//! `skill` tool (skill-injection design D2): model-initiated skill body loader.
//!
//! When the model sees a skill in the system-prompt catalog, it calls this tool
//! with the skill's `name` to load the full instruction body. The resolver does
//! an exact name match (the model has the exact name from the catalog) and
//! returns the resolved body.

use std::sync::Arc;

use agent_base::{AgentResult, Content, Tool, ToolContext, ToolMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::task;

use super::resolver::{SkillLookupError, SkillResolver};
use super::telemetry::SkillTelemetry;

/// Output budget for a single skill body call. When the body exceeds this
/// limit, the tool returns a truncated excerpt plus the `SKILL.md` absolute
/// path so the model can call `read_file` to finish (Codex fallback).
const MAX_SKILL_BODY_CHARS: usize = 16_000;

/// How many names the not-found error lists before "(+N more)".
const MAX_ERROR_LISTING: usize = 20;

/// Model-facing skill body loader (read-only, no approval needed).
pub struct SkillTool {
    resolver: Arc<SkillResolver>,
    telemetry: Arc<SkillTelemetry>,
}

impl SkillTool {
    pub fn new(resolver: Arc<SkillResolver>, telemetry: Arc<SkillTelemetry>) -> Self {
        Self {
            resolver,
            telemetry,
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &'static str {
        "skill"
    }

    fn description(&self) -> &'static str {
        "Load a skill's full instruction body by name. Use this when the user's request matches an available skill (e.g., 'review code' → 'review', 'deploy' → 'ship', 'investigate bug' → 'investigate'). The skill body contains detailed instructions for the task. Returns the skill body, or an error with available skill names if not found."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill's exact name (as listed in the system prompt catalog)."
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments passed to the skill ($ARGUMENTS placeholder in the body)."
                }
            },
            "required": ["name"]
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: "Load a skill body by exact name.".to_string(),
            origin: "agent-works".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let name = match args.get("name").and_then(Value::as_str) {
            Some(n) => n.to_string(),
            None => {
                return Ok(vec![Content::text(
                    "Error: missing required field `name` (string).".to_string(),
                )]);
            }
        };
        let raw_args = args
            .get("args")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let resolver = Arc::clone(&self.resolver);
        let telemetry = Arc::clone(&self.telemetry);
        let max_chars = ctx.max_output_chars.unwrap_or(MAX_SKILL_BODY_CHARS);

        // All work is in-memory; spawn_blocking keeps the async runtime free.
        let result = task::spawn_blocking(move || {
            match resolver.resolve_by_name(&name, &raw_args) {
                Ok((body, matched_name)) => {
                    // Record model-triggered skill load for telemetry (M3b).
                    telemetry.record_model(matched_name);
                    tracing::info!(
                        name = %name,
                        matched = %matched_name,
                        body_len = body.len(),
                        "resolved skill via tool"
                    );
                    if body.chars().count() <= max_chars {
                        Ok(body)
                    } else {
                        // Overflow: truncated excerpt + path so the model can
                        // use read_file to finish the read. The hint is carved
                        // out of the budget — the pipeline rejects output over
                        // max_output_chars, so truncating to the full cap and
                        // then appending would blow the limit.
                        let path_hint = resolver
                            .source_path_for(matched_name)
                            .map(|p| format!("\n\n[truncated - full body at: {}]", p.display()))
                            .unwrap_or_default();
                        let budget = max_chars.saturating_sub(path_hint.chars().count());
                        let truncated: String = body.chars().take(budget).collect();
                        Ok(format!("{truncated}{path_hint}"))
                    }
                }
                Err(SkillLookupError::ModelInvocationDenied) => {
                    tracing::info!(name = %name, "skill tool: model invocation denied");
                    Err(format!(
                        "Error: skill \"{name}\" has disable-model-invocation set and cannot be loaded by the model. It can only be run explicitly by the user."
                    ))
                }
                Err(SkillLookupError::NotFound) => {
                    let available = resolver.skill_names();
                    let listing: Vec<&str> =
                        available.iter().take(MAX_ERROR_LISTING).copied().collect();
                    let omitted = available.len().saturating_sub(MAX_ERROR_LISTING);
                    let mut msg = format!(
                        "Error: no skill named \"{name}\". Available skills: {listing:?}"
                    );
                    if omitted > 0 {
                        msg.push_str(&format!(" (+{omitted} more)"));
                    }
                    tracing::info!(name = %name, "skill tool: no match");
                    Err(msg)
                }
            }
        })
        .await;

        match result {
            Ok(Ok(body)) => Ok(vec![Content::text(body)]),
            Ok(Err(msg)) => Ok(vec![Content::text(msg)]),
            Err(e) => Ok(vec![Content::text(format!(
                "[Error]: skill tool task failed: {e}"
            ))]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temporary skill directory with one or more skills.
    ///
    /// Returns `(Arc<SkillResolver>, TempDir)` — the guard must live as long as
    /// the resolver, because `source_path_for` points into the temp directory
    /// and the overflow test relies on that path being readable.
    fn make_resolver(skills: &[(&str, &str, bool)]) -> (Arc<SkillResolver>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        for (name, body, invocable) in skills {
            let dir = tmp.path().join("skills").join(name);
            fs::create_dir_all(&dir).unwrap();
            let inv = if *invocable { "true" } else { "false" };
            fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: test skill\nuser-invocable: {inv}\n---\n\n{body}"),
            )
            .unwrap();
        }
        (
            Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")])),
            tmp,
        )
    }

    fn ctx() -> ToolContext {
        ToolContext::for_test()
    }

    fn text(out: Vec<Content>) -> String {
        out.into_iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn happy_path_returns_body() {
        let (resolver, _guard) =
            make_resolver(&[("code-review", "Review this PR carefully.", true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = tool
            .call(&json!({"name": "code-review"}), &ctx())
            .await
            .unwrap();
        assert_eq!(text(out), "Review this PR carefully.");
    }

    #[tokio::test]
    async fn happy_path_with_args_substitution() {
        let (resolver, _guard) = make_resolver(&[("greet", "Hello $ARGUMENTS!", true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = tool
            .call(&json!({"name": "greet", "args": "world"}), &ctx())
            .await
            .unwrap();
        assert_eq!(text(out), "Hello world!");
    }

    #[tokio::test]
    async fn not_found_returns_available_names() {
        let (resolver, _guard) = make_resolver(&[("alpha", "a", true), ("beta", "b", true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = text(tool.call(&json!({"name": "gamma"}), &ctx()).await.unwrap());
        assert!(out.contains("no skill named"), "{out}");
        assert!(out.contains("alpha"), "{out}");
        assert!(out.contains("beta"), "{out}");
    }

    #[tokio::test]
    async fn non_user_invocable_is_accessible_by_tool() {
        // user_invocable:false only blocks the user slash path, not the model tool.
        let (resolver, _guard) = make_resolver(&[("internal", "internal body", false)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = text(
            tool.call(&json!({"name": "internal"}), &ctx())
                .await
                .unwrap(),
        );
        assert_eq!(out, "internal body");
    }

    #[tokio::test]
    async fn long_body_truncates_with_path_hint() {
        // No custom max_output_chars constructible from outside agent-base
        // (event_bus is pub(crate)), so we test against the tool's internal
        // default (MAX_SKILL_BODY_CHARS = 16_000).
        let long_body = "x".repeat(17_000);
        let (resolver, _guard) = make_resolver(&[("long", &long_body, true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = text(tool.call(&json!({"name": "long"}), &ctx()).await.unwrap());
        assert!(out.contains("[truncated"), "{out}");
        assert!(
            out.contains("SKILL.md"),
            "must include path for read_file continuation: {out}"
        );
        // The truncated excerpt + hint must not exceed max_output_chars overall —
        // the pipeline rejects over-limit output.
        assert!(
            out.chars().count() <= MAX_SKILL_BODY_CHARS,
            "overflow output must stay within the budget: {}",
            out.chars().count()
        );
    }

    #[tokio::test]
    async fn body_exactly_at_char_limit_not_truncated() {
        let exact_body = "y".repeat(MAX_SKILL_BODY_CHARS);
        let (resolver, _guard) = make_resolver(&[("exact", &exact_body, true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        let out = text(tool.call(&json!({"name": "exact"}), &ctx()).await.unwrap());
        assert_eq!(out.chars().count(), MAX_SKILL_BODY_CHARS);
        assert!(
            !out.contains("[truncated"),
            "at-limit body must not gain a path hint: {out}"
        );
    }

    #[tokio::test]
    async fn missing_or_non_string_name_returns_error() {
        let (resolver, _guard) = make_resolver(&[("alpha", "a", true)]);
        let tool = SkillTool::new(resolver, Arc::new(SkillTelemetry::new()));
        for bad in [json!({}), json!({"name": 42})] {
            let out = text(tool.call(&bad, &ctx()).await.unwrap());
            assert!(
                out.contains("missing required field `name`"),
                "input {bad} should yield a field error: {out}"
            );
        }
    }

    #[tokio::test]
    async fn not_found_listing_caps_at_20_with_omitted_count() {
        // 22 skills: on error only 20 names are listed + "(+2 more)".
        let tmp = tempfile::tempdir().unwrap();
        let names: Vec<String> = (0..22).map(|i| format!("skill-{i:02}")).collect();
        for name in &names {
            let dir = tmp.path().join("skills").join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: d\nuser-invocable: true\n---\n\nbody"),
            )
            .unwrap();
        }
        let tool = SkillTool::new(
            Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")])),
            Arc::new(SkillTelemetry::new()),
        );
        let out = text(tool.call(&json!({"name": "absent"}), &ctx()).await.unwrap());
        assert!(out.contains("no skill named \"absent\""), "{out}");
        assert!(out.contains("skill-00"), "{out}");
        assert!(out.contains("skill-19"), "{out}");
        assert!(
            !out.contains("skill-20"),
            "listing must cap at {MAX_ERROR_LISTING}: {out}"
        );
        assert!(out.contains("(+2 more)"), "{out}");
    }

    #[tokio::test]
    async fn model_denied_skill_returns_error_not_body() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("skills").join("deploy");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: d\nuser-invocable: true\ndisable-model-invocation: true\n---\n\nsecret ship steps",
        )
        .unwrap();
        let tool = SkillTool::new(
            Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")])),
            Arc::new(SkillTelemetry::new()),
        );
        let out = text(tool.call(&json!({"name": "deploy"}), &ctx()).await.unwrap());
        assert!(out.contains("Error:"), "{out}");
        assert!(out.contains("disable-model-invocation"), "{out}");
        assert!(
            !out.contains("secret ship steps"),
            "denied body must not leak: {out}"
        );
    }
}
