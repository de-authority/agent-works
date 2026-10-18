//! Fluent child-agent builder (design doc §5.3).
//!
//! The one merge rule in the whole API: **setters record explicit user
//! intent; `preset()` only fills fields the user has not set, never
//! overwrites.** (`ChildFieldSet` is what "explicit" means — a draft field
//! alone can't tell you, since preset-filled values also land in the draft.)

use std::collections::BTreeSet;
use std::sync::Arc;

use agent_base::{AgentError, SessionId};

use super::child::{ChildGuard, ChildHandle};
use super::child_config::ChildConfig;
use super::preset::ChildPreset;
use super::runtime::MultiAgentRuntime;

/// Which fields the user set through a setter (as opposed to `preset()`).
#[derive(Default)]
struct ChildFieldSet {
    system_prompt: bool,
    tool_names: bool,
    max_turns: bool,
    context_window: bool,
    full_permission: bool,
}

/// Builder for one spawned child agent. Entry point:
/// [`MultiAgentRuntime::child`](super::MultiAgentRuntime::child).
pub struct ChildBuilder {
    runtime: Arc<MultiAgentRuntime>,
    draft: ChildConfig,
    explicit: ChildFieldSet,
    /// Legacy `fork_history` route — deliberately **not** a `ChildConfig`
    /// field: the parent session id is per-call tool-layer knowledge
    /// (`ToolContext::session_id`), not spawn-template knowledge, so it
    /// lives on the builder alone (design §7.5 "no new fields" preserved).
    fork: Option<(String, SessionId)>,
}

impl ChildBuilder {
    pub(crate) fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self {
            runtime,
            draft: ChildConfig::default(),
            explicit: ChildFieldSet::default(),
            fork: None,
        }
    }

    /// Set the system prompt (required — `spawn` fails fast without it).
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.draft.system_prompt = Some(prompt.into());
        self.explicit.system_prompt = true;
        self
    }

    /// Replace the tool whitelist wholesale (explicit intent).
    pub fn tool_names(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.draft.tool_names = Some(names.into_iter().map(Into::into).collect());
        self.explicit.tool_names = true;
        self
    }

    /// Add one tool to the whitelist. Starts a fresh set if none exists yet —
    /// either way the whitelist becomes explicit intent, so `preset()` won't
    /// touch it afterwards.
    pub fn add_tool_name(mut self, name: impl Into<String>) -> Self {
        self.draft
            .tool_names
            .get_or_insert_with(BTreeSet::new)
            .insert(name.into());
        self.explicit.tool_names = true;
        self
    }

    /// Max turns (→ `AgentBuilder::execution_max_turns`).
    pub fn max_turns(mut self, max: u32) -> Self {
        self.draft.max_turns = Some(max);
        self.explicit.max_turns = true;
        self
    }

    /// Context window in tokens (→ `AgentBuilder::context_window`).
    pub fn context_window(mut self, tokens: usize) -> Self {
        self.draft.context_window = Some(tokens);
        self.explicit.context_window = true;
        self
    }

    /// Per-spawn permission override (effective only under
    /// `ChildPermissionMode::PerSpawn`, §10.1 B4).
    pub fn full_permission(mut self, full: bool) -> Self {
        self.draft.full_permission = Some(full);
        self.explicit.full_permission = true;
        self
    }

    /// Apply a preset: fill only the fields the user has not set. Order
    /// relative to the setters does not matter for precedence — a later
    /// setter still wins (it marks the field explicit).
    pub fn preset(mut self, preset: &ChildPreset) -> Self {
        let p = &preset.config;
        if !self.explicit.system_prompt && p.system_prompt.is_some() {
            self.draft.system_prompt = p.system_prompt.clone();
        }
        if !self.explicit.tool_names && p.tool_names.is_some() {
            self.draft.tool_names = p.tool_names.clone();
        }
        if !self.explicit.max_turns && p.max_turns.is_some() {
            self.draft.max_turns = p.max_turns;
        }
        if !self.explicit.context_window && p.context_window.is_some() {
            self.draft.context_window = p.context_window;
        }
        if !self.explicit.full_permission && p.full_permission.is_some() {
            self.draft.full_permission = p.full_permission;
        }
        self
    }

    /// Inherit parent conversation history — the legacy `fork_history`
    /// parameter surfaced on the builder (§8.2 "no legacy behavior lost in one release": the tool
    /// schema kept promising it, and the production wiring does have a
    /// session manager). `mode` is `"none"` | `"all"` | a number N of last
    /// turns; `parent_session` is the calling agent's session
    /// (`ToolContext::session_id`). Resolution is lenient exactly like the
    /// legacy path: no session manager or a bad mode degrades to no history.
    pub fn fork_history(self, mode: impl Into<String>, parent_session: SessionId) -> Self {
        Self {
            fork: Some((mode.into(), parent_session)),
            ..self
        }
    }

    /// Build and spawn. The required-field check on `system_prompt` runs
    /// here (fail-fast, before any gate is touched).
    pub async fn spawn(self, name: impl Into<String>) -> Result<ChildHandle, AgentError> {
        let name = name.into();
        if self.draft.system_prompt.as_deref().unwrap_or("").is_empty() {
            return Err(AgentError::ConfigError(
                "ChildConfig.system_prompt is required (set it directly or use a preset)".into(),
            ));
        }
        let parent_messages = match &self.fork {
            Some((mode, sid)) => {
                self.runtime
                    .resolve_fork_history(Some(mode.clone()), sid)
                    .await
            }
            None => Vec::new(),
        };
        let spawned = self
            .runtime
            .spawn_with_config_forked(name, self.draft, parent_messages, None)
            .await?;
        Ok(ChildHandle::new(
            Arc::clone(&self.runtime),
            spawned.agent_path().clone(),
            spawned.spawned_tools().clone(),
        ))
    }

    /// Convenience: spawn and immediately wrap in a close-on-drop guard
    /// (the one direction of the handle↔guard conversion, D1).
    pub async fn spawn_guarded(self, name: impl Into<String>) -> Result<ChildGuard, AgentError> {
        Ok(self.spawn(name).await?.into_guard())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use agent_base::llm_trait::LlmProvider;

    use super::*;
    use crate::multi_agent::child::ChildOutcome;
    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::preset::tool;

    struct EchoLlm;

    #[async_trait::async_trait]
    impl LlmProvider for EchoLlm {
        async fn stream(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
            Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::iter(vec![
                    Ok(agent_base::StreamChunk::Text("done".to_string())),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    }),
                ]),
            )))
        }
        async fn chat(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
            unreachable!("unused")
        }
        fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
            agent_base::llm_trait::Capabilities::default()
        }
        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "echo".into(),
                model: "echo".into(),
                version: None,
            }
        }
    }

    fn runtime() -> Arc<MultiAgentRuntime> {
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(EchoLlm),
            vec![],
            tokio_util::sync::CancellationToken::new(),
            None,
            agent_base::Language::En,
            None,
            None,
        ))
    }

    /// Preset with every field explicitly set.
    fn full_preset() -> ChildPreset {
        ChildPreset::custom(
            "full",
            "d",
            ChildConfig {
                system_prompt: Some("preset prompt".into()),
                tool_names: Some([tool::READ_FILE.to_string()].into_iter().collect()),
                max_turns: Some(32),
                context_window: Some(4096),
                full_permission: Some(true),
                ..Default::default()
            },
        )
    }

    /// Preset with no field set (inherits everything).
    fn empty_preset() -> ChildPreset {
        ChildPreset::custom("empty", "d", ChildConfig::default())
    }

    // ── merge matrix (§12 stage-1 acceptance) ──────────────────────────────
    //
    // Per field, five cells:
    //   1. user → full preset          ⇒ user wins (preset never overwrites)
    //   2. user → empty preset         ⇒ user survives
    //   3. (no user) → full preset     ⇒ preset value fills in
    //   4. (no user) → empty preset    ⇒ stays None (inherit)
    //   5. full preset → user          ⇒ user wins (setter after preset)

    fn merge_cells<T>(
        field: &str,
        user_val: T,
        preset_val: T,
        set: impl Fn(ChildBuilder, T) -> ChildBuilder,
        get: impl Fn(&ChildBuilder) -> Option<T>,
    ) where
        T: PartialEq + Clone + std::fmt::Debug,
    {
        let ma = runtime();
        let full = full_preset();
        let empty = empty_preset();

        let b = set(ChildBuilder::new(Arc::clone(&ma)), user_val.clone()).preset(&full);
        assert_eq!(get(&b), Some(user_val.clone()), "{field}: cell 1");
        let b = set(ChildBuilder::new(Arc::clone(&ma)), user_val.clone()).preset(&empty);
        assert_eq!(get(&b), Some(user_val.clone()), "{field}: cell 2");
        let b = ChildBuilder::new(Arc::clone(&ma)).preset(&full);
        assert_eq!(get(&b), Some(preset_val), "{field}: cell 3");
        let b = ChildBuilder::new(Arc::clone(&ma)).preset(&empty);
        assert_eq!(get(&b), None, "{field}: cell 4");
        let b = set(ChildBuilder::new(ma).preset(&full), user_val.clone());
        assert_eq!(get(&b), Some(user_val), "{field}: cell 5");
    }

    #[test]
    fn merge_matrix_system_prompt() {
        merge_cells(
            "system_prompt",
            "user prompt".to_string(),
            "preset prompt".to_string(),
            |b, v| b.system_prompt(v),
            |b| b.draft.system_prompt.clone(),
        );
    }

    #[test]
    fn merge_matrix_tool_names() {
        merge_cells(
            "tool_names",
            [tool::WRITE_FILE.to_string()].into_iter().collect(),
            [tool::READ_FILE.to_string()].into_iter().collect(),
            |b, v| b.tool_names(v),
            |b| b.draft.tool_names.clone(),
        );
    }

    #[test]
    fn merge_matrix_max_turns() {
        merge_cells(
            "max_turns",
            8u32,
            32u32,
            |b, v| b.max_turns(v),
            |b| b.draft.max_turns,
        );
    }

    #[test]
    fn merge_matrix_context_window() {
        merge_cells(
            "context_window",
            1024usize,
            4096usize,
            |b, v| b.context_window(v),
            |b| b.draft.context_window,
        );
    }

    #[test]
    fn merge_matrix_full_permission() {
        merge_cells(
            "full_permission",
            false,
            true,
            |b, v| b.full_permission(v),
            |b| b.draft.full_permission,
        );
    }

    #[test]
    fn add_tool_name_accumulates_and_blocks_preset() {
        // add_tool_name marks the whitelist explicit even without a prior
        // tool_names() call, and appends across calls.
        let b = ChildBuilder::new(runtime())
            .add_tool_name("a")
            .add_tool_name("b")
            .preset(&full_preset());
        assert_eq!(
            b.draft.tool_names,
            Some(["a".to_string(), "b".to_string()].into_iter().collect())
        );
    }

    #[test]
    fn add_tool_name_after_preset_appends_to_preset_set() {
        let b = ChildBuilder::new(runtime())
            .preset(&full_preset())
            .add_tool_name("extra");
        assert_eq!(
            b.draft.tool_names,
            Some(
                [tool::READ_FILE.to_string(), "extra".to_string()]
                    .into_iter()
                    .collect()
            )
        );
    }

    // ── spawn behaviour ────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_requires_system_prompt() {
        let err = runtime()
            .child()
            .max_turns(4)
            .spawn("worker")
            .await
            .map(|_h| ())
            .expect_err("no prompt must fail");
        assert!(
            matches!(&err, AgentError::ConfigError(s)
                if s == "ChildConfig.system_prompt is required (set it directly or use a preset)"),
            "got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_with_preset_prompt_succeeds() {
        // A preset's prompt satisfies the required-field check.
        let ma = runtime();
        let handle = ma
            .child()
            .preset(&ChildPreset::custom(
                "p",
                "d",
                ChildConfig {
                    system_prompt: Some("prompt".into()),
                    ..Default::default()
                },
            ))
            .spawn("w")
            .await
            .expect("preset prompt spawns");
        assert_eq!(handle.agent_path(), "root/w");
        // Full round trip: task → wait → close.
        assert!(handle.task("go").unwrap());
        let outcome = handle.wait(Duration::from_secs(3)).await;
        assert!(matches!(outcome, ChildOutcome::Ok { .. }));
        assert!(handle.close().unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn spawn_guarded_closes_on_drop() {
        let ma = runtime();
        let guard = ma
            .child()
            .system_prompt("prompt")
            .spawn_guarded("g")
            .await
            .expect("spawn_guarded");
        assert_eq!(guard.handle().agent_path(), "root/g");
        drop(guard);
        // Deferred cleanup (§4): the registry empties asynchronously.
        for _ in 0..100 {
            if ma.registry().lock().unwrap().count() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(ma.registry().lock().unwrap().count(), 0);
    }
}
