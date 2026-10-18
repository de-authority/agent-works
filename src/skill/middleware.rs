//! Per-turn system prompt refresh for the skills catalog (skill-injection M3a).
//!
//! Host apps embed the catalog in the stored system prompt at build time and
//! let this middleware swap in a fresh one before each LLM call
//! (`refresh_catalog` — idempotent, in place), so resolver changes show up
//! without restarting and the prompt never accumulates duplicate catalogs.
//!
//! Uses `agent_base::Middleware::on_pre_llm`, which runs before every LLM
//! call — zero cross-repo changes needed on the engine side.

use std::sync::Arc;

use agent_base::llm_trait::ChatMessage;
use agent_base::{AgentResult, Middleware, PreLlmCtx};
use async_trait::async_trait;

use super::catalog::render_catalog;
use super::resolver::SkillResolver;

/// Middleware that refreshes the system prompt (base + skills catalog) before
/// each LLM call. This lets the model see skill catalog changes (additions,
/// removals, description edits) without restarting the session.
///
/// The resolver is `Arc`-shared with the host app's slash path and the
/// `SkillTool` — all three read the same immutable skill list. To pick up
/// filesystem changes, the resolver would need to be rebuilt (future work;
/// the crate's `hot-reload` feature is the natural home); this middleware
/// handles the *wiring* half: ensuring the prompt the model sees always
/// reflects the resolver's current state, without duplication.
pub struct SkillCatalogRefreshMiddleware {
    resolver: Arc<SkillResolver>,
}

impl SkillCatalogRefreshMiddleware {
    pub fn new(resolver: Arc<SkillResolver>) -> Self {
        Self { resolver }
    }
}

#[async_trait]
impl Middleware for SkillCatalogRefreshMiddleware {
    async fn on_pre_llm(&self, ctx: &mut PreLlmCtx) -> AgentResult<()> {
        // Only touch the system message when there are skills to inject.
        // Without skills, the builder's original prompt is left byte-identical.
        let Some(catalog) = render_catalog(&self.resolver) else {
            return Ok(());
        };
        match ctx.messages.first_mut() {
            Some(ChatMessage::System { content, .. }) => {
                // Idempotent refresh: swap the previously injected catalog (the
                // builder already composed one into the stored prompt) for the
                // current one, in place — trailing sections keep their position.
                // A plain append made the model see a duplicate catalog on every
                // LLM call.
                *content = super::catalog::refresh_catalog(content, &catalog);
            }
            // Silently skipping would mask wiring errors: skills are assembled
            // yet the catalog never refreshes, and the catalog the model sees
            // quietly goes stale — a regression that is hard to notice, so at
            // least leave a log.
            _ => tracing::warn!(
                "skill catalog refresh skipped: first message is not System \
                 (skills are loaded but the catalog cannot be refreshed)"
            ),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::SessionId;
    use std::path::Path;

    const BASE: &str = "You are a coding agent.";

    fn make_skill_dir(tmp: &Path, name: &str, body: &str) {
        let dir = tmp.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: test\nuser-invocable: true\n---\n\n{body}"),
        )
        .unwrap();
    }

    fn make_pre_llm_ctx(system_content: &str) -> PreLlmCtx {
        PreLlmCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
            messages: vec![ChatMessage::System {
                content: system_content.to_string(),
                ephemeral: false,
            }],
            tools: vec![],
            emit_fn: None,
            turn_count: 1,
            max_turns: 10,
        }
    }

    /// Equivalent construction of the builder's output: base + "\n\n" + catalog —
    /// this is how the host app composes the catalog into the stored system
    /// prompt at build() time.
    fn stored_prompt(resolver: &SkillResolver) -> String {
        format!("{BASE}\n\n{}", render_catalog(resolver).unwrap())
    }

    #[tokio::test]
    async fn refresh_replaces_system_message_with_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver);
        let mut ctx = make_pre_llm_ctx("original prompt");

        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(
                    content.starts_with("original prompt"),
                    "base prompt must be preserved"
                );
                assert!(content.contains("## Skills"), "catalog must be appended");
                assert!(content.contains("- alpha: test"), "skill must appear");
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_without_skills_keeps_original_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("nonexistent")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver);
        let original = "base prompt only";
        let mut ctx = make_pre_llm_ctx(original);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(content, original, "must be byte-identical without skills");
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_reflects_newly_added_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());

        // First turn: no skills
        let mut ctx = make_pre_llm_ctx("prompt");
        mw.on_pre_llm(&mut ctx).await.unwrap();
        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(!content.contains("## Skills"));
            }
            _ => panic!(),
        }

        // Add a skill and rebuild resolver
        make_skill_dir(tmp.path(), "new-skill", "new body");
        let resolver2 = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw2 = SkillCatalogRefreshMiddleware::new(resolver2);

        // Second turn: skill appears
        let mut ctx = make_pre_llm_ctx("prompt");
        mw2.on_pre_llm(&mut ctx).await.unwrap();
        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(
                    content.contains("- new-skill: test"),
                    "new skill must appear"
                );
            }
            _ => panic!(),
        }
    }

    // ── Catalog duplication regression (build-time M1 × per-turn M3a) ──
    //
    // build() already composed the catalog into the stored system prompt; the
    // middleware must swap the old one out before injecting, not append another
    // copy (the model used to see two catalogs on every LLM call).

    #[tokio::test]
    async fn refresh_strips_builder_embedded_catalog_instead_of_duplicating() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        // Real wiring: what the builder stores into the session is the prompt
        // with the catalog in it
        let stored = stored_prompt(&resolver);
        assert!(stored.contains("## Skills"));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(
                    content.matches("## Skills").count(),
                    1,
                    "catalog must appear exactly once, not duplicated per LLM call"
                );
                assert_eq!(
                    content,
                    &stored_prompt(&resolver),
                    "unchanged resolver → byte-identical recomposition"
                );
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_preserves_trailing_sections_after_catalog() {
        // token-budget-style hosts: the prompt suffix is push_str'd straight
        // after the catalog (the suffix carries its own \n\n lead-in); refreshing
        // the old catalog must not swallow what follows it.
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let suffix = "\n\n## Context Management\n\nuse history and notes tools.";
        let stored = format!("{}{suffix}", stored_prompt(&resolver));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(content.matches("## Skills").count(), 1, "{content}");
                assert!(
                    content.ends_with(suffix),
                    "trailing section must survive: {content}"
                );
                assert_eq!(
                    content, &stored,
                    "unchanged resolver + suffix → byte-identical to builder output"
                );
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_is_idempotent_across_repeated_calls() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored_prompt(&resolver));
        mw.on_pre_llm(&mut ctx).await.unwrap();

        let after_first = match &ctx.messages[0] {
            ChatMessage::System { content, .. } => content.clone(),
            _ => panic!(),
        };

        // Second call (the next LLM call; messages re-cloned from the session)
        let mut ctx2 = make_pre_llm_ctx(&after_first);
        mw.on_pre_llm(&mut ctx2).await.unwrap();

        match &ctx2.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(
                    content, &after_first,
                    "repeated refresh must be a fixed point"
                );
                assert_eq!(content.matches("## Skills").count(), 1);
            }
            _ => panic!("expected System message"),
        }
    }

    // ── Robustness when the first message is not System: skip + warn log,
    //    no panic, no mis-edit ──

    #[tokio::test]
    async fn non_system_first_message_left_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver);

        let mut ctx = make_pre_llm_ctx("unused");
        ctx.messages = vec![ChatMessage::User {
            content: "hi".to_string(),
            images: vec![],
            ephemeral: false,
        }];
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::User { content, .. } => {
                assert_eq!(content, "hi", "user message must survive")
            }
            other => panic!("expected User message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_messages_do_not_panic() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver);

        let mut ctx = make_pre_llm_ctx("unused");
        ctx.messages.clear();
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }
}
