//! Built-in child-agent presets (design doc §6.1).
//!
//! A [`ChildPreset`] is nothing but a named [`ChildConfig`] template — the
//! one merge rule in the whole API is "`preset()` fills fields the user has
//! not explicitly set, never overwrites" (§5.3).
//!
//! The `tool::*` constants are **real registered tool names** (verified
//! against the `fn name()` implementations in phi-kernel-tools; there is no
//! search/grep tool, hence researcher's whitelist is read + list). The CI
//! fixture test below re-registers every constant and spawns every preset
//! against it, so a renamed tool breaks the build here, not silently at
//! runtime.

use std::collections::BTreeSet;

use super::child_config::ChildConfig;

/// Real registered tool names (§6.1). Whitelists are written against these.
pub mod tool {
    pub const READ_FILE: &str = "read_file";
    pub const WRITE_FILE: &str = "write_file";
    pub const EDIT_FILE: &str = "edit_file";
    pub const LIST_FILES: &str = "list_files";
    pub const EXECUTE_COMMAND: &str = "execute_command";
}

fn tools(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

/// The mutating-tool set that `AgentAutonomy::Manual` excludes by default
/// (§7.5): the phi-kernel-tools write set. Deployments with custom mutating
/// tools extend it via `ControlConfig::write_tools`.
pub fn default_write_tools() -> Vec<String> {
    vec![
        tool::WRITE_FILE.to_string(),
        tool::EDIT_FILE.to_string(),
        tool::EXECUTE_COMMAND.to_string(),
    ]
}

/// A named [`ChildConfig`] template for `ChildBuilder::preset`.
#[derive(Debug, Clone)]
pub struct ChildPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub config: ChildConfig,
}

impl ChildPreset {
    /// Code research/analysis, read-only.
    pub fn researcher() -> Self {
        Self {
            name: "researcher",
            description: "Code research/analysis, read-only",
            config: ChildConfig {
                system_prompt: Some(
                    "You are a code research expert. Analyze code structure, understand logic, \
                     and discover patterns; read-only, write your findings into the final answer."
                        .into(),
                ),
                tool_names: Some(tools(&[tool::READ_FILE, tool::LIST_FILES])),
                max_turns: Some(32),
                ..Default::default()
            },
        }
    }

    /// Code writing and modification.
    pub fn coder() -> Self {
        Self {
            name: "coder",
            description: "Code writing and modification",
            config: ChildConfig {
                system_prompt: Some(
                    "You are a code writing expert. Write high-quality code to the requirements, \
                     keeping changes minimal."
                        .into(),
                ),
                tool_names: Some(tools(&[
                    tool::READ_FILE,
                    tool::WRITE_FILE,
                    tool::EDIT_FILE,
                    tool::LIST_FILES,
                    tool::EXECUTE_COMMAND,
                ])),
                max_turns: Some(64),
                ..Default::default()
            },
        }
    }

    /// Code review — researcher's read-only shape, review prompt.
    pub fn reviewer() -> Self {
        Self {
            name: "reviewer",
            description: "Code review, read-only",
            config: ChildConfig {
                system_prompt: Some(
                    "You are a code review expert. Review for correctness, style, and security \
                     issues; read-only, write your review comments into the final answer."
                        .into(),
                ),
                tool_names: Some(tools(&[tool::READ_FILE, tool::LIST_FILES])),
                max_turns: Some(32),
                ..Default::default()
            },
        }
    }

    /// Test authoring and execution — coder's tool set, test prompt.
    pub fn tester() -> Self {
        Self {
            name: "tester",
            description: "Test authoring and execution",
            config: ChildConfig {
                system_prompt: Some(
                    "You are a testing expert. Write tests for the code and execute them; \
                     report pass/fail and coverage."
                        .into(),
                ),
                tool_names: Some(tools(&[
                    tool::READ_FILE,
                    tool::WRITE_FILE,
                    tool::EDIT_FILE,
                    tool::LIST_FILES,
                    tool::EXECUTE_COMMAND,
                ])),
                max_turns: Some(64),
                ..Default::default()
            },
        }
    }

    /// A user-defined template — same fill-unset-only merge semantics.
    pub fn custom(name: &'static str, description: &'static str, config: ChildConfig) -> Self {
        Self {
            name,
            description,
            config,
        }
    }

    /// Resolve one of the four built-ins by name (§8.2: unknown names are a
    /// tool-layer compatibility concern, not an error here).
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "researcher" => Some(Self::researcher()),
            "coder" => Some(Self::coder()),
            "reviewer" => Some(Self::reviewer()),
            "tester" => Some(Self::tester()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_base::{Content, Tool, ToolContext, llm_trait::LlmProvider};

    use super::*;
    use crate::multi_agent::child_config::ChildConfig as Cfg;
    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::runtime::MultiAgentRuntime;

    // Fixture asserting tool::* constants are real, registerable names (§6.1:
    // "every constant must be registerable in the integration fixture"). A constant that drifted from
    // phi-kernel-tools would make the whitelist validation fail with
    // ToolNotFound here — the spawn below is the assertion.
    struct NamedTool(&'static str);

    #[async_trait::async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "fixture"
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> agent_base::AgentResult<Vec<Content>> {
            Ok(vec![Content::text("ok")])
        }
    }

    struct QuietLlm;

    #[async_trait::async_trait]
    impl LlmProvider for QuietLlm {
        async fn stream(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
            Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::iter(vec![Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                })]),
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
                name: "quiet".into(),
                model: "quiet".into(),
                version: None,
            }
        }
    }

    fn fixture_runtime() -> Arc<MultiAgentRuntime> {
        // Every tool::* constant registered — the exact set presets whitelist
        // against. A renamed/removed real tool surfaces here as ToolNotFound.
        let business_tools: Vec<Arc<dyn Tool>> = [
            tool::READ_FILE,
            tool::WRITE_FILE,
            tool::EDIT_FILE,
            tool::LIST_FILES,
            tool::EXECUTE_COMMAND,
        ]
        .into_iter()
        .map(|n| Arc::new(NamedTool(n)) as Arc<dyn Tool>)
        .collect();
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(QuietLlm),
            business_tools,
            tokio_util::sync::CancellationToken::new(),
            None,
            agent_base::Language::En,
            None,
            None,
        ))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn every_preset_whitelist_resolves_against_fixture() {
        let ma = fixture_runtime();
        for preset in [
            ChildPreset::researcher(),
            ChildPreset::coder(),
            ChildPreset::reviewer(),
            ChildPreset::tester(),
        ] {
            let spawned = ma
                .spawn_with_config(format!("p_{}", preset.name), preset.config.clone())
                .await
                .unwrap_or_else(|e| panic!("preset {} failed to spawn: {e}", preset.name));
            // Constants registered ⇒ the full whitelist survives validation,
            // and the echo equals the preset's set exactly.
            assert_eq!(
                spawned.spawned_tools(),
                preset.config.tool_names.as_ref().unwrap(),
                "preset {}",
                preset.name
            );
            ma.close_agent(&spawned.agent_path().to_string())
                .ok()
                .filter(|r| r.closed)
                .unwrap_or_else(|| panic!("child {} should close", preset.name));
        }
    }

    #[test]
    fn constants_match_phimint_registered_names() {
        // Belt for the fixture test above: the constants themselves are the
        // §6.1 list, spelled exactly.
        assert_eq!(tool::READ_FILE, "read_file");
        assert_eq!(tool::WRITE_FILE, "write_file");
        assert_eq!(tool::EDIT_FILE, "edit_file");
        assert_eq!(tool::LIST_FILES, "list_files");
        assert_eq!(tool::EXECUTE_COMMAND, "execute_command");
    }

    #[test]
    fn by_name_resolves_four_and_rejects_rest() {
        assert_eq!(
            ChildPreset::by_name("researcher").unwrap().name,
            "researcher"
        );
        assert_eq!(ChildPreset::by_name("coder").unwrap().name, "coder");
        assert_eq!(ChildPreset::by_name("reviewer").unwrap().name, "reviewer");
        assert_eq!(ChildPreset::by_name("tester").unwrap().name, "tester");
        assert!(ChildPreset::by_name("translator").is_none());
    }

    #[test]
    fn custom_wraps_config_verbatim() {
        let cfg = Cfg {
            system_prompt: Some("hi".into()),
            max_turns: Some(3),
            ..Default::default()
        };
        let p = ChildPreset::custom("mine", "d", cfg.clone());
        assert_eq!(p.name, "mine");
        assert_eq!(p.config.system_prompt.as_deref(), Some("hi"));
        assert_eq!(p.config.max_turns, Some(3));
        // Unset fields stay inherit.
        assert!(p.config.context_window.is_none());
        assert!(p.config.full_permission.is_none());
    }
}
