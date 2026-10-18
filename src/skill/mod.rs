use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use agent_base::{PlanItem, Tool};
use serde::Serialize;

pub mod prompter;
pub mod registry;

#[cfg(feature = "prompt_skill")]
pub mod prompt_skill;

#[cfg(feature = "yaml_skill")]
pub mod yaml_skill;

// Filesystem discovery + prompt injection capability (skill-injection design
// D1/D2/M3). Gated on `prompt_skill` because the resolver builds on
// `PromptSkill::scan_dir`.
#[cfg(feature = "prompt_skill")]
pub mod catalog;
#[cfg(feature = "prompt_skill")]
pub mod middleware;
#[cfg(feature = "prompt_skill")]
pub mod resolver;
#[cfg(feature = "prompt_skill")]
pub mod telemetry;
#[cfg(feature = "prompt_skill")]
pub mod tool;

pub use prompter::{FullDetailPrompter, LazySkillPrompter};
pub use registry::{SkillRegistry, SkillSummary};

#[cfg(feature = "prompt_skill")]
pub use catalog::{MAX_CATALOG_SKILLS, refresh_catalog, render_catalog, strip_catalog};
#[cfg(feature = "prompt_skill")]
pub use middleware::SkillCatalogRefreshMiddleware;
#[cfg(feature = "prompt_skill")]
pub use resolver::SkillResolver;
#[cfg(feature = "prompt_skill")]
pub use telemetry::SkillTelemetry;
#[cfg(feature = "prompt_skill")]
pub use tool::SkillTool;

// ── Skill parameter types ──

/// Parameter type for template-based skills.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum SkillParamType {
    String,
    Number,
    HostRef,
}

/// Parameter definition for a template-based skill.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SkillParam {
    pub name: String,
    pub description: String,
    pub param_type: SkillParamType,
    pub required: bool,
    pub default: Option<String>,
}

// ── Skill trait ──

pub trait Skill: Send + Sync {
    fn name(&self) -> &'static str;
    fn brief_description(&self) -> String;
    fn detailed_description(&self) -> String;

    /// Tools provided by knowledge-type skills.
    /// Template-type skills should return `vec![]`.
    fn tools(&self) -> Vec<Arc<dyn Tool>>;

    /// Generate PlanItems from this template skill with the given params.
    /// Template-type skills override this; knowledge-type skills return None.
    fn plan_steps(&self, _params: &HashMap<String, String>) -> Option<Vec<PlanItem>> {
        None
    }

    /// Parameter definitions for template-type skills.
    fn parameters(&self) -> &[SkillParam] {
        &[]
    }

    fn version(&self) -> &'static str {
        "0.1.0"
    }

    fn tags(&self) -> &[&'static str] {
        &[]
    }

    fn author(&self) -> &'static str {
        ""
    }

    fn category(&self) -> &'static str {
        ""
    }

    // ── Agent Skills open standard fields (agentskills.io) ──

    /// Tools allowed when this skill is active (empty = all tools allowed).
    fn allowed_tools(&self) -> &[String] {
        &[]
    }

    /// Tools disallowed when this skill is active (empty = no restrictions).
    fn disallowed_tools(&self) -> &[String] {
        &[]
    }

    /// Override the model selection. `None` means inherit from parent.
    fn model_override(&self) -> Option<&str> {
        None
    }

    /// Whether the user can manually invoke this skill via `/skill-name`.
    fn is_user_invocable(&self) -> bool {
        true
    }

    /// If true, the LLM cannot auto-trigger this skill (for side-effectful skills
    /// like deploy/commit). The user must explicitly invoke it.
    fn disable_model_invocation(&self) -> bool {
        false
    }

    /// If `Some("fork")`, the skill runs in an isolated sub-agent context.
    fn context_mode(&self) -> Option<&str> {
        None
    }

    /// Gitignore-style glob patterns. The skill is only activated when changed
    /// files match these patterns.
    fn path_patterns(&self) -> &[String] {
        &[]
    }

    /// The skill directory on disk (for `$PHI_SKILL_DIR` variable substitution).
    fn skill_dir(&self) -> Option<&Path> {
        None
    }

    /// The path to the skill's main source file (e.g. SKILL.md).
    /// Used in prompt-injection mode to tell the LLM where to read the full skill.
    fn source_path(&self) -> Option<&Path> {
        None
    }

    /// Read a file from the skill's `references/` directory.
    fn read_reference(&self, _relative_path: &str) -> Result<String, String> {
        Err("references not supported".into())
    }
}

// ── SkillPrompter trait ──

pub trait SkillPrompter: Send + Sync {
    /// Build the system prompt snippet for the given skills.
    /// `detail_tool_name` is the name of the tool the LLM can call to get
    /// detailed skill descriptions (used by `LazySkillPrompter`).
    fn build_prompt(&self, skills: &[Arc<dyn Skill>], detail_tool_name: &str) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSkill;

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            "test_skill"
        }

        fn brief_description(&self) -> String {
            "brief".to_string()
        }

        fn detailed_description(&self) -> String {
            "detailed".to_string()
        }

        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![]
        }
    }

    #[test]
    fn test_default_trait_methods() {
        let skill = TestSkill;
        let params: HashMap<String, String> = HashMap::new();
        assert!(skill.plan_steps(&params).is_none());
        assert!(skill.parameters().is_empty());
        assert_eq!(skill.version(), "0.1.0");
        assert!(skill.tags().is_empty());
        assert_eq!(skill.author(), "");
        assert_eq!(skill.category(), "");
        assert!(skill.allowed_tools().is_empty());
        assert!(skill.disallowed_tools().is_empty());
        assert_eq!(skill.model_override(), None);
        assert!(skill.is_user_invocable());
        assert!(!skill.disable_model_invocation());
        assert_eq!(skill.context_mode(), None);
        assert!(skill.path_patterns().is_empty());
        assert_eq!(skill.skill_dir(), None);
        assert_eq!(skill.source_path(), None);
        assert!(skill.read_reference("x").is_err());
    }

    #[test]
    fn test_skill_param_types() {
        assert_eq!(SkillParamType::String, SkillParamType::String);
        assert_ne!(SkillParamType::Number, SkillParamType::HostRef);

        let p = SkillParam {
            name: "x".to_string(),
            description: "d".to_string(),
            param_type: SkillParamType::String,
            required: true,
            default: None,
        };
        assert_eq!(p.name, "x");
        assert!(p.required);
        assert_eq!(p.default, None);
    }
}
