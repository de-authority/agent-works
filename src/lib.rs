#![allow(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "skill")]
pub mod skill;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "memory")]
pub mod tools;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "focus")]
pub mod focus;

#[cfg(feature = "multi_agent")]
pub mod multi_agent;

#[cfg(feature = "compression")]
pub mod compression;

pub mod approval_ledger;
pub mod env_ref;
pub mod guard;
pub mod prompt;
pub mod rotation_policy;
pub mod script_tool;
pub mod tool_provider;

pub mod plugin;

mod builder;
pub mod handle;

#[cfg(feature = "skill")]
pub use crate::builder::SkillDetailToolFactory;
#[cfg(feature = "memory")]
pub use crate::builder::build_memory_system_prompt_with_config;
pub use crate::builder::{AgentBuilder, build_memory_system_prompt};
#[cfg(feature = "multi_agent")]
pub use crate::builder::{
    MultiAgentToolFactory, build_multi_agent_system_prompt, setup_multi_agent,
};
pub use crate::handle::{AgentHandle, SendError};
#[cfg(feature = "memory")]
pub use crate::memory::{
    CLAUDE_COMPATIBLE_TEMPLATE, DEFAULT_INDEX_FILENAME, MEMORY_TYPES, MemoryConfig, MemoryDoc,
    MemoryFrontmatter, MemoryStore, is_valid_memory_name, now_iso8601, parse_memory_file,
    parse_memory_str, project_slug, rebuild_index, render_memory_file, validate_memory_name,
    validate_memory_type,
};
pub use crate::prompt::{
    DynamicToolsFragment, EnvironmentFragment, FragmentContext, PromptFragment, compose_fragments,
};
#[cfg(feature = "memory")]
pub use crate::tools::{
    MemoryDeleteTool, MemoryListTool, MemoryReadTool, MemoryWriteTool, create_memory_tools,
};

#[cfg(feature = "skill")]
pub use skill::{
    FullDetailPrompter, LazySkillPrompter, Skill, SkillParam, SkillParamType, SkillRegistry,
    SkillSummary,
};

#[cfg(feature = "prompt_skill")]
pub use skill::prompt_skill::PromptSkill;

#[cfg(feature = "yaml_skill")]
pub use skill::yaml_skill::YamlSkill;

#[cfg(feature = "fuzzing")]
pub mod fuzz {
    #[cfg(feature = "mcp")]
    pub use super::mcp::server::fuzz_exports as mcp_server;
    #[cfg(feature = "skill")]
    pub use super::skill::prompt_skill::fuzz_exports as prompt_skill;
}
