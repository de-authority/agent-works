//! PromptSkill: implements `Skill` from a Markdown file with YAML frontmatter.
//!
//! Prompt Skills are lightweight instruction files that get injected into the
//! system prompt to guide AI behavior. Compatible with Trae/Cursor/Claude rules format.
//!
//! Supports the [Agent Skills](https://agentskills.io) open standard
//! (Anthropic 2025.12).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::skill::{Skill, SkillParam};
use serde::Deserialize;

// ── Frontmatter schema ──

/// Skill argument/parameter placeholder definition.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillArg {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

/// Lifetime scope of a skill body once triggered (skill-lifetime spec D1-D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillScope {
    /// Default: the host injects the body into its system prompt for the rest
    /// of the session (matches the Claude Code ecosystem contract — loaded
    /// skill content stays in context).
    Session,
    /// One-shot: the host injects the body as an ephemeral turn input and
    /// strips it at turn end (the ephemeral-injection path).
    Turn,
}

impl SkillScope {
    /// Parse the raw frontmatter value. Missing/empty/unknown values fall back
    /// to [`SkillScope::Session`] — the ecosystem default; unknown values warn.
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::trim).filter(|s| !s.is_empty()) {
            None => Self::Session,
            Some("session") => Self::Session,
            Some("turn") => Self::Turn,
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "unknown skill scope in SKILL.md frontmatter, falling back to 'session'"
                );
                Self::Session
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub author: Option<String>,

    // ── Agent Skills open-standard fields (agentskills.io) ──
    /// Tools allowed when this skill is active (comma-separated or YAML list).
    /// Empty = all tools allowed.
    #[serde(default, alias = "allowed-tools")]
    pub allowed_tools: Vec<String>,

    /// Tools disallowed when this skill is active.
    #[serde(default, alias = "disallowed-tools")]
    pub disallowed_tools: Vec<String>,

    /// Override model selection. `None` / `"inherit"` = use parent model.
    #[serde(default)]
    pub model: Option<String>,

    /// Whether the user can manually invoke this skill via `/skill-name`.
    #[serde(default = "default_true", alias = "user-invocable")]
    pub user_invocable: bool,

    /// If true, the LLM cannot auto-trigger this skill (for side-effectful skills
    /// like deploy/commit).
    #[serde(default, alias = "disable-model-invocation")]
    pub disable_model_invocation: bool,

    /// Lifetime of the body once triggered. `session` (default): the host
    /// injects it into its system prompt until session end. `turn`: one-shot —
    /// the host injects it as an ephemeral turn input and strips it at turn
    /// end. Raw string; parsed by [`SkillScope::parse`] with warn-on-unknown so
    /// a typo can never fail the whole SKILL.md load.
    #[serde(default)]
    pub scope: Option<String>,

    /// Parameter placeholders (e.g. `[{name: branch, description: "Target branch"}]`).
    /// Body uses `$branch` to reference these.
    #[serde(default)]
    pub arguments: Vec<SkillArg>,

    /// If `"fork"`, the skill runs in an isolated sub-agent context.
    #[serde(default)]
    pub context: Option<String>,

    /// Gitignore-style glob patterns. The skill is only activated when changed
    /// files match these patterns.
    #[serde(default)]
    pub paths: Vec<String>,
}

// ── PromptSkill ──

/// A skill loaded from a Markdown file with YAML frontmatter.
///
/// Implements `Skill` — can be registered into `SkillRegistry`
/// and used for system prompt injection.
///
/// Supports two loading modes:
/// - `from_markdown(content)` — parse a single SKILL.md string (no directory context)
/// - `from_dir(path)` — load from a standard skill directory with SKILL.md,
///   optional scripts/, references/, and templates/ subdirectories
///
/// ## Memory note
///
/// `PromptSkill` uses `Box::leak` to convert `String` fields into `&'static str`
/// references required by the `Skill` trait. This is acceptable for one-time
/// startup loading (a few KB of "leaked" memory), but is **not** compatible with
/// frequent hot-reload cycles — each reload adds new strings to the leaked pool.
/// If you use `hot-reload`, prefer event-triggered rather than timer-based reloading.
#[derive(Debug)]
pub struct PromptSkill {
    frontmatter: SkillFrontmatter,
    /// The full Markdown content (including frontmatter)
    content: String,
    /// The Markdown body (excluding frontmatter)
    body: String,
    /// Directory on disk (set via `from_dir()`)
    skill_dir: Option<PathBuf>,
    /// Path to SKILL.md file (set via `from_dir()`)
    source_path: Option<PathBuf>,
    /// Leaked name for &'static str return
    static_name: &'static str,
    /// Leaked version for &'static str return
    static_version: &'static str,
    /// Leaked category for &'static str return
    static_category: &'static str,
    /// Leaked author for &'static str return
    static_author: &'static str,
    /// Leaked tags for &'static [&'static str] return
    static_tags: Vec<&'static str>,
    /// Leaked allowed_tools
    static_allowed_tools: Vec<String>,
    /// Leaked disallowed_tools
    static_disallowed_tools: Vec<String>,
    /// Leaked paths
    static_paths: Vec<String>,
}

impl PromptSkill {
    /// Parse a Markdown string with YAML frontmatter into a `PromptSkill`.
    ///
    /// Expected format:
    /// ```markdown
    /// ---
    /// name: my-skill
    /// description: A short description
    /// category: 运维
    /// tags: [tag1, tag2]
    /// version: "1.0"
    /// author: ops-agent
    /// ---
    ///
    /// # Skill content here
    /// ```
    pub fn from_markdown(content: &str) -> Result<Self, String> {
        let (frontmatter_str, body) = split_frontmatter(content)?;

        let frontmatter: SkillFrontmatter = serde_yaml::from_str(frontmatter_str)
            .map_err(|e| format!("Frontmatter parse error: {e}"))?;

        if frontmatter.name.is_empty() {
            return Err("Skill name is empty".to_string());
        }

        if frontmatter.description.is_empty() {
            return Err("Skill description is empty".to_string());
        }

        // Validate name format (alphanumeric + hyphens)
        if !frontmatter
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!(
                "Invalid skill name '{}': only alphanumeric, hyphens, and underscores allowed",
                frontmatter.name
            ));
        }

        let static_name: &'static str = Box::leak(frontmatter.name.clone().into_boxed_str());
        let static_version: &'static str = Box::leak(
            frontmatter
                .version
                .clone()
                .unwrap_or_else(|| "1.0".to_string())
                .into_boxed_str(),
        );
        let static_category: &'static str =
            Box::leak(frontmatter.category.clone().into_boxed_str());
        let static_author: &'static str = Box::leak(
            frontmatter
                .author
                .clone()
                .unwrap_or_default()
                .into_boxed_str(),
        );
        let static_tags: Vec<&'static str> = frontmatter
            .tags
            .iter()
            .map(|t| Box::leak(t.clone().into_boxed_str()) as &'static str)
            .collect();
        let static_allowed_tools: Vec<String> = frontmatter.allowed_tools.clone();
        let static_disallowed_tools: Vec<String> = frontmatter.disallowed_tools.clone();
        let static_paths: Vec<String> = frontmatter.paths.clone();

        Ok(Self {
            frontmatter,
            content: content.to_string(),
            body: body.to_string(),
            skill_dir: None,
            source_path: None,
            static_name,
            static_version,
            static_category,
            static_author,
            static_tags,
            static_allowed_tools,
            static_disallowed_tools,
            static_paths,
        })
    }

    /// The raw frontmatter (for introspection).
    pub fn frontmatter(&self) -> &SkillFrontmatter {
        &self.frontmatter
    }

    /// Lifetime scope once triggered (frontmatter `scope:`; missing/unknown
    /// values fall back to `session`).
    pub fn scope(&self) -> SkillScope {
        SkillScope::parse(self.frontmatter.scope.as_deref())
    }

    /// The Markdown body (without frontmatter).
    pub fn body(&self) -> &str {
        &self.body
    }

    /// The skill directory on disk, if loaded via `from_dir()`.
    pub fn skill_dir_path(&self) -> Option<&Path> {
        self.skill_dir.as_deref()
    }

    /// Path to the `scripts/` subdirectory, if it exists.
    pub fn scripts_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("scripts"))
            .filter(|p| p.is_dir())
    }

    /// Path to the `references/` subdirectory, if it exists.
    pub fn references_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("references"))
            .filter(|p| p.is_dir())
    }

    /// Path to the `templates/` subdirectory, if it exists.
    pub fn templates_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("templates"))
            .filter(|p| p.is_dir())
    }

    /// Load a skill from a standard directory structure.
    ///
    /// Expected layout:
    /// ```text
    /// skill-name/
    ///   SKILL.md          (required — YAML frontmatter + Markdown body)
    ///   scripts/          (optional)
    ///   references/       (optional)
    ///   templates/        (optional)
    /// ```
    ///
    /// The directory name must match the `name` field in SKILL.md frontmatter.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let skill_md = path.join("SKILL.md");

        if !skill_md.exists() {
            return Err(format!("SKILL.md not found in {}", path.display()));
        }

        let content = std::fs::read_to_string(&skill_md)
            .map_err(|e| format!("Failed to read {}: {}", skill_md.display(), e))?;

        let mut skill = Self::from_markdown(&content)?;

        // Validate directory name matches skill name
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str())
            && dir_name != skill.frontmatter.name
        {
            return Err(format!(
                "Directory name '{}' does not match skill name '{}'",
                dir_name, skill.frontmatter.name
            ));
        }

        skill.skill_dir = Some(path.to_path_buf());
        skill.source_path = Some(path.join("SKILL.md"));
        Ok(skill)
    }

    /// Scan a directory for skill subdirectories.
    ///
    /// Each subdirectory that contains a `SKILL.md` file is loaded as a
    /// [`PromptSkill`]. Subdirectories without `SKILL.md` are silently skipped.
    /// Subdirectories whose name starts with `_` are treated as disabled by
    /// user convention and silently skipped, even if they contain a SKILL.md.
    ///
    /// Returns an empty `Vec` if the directory does not exist.
    pub fn scan_dir(path: impl AsRef<Path>) -> Result<Vec<Self>, String> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Ok(Vec::new());
        }

        let mut skills = Vec::new();
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory {}: {}", path.display(), e))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let entry_path = entry.path();
            // `_`-prefixed directories are a user convention for disabled/hidden
            // skills (users share `~/.claude/skills` with other agents and park
            // skills there by renaming) — skip silently, no warning.
            if entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('_'))
            {
                continue;
            }
            if entry_path.is_dir() && entry_path.join("SKILL.md").exists() {
                match Self::from_dir(&entry_path) {
                    Ok(skill) => skills.push(skill),
                    Err(e) => {
                        tracing::warn!(
                            dir = %entry_path.display(),
                            error = %e,
                            "skipping invalid skill directory"
                        );
                    }
                }
            }
        }

        // Sort by name for deterministic ordering
        skills.sort_by(|a, b| a.frontmatter.name.cmp(&b.frontmatter.name));
        Ok(skills)
    }

    /// Resolve variables in the skill body.
    ///
    /// Supported variables:
    /// - `$ARGUMENTS` — raw text after `/skill-name`
    /// - `$name` — named parameter value (from frontmatter `arguments` list)
    /// - `$PHI_SKILL_DIR` — path to the skill directory (for referencing scripts/)
    pub fn resolve_body(&self, params: &HashMap<String, String>, raw_arguments: &str) -> String {
        let mut result = self.body.clone();

        // $ARGUMENTS
        result = result.replace("$ARGUMENTS", raw_arguments);

        // $PHI_SKILL_DIR
        if let Some(dir) = &self.skill_dir {
            result = result.replace("$PHI_SKILL_DIR", &dir.to_string_lossy());
        }

        // Named parameters: $param_name
        // Sort by key length descending to prevent prefix clobbering.
        // Example: with keys ["host", "hostname"], $hostname must be substituted
        // before $host, otherwise $hostname becomes "Xname" instead of "Y".
        let mut sorted_keys: Vec<&String> = params.keys().collect();
        sorted_keys.sort_by_key(|b| std::cmp::Reverse(b.len()));
        for key in sorted_keys {
            let value = &params[key];
            let placeholder = format!("${}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }
}

/// Split a Markdown file into (frontmatter, body) at the `---` delimiters.
fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        return Err("Missing YAML frontmatter (file must start with ---)".to_string());
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let closing = after_first
        .find("\n---")
        .or_else(|| after_first.find("\r\n---"))
        .ok_or("Missing closing --- for frontmatter")?;

    let frontmatter_str = &after_first[..closing];
    let body_start = closing + 4; // skip past "\n---"
    let body = after_first[body_start..]
        .trim_start_matches('\n')
        .trim_start_matches('\r');

    Ok((frontmatter_str, body))
}

impl Skill for PromptSkill {
    fn name(&self) -> &'static str {
        self.static_name
    }

    fn brief_description(&self) -> String {
        self.frontmatter.description.clone()
    }

    fn detailed_description(&self) -> String {
        self.content.clone()
    }

    fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
        vec![]
    }

    fn parameters(&self) -> &[SkillParam] {
        &[]
    }

    fn version(&self) -> &'static str {
        self.static_version
    }

    fn tags(&self) -> &[&'static str] {
        &self.static_tags
    }

    fn author(&self) -> &'static str {
        self.static_author
    }

    fn category(&self) -> &'static str {
        self.static_category
    }

    fn allowed_tools(&self) -> &[String] {
        &self.static_allowed_tools
    }

    fn disallowed_tools(&self) -> &[String] {
        &self.static_disallowed_tools
    }

    fn model_override(&self) -> Option<&str> {
        self.frontmatter
            .model
            .as_deref()
            .filter(|m| *m != "inherit")
    }

    fn is_user_invocable(&self) -> bool {
        self.frontmatter.user_invocable
    }

    fn disable_model_invocation(&self) -> bool {
        self.frontmatter.disable_model_invocation
    }

    fn context_mode(&self) -> Option<&str> {
        self.frontmatter.context.as_deref()
    }

    fn path_patterns(&self) -> &[String] {
        &self.static_paths
    }

    fn skill_dir(&self) -> Option<&Path> {
        self.skill_dir.as_deref()
    }

    fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    fn read_reference(&self, relative_path: &str) -> Result<String, String> {
        let ref_dir = self.references_dir().ok_or_else(|| {
            format!(
                "Skill '{}' has no references/ directory",
                self.frontmatter.name
            )
        })?;

        // Prevent path traversal: canonicalize both paths for reliable comparison
        // (symlinks, case-insensitive FS, etc.)
        let ref_dir_canon = ref_dir
            .canonicalize()
            .map_err(|e| format!("Failed to resolve references/ directory: {}", e))?;

        let resolved = ref_dir.join(relative_path);
        let canonical = resolved.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve reference path '{}': {}",
                relative_path, e
            )
        })?;

        if !canonical.starts_with(&ref_dir_canon) {
            return Err(format!(
                "Path traversal denied: '{}' is outside references/",
                relative_path
            ));
        }

        std::fs::read_to_string(&canonical)
            .map_err(|e| format!("Failed to read reference '{}': {}", relative_path, e))
    }
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    pub fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
        super::split_frontmatter(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PROMPT_SKILL: &str = r#"---
name: test-prompt
description: A test prompt skill
category: 测试
tags: [test, example]
version: "1.0"
author: test
---

# Test Prompt Skill

## 角色
你是一个测试助手。

## 指导
- 始终用中文回复
- 保持简洁
"#;

    #[test]
    fn test_parse_markdown() {
        let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
        assert_eq!(skill.name(), "test-prompt");
        assert_eq!(skill.brief_description(), "A test prompt skill");
        assert_eq!(skill.category(), "测试");
        assert_eq!(skill.version(), "1.0");
        assert_eq!(skill.author(), "test");
        assert_eq!(skill.tags().len(), 2);
        assert!(skill.body().contains("# Test Prompt Skill"));
    }

    #[test]
    fn test_full_content() {
        let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
        let desc = skill.detailed_description();
        assert!(desc.contains("---"));
        assert!(desc.contains("name: test-prompt"));
        assert!(desc.contains("# Test Prompt Skill"));
    }

    #[test]
    fn test_no_frontmatter() {
        let md = "# Just a heading\nSome content";
        assert!(PromptSkill::from_markdown(md).is_err());
    }

    #[test]
    fn test_empty_name() {
        let md = "---\nname: ''\ndescription: test\n---\nContent";
        assert!(PromptSkill::from_markdown(md).is_err());
    }

    #[test]
    fn test_empty_description() {
        let md = "---\nname: test-skill\ndescription: ''\n---\nContent";
        assert!(PromptSkill::from_markdown(md).is_err());
    }

    #[test]
    fn test_invalid_name() {
        let md = "---\nname: invalid name!\ndescription: test\n---\nContent";
        assert!(PromptSkill::from_markdown(md).is_err());
    }

    #[test]
    fn test_minimal_frontmatter() {
        let md = "---\nname: minimal\ndescription: Minimal skill\n---\nBody here";
        let skill = PromptSkill::from_markdown(md).unwrap();
        assert_eq!(skill.name(), "minimal");
        assert_eq!(skill.version(), "1.0"); // default
        assert_eq!(skill.author(), ""); // default
        assert_eq!(skill.category(), ""); // default
        assert!(skill.tags().is_empty());
    }

    // ── Phase 4: new frontmatter fields ──

    const SKILL_WITH_NEW_FIELDS: &str = r#"---
name: advanced-skill
description: An advanced skill with new fields
category: ops
tags: [deploy, production]
version: "2.0"
author: ops-team
allowed-tools: [bash, git, docker]
disallowed-tools: [rm, drop]
model: opus
user-invocable: true
disable-model-invocation: true
arguments:
  - name: branch
    description: Target branch
    required: true
  - name: env
    description: Target environment
context: fork
paths: ["src/**", "deploy/**"]
---

# Advanced Skill

Deploy to $env on branch $branch.
Scripts are in $PHI_SKILL_DIR/scripts/
Full arguments: $ARGUMENTS
"#;

    #[test]
    fn test_new_frontmatter_fields() {
        let skill = PromptSkill::from_markdown(SKILL_WITH_NEW_FIELDS).unwrap();
        assert_eq!(skill.name(), "advanced-skill");

        // allowed-tools / disallowed-tools
        assert_eq!(skill.allowed_tools(), &["bash", "git", "docker"]);
        assert_eq!(skill.disallowed_tools(), &["rm", "drop"]);

        // model
        assert_eq!(skill.model_override(), Some("opus"));

        // user-invocable
        assert!(skill.is_user_invocable());

        // disable-model-invocation
        assert!(skill.disable_model_invocation());

        // context
        assert_eq!(skill.context_mode(), Some("fork"));

        // paths
        assert_eq!(skill.path_patterns(), &["src/**", "deploy/**"]);

        // Frontmatter introspection
        let fm = skill.frontmatter();
        assert_eq!(fm.arguments.len(), 2);
        assert_eq!(fm.arguments[0].name, "branch");
        assert!(fm.arguments[0].required);
        assert_eq!(fm.arguments[1].name, "env");
        assert!(!fm.arguments[1].required);
    }

    #[test]
    fn test_new_fields_defaults() {
        // Old format without new fields should still parse with defaults
        let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
        assert!(skill.allowed_tools().is_empty());
        assert!(skill.disallowed_tools().is_empty());
        assert_eq!(skill.model_override(), None);
        assert!(skill.is_user_invocable()); // default true
        assert!(!skill.disable_model_invocation()); // default false
        assert_eq!(skill.context_mode(), None);
        assert!(skill.path_patterns().is_empty());
    }

    #[test]
    fn test_model_inherit_treated_as_none() {
        let md = "---\nname: inherit-skill\ndescription: test\nmodel: inherit\n---\nBody";
        let skill = PromptSkill::from_markdown(md).unwrap();
        assert_eq!(skill.model_override(), None);
    }

    #[test]
    fn test_resolve_body() {
        let skill = PromptSkill::from_markdown(SKILL_WITH_NEW_FIELDS).unwrap();

        let mut params = HashMap::new();
        params.insert("branch".to_string(), "main".to_string());
        params.insert("env".to_string(), "staging".to_string());

        let resolved = skill.resolve_body(&params, "--force");

        assert!(resolved.contains("Deploy to staging on branch main."));
        assert!(resolved.contains("Full arguments: --force"));
        // $PHI_SKILL_DIR should NOT be replaced since skill was loaded from markdown (no dir)
        assert!(resolved.contains("$PHI_SKILL_DIR"));
    }

    #[test]
    fn test_skill_dir_none_when_from_markdown() {
        let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
        assert!(skill.skill_dir().is_none());
        assert!(skill.scripts_dir().is_none());
        assert!(skill.references_dir().is_none());
        assert!(skill.templates_dir().is_none());
    }

    #[test]
    fn test_read_reference_no_dir_returns_error() {
        let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
        let result = skill.read_reference("guide.md");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no references/"));
    }

    // ── from_dir / scan_dir tests ──

    #[test]
    fn test_from_dir_loads_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("test-skill");
        std::fs::create_dir(&skill_dir).unwrap();

        let skill_md = r#"---
name: test-skill
description: A skill from directory
---
# Skill Body
"#;
        std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();
        std::fs::create_dir(skill_dir.join("scripts")).unwrap();
        std::fs::create_dir(skill_dir.join("references")).unwrap();

        let skill = PromptSkill::from_dir(&skill_dir).unwrap();
        assert_eq!(skill.name(), "test-skill");
        assert_eq!(skill.brief_description(), "A skill from directory");
        assert!(skill.skill_dir().is_some());
        assert!(skill.scripts_dir().is_some());
        assert!(skill.references_dir().is_some());
        assert!(skill.templates_dir().is_none());
    }

    #[test]
    fn test_from_dir_name_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("wrong-name");
        std::fs::create_dir(&skill_dir).unwrap();

        let skill_md = r#"---
name: correct-name
description: Test
---
Body
"#;
        std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

        let result = PromptSkill::from_dir(&skill_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }

    #[test]
    fn test_from_dir_missing_skill_md() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("no-skill-md");
        std::fs::create_dir(&skill_dir).unwrap();

        let result = PromptSkill::from_dir(&skill_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SKILL.md not found"));
    }

    #[test]
    fn test_scan_dir_discovers_skills() {
        let dir = tempfile::tempdir().unwrap();

        // Create skill directories
        for name in &["deploy", "review", "test"] {
            let skill_dir = dir.path().join(name);
            std::fs::create_dir(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!(
                    "---\nname: {}\ndescription: {} skill\n---\nBody",
                    name, name
                ),
            )
            .unwrap();
        }

        // Create a non-skill directory (no SKILL.md)
        let misc_dir = dir.path().join("misc");
        std::fs::create_dir(&misc_dir).unwrap();

        let skills = PromptSkill::scan_dir(dir.path()).unwrap();
        assert_eq!(skills.len(), 3);

        let names: Vec<&str> = skills.iter().map(|s| s.name()).collect();
        assert_eq!(names, &["deploy", "review", "test"]); // sorted
    }

    #[test]
    fn test_scan_dir_skips_underscore_prefixed_dirs_silently() {
        let dir = tempfile::tempdir().unwrap();

        // A live skill alongside a `_`-parked one whose SKILL.md would fail
        // validation anyway (dir name != skill name). The `_` dir must be
        // skipped without an error and without entering the result.
        for (name, skill_name) in &[("deploy", "deploy"), ("_disabled", "realname")] {
            let skill_dir = dir.path().join(name);
            std::fs::create_dir(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: {skill_name}\ndescription: d\n---\nBody"),
            )
            .unwrap();
        }

        let skills = PromptSkill::scan_dir(dir.path()).unwrap();
        let names: Vec<&str> = skills.iter().map(|s| s.name()).collect();
        assert_eq!(names, &["deploy"]);
    }

    #[test]
    fn test_scan_dir_nonexistent_returns_empty() {
        let skills = PromptSkill::scan_dir("/nonexistent/path/12345").unwrap();
        assert!(skills.is_empty());
    }

    #[test]
    fn test_read_reference_success() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("ref-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        let refs_dir = skill_dir.join("references");
        std::fs::create_dir(&refs_dir).unwrap();

        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: ref-skill\ndescription: test\n---\nBody",
        )
        .unwrap();
        std::fs::write(
            refs_dir.join("guide.md"),
            "# Reference Guide\n\nImportant info.",
        )
        .unwrap();

        let skill = PromptSkill::from_dir(&skill_dir).unwrap();
        let content = skill.read_reference("guide.md").unwrap();
        assert!(content.contains("Reference Guide"));
        assert!(content.contains("Important info"));
    }

    #[test]
    fn test_read_reference_path_traversal_prevented() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("secure-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        let refs_dir = skill_dir.join("references");
        std::fs::create_dir(&refs_dir).unwrap();

        // Create a file INSIDE the skill dir but OUTSIDE references/
        // (e.g. skill_dir/secret.txt, accessible via ../secret.txt from references/)
        std::fs::write(skill_dir.join("secret.txt"), "top secret").unwrap();

        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: secure-skill\ndescription: test\n---\nBody",
        )
        .unwrap();

        let skill = PromptSkill::from_dir(&skill_dir).unwrap();
        let result = skill.read_reference("../secret.txt");
        assert!(
            result.is_err(),
            "expected error for path traversal, got {:?}",
            result.ok()
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("Path traversal") || err.contains("outside references"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_resolve_body_with_skill_dir() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("resolve-skill");
        std::fs::create_dir(&skill_dir).unwrap();

        let skill_md = r#"---
name: resolve-skill
description: test
---
Run scripts from $PHI_SKILL_DIR/scripts/
Arguments: $ARGUMENTS
"#;
        std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

        let skill = PromptSkill::from_dir(&skill_dir).unwrap();
        let params = HashMap::new();
        let resolved = skill.resolve_body(&params, "extra args");

        assert!(resolved.contains(&*skill_dir.to_string_lossy()));
        assert!(resolved.contains("extra args"));
        assert!(!resolved.contains("$PHI_SKILL_DIR"));
        assert!(!resolved.contains("$ARGUMENTS"));
    }

    #[test]
    fn test_resolve_body_prefix_clobbering_prevented() {
        // If $host is substituted before $hostname, "Connect to $hostname"
        // becomes "Connect to Xname" instead of "Connect to Y".
        let md = "---\nname: prefix-skill\ndescription: test\n---\nConnect to $hostname at $host";
        let skill = PromptSkill::from_markdown(md).unwrap();

        let mut params = HashMap::new();
        params.insert("host".to_string(), "192.168.1.1".to_string());
        params.insert("hostname".to_string(), "web-prod".to_string());

        let resolved = skill.resolve_body(&params, "");
        assert!(
            resolved.contains("Connect to web-prod at 192.168.1.1"),
            "expected correct substitution order, got: {}",
            resolved
        );
    }

    // ── proptest: split_frontmatter / from_markdown ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn from_markdown_never_panics(content in ".*") {
                let _ = PromptSkill::from_markdown(&content);
            }

            #[test]
            fn split_frontmatter_valid_input(
                fm_content in r"[a-z: \n]{5,100}",
                body_content in r"[a-zA-Z0-9 .,\n]{0,200}"
            ) {
                // Build valid markdown with frontmatter
                let name = "test-skill";
                let desc = "test description";
                let input = format!("---\nname: {}\ndescription: {}\n{}\n---\n{}", name, desc, fm_content, body_content);
                let result = split_frontmatter(&input);
                if let Ok((fm, body)) = result {
                    assert!(fm.contains("test-skill"), "frontmatter should contain name");
                    assert!(fm.contains("test description"), "frontmatter should contain description");
                    // Body should not contain the closing ---
                    assert!(!body.starts_with("---"), "body should not start with ---");
                }
            }

            #[test]
            fn from_markdown_preserves_body(
                body_text in r"[a-zA-Z0-9 .,\n]{1,200}"
            ) {
                let input = format!("---\nname: test-skill\ndescription: test desc\n---\n{}", body_text);
                if let Ok(skill) = PromptSkill::from_markdown(&input) {
                    // Body should contain the original text (trimmed)
                    let trimmed_body = body_text.trim();
                    if !trimmed_body.is_empty() {
                        assert!(skill.body().contains(trimmed_body),
                            "body {:?} should contain {:?}", skill.body(), trimmed_body);
                    }
                }
            }
        }
    }
}
