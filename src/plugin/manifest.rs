//! plugin.json manifest (schemaVersion 1) — parsing, fail-closed validation,
//! pre-install human review digest.
//!
//! The contract itself = `agent-works/schemas/plugin.schema.json` (a P0 asset;
//! the schema is the authority published to authors; this module is the runtime
//! fail-closed enforcement — both sides share the same law).
//!
//! - Name law / reserved words go through [`crate::tool_provider::ToolIdentity`]
//!   (defined in one place);
//! - Path law goes through [`crate::script_tool::resolve_under_root`] (resolve
//!   relative to the plugin root + reject escape from it), the same law as the
//!   global script anti-substitution rules;
//! - Form exclusivity is enforced per form by serde's `deny_unknown_fields`
//!   (wasm has no `run`, script has no `capabilities`, mcp has no
//!   `parameters`…);
//! - [`PluginManifest::review`] produces the pre-install review digest: full
//!   manifest + capability breakdown (**resolved concrete values** for
//!   paths/hosts) — nothing is written to disk before a human signs off
//!   (product-level gate).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::script_tool::{
    ExposureSpec, ParamSpec, ScriptLawError, pin_from_plugin_root, resolve_under_root,
};
use crate::tool_provider::{IdentityError, ToolIdentity};

/// Default timeout (seconds) for the wasm form.
pub const WASM_DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Manifest format version (schemaVersion starts at 1; evolution bumps the
/// version instead of changing semantics).
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Manifest parse/validation errors. fail-closed: reject anything invalid, no
/// compromises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// Wrong JSON syntax or shape (serde).
    Json(String),
    /// `schemaVersion` is not 1.
    BadSchemaVersion(u32),
    /// Plugin name or tool name is invalid (charset / reserved word / too long).
    BadName { name: String, source: IdentityError },
    /// `tools` is empty.
    EmptyTools,
    /// Duplicate tool name (including the mcp server label).
    DuplicateTool { name: String },
    /// `timeout_secs` is 0 (schema minimum 1).
    BadTimeout { tool: String },
    /// `exposure` uses a tier outside the schema (v1 has only direct/deferred).
    BadExposure { tool: String },
    /// Invalid capability entry (`fs:read` without paths / `net:http` without
    /// hosts / empty list).
    BadCapability { tool: String, reason: String },
    /// Invalid mcp `server` (command/url not mutually exclusive, env/headers on
    /// the wrong side, url shape).
    BadServer { tool: String, reason: String },
    /// Path-law violation (relative path escaping the plugin root, etc.).
    PathLaw {
        tool: String,
        source: ScriptLawError,
    },
    /// `version` is empty.
    EmptyVersion,
    /// `skills` entry is an empty string or duplicated.
    BadSkills { reason: String },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "plugin.json parse error: {e}"),
            Self::BadSchemaVersion(v) => write!(
                f,
                "plugin.json schemaVersion {v} is not supported (this build understands {MANIFEST_SCHEMA_VERSION})"
            ),
            Self::BadName { name, source } => write!(f, "invalid name `{name}`: {source}"),
            Self::EmptyTools => write!(f, "plugin.json must declare at least one tool"),
            Self::DuplicateTool { name } => {
                write!(f, "duplicate tool name `{name}` in plugin.json")
            }
            Self::BadTimeout { tool } => {
                write!(f, "tool `{tool}`: timeout_secs must be >= 1")
            }
            Self::BadExposure { tool } => {
                write!(f, "tool `{tool}`: exposure must be `direct` or `deferred`")
            }
            Self::BadCapability { tool, reason } => {
                write!(f, "tool `{tool}`: invalid capability ({reason})")
            }
            Self::BadServer { tool, reason } => {
                write!(f, "tool `{tool}`: invalid mcp server spec ({reason})")
            }
            Self::PathLaw { tool, source } => write!(f, "tool `{tool}`: {source}"),
            Self::EmptyVersion => write!(f, "plugin.json `version` must not be empty"),
            Self::BadSkills { reason } => write!(f, "plugin.json `skills`: {reason}"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Payload of a `fs:read` capability entry (`paths`; extra fields rejected —
/// strict oneOf law).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FsReadSpec {
    /// Readable directories/files relative to the plugin root.
    pub paths: Vec<String>,
}

/// Payload of a `net:http` capability entry (`hosts`; extra fields rejected —
/// strict oneOf law).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetHttpSpec {
    /// Outbound HTTP host allow-list.
    pub hosts: Vec<String>,
}

/// Capability allow-list entry (declaration form; fail-closed, undeclared =
/// not granted). wasm form only.
///
/// `fs:read` must come with `paths` (WASI preopens are path-scoped: no preopen
/// = can read nothing, preopening `/` = giving everything); `net:http` must
/// come with a `hosts` allow-list (filtered at outbound-request time, not on
/// the list = rejected). Cross fields (fs with hosts / net with paths) are
/// enforced by the payload structs' `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CapabilitySpec {
    /// Readable directories/files relative to the plugin root (resolved to
    /// absolute paths at install time and pinned into the registry).
    #[serde(rename = "fs:read")]
    FsRead(FsReadSpec),
    /// Outbound HTTP host allow-list (normalized and pinned into the registry at
    /// install time).
    #[serde(rename = "net:http")]
    NetHttp(NetHttpSpec),
}

/// Capability allow-list entry (resolved form: paths absolutized / hosts
/// normalized) — the pinned values in the registry.
///
/// Capability enforcement **reads only this** (the external registry), never
/// the plugin's own manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResolvedCapability {
    /// Absolute path list (pinned at install time via [`resolve_under_root`]).
    #[serde(rename = "fs:read")]
    FsRead { paths: Vec<String> },
    /// Normalized host allow-list (lowercased, whitespace stripped).
    #[serde(rename = "net:http")]
    NetHttp { hosts: Vec<String> },
}

/// Resolve a batch of capability declarations (relative to `plugin_root`).
pub fn resolve_capabilities(
    caps: &[CapabilitySpec],
    plugin_root: &Path,
) -> Result<Vec<ResolvedCapability>, ScriptLawError> {
    caps.iter()
        .map(|c| match c {
            CapabilitySpec::FsRead(spec) => {
                let mut resolved = Vec::with_capacity(spec.paths.len());
                for p in &spec.paths {
                    resolved.push(resolve_under_root(plugin_root, p)?);
                }
                Ok(ResolvedCapability::FsRead { paths: resolved })
            }
            CapabilitySpec::NetHttp(spec) => Ok(ResolvedCapability::NetHttp {
                hosts: spec
                    .hosts
                    .iter()
                    .map(|h| h.trim().to_ascii_lowercase())
                    .collect(),
            }),
        })
        .collect()
}

/// In-package MCP server (stdio command or remote URL; env/headers values
/// reference secrets only as `${ENV:VAR}`, resolved at product-layer
/// registration).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerSpec {
    /// stdio form: argv array, no shell. In-package relative paths are pinned
    /// absolute at install time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    /// stdio environment overrides (inherit + override).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Remote URL form (`http(s)://`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Remote URL request headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// wasm-form tool entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WasmToolSpec {
    /// Identity-key tool segment.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// Component path (relative to the plugin root). WIT world `phi:tool@0.1.0`.
    pub module: String,
    /// Parameter sketch (the host validates against it before entering wasm).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, ParamSpec>,
    /// Capability allow-list (fail-closed, undeclared = not granted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<CapabilitySpec>,
    /// Timeout (seconds). Defaults to [`WASM_DEFAULT_TIMEOUT_SECS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Tool-table budget tier, default direct.
    #[serde(default)]
    pub exposure: ExposureSpec,
}

/// script-form tool entry (in-package relative paths pinned at install time,
/// the same law as global scripts).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptToolSpec {
    /// Identity-key tool segment.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// argv array, `${param}` substituted per element, no shell.
    pub run: Vec<String>,
    /// Parameter sketch.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, ParamSpec>,
    /// Timeout (seconds). Defaults to 60s ([`crate::script_tool::DEFAULT_TIMEOUT_MS`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Tool-table budget tier, default direct.
    #[serde(default)]
    pub exposure: ExposureSpec,
}

/// mcp-form tool entry. `name` is a **server label** (not a tool name); each
/// tool exposed by that server gets one `plugin.{plugin}.{tool-name}` key
/// (normalized at load time).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolSpec {
    /// Server label (unique within the manifest; not part of the identity key).
    pub name: String,
    /// In-package MCP server (owned by the plugin, not written to
    /// connections.json).
    pub server: McpServerSpec,
    /// Timeout (seconds; reserved for call-level timeouts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Tool-table budget tier, default direct.
    #[serde(default)]
    pub exposure: ExposureSpec,
}

/// One tool line in plugin.json (the form is decided by `form`).
///
/// Form exclusivity (wasm without run/server, script without
/// module/server/capabilities, mcp without
/// module/run/capabilities/parameters/description) is enforced by each form
/// struct's `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub enum PluginToolSpec {
    /// In-process WASM sandbox (WIT `phi:tool@0.1.0`).
    Wasm(WasmToolSpec),
    /// argv-spawned script.
    Script(ScriptToolSpec),
    /// In-package MCP server (the answer for multi-language).
    Mcp(McpToolSpec),
}

impl PluginToolSpec {
    /// The `name` field in the manifest (wasm/script = tool name; mcp = server
    /// label).
    pub fn name(&self) -> &str {
        match self {
            Self::Wasm(t) => &t.name,
            Self::Script(t) => &t.name,
            Self::Mcp(t) => &t.name,
        }
    }

    /// Form name (for the review digest / registry).
    pub fn form(&self) -> &'static str {
        match self {
            Self::Wasm(_) => "wasm",
            Self::Script(_) => "script",
            Self::Mcp(_) => "mcp",
        }
    }

    /// Timeout (seconds); `None` if unset (the runtime uses the form's default).
    pub fn timeout_secs(&self) -> Option<u64> {
        match self {
            Self::Wasm(t) => t.timeout_secs,
            Self::Script(t) => t.timeout_secs,
            Self::Mcp(t) => t.timeout_secs,
        }
    }

    /// Identity key. For wasm/script it is directly `plugin.{plugin}.{name}`;
    /// for the mcp form the entry name is a server label, and the real tool keys
    /// are generated at load time from the server's tool names — the label key
    /// returned here is only used for de-duplication within the manifest and
    /// review annotations.
    pub fn identity(&self, plugin_name: &str) -> Result<String, IdentityError> {
        ToolIdentity::plugin(plugin_name, self.name()).map(|id| id.to_key())
    }
}

/// Runtime shape of plugin.json (schemaVersion 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    /// Manifest format version (only [`MANIFEST_SCHEMA_VERSION`] is accepted).
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    /// Plugin name (`[a-z0-9_-]{1,32}`; hitting a reserved word is rejected).
    pub name: String,
    /// Author-declared version (integrity is governed by the manifest hash, not
    /// by this field).
    pub version: String,
    /// Tools contributed by this plugin (at least one).
    pub tools: Vec<PluginToolSpec>,
    /// Skill root directory inside the plugin (relative to the plugin root).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<String>,
}

/// One line of the pre-install review digest: the full picture of one tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewTool {
    /// Identity key (the label key for the mcp form; server tools are normalized
    /// separately at load time).
    pub identity: String,
    /// Form (wasm / script / mcp).
    pub form: &'static str,
    /// Execution summary (script = pinned argv; wasm = absolute component path;
    /// mcp = command/URL).
    pub summary: String,
    /// Timeout (seconds; `None` = the form's default).
    pub timeout_secs: Option<u64>,
    /// This tool's capability breakdown (**resolved** concrete paths/hosts).
    pub capabilities: Vec<ResolvedCapability>,
}

/// Pre-install review digest: the full manifest picture (tools / execution
/// summaries / capability breakdown).
///
/// "Write to disk only after human confirmation": the product layer renders
/// this, and only after a human nods is
/// [`crate::plugin::install::commit_install`] allowed (plugins requested by an
/// agent are never allowed through by default — guaranteed by the product
/// layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewDigest {
    /// Plugin name.
    pub name: String,
    /// Author version.
    pub version: String,
    /// Full picture per tool.
    pub tools: Vec<ReviewTool>,
    /// Skill root directories (relative to the plugin root).
    pub skills: Vec<String>,
}

impl PluginManifest {
    /// Parse + fail-closed validation (serde shape + name law + form law +
    /// capability law).
    pub fn from_json(raw: &str) -> Result<Self, ManifestError> {
        let manifest: Self =
            serde_json::from_str(raw).map_err(|e| ManifestError::Json(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// fail-closed validation (rule by rule after parsing).
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::BadSchemaVersion(self.schema_version));
        }
        check_name(&self.name, &self.name)?;
        if self.version.trim().is_empty() {
            return Err(ManifestError::EmptyVersion);
        }
        if self.tools.is_empty() {
            return Err(ManifestError::EmptyTools);
        }

        let mut seen = BTreeSet::new();
        for tool in &self.tools {
            check_name(&self.name, tool.name())?;
            if !seen.insert(tool.name().to_string()) {
                return Err(ManifestError::DuplicateTool {
                    name: tool.name().to_string(),
                });
            }
            self.validate_tool(tool)?;
        }

        let mut skills_seen = BTreeSet::new();
        for skill in &self.skills {
            if skill.trim().is_empty() {
                return Err(ManifestError::BadSkills {
                    reason: "empty skill directory".into(),
                });
            }
            if !skills_seen.insert(skill.clone()) {
                return Err(ManifestError::BadSkills {
                    reason: format!("duplicate skill directory `{skill}`"),
                });
            }
        }
        Ok(())
    }

    fn validate_tool(&self, tool: &PluginToolSpec) -> Result<(), ManifestError> {
        let name = tool.name();
        if matches!(tool.timeout_secs(), Some(0)) {
            return Err(ManifestError::BadTimeout {
                tool: name.to_string(),
            });
        }
        match tool {
            PluginToolSpec::Wasm(t) => {
                if t.module.trim().is_empty() {
                    return Err(ManifestError::PathLaw {
                        tool: name.to_string(),
                        source: ScriptLawError::EmptyRun,
                    });
                }
                if t.exposure == ExposureSpec::Hidden {
                    return Err(ManifestError::BadExposure {
                        tool: name.to_string(),
                    });
                }
                for cap in &t.capabilities {
                    match cap {
                        CapabilitySpec::FsRead(spec) if spec.paths.is_empty() => {
                            return Err(ManifestError::BadCapability {
                                tool: name.to_string(),
                                reason: "fs:read needs at least one path".into(),
                            });
                        }
                        CapabilitySpec::NetHttp(spec) if spec.hosts.is_empty() => {
                            return Err(ManifestError::BadCapability {
                                tool: name.to_string(),
                                reason: "net:http needs at least one host".into(),
                            });
                        }
                        CapabilitySpec::NetHttp(spec)
                            if spec.hosts.iter().any(|h| h.trim().is_empty()) =>
                        {
                            return Err(ManifestError::BadCapability {
                                tool: name.to_string(),
                                reason: "net:http host must not be empty".into(),
                            });
                        }
                        _ => {}
                    }
                }
            }
            PluginToolSpec::Script(t) => {
                if t.exposure == ExposureSpec::Hidden {
                    return Err(ManifestError::BadExposure {
                        tool: name.to_string(),
                    });
                }
                // Here we only check the shape (non-empty); the path law
                // (`run[0]` absolute / interpreter allow-list) is checked in the
                // [`pin_from_plugin_root`] of review / install — in-plugin `run`
                // is **relative to the plugin root**, so before pinning it is
                // not an absolute path in the first place.
                if t.run.is_empty() {
                    return Err(ManifestError::PathLaw {
                        tool: name.to_string(),
                        source: ScriptLawError::EmptyRun,
                    });
                }
                if t.run.iter().any(|c| c.trim().is_empty()) {
                    return Err(ManifestError::PathLaw {
                        tool: name.to_string(),
                        source: ScriptLawError::EmptyRun,
                    });
                }
            }
            PluginToolSpec::Mcp(t) => {
                if t.exposure == ExposureSpec::Hidden {
                    return Err(ManifestError::BadExposure {
                        tool: name.to_string(),
                    });
                }
                validate_server(name, &t.server)?;
            }
        }
        Ok(())
    }

    /// Pre-install review digest: full manifest + capability breakdown (the
    /// resolved concrete values of paths/hosts).
    ///
    /// `plugin_root` is the **install target root** (`~/.phimint/plugins/<name>`)
    /// — the absolute paths in the digest are the final authorization values a
    /// human reviews. Path-law violations fail-closed here too.
    pub fn review(&self, plugin_root: &Path) -> Result<ReviewDigest, ManifestError> {
        let mut tools = Vec::with_capacity(self.tools.len());
        for tool in &self.tools {
            let identity = tool
                .identity(&self.name)
                .map_err(|source| ManifestError::BadName {
                    name: tool.name().to_string(),
                    source,
                })?;
            let (summary, capabilities) = match tool {
                PluginToolSpec::Wasm(t) => {
                    let module = resolve_under_root(plugin_root, &t.module).map_err(|source| {
                        ManifestError::PathLaw {
                            tool: t.name.clone(),
                            source,
                        }
                    })?;
                    let caps =
                        resolve_capabilities(&t.capabilities, plugin_root).map_err(|source| {
                            ManifestError::PathLaw {
                                tool: t.name.clone(),
                                source,
                            }
                        })?;
                    (format!("wasm module {module}"), caps)
                }
                PluginToolSpec::Script(t) => {
                    let run = pin_from_plugin_root(&t.run, plugin_root).map_err(|source| {
                        ManifestError::PathLaw {
                            tool: t.name.clone(),
                            source,
                        }
                    })?;
                    (run.join(" "), Vec::new())
                }
                PluginToolSpec::Mcp(t) => {
                    let summary = match (&t.server.command, &t.server.url) {
                        (Some(cmd), _) => format!("mcp stdio: {}", cmd.join(" ")),
                        (_, Some(url)) => format!("mcp http: {url}"),
                        _ => "mcp server (invalid)".into(),
                    };
                    (summary, Vec::new())
                }
            };
            tools.push(ReviewTool {
                identity,
                form: tool.form(),
                summary,
                timeout_secs: tool.timeout_secs(),
                capabilities,
            });
        }
        Ok(ReviewDigest {
            name: self.name.clone(),
            version: self.version.clone(),
            tools,
            skills: self.skills.clone(),
        })
    }
}

fn check_name(plugin_name: &str, segment: &str) -> Result<(), ManifestError> {
    ToolIdentity::plugin(plugin_name, segment)
        .map(|_| ())
        .map_err(|source| ManifestError::BadName {
            name: segment.to_string(),
            source,
        })
}

fn validate_server(tool: &str, server: &McpServerSpec) -> Result<(), ManifestError> {
    let bad = |reason: &str| ManifestError::BadServer {
        tool: tool.to_string(),
        reason: reason.to_string(),
    };
    match (&server.command, &server.url) {
        (Some(cmd), None) => {
            if cmd.is_empty() {
                return Err(bad("command must not be empty"));
            }
            if cmd.iter().any(|c| c.trim().is_empty()) {
                return Err(bad("command argv entries must not be empty"));
            }
            if !server.headers.is_empty() {
                return Err(bad("headers belong to the url form"));
            }
        }
        (None, Some(url)) => {
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                return Err(bad("url must be http(s)://"));
            }
            if !server.env.is_empty() {
                return Err(bad("env belongs to the stdio command form"));
            }
        }
        (Some(_), Some(_)) => return Err(bad("command and url are mutually exclusive")),
        (None, None) => return Err(bad("either command or url is required")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        // The platform absolute-path law is the same as Path::is_absolute
        // (runs on unix/windows alike).
        std::env::current_dir().unwrap().join("plugin-root-test")
    }

    fn minimal_json() -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 1,
            "name": "my-tools",
            "version": "0.1.0",
            "tools": [{
                "name": "deploy",
                "form": "script",
                "description": "deploy",
                "run": ["scripts/deploy.sh", "${env}"],
                "parameters": {"env": {"type": "string"}}
            }]
        })
    }

    #[test]
    fn parses_design_example_shape() {
        // The three-form example from the design doc "the shape of a plugin set".
        let raw = serde_json::json!({
            "schemaVersion": 1,
            "name": "my-tools",
            "version": "0.1.0",
            "tools": [
                {
                    "name": "search", "form": "wasm",
                    "module": "wasm/search.wasm",
                    "description": "search",
                    "parameters": {"q": {"type": "string", "description": "query"}},
                    "capabilities": [{"type": "fs:read", "paths": ["./data"]}],
                    "timeout_secs": 30
                },
                {
                    "name": "deploy", "form": "script",
                    "run": ["scripts/deploy.sh", "${env}"],
                    "description": "deploy",
                    "parameters": {"env": {"type": "string"}},
                    "timeout_secs": 120
                },
                {
                    "name": "wiki", "form": "mcp",
                    "server": {
                        "command": ["python", "server/main.py"],
                        "env": {"WIKI_TOKEN": "${ENV:WIKI_TOKEN}"}
                    }
                }
            ],
            "skills": ["skills"]
        });
        let manifest = PluginManifest::from_json(&raw.to_string()).expect("design example parses");
        assert_eq!(manifest.tools.len(), 3);
        assert_eq!(manifest.tools[0].form(), "wasm");
        assert_eq!(manifest.tools[1].form(), "script");
        assert_eq!(manifest.tools[2].form(), "mcp");
        assert_eq!(manifest.skills, vec!["skills"]);
    }

    #[test]
    fn rejects_bad_schema_version() {
        let mut raw = minimal_json();
        raw["schemaVersion"] = serde_json::json!(2);
        let err = PluginManifest::from_json(&raw.to_string()).unwrap_err();
        assert!(matches!(err, ManifestError::BadSchemaVersion(2)), "{err}");
    }

    #[test]
    fn rejects_reserved_plugin_name_and_bad_segments() {
        for name in [
            "mcp",
            "script",
            "plugin",
            "My Tools",
            "",
            "a".repeat(33).as_str(),
        ] {
            let mut raw = minimal_json();
            raw["name"] = serde_json::json!(name);
            let err = PluginManifest::from_json(&raw.to_string()).unwrap_err();
            assert!(
                matches!(err, ManifestError::BadName { .. } | ManifestError::Json(_)),
                "{name:?} → {err}"
            );
        }
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        let mut raw = minimal_json();
        raw["tools"]
            .as_array_mut()
            .unwrap()
            .push(minimal_json()["tools"][0].clone());
        let err = PluginManifest::from_json(&raw.to_string()).unwrap_err();
        assert!(
            matches!(err, ManifestError::DuplicateTool { ref name } if name == "deploy"),
            "{err}"
        );
    }

    #[test]
    fn rejects_form_field_bleed() {
        // Form exclusivity: script cannot carry capabilities / module / server.
        let mut raw = minimal_json();
        raw["tools"][0]["capabilities"] = serde_json::json!([{"type": "fs:read", "paths": ["."]}]);
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::Json(_))
        ));

        let mut raw = minimal_json();
        raw["tools"][0]["module"] = serde_json::json!("wasm/x.wasm");
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::Json(_))
        ));

        // mcp cannot carry description / parameters.
        let mut raw = minimal_json();
        raw["tools"][0] = serde_json::json!({
            "name": "wiki", "form": "mcp",
            "description": "nope",
            "server": {"url": "https://x"}
        });
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::Json(_))
        ));
    }

    #[test]
    fn rejects_bad_capability_shapes() {
        let mut raw = minimal_json();
        raw["tools"][0] = serde_json::json!({
            "name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d",
            "capabilities": [{"type": "fs:read", "paths": []}]
        });
        let err = PluginManifest::from_json(&raw.to_string()).unwrap_err();
        assert!(matches!(err, ManifestError::BadCapability { .. }), "{err}");

        // fs:read cannot carry hosts; net:http cannot carry paths (schema oneOf).
        let mut raw = minimal_json();
        raw["tools"][0] = serde_json::json!({
            "name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d",
            "capabilities": [{"type": "fs:read", "paths": ["."], "hosts": ["x"]}]
        });
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::Json(_))
        ));
    }

    #[test]
    fn rejects_bad_mcp_server_spec() {
        // command + url not mutually exclusive / neither present / env hung on
        // the url side.
        for server in [
            serde_json::json!({"command": ["python", "s.py"], "url": "https://x"}),
            serde_json::json!({}),
            serde_json::json!({"url": "https://x", "env": {"A": "b"}}),
            serde_json::json!({"url": "ftp://x"}),
            serde_json::json!({"command": []}),
        ] {
            let mut raw = minimal_json();
            raw["tools"][0] = serde_json::json!({
                "name": "wiki", "form": "mcp", "server": server
            });
            let err = PluginManifest::from_json(&raw.to_string()).unwrap_err();
            assert!(
                matches!(
                    err,
                    ManifestError::BadServer { .. } | ManifestError::Json(_)
                ),
                "{server} → {err}"
            );
        }
    }

    #[test]
    fn rejects_zero_timeout_and_hidden_exposure() {
        let mut raw = minimal_json();
        raw["tools"][0]["timeout_secs"] = serde_json::json!(0);
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::BadTimeout { .. })
        ));

        let mut raw = minimal_json();
        raw["tools"][0]["exposure"] = serde_json::json!("hidden");
        assert!(matches!(
            PluginManifest::from_json(&raw.to_string()),
            Err(ManifestError::BadExposure { .. })
        ));
    }

    #[test]
    fn review_shows_full_view_with_resolved_paths() {
        let raw = serde_json::json!({
            "schemaVersion": 1,
            "name": "my-tools",
            "version": "0.1.0",
            "tools": [
                {
                    "name": "search", "form": "wasm",
                    "module": "wasm/search.wasm",
                    "description": "search",
                    "capabilities": [
                        {"type": "fs:read", "paths": ["./data"]},
                        {"type": "net:http", "hosts": ["API.Example.com"]}
                    ],
                    "timeout_secs": 30
                },
                {
                    "name": "deploy", "form": "script",
                    "run": ["scripts/deploy.sh", "--env", "${env}"],
                    "description": "deploy"
                }
            ]
        });
        let manifest = PluginManifest::from_json(&raw.to_string()).unwrap();
        let root = root();
        let digest = manifest.review(&root).unwrap();

        assert_eq!(digest.name, "my-tools");
        assert_eq!(digest.tools.len(), 2);

        // wasm: the capability breakdown must be resolved concrete values
        // (absolute paths / lowercase hosts).
        let wasm = &digest.tools[0];
        assert_eq!(wasm.identity, "plugin.my-tools.search");
        let data_abs = root.join("data").to_string_lossy().into_owned();
        assert_eq!(
            wasm.capabilities,
            vec![
                ResolvedCapability::FsRead {
                    paths: vec![data_abs]
                },
                ResolvedCapability::NetHttp {
                    hosts: vec!["api.example.com".into()]
                }
            ]
        );

        // script: the execution summary = the pinned argv (absolute paths).
        let script = &digest.tools[1];
        assert_eq!(script.identity, "plugin.my-tools.deploy");
        let run_abs = root
            .join("scripts/deploy.sh")
            .to_string_lossy()
            .into_owned();
        assert_eq!(script.summary, format!("{run_abs} --env ${{env}}"));
    }

    #[test]
    fn review_rejects_escaping_paths() {
        // Capability/script paths escaping the plugin root → rejected (same law
        // as anti-substitution).
        let raw = serde_json::json!({
            "schemaVersion": 1,
            "name": "my-tools",
            "version": "0.1.0",
            "tools": [{
                "name": "search", "form": "wasm",
                "module": "wasm/search.wasm", "description": "d",
                "capabilities": [{"type": "fs:read", "paths": ["../outside"]}]
            }]
        });
        let manifest = PluginManifest::from_json(&raw.to_string()).unwrap();
        let err = manifest.review(&root()).unwrap_err();
        assert!(matches!(err, ManifestError::PathLaw { .. }), "{err}");
    }

    #[test]
    fn round_trips_through_json() {
        let manifest = PluginManifest::from_json(&minimal_json().to_string()).unwrap();
        let back: PluginManifest =
            serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
        assert_eq!(manifest, back);
    }
}
