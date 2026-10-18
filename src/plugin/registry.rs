//! Install integrity registry (`<plugins_root>/registry.json`) — the pins live
//! outside the directory they protect.
//!
//! An install.json placed inside the plugin directory can be rewritten by the same
//! hand — anchor and anchored object in the same cave = self-deception. So the
//! authoritative pins live only in this registry, **outside the plugin directory**,
//! written exclusively by `plugin add` / `plugin remove`; no authoritative copy is
//! stored inside the plugin directory.
//!
//! What gets pinned (design section "Plugin install"): **manifest hash + capabilities
//! (paths/hosts resolved to absolute/normalized values) + the hash of every .wasm**.
//! Re-verified on every load:
//!
//! - edit the manifest / swap the wasm / add a tool → **post-install tampering**:
//!   all of that plugin's tools are fail-closed disabled + it reports "the plugin has
//!   been modified; run `plugin add` again to re-review";
//! - plugin.json lists a tool but the .wasm is missing → **corruption handling**:
//!   that tool is disabled + an error is reported; the whole package is not silently
//!   skipped (the other tools work as usual);
//! - capability tightening reads **only the pinned values in this registry**, never
//!   the plugin's own manifest.
//!
//! Secret rule: env/headers store only `${ENV:VAR}` references, **never** resolved
//! token literals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::manifest::ResolvedCapability;

/// Registry file name (`<plugins_root>/registry.json`).
pub const REGISTRY_FILE: &str = "registry.json";

/// Registry format version.
pub const REGISTRY_SCHEMA_VERSION: u32 = 1;

/// Registry read/write errors.
#[derive(Debug)]
pub enum RegistryError {
    /// IO failure.
    Io(std::io::Error),
    /// Corrupt JSON (**not** silently treated as empty — this is the user's install ledger).
    Json(String),
    /// Unrecognized `schemaVersion`.
    BadSchemaVersion(u32),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "registry io: {e}"),
            Self::Json(e) => write!(f, "registry.json is corrupt: {e}"),
            Self::BadSchemaVersion(v) => write!(f, "registry schemaVersion {v} is not supported"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Pinned values for one tool (enforcement accepts only these, not the plugin manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "lowercase")]
pub enum ToolPin {
    /// wasm form: absolute component path + content hash + resolved capabilities.
    Wasm {
        /// Identity key `plugin.{plugin}.{tool}`.
        identity: String,
        /// Absolute component path (pinned at install time).
        module: String,
        /// Component content sha256 (hex).
        wasm_sha256: String,
        /// Pinned capability allowlist values (paths absolute / hosts normalized).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        capabilities: Vec<ResolvedCapability>,
    },
    /// script form: pinned argv (absolute paths to prevent substitution).
    Script {
        /// Identity key `plugin.{plugin}.{tool}`.
        identity: String,
        /// argv (`run[0]`, or the interpreter's script argument, has been absolutized).
        run: Vec<String>,
    },
    /// mcp form: pinned config for an in-package server (owned by the plugin; not written to connections.json).
    Mcp {
        /// Server label (the `name` in the manifest).
        label: String,
        /// stdio form argv (in-package paths absolutized).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command: Option<Vec<String>>,
        /// stdio environment variable overrides (**store only `${ENV:VAR}` references**).
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
        /// Remote URL form.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Remote request headers (**store only `${ENV:VAR}` references**).
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
    },
}

impl ToolPin {
    /// Identity key (for the mcp form, the normalized placeholder key of the label, `plugin.{plugin}.{label}`).
    pub fn identity(&self, plugin: &str) -> String {
        match self {
            Self::Wasm { identity, .. } | Self::Script { identity, .. } => identity.clone(),
            Self::Mcp { label, .. } => format!("plugin.{plugin}.{label}"),
        }
    }

    /// Form name (wasm / script / mcp) — the error surface when a load-time pin doesn't match the manifest's form.
    pub fn form(&self) -> &'static str {
        match self {
            Self::Wasm { .. } => "wasm",
            Self::Script { .. } => "script",
            Self::Mcp { .. } => "mcp",
        }
    }
}

/// The pins for one installed plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginPin {
    /// Plugin name.
    pub name: String,
    /// Author's version (for audit display; integrity is judged by the hash).
    pub version: String,
    /// Source (path / git URL / archive URL) — for provenance, not a root of trust.
    pub source: String,
    /// sha256 (hex) of the raw plugin.json bytes.
    pub manifest_sha256: String,
    /// Per-tool pinned values.
    pub tools: Vec<ToolPin>,
}

/// Package-level integrity verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageStatus {
    /// manifest / wasm hashes match the pins.
    Trusted,
    /// **Post-install tampering**: manifest edited / wasm swapped / tool added → all tools disabled,
    /// "the plugin has been modified; run `plugin add` again to re-review".
    Tampered { reason: String },
}

/// Per-tool load status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    /// Loadable (hash matches, file present).
    Ready,
    /// Missing/corrupt → **that tool** is disabled + an error is reported; the whole package is not skipped.
    Broken { detail: String },
}

/// One re-verification finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyIssue {
    /// plugin.json content does not match the pin.
    ManifestHashMismatch,
    /// The tool set in the manifest does not match the pin (tool added / removed / renamed).
    ToolSetChanged { detail: String },
    /// Some .wasm content does not match the pin (wasm swapped).
    WasmHashMismatch { identity: String },
    /// Some .wasm is missing.
    MissingModule { identity: String, path: String },
}

/// The report of one load-time re-verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    /// Plugin name.
    pub plugin: String,
    /// Package-level verdict (Tampered ⇒ all of this plugin's tools are disabled).
    pub package: PackageStatus,
    /// Per-tool status (identity → status).
    pub tools: Vec<(String, ToolStatus)>,
    /// Finding list (for panel error display).
    pub issues: Vec<VerifyIssue>,
}

impl VerifyReport {
    /// Whether it was tampered with after install (all tools disabled + "re-run add to re-review").
    pub fn tampered(&self) -> bool {
        self.package != PackageStatus::Trusted
    }

    /// One-line panel message (design copy: "the plugin has been modified; run `plugin add` again to re-review").
    pub fn user_message(&self) -> Option<String> {
        match &self.package {
            PackageStatus::Tampered { reason } => Some(format!(
                "plugin `{}` has been modified ({reason}); run `plugin add` again to re-review",
                self.plugin
            )),
            PackageStatus::Trusted => None,
        }
    }
}

/// Central registry (`<plugins_root>/registry.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginRegistry {
    /// Format version.
    pub schema_version: u32,
    /// Plugin name → pin.
    pub plugins: BTreeMap<String, PluginPin>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Empty registry (schemaVersion 1).
    pub fn new() -> Self {
        Self {
            schema_version: REGISTRY_SCHEMA_VERSION,
            plugins: BTreeMap::new(),
        }
    }

    /// Read the registry. **Missing = empty table; corrupt = error** (never silently
    /// treated as empty — this is the user's install ledger, and silently clearing it
    /// would throw away the anchors of every reviewed plugin).
    pub fn load(path: &Path) -> Result<Self, RegistryError> {
        match std::fs::read_to_string(path) {
            Ok(raw) => {
                let registry: Self =
                    serde_json::from_str(&raw).map_err(|e| RegistryError::Json(e.to_string()))?;
                if registry.schema_version != REGISTRY_SCHEMA_VERSION {
                    return Err(RegistryError::BadSchemaVersion(registry.schema_version));
                }
                Ok(registry)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(RegistryError::Io(e)),
        }
    }

    /// Write back (pure JSON, machine-written; called only by `plugin add` / `plugin remove`).
    pub fn save(&self, path: &Path) -> Result<(), RegistryError> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir).map_err(RegistryError::Io)?;
        }
        let raw =
            serde_json::to_string_pretty(self).map_err(|e| RegistryError::Json(e.to_string()))?;
        std::fs::write(path, raw).map_err(RegistryError::Io)
    }

    /// Registry path (convenience: `registry.json` under plugins_root).
    pub fn path_in(plugins_root: &Path) -> PathBuf {
        plugins_root.join(REGISTRY_FILE)
    }

    /// Get one plugin's pin.
    pub fn get(&self, name: &str) -> Option<&PluginPin> {
        self.plugins.get(name)
    }

    /// Pin in / overwrite (for `plugin add` only).
    pub fn insert(&mut self, pin: PluginPin) {
        self.plugins.insert(pin.name.clone(), pin);
    }

    /// Pull one plugin's pin (for `plugin remove` only).
    pub fn remove(&mut self, name: &str) -> Option<PluginPin> {
        self.plugins.remove(name)
    }

    /// Number of pinned plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Load-time re-verification: pins vs the plugin directory on disk.
    ///
    /// - manifest hash mismatch / tool set mismatch / any .wasm swapped → package-level
    ///   [`PackageStatus::Tampered`] (all tools disabled + "re-run add to re-review");
    /// - .wasm missing → per-tool [`ToolStatus::Broken`] (the package is not skipped);
    /// - "manifest unreadable but hash matches" cannot happen (hash = content), so an
    ///   unreadable manifest means tampering.
    pub fn verify(&self, plugin_dir: &Path) -> VerifyReport {
        // No file_name (root path) or no pin found by name → the tampering path.
        let Some(pin) = plugin_dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| self.plugins.get(n))
        else {
            let name = plugin_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            return VerifyReport {
                plugin: name,
                package: PackageStatus::Tampered {
                    reason: "not installed (no registry pin)".into(),
                },
                tools: Vec::new(),
                issues: vec![VerifyIssue::ManifestHashMismatch],
            };
        };
        self.verify_pin(pin, plugin_dir)
    }

    /// Re-verify an installed plugin by name (used by the loader).
    pub fn verify_plugin(&self, name: &str, plugin_dir: &Path) -> VerifyReport {
        let Some(pin) = self.plugins.get(name) else {
            return VerifyReport {
                plugin: name.to_string(),
                package: PackageStatus::Tampered {
                    reason: "not installed (no registry pin)".into(),
                },
                tools: Vec::new(),
                issues: vec![VerifyIssue::ManifestHashMismatch],
            };
        };
        self.verify_pin(pin, plugin_dir)
    }

    fn verify_pin(&self, pin: &PluginPin, plugin_dir: &Path) -> VerifyReport {
        let mut issues = Vec::new();
        let mut tool_status: BTreeMap<String, ToolStatus> = BTreeMap::new();

        // 1. manifest hash: manifest edited → tampered.
        let manifest_path = plugin_dir.join("plugin.json");
        let manifest_raw = std::fs::read(&manifest_path).ok();
        let manifest_ok = match &manifest_raw {
            Some(raw) if hex_sha256(raw) == pin.manifest_sha256 => true,
            Some(_) => {
                issues.push(VerifyIssue::ManifestHashMismatch);
                false
            }
            None => {
                issues.push(VerifyIssue::ManifestHashMismatch);
                false
            }
        };

        // 2. tool set consistency (tool added / removed / renamed → tampered).
        if manifest_ok
            && let Some(raw) = &manifest_raw
            && let Ok(text) = std::str::from_utf8(raw)
            && let Ok(manifest) = super::manifest::PluginManifest::from_json(text)
        {
            let manifest_keys: Vec<String> = manifest
                .tools
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            let pin_keys: Vec<String> = pin
                .tools
                .iter()
                .map(|t| match t {
                    ToolPin::Wasm { identity, .. } | ToolPin::Script { identity, .. } => {
                        // identity = plugin.{name}.{tool} → take the tool segment
                        identity.rsplit('.').next().unwrap_or_default().to_string()
                    }
                    ToolPin::Mcp { label, .. } => label.clone(),
                })
                .collect();
            if manifest_keys != pin_keys {
                issues.push(VerifyIssue::ToolSetChanged {
                    detail: format!("manifest tools {manifest_keys:?} vs pinned {pin_keys:?}"),
                });
            }
        }

        // 3. per tool: wasm hash (wasm swapped → tampered); file missing → that tool Broken.
        for tool in &pin.tools {
            let (identity, status) = match tool {
                ToolPin::Wasm {
                    identity,
                    module,
                    wasm_sha256,
                    ..
                } => {
                    let path = Path::new(module);
                    if !path.is_file() {
                        issues.push(VerifyIssue::MissingModule {
                            identity: identity.clone(),
                            path: module.clone(),
                        });
                        (
                            identity.clone(),
                            ToolStatus::Broken {
                                detail: format!("module missing: {module}"),
                            },
                        )
                    } else {
                        match std::fs::read(path) {
                            Ok(bytes) if hex_sha256(&bytes) == *wasm_sha256 => {
                                (identity.clone(), ToolStatus::Ready)
                            }
                            Ok(_) => {
                                issues.push(VerifyIssue::WasmHashMismatch {
                                    identity: identity.clone(),
                                });
                                (
                                    identity.clone(),
                                    ToolStatus::Broken {
                                        detail: "wasm hash mismatch".into(),
                                    },
                                )
                            }
                            Err(e) => (
                                identity.clone(),
                                ToolStatus::Broken {
                                    detail: format!("module unreadable: {e}"),
                                },
                            ),
                        }
                    }
                }
                ToolPin::Script { identity, run } => {
                    // Missing script file = corruption handling (that tool is disabled;
                    // the package is not skipped).
                    let missing = std::fs::metadata(&run[0])
                        .map(|m| !m.is_file())
                        .unwrap_or(true);
                    if missing {
                        (
                            identity.clone(),
                            ToolStatus::Broken {
                                detail: format!("script missing: {}", run[0]),
                            },
                        )
                    } else {
                        (identity.clone(), ToolStatus::Ready)
                    }
                }
                ToolPin::Mcp { label, .. } => {
                    (format!("plugin.{}.{}", pin.name, label), ToolStatus::Ready)
                }
            };
            tool_status.insert(identity, status);
        }

        // Package-level verdict: any mismatch of manifest / tool set / wasm hash = tampered (disable all).
        let tampered = issues.iter().any(|i| {
            matches!(
                i,
                VerifyIssue::ManifestHashMismatch
                    | VerifyIssue::ToolSetChanged { .. }
                    | VerifyIssue::WasmHashMismatch { .. }
            )
        });
        let package = if tampered {
            PackageStatus::Tampered {
                reason: match issues.first() {
                    Some(VerifyIssue::ManifestHashMismatch) => "manifest changed".into(),
                    Some(VerifyIssue::ToolSetChanged { detail }) => {
                        format!("tool set changed ({detail})")
                    }
                    Some(VerifyIssue::WasmHashMismatch { identity }) => {
                        format!("wasm replaced ({identity})")
                    }
                    _ => "integrity check failed".into(),
                },
            }
        } else {
            PackageStatus::Trusted
        };

        // On tampering, all tools are disabled (even if individual files are still present).
        let tools = pin
            .tools
            .iter()
            .map(|t| {
                let identity = t.identity(&pin.name);
                let status = if tampered {
                    ToolStatus::Broken {
                        detail: "plugin tampered; re-run plugin add".into(),
                    }
                } else {
                    tool_status.remove(&identity).unwrap_or(ToolStatus::Ready)
                };
                (identity, status)
            })
            .collect();

        VerifyReport {
            plugin: pin.name.clone(),
            package,
            tools,
            issues,
        }
    }
}

/// sha256 → hex (lowercase). The one canonical encoding for integrity hashes.
pub fn hex_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin_with_wasm(
        name: &str,
        manifest_raw: &str,
        wasm_path: &str,
        wasm_bytes: &[u8],
    ) -> PluginPin {
        PluginPin {
            name: name.into(),
            version: "0.1.0".into(),
            source: "path".into(),
            manifest_sha256: hex_sha256(manifest_raw.as_bytes()),
            tools: vec![ToolPin::Wasm {
                identity: format!("plugin.{name}.search"),
                module: wasm_path.into(),
                wasm_sha256: hex_sha256(wasm_bytes),
                capabilities: vec![ResolvedCapability::FsRead {
                    paths: vec!["/tmp/data".into()],
                }],
            }],
        }
    }

    fn manifest_raw() -> String {
        serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [{"name": "search", "form": "wasm", "module": "wasm/search.wasm", "description": "d"}]
        })
        .to_string()
    }

    #[test]
    fn registry_round_trips_json() {
        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            "/abs/search.wasm",
            b"wasm!",
        ));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(REGISTRY_FILE);
        reg.save(&path).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["plugins"]["my-tools"]["version"], "0.1.0");
        assert_eq!(parsed["plugins"]["my-tools"]["tools"][0]["form"], "wasm");
        assert_eq!(
            parsed["plugins"]["my-tools"]["tools"][0]["capabilities"][0]["paths"][0],
            "/tmp/data"
        );

        let back = PluginRegistry::load(&path).unwrap();
        assert_eq!(back, reg);
    }

    #[test]
    fn load_missing_is_fresh_but_corrupt_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(REGISTRY_FILE);
        assert!(PluginRegistry::load(&path).unwrap().is_empty());

        std::fs::write(&path, "{broken").unwrap();
        assert!(matches!(
            PluginRegistry::load(&path),
            Err(RegistryError::Json(_))
        ));
    }

    #[test]
    fn verify_trusted_when_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let plugin_dir = dir.path().join("my-tools");
        std::fs::create_dir_all(plugin_dir.join("wasm")).unwrap();
        std::fs::write(plugin_dir.join("plugin.json"), manifest_raw()).unwrap();
        std::fs::write(plugin_dir.join("wasm/search.wasm"), b"wasm!").unwrap();

        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            plugin_dir.join("wasm/search.wasm").to_str().unwrap(),
            b"wasm!",
        ));

        let report = reg.verify(&plugin_dir);
        assert!(!report.tampered());
        assert_eq!(report.package, PackageStatus::Trusted);
        assert_eq!(
            report.tools,
            vec![("plugin.my-tools.search".into(), ToolStatus::Ready)]
        );
    }

    #[test]
    fn verify_tampered_on_manifest_edit() {
        // Edit the manifest (add capability/add tool) → tampered → disable all + "re-run add to re-review".
        let dir = tempfile::tempdir().unwrap();
        let plugin_dir = dir.path().join("my-tools");
        std::fs::create_dir_all(plugin_dir.join("wasm")).unwrap();
        std::fs::write(plugin_dir.join("wasm/search.wasm"), b"wasm!").unwrap();

        let tampered = serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [
                {"name": "search", "form": "wasm", "module": "wasm/search.wasm", "description": "d",
                 "capabilities": [{"type": "fs:read", "paths": ["/"]}]},
                {"name": "evil", "form": "script", "description": "d", "run": ["scripts/x.sh"]}
            ]
        })
        .to_string();
        std::fs::write(plugin_dir.join("plugin.json"), &tampered).unwrap();

        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            plugin_dir.join("wasm/search.wasm").to_str().unwrap(),
            b"wasm!",
        ));

        let report = reg.verify(&plugin_dir);
        assert!(report.tampered());
        let msg = report.user_message().unwrap();
        assert!(msg.contains("plugin add"), "{msg}");
        // All tools are disabled (even the one that was not swapped).
        assert!(
            report
                .tools
                .iter()
                .all(|(_, s)| matches!(s, ToolStatus::Broken { .. }))
        );
    }

    #[test]
    fn verify_tampered_on_wasm_swap() {
        let dir = tempfile::tempdir().unwrap();
        let plugin_dir = dir.path().join("my-tools");
        std::fs::create_dir_all(plugin_dir.join("wasm")).unwrap();
        std::fs::write(plugin_dir.join("plugin.json"), manifest_raw()).unwrap();
        let wasm_path = plugin_dir.join("wasm/search.wasm");
        std::fs::write(&wasm_path, b"EVIL!!").unwrap(); // swap the wasm

        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            wasm_path.to_str().unwrap(),
            b"wasm!",
        ));

        let report = reg.verify(&plugin_dir);
        assert!(report.tampered());
        assert!(report.user_message().unwrap().contains("plugin add"));
    }

    #[test]
    fn verify_missing_wasm_disables_one_tool_not_package() {
        // Corruption handling: .wasm missing → that tool is disabled + an error is
        // reported; the package is not skipped.
        let dir = tempfile::tempdir().unwrap();
        let plugin_dir = dir.path().join("my-tools");
        std::fs::create_dir_all(plugin_dir.join("wasm")).unwrap();
        std::fs::write(plugin_dir.join("plugin.json"), manifest_raw()).unwrap();
        // search.wasm does not exist.

        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            plugin_dir.join("wasm/search.wasm").to_str().unwrap(),
            b"wasm!",
        ));

        let report = reg.verify(&plugin_dir);
        assert!(!report.tampered(), "missing is corruption, not tampering");
        assert_eq!(report.package, PackageStatus::Trusted);
        assert!(matches!(
            &report.tools[0],
            (id, ToolStatus::Broken { detail }) if id == "plugin.my-tools.search" && detail.contains("missing")
        ));
        assert!(matches!(
            report.issues[0],
            VerifyIssue::MissingModule { .. }
        ));
    }

    #[test]
    fn verify_unpinned_plugin_is_tampered() {
        let dir = tempfile::tempdir().unwrap();
        let plugin_dir = dir.path().join("stranger");
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(plugin_dir.join("plugin.json"), manifest_raw()).unwrap();

        let reg = PluginRegistry::new();
        let report = reg.verify(&plugin_dir);
        assert!(report.tampered());
    }

    #[test]
    fn remove_clears_registry_entry() {
        let mut reg = PluginRegistry::new();
        reg.insert(pin_with_wasm(
            "my-tools",
            &manifest_raw(),
            "/abs/s.wasm",
            b"w",
        ));
        assert_eq!(reg.len(), 1);
        let removed = reg.remove("my-tools").unwrap();
        assert_eq!(removed.name, "my-tools");
        assert!(reg.is_empty());
        assert!(reg.remove("my-tools").is_none());
    }
}
