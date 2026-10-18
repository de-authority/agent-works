//! Plugin loading (the P2 load surface) — re-verification gate → build tools per form
//! → disable per tool → three places cleaned on unload.
//!
//! Criteria (`notes/tool-socket-design.md`, "Plugin package mechanism" acceptance):
//!
//! - **Post-install tampering is refused**: re-verification [`PackageStatus::Tampered`] ⇒ **all** tools
//!   of that plugin are disabled, reason = "…run `plugin add` again to re-review" ([`VerifyReport::user_message`]);
//! - **Corruption does not skip the whole package**: missing .wasm (registry reports Broken) or
//!   compile failure at load time ⇒ **that tool** is disabled, the rest of the same
//!   package work as usual;
//! - **Unload cleans three places**: the tool table ([`LoadedPlugin::identity_keys`] handed back to
//!   the caller for unregistering) / the ledger ([`crate::approval_ledger::ApprovalLedger::remove_prefix`]) /
//!   the registry ([`super::install::remove_plugin`]) — each has its own test;
//! - **The mcp form belongs to the plugin**: server pins live in [`PluginMcpServer`] (connections are
//!   made by the product layer, nothing written to connections.json); key assignment
//!   disambiguates collisions with numeric suffixes ([`assign_mcp_tool_keys`]).
//!
//! The load surface **does not touch the network and does not spawn**: MCP only registers
//! pins, scripts only register argv pins — the execution surfaces hang separately
//! ([`crate::script_tool::ScriptTool`] / [`super::wasm::WasmTool`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use agent_base::Tool;

use super::manifest::{
    PluginManifest, PluginToolSpec, ScriptToolSpec, WASM_DEFAULT_TIMEOUT_SECS, WasmToolSpec,
};
use super::registry::{PluginRegistry, ToolPin, ToolStatus, VerifyReport};
use crate::script_tool::{ScriptTool, ScriptToolEntry};
use crate::tool_provider::{NAME_SEGMENT_MAX, normalize_tool_segment};

/// A disabled tool (post-install tampering / missing file / compile failure / not
/// compiled in), with a reason the panel can report directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisabledTool {
    /// identity key `plugin.{plugin}.{tool}`.
    pub identity: String,
    /// human-readable reason (shown directly by the panel / `/tools`).
    pub reason: String,
}

/// Pin for a plugin-owned MCP server (nothing written to connections.json — it belongs
/// to the plugin).
///
/// env / headers only contain `${ENV:VAR}` references (the registry's law; the product
/// layer resolves and injects them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMcpServer {
    /// the server label from the manifest (not part of the identity key).
    pub label: String,
    /// stdio form argv (in-package paths already absolutized).
    pub command: Option<Vec<String>>,
    /// environment variable overrides (only `${ENV:VAR}` references are stored).
    pub env: BTreeMap<String, String>,
    /// remote URL form.
    pub url: Option<String>,
    /// remote request headers (only `${ENV:VAR}` references are stored).
    pub headers: BTreeMap<String, String>,
}

/// A loaded plugin package: usable tools + disabled list + re-verification report + own server pins.
pub struct LoadedPlugin {
    /// plugin name.
    pub name: String,
    /// re-verification report for this load (panel display / logging).
    pub verify: VerifyReport,
    /// usable tools (can be passed straight to `ToolRegistry::register_arc`).
    pub tools: Vec<Arc<dyn Tool>>,
    /// disabled tools (the product of not skipping the whole package on corruption).
    pub disabled: Vec<DisabledTool>,
    /// own MCP server pins (connections are made by the product layer).
    pub mcp_servers: Vec<PluginMcpServer>,
}

impl LoadedPlugin {
    /// All identity keys touched by this plugin (usable + disabled) — used to clean the
    /// tool table on unload.
    pub fn identity_keys(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|t| t.name().to_string())
            .chain(self.disabled.iter().map(|d| d.identity.clone()))
            .collect()
    }

    /// One-line panel message (the tamper text from the re-verification report; Trusted = none).
    pub fn user_message(&self) -> Option<String> {
        self.verify.user_message()
    }
}

/// Plugin loader: shares one wasm runtime across plugins (feature `wasm`).
#[derive(Default)]
pub struct PluginLoader {
    #[cfg(feature = "wasm")]
    wasm: Option<Arc<super::wasm::WasmRuntime>>,
}

impl PluginLoader {
    /// Without a wasm runtime (wasm-form tools are disabled with the reason stated).
    pub fn new() -> Self {
        Self::default()
    }

    /// With a wasm runtime (one per process, shared across plugins).
    #[cfg(feature = "wasm")]
    pub fn with_wasm(runtime: Arc<super::wasm::WasmRuntime>) -> Self {
        Self {
            wasm: Some(runtime),
        }
    }

    /// Load one plugin: re-verify → build tool by tool.
    ///
    /// - package-level tampering ⇒ disable everything (`tools` empty), reason = the
    ///   re-verification text;
    /// - a single tool missing/corrupt ⇒ that tool is disabled, the rest as usual;
    /// - the load surface neither touches the network nor spawns: the mcp form only
    ///   yields server pins.
    pub fn load(&self, name: &str, registry: &PluginRegistry, plugins_root: &Path) -> LoadedPlugin {
        let plugin_dir = plugins_root.join(name);
        let verify = registry.verify_plugin(name, &plugin_dir);
        let mut loaded = LoadedPlugin {
            name: name.to_string(),
            verify,
            tools: Vec::new(),
            disabled: Vec::new(),
            mcp_servers: Vec::new(),
        };

        let Some(pin) = registry.get(name) else {
            loaded.disabled.push(DisabledTool {
                identity: format!("plugin.{name}"),
                reason: "not installed (no registry pin)".to_string(),
            });
            return loaded;
        };

        // Post-install tampering ⇒ fail-closed, disable everything (don't read the manifest
        // on disk — it is the very thing that was modified).
        if loaded.verify.tampered() {
            let reason = loaded
                .verify
                .user_message()
                .unwrap_or_else(|| "plugin has been modified".to_string());
            for tool in &pin.tools {
                loaded.disabled.push(DisabledTool {
                    identity: tool.identity(name),
                    reason: reason.clone(),
                });
            }
            return loaded;
        }

        // Trusted ⇒ the manifest content is what the pinned hash corresponds to
        // (re-verification already checked the hash).
        let manifest = match std::fs::read_to_string(plugin_dir.join("plugin.json"))
            .map_err(|e| e.to_string())
            .and_then(|raw| PluginManifest::from_json(&raw).map_err(|e| e.to_string()))
        {
            Ok(m) => m,
            Err(e) => {
                // Hash matches but the file is unreadable = impossible path; still fail-closed
                // to disabling everything.
                for tool in &pin.tools {
                    loaded.disabled.push(DisabledTool {
                        identity: tool.identity(name),
                        reason: format!("manifest unreadable: {e}"),
                    });
                }
                return loaded;
            }
        };

        let pins: BTreeMap<String, &ToolPin> = pin
            .tools
            .iter()
            .map(|t| (t.identity(name), t))
            .collect::<BTreeMap<_, _>>();
        // Per-tool load status (missing → Broken).
        let mut tool_status: BTreeMap<String, ToolStatus> = BTreeMap::new();
        for (identity, status) in &loaded.verify.tools {
            tool_status.insert(identity.clone(), status.clone());
        }

        for spec in &manifest.tools {
            let identity = match spec.identity(name) {
                Ok(key) => key,
                Err(e) => {
                    loaded.disabled.push(DisabledTool {
                        identity: format!("plugin.{name}.{}", spec.name()),
                        reason: e.to_string(),
                    });
                    continue;
                }
            };
            let Some(tool_pin) = pins.get(&identity) else {
                loaded.disabled.push(DisabledTool {
                    identity,
                    reason: "no registry pin for this tool".to_string(),
                });
                continue;
            };

            match (spec, tool_pin) {
                (PluginToolSpec::Script(spec), ToolPin::Script { run, .. }) => {
                    match build_script_tool(&identity, spec, run.clone()) {
                        Ok(tool) => loaded.tools.push(tool),
                        Err(reason) => loaded.disabled.push(DisabledTool { identity, reason }),
                    }
                }
                (
                    PluginToolSpec::Wasm(spec),
                    ToolPin::Wasm {
                        module,
                        capabilities,
                        ..
                    },
                ) => {
                    // Missing file → registry already judged Broken (disable this tool,
                    // don't skip the whole package).
                    if let Some(ToolStatus::Broken { detail }) = tool_status.get(&identity) {
                        loaded.disabled.push(DisabledTool {
                            identity,
                            reason: format!("broken: {detail}"),
                        });
                        continue;
                    }
                    match self.build_wasm_tool(&identity, spec, module, capabilities.clone()) {
                        Ok(tool) => loaded.tools.push(tool),
                        Err(reason) => loaded.disabled.push(DisabledTool { identity, reason }),
                    }
                }
                (PluginToolSpec::Mcp(spec), ToolPin::Mcp { .. }) => {
                    loaded.mcp_servers.push(PluginMcpServer {
                        label: spec.name.clone(),
                        command: tool_pin_mcp_command(tool_pin),
                        env: tool_pin_mcp_env(tool_pin),
                        url: tool_pin_mcp_url(tool_pin),
                        headers: tool_pin_mcp_headers(tool_pin),
                    });
                }
                (spec, _) => {
                    loaded.disabled.push(DisabledTool {
                        identity,
                        reason: format!(
                            "pin/manifest form mismatch: manifest={} pin={}",
                            spec.form(),
                            tool_pin.form()
                        ),
                    });
                }
            }
        }
        loaded
    }

    /// One wasm-form row → tool table entry (or the reason it is disabled).
    fn build_wasm_tool(
        &self,
        identity: &str,
        spec: &WasmToolSpec,
        module: &str,
        capabilities: Vec<super::manifest::ResolvedCapability>,
    ) -> Result<Arc<dyn Tool>, String> {
        #[cfg(feature = "wasm")]
        {
            let Some(runtime) = self.wasm.as_ref() else {
                return Err("wasm runtime not configured".to_string());
            };
            let component = runtime
                .compile(Path::new(module))
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(super::wasm::WasmTool::new(
                identity,
                &spec.description,
                &spec.parameters,
                Arc::clone(runtime),
                component,
                &spec.name,
                capabilities,
                std::time::Duration::from_secs(
                    spec.timeout_secs.unwrap_or(WASM_DEFAULT_TIMEOUT_SECS),
                ),
            )))
        }
        #[cfg(not(feature = "wasm"))]
        {
            let _ = (
                identity,
                spec,
                module,
                capabilities,
                WASM_DEFAULT_TIMEOUT_SECS,
            );
            Err("this build has no plugin wasm support (feature `wasm`)".to_string())
        }
    }
}

fn tool_pin_mcp_command(pin: &ToolPin) -> Option<Vec<String>> {
    match pin {
        ToolPin::Mcp { command, .. } => command.clone(),
        _ => None,
    }
}

fn tool_pin_mcp_env(pin: &ToolPin) -> BTreeMap<String, String> {
    match pin {
        ToolPin::Mcp { env, .. } => env.clone(),
        _ => BTreeMap::new(),
    }
}

fn tool_pin_mcp_url(pin: &ToolPin) -> Option<String> {
    match pin {
        ToolPin::Mcp { url, .. } => url.clone(),
        _ => None,
    }
}

fn tool_pin_mcp_headers(pin: &ToolPin) -> BTreeMap<String, String> {
    match pin {
        ToolPin::Mcp { headers, .. } => headers.clone(),
        _ => BTreeMap::new(),
    }
}

/// One script-form row → tool table entry (argv uses the pinned registry values, not the manifest's).
fn build_script_tool(
    identity: &str,
    spec: &ScriptToolSpec,
    run: Vec<String>,
) -> Result<Arc<dyn Tool>, String> {
    let entry = ScriptToolEntry {
        name: spec.name.clone(),
        description: spec.description.clone(),
        parameters: spec.parameters.clone(),
        run,
        timeout_ms: spec.timeout_secs.map(|s| s.saturating_mul(1_000)),
        env: BTreeMap::new(),
        cwd: None,
        exposure: spec.exposure,
    };
    entry.check_laws().map_err(|e| e.to_string())?;
    Ok(Arc::new(ScriptTool::with_key(identity, entry)))
}

/// MCP server tool name list → `plugin.{plugin}.{segment}` key assignment.
///
/// - normalization goes through [`normalize_tool_segment`] (lowercase, illegal chars → `-`,
///   truncate to 32);
/// - names that collide after normalization get `-2` / `-3`… (the suffix counts toward the
///   32 cap: truncate the base first, then append the suffix);
/// - names that normalize to an empty string go into `rejected` (fail-closed to "not given",
///   never given wrong).
///
/// Returned keys follow the input order; `(server original tool name, identity key)`.
pub fn assign_mcp_tool_keys(plugin: &str, raw_tool_names: &[String]) -> McpKeyAssignment {
    let mut taken: BTreeSet<String> = BTreeSet::new();
    let mut keys = Vec::new();
    let mut rejected = Vec::new();

    for raw in raw_tool_names {
        let base = normalize_tool_segment(raw);
        if base.is_empty() {
            rejected.push(raw.clone());
            continue;
        }
        let mut segment = base.clone();
        let mut n = 2;
        while taken.contains(&segment) {
            segment = suffixed_segment(&base, n);
            n += 1;
        }
        taken.insert(segment.clone());
        keys.push((raw.clone(), format!("plugin.{plugin}.{segment}")));
    }
    McpKeyAssignment { keys, rejected }
}

/// Collision suffix (`base-2`…): the suffix counts toward [`NAME_SEGMENT_MAX`];
/// if too long, truncate the base first.
fn suffixed_segment(base: &str, n: usize) -> String {
    let suffix = format!("-{n}");
    let keep = NAME_SEGMENT_MAX
        .saturating_sub(suffix.len())
        .min(base.len());
    format!("{}{suffix}", &base[..keep])
}

/// Result of [`assign_mcp_tool_keys`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpKeyAssignment {
    /// `(server original tool name, identity key)`, in input order.
    pub keys: Vec<(String, String)>,
    /// original names unusable as identity segments after normalization (rejected at load
    /// time, they don't take up a key).
    pub rejected: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::super::install::{PluginSource, commit_install, plan_install, remove_plugin};
    use super::super::registry::PluginRegistry;
    use super::*;
    use crate::approval_ledger::{ApprovalLedger, ApprovalTier};
    use agent_base::ToolRegistry;
    use serde_json::Value;

    /// A stub registered manually into the tool table (stand-in for a tool registered
    /// after an mcp form connects).
    struct StubTool(&'static str);

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "stub"
        }
        fn schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn call(
            &self,
            _args: &Value,
            _ctx: &agent_base::ToolContext,
        ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
            Ok(vec![])
        }
    }

    fn dispatcher_wasm() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dispatcher.wasm")
    }

    /// Write a script and mark it executable (unix) — the install law requires a directly
    /// executed `run[0]` to be executable.
    fn write_script(path: &std::path::Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(path).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(path, perm).unwrap();
        }
    }

    /// Install a basic script + mcp plugin (same shape as the install.rs tests).
    async fn install_basic(plugins_root: &Path, source_dir: &Path) -> PluginRegistry {
        std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
        std::fs::create_dir_all(source_dir.join("server")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1,
                "name": "my-tools",
                "version": "0.1.0",
                "tools": [
                    {
                        "name": "deploy", "form": "script",
                        "description": "deploy it",
                        "run": ["scripts/deploy.sh", "--env", "${env}"],
                        "parameters": {"env": {"type": "string"}}
                    },
                    {
                        "name": "wiki", "form": "mcp",
                        "server": {
                            "command": ["python", "server/main.py"],
                            "env": {"WIKI_TOKEN": "${ENV:WIKI_TOKEN}"}
                        }
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        write_script(
            &source_dir.join("scripts/deploy.sh"),
            "#!/bin/sh\necho deployed\n",
        );
        std::fs::write(source_dir.join("server/main.py"), "print('wiki')\n").unwrap();

        let mut registry = PluginRegistry::new();
        let plan = plan_install(&PluginSource::Path(source_dir.to_path_buf()), plugins_root)
            .await
            .expect("plan");
        commit_install(&plan, &mut registry).expect("commit");
        registry
    }

    #[tokio::test]
    async fn tampered_package_disables_all_tools() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let registry = install_basic(&plugins_root, &tmp.path().join("src")).await;

        // Post-install tampering: change the manifest content (hash no longer matches).
        let manifest_path = plugins_root.join("my-tools/plugin.json");
        let raw = std::fs::read_to_string(&manifest_path).unwrap();
        std::fs::write(&manifest_path, raw.replace("deploy it", "tampered!")).unwrap();

        let loader = PluginLoader::new();
        let loaded = loader.load("my-tools", &registry, &plugins_root);
        assert!(loaded.verify.tampered());
        assert!(loaded.tools.is_empty(), "tampering ⇒ everything disabled");
        assert_eq!(loaded.disabled.len(), 2);
        for tool in &loaded.disabled {
            assert!(
                tool.reason.contains("run `plugin add` again"),
                "reason must point to re-review: {}",
                tool.reason
            );
        }
        let mut keys: Vec<String> = loaded.disabled.iter().map(|d| d.identity.clone()).collect();
        keys.sort();
        assert_eq!(keys, ["plugin.my-tools.deploy", "plugin.my-tools.wiki"]);
    }

    #[tokio::test]
    async fn missing_wasm_disables_that_tool_only() {
        // .wasm deleted after install → that tool is disabled as Broken, the script in the
        // same package works as usual (the whole package is not skipped).
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src");
        std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
        std::fs::create_dir_all(source_dir.join("wasm")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "mix-pack", "version": "0.1.0",
                "tools": [
                    {"name": "cap", "form": "wasm", "description": "capture",
                     "module": "wasm/cap.wasm"},
                    {"name": "ship", "form": "script", "description": "ship it",
                     "run": ["scripts/ship.sh"]}
                ]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::copy(dispatcher_wasm(), source_dir.join("wasm/cap.wasm")).unwrap();
        write_script(
            &source_dir.join("scripts/ship.sh"),
            "#!/bin/sh\necho shipped\n",
        );

        let mut registry = PluginRegistry::new();
        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .expect("plan");
        commit_install(&plan, &mut registry).expect("commit");

        // Delete the .wasm after install (manifest untouched = not tampering, just missing pieces).
        std::fs::remove_file(plugins_root.join("mix-pack/wasm/cap.wasm")).unwrap();

        let loader = PluginLoader::new();
        let loaded = loader.load("mix-pack", &registry, &plugins_root);
        assert!(
            !loaded.verify.tampered(),
            "missing pieces are not tampering"
        );
        assert_eq!(loaded.tools.len(), 1, "script as usual");
        assert_eq!(loaded.tools[0].name(), "plugin.mix-pack.ship");
        assert_eq!(loaded.disabled.len(), 1);
        assert_eq!(loaded.disabled[0].identity, "plugin.mix-pack.cap");
        assert!(
            loaded.disabled[0].reason.contains("broken"),
            "reason must say the piece is missing: {}",
            loaded.disabled[0].reason
        );
    }

    #[cfg(feature = "wasm")]
    #[tokio::test]
    async fn corrupt_wasm_disables_that_tool_only() {
        // A bad .wasm installs fine (the hash is only a content pin), but fails to compile
        // at load time ⇒ only it gets disabled.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src");
        std::fs::create_dir_all(source_dir.join("wasm")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "wasm-pack", "version": "0.1.0",
                "tools": [
                    {"name": "echo", "form": "wasm", "description": "roundtrip",
                     "module": "wasm/good.wasm"},
                    {"name": "cursed", "form": "wasm", "description": "broken bits",
                     "module": "wasm/bad.wasm"}
                ]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::copy(dispatcher_wasm(), source_dir.join("wasm/good.wasm")).unwrap();
        std::fs::write(source_dir.join("wasm/bad.wasm"), b"not a wasm component").unwrap();

        let mut registry = PluginRegistry::new();
        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .expect("plan");
        commit_install(&plan, &mut registry).expect("commit");

        let runtime = Arc::new(super::super::wasm::WasmRuntime::new().expect("runtime"));
        let loader = PluginLoader::with_wasm(Arc::clone(&runtime));
        let loaded = loader.load("wasm-pack", &registry, &plugins_root);
        assert!(!loaded.verify.tampered());
        assert_eq!(
            loaded.tools.len(),
            1,
            "the good half of the package works as usual"
        );
        assert_eq!(loaded.tools[0].name(), "plugin.wasm-pack.echo");
        assert_eq!(loaded.disabled.len(), 1);
        assert_eq!(loaded.disabled[0].identity, "plugin.wasm-pack.cursed");
        assert!(
            loaded.disabled[0].reason.contains("compile"),
            "reason must say the compile failed: {}",
            loaded.disabled[0].reason
        );

        // The good tool really runs (the load surface didn't break the runtime).
        let out = loaded.tools[0]
            .call(
                &serde_json::json!({"hello": "wasm"}),
                &agent_base::ToolContext::for_test(),
            )
            .await
            .expect("call");
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn unload_clears_tool_table_ledger_registry() {
        // Three places cleaned: tool table / ledger / registry.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let registry = install_basic(&plugins_root, &tmp.path().join("src")).await;

        let loader = PluginLoader::new();
        let loaded = loader.load("my-tools", &registry, &plugins_root);

        // Into the tool table + ledger (the user's everyday state). mcp-form tools are
        // registered after the product layer connects; here a stub stands in for that row,
        // likewise in the table.
        let mut table = ToolRegistry::default();
        for tool in &loaded.tools {
            table.register_arc(Arc::clone(tool));
        }
        table.register_arc(Arc::new(StubTool("plugin.my-tools.wiki")));
        let mut ledger = ApprovalLedger::new();
        ledger.set_tier("plugin.my-tools.deploy", ApprovalTier::AlwaysAllow);
        ledger.set_tier("plugin.my-tools.wiki", ApprovalTier::Deny);
        ledger.set_tier("unrelated.keep", ApprovalTier::AlwaysAllow);

        // Unload: registry + directory + ledger prefix + tool table.
        let mut registry = registry;
        let report = remove_plugin(&plugins_root, &mut registry, "my-tools").expect("remove");
        let swept = report.sweep_ledger(&mut ledger);
        assert_eq!(swept, 2, "all plugin.my-tools.* swept");
        for key in &report.identity_keys {
            table.remove(key);
        }

        // Each of the three places is clean on its own.
        for key in &report.identity_keys {
            assert!(table.get(key).is_none(), "tool table has leftovers {key}");
        }
        assert_eq!(ledger.tier_of("plugin.my-tools.deploy"), ApprovalTier::Ask);
        assert_eq!(ledger.tier_of("plugin.my-tools.wiki"), ApprovalTier::Ask);
        assert_eq!(
            ledger.tier_of("unrelated.keep"),
            ApprovalTier::AlwaysAllow,
            "other plugins' ledgers must not be touched"
        );
        assert!(registry.get("my-tools").is_none());
        assert!(!plugins_root.join("my-tools").exists());
    }

    #[test]
    fn assign_mcp_tool_keys_normalizes_and_disambiguates() {
        let names: Vec<String> = ["Search", "search", "get file", "Other", ""]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = assign_mcp_tool_keys("my-tools", &names);
        assert_eq!(
            out.keys,
            vec![
                ("Search".to_string(), "plugin.my-tools.search".to_string()),
                ("search".to_string(), "plugin.my-tools.search-2".to_string()),
                (
                    "get file".to_string(),
                    "plugin.my-tools.get-file".to_string()
                ),
                ("Other".to_string(), "plugin.my-tools.other".to_string()),
            ]
        );
        assert_eq!(
            out.rejected,
            vec!["".to_string()],
            "empty name fail-closed reject"
        );
    }

    #[test]
    fn assign_mcp_tool_keys_suffix_respects_32_char_cap() {
        let base = "a".repeat(NAME_SEGMENT_MAX);
        let names: Vec<String> = vec![base.clone(), base.clone(), base.clone()];
        let out = assign_mcp_tool_keys("p", &names);
        assert_eq!(out.keys[0].1, format!("plugin.p.{base}"));
        assert_eq!(
            out.keys[1].1,
            format!("plugin.p.{}-2", "a".repeat(NAME_SEGMENT_MAX - 2))
        );
        assert_eq!(
            out.keys[2].1,
            format!("plugin.p.{}-3", "a".repeat(NAME_SEGMENT_MAX - 2))
        );
        // Suffix counts toward the 32 cap + all three keys distinct.
        let segments: Vec<&str> = out
            .keys
            .iter()
            .map(|(_, k)| k.strip_prefix("plugin.p.").unwrap())
            .collect();
        for segment in &segments {
            assert!(segment.len() <= NAME_SEGMENT_MAX, "{segment}");
        }
        let mut dedup = segments.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), 3);
    }
}
