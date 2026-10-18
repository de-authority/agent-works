//! Plugin package machinery (P2) — manifest / registry integrity / install lifecycle / loading.
//!
//! From the user's point of view, in one sentence: "install a plugin". After the
//! install, a few more tools show up in the tool table; the user doesn't need to know
//! which one is wasm and which one is python. For the mechanism's main text, see
//! `phimint/notes/tool-socket-design.md`, sections "Shape of a plugin package /
//! Plugin install":
//!
//! - **Three forms, one key scheme**: tools in the wasm / script / mcp forms share
//!   the unified identity `plugin.{plugin}.{tool}` ([crate::tool_provider]);
//! - **Human review before install**: [`manifest::ReviewDigest`] lays out the full
//!   manifest (tools / execution summary / capabilities with concrete paths/hosts),
//!   and only after human confirmation does [`install::commit_install`] write to
//!   disk; a plugin requested for install on an agent's behalf is never allowed
//!   through by default (guaranteed at the product layer);
//! - **Install integrity is pinned outside the plugin directory**:
//!   [`registry::PluginRegistry`] (`<plugins_root>/registry.json`) stores the manifest
//!   hash + the hash of every .wasm + pinned resolved capabilities/absolute paths.
//!   **No authoritative copy** is kept inside the plugin directory — anchor and
//!   anchored object in the same cave = self-deception. Re-verified on every load;
//!   tampering = all of that plugin's tools are disabled + "run `plugin add` again to
//!   re-review";
//! - **Capability tightening reads the pinned values in the registry**, never the
//!   plugin's own manifest;
//! - **Corruption does not skip the whole package**: if plugin.json lists a tool but
//!   the .wasm is missing/corrupt → that tool is disabled + an error is reported,
//!   while the other tools work as usual;
//! - **Uninstall cleans three places**: the tool table (the provider no longer lists
//!   it) / the ledger (`plugin.{name}.*`) / the registry entry.
//!
//! Transport notes (explicitly documented, phased discipline): git URLs depend on
//! the git executable (must be on PATH on Windows; gix could be embedded later);
//! archive extraction performs a zip-slip path check (a member path that escapes the
//! target directory is rejected).

pub mod install;
pub mod loader;
pub mod manifest;
pub mod registry;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use install::{
    ArchiveFormat, InstallError, InstallPlan, InstalledPlugin, PluginSource, RemovalReport,
    StagingLayout, commit_install, plan_install, remove_plugin,
};
pub use loader::{
    DisabledTool, LoadedPlugin, McpKeyAssignment, PluginLoader, PluginMcpServer,
    assign_mcp_tool_keys,
};
pub use manifest::{
    CapabilitySpec, FsReadSpec, ManifestError, McpServerSpec, McpToolSpec, NetHttpSpec,
    PluginManifest, PluginToolSpec, ResolvedCapability, ReviewDigest, ReviewTool, ScriptToolSpec,
    WasmToolSpec, resolve_capabilities,
};
pub use registry::{
    PackageStatus, PluginPin, PluginRegistry, RegistryError, ToolPin, ToolStatus, VerifyIssue,
    VerifyReport,
};
