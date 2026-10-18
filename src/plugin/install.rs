//! Plugin install lifecycle: `plugin add` / `plugin remove` (human review before
//! install → write to disk → pin → uninstall).
//!
//! The channels are capped at **path + git + archive**; this does not grow into a
//! package manager (marketplace / central catalog deferred).
//!
//! - **Human review before install**: [`plan_install`] only stages + produces a
//!   [`ReviewDigest`] (full manifest + capability breakdown); only **after human
//!   confirmation** does [`commit_install`] write to disk (product-level gate;
//!   plugins requested by an agent are never allowed through by default);
//! - **Pinned values**: manifest hash, each .wasm hash, `run`/module/server paths
//!   absolutized, resolved values for capabilities' paths/hosts — all pinned into
//!   [`PluginRegistry`] (the authority **outside** the plugin directory);
//! - **zip-slip**: reject if an archive member path escapes the extraction
//!   directory;
//! - **git dependency made explicit** (Windows): the git URL channel depends on a
//!   `git` executable on `PATH` (may embed gix later);
//! - **Uninstall cleans all three places**: tool table (identity_keys returned to
//!   the caller for deregistration) / ledger ([`RemovalReport::sweep_ledger`]) /
//!   registry entries.
//!
//! secret rule: env/headers only flow as `${ENV:VAR}` references; this module
//! never resolves them and never writes literals.

use std::path::{Path, PathBuf};

use super::manifest::{
    ManifestError, PluginManifest, PluginToolSpec, ReviewDigest, resolve_capabilities,
};
use super::registry::{PluginPin, PluginRegistry, RegistryError, ToolPin, hex_sha256};
use crate::script_tool::{
    ScriptLawError, check_script_file_on_disk, pin_from_plugin_root, resolve_under_root,
    script_file_index, script_file_is_program,
};

/// Archive format (archive channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    /// `.zip`。
    Zip,
    /// `.tar.gz` / `.tgz`。
    TarGz,
}

/// Install source (channels capped at path + git + archive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginSource {
    /// Local directory (copy install).
    Path(PathBuf),
    /// git URL (**depends on a git executable**; on Windows it must be on
    /// PATH — made explicit).
    Git {
        /// Repository URL.
        url: String,
        /// Optional rev/tag/branch.
        rev: Option<String>,
    },
    /// Local archive.
    ArchivePath {
        /// Archive path.
        path: PathBuf,
        /// Format.
        format: ArchiveFormat,
    },
    /// Remote archive URL.
    ArchiveUrl {
        /// Download URL.
        url: String,
        /// Format.
        format: ArchiveFormat,
    },
}

impl PluginSource {
    /// Parse the source from `plugin add` arguments (used by the product CLI).
    ///
    /// `*.git` / `git@` → git; `http(s)://` + `.zip|.tar.gz|.tgz` → archive URL;
    /// anything else is treated as a local path (directory or archive file).
    pub fn parse(spec: &str) -> Result<Self, InstallError> {
        let lower = spec.to_ascii_lowercase();
        if lower.ends_with(".git") || spec.starts_with("git@") {
            return Ok(Self::Git {
                url: spec.to_string(),
                rev: None,
            });
        }
        if spec.starts_with("https://") || spec.starts_with("http://") {
            if let Some(format) = archive_format_of(&lower) {
                return Ok(Self::ArchiveUrl {
                    url: spec.to_string(),
                    format,
                });
            }
            return Err(InstallError::BadRequest {
                reason: format!("unsupported URL `{spec}` (git URL or .zip/.tar.gz/.tgz archive)"),
            });
        }
        let path = PathBuf::from(spec);
        if let Some(format) = archive_format_of(&lower) {
            return Ok(Self::ArchivePath { path, format });
        }
        if path.is_dir() {
            return Ok(Self::Path(path));
        }
        Err(InstallError::BadRequest {
            reason: format!("`{spec}` is not a plugin directory or archive"),
        })
    }
}

fn archive_format_of(lower_spec: &str) -> Option<ArchiveFormat> {
    if lower_spec.ends_with(".zip") {
        Some(ArchiveFormat::Zip)
    } else if lower_spec.ends_with(".tar.gz") || lower_spec.ends_with(".tgz") {
        Some(ArchiveFormat::TarGz)
    } else {
        None
    }
}

/// Install errors. fail-closed: nothing is written to disk if any step is invalid.
#[derive(Debug)]
pub enum InstallError {
    /// IO failure.
    Io(std::io::Error),
    /// Manifest parse/validation failure.
    Manifest(ManifestError),
    /// Path-law violation (escape from the plugin root, etc.).
    PathLaw { source: ScriptLawError },
    /// git channel failure (including **git not on PATH** — the real-world
    /// friction made explicit for Windows).
    Git(String),
    /// Download failure.
    Download(String),
    /// Archive extraction failure.
    Archive(String),
    /// **zip-slip**: member path escapes the extraction directory.
    ZipSlip { entry: String },
    /// Registry write failure.
    Registry(RegistryError),
    /// Invalid arguments.
    BadRequest { reason: String },
    /// Plugin not installed (on remove).
    NotInstalled { name: String },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "install io: {e}"),
            Self::Manifest(e) => write!(f, "install manifest: {e}"),
            Self::PathLaw { source } => write!(f, "install path law: {source}"),
            Self::Git(e) => write!(
                f,
                "git channel failed: {e} (git URLs need the git executable on PATH; on Windows install Git for Windows)"
            ),
            Self::Download(e) => write!(f, "download failed: {e}"),
            Self::Archive(e) => write!(f, "archive extract failed: {e}"),
            Self::ZipSlip { entry } => {
                write!(
                    f,
                    "archive entry `{entry}` escapes the extract root (zip-slip)"
                )
            }
            Self::Registry(e) => write!(f, "registry: {e}"),
            Self::BadRequest { reason } => write!(f, "{reason}"),
            Self::NotInstalled { name } => write!(f, "plugin `{name}` is not installed"),
        }
    }
}

impl std::error::Error for InstallError {}

/// Staging layout (the output of `plan_install`, the input of `commit_install`).
#[derive(Debug)]
pub struct StagingLayout {
    /// Staging directory (contents already in the plugin's full layout).
    pub dir: PathBuf,
}

/// An install plan awaiting human review: staged content + review digest + pinned values.
///
/// **[`commit_install`] only after human confirmation** (product-level gate);
/// [`InstallPlan::discard`] clears the staging area if not installing.
#[derive(Debug)]
pub struct InstallPlan {
    /// Review digest (full manifest + capability breakdown + absolute paths).
    pub digest: ReviewDigest,
    /// Pinned values (paths already absolutized against the **final install dir**).
    pub pin: PluginPin,
    /// Final install directory (`<plugins_root>/<name>`).
    pub plugin_root: PathBuf,
    /// Staging layout.
    pub staging: StagingLayout,
    plugins_root: PathBuf,
}

impl InstallPlan {
    /// Abandon the install: clear staging, write nothing to disk.
    pub fn discard(self) {
        let _ = std::fs::remove_dir_all(&self.staging.dir);
    }
}

/// Install result.
#[derive(Debug, Clone)]
pub struct InstalledPlugin {
    /// The pins (already written into the registry).
    pub pin: PluginPin,
    /// Install directory.
    pub plugin_dir: PathBuf,
    /// Tool identity keys (for tool-table registration / ledger cleanup).
    pub identity_keys: Vec<String>,
}

/// Uninstall report: receipts for the three-place cleanup (tool table / ledger /
/// registry).
#[derive(Debug, Clone)]
pub struct RemovalReport {
    /// Plugin name.
    pub name: String,
    /// Tool identity keys (the caller deregisters these from the tool table).
    pub identity_keys: Vec<String>,
    /// Ledger prefix (`plugin.{name}.`).
    pub ledger_prefix: String,
    /// Whether the install directory has been deleted.
    pub dir_removed: bool,
    /// Whether the registry entry has been removed.
    pub registry_removed: bool,
}

impl RemovalReport {
    /// Sweep the ledger: wipe all tiers of `plugin.{name}.*` (no orphaned memory).
    pub fn sweep_ledger(&self, ledger: &mut crate::approval_ledger::ApprovalLedger) -> usize {
        ledger.remove_prefix(&self.ledger_prefix)
    }
}

/// `plugin add` step one: materialize the source into staging, validate the
/// manifest, compute pinned values, produce the review digest.
///
/// **Writes nothing to disk** (nothing into `<plugins_root>/<name>/`, nothing to
/// the registry) — only after a human has reviewed [`InstallPlan::digest`] and
/// confirmed does [`commit_install`] run.
pub async fn plan_install(
    source: &PluginSource,
    plugins_root: &Path,
) -> Result<InstallPlan, InstallError> {
    let staging_dir = staging_dir_for(plugins_root);
    std::fs::create_dir_all(&staging_dir).map_err(InstallError::Io)?;

    match materialize(source, &staging_dir).await {
        Ok(()) => {}
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging_dir);
            return Err(e);
        }
    }
    match build_plan(staging_dir, plugins_root) {
        Ok(plan) => Ok(plan),
        Err(e) => Err(e),
    }
}

/// `plugin add` step two (**after human confirmation**): staging → final
/// directory, pinned into the registry.
pub fn commit_install(
    plan: &InstallPlan,
    registry: &mut PluginRegistry,
) -> Result<InstalledPlugin, InstallError> {
    // Replacement semantics for re-adding: delete the old directory first
    // (same-name plugin slot).
    if plan.plugin_root.exists() {
        std::fs::remove_dir_all(&plan.plugin_root).map_err(InstallError::Io)?;
    }
    if let Some(parent) = plan.plugin_root.parent() {
        std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
    }
    std::fs::rename(&plan.staging.dir, &plan.plugin_root).map_err(InstallError::Io)?;

    registry.insert(plan.pin.clone());
    let registry_path = PluginRegistry::path_in(&plan.plugins_root);
    registry
        .save(&registry_path)
        .map_err(InstallError::Registry)?;

    let identity_keys = plan
        .pin
        .tools
        .iter()
        .flat_map(|t| match t {
            // The real tool keys of the mcp form expand at load time by server
            // tool name; here we give label keys, and the loader fills them in.
            ToolPin::Wasm { identity, .. } | ToolPin::Script { identity, .. } => {
                vec![identity.clone()]
            }
            ToolPin::Mcp { label, .. } => vec![format!("plugin.{}.{label}", plan.pin.name)],
        })
        .collect();

    Ok(InstalledPlugin {
        pin: plan.pin.clone(),
        plugin_dir: plan.plugin_root.clone(),
        identity_keys,
    })
}

/// `plugin remove`: clean all three places (tool-table keys returned, ledger
/// prefix sweepable, registry pins pulled), delete the install directory.
pub fn remove_plugin(
    plugins_root: &Path,
    registry: &mut PluginRegistry,
    name: &str,
) -> Result<RemovalReport, InstallError> {
    let pin = registry
        .remove(name)
        .ok_or_else(|| InstallError::NotInstalled {
            name: name.to_string(),
        })?;

    let registry_path = PluginRegistry::path_in(plugins_root);
    registry
        .save(&registry_path)
        .map_err(InstallError::Registry)?;

    let plugin_dir = plugins_root.join(name);
    let dir_removed = if plugin_dir.exists() {
        std::fs::remove_dir_all(&plugin_dir).map_err(InstallError::Io)?;
        true
    } else {
        false
    };

    let mut identity_keys = Vec::new();
    for tool in &pin.tools {
        match tool {
            ToolPin::Wasm { identity, .. } | ToolPin::Script { identity, .. } => {
                identity_keys.push(identity.clone());
            }
            ToolPin::Mcp { label, .. } => {
                identity_keys.push(format!("plugin.{name}.{label}"));
            }
        }
    }

    Ok(RemovalReport {
        name: name.to_string(),
        identity_keys,
        ledger_prefix: format!("plugin.{name}."),
        dir_removed,
        registry_removed: true,
    })
}

// ── internal: materialize / plan ──────────────────────────────────────────

fn staging_dir_for(plugins_root: &Path) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    plugins_root.join(format!(".staging-{stamp}"))
}

async fn materialize(source: &PluginSource, staging: &Path) -> Result<(), InstallError> {
    match source {
        PluginSource::Path(path) => {
            copy_dir(path, staging)?;
            Ok(())
        }
        PluginSource::Git { url, rev } => git_clone(url, rev.as_deref(), staging).await,
        PluginSource::ArchivePath { path, format } => {
            let raw = std::fs::read(path).map_err(InstallError::Io)?;
            extract_archive(&raw, *format, staging)
        }
        PluginSource::ArchiveUrl { url, format } => {
            let raw = download(url).await?;
            extract_archive(&raw, *format, staging)
        }
    }
}

async fn download(url: &str) -> Result<Vec<u8>, InstallError> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| InstallError::Download(format!("{url}: {e}")))?;
    if !response.status().is_success() {
        return Err(InstallError::Download(format!(
            "{url}: HTTP {}",
            response.status()
        )));
    }
    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| InstallError::Download(format!("{url}: {e}")))
}

/// git channel: `git clone` (+ optional rev). Depends on a git executable
/// (explicit for Windows).
async fn git_clone(url: &str, rev: Option<&str>, staging: &Path) -> Result<(), InstallError> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(url)
        .arg(staging)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    if let Some(rev) = rev {
        cmd.arg("--branch").arg(rev);
    }
    let output = cmd
        .output()
        .await
        .map_err(|e| InstallError::Git(format!("spawn git: {e}")))?;
    if !output.status.success() {
        return Err(InstallError::Git(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(to).map_err(InstallError::Io)?;
    for entry in std::fs::read_dir(from).map_err(InstallError::Io)? {
        let entry = entry.map_err(InstallError::Io)?;
        let target = to.join(entry.file_name());
        let kind = entry.file_type().map_err(InstallError::Io)?;
        if kind.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

/// Extract into staging (zip-slip check: reject if a member path escapes the
/// extraction directory).
fn extract_archive(raw: &[u8], format: ArchiveFormat, staging: &Path) -> Result<(), InstallError> {
    match format {
        ArchiveFormat::Zip => extract_zip(raw, staging),
        ArchiveFormat::TarGz => extract_tar_gz(raw, staging),
    }
}

fn extract_zip(raw: &[u8], staging: &Path) -> Result<(), InstallError> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(raw))
        .map_err(|e| InstallError::Archive(e.to_string()))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| InstallError::Archive(e.to_string()))?;
        let Some(rel) = file.enclosed_name() else {
            return Err(InstallError::ZipSlip {
                entry: file.name().to_string(),
            });
        };
        let target = staging.join(&rel);
        if file.is_dir() {
            std::fs::create_dir_all(&target).map_err(InstallError::Io)?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
            }
            let mut out = std::fs::File::create(&target).map_err(InstallError::Io)?;
            std::io::copy(&mut file, &mut out).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

fn extract_tar_gz(raw: &[u8], staging: &Path) -> Result<(), InstallError> {
    let decoder = flate2::read::GzDecoder::new(raw);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive
        .entries()
        .map_err(|e| InstallError::Archive(e.to_string()))?
    {
        let mut entry = entry.map_err(|e| InstallError::Archive(e.to_string()))?;
        let path = entry
            .path()
            .map_err(|e| InstallError::Archive(e.to_string()))?;
        // zip-slip: after lexical normalization it must still be under the
        // extraction directory.
        let Some(rel) = normalize_relative(&path) else {
            return Err(InstallError::ZipSlip {
                entry: path.display().to_string(),
            });
        };
        let target = staging.join(&rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&target).map_err(InstallError::Io)?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
            }
            let mut out = std::fs::File::create(&target).map_err(InstallError::Io)?;
            std::io::copy(&mut entry, &mut out).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

/// Lexical normalization of an archive member path: absolute path / `..`
/// escape → `None` (zip-slip).
fn normalize_relative(path: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if path.is_absolute() {
        return None;
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None, // reject ParentDir / RootDir / Prefix outright
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Staging → plan: parse the manifest, compute pinned values against the
/// **final directory**, produce the review digest.
fn build_plan(staging_dir: PathBuf, plugins_root: &Path) -> Result<InstallPlan, InstallError> {
    let manifest_path = staging_dir.join("plugin.json");
    let manifest_raw = std::fs::read(&manifest_path).map_err(InstallError::Io)?;
    let manifest_text = String::from_utf8_lossy(&manifest_raw).into_owned();
    let manifest = PluginManifest::from_json(&manifest_text).map_err(InstallError::Manifest)?;

    let plugin_root = plugins_root.join(&manifest.name);
    let digest = manifest
        .review(&plugin_root)
        .map_err(InstallError::Manifest)?;

    let pin = build_pin(&manifest, &manifest_raw, &plugin_root, &staging_dir)?;

    Ok(InstallPlan {
        digest,
        pin,
        plugin_root,
        staging: StagingLayout { dir: staging_dir },
        plugins_root: plugins_root.to_path_buf(),
    })
}

fn build_pin(
    manifest: &PluginManifest,
    manifest_raw: &[u8],
    plugin_root: &Path,
    staging_dir: &Path,
) -> Result<PluginPin, InstallError> {
    let mut tools = Vec::with_capacity(manifest.tools.len());
    for tool in &manifest.tools {
        let identity = tool.identity(&manifest.name).map_err(|source| {
            InstallError::Manifest(ManifestError::BadName {
                name: tool.name().to_string(),
                source,
            })
        })?;
        match tool {
            PluginToolSpec::Wasm(t) => {
                let module = resolve_under_root(plugin_root, &t.module)
                    .map_err(|source| InstallError::PathLaw { source })?;
                // Hashes are computed over the staged content (bytes unchanged
                // after the rename).
                let staged_module = staging_dir.join(&t.module);
                let bytes = std::fs::read(&staged_module).map_err(InstallError::Io)?;
                let capabilities = resolve_capabilities(&t.capabilities, plugin_root)
                    .map_err(|source| InstallError::PathLaw { source })?;
                tools.push(ToolPin::Wasm {
                    identity,
                    module,
                    wasm_sha256: hex_sha256(&bytes),
                    capabilities,
                });
            }
            PluginToolSpec::Script(t) => {
                let run = pin_from_plugin_root(&t.run, plugin_root)
                    .map_err(|source| InstallError::PathLaw { source })?;
                // At install time the script files must be present in the
                // package — the same fail-closed law as missing wasm files
                // (acceptance fixture: a missing script file must block a
                // half-package install). We check the staged content (bytes
                // unchanged after the rename); if something is missing, report
                // which file is missing.
                let index =
                    script_file_index(&t.run).map_err(|source| InstallError::PathLaw { source })?;
                let require_executable = script_file_is_program(&t.run)
                    .map_err(|source| InstallError::PathLaw { source })?;
                check_script_file_on_disk(&staging_dir.join(&t.run[index]), require_executable)
                    .map_err(|source| InstallError::PathLaw { source })?;
                tools.push(ToolPin::Script { identity, run });
            }
            PluginToolSpec::Mcp(t) => {
                let command = match &t.server.command {
                    Some(cmd) => Some(
                        pin_server_argv(cmd, plugin_root)
                            .map_err(|source| InstallError::PathLaw { source })?,
                    ),
                    None => None,
                };
                tools.push(ToolPin::Mcp {
                    label: t.name.clone(),
                    command,
                    env: t.server.env.clone(),
                    url: t.server.url.clone(),
                    headers: t.server.headers.clone(),
                });
            }
        }
    }

    Ok(PluginPin {
        name: manifest.name.clone(),
        version: manifest.version.clone(),
        source: "installed".into(),
        manifest_sha256: hex_sha256(manifest_raw),
        tools,
    })
}

/// Argv pinning law for an in-package MCP server (same anti-substitution intent
/// as `run[0]`, trimmed to the server context):
///
/// - **Path-shaped element** (contains a separator / starts with `.`) =
///   in-package relative path → resolve to an absolute path under the plugin
///   root and pin it (lexical normalization + reject escape from the plugin
///   root);
/// - **Bare name** (`python` / `npx` …) = interpreter/launcher resolved via
///   PATH lookup, kept as-is — the server's author experience comes first
///   (design example `["python", "server/main.py"]`);
/// - **Absolute path** (`/usr/bin/python3`) = explicit interpreter, kept as-is.
fn pin_server_argv(argv: &[String], plugin_root: &Path) -> Result<Vec<String>, ScriptLawError> {
    argv.iter()
        .map(|arg| {
            let pathish = arg.contains('/') || arg.contains('\\') || arg.starts_with('.');
            if !pathish || Path::new(arg).is_absolute() {
                return Ok(arg.clone());
            }
            resolve_under_root(plugin_root, arg)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::manifest::ResolvedCapability;
    use super::super::registry::PackageStatus;
    use super::*;

    /// Platform-independent absolute path (test criterion under the same law
    /// as `Path::is_absolute`).
    fn abs(rel: &str) -> PathBuf {
        std::env::current_dir().unwrap().join(rel)
    }

    /// Write a script and set the executable bit (unix) — the install law
    /// requires directly-executed `run[0]` to be executable.
    fn write_script(path: &Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(path).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(path, perm).unwrap();
        }
    }

    fn write_basic_plugin(dir: &Path) {
        std::fs::create_dir_all(dir.join("scripts")).unwrap();
        std::fs::create_dir_all(dir.join("wasm")).unwrap();
        std::fs::create_dir_all(dir.join("server")).unwrap();
        std::fs::write(
            dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1,
                "name": "my-tools",
                "version": "0.1.0",
                "tools": [
                    {
                        "name": "deploy", "form": "script",
                        "description": "deploy it",
                        "run": ["scripts/deploy.sh", "--env", "${env}"],
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
                ]
            })
            .to_string(),
        )
        .unwrap();
        write_script(&dir.join("scripts/deploy.sh"), "#!/bin/sh\necho deployed\n");
        std::fs::write(dir.join("server/main.py"), "print('wiki')\n").unwrap();
    }

    #[tokio::test]
    async fn plan_then_commit_installs_and_pins() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        write_basic_plugin(&source_dir);

        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .expect("plan");

        // Review digest: full argv + absolute paths (mechanism side of UX
        // criterion 3).
        let deploy = plan
            .digest
            .tools
            .iter()
            .find(|t| t.identity == "plugin.my-tools.deploy")
            .unwrap();
        let run_abs = plugins_root
            .join("my-tools")
            .join("scripts/deploy.sh")
            .to_string_lossy()
            .into_owned();
        assert_eq!(deploy.summary, format!("{run_abs} --env ${{env}}"));
        assert_eq!(deploy.timeout_secs, Some(120));

        // Before human confirmation: neither the final directory nor the
        // registry exists.
        assert!(!plugins_root.join("my-tools").exists());
        assert!(!PluginRegistry::path_in(&plugins_root).exists());

        // Standard install trio: env only flows as ${ENV:VAR} references
        // (never resolved).
        let mut registry = PluginRegistry::new();
        let installed = commit_install(&plan, &mut registry).expect("commit");
        assert!(plugins_root.join("my-tools/plugin.json").is_file());
        assert!(registry.get("my-tools").is_some());

        let pins = &registry.get("my-tools").unwrap().tools;
        let script_pin = pins
            .iter()
            .find_map(|t| match t {
                ToolPin::Script { identity, run } if identity == "plugin.my-tools.deploy" => {
                    Some(run.clone())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(script_pin[0], run_abs, "run[0] pinned to an absolute path");
        assert!(matches!(
            &pins[1],
            ToolPin::Mcp { env, command: Some(cmd), .. }
                if env.get("WIKI_TOKEN").map(String::as_str) == Some("${ENV:WIKI_TOKEN}")
                    && cmd[0] == "python"
                    && cmd[1].ends_with("server/main.py")
                    && Path::new(&cmd[1]).is_absolute()
        ));
        assert_eq!(
            installed.identity_keys,
            vec!["plugin.my-tools.deploy", "plugin.my-tools.wiki"]
        );

        // The manifest hash in the registry matches the one on disk (load-time
        // re-verification passes).
        let report = registry.verify(&plugins_root.join("my-tools"));
        assert_eq!(report.package, PackageStatus::Trusted);
    }

    #[tokio::test]
    async fn discard_leaves_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        write_basic_plugin(&source_dir);

        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .expect("plan");
        plan.discard();
        assert!(!plugins_root.join("my-tools").exists());
        let leftovers = std::fs::read_dir(&plugins_root)
            .map(|rd| rd.count())
            .unwrap_or(0);
        assert_eq!(leftovers, 0, "staging must be cleaned up too");
    }

    #[tokio::test]
    async fn replace_reinstall_swaps_pin() {
        // "Re-run plugin add to re-review": installing the same name again =
        // replace + re-pin.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        write_basic_plugin(&source_dir);

        let mut registry = PluginRegistry::new();
        let plan = plan_install(&PluginSource::Path(source_dir.clone()), &plugins_root)
            .await
            .unwrap();
        commit_install(&plan, &mut registry).unwrap();

        // The author changed the script content and installed again.
        write_script(
            &source_dir.join("scripts/deploy.sh"),
            "#!/bin/sh\necho v2\n",
        );
        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap();
        commit_install(&plan, &mut registry).unwrap();

        let report = registry.verify(&plugins_root.join("my-tools"));
        assert_eq!(report.package, PackageStatus::Trusted);
    }

    #[tokio::test]
    async fn remove_clears_dir_registry_and_ledger() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        write_basic_plugin(&source_dir);

        let mut registry = PluginRegistry::new();
        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap();
        let installed = commit_install(&plan, &mut registry).unwrap();

        // Record a ledger tier first (simulate prior use).
        let mut ledger = crate::approval_ledger::ApprovalLedger::new();
        ledger.set_tier(
            "plugin.my-tools.deploy",
            crate::approval_ledger::ApprovalTier::AlwaysAllow,
        );
        ledger.set_tier(
            "plugin.other.search",
            crate::approval_ledger::ApprovalTier::AlwaysAllow,
        );

        let report = remove_plugin(&plugins_root, &mut registry, "my-tools").expect("remove");
        assert!(report.dir_removed);
        assert!(report.registry_removed);
        assert!(registry.get("my-tools").is_none());
        assert!(!installed.plugin_dir.exists());

        // One of the three places: plugin.my-tools.* wiped, others untouched.
        let swept = report.sweep_ledger(&mut ledger);
        assert_eq!(swept, 1);
        assert_eq!(
            ledger.tier_of("plugin.my-tools.deploy"),
            crate::approval_ledger::ApprovalTier::Ask
        );
        assert_eq!(
            ledger.tier_of("plugin.other.search"),
            crate::approval_ledger::ApprovalTier::AlwaysAllow
        );
        assert_eq!(
            report.identity_keys,
            vec!["plugin.my-tools.deploy", "plugin.my-tools.wiki"]
        );
    }

    #[tokio::test]
    async fn remove_unknown_plugin_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let mut registry = PluginRegistry::new();
        let err = remove_plugin(&plugins_root, &mut registry, "ghost").unwrap_err();
        assert!(matches!(err, InstallError::NotInstalled { name } if name == "ghost"));
    }

    // ── zip-slip / archive channel ────────────────────────────────────

    #[test]
    fn normalize_relative_rejects_traversal() {
        assert!(normalize_relative(Path::new("ok/inner.txt")).is_some());
        assert!(normalize_relative(Path::new("./ok")).is_some());
        assert!(normalize_relative(Path::new("../escape")).is_none());
        assert!(normalize_relative(Path::new("a/../../b")).is_none());
        assert!(normalize_relative(Path::new("/abs")).is_none());
        assert!(normalize_relative(Path::new("")).is_none());
    }

    #[test]
    fn tar_gz_zip_slip_is_rejected() {
        // Build a tar.gz with a `../evil` entry (bypassing tar::Builder's path
        // law by writing the GNU header directly — exactly how real malicious
        // packages are made).
        let mut raw = Vec::new();
        {
            let encoder = flate2::write::GzEncoder::new(&mut raw, flate2::Compression::default());
            let mut builder = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            header.set_size(5);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            header.as_gnu_mut().unwrap().name[..7].copy_from_slice(b"../evil");
            header.set_cksum();
            builder.append(&header, &b"pwned"[..]).expect("append");
            builder.finish().expect("finish");
        }

        let tmp = tempfile::tempdir().unwrap();
        let err = extract_archive(&raw, ArchiveFormat::TarGz, tmp.path()).unwrap_err();
        assert!(
            matches!(err, InstallError::ZipSlip { ref entry } if entry.contains("evil")),
            "{err:?}"
        );
        assert!(!tmp.path().join("../evil").exists());
    }

    #[test]
    fn tar_gz_extracts_clean_archive() {
        let mut raw = Vec::new();
        {
            let encoder = flate2::write::GzEncoder::new(&mut raw, flate2::Compression::default());
            let mut builder = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            header.set_size(5);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "plugin.json", &b"hello"[..])
                .unwrap();
            builder.finish().unwrap();
        }
        let tmp = tempfile::tempdir().unwrap();
        extract_archive(&raw, ArchiveFormat::TarGz, tmp.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("plugin.json")).unwrap(),
            "hello"
        );
    }

    // ── source parsing ────────────────────────────────────────────────────

    #[test]
    fn source_parse_channels() {
        assert_eq!(
            PluginSource::parse("https://x.example/my-tools.git").unwrap(),
            PluginSource::Git {
                url: "https://x.example/my-tools.git".into(),
                rev: None
            }
        );
        assert_eq!(
            PluginSource::parse("git@github.com:me/my-tools.git").unwrap(),
            PluginSource::Git {
                url: "git@github.com:me/my-tools.git".into(),
                rev: None
            }
        );
        assert_eq!(
            PluginSource::parse("https://x.example/my-tools.tar.gz").unwrap(),
            PluginSource::ArchiveUrl {
                url: "https://x.example/my-tools.tar.gz".into(),
                format: ArchiveFormat::TarGz
            }
        );
        assert!(matches!(
            PluginSource::parse("https://x.example/page.html"),
            Err(InstallError::BadRequest { .. })
        ));
        assert!(matches!(
            PluginSource::parse("/definitely/not/here"),
            Err(InstallError::BadRequest { .. })
        ));
        assert_eq!(
            PluginSource::parse("packs/my-tools.zip").unwrap(),
            PluginSource::ArchivePath {
                path: PathBuf::from("packs/my-tools.zip"),
                format: ArchiveFormat::Zip
            }
        );
    }

    #[test]
    fn pin_server_argv_pins_paths_keeps_interpreters() {
        let root = abs("plugin-root-test");
        let pinned = pin_server_argv(
            &[
                "python".into(),
                "server/main.py".into(),
                abs("/usr/bin/python3").to_string_lossy().into_owned(),
            ],
            &root,
        )
        .unwrap();
        assert_eq!(pinned[0], "python", "bare-name interpreter kept as-is");
        assert_eq!(
            pinned[1],
            root.join("server/main.py").to_string_lossy().into_owned(),
            "in-package relative path pinned absolute"
        );
        assert!(
            Path::new(&pinned[2]).is_absolute(),
            "explicit absolute interpreter kept as-is"
        );

        // Path-shaped arguments that escape the plugin root are rejected.
        let err = pin_server_argv(&["python".into(), "../evil.py".into()], &root).unwrap_err();
        assert!(
            matches!(err, ScriptLawError::EscapedPluginRoot { .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn wasm_missing_at_install_is_reported() {
        // plugin.json lists a wasm tool but the .wasm is missing → cannot
        // install (fail-closed); a half package is never pinned into the
        // registry.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{"name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d"}]
            })
            .to_string(),
        )
        .unwrap();

        let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap_err();
        assert!(matches!(err, InstallError::Io(_)), "{err:?}");
    }

    #[tokio::test]
    async fn script_missing_at_install_is_reported() {
        // Same law as missing wasm files: the manifest lists a script tool but
        // the file is absent (acceptance fixture: a missing script file must
        // block a half-package install) → cannot install, and the error says
        // which file is missing.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{
                    "name": "deploy", "form": "script",
                    "description": "deploy it",
                    "run": ["scripts/deploy.sh", "--env", "${env}"],
                    "parameters": {"env": {"type": "string"}}
                }]
            })
            .to_string(),
        )
        .unwrap();

        let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("scripts/deploy.sh")
                && err.to_string().contains("does not exist"),
            "missing files must be named: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn script_not_executable_at_install_is_reported() {
        // `run[0]` is the script body itself, executed directly — no
        // executable bit = it would not run even if installed, so reject at
        // install time (chmod +x and come back).
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{
                    "name": "deploy", "form": "script",
                    "description": "deploy it",
                    "run": ["scripts/deploy.sh"]
                }]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(source_dir.join("scripts/deploy.sh"), "#!/bin/sh\necho hi\n").unwrap();
        // 0644 — no executable bit.

        let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("not executable"),
            "a missing executable bit must be stated clearly: {err}"
        );
    }

    #[tokio::test]
    async fn script_interpreter_form_checks_the_script_arg_not_the_interpreter() {
        // Interpreter form: the executable bit is checked on the interpreter
        // (PATH resolution, not the installer's business), but the script
        // argument must be in the package.
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{
                    "name": "deploy", "form": "script",
                    "description": "deploy it",
                    "run": ["cmd", "/c", "scripts/deploy.bat"]
                }]
            })
            .to_string(),
        )
        .unwrap();

        let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("deploy.bat") && err.to_string().contains("does not exist"),
            "the interpreter form checks the script argument: {err}"
        );
    }

    #[tokio::test]
    async fn capability_pinned_with_resolved_values() {
        let tmp = tempfile::tempdir().unwrap();
        let plugins_root = tmp.path().join("plugins");
        let source_dir = tmp.path().join("src-plugin");
        std::fs::create_dir_all(source_dir.join("wasm")).unwrap();
        std::fs::write(source_dir.join("wasm/s.wasm"), b"wasm-bytes").unwrap();
        std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{
                    "name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d",
                    "capabilities": [
                        {"type": "fs:read", "paths": ["./data"]},
                        {"type": "net:http", "hosts": ["API.Example.com"]}
                    ]
                }]
            })
            .to_string(),
        )
        .unwrap();

        let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
            .await
            .expect("plan");
        match &plan.pin.tools[0] {
            ToolPin::Wasm {
                capabilities,
                wasm_sha256,
                module,
                ..
            } => {
                let data_abs = plugins_root
                    .join("my-tools")
                    .join("data")
                    .to_string_lossy()
                    .into_owned();
                assert_eq!(
                    capabilities,
                    &vec![
                        ResolvedCapability::FsRead {
                            paths: vec![data_abs]
                        },
                        ResolvedCapability::NetHttp {
                            hosts: vec!["api.example.com".into()]
                        }
                    ]
                );
                assert_eq!(*wasm_sha256, hex_sha256(b"wasm-bytes"));
                assert!(Path::new(module).is_absolute());
            }
            other => panic!("expected wasm pin, got {other:?}"),
        }
    }
}
