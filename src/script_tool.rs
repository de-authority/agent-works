//! Script-tier adapter (P1) — the runtime law for global script tools
//! (`script.{name}`) and the plugin `script` form.
//!
//! Spec text: `phimint/notes/tool-socket-design.md`, sections
//! "global script tool entry / script runtime semantics":
//!
//! - **Absolute path pinning (anti-shadowing)**: `run[0]` must be an absolute
//!   path. The only exception is the interpreter whitelist
//!   (`powershell` / `pwsh` / `cmd`), in which case the **script file argument**
//!   must be an absolute path (powershell: after `-File`/`-f`; cmd: after `/c`).
//!   A relative path = running that other project's same-named script in that
//!   project; rejected at definition/install time.
//! - **Scripts inside plugins obey the same law**: package-relative paths are
//!   resolved to absolute paths at install time by [`pin_from_plugin_root`]
//!   and pinned (lexical normalization + escape from the plugin root → reject).
//! - **argv spawn, no shell**: `run` substitutes `${param}` per element; a
//!   missing or non-scalar parameter → tool error for that call (fail-closed);
//!   parameter values are not re-expanded.
//! - cwd = current project root (pinned at product-layer registration in
//!   [`ScriptToolEntry::cwd`]; the framework does not sniff paths);
//!   env = inherited + entry overrides.
//! - timeout defaults to 60s; on timeout the **whole process tree** is killed
//!   (POSIX process group + `killpg`; Windows `taskkill /T /F` +
//!   `CREATE_NO_WINDOW`).
//! - Output exceeding `max_output_chars` reuses the kernel-tools truncation
//!   convention (first 1/3 + last 2/3 + `...[output truncated]`; the convention
//!   is copied verbatim, no cross-crate code import).
//! - exit ≠ 0 → tool error (with a stderr summary). If the script file is gone
//!   at execution time → a "script is gone" error, **no silent file swap**.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use agent_base::tool::Tool;
use agent_base::{AgentError, AgentResult, Content, ToolContext, ToolExposure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::tool_provider::{ToolDescriptor, ToolIdentity, ToolProvider};

/// Default timeout (milliseconds).
pub const DEFAULT_TIMEOUT_MS: u64 = 60_000;

/// Interpreter whitelist: the **only** exception where `run[0]` need not be an absolute path.
pub const INTERPRETER_WHITELIST: [&str; 3] = ["powershell", "pwsh", "cmd"];

/// Output truncation marker (same convention as phi-kernel-tools `local_shell::truncate_output`).
const TRUNCATION_MARKER: &str = "...[output truncated]";

/// Path-law violation at definition/install time. Fail-closed: reject what is illegal, no compromises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptLawError {
    /// Tool name is invalid (character set / reserved word / too long).
    BadName { name: String },
    /// `run` is empty.
    EmptyRun,
    /// `run[0]` is not an absolute path and not in the interpreter whitelist — rejected to prevent shadowing.
    RelativeProgram { program: String },
    /// The interpreter lacks a script file argument (`-File`/`-f`/`/c` not found), so the script path cannot be pinned.
    InterpreterMissingScriptArg { interpreter: String },
    /// The interpreter's script file argument is not an absolute path.
    RelativeScriptArg { interpreter: String, arg: String },
    /// A path inside a plugin escapes the plugin root after resolution (`../` escape / absolute path outside).
    EscapedPluginRoot { path: String },
    /// The script file does not exist at install/mount time (or is not a regular file).
    ScriptFileMissing { path: String },
    /// The script body (the non-interpreter form's `run[0]`) has no executable bit.
    ScriptFileNotExecutable { path: String },
}

impl std::fmt::Display for ScriptLawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadName { name } => write!(f, "script tool name `{name}` is invalid"),
            Self::EmptyRun => write!(f, "`run` must not be empty"),
            Self::RelativeProgram { program } => write!(
                f,
                "`run[0]` must be an absolute path (got `{program}`); \
                 the only exception is the interpreter whitelist {INTERPRETER_WHITELIST:?} \
                 with an absolute script file argument"
            ),
            Self::InterpreterMissingScriptArg { interpreter } => write!(
                f,
                "interpreter `{interpreter}` needs a script file argument \
                 (`-File`/`-f` for powershell/pwsh, `/c` for cmd) so its absolute path can be pinned"
            ),
            Self::RelativeScriptArg { interpreter, arg } => write!(
                f,
                "script file argument `{arg}` of interpreter `{interpreter}` must be an absolute path"
            ),
            Self::EscapedPluginRoot { path } => {
                write!(f, "script path `{path}` escapes the plugin root")
            }
            Self::ScriptFileMissing { path } => write!(
                f,
                "script file `{path}` does not exist (must be a regular file)"
            ),
            Self::ScriptFileNotExecutable { path } => write!(
                f,
                "script file `{path}` is not executable (chmod +x) — it runs directly as `run[0]`"
            ),
        }
    }
}

impl std::error::Error for ScriptLawError {}

/// Parameter-table entry (the author-facing side of plugin manifest / tools.json): scalars only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    String,
    Number,
    Boolean,
}

/// A single parameter: `{ "type": "string", "description": "…" }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamSpec {
    #[serde(rename = "type")]
    pub ty: ParamType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Tool-table budget tier (`ToolExposure` has no serde; the author-facing side uses this string form).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureSpec {
    #[default]
    Direct,
    Deferred,
    Hidden,
}

impl From<ExposureSpec> for ToolExposure {
    fn from(value: ExposureSpec) -> Self {
        match value {
            ExposureSpec::Direct => ToolExposure::Direct,
            ExposureSpec::Deferred => ToolExposure::Deferred,
            ExposureSpec::Hidden => ToolExposure::Hidden,
        }
    }
}

/// A script tool entry (one row of tools.json / one row of the plugin manifest `script` form).
///
/// JSON form (the design's "global script tool entry"):
/// ```json
/// {
///   "name": "deploy",
///   "description": "部署当前项目到 staging",
///   "parameters": { "env": { "type": "string", "description": "目标环境" } },
///   "run": ["/Users/me/proj/deploy.sh", "--env", "${env}"],
///   "timeout_secs": 120
/// }
/// ```
///
/// serde parsing enforces the path law and the name law (fail-closed: failing to parse a legal entry = parse failure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ScriptEntryRaw", into = "ScriptEntryRaw")]
pub struct ScriptToolEntry {
    /// Name (`script.{name}` / the plugin's tool segment). `[a-z0-9_-]{1,32}`, not a reserved word.
    pub name: String,
    /// Model-visible description.
    pub description: String,
    /// Parameter table (scalars only).
    pub parameters: BTreeMap<String, ParamSpec>,
    /// argv template: `${param}` substitution per element, no shell. The path law has already been enforced.
    pub run: Vec<String>,
    /// Hard timeout (milliseconds). `None` = [`DEFAULT_TIMEOUT_MS`].
    pub timeout_ms: Option<u64>,
    /// Environment variable overrides (layered on top of the inherited process env).
    pub env: BTreeMap<String, String>,
    /// Working directory (the current project root; pinned at product-layer registration, the framework does not sniff paths).
    /// Not stored in tools.json (machine-specific path, injected at product-layer load time).
    pub cwd: Option<PathBuf>,
    /// Tool-table budget tier, defaults to direct.
    pub exposure: ExposureSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScriptEntryRaw {
    name: String,
    description: String,
    #[serde(default)]
    parameters: BTreeMap<String, ParamSpec>,
    run: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "is_default_exposure")]
    exposure: ExposureSpec,
}

fn is_default_exposure(e: &ExposureSpec) -> bool {
    *e == ExposureSpec::default()
}

impl TryFrom<ScriptEntryRaw> for ScriptToolEntry {
    type Error = ScriptLawError;

    fn try_from(raw: ScriptEntryRaw) -> Result<Self, Self::Error> {
        let entry = Self {
            name: raw.name,
            description: raw.description,
            parameters: raw.parameters,
            run: raw.run,
            timeout_ms: raw.timeout_secs.map(|s| s.saturating_mul(1_000)),
            env: raw.env,
            cwd: None,
            exposure: raw.exposure,
        };
        entry.check_laws()?;
        Ok(entry)
    }
}

impl From<ScriptToolEntry> for ScriptEntryRaw {
    fn from(entry: ScriptToolEntry) -> Self {
        Self {
            name: entry.name,
            description: entry.description,
            parameters: entry.parameters,
            run: entry.run,
            timeout_secs: entry.timeout_ms.map(|ms| ms.div_ceil(1_000)),
            env: entry.env,
            exposure: entry.exposure,
        }
    }
}

impl ScriptToolEntry {
    /// Construct + enforce the laws (name law + path law). Manual construction also goes through here; never assemble the struct bare.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        run: Vec<String>,
    ) -> Result<Self, ScriptLawError> {
        let entry = Self {
            name: name.into(),
            description: description.into(),
            parameters: BTreeMap::new(),
            run,
            timeout_ms: None,
            env: BTreeMap::new(),
            cwd: None,
            exposure: ExposureSpec::default(),
        };
        entry.check_laws()?;
        Ok(entry)
    }

    /// Name law + path law (called at definition/install time; at execution time only check whether the script file is still there).
    pub fn check_laws(&self) -> Result<(), ScriptLawError> {
        ToolIdentity::script(&self.name).map_err(|_| ScriptLawError::BadName {
            name: self.name.clone(),
        })?;
        check_run_law(&self.run)
    }

    /// Hard timeout (milliseconds): the entry's value, defaulting to [`DEFAULT_TIMEOUT_MS`].
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS)
    }

    /// LLM parameter envelope (MCP `inputSchema` form: object + properties + required=all).
    pub fn parameters_envelope(&self) -> Value {
        parameters_envelope(&self.parameters)
    }

    /// Identity key (global-script form `script.{name}`). The plugin form is assembled by the plugin provider itself.
    pub fn identity(&self) -> ToolIdentity {
        // check_laws already guarantees validity; construction failure = programming error.
        ToolIdentity::script(&self.name).expect("entry name validated at construction")
    }
}

/// Parameter table → JSON Schema envelope: `type: object` + properties + required=all +
/// `additionalProperties: false` (in v1 all parameters are de facto required, see contract §appendix 3).
pub fn parameters_envelope(params: &BTreeMap<String, ParamSpec>) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, spec) in params {
        let mut schema = serde_json::Map::new();
        let ty = match spec.ty {
            ParamType::String => "string",
            ParamType::Number => "number",
            ParamType::Boolean => "boolean",
        };
        schema.insert("type".into(), json!(ty));
        if let Some(desc) = &spec.description {
            schema.insert("description".into(), json!(desc));
        }
        properties.insert(name.clone(), Value::Object(schema));
        required.push(json!(name));
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// Interpreter whitelist check: takes the file stem of `run[0]` (lowercased).
fn interpreter_of(program: &str) -> Option<&'static str> {
    let stem = Path::new(program)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    INTERPRETER_WHITELIST
        .iter()
        .find(|name| **name == stem)
        .copied()
}

/// Index of the interpreter's "script file argument": powershell/pwsh → after `-File`/`-f`;
/// cmd → after `/c`. Not found (or no argument after the flag) = cannot pin → error.
/// Not an interpreter → `Ok(None)` (the script is `run[0]` itself).
fn interpreter_script_arg_index(run: &[String]) -> Result<Option<usize>, ScriptLawError> {
    let Some((program, args)) = run.split_first() else {
        return Err(ScriptLawError::EmptyRun);
    };
    let Some(interpreter) = interpreter_of(program) else {
        return Ok(None);
    };
    let flags: &[&str] = if interpreter == "cmd" {
        &["/c"]
    } else {
        &["-file", "-f"]
    };
    for (offset, arg) in args.iter().enumerate() {
        if flags.iter().any(|f| arg.eq_ignore_ascii_case(f)) {
            // absolute index of the flag itself within run
            let flag_index = offset + 1;
            return run.get(flag_index + 1).map(|_| Some(flag_index + 1)).ok_or(
                ScriptLawError::InterpreterMissingScriptArg {
                    interpreter: interpreter.to_string(),
                },
            );
        }
    }
    Err(ScriptLawError::InterpreterMissingScriptArg {
        interpreter: interpreter.to_string(),
    })
}

/// Index of the "script file" element: interpreter → its script argument position; otherwise 0.
pub fn script_file_index(run: &[String]) -> Result<usize, ScriptLawError> {
    if run.is_empty() {
        return Err(ScriptLawError::EmptyRun);
    }
    Ok(interpreter_script_arg_index(run)?.unwrap_or(0))
}

/// Path law (anti-shadowing): `run[0]` is an absolute path, or the interpreter whitelist + an absolute script file argument.
pub fn check_run_law(run: &[String]) -> Result<(), ScriptLawError> {
    let Some((program, _)) = run.split_first() else {
        return Err(ScriptLawError::EmptyRun);
    };
    match interpreter_script_arg_index(run)? {
        Some(index) => {
            let arg = &run[index];
            if Path::new(arg).is_absolute() {
                Ok(())
            } else {
                Err(ScriptLawError::RelativeScriptArg {
                    interpreter: program.clone(),
                    arg: arg.clone(),
                })
            }
        }
        None => {
            if Path::new(program).is_absolute() {
                Ok(())
            } else {
                Err(ScriptLawError::RelativeProgram {
                    program: program.clone(),
                })
            }
        }
    }
}

/// Whether the script body is `run[0]` itself (non-interpreter form). Interpreter form → `false`
/// (the script is read in, no executable bit needed). Invalid/empty errors out on the same criteria as [`check_run_law`].
pub fn script_file_is_program(run: &[String]) -> Result<bool, ScriptLawError> {
    Ok(interpreter_script_arg_index(run)?.is_none())
}

/// Script file check at install/mount time (fail-closed, the same law as a missing wasm file):
/// it must be a regular file; when the script body is executed directly it must also have the
/// executable bit on unix.
///
/// `path` is supplied by the caller: plugin install checks the **staging area** contents (bytes
/// unchanged after the rename), global mount checks the final path. Not checked at load time —
/// archived entries are left to self-check at execution time (see [`ScriptToolEntry::check_laws`]);
/// once the file comes back the tool is alive again.
pub fn check_script_file_on_disk(
    path: &Path,
    require_executable: bool,
) -> Result<(), ScriptLawError> {
    let missing = ScriptLawError::ScriptFileMissing {
        path: path.display().to_string(),
    };
    let meta = std::fs::metadata(path).map_err(|_| missing.clone())?;
    if !meta.is_file() {
        return Err(missing);
    }
    #[cfg(unix)]
    if require_executable {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(ScriptLawError::ScriptFileNotExecutable {
                path: path.display().to_string(),
            });
        }
    }
    #[cfg(not(unix))]
    let _ = require_executable;
    Ok(())
}

/// Lexical normalization (does not touch the file system): consumes `.`, `..` steps back;
/// stepping back past the start = escape → `None`.
fn normalize_lexical(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

/// Resolve an in-plugin path (relative to the plugin root) to an absolute path and pin it.
///
/// - Relative path → resolved under `plugin_root`; absolute path → verified to be inside the plugin root;
/// - After lexical normalization it must still be under the plugin root (`../` escape → [`ScriptLawError::EscapedPluginRoot`]).
pub fn resolve_under_root(plugin_root: &Path, path: &str) -> Result<String, ScriptLawError> {
    let escape = |p: &str| ScriptLawError::EscapedPluginRoot {
        path: p.to_string(),
    };
    let root_norm = normalize_lexical(plugin_root).ok_or_else(|| escape(path))?;
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root_norm.join(path)
    };
    let normalized = normalize_lexical(&joined).ok_or_else(|| escape(path))?;
    if normalized.starts_with(&root_norm) {
        Ok(normalized.to_string_lossy().into_owned())
    } else {
        Err(escape(path))
    }
}

/// Plugin script pinning at install time: resolve the script file element to an absolute path
/// under the plugin root, then enforce the path law.
///
/// The same law as global scripts' "`run[0]` anti-shadowing" rule; plugins are not exempt.
pub fn pin_from_plugin_root(
    run: &[String],
    plugin_root: &Path,
) -> Result<Vec<String>, ScriptLawError> {
    let index = script_file_index(run)?;
    let mut pinned = run.to_vec();
    pinned[index] = resolve_under_root(plugin_root, &run[index])?;
    check_run_law(&pinned)?;
    Ok(pinned)
}

/// `${param}` substitution per element (argv, no shell).
///
/// - Only the `${name}` form is recognized; bare text such as `$FOO` or anything outside the
///   `${…}` form is left untouched (`${ENV:…}` has already been resolved to a literal at
///   product-layer registration, so none should remain by this point);
/// - Parameter values are inserted verbatim, **no re-expansion**;
/// - Missing, non-scalar (null/array/object), or unclosed `${` → [`AgentError::ToolArgsInvalid`]
///   (fail-closed, tool error for that call).
pub fn substitute_params(tool_key: &str, run: &[String], args: &Value) -> AgentResult<Vec<String>> {
    let Some(obj) = args.as_object() else {
        return Err(AgentError::ToolArgsInvalid {
            name: tool_key.to_string(),
            raw: args.to_string(),
        });
    };
    run.iter()
        .map(|element| substitute_element(tool_key, element, obj, args))
        .collect()
}

fn invalid_args(tool_key: &str, raw_args: &Value) -> AgentError {
    AgentError::ToolArgsInvalid {
        name: tool_key.to_string(),
        raw: raw_args.to_string(),
    }
}

fn substitute_element(
    tool_key: &str,
    element: &str,
    obj: &serde_json::Map<String, Value>,
    raw_args: &Value,
) -> AgentResult<String> {
    let mut out = String::with_capacity(element.len());
    let mut rest = element;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| invalid_args(tool_key, raw_args))?;
        let name = &after[..end];
        let value = obj
            .get(name)
            .ok_or_else(|| invalid_args(tool_key, raw_args))?;
        let scalar = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => return Err(invalid_args(tool_key, raw_args)), // null/array/object are non-scalar
        };
        out.push_str(&scalar); // inserted verbatim, no re-expansion
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Output truncation convention (copied verbatim from phi-kernel-tools `local_shell::truncate_output`):
/// first 1/3 + `...[output truncated]` + last 2/3.
fn truncate_output(s: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let available = max_chars.saturating_sub(TRUNCATION_MARKER.chars().count() + 2);
    let head_chars = available / 3;
    let tail_chars = available - head_chars;
    let head: String = s.chars().take(head_chars).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}\n{TRUNCATION_MARKER}\n{tail}")
}

/// Kill the whole process tree (on timeout/cancellation).
///
/// - POSIX: the script starts as a process group leader (`process_group(0)`), `killpg` SIGKILL;
/// - Windows: `taskkill /T /F /PID` (Job Object comes later; this semantically equivalent fallback
///   covers us for now).
async fn kill_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    #[cfg(unix)]
    {
        // process group = leader pid (process_group(0) at spawn).
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

/// Execute one script entry (shared by `ScriptProvider` / `ScriptTool` / the future plugin `script` form).
///
/// `tool_key` is the identity key (carried in errors); `args` MUST already have been validated by
/// the host against the envelope.
pub async fn execute_entry(
    tool_key: &str,
    entry: &ScriptToolEntry,
    args: &Value,
    ctx: &ToolContext,
) -> AgentResult<Vec<Content>> {
    let argv = substitute_params(tool_key, &entry.run, args)?;

    // Script gone → "script is gone" error, no silent file swap.
    let script_index = script_file_index(&entry.run).map_err(|e| AgentError::ToolExecution {
        name: tool_key.to_string(),
        source: Box::new(AgentError::Internal(e.to_string())),
    })?;
    let script_path = Path::new(&argv[script_index]);
    if !script_path.is_file() {
        return Err(AgentError::ToolExecution {
            name: tool_key.to_string(),
            source: Box::new(AgentError::Internal(format!(
                "script is gone: {}",
                script_path.display()
            ))),
        });
    }

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .envs(&entry.env) // inherit process env + entry overrides
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = &entry.cwd {
        cmd.current_dir(dir);
    }
    #[cfg(unix)]
    cmd.process_group(0); // own process group, leader = this script → killpg the whole tree
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

    let mut child = cmd.spawn().map_err(|e| AgentError::ToolExecution {
        name: tool_key.to_string(),
        source: Box::new(AgentError::Internal(format!(
            "failed to spawn {}: {e}",
            argv[0]
        ))),
    })?;
    let pid = child.id();

    // Concurrent pipe reads (wait and the read ends don't exclude each other; leaked background
    // children holding the pipes get a fallback deadline).
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });

    // Timeout / cancellation / normal exit. Bind before matching so the wait future is dropped
    // before `child` is finalized (a match scrutinee temporary lives until the match ends).
    enum WaitEnd {
        Exited(std::io::Result<std::process::ExitStatus>),
        Cancelled,
    }
    let timeout = Duration::from_millis(entry.timeout_ms());
    let wait_future = async {
        tokio::select! {
            _ = ctx.cancel_token.cancelled() => WaitEnd::Cancelled,
            status = child.wait() => WaitEnd::Exited(status),
        }
    };
    let ended = tokio::time::timeout(timeout, wait_future).await;

    let status = match ended {
        Err(_elapsed) => {
            // Timeout: kill the whole tree (the tool dies, not the host).
            kill_tree(pid).await;
            let _ = child.wait().await;
            return Err(AgentError::ToolTimeout);
        }
        Ok(WaitEnd::Cancelled) => {
            kill_tree(pid).await;
            let _ = child.wait().await;
            return Err(AgentError::Cancelled);
        }
        Ok(WaitEnd::Exited(Err(e))) => {
            return Err(AgentError::ToolExecution {
                name: tool_key.to_string(),
                source: Box::new(AgentError::Internal(format!("wait failed: {e}"))),
            });
        }
        Ok(WaitEnd::Exited(Ok(status))) => status,
    };

    let read_deadline = Duration::from_secs(5);
    let stdout = tokio::time::timeout(read_deadline, stdout_task)
        .await
        .ok()
        .and_then(|t| t.ok())
        .unwrap_or_default();
    let stderr = tokio::time::timeout(read_deadline, stderr_task)
        .await
        .ok()
        .and_then(|t| t.ok())
        .unwrap_or_default();
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();

    if !status.success() {
        // exit ≠ 0 → tool error (with a stderr summary).
        let summary = if stderr.trim().is_empty() {
            stdout.clone()
        } else {
            stderr.clone()
        };
        let summary = truncate_output(summary.trim(), ctx.max_output_chars);
        return Err(AgentError::ToolExecution {
            name: tool_key.to_string(),
            source: Box::new(AgentError::Internal(format!(
                "script exited with {status}: {summary}"
            ))),
        });
    }

    let body = if stdout.trim().is_empty() {
        stderr
    } else {
        stdout
    };
    Ok(vec![Content::text(truncate_output(
        body.trim_end_matches('\n'),
        ctx.max_output_chars,
    ))])
}

/// Global script tool supply: namespace `script.{name}`, a name → entry table.
///
/// Contract invariant: `call` rejects every foreign key (builtin / `mcp.` / `plugin.` all get
/// [`AgentError::ToolNotFound`]).
#[derive(Debug, Default)]
pub struct ScriptProvider {
    entries: BTreeMap<String, ScriptToolEntry>,
}

impl ScriptProvider {
    /// Build the table in bulk; **duplicate names keep the first one** (design: skip the later one
    /// at load time and warn — the warning lives at the product layer).
    pub fn new(entries: impl IntoIterator<Item = ScriptToolEntry>) -> Self {
        let mut provider = Self::default();
        for entry in entries {
            provider.entries.entry(entry.name.clone()).or_insert(entry);
        }
        provider
    }

    /// Add/replace one entry.
    pub fn insert(&mut self, entry: ScriptToolEntry) -> Option<ScriptToolEntry> {
        self.entries.insert(entry.name.clone(), entry)
    }

    /// Remove one entry (used by `/tools remove`; ledger cleanup is the product layer's job, see approval_ledger).
    pub fn remove(&mut self, name: &str) -> Option<ScriptToolEntry> {
        self.entries.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&ScriptToolEntry> {
        self.entries.get(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[async_trait]
impl ToolProvider for ScriptProvider {
    async fn list(&self) -> AgentResult<Vec<ToolDescriptor>> {
        Ok(self
            .entries
            .values()
            .map(|entry| ToolDescriptor {
                identity: entry.identity(),
                description: entry.description.clone(),
                parameters: entry.parameters_envelope(),
                timeout_ms: Some(entry.timeout_ms()),
                exposure: entry.exposure.into(),
            })
            .collect())
    }

    async fn call(
        &self,
        identity: &ToolIdentity,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<Vec<Content>> {
        let ToolIdentity::Script { name } = identity else {
            return Err(AgentError::ToolNotFound {
                name: identity.to_key(),
            });
        };
        let Some(entry) = self.entries.get(name) else {
            return Err(AgentError::ToolNotFound {
                name: identity.to_key(),
            });
        };
        execute_entry(&identity.to_key(), entry, args, ctx).await
    }

    async fn health(&self) -> AgentResult<()> {
        // Cheap probe: is the script file still there.
        for entry in self.entries.values() {
            let index = script_file_index(&entry.run)
                .map_err(|e| AgentError::Internal(format!("script.{}: {e}", entry.name)))?;
            let path = Path::new(&entry.run[index]);
            if !path.is_file() {
                return Err(AgentError::Internal(format!(
                    "script.{} is gone: {}",
                    entry.name,
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

/// A single script tool (plugs into the [`Tool`] table; `Box::leak` naming follows the existing
/// `McpToolAdapter` precedent).
///
/// Global scripts use [`ScriptTool::new`] (key = `script.{name}`); the plugin `script` form uses
/// [`ScriptTool::with_key`] in P2 to mount `plugin.{plugin}.{tool}`.
pub struct ScriptTool {
    name: &'static str,
    description: &'static str,
    schema: Value,
    entry: ScriptToolEntry,
}

impl ScriptTool {
    /// Global script tool: key = `script.{name}`.
    pub fn new(entry: ScriptToolEntry) -> Self {
        let key = format!("script.{}", entry.name);
        Self::with_key(&key, entry)
    }

    /// Explicit identity key (plugin `script` form: `plugin.{plugin}.{tool}`).
    pub fn with_key(identity_key: &str, entry: ScriptToolEntry) -> Self {
        let name: &'static str = Box::leak(identity_key.to_string().into_boxed_str());
        let description: &'static str = Box::leak(entry.description.clone().into_boxed_str());
        let schema = entry.parameters_envelope();
        Self {
            name,
            description,
            schema,
            entry,
        }
    }

    pub fn entry(&self) -> &ScriptToolEntry {
        &self.entry
    }
}

#[async_trait]
impl Tool for ScriptTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        execute_entry(self.name, &self.entry, args, ctx).await
    }

    fn timeout_ms(&self) -> Option<u64> {
        Some(self.entry.timeout_ms())
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name.to_string(),
            description: self.description.to_string(),
            origin: "script".to_string(),
            version: "unknown".to_string(),
            requirements: vec![],
        }
    }

    fn exposure(&self) -> ToolExposure {
        self.entry.exposure.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn s(v: &str) -> String {
        v.to_string()
    }

    /// Platform absolute path (the test criterion follows the same law as `Path::is_absolute`, runs
    /// the same on unix/windows).
    fn abs(rel: &str) -> String {
        std::env::current_dir()
            .unwrap()
            .join(rel)
            .to_string_lossy()
            .into_owned()
    }

    fn entry(run: &[&str]) -> ScriptToolEntry {
        ScriptToolEntry::new(
            "deploy",
            "deploy the project",
            run.iter().map(|s| s.to_string()).collect(),
        )
        .expect("entry must pass the path law")
    }

    // ── Path law (anti-shadowing) ────────────────────────────────────────────

    #[test]
    fn run_law_accepts_absolute_program() {
        let program = abs("deploy.sh");
        assert!(check_run_law(&[s(&program), s("--env"), s("prod")]).is_ok());
    }

    #[test]
    fn run_law_rejects_relative_program_global_anti_shadowing() {
        // Global tools.json + relative path = another project shadowing a same-named script → reject.
        for program in ["./deploy.sh", "deploy.sh", "../deploy.sh", "~/deploy.sh"] {
            let err = check_run_law(&[s(program)]).unwrap_err();
            assert!(
                matches!(err, ScriptLawError::RelativeProgram { .. }),
                "{program}: {err}"
            );
        }
    }

    #[test]
    fn run_law_interpreter_whitelist_requires_absolute_script_arg() {
        // Whitelisted interpreter: the script file argument must be an absolute path.
        for (interpreter, flag, script_rel) in [
            ("powershell", "-File", "deploy.ps1"),
            ("pwsh", "-f", "deploy.ps1"),
            ("powershell.exe", "-File", "deploy.ps1"),
            ("cmd", "/c", "deploy.cmd"),
            ("cmd.exe", "/C", "deploy.cmd"),
        ] {
            let script = abs(script_rel);
            let ok = vec![s(interpreter), s(flag), s(&script), s("prod")];
            assert!(check_run_law(&ok).is_ok(), "{interpreter} absolute script");

            let bad = vec![s(interpreter), s(flag), s("./deploy.ps1")];
            let err = check_run_law(&bad).unwrap_err();
            assert!(
                matches!(err, ScriptLawError::RelativeScriptArg { .. }),
                "{interpreter} relative script: {err}"
            );

            // No script argument found → cannot pin → reject.
            let missing = vec![s(interpreter), s("-Command"), s("echo hi")];
            let err = check_run_law(&missing).unwrap_err();
            assert!(
                matches!(err, ScriptLawError::InterpreterMissingScriptArg { .. }),
                "{interpreter} missing script arg: {err}"
            );
        }
    }

    #[test]
    fn run_law_empty_run_rejected() {
        assert_eq!(check_run_law(&[]).unwrap_err(), ScriptLawError::EmptyRun);
    }

    #[test]
    fn non_interpreter_bare_name_is_not_exempt() {
        // `npx` is not on the whitelist → run[0] must still be an absolute path.
        let err = check_run_law(&[s("npx"), s("-y"), s("pkg")]).unwrap_err();
        assert!(matches!(err, ScriptLawError::RelativeProgram { .. }));
    }

    // ── In-plugin path pinning (the same law, plugins are not exempt) ─────────────────────

    #[test]
    fn pin_from_plugin_root_pins_relative_script_absolute() {
        let root = PathBuf::from(abs("plugins/my-tools"));
        let pinned =
            pin_from_plugin_root(&[s("scripts/deploy.sh"), s("--env"), s("${env}")], &root)
                .unwrap();
        assert_eq!(
            Path::new(&pinned[0]),
            root.join("scripts/deploy.sh").as_path(),
            "package-relative path pinned to an absolute path under the plugin root"
        );
        assert!(Path::new(&pinned[0]).is_absolute());
        // After pinning it must pass the same path law
        assert!(check_run_law(&pinned).is_ok());
    }

    #[test]
    fn pin_from_plugin_root_rejects_escape_relative_and_absolute() {
        let root = PathBuf::from(abs("plugins/my-tools"));
        let outside = abs("plugins/other/x.sh");
        for bad in [
            vec![s("../outside.sh")],
            vec![s("scripts/../../outside.sh")],
            vec![s(&outside)],
        ] {
            let err = pin_from_plugin_root(&bad, &root).unwrap_err();
            assert!(
                matches!(err, ScriptLawError::EscapedPluginRoot { .. }),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn pin_from_plugin_root_pins_interpreter_script_arg() {
        let root = PathBuf::from(abs("plugins/my-tools"));
        let pinned = pin_from_plugin_root(
            &[
                s("powershell"),
                s("-File"),
                s("scripts/deploy.ps1"),
                s("-env"),
                s("prod"),
            ],
            &root,
        )
        .unwrap();
        assert_eq!(
            pinned[0], "powershell",
            "the interpreter itself is not pinned (whitelist)"
        );
        assert_eq!(
            Path::new(&pinned[2]),
            root.join("scripts/deploy.ps1").as_path(),
            "the interpreter's script argument is pinned"
        );
        // An escaping interpreter script argument is rejected just the same
        let err =
            pin_from_plugin_root(&[s("powershell"), s("-File"), s("../x.ps1")], &root).unwrap_err();
        assert!(matches!(err, ScriptLawError::EscapedPluginRoot { .. }));
    }

    // ── ${param} substitution (argv, fail-closed, no re-expansion) ──────────────────────

    #[test]
    fn substitute_replaces_whole_and_inline_placeholders() {
        let run = [
            s("/abs/deploy.sh"),
            s("--env"),
            s("${env}"),
            s("--tag=${tag}"),
        ];
        let args = json!({"env": "prod", "tag": "v1"});
        let got = substitute_params("script.deploy", &run, &args).unwrap();
        assert_eq!(got, vec!["/abs/deploy.sh", "--env", "prod", "--tag=v1"]);
    }

    #[test]
    fn substitute_missing_param_is_tool_error() {
        let run = [s("/abs/deploy.sh"), s("${env}")];
        let err = substitute_params("script.deploy", &run, &json!({})).unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { name, .. } if name == "script.deploy"));
    }

    #[test]
    fn substitute_non_scalar_param_is_tool_error() {
        let run = [s("/abs/deploy.sh"), s("${env}")];
        for bad in [
            json!({"env": null}),
            json!({"env": ["a"]}),
            json!({"env": {"a": 1}}),
        ] {
            let err = substitute_params("script.deploy", &run, &bad).unwrap_err();
            assert!(matches!(err, AgentError::ToolArgsInvalid { .. }), "{bad}");
        }
    }

    #[test]
    fn substitute_scalar_params_stringify() {
        let run = [s("${n}"), s("${b}"), s("${s}")];
        let args = json!({"n": 42, "b": true, "s": "txt"});
        let got = substitute_params("script.deploy", &run, &args).unwrap();
        assert_eq!(got, vec!["42", "true", "txt"]);
    }

    #[test]
    fn substitute_does_not_reexpand_values() {
        // ${x} inside a parameter value is not re-expanded; bare $FOO / non-${ forms are untouched.
        let run = [s("${a}"), s("$FOO"), s("plain")];
        let args = json!({"a": "${b}", "b": "NOPE"});
        let got = substitute_params("script.deploy", &run, &args).unwrap();
        assert_eq!(got, vec!["${b}", "$FOO", "plain"]);
    }

    #[test]
    fn substitute_unclosed_placeholder_fails_closed() {
        let err = substitute_params(
            "script.deploy",
            &[s("/abs/x.sh ${env")],
            &json!({"env": "p"}),
        )
        .unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { .. }));
    }

    #[test]
    fn substitute_non_object_args_fails_closed() {
        let err = substitute_params("script.deploy", &[s("x")], &json!(["a"])).unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { .. }));
    }

    // ── serde (tools.json entry form, parsing enforces the laws) ───────────────────────────

    #[test]
    fn entry_serde_round_trips_design_shape() {
        let script = abs("proj/deploy.sh");
        let raw = json!({
            "name": "deploy",
            "description": "部署当前项目到 staging",
            "parameters": { "env": { "type": "string", "description": "目标环境" } },
            "run": [script, "--env", "${env}"],
            "timeout_secs": 120
        });
        let entry: ScriptToolEntry = serde_json::from_value(raw).unwrap();
        assert_eq!(entry.name, "deploy");
        assert_eq!(entry.timeout_ms(), 120_000);
        assert_eq!(entry.parameters["env"].ty, ParamType::String);
        let back = serde_json::to_value(&entry).unwrap();
        assert_eq!(back["timeout_secs"], 120);
        assert_eq!(back["run"][0], script);
    }

    #[test]
    fn entry_serde_rejects_shadowable_run() {
        let raw = json!({
            "name": "deploy",
            "description": "d",
            "run": ["deploy.sh"]
        });
        let err = serde_json::from_value::<ScriptToolEntry>(raw).unwrap_err();
        assert!(err.to_string().contains("deploy.sh"), "{err}");
    }

    #[test]
    fn entry_serde_rejects_bad_name() {
        let raw = json!({
            "name": "Deploy",
            "description": "d",
            "run": [abs("deploy.sh")]
        });
        assert!(serde_json::from_value::<ScriptToolEntry>(raw).is_err());
    }

    #[test]
    fn parameters_envelope_requires_all_scalars() {
        let script = abs("deploy.sh");
        let mut entry = entry(&[script.as_str()]);
        entry.parameters.insert(
            "env".into(),
            ParamSpec {
                ty: ParamType::String,
                description: Some("目标环境".into()),
            },
        );
        let envelope = entry.parameters_envelope();
        assert_eq!(envelope["type"], "object");
        assert_eq!(envelope["properties"]["env"]["type"], "string");
        assert_eq!(envelope["required"], json!(["env"]));
        assert_eq!(envelope["additionalProperties"], false);
    }

    // ── Execution semantics (actually run on unix; Windows semantics see the §win tests + docs) ────

    #[cfg(unix)]
    mod unix_exec {
        use super::*;

        #[tokio::test]
        async fn runs_argv_without_shell_and_substitutes_params() {
            let entry = entry(&["/bin/echo", "${msg}"]);
            let ctx = ToolContext::for_test();
            let out = execute_entry("script.echo", &entry, &json!({"msg": "hello"}), &ctx)
                .await
                .unwrap();
            assert_eq!(out.len(), 1);
            assert_eq!(agent_base::tool::content_text(&out), "hello");
        }

        #[tokio::test]
        async fn env_inherits_and_overrides() {
            // argv goes through no shell; env override verification relies on /bin/sh -c (an absolute run[0] is legal).
            // Since edition 2024 env::set_var is unsafe (races with multithreaded env access).
            unsafe { std::env::set_var("SCRIPT_TOOL_TEST_INHERITED", "inh") };
            let mut entry = entry(&[
                "/bin/sh",
                "-c",
                r#"echo "$SCRIPT_TOOL_TEST_INHERITED/$SCRIPT_TOOL_TEST_SET""#,
            ]);
            entry
                .env
                .insert("SCRIPT_TOOL_TEST_SET".into(), "set".into());
            let ctx = ToolContext::for_test();
            let out = execute_entry("script.env", &entry, &json!({}), &ctx)
                .await
                .unwrap();
            assert_eq!(agent_base::tool::content_text(&out), "inh/set");
        }

        #[tokio::test]
        async fn non_zero_exit_reports_stderr_summary() {
            let entry = entry(&["/bin/sh", "-c", "echo boom >&2; exit 3"]);
            let ctx = ToolContext::for_test();
            let err = execute_entry("script.fail", &entry, &json!({}), &ctx)
                .await
                .unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("boom"), "{msg}");
            assert!(msg.contains("3"), "{msg}");
        }

        #[tokio::test]
        async fn timeout_kills_whole_process_tree() {
            // Grandchild process sleeps 30 and records its pid; after the timeout the whole tree must die.
            let dir = tempfile::tempdir().unwrap();
            let pidfile = dir.path().join("grandchild.pid");
            let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
            let mut entry = entry(&["/bin/sh", "-c", &script]);
            entry.timeout_ms = Some(1_000);
            let ctx = ToolContext::for_test();
            let err = execute_entry("script.tree", &entry, &json!({}), &ctx)
                .await
                .unwrap_err();
            assert!(matches!(err, AgentError::ToolTimeout), "{err}");

            let pid_raw = std::fs::read_to_string(&pidfile).expect("grandchild pidfile");
            let pid: i32 = pid_raw.trim().parse().unwrap();
            for _ in 0..50 {
                let alive = unsafe { libc::kill(pid, 0) == 0 };
                if !alive {
                    return; // the grandchild died with the tree
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            panic!("grandchild {pid} survived the tree kill");
        }

        #[tokio::test]
        async fn cancel_token_stops_script_and_reports_cancelled() {
            let dir = tempfile::tempdir().unwrap();
            let pidfile = dir.path().join("grandchild.pid");
            let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
            let entry = entry(&["/bin/sh", "-c", &script]);
            let ctx = ToolContext::for_test();
            let token = ctx.cancel_token.clone();
            let run = tokio::spawn(async move {
                execute_entry("script.cancel", &entry, &json!({}), &ctx).await
            });
            // Wait for the pidfile to appear (the tree has grown) before cancelling
            for _ in 0..100 {
                if pidfile.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            token.cancel();
            let err = run.await.unwrap().unwrap_err();
            assert!(matches!(err, AgentError::Cancelled), "{err}");
        }

        #[tokio::test]
        async fn output_truncation_follows_kernel_convention() {
            let entry = entry(&["/bin/sh", "-c", "seq 1 1000"]);
            let mut ctx = ToolContext::for_test();
            ctx.max_output_chars = Some(60);
            let out = execute_entry("script.big", &entry, &json!({}), &ctx)
                .await
                .unwrap();
            let text = agent_base::tool::content_text(&out);
            assert!(text.contains(TRUNCATION_MARKER), "{text}");
            assert!(
                text.chars().count() < 200,
                "still huge: {}",
                text.chars().count()
            );
            // Both head and tail are present (seq runs from 1 to 1000)
            assert!(text.starts_with('1'), "{text}");
            assert!(text.contains("1000"), "{text}");
        }

        #[tokio::test]
        async fn missing_script_file_reports_script_is_gone() {
            let entry = entry(&["/definitely/not/here/deploy.sh"]);
            let ctx = ToolContext::for_test();
            let err = execute_entry("script.gone", &entry, &json!({}), &ctx)
                .await
                .unwrap_err();
            assert!(err.to_string().contains("script is gone"), "{err}");
        }
    }

    // ── Provider / Tool wiring ───────────────────────────────────────────────

    #[tokio::test]
    async fn provider_lists_scripts_and_rejects_foreign_keys() {
        #[cfg(unix)]
        let entry = entry(&["/bin/echo", "hi"]);
        #[cfg(windows)]
        let entry = entry(&[r"C:\Windows\System32\cmd.exe", "/c", r"C:\proj\deploy.cmd"]);

        let provider = ScriptProvider::new([entry.clone()]);
        let listed = provider.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].identity.to_key(), "script.deploy");
        assert_eq!(listed[0].timeout_ms, Some(DEFAULT_TIMEOUT_MS));

        let ctx = ToolContext::for_test();
        for foreign in [
            ToolIdentity::builtin("read_file").unwrap(),
            ToolIdentity::mcp("figma", "get_file").unwrap(),
            ToolIdentity::plugin("my-tools", "deploy").unwrap(),
            ToolIdentity::script("other").unwrap(),
        ] {
            let err = provider.call(&foreign, &json!({}), &ctx).await.unwrap_err();
            assert!(
                matches!(err, AgentError::ToolNotFound { .. }),
                "{foreign:?}"
            );
        }

        let own = entry.identity();
        provider.call(&own, &json!({}), &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn provider_health_detects_gone_script() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("deploy.sh");
        std::fs::write(&script, "#!/bin/sh\necho ok\n").unwrap();
        let entry = entry(&[script.to_str().unwrap()]);
        let provider = ScriptProvider::new([entry.clone()]);
        provider.health().await.unwrap();

        std::fs::remove_file(&script).unwrap();
        let err = provider.health().await.unwrap_err();
        assert!(err.to_string().contains("is gone"), "{err}");
    }

    #[tokio::test]
    async fn script_tool_plugs_into_tool_table() {
        let entry = entry(&["/bin/echo", "hi"]);
        let tool = ScriptTool::new(entry.clone());
        assert_eq!(tool.name(), "script.deploy");
        assert_eq!(tool.timeout_ms(), Some(DEFAULT_TIMEOUT_MS));
        assert_eq!(tool.metadata().origin, "script");
        assert_eq!(tool.schema()["type"], "object");

        let plugin_tool = ScriptTool::with_key("plugin.my-tools.deploy", entry);
        assert_eq!(plugin_tool.name(), "plugin.my-tools.deploy");
    }
}
