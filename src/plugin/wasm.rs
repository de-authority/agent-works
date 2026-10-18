//! Plugin wasm form: loading and execution via the wasmtime Component Model.
//!
//! Criteria (`notes/tool-socket-design.md`, "Plugin package mechanism"):
//!
//! - **WIT contract** `phi:tool@0.1.0` `executor.invoke(tool, args)` — both the host
//!   and author-side bindings consume `wit/phi-tool.wit` (`wit/tool-guest-world.wit` is
//!   just build glue);
//! - **Capability tightening only honors pinned registry values** ([`ResolvedCapability`]),
//!   not the plugin manifest: fs only preopens pinned paths (undeclared = no preopen =
//!   cannot read anything); net only allows pinned hosts ([`HostFilterHooks`] filters
//!   per request, anything not on the whitelist is denied); raw sockets are denied by
//!   default (no `inherit_network`, all wasi:sockets operations return
//!   PermissionDenied); when net is undeclared, `wasi:http` is not even linked
//!   (cannot be called);
//! - **Timeout/suspension kills the tool, not the host**: epoch interruption cuts it
//!   off ([`Trap::Interrupt`] → [`WasmError::Timeout`]), guest panic/trap
//!   → [`WasmError::Trap`] — the host keeps serving other tools (the second call in
//!   tests verifies liveness);
//! - **Compile at load time**: a corrupt .wasm errors at the moment of
//!   [`WasmRuntime::compile`], and the loader disables just that tool instead of
//!   skipping the whole package.
//!
//! secret rule: the wasm side only sees resolved literals (`${ENV:VAR}` is resolved
//! by the product layer before connection/environment injection); the host never
//! resolves or persists any reference.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext, ToolExposure, ToolMetadata};
use async_trait::async_trait;
use serde_json::Value;
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store, Trap};
use wasmtime_wasi::p2::pipe::MemoryOutputPipe;
use wasmtime_wasi::{FsPerms, ResourceTable, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wasmtime_wasi_http::p2::add_only_http_to_linker_async;
use wasmtime_wasi_http::{
    Error as HttpError, RequestOptions, WasiBody, WasiHttpCtx, WasiHttpCtxView, WasiHttpHooks,
    WasiHttpView,
};

use super::manifest::ResolvedCapability;
use crate::script_tool::{ParamSpec, parameters_envelope};
use std::collections::BTreeMap;

/// epoch tick granularity (timeout quantization unit).
const EPOCH_TICK_MS: u64 = 10;

wasmtime::component::bindgen!({
    path: "wit",
    world: "tool-guest",
    exports: { default: async },
});

/// Load/execution errors for the wasm form. All of them hang only on the tool side
/// (call or load), never on the host — a host panic is not covered here, that is a bug.
#[derive(Debug)]
pub enum WasmError {
    /// Failed to read the component file.
    Io(std::io::Error),
    /// Component compilation failed (corrupt / not a component / version mismatch)
    /// → disable the tool at load time.
    Compile(String),
    /// epoch interruption cut the call off (timed out / stuck).
    Timeout { timeout: Duration },
    /// guest trap / panic.
    Trap(String),
    /// the guest explicitly returned an error (`result::err`).
    Guest(String),
    /// capability denied (e.g. net outside the whitelist).
    Denied(String),
    /// host linking/instantiation failed.
    Host(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "wasm io: {e}"),
            Self::Compile(e) => write!(f, "wasm compile failed: {e}"),
            Self::Timeout { timeout } => {
                write!(f, "wasm call timed out after {}ms", timeout.as_millis())
            }
            Self::Trap(e) => write!(f, "wasm trapped: {e}"),
            Self::Guest(e) => write!(f, "wasm tool error: {e}"),
            Self::Denied(e) => write!(f, "wasm capability denied: {e}"),
            Self::Host(e) => write!(f, "wasm host error: {e}"),
        }
    }
}

impl std::error::Error for WasmError {}

/// Input for one wasm call. `capabilities` MUST be pinned registry values
/// (enforcement does not honor the plugin's own manifest).
pub struct WasmCall<'a> {
    /// the `tool` parameter of `executor.invoke` (the `name` of the wasm entry in the manifest).
    pub tool: &'a str,
    /// validated JSON argument text (the host MUST validate against the parameters schema first).
    pub args_json: &'a str,
    /// pinned capability whitelist values.
    pub capabilities: &'a [ResolvedCapability],
    /// hard timeout.
    pub timeout: Duration,
}

/// wasm runtime: one engine + one epoch tick thread (shared within the process).
pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    /// Build a runtime (component model + async + epoch interruption).
    pub fn new() -> Result<Self, WasmError> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| WasmError::Host(e.to_string()))?;

        // The heartbeat behind the timeout criterion: bump the epoch every tick;
        // callers convert the timeout into a deadline.
        let ticker = engine.clone();
        std::thread::Builder::new()
            .name("plugin-wasm-epoch".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_millis(EPOCH_TICK_MS));
                    ticker.increment_epoch();
                }
            })
            .map_err(WasmError::Io)?;
        Ok(Self { engine })
    }

    /// Compile one component at load time (failure = the error surface that disables
    /// this tool, not skipping the whole package).
    pub fn compile(&self, module_path: &Path) -> Result<Component, WasmError> {
        Component::from_file(&self.engine, module_path)
            .map_err(|e| WasmError::Compile(e.to_string()))
    }

    /// Execute one call. Timeout/trap/capability denial only hangs this call.
    pub async fn invoke(
        &self,
        component: &Component,
        call: &WasmCall<'_>,
    ) -> Result<String, WasmError> {
        let state = HostState::new(call.capabilities)?;
        let mut linker: Linker<HostState> = Linker::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| WasmError::Host(e.to_string()))?;
        // net capability: undeclared = don't even link wasi:http (the guest cannot call it).
        if call
            .capabilities
            .iter()
            .any(|c| matches!(c, ResolvedCapability::NetHttp { .. }))
        {
            add_only_http_to_linker_async(&mut linker)
                .map_err(|e| WasmError::Host(e.to_string()))?;
        }

        let mut store = Store::new(&self.engine, state);
        let ticks = (call.timeout.as_millis() as u64 / EPOCH_TICK_MS).max(1);
        store.set_epoch_deadline(ticks);

        let bindings = ToolGuest::instantiate_async(&mut store, component, &linker)
            .await
            .map_err(classify_host_error)?;

        // When the exported interface comes from package `phi:tool`, the accessor name
        // carries the package-name prefix (bindgen naming).
        let result = bindings
            .phi_tool_executor()
            .call_invoke(&mut store, call.tool, call.args_json)
            .await;

        match result {
            Ok(Ok(json)) => Ok(json),
            Ok(Err(message)) => Err(WasmError::Guest(message)),
            Err(e) => Err(classify_trap(e, call.timeout)),
        }
    }
}

/// Find the [`Trap`] in the error (including the root cause — the component layer may
/// wrap a trap with context).
fn as_trap(e: &wasmtime::Error) -> Option<Trap> {
    e.downcast_ref::<Trap>()
        .copied()
        .or_else(|| e.root_cause().downcast_ref::<Trap>().copied())
}

/// Call-time error classification: epoch interruption = timeout, any other trap = the tool hangs.
fn classify_trap(e: wasmtime::Error, timeout: Duration) -> WasmError {
    match as_trap(&e) {
        Some(Trap::Interrupt) => WasmError::Timeout { timeout },
        Some(trap) => WasmError::Trap(trap.to_string()),
        None => WasmError::Host(e.to_string()),
    }
}

/// Instantiation-time errors (missing import / linker mismatch) likewise hang only the tool side.
fn classify_host_error(e: wasmtime::Error) -> WasmError {
    match as_trap(&e) {
        Some(trap) => WasmError::Trap(trap.to_string()),
        None => WasmError::Host(e.to_string()),
    }
}

// ── Host state and capability tightening ────────────────────────────────────

struct HostState {
    wasi: WasiCtx,
    table: ResourceTable,
    http: WasiHttpCtx,
    hooks: HostFilterHooks,
}

impl HostState {
    /// Assemble from pinned capabilities: fs only preopens pinned paths; the net whitelist
    /// goes into hooks; sockets are denied by default (no inherit_network).
    fn new(capabilities: &[ResolvedCapability]) -> Result<Self, WasmError> {
        let mut builder = WasiCtxBuilder::new();
        // The guest's println/panic goes to in-memory pipes — fixtures don't pollute test
        // output; a plugin tool's output goes through the invoke return value, not stdout.
        builder.stdout(MemoryOutputPipe::new(64 * 1024));
        builder.stderr(MemoryOutputPipe::new(16 * 1024));

        for cap in capabilities {
            if let ResolvedCapability::FsRead { paths } = cap {
                for path in paths {
                    // guest name = pinned absolute path: this is the name authorized in the model/manifest.
                    builder
                        .preopened_dir(path, path.as_str(), FsPerms::ReadOnly)
                        .map_err(|e| WasmError::Host(format!("preopen {path}: {e}")))?;
                }
            }
        }

        let mut allowed_hosts = Vec::new();
        for cap in capabilities {
            if let ResolvedCapability::NetHttp { hosts } = cap {
                allowed_hosts.extend(hosts.iter().cloned());
            }
        }

        Ok(Self {
            wasi: builder.build(),
            table: ResourceTable::new(),
            http: WasiHttpCtx::new(),
            hooks: HostFilterHooks { allowed_hosts },
        })
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for HostState {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            hooks: &mut self.hooks,
            table: &mut self.table,
            ctx: &mut self.http,
        }
    }
}

/// net:http capability tightening: filter each request against the pinned host
/// whitelist; not on the whitelist = denied (fail-closed; an empty list denies everything).
struct HostFilterHooks {
    allowed_hosts: Vec<String>,
}

impl WasiHttpHooks for HostFilterHooks {
    fn send_request(
        &mut self,
        request: http::Request<WasiBody>,
        options: Option<RequestOptions>,
        fut: Box<dyn std::future::Future<Output = Result<(), HttpError>> + Send>,
    ) -> Box<
        dyn std::future::Future<
                Output = Result<
                    (
                        http::Response<WasiBody>,
                        Box<dyn std::future::Future<Output = Result<(), HttpError>> + Send>,
                    ),
                    HttpError,
                >,
            > + Send,
    > {
        let Some(host) = request.uri().host().map(str::to_ascii_lowercase) else {
            return Box::new(std::future::ready(Err(HttpError::InternalError(Some(
                "plugin net:http: request has no host".to_string(),
            )))));
        };
        if !host_allowed(&self.allowed_hosts, &host) {
            tracing::warn!(target: "plugin", host = %host, "net:http denied by capability whitelist");
            return Box::new(std::future::ready(Err(HttpError::HttpRequestDenied)));
        }
        // Allowed → go through the default implementation (real outbound).
        DefaultHooks.send_request(request, options, fut)
    }
}

/// Delegate to the default outbound implementation (the default `send_request`
/// method body is reused as-is).
struct DefaultHooks;

impl WasiHttpHooks for DefaultHooks {}

/// Host whitelist criterion (already normalized to lowercase / whitespace-stripped at
/// install time; an empty list denies everything).
///
/// Only compares the host itself (without port); exact match, no suffix wildcarding —
/// wildcards are a future product-layer thing; the secure default is tight first.
pub fn host_allowed(allowed_hosts: &[String], host: &str) -> bool {
    let host = host.trim().to_ascii_lowercase();
    !host.is_empty() && allowed_hosts.contains(&host)
}

// ── Tool table adapter (`plugin.{plugin}.{tool}`) ───────────────────────────

/// One row of the wasm tool table: in-process component + pinned capabilities + hard timeout.
///
/// Naming follows the same law as [`crate::script_tool::ScriptTool::with_key`]
/// (`Box::leak` follows the existing `McpToolAdapter` precedent). The execution surface
/// hangs tools, not the host: timeout/trap/capability denial all come back for this row
/// as [`AgentError`], while the component and runtime keep serving other tools.
pub struct WasmTool {
    name: &'static str,
    description: &'static str,
    schema: Value,
    runtime: Arc<WasmRuntime>,
    component: Component,
    /// the `tool` parameter of `executor.invoke` (the `name` of the wasm entry in the manifest).
    tool: String,
    /// pinned capability whitelist values (from the registry, not the plugin manifest).
    capabilities: Vec<ResolvedCapability>,
    timeout: Duration,
}

impl WasmTool {
    /// Build one tool table entry. `identity_key` = `plugin.{plugin}.{tool}`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity_key: &str,
        description: &str,
        parameters: &BTreeMap<String, ParamSpec>,
        runtime: Arc<WasmRuntime>,
        component: Component,
        tool: &str,
        capabilities: Vec<ResolvedCapability>,
        timeout: Duration,
    ) -> Self {
        Self {
            name: Box::leak(identity_key.to_string().into_boxed_str()),
            description: Box::leak(description.to_string().into_boxed_str()),
            schema: parameters_envelope(parameters),
            runtime,
            component,
            tool: tool.to_string(),
            capabilities,
            timeout,
        }
    }

    async fn invoke(&self, args_json: &str) -> Result<String, WasmError> {
        self.runtime
            .invoke(
                &self.component,
                &WasmCall {
                    tool: &self.tool,
                    args_json,
                    capabilities: &self.capabilities,
                    timeout: self.timeout,
                },
            )
            .await
    }
}

fn wasm_error_to_agent(name: &str, e: WasmError) -> AgentError {
    match e {
        WasmError::Timeout { .. } => AgentError::ToolTimeout,
        other => AgentError::ToolExecution {
            name: name.to_string(),
            source: Box::new(AgentError::Internal(other.to_string())),
        },
    }
}

#[async_trait]
impl Tool for WasmTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let json = self
            .invoke(&args.to_string())
            .await
            .map_err(|e| wasm_error_to_agent(self.name, e))?;
        Ok(vec![Content::text(json)])
    }

    fn timeout_ms(&self) -> Option<u64> {
        Some(self.timeout.as_millis() as u64)
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name.to_string(),
            description: self.description.to_string(),
            origin: "plugin-wasm".to_string(),
            version: "unknown".to_string(),
            requirements: vec![],
        }
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dispatcher.wasm")
    }

    fn runtime() -> WasmRuntime {
        WasmRuntime::new().expect("runtime")
    }

    async fn echo(rt: &WasmRuntime, component: &Component) -> String {
        rt.invoke(
            component,
            &WasmCall {
                tool: "echo",
                args_json: "{\"hello\":\"wasm\"}",
                capabilities: &[],
                timeout: Duration::from_secs(5),
            },
        )
        .await
        .expect("echo")
    }

    // ── Whitelist criterion (pure function, fast) ───────────────────────

    #[test]
    fn host_allowlist_is_fail_closed_exact_match() {
        let allow = vec!["api.example.com".to_string()];
        assert!(host_allowed(&allow, "api.example.com"));
        assert!(host_allowed(&allow, "API.Example.com"), "case normalized");
        assert!(!host_allowed(&allow, "evil.com"));
        assert!(!host_allowed(&allow, "example.com"), "no suffix wildcards");
        assert!(
            !host_allowed(&allow, "api.example.com.evil"),
            "no suffix wildcards"
        );
        assert!(
            !host_allowed(&[], "api.example.com"),
            "empty list denies all"
        );
        assert!(!host_allowed(&allow, ""), "empty host denied");
    }

    // ── net whitelist hook (the send_request surface) ────────────────────

    fn empty_body() -> WasiBody {
        use http_body_util::BodyExt;
        http_body_util::Empty::<bytes::Bytes>::new()
            .map_err(|never: std::convert::Infallible| match never {})
            .boxed_unsync()
    }

    fn ready_fut() -> Box<dyn std::future::Future<Output = Result<(), HttpError>> + Send> {
        Box::new(std::future::ready(Ok(())))
    }

    /// send_request returns `Box<dyn Future>` (the target is not Unpin, so it cannot be
    /// awaited directly): use `Box::into_pin` to get `Pin<Box<_>>` and await that. The err
    /// goes through Option (the Ok side is not Debug).
    async fn deny_reason(
        hooks: &mut HostFilterHooks,
        request: http::Request<WasiBody>,
    ) -> HttpError {
        Box::into_pin(hooks.send_request(request, None, ready_fut()))
            .await
            .err()
            .expect("expected rejection")
    }

    #[tokio::test]
    async fn net_whitelist_denies_unlisted_host_at_send_request() {
        let mut hooks = HostFilterHooks {
            allowed_hosts: vec!["api.example.com".to_string()],
        };
        let request = http::Request::builder()
            .uri("https://evil.example.net/x")
            .body(empty_body())
            .unwrap();
        let err = deny_reason(&mut hooks, request).await;
        assert!(
            matches!(err, HttpError::HttpRequestDenied),
            "deny must be HttpRequestDenied: {err:?}"
        );

        // An empty whitelist denies everything
        let mut hooks = HostFilterHooks {
            allowed_hosts: Vec::new(),
        };
        let request = http::Request::builder()
            .uri("https://api.example.com/x")
            .body(empty_body())
            .unwrap();
        let err = deny_reason(&mut hooks, request).await;
        assert!(matches!(err, HttpError::HttpRequestDenied), "{err:?}");

        // A request without a host is denied too
        let request = http::Request::builder()
            .uri("/relative-only")
            .body(empty_body())
            .unwrap();
        let err = deny_reason(&mut hooks, request).await;
        assert!(matches!(err, HttpError::InternalError(_)), "{err:?}");
    }

    #[tokio::test]
    async fn net_whitelist_passes_allowed_host_downstream() {
        // Allowed goes through the default outbound implementation: start a local server,
        // put 127.0.0.1 on the whitelist, and let the loopback go all the way through.
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/ok"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("fine"))
            .mount(&server)
            .await;

        let mut hooks = HostFilterHooks {
            allowed_hosts: vec!["127.0.0.1".to_string()],
        };
        let request = http::Request::builder()
            .uri(format!("http://127.0.0.1:{}/ok", server.address().port()))
            .body(empty_body())
            .unwrap();
        let (response, _io) = Box::into_pin(hooks.send_request(request, None, ready_fut()))
            .await
            .unwrap_or_else(|e| panic!("allowed host must reach the wire: {e:?}"));
        assert_eq!(response.status(), 200);
    }

    // ── Execution surface (real component; hangs tools, not the host) ────

    #[tokio::test]
    async fn echo_roundtrips_json() {
        let rt = runtime();
        let component = rt.compile(&fixture()).expect("compile fixture");
        assert_eq!(echo(&rt, &component).await, "{\"hello\":\"wasm\"}");
    }

    #[tokio::test]
    async fn fs_without_preopen_reads_nothing() {
        // Undeclared fs:read → no preopen → cannot read (tool errors, host lives).
        let rt = runtime();
        let component = rt.compile(&fixture()).expect("compile fixture");
        let err = rt
            .invoke(
                &component,
                &WasmCall {
                    tool: "read_file",
                    args_json: "/etc/passwd",
                    capabilities: &[],
                    timeout: Duration::from_secs(5),
                },
            )
            .await
            .expect_err("no preopen = no reads");
        assert!(matches!(err, WasmError::Guest(_)), "{err}");
        // The host keeps serving
        assert_eq!(echo(&rt, &component).await, "{\"hello\":\"wasm\"}");
    }

    #[tokio::test]
    async fn fs_preopen_grants_only_pinned_path() {
        let rt = runtime();
        let component = rt.compile(&fixture()).expect("compile fixture");

        let dir = tempfile::tempdir().unwrap();
        let pinned = dir.path().join("data");
        std::fs::create_dir(&pinned).unwrap();
        let file = pinned.join("ok.txt");
        std::fs::write(&file, "from-preopen").unwrap();

        let outside = dir.path().join("outside.txt");
        std::fs::write(&outside, "secret").unwrap();

        let caps = vec![ResolvedCapability::FsRead {
            paths: vec![pinned.to_string_lossy().into_owned()],
        }];
        let read = |path: std::path::PathBuf| {
            let caps = &caps;
            let rt = &rt;
            let component = &component;
            async move {
                let args = path.to_string_lossy().into_owned();
                rt.invoke(
                    component,
                    &WasmCall {
                        tool: "read_file",
                        args_json: &args,
                        capabilities: caps,
                        timeout: Duration::from_secs(5),
                    },
                )
                .await
            }
        };

        // Readable inside the pinned directory
        assert_eq!(read(file).await.expect("preopen read"), "from-preopen");
        // Cannot read outside the directory (the preopen boundary is the capability boundary)
        let err = read(outside).await.expect_err("outside preopen");
        assert!(matches!(err, WasmError::Guest(_)), "{err}");
    }

    #[tokio::test]
    async fn timeout_kills_call_not_host() {
        let rt = runtime();
        let component = rt.compile(&fixture()).expect("compile fixture");
        let err = rt
            .invoke(
                &component,
                &WasmCall {
                    tool: "spin",
                    args_json: "{}",
                    capabilities: &[],
                    timeout: Duration::from_millis(200),
                },
            )
            .await
            .expect_err("spin must time out");
        assert!(matches!(err, WasmError::Timeout { .. }), "{err}");
        // The tool hangs, not the host: the next call still works
        assert_eq!(echo(&rt, &component).await, "{\"hello\":\"wasm\"}");
    }

    #[tokio::test]
    async fn panic_traps_call_not_host() {
        let rt = runtime();
        let component = rt.compile(&fixture()).expect("compile fixture");
        let err = rt
            .invoke(
                &component,
                &WasmCall {
                    tool: "boom",
                    args_json: "{}",
                    capabilities: &[],
                    timeout: Duration::from_secs(5),
                },
            )
            .await
            .expect_err("boom must trap");
        assert!(matches!(err, WasmError::Trap(_)), "{err}");
        assert_eq!(echo(&rt, &component).await, "{\"hello\":\"wasm\"}");
    }

    #[tokio::test]
    async fn corrupt_wasm_fails_at_compile() {
        // Compile failure at load time → the error surface that disables this tool
        // (the loader does not skip the whole package).
        let rt = runtime();
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("bogus.wasm");
        std::fs::write(&bogus, b"not a wasm component").unwrap();
        let err = rt.compile(&bogus).expect_err("garbage must not compile");
        assert!(matches!(err, WasmError::Compile(_)), "{err}");
    }
}
