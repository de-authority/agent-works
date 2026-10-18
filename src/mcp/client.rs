use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use super::types::{McpToolInfo, McpTransport};

struct StdioProcess {
    #[allow(dead_code)] // held for RAII: child process is killed on drop
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Spawn a stdio MCP server under the one contract that keeps it away from
/// the user's terminal:
///
/// - stdin/stdout are the transport (piped);
/// - stderr is **piped** — never `inherit()`. An inherited stderr paints
///   straight onto the TUI's alternate screen (npm deprecation warnings were
///   observed landing mid-frame) and ratatui's diffing never repaints those
///   cells;
/// - the child `setsid`s out of the controlling terminal, so it can neither
///   `open("/dev/tty")` nor `tcsetattr` the user's tty (a clobbered termios
///   silently disables keyboard input — the "stuck TUI" incident).
///
/// Returns the stderr handle so the caller can drain it into the log — the
/// original intent of `inherit()` (server errors stay visible) without the
/// screen damage.
fn spawn_stdio_process(
    command: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
) -> std::io::Result<(StdioProcess, ChildStderr)> {
    let mut cmd = Command::new(command);
    cmd.args(args)
        .envs(env) // inherit process env + per-entry overrides (server token ports)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt as _;
        // From here on the child is its own session with no controlling terminal.
        // pre_exec runs async-signal-safe syscalls only; if setsid fails we refuse
        // to spawn (fail-closed — never quietly run with a tty attached).
        cmd.as_std_mut().pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("no stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("no stderr"))?;
    Ok((
        StdioProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        },
        stderr,
    ))
}

/// Drain a MCP server's stderr into the log, one line per warning.
async fn drain_stderr(command: String, stderr: ChildStderr) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::warn!(server = %command, %line, "MCP server stderr");
    }
}

#[allow(clippy::large_enum_variant)] // Http carries reqwest::Client; Stdio carries a child process
enum TransportInner {
    Http {
        url: String,
        /// Extra request headers (token etc.; `${ENV:}` has already been resolved
        /// to literals by the product layer).
        headers: BTreeMap<String, String>,
        client: reqwest::Client,
    },
    Stdio {
        process: Mutex<StdioProcess>,
    },
}

pub struct McpClient {
    transport: TransportInner,
    request_id: AtomicU64,
    /// Streamable HTTP session ID: captured from the `Mcp-Session-Id` response
    /// header of `initialize` and replayed on every subsequent request. Required
    /// by the spec (without it the server answers -32000 “no valid session ID
    /// provided”, as implementations like Taobao Desktop do). HTTP-only.
    session: Mutex<Option<String>>,
    /// Protocol handshake (`initialize` + `notifications/initialized`) happens
    /// only once for the whole process. When the server invalidates the session,
    /// this flag is cleared along with [`Self::session`] and everything starts over.
    handshaken: Mutex<bool>,
}

impl McpClient {
    pub async fn new(transport: McpTransport) -> AgentResult<Self> {
        match transport {
            McpTransport::Http { url, headers } => Ok(Self {
                transport: TransportInner::Http {
                    url,
                    headers,
                    client: reqwest::Client::new(),
                },
                request_id: AtomicU64::new(1),
                session: Mutex::new(None),
                handshaken: Mutex::new(false),
            }),
            McpTransport::Stdio { command, args, env } => {
                let (process, stderr) = spawn_stdio_process(&command, &args, &env)
                    .map_err(|e| AgentError::internal(format!("spawn MCP process: {e}")))?;
                tokio::spawn(drain_stderr(command.clone(), stderr));

                Ok(Self {
                    transport: TransportInner::Stdio {
                        process: Mutex::new(process),
                    },
                    request_id: AtomicU64::new(1),
                    session: Mutex::new(None),
                    handshaken: Mutex::new(false),
                })
            }
        }
    }

    /// MCP protocol handshake: `initialize` → record `Mcp-Session-Id` →
    /// `notifications/initialized`. Done only once per process (flag); on failure
    /// the flag is not set, so the next call retries.
    ///
    /// The spec-mandated first step — before any business request. Consequences
    /// of skipping it: lenient servers silently tolerate it, spec-compliant
    /// servers (Taobao Desktop MCP etc.) reject `tools/list` outright
    /// (-32000 "no valid session ID provided").
    async fn ensure_init(&self) -> AgentResult<()> {
        let mut handshaken = self.handshaken.lock().await;
        if *handshaken {
            return Ok(());
        }
        self.send_message(
            &json!({
                "jsonrpc": "2.0",
                "id": self.request_id.fetch_add(1, Ordering::SeqCst),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "phi-agent",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }),
            true,
        )
        .await?;
        self.send_message(
            &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            false,
        )
        .await?;
        *handshaken = true;
        Ok(())
    }

    /// Session invalidated (server restarted / expired): clear the session ID and
    /// the handshake flag together, and redo the whole flow on the next call.
    async fn forget_session(&self) {
        *self.session.lock().await = None;
        *self.handshaken.lock().await = false;
    }

    async fn send_request(&self, method: &str, params: Value) -> AgentResult<Value> {
        self.ensure_init().await?;
        match self.send_rpc(method, &params).await {
            // The server invalidated the session (bridge restart / expiry):
            // redo the full handshake once and retry.
            // **Only one retry** — tools/call is not idempotent, no endless retries.
            Err(e)
                if is_session_rejected(&e)
                    && matches!(self.transport, TransportInner::Http { .. }) =>
            {
                self.forget_session().await;
                self.ensure_init().await?;
                self.send_rpc(method, &params).await
            }
            other => other,
        }
    }

    /// One business request (fresh id + response).
    async fn send_rpc(&self, method: &str, params: &Value) -> AgentResult<Value> {
        self.send_message(
            &json!({
                "jsonrpc": "2.0",
                "id": self.request_id.fetch_add(1, Ordering::SeqCst),
                "method": method,
                "params": params,
            }),
            true,
        )
        .await
    }

    /// One round trip of a JSON-RPC envelope. `expect_response` = false means a
    /// notification (one-way, no response read — servers reply 202/empty body to
    /// notifications).
    async fn send_message(&self, body: &Value, expect_response: bool) -> AgentResult<Value> {
        match &self.transport {
            TransportInner::Http {
                url,
                headers,
                client,
            } => {
                self.send_http(url, headers, client, body, expect_response)
                    .await
            }
            TransportInner::Stdio { process } => {
                self.send_stdio(process, body, expect_response).await
            }
        }
    }

    async fn send_http(
        &self,
        url: &str,
        headers: &BTreeMap<String, String>,
        client: &reqwest::Client,
        body: &Value,
        expect_response: bool,
    ) -> AgentResult<Value> {
        let mut request = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        // Session replay: attach the `Mcp-Session-Id` obtained from the initialize
        // response header to every subsequent request.
        if let Some(sid) = self.session.lock().await.as_deref() {
            request = request.header("Mcp-Session-Id", sid);
        }

        let response = request
            .json(body)
            .send()
            .await
            .map_err(|e| transport_error(url, e))?;

        // The server may send the session header on any response; it must on
        // initialize (spec).
        if let Some(sid) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            *self.session.lock().await = Some(sid.to_string());
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = response
            .text()
            .await
            .map_err(|e| AgentError::json(format!("MCP response read: {e}")))?;

        if !expect_response {
            return Ok(Value::Null);
        }

        let res = parse_jsonrpc_body(&content_type, &text).map_err(|e| {
            AgentError::json(format!(
                "MCP response parse: {e} (HTTP {status}, content-type {content_type})"
            ))
        })?;
        if let Some(error) = res.get("error") {
            return Err(AgentError::internal(format!("MCP error: {error}")));
        }
        if !status.is_success() {
            return Err(AgentError::internal(format!(
                "MCP HTTP {status}: {}",
                text.chars().take(200).collect::<String>()
            )));
        }
        Ok(res.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn send_stdio(
        &self,
        process: &Mutex<StdioProcess>,
        body: &Value,
        expect_response: bool,
    ) -> AgentResult<Value> {
        let mut request_line =
            serde_json::to_string(body).map_err(|e| AgentError::json(format!("serialize: {e}")))?;
        request_line.push('\n');

        let response_line = {
            let mut proc = process.lock().await;
            proc.stdin
                .write_all(request_line.as_bytes())
                .await
                .map_err(|e| AgentError::internal(format!("stdio write: {e}")))?;
            if !expect_response {
                // Notification: do not read a response (the server does not answer).
                return Ok(Value::Null);
            }
            let mut line = String::new();
            proc.stdout
                .read_line(&mut line)
                .await
                .map_err(|e| AgentError::internal(format!("stdio read: {e}")))?;
            line
        };

        let res: Value = serde_json::from_str(&response_line)
            .map_err(|e| AgentError::json(format!("parse: {e}")))?;

        if let Some(error) = res.get("error") {
            return Err(AgentError::internal(format!("MCP error: {error}")));
        }

        Ok(res.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn list_tools(&self) -> AgentResult<Vec<McpToolInfo>> {
        let result = self.send_request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| AgentError::internal("MCP: invalid tools/list response"))?;

        let mut infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            infos.push(McpToolInfo {
                name,
                description,
                input_schema,
            });
        }
        Ok(infos)
    }

    pub async fn call_tool(&self, tool_name: &str, arguments: &Value) -> AgentResult<Value> {
        self.send_request(
            "tools/call",
            json!({
                "name": tool_name,
                "arguments": arguments,
            }),
        )
        .await
    }
}

/// Response body → JSON-RPC envelope. Accepts both shapes: plain
/// `application/json`, and Streamable HTTP `text/event-stream` (`data:` frames —
/// servers like Taobao Desktop wrap successful responses in SSE).
fn parse_jsonrpc_body(content_type: &str, body: &str) -> Result<Value, String> {
    let looks_sse = content_type.contains("text/event-stream")
        || body.trim_start().starts_with("event:")
        || body.lines().any(|l| l.starts_with("data:"));
    if !looks_sse {
        return serde_json::from_str(body).map_err(|e| e.to_string());
    }
    // SSE: multi-frame messages separated by blank lines; each frame's `data:`
    // lines are concatenated back into one JSON.
    let mut payload = String::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if !payload.is_empty() {
                payload.push('\n');
            }
            payload.push_str(data.trim_start());
            continue;
        }
        if line.is_empty() && !payload.is_empty() {
            let parsed = sse_payload_value(&payload);
            payload.clear();
            if let Some(value) = parsed {
                return Ok(value);
            }
        }
    }
    if let Some(value) = sse_payload_value(&payload) {
        return Ok(value);
    }
    Err("no JSON-RPC payload in SSE stream".to_string())
}

/// Reassembled `data:` payload → JSON-RPC envelope (must be a JSON object;
/// empty/non-object doesn't count).
fn sse_payload_value(payload: &str) -> Option<Value> {
    serde_json::from_str::<Value>(payload)
        .ok()
        .filter(Value::is_object)
}

/// A decent error for transport failures: say plainly "the server is gone /
/// timed out", don't just dump the raw reqwest text
/// (UX checklist ⑥ "server unreachable" error surface — the model relays this
/// sentence to the user).
fn transport_error(url: &str, e: reqwest::Error) -> AgentError {
    let timed_out = e.is_timeout();
    let detail = e.without_url();
    if timed_out {
        AgentError::internal(format!(
            "MCP server at {url} timed out — it is up but did not answer ({detail})"
        ))
    } else {
        AgentError::internal(format!(
            "MCP server at {url} is unreachable — the app that provides it may have quit or its bridge died ({detail})"
        ))
    }
}

/// The server invalidated the session (-32000 invalid session ID / the spec
/// answers HTTP 404 for expired sessions). On a match, redo the full handshake
/// once (see the recovery path in [`McpClient::send_request`]).
fn is_session_rejected(err: &AgentError) -> bool {
    let msg = err.to_string();
    // The JSON-RPC error object is embedded in the message verbatim:
    // `MCP error: {"code":-32000,...}`.
    msg.contains("\"code\":-32000") || msg.contains("MCP HTTP 404")
}

/// Wraps an MCP server tool as an agent-base `Tool`.
///
/// Tool names are prefixed with `mcp.{server_name}.` to prevent collisions
/// between tools from different MCP servers.
pub struct McpToolAdapter {
    /// Prefixed name exposed to the LLM: `mcp.{server}.{tool}`
    name: &'static str,
    /// Original tool name used when calling the MCP server
    original_name: String,
    description: &'static str,
    input_schema: Value,
    mcp_client: Arc<McpClient>,
}

impl McpToolAdapter {
    pub fn new(info: McpToolInfo, mcp_client: Arc<McpClient>, server_name: &str) -> Self {
        let canonical = format!("mcp.{}.{}", server_name, info.name);
        Self::with_identity_key(&canonical, info, mcp_client)
    }

    /// Constructor with an explicit identity key (MCP servers owned by a plugin
    /// use `plugin.{plugin}.{tool}`; keys are assigned at the product layer in
    /// [`crate::plugin::loader::assign_mcp_tool_keys`]).
    pub fn with_identity_key(
        identity_key: &str,
        info: McpToolInfo,
        mcp_client: Arc<McpClient>,
    ) -> Self {
        let original_name = info.name;
        let static_name: &'static str = Box::leak(identity_key.to_string().into_boxed_str());
        let static_description: &'static str = Box::leak(info.description.into_boxed_str());
        Self {
            name: static_name,
            original_name,
            description: static_description,
            input_schema: info.input_schema,
            mcp_client,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let result = self.mcp_client.call_tool(&self.original_name, args).await?;
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| result.to_string());

        Ok(vec![Content::text(content)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn tool_info() -> McpToolInfo {
        McpToolInfo {
            name: "search".to_string(),
            description: "search the web".to_string(),
            input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }
    }

    async fn http_client(server: &MockServer) -> McpClient {
        McpClient::new(McpTransport::http(format!("{}/mcp", server.uri())))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_new_http_transport() {
        let server = MockServer::start().await;
        let client = http_client(&server).await;
        assert!(matches!(client.transport, TransportInner::Http { .. }));
    }

    #[tokio::test]
    async fn test_new_stdio_spawn_error() {
        let err = McpClient::new(McpTransport::stdio(
            "definitely-not-a-real-command-xyz",
            Vec::<String>::new(),
        ))
        .await
        .err()
        .unwrap();
        assert!(format!("{err}").contains("spawn MCP process"));
    }

    /// Regression (BUG 7): a stdio server's stderr must come back to us, not be
    /// inherited into the user's terminal. Observed in practice: npm deprecation
    /// warnings used to land as whole lines in the alt-screen.
    #[tokio::test]
    async fn test_stdio_child_stderr_is_piped_not_inherited() {
        let (mut process, stderr) = spawn_stdio_process(
            "sh",
            &["-c".to_string(), "echo NOISE-XYZZY >&2".to_string()],
            &BTreeMap::new(),
        )
        .unwrap();
        let mut lines = BufReader::new(stderr).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        assert_eq!(line, "NOISE-XYZZY");
        let _ = process.child.wait().await;
    }

    /// Regression (BUG 7): the child must setsid away from the controlling
    /// terminal, otherwise it can open /dev/tty and tcsetattr the user's tty
    /// (setting termios back to cooked → keyboard silently dies). Criterion:
    /// after setsid the child is the session leader, sid == pid.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_stdio_child_leaves_the_controlling_terminal() {
        let (mut process, _stderr) =
            spawn_stdio_process("sleep", &["5".to_string()], &BTreeMap::new()).unwrap();
        let pid = process.child.id().unwrap();
        // After setsid the child is its own session: from our side getsid(child)
        // returns either EPERM (different session, -1) or the child's own
        // sid == pid. Without setsid we would get this process's session id,
        // satisfying neither — the criterion would fail.
        let sid = unsafe { libc::getsid(pid as i32) };
        assert!(
            sid == -1 || sid == pid as i32,
            "child is still in the parent's session (sid={sid}, pid={pid})"
        );
        let _ = process.child.start_kill();
        let _ = process.child.wait().await;
    }

    #[tokio::test]
    async fn test_send_request_http_returns_result() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"tools": [{"name": "run"}]}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let res = client.send_request("tools/list", json!({})).await.unwrap();
        assert_eq!(res["tools"][0]["name"], "run");
    }

    #[tokio::test]
    async fn test_send_request_http_error_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "error": {"code": -32601, "message": "Method not found"}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let err = client.send_request("nope", json!({})).await.unwrap_err();
        assert!(format!("{err}").contains("MCP error"));
    }

    #[tokio::test]
    async fn test_send_request_http_network_error() {
        // No mock mounted + pointing at an unbound port should fail at send().
        // The error surface must be human: say clearly "the server is gone",
        // don't just dump the raw reqwest text (UX⑥).
        let client = McpClient::new(McpTransport::http("http://127.0.0.1:1/mcp"))
            .await
            .unwrap();
        let err = client
            .send_request("tools/list", json!({}))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unreachable"), "{msg}");
        assert!(msg.contains("http://127.0.0.1:1/mcp"), "{msg}");
    }

    #[tokio::test]
    async fn test_send_request_http_sends_headers() {
        // headers port: Authorization must appear on the request (wiremock 404s
        // if the header matcher doesn't hit).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(wiremock::matchers::header("authorization", "Bearer tok-9"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": {"ok": true}
            })))
            .mount(&server)
            .await;

        let client = McpClient::new(
            McpTransport::http(format!("{}/mcp", server.uri()))
                .with_header("Authorization", "Bearer tok-9"),
        )
        .await
        .unwrap();
        let res = client.send_request("tools/list", json!({})).await.unwrap();
        assert_eq!(res["ok"], true);
    }

    /// Replays Taobao Desktop's failure mode: `initialize` issues a session ID,
    /// then subsequent requests that do **not** carry `Mcp-Session-Id` get
    /// -32000 “no valid session ID provided”. The client must handshake and
    /// replay it.
    struct SessionEchoServer {
        initialized: std::sync::atomic::AtomicU64,
    }

    impl wiremock::Respond for SessionEchoServer {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body = std::str::from_utf8(&request.body).unwrap_or("");
            if body.contains("\"initialize\"") {
                self.initialized
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                return ResponseTemplate::new(200)
                    .insert_header("Mcp-Session-Id", "sess-77")
                    .set_body_json(json!({
                        "jsonrpc": "2.0", "id": 1,
                        "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
                    }));
            }
            if body.contains("notifications/initialized") {
                return ResponseTemplate::new(202);
            }
            let has_session = request
                .headers
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
                == Some("sess-77");
            if has_session {
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {"tools": [{"name": "search_products", "description": "d", "inputSchema": {}}]}
                }))
            } else {
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "error": {"code": -32000, "message": "错误请求：未提供有效的会话ID"}
                }))
            }
        }
    }

    #[tokio::test]
    async fn test_handshake_captures_session_id_and_replays_it() {
        let server = MockServer::start().await;
        let responder = SessionEchoServer {
            initialized: std::sync::atomic::AtomicU64::new(0),
        };
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(responder)
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        // Handshake (initialize + notification) happens automatically;
        // tools/list carries the session header → tools are returned.
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search_products");
    }

    #[tokio::test]
    async fn test_sse_response_is_parsed() {
        // A Streamable HTTP success response may be `text/event-stream`
        // (Taobao Desktop does exactly this).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[{\"name\":\"buy_now\",\"description\":\"d\",\"inputSchema\":{}}]}}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "buy_now");
    }

    #[tokio::test]
    async fn test_handshake_happens_once_per_client() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let server = MockServer::start().await;
        let counter = Arc::new(AtomicU64::new(0));
        struct CountInit(Arc<AtomicU64>);
        impl wiremock::Respond for CountInit {
            fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
                let body = std::str::from_utf8(&request.body).unwrap_or("");
                if body.contains("\"initialize\"") {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 1, "result": {"tools": []}
                }))
            }
        }
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(CountInit(Arc::clone(&counter)))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let _ = client.list_tools().await.unwrap();
        let _ = client.list_tools().await.unwrap();
        let _ = client.call_tool("t", &json!({})).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1, "initialize exactly once");
    }

    /// Bridge-restart scenario: after the first business request `sess-1` is
    /// invalidated (-32000); only a re-handshake yielding `sess-2` is accepted.
    /// The client must automatically re-handshake and successfully replay the
    /// original call, and **only once**.
    #[tokio::test]
    async fn test_invalid_session_rehandshakes_and_retries_once() {
        use std::sync::atomic::{AtomicU64, Ordering};
        struct RestartingBridge {
            inits: Arc<AtomicU64>,
            business: Arc<AtomicU64>,
        }
        impl wiremock::Respond for RestartingBridge {
            fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
                let body = std::str::from_utf8(&request.body).unwrap_or("");
                if body.contains("\"initialize\"") {
                    let n = self.inits.fetch_add(1, Ordering::SeqCst) + 1;
                    return ResponseTemplate::new(200)
                        .insert_header("Mcp-Session-Id", format!("sess-{n}"))
                        .set_body_json(json!({
                            "jsonrpc": "2.0", "id": 1,
                            "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
                        }));
                }
                if body.contains("notifications/initialized") {
                    return ResponseTemplate::new(202);
                }
                let sid = request
                    .headers
                    .get("mcp-session-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                let nth = self.business.fetch_add(1, Ordering::SeqCst);
                // The old session is invalidated after the first business request;
                // the new session (from the re-handshake) is accepted forever.
                if sid == "sess-1" && nth >= 1 {
                    return ResponseTemplate::new(200).set_body_json(json!({
                        "jsonrpc": "2.0", "id": 2,
                        "error": {"code": -32000, "message": "错误请求：未提供有效的会话ID"}
                    }));
                }
                ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {"tools": [{"name": "search_products", "description": "d", "inputSchema": {}}]}
                }))
            }
        }
        let server = MockServer::start().await;
        let inits = Arc::new(AtomicU64::new(0));
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(RestartingBridge {
                inits: Arc::clone(&inits),
                business: Arc::new(AtomicU64::new(0)),
            })
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        // First call: handshake + normal response.
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        // Second call: sess-1 is now invalid → automatic re-handshake (sess-2)
        // → replay succeeds.
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        // Afterwards sess-2 is used directly, no further handshake
        // (the recovery happens exactly once).
        let _ = client
            .call_tool("search_products", &json!({}))
            .await
            .unwrap();
        assert_eq!(inits.load(Ordering::SeqCst), 2, "exactly one re-handshake");
    }

    #[tokio::test]
    async fn test_send_request_stdio_returns_result() {
        // Multi-turn shell: only reply to requests that carry an id
        // (notifications get no reply — the handshake includes one notification).
        let client = McpClient::new(McpTransport::stdio(
            "sh",
            [
                "-c",
                r#"while read line; do case "$line" in *'"id"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}';; esac; done"#,
            ],
        ))
        .await
        .unwrap();

        let res = client.send_request("tools/list", json!({})).await.unwrap();
        assert!(res["tools"].is_array());
    }

    #[cfg(unix)]
    // file-write probe goes through sh; Windows semantics are covered by a separate spawn-env test (see docs)
    #[tokio::test]
    async fn test_stdio_env_override_reaches_child() {
        // env port: takes effect at spawn (no protocol round trip needed) — the
        // child writes $MCP_TEST_TOKEN to a file.
        let dir = tempfile::tempdir().unwrap();
        let outfile = dir.path().join("env.txt");
        let script = format!("printf '%s' \"$MCP_TEST_TOKEN\" > {}", outfile.display());
        let _client = McpClient::new(
            McpTransport::stdio("sh", ["-c", &script]).with_env("MCP_TEST_TOKEN", "tok-from-env"),
        )
        .await
        .unwrap();

        let mut content = String::new();
        for _ in 0..100 {
            if let Ok(text) = std::fs::read_to_string(&outfile) {
                content = text;
                if !content.is_empty() {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(content, "tok-from-env");
    }

    #[tokio::test]
    async fn test_list_tools_parses_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {
                    "tools": [
                        {"name": "search", "description": "search the web",
                         "inputSchema": {"type": "object"}},
                        {"name": "run", "description": ""}
                    ]
                }
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[0].description, "search the web");
        // Missing inputSchema falls back to {"type": "object"}
        assert_eq!(tools[1].input_schema, json!({"type": "object"}));
    }

    #[tokio::test]
    async fn test_list_tools_invalid_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": {"not_tools": []}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let err = client.list_tools().await.unwrap_err();
        assert!(format!("{err}").contains("invalid tools/list response"));
    }

    #[tokio::test]
    async fn test_call_tool_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text", "text": "done"}]}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let result = client
            .call_tool("search", &json!({"q": "rust"}))
            .await
            .unwrap();
        assert_eq!(result["content"][0]["text"], "done");
    }

    #[tokio::test]
    async fn test_adapter_new_and_metadata() {
        let server = MockServer::start().await;
        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");

        assert_eq!(adapter.name(), "mcp.brave.search");
        assert_eq!(adapter.description(), "search the web");
        assert_eq!(adapter.schema()["properties"]["q"]["type"], "string");
    }

    #[tokio::test]
    async fn test_adapter_call_returns_text_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text", "text": "alpha"},
                                       {"type": "text", "text": "beta"}]}
            })))
            .mount(&server)
            .await;

        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");
        let out = adapter
            .call(&json!({"q": "rust"}), &ToolContext::for_test())
            .await
            .unwrap();
        // Text items are joined by newline.
        assert_eq!(content_text(&out), "alpha\nbeta");
    }

    #[tokio::test]
    async fn test_adapter_call_without_content_falls_back_to_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"plain": "no content field"}
            })))
            .mount(&server)
            .await;

        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");
        let out = adapter
            .call(&json!({"q": "rust"}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&out).contains("no content field"));
    }
}
