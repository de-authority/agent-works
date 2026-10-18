//! MCP Server — expose an Agent as an MCP-compatible server.
//!
//! This module implements the **server side** of the Model Context Protocol,
//! allowing external orchestrators (LangGraph, CrewAI, custom tools) to delegate
//! tasks to a phi-agent over stdio or HTTP.
//!
//! ## Protocol
//!
//! The server exposes a single tool called `run`:
//!
//! ```text
//! tools/list → [{ name: "run", description: "Execute a task", inputSchema: {...} }]
//! tools/call { name: "run", arguments: { prompt: "..." } }
//!   → Agent runs full ReAct loop
//!   → Progress notifications for each step
//!   → Final result returned
//! ```
//!
//! ## Transports
//!
//! - **stdio** (subprocess mode): line-delimited JSON-RPC 2.0 on stdin/stdout
//! - **HTTP** (service mode): POST `/mcp` with SSE streaming progress
//!
//! ## Usage
//!
//! ```ignore
//! use agent_works::mcp::server::{McpServer, McpServerTransport};
//!
//! let server = McpServer::new(runtime, McpServerTransport::Stdio);
//! server.serve().await?;
//! ```

use std::sync::Arc;

use agent_base::{AgentResult, AgentRuntime, RunOutcome, RuntimeEvent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

// ── JSON-RPC 2.0 types ──

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

// ── MCP Server types ──

/// Transport mode for the MCP server.
#[derive(Debug, Clone)]
pub enum McpServerTransport {
    /// Line-delimited JSON on stdin/stdout (subprocess mode).
    Stdio,
    /// HTTP + SSE streaming.
    Http { host: String, port: u16 },
}

/// Configuration for running as an MCP server.
#[derive(Debug, Clone)]
pub struct McpServeConfig {
    /// Human-readable name shown in `tools/list`.
    pub name: String,
    /// Server version reported during `initialize`.
    pub version: String,
    /// Transport mode.
    pub transport: McpServerTransport,
}

impl Default for McpServeConfig {
    fn default() -> Self {
        Self {
            name: "phi-agent".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            transport: McpServerTransport::Stdio,
        }
    }
}

/// An MCP Server that wraps an [`AgentRuntime`] and exposes it as an MCP tool.
///
/// External orchestrators call `tools/list` to discover the `run` tool,
/// then `tools/call` with a `prompt` argument to execute tasks.
pub struct McpServer {
    runtime: AgentRuntime,
    config: McpServeConfig,
}

impl McpServer {
    /// Create a new MCP server wrapping the given runtime.
    pub fn new(runtime: AgentRuntime, config: McpServeConfig) -> Self {
        Self { runtime, config }
    }

    /// Start serving requests. This blocks until the transport closes.
    pub async fn serve(&self) -> AgentResult<()> {
        match &self.config.transport {
            McpServerTransport::Stdio => self.serve_stdio().await,
            McpServerTransport::Http { host, port } => self.serve_http(host, *port).await,
        }
    }

    // ── stdio transport ──

    async fn serve_stdio(&self) -> AgentResult<()> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let writer = Arc::new(Mutex::new(stdout));
        let mut lines = reader.lines();

        tracing::info!(name = %self.config.name, "MCP server listening on stdio");

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let err = JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {}", e),
                            data: None,
                        }),
                    };
                    Self::write_response(&writer, &err).await;
                    continue;
                }
            };

            let response = self.handle_request(&request, Some(writer.clone())).await;

            if let Some(resp) = response {
                Self::write_response(&writer, &resp).await;
            }
        }

        tracing::info!("MCP server stdin closed, shutting down");
        Ok(())
    }

    // ── HTTP transport ──

    async fn serve_http(&self, host: &str, port: u16) -> AgentResult<()> {
        use tokio::net::TcpListener;

        let addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(&addr).await.map_err(|e| {
            agent_base::AgentError::internal(format!("Failed to bind {}: {}", addr, e))
        })?;

        tracing::info!(addr = %addr, name = %self.config.name, "MCP server listening on HTTP");

        // For now, handle one connection at a time. A production server would
        // use hyper/axum for proper HTTP + SSE. This is a minimal implementation
        // suitable for local use.
        loop {
            let (mut socket, peer) = listener
                .accept()
                .await
                .map_err(|e| agent_base::AgentError::internal(format!("Accept error: {}", e)))?;

            tracing::debug!(peer = %peer, "MCP HTTP connection");

            let (reader, mut writer) = socket.split();
            let buf_reader = BufReader::new(reader);
            let mut lines = buf_reader.lines();

            // Simple line-delimited JSON over TCP (same as stdio but over socket)
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                let request: JsonRpcRequest = match serde_json::from_str(&line) {
                    Ok(req) => req,
                    Err(_) => continue,
                };

                let response = self.handle_request(&request, None).await;

                if let Some(resp) = response {
                    let mut json = serde_json::to_string(&resp).unwrap_or_default();
                    json.push('\n');
                    let _ = writer.write_all(json.as_bytes()).await;
                }
            }
        }
    }

    // ── Request handling ──

    async fn handle_request(
        &self,
        request: &JsonRpcRequest,
        progress_writer: Option<Arc<Mutex<tokio::io::Stdout>>>,
    ) -> Option<JsonRpcResponse> {
        match request.method.as_str() {
            "initialize" => Some(self.handle_initialize(request)),
            "initialized" => {
                // Client confirms initialization is complete. No response needed.
                tracing::info!("MCP client initialized");
                None
            }
            "tools/list" => Some(self.handle_tools_list(request)),
            "tools/call" => Some(self.handle_tools_call(request, progress_writer).await),
            "notifications/initialized" => None,
            _ => Some(JsonRpcResponse {
                jsonrpc: "2.0",
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            }),
        }
    }

    // ── initialize ──

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let result = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.config.name,
                "version": self.config.version,
            },
            "capabilities": {
                "tools": {},
            },
        });

        JsonRpcResponse {
            jsonrpc: "2.0",
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    // ── tools/list ──

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tool_def = serde_json::json!({
            "name": "run",
            "description": "Execute a task using the phi-agent AI agent. The agent will perform a full ReAct loop — thinking, calling internal tools, and iterating — to complete the task.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description or question for the agent to handle."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model override (e.g. 'opus', 'sonnet', 'gpt-4o')."
                    }
                },
                "required": ["prompt"]
            }
        });

        let result = serde_json::json!({
            "tools": [tool_def],
        });

        JsonRpcResponse {
            jsonrpc: "2.0",
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    // ── tools/call ──

    async fn handle_tools_call(
        &self,
        request: &JsonRpcRequest,
        progress_writer: Option<Arc<Mutex<tokio::io::Stdout>>>,
    ) -> JsonRpcResponse {
        let params = match request.params.as_ref().and_then(|p| p.get("arguments")) {
            Some(args) => args.clone(),
            None => {
                return JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: request.id.clone(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing arguments".to_string(),
                        data: None,
                    }),
                };
            }
        };

        // Validate the tool name — we only expose "run"
        let tool_name = request
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str);
        if tool_name != Some("run") {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: format!(
                        "Unknown tool: {:?}. The only available tool is \"run\".",
                        tool_name
                    ),
                    data: None,
                }),
            };
        }

        let prompt = params
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        if prompt.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Missing required parameter: prompt".to_string(),
                    data: None,
                }),
            };
        }

        // Create a session for this task
        let session_id = self.runtime.create_session().await;

        // Collect events into a result
        let thought_texts: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let tool_calls: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let final_text: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let thought_texts_clone = thought_texts.clone();
        let tool_calls_clone = tool_calls.clone();
        let final_text_clone = final_text.clone();
        let progress_writer_clone = progress_writer.clone();

        let prompt_clone = prompt.clone();
        let outcome = self
            .runtime
            .run_turn(session_id, &prompt_clone, move |event| {
                let thought_texts = thought_texts_clone.clone();
                let tool_calls = tool_calls_clone.clone();
                let final_text = final_text_clone.clone();
                let progress_writer = progress_writer_clone.clone();

                // Emit progress notifications
                let progress_msg = match &event {
                    RuntimeEvent::ThoughtDelta { text, .. } => {
                        thought_texts.lock().unwrap().push(text.clone());
                        Some(serde_json::json!({
                            "type": "thought",
                            "text": text,
                        }))
                    }
                    RuntimeEvent::ToolCallStarted { tool_name, .. } => {
                        tool_calls.lock().unwrap().push(tool_name.clone());
                        Some(serde_json::json!({
                            "type": "tool_call",
                            "name": tool_name,
                        }))
                    }
                    RuntimeEvent::ToolCallFinished {
                        tool_name, summary, ..
                    } => Some(serde_json::json!({
                        "type": "tool_result",
                        "name": tool_name,
                        "summary": summary,
                    })),
                    RuntimeEvent::TextDelta { text, .. } => {
                        final_text.lock().unwrap().push(text.clone());
                        Some(serde_json::json!({
                            "type": "text",
                            "text": text,
                        }))
                    }
                    RuntimeEvent::RunFinished { .. } => None, // handled below
                    _ => None,
                };

                // Write progress to the shared writer (same as the response writer),
                // avoiding interleaved output on stdout.
                if let (Some(progress), Some(writer)) = (progress_msg, &progress_writer) {
                    let notification = JsonRpcNotification {
                        jsonrpc: "2.0",
                        method: "notifications/progress".to_string(),
                        params: Some(serde_json::json!({
                            "progress": progress,
                        })),
                    };
                    if let Ok(json) = serde_json::to_string(&notification) {
                        let json_line = format!("{}\n", json);
                        // Use block_in_place to write async from sync context.
                        // This requires a multi-threaded tokio runtime (same
                        // requirement as AgentBuilder::build).
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let mut stdout = writer.lock().await;
                                let _ = stdout.write_all(json_line.as_bytes()).await;
                                let _ = stdout.flush().await;
                            })
                        });
                    }
                }

                Ok(())
            })
            .await;

        // Build the result
        let final_text = final_text.lock().unwrap();
        let thought_texts = thought_texts.lock().unwrap();
        let tool_calls = tool_calls.lock().unwrap();

        let result_text = match &outcome {
            Ok(RunOutcome::Completed) => final_text.join("\n"),
            Ok(RunOutcome::Continuing) => final_text.join("\n"),
            Ok(RunOutcome::Cancelled) => {
                if final_text.is_empty() {
                    "Task cancelled.".to_string()
                } else {
                    final_text.join("\n")
                }
            }
            Ok(RunOutcome::MaxTurnsExceeded { turns }) => {
                format!(
                    "Reached max turns ({}).\n\nPartial output:\n{}",
                    turns,
                    final_text.join("\n")
                )
            }
            Ok(RunOutcome::Failed { error }) => {
                format!(
                    "Error: {}\n\nPartial output:\n{}",
                    error,
                    final_text.join("\n")
                )
            }
            Err(e) => format!("Agent error: {}", e),
        };

        // Build tool list summary
        let tool_summary: Vec<String> = tool_calls.iter().map(|t| format!("- {}", t)).collect();

        let result_content = if tool_summary.is_empty() {
            format!("{}\n\n{}", result_text, thought_texts.join("\n"))
        } else {
            format!(
                "{}\n\nTool calls:\n{}\n\nThought process:\n{}",
                result_text,
                tool_summary.join("\n"),
                thought_texts.join("\n")
            )
        };

        let result = serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": result_content,
                }
            ],
        });

        JsonRpcResponse {
            jsonrpc: "2.0",
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    // ── Helpers ──

    async fn write_response(writer: &Arc<Mutex<tokio::io::Stdout>>, response: &JsonRpcResponse) {
        let mut json = serde_json::to_string(response).unwrap_or_default();
        json.push('\n');
        let mut stdout = writer.lock().await;
        let _ = stdout.write_all(json.as_bytes()).await;
        let _ = stdout.flush().await;
    }
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    use serde::Deserialize;
    use serde_json::Value;

    #[derive(Debug, Deserialize)]
    pub struct JsonRpcRequest {
        pub jsonrpc: String,
        #[serde(default)]
        pub id: Option<Value>,
        pub method: String,
        #[serde(default)]
        pub params: Option<Value>,
    }

    pub fn parse_json_rpc(json_str: &str) -> Result<JsonRpcRequest, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use agent_base::llm_trait::response::FinishReason;
    use agent_base::llm_trait::types::UsageInfo;
    use agent_base::llm_trait::{
        Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
    };
    use agent_base::{AgentBuilder, StreamChunk};

    struct StubProvider;

    #[async_trait::async_trait]
    impl LlmProvider for StubProvider {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::Text("hello".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]))))
        }

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: "hello".to_string(),
                tool_calls: vec![],
                usage: UsageInfo::default(),
                finish_reason: FinishReason::Stop,
                raw: None,
                reasoning_content: None,
                thinking_signature: None,
            })
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "stub".to_string(),
                model: "stub-model".to_string(),
                version: None,
            }
        }
    }

    fn runtime() -> AgentRuntime {
        AgentBuilder::new(Arc::new(StubProvider)).build().unwrap()
    }

    fn server() -> McpServer {
        McpServer::new(runtime(), McpServeConfig::default())
    }

    fn req(method: &str, params: Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: method.to_string(),
            params: Some(params),
        }
    }

    #[test]
    fn test_default_config() {
        let config = McpServeConfig::default();
        assert_eq!(config.name, "phi-agent");
        assert!(!config.version.is_empty());
    }

    #[test]
    fn test_json_rpc_request_parse() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Value::Number(1.into())));
    }

    #[test]
    fn test_json_rpc_response_serialize() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0",
            id: Some(Value::Number(1.into())),
            result: Some(serde_json::json!({"ok": true})),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"ok\":true"));
    }

    #[test]
    fn test_json_rpc_error_serialize() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0",
            id: None,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "Method not found".to_string(),
                data: None,
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Method not found"));
    }

    #[test]
    fn test_notification_serialize() {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0",
            method: "notifications/progress".to_string(),
            params: Some(serde_json::json!({
                "type": "thought",
                "text": "thinking..."
            })),
        };
        let json = serde_json::to_string(&notif).unwrap();
        assert!(json.contains("notifications/progress"));
        assert!(json.contains("thinking"));
        // Notification should not have id
        assert!(!json.contains("\"id\""));
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let resp = server().handle_initialize(&req("initialize", serde_json::json!({})));
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "phi-agent");
        assert!(!result["serverInfo"]["version"].as_str().unwrap().is_empty());
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let resp = server().handle_tools_list(&req("tools/list", serde_json::json!({})));
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "run");
        assert!(
            tools[0]["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("prompt"))
        );
    }

    #[tokio::test]
    async fn test_handle_request_unknown_method() {
        let resp = server()
            .handle_request(&req("foo/bar", serde_json::json!({})), None)
            .await
            .unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("Method not found"));
    }

    #[tokio::test]
    async fn test_handle_request_notifications_return_none() {
        let server = server();
        assert!(
            server
                .handle_request(&req("initialized", serde_json::json!({})), None)
                .await
                .is_none()
        );
        assert!(
            server
                .handle_request(
                    &req("notifications/initialized", serde_json::json!({})),
                    None
                )
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_handle_tools_call_missing_arguments() {
        let resp = server()
            .handle_tools_call(&req("tools/call", serde_json::json!({"name": "run"})), None)
            .await;
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_unknown_tool() {
        let resp = server()
            .handle_tools_call(
                &req(
                    "tools/call",
                    serde_json::json!({"name": "nope", "arguments": {"prompt": "x"}}),
                ),
                None,
            )
            .await;
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_missing_prompt() {
        let resp = server()
            .handle_tools_call(
                &req(
                    "tools/call",
                    serde_json::json!({"name": "run", "arguments": {}}),
                ),
                None,
            )
            .await;
        let err = resp.error.unwrap();
        assert!(err.message.contains("Missing required parameter: prompt"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_success() {
        let resp = server()
            .handle_tools_call(
                &req(
                    "tools/call",
                    serde_json::json!({"name": "run", "arguments": {"prompt": "hi"}}),
                ),
                None,
            )
            .await;
        assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
        let result = resp.result.unwrap();
        let content = result["content"][0]["text"].as_str().unwrap();
        assert!(content.contains("hello"), "content: {content}");
    }

    // ── tools/call validation (Phase 4.1 — tool name check) ──

    /// Validate the tool-name extraction logic used in `handle_tools_call`.
    /// We can't call the full handler without an AgentRuntime, so we test
    /// the params-parsing logic directly.
    #[test]
    fn test_tools_call_tool_name_validation() {
        // Valid: name = "run"
        let params = Some(serde_json::json!({
            "name": "run",
            "arguments": {"prompt": "do something"}
        }));
        let name = params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str);
        assert_eq!(name, Some("run"));

        // Invalid: unknown tool name
        let params = Some(serde_json::json!({
            "name": "nonexistent",
            "arguments": {}
        }));
        let name = params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str);
        assert_ne!(name, Some("run"));
        assert_eq!(name, Some("nonexistent"));

        // Missing name field entirely
        let params = Some(serde_json::json!({
            "arguments": {"prompt": "test"}
        }));
        let name = params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str);
        assert_ne!(name, Some("run"));
        assert_eq!(name, None);

        // Missing params entirely
        let name = None::<Value>
            .as_ref()
            .and_then(|p: &Value| p.get("name"))
            .and_then(Value::as_str);
        assert_eq!(name, None);

        // Empty name
        let params = Some(serde_json::json!({
            "name": "",
            "arguments": {}
        }));
        let name = params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str);
        assert_ne!(name, Some("run"));
    }
}
