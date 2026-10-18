use super::*;
use agent_base::RunOutcome;

use crate::multi_agent::config::ControlConfig;

// ---------------------------------------------------------------------------
// Shared fixtures (used by more than one scenario module)
// ---------------------------------------------------------------------------

struct StreamingStub;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for StreamingStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(vec![
                Ok(agent_base::StreamChunk::Text("child ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatResponse {
            content: "child ok".to_string(),
            tool_calls: vec![],
            usage: agent_base::llm_trait::types::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

/// Stub provider that emits one tool call on its first turn and plain text
/// afterwards: drives a real child task through the full
/// "LLM → tool execution → LLM → completion" chain (the write-gate and
/// recycle tests depend on that timing). The tool-call chunk shape matches
/// the llm_engine aggregator (OpenAI delta form).
pub(crate) struct ToolCallOnceStub {
    pub tool: &'static str,
    pub arguments: &'static str,
    turns: std::sync::atomic::AtomicUsize,
}

impl ToolCallOnceStub {
    pub(crate) fn new(tool: &'static str, arguments: &'static str) -> Self {
        Self {
            tool,
            arguments,
            turns: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for ToolCallOnceStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let chunks: Vec<Result<agent_base::StreamChunk, agent_base::llm_trait::LlmError>> =
            if turn == 0 {
                vec![
                    Ok(agent_base::StreamChunk::ToolCall(serde_json::json!({
                        "delta": { "tool_calls": [
                            { "index": 0, "id": "call_1",
                              "function": { "name": self.tool, "arguments": self.arguments } }
                        ] }
                    }))),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("tool_calls".to_string()),
                    }),
                ]
            } else {
                vec![
                    Ok(agent_base::StreamChunk::Text("stub done".to_string())),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    }),
                ]
            };
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(chunks),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatResponse {
            content: "stub done".to_string(),
            tool_calls: vec![],
            usage: agent_base::llm_trait::types::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "tool-call-once-stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

fn make_ma_runtime_with(
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
) -> Arc<MultiAgentRuntime> {
    make_ma_runtime_with_tools(client, vec![])
}

fn make_ma_runtime_with_tools(
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
) -> Arc<MultiAgentRuntime> {
    Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        client,
        tools,
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

fn make_ma_runtime() -> Arc<MultiAgentRuntime> {
    make_ma_runtime_with(Arc::new(StreamingStub))
}

/// Business tools + per-deployment exclusion (used by the whitelist and
/// external-tool scenarios).
fn make_runtime_with_tools(
    business_tools: Vec<Arc<dyn Tool>>,
    excluded: Vec<String>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        child_excluded_tools: excluded,
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(StreamingStub),
        business_tools,
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

struct NoopReadFileTool;

#[async_trait::async_trait]
impl Tool for NoopReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a file's contents"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        })
    }

    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &agent_base::ToolContext,
    ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
        Ok(vec![agent_base::Content::text("contents")])
    }
}

/// Poll a teardown condition (cleanup runs on a worker thread, so the
/// assertions after `close_agent` / panic / `drop` are eventually-true).
async fn poll_until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("teardown did not complete within 2s: {what}");
}

/// Provider whose `stream` never returns — keeps the child inside
/// `run_turn` so parent Drop can only end it via JoinSet abort.
struct HangingLlm;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for HangingLlm {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        unreachable!("the child task is aborted long before this returns")
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        unreachable!("chat is not called by the child loop")
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "hanging".to_string(),
            model: "hanging".to_string(),
            version: None,
        }
    }
}

/// Provider whose stream completes after a short delay — keeps the child
/// inside `run_turn` long enough for a mid-task `close_agent`, then lets the
/// task finish (close is only honored between tasks, so the late result must
/// still be delivered).
struct DelayedStub;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for DelayedStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(vec![
                Ok(agent_base::StreamChunk::Text("late ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        Ok(agent_base::llm_trait::ChatResponse {
            content: "late ok".to_string(),
            tool_calls: vec![],
            usage: agent_base::llm_trait::types::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "delayed".to_string(),
            model: "delayed".to_string(),
            version: None,
        }
    }
}

// scenario modules (split from the former inline `mod tests`)
mod autonomy;
mod capability;
mod cleanup;
mod control;
mod denied;
mod external;
mod fork;
mod lifecycle;
mod outcome;
mod permission;
mod whitelist;
mod write_gate;
