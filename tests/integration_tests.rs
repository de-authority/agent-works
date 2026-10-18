use std::sync::Arc;
use std::sync::Mutex;

use agent_base::llm_trait::response::FinishReason as LlmFinishReason;
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
};
use agent_base::{AgentResult, Content, RunOutcome, StreamChunk, Tool, ToolContext};
use agent_works::AgentBuilder;
use async_trait::async_trait;
use serde_json::{Value, json};

struct MockLlmClient {
    responses: Mutex<std::vec::IntoIter<Vec<StreamChunk>>>,
    call_count: Mutex<usize>,
}

impl MockLlmClient {
    fn new(scripted_responses: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            responses: Mutex::new(scripted_responses.into_iter()),
            call_count: Mutex::new(0),
        }
    }

    fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl LlmProvider for MockLlmClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        *self.call_count.lock().unwrap() += 1;
        let chunks: Vec<Result<StreamChunk, LlmError>> = self
            .responses
            .lock()
            .unwrap()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(Ok)
            .collect();
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: "{\"done\":true,\"reason\":\"task complete\"}".to_string(),
            tool_calls: vec![],
            usage: agent_base::UsageInfo::default(),
            finish_reason: LlmFinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: None,
            max_output_tokens: None,
        }
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "echo back the message"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let msg = args["message"].as_str().unwrap_or("");
        Ok(vec![Content::text(format!("echo: {msg}"))])
    }
}

// ---------------------------------------------------------------------------
// Builder forwarding tests (without skill feature)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_builder_forwarding_text_reply() {
    let mock = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("Hello, world!".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));
    let llm: Arc<dyn LlmProvider> = mock.clone();

    let runtime = AgentBuilder::new(llm)
        .system_prompt("You are a helpful assistant")
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id.clone(), "Hi").await;
    assert!(result.is_ok(), "Expected ok, got: {result:?}");
    let (_events, outcome) = result.unwrap();
    assert_eq!(outcome, RunOutcome::Completed);
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn test_builder_forwarding_with_tool() {
    let mock = Arc::new(MockLlmClient::new(vec![
        vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"message\": \"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Done!".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ]));
    let llm: Arc<dyn LlmProvider> = mock.clone();

    let runtime = AgentBuilder::new(llm)
        .register_tool(EchoTool)
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "Echo hello").await;
    assert!(result.is_ok(), "Expected ok, got: {result:?}");
    assert_eq!(mock.call_count(), 2);
}

#[tokio::test]
async fn test_builder_forwarding_middleware() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let triggered = Arc::new(AtomicBool::new(false));

    struct FlagMiddleware {
        flag: Arc<AtomicBool>,
    }

    #[async_trait]
    impl agent_base::Middleware for FlagMiddleware {
        async fn on_post_llm(&self, _ctx: &mut agent_base::PostLlmCtx) -> AgentResult<()> {
            self.flag.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("reply".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let runtime = AgentBuilder::new(llm)
        .system_prompt("sys")
        .middleware(FlagMiddleware {
            flag: triggered.clone(),
        })
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "test").await;
    assert!(result.is_ok());
    assert!(
        triggered.load(Ordering::SeqCst),
        "Middleware should be triggered"
    );
}

// ---------------------------------------------------------------------------
// Builder forwarding - error recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_builder_forwarding_error_recovery() {
    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::ToolCall(json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "echo",
                        "arguments": "{\"message\": \"test\"}"
                    }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let runtime = AgentBuilder::new(llm)
        .register_tool(EchoTool)
        .tool_timeout(30_000)
        .max_tool_output_chars(4096)
        .language(agent_base::Language::Zh)
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "test").await;
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// ToolEnforcementMiddleware tests (in agent-base, re-exported by agent-works)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_tool_enforcement_available_via_works() {
    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("I will do it...".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let config = agent_base::ToolEnforcementConfig::default();
    let runtime = AgentBuilder::new(llm)
        .register_tool(EchoTool)
        .system_prompt("sys")
        .middleware(agent_base::ToolEnforcementMiddleware::new(config))
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "do something").await;
    assert!(result.is_ok(), "Expected ok: {result:?}");
}

// ---------------------------------------------------------------------------
// Guard integration tests (Phase 3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_guard_default_noop_text_reply() {
    // Default (NoopGuard): text reply → Completed
    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("Hello!".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let runtime = AgentBuilder::new(llm)
        .system_prompt("You are helpful")
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "Hi").await;
    assert!(result.is_ok(), "Expected ok: {result:?}");
    let (_events, outcome) = result.unwrap();
    assert_eq!(outcome, RunOutcome::Completed);
}

#[tokio::test]
async fn test_guard_default_noop_with_tool() {
    // Default (NoopGuard): tool call + text → Completed
    let llm = Arc::new(MockLlmClient::new(vec![
        vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"message\": \"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Done!".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ]));

    let runtime = AgentBuilder::new(llm)
        .register_tool(EchoTool)
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "Echo hello").await;
    assert!(result.is_ok(), "Expected ok: {result:?}");
    let (_events, outcome) = result.unwrap();
    assert_eq!(outcome, RunOutcome::Completed);
}

#[tokio::test]
async fn test_guard_explicit_default_guard() {
    use agent_works::guard::{DefaultGuard, DefaultGuardConfig};

    // Explicitly set DefaultGuard — text reply should Complete
    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("response".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let runtime = AgentBuilder::new(llm).guard(guard).build().unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "test").await;
    assert!(result.is_ok(), "Expected ok: {result:?}");
    let (_events, outcome) = result.unwrap();
    assert_eq!(outcome, RunOutcome::Completed);
}

#[tokio::test]
async fn test_guard_explicit_noop_guard() {
    use agent_works::guard::NoopGuard;

    // Explicitly set NoopGuard
    let llm = Arc::new(MockLlmClient::new(vec![vec![
        StreamChunk::Text("reply".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));

    let runtime = AgentBuilder::new(llm).guard(NoopGuard).build().unwrap();

    let session_id = runtime.create_session().await;
    let result = runtime.run_turn_collect(session_id, "test").await;
    assert!(result.is_ok(), "Expected ok: {result:?}");
    let (_events, outcome) = result.unwrap();
    assert_eq!(outcome, RunOutcome::Completed);
}

// ---------------------------------------------------------------------------
// Skill feature tests
// ---------------------------------------------------------------------------

#[cfg(feature = "skill")]
mod skill_tests {
    use super::*;
    use agent_base::RuntimeEvent;
    use agent_works::skill::{LazySkillPrompter, Skill, SkillPrompter};
    use serde_json::Value;
    use std::sync::Arc;

    struct AddTool;

    #[async_trait]
    impl Tool for AddTool {
        fn name(&self) -> &'static str {
            "add"
        }

        fn description(&self) -> &'static str {
            "Calculate the sum of two integers"
        }

        fn schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer", "description": "First addend" },
                    "b": { "type": "integer", "description": "Second addend" }
                },
                "required": ["a", "b"]
            })
        }

        async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(vec![Content::text(format!("{a} + {b} = {}", a + b))])
        }
    }

    struct MathSkill;

    impl Skill for MathSkill {
        fn name(&self) -> &'static str {
            "math"
        }

        fn brief_description(&self) -> String {
            "Math: supports addition".to_string()
        }

        fn detailed_description(&self) -> String {
            "## Math Skill\n\n- **add**: Calculate the sum of two integers".to_string()
        }

        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![Arc::new(AddTool)]
        }
    }

    #[tokio::test]
    async fn test_register_skill_with_builder() {
        let llm = Arc::new(super::MockLlmClient::new(vec![vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "add",
                            "arguments": "{\"a\": 1, \"b\": 2}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let runtime = AgentBuilder::new(llm)
            .system_prompt("You are a math assistant")
            .register_skill(MathSkill)
            .build()
            .unwrap();

        let session_id = runtime.create_session().await;
        let result = runtime.run_turn_collect(session_id, "1+2=?").await;
        assert!(result.is_ok(), "Expected ok, got: {result:?}");

        let (events, _outcome) = result.unwrap();
        let tool_done = events.iter().any(
            |e| matches!(e, RuntimeEvent::ToolCallFinished { tool_name, .. } if tool_name == "add"),
        );
        assert!(tool_done, "add tool should be called");
    }

    #[tokio::test]
    async fn test_skill_disable_prompt_injection() {
        let llm = Arc::new(super::MockLlmClient::new(vec![vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "add",
                            "arguments": "{\"a\": 3, \"b\": 4}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let runtime = AgentBuilder::new(llm)
            .system_prompt("My custom prompt")
            .register_skill(MathSkill)
            .disable_skill_prompt_injection()
            .build()
            .unwrap();

        let session_id = runtime.create_session().await;
        let result = runtime.run_turn_collect(session_id, "3+4=?").await;
        assert!(result.is_ok(), "Expected ok, got: {result:?}");
    }

    #[tokio::test]
    #[ignore = "SkillDetailTool lives in phi-kernel-tools; this test needs a factory set via with_skill_detail_tool_factory"]
    async fn test_skill_custom_detail_tool_name() {
        let llm = Arc::new(super::MockLlmClient::new(vec![vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "skill_info",
                            "arguments": "{\"name\": \"math\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let runtime = AgentBuilder::new(llm)
            .system_prompt("sys")
            .register_skill(MathSkill)
            .skill_detail_tool_name("skill_info")
            .build()
            .unwrap();

        let session_id = runtime.create_session().await;
        let result = runtime
            .run_turn_collect(session_id, "tell me about math skill")
            .await;
        assert!(result.is_ok(), "Expected ok, got: {result:?}");

        let (events, _outcome) = result.unwrap();
        let skill_loaded = events.iter().any(|e| {
            matches!(e, RuntimeEvent::ToolCallFinished { tool_name, summary, .. }
                if tool_name == "skill_info" && summary.contains("Math Skill"))
        });
        assert!(
            skill_loaded,
            "skill_info tool should return Math Skill detail"
        );
    }

    #[tokio::test]
    async fn test_skill_tool_name_conflict() {
        let llm = Arc::new(super::MockLlmClient::new(vec![]));

        let result = AgentBuilder::new(llm)
            .register_tool(AddTool) // registers "add" directly
            .register_skill(MathSkill) // MathSkill also registers "add"
            .build();

        assert!(result.is_err(), "Tool name conflict should be detected");
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(
            err_msg.contains("Tool name conflict"),
            "Error should mention tool name conflict: {err_msg}"
        );
    }

    #[test]
    fn test_lazy_skill_prompter() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(MathSkill)];
        let prompter = LazySkillPrompter::new();
        let prompt = prompter.build_prompt(&skills, "get_skill_detail");
        assert!(prompt.contains("math"), "Prompt should contain skill name");
        assert!(
            prompt.contains("read_file"),
            "Prompt should mention read_file (prompt-injection mode)"
        );
    }

    #[test]
    fn test_lazy_skill_prompter_custom_config() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(MathSkill)];
        let prompter = LazySkillPrompter::new()
            .title("## My Skills")
            .instruction("> Use read_file to see details")
            .item_prefix("+ ");
        let prompt = prompter.build_prompt(&skills, "my_get_detail");
        assert!(prompt.contains("## My Skills"));
        assert!(prompt.contains("> Use read_file to see details"));
        assert!(prompt.contains("+ "));
    }

    #[test]
    fn test_full_detail_prompter() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(MathSkill)];
        let prompter = agent_works::skill::FullDetailPrompter;
        let prompt = prompter.build_prompt(&skills, "get_skill_detail");
        assert!(prompt.contains("math"), "Should contain skill name");
        assert!(
            prompt.contains("Math Skill"),
            "Should contain detailed description"
        );
    }

    #[test]
    fn test_skill_default_methods() {
        assert_eq!(MathSkill.version(), "0.1.0");
        assert!(MathSkill.tags().is_empty());
        assert_eq!(MathSkill.author(), "");
    }
}

// ---------------------------------------------------------------------------
// MCP module tests
// ---------------------------------------------------------------------------

#[cfg(feature = "mcp")]
mod mcp_tests {
    use super::*;
    use agent_works::mcp::*;

    #[test]
    fn test_mcp_tool_info_creation() {
        let info = McpToolInfo {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({"type": "object"}),
        };
        assert_eq!(info.name, "test_tool");
        assert_eq!(info.description, "A test tool");
    }

    #[test]
    fn test_mcp_transport_variants() {
        let http =
            McpTransport::http("http://localhost:8080").with_header("Authorization", "Bearer t");
        assert!(matches!(http, McpTransport::Http { .. }));

        let stdio = McpTransport::stdio("npx", ["-y", "mcp-server"]).with_env("TOKEN", "t");
        assert!(matches!(stdio, McpTransport::Stdio { .. }));
    }

    #[test]
    fn test_mcp_server_config() {
        let config = McpServerConfig {
            name: "my-server".to_string(),
            transport: McpTransport::http("http://localhost:8080"),
            auto_reconnect: true,
        };
        assert_eq!(config.name, "my-server");
        assert!(config.auto_reconnect);
    }
}

// ---------------------------------------------------------------------------
// Skill detail tool standalone tests
// NOTE: SkillDetailTool is defined in phi-kernel-tools, not agent-works.
// These tests were moved to phi-kernel-tests/src/skill/detail_tool.rs.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// LazySkillPrompter dynamic tool name tests
// ---------------------------------------------------------------------------

#[cfg(feature = "skill")]
mod prompter_tests {
    use agent_base::Tool;
    use agent_works::skill::{FullDetailPrompter, LazySkillPrompter, Skill, SkillPrompter};
    use std::sync::Arc;

    struct TestSkill;
    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            "test_skill"
        }
        fn brief_description(&self) -> String {
            "A test skill".to_string()
        }
        fn detailed_description(&self) -> String {
            "Detailed test instructions".to_string()
        }
        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![]
        }
    }

    #[test]
    fn test_lazy_prompter_default_read_file_instruction() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(TestSkill)];
        let prompter = LazySkillPrompter::new();
        let prompt = prompter.build_prompt(&skills, "get_skill_detail");
        assert!(
            prompt.contains("read_file"),
            "default prompt should use read_file mode"
        );
        assert!(prompt.contains("test_skill"));
    }

    #[test]
    fn test_lazy_prompter_ignores_tool_name() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(TestSkill)];
        let prompter = LazySkillPrompter::new();
        let prompt = prompter.build_prompt(&skills, "my_custom_detail");
        assert!(
            !prompt.contains("my_custom_detail"),
            "read_file mode should not mention the detail tool name"
        );
        assert!(
            !prompt.contains("get_skill_detail"),
            "should NOT contain default name"
        );
        assert!(prompt.contains("read_file"));
    }

    #[test]
    fn test_lazy_prompter_custom_instruction_verbatim() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(TestSkill)];
        let prompter = LazySkillPrompter::new().instruction("Use `read_file` with the skill path.");
        let prompt = prompter.build_prompt(&skills, "detail_query");
        assert!(
            prompt.contains("Use `read_file` with the skill path."),
            "custom instruction should be emitted verbatim: {}",
            prompt
        );
    }

    #[test]
    fn test_full_detail_prompter_ignores_tool_name() {
        let skills: Vec<Arc<dyn Skill>> = vec![Arc::new(TestSkill)];
        let prompter = FullDetailPrompter;
        let prompt = prompter.build_prompt(&skills, "whatever");
        assert!(prompt.contains("Detailed test instructions"));
        assert!(
            !prompt.contains("whatever"),
            "FullDetailPrompter should not use tool name"
        );
    }

    #[test]
    fn test_lazy_prompter_empty_skills() {
        let skills: Vec<Arc<dyn Skill>> = vec![];
        let prompter = LazySkillPrompter::new();
        let prompt = prompter.build_prompt(&skills, "get_skill_detail");
        assert!(
            prompt.is_empty(),
            "empty skills should produce empty prompt"
        );
    }
}

// ─────────────────────────────────────────────────────────────
// AgentHandle tests (17. session scheduling tests)
// ─────────────────────────────────────────────────────────────

mod agent_handle_tests {
    use super::*;
    use agent_base::RuntimeEvent;
    use agent_works::AgentHandle;

    /// Scenario 1: basic session scheduling — send_input → recv_event → RunFinished
    #[tokio::test]
    async fn test_handle_basic_session() {
        let llm = Arc::new(MockLlmClient::new(vec![vec![
            StreamChunk::Text("disk: 37% used".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let runtime = AgentBuilder::new(llm)
            .system_prompt("You are a helpful assistant")
            .build()
            .unwrap();

        let mut handle = AgentHandle::new(runtime);
        handle.send_input("check disk space").await.unwrap();

        let mut all_events = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, handle.recv_event()).await {
            all_events.push(format!("{:?}", event));
            if matches!(event, RuntimeEvent::RunFinished { .. }) {
                break;
            }
        }

        assert!(
            all_events.iter().any(|e| e.contains("TextDelta")),
            "should have received TextDelta, got: {:?}",
            all_events
        );
        assert!(
            all_events.iter().any(|e| e.contains("RunFinished")),
            "should have received RunFinished, got: {:?}",
            all_events
        );
    }

    /// Scenario 2: cancel a session — send_input → cancel → RunCancelled
    #[tokio::test]
    async fn test_handle_cancel() {
        // Create a mock that uses a channel-based stream that blocks
        type BlockingTx =
            Arc<Mutex<Option<tokio::sync::mpsc::UnboundedSender<Result<StreamChunk, LlmError>>>>>;
        struct BlockingLlm {
            tx: BlockingTx,
        }
        #[async_trait]
        impl LlmProvider for BlockingLlm {
            async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
                let (tx, rx) =
                    tokio::sync::mpsc::unbounded_channel::<Result<StreamChunk, LlmError>>();
                *self.tx.lock().unwrap() = Some(tx);
                let stream = futures_util::stream::unfold(rx, |mut rx| async move {
                    let item = rx.recv().await?;
                    Some((item, rx))
                });
                Ok(ChatStream::new(Box::pin(stream)))
            }
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
                unimplemented!()
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::default()
            }
            fn info(&self) -> agent_base::llm_trait::ProviderInfo {
                agent_base::llm_trait::ProviderInfo {
                    name: "blocking".to_string(),
                    model: "blocking-model".to_string(),
                    version: None,
                }
            }
        }

        let blocking = Arc::new(BlockingLlm {
            tx: Arc::new(Mutex::new(None)),
        });
        let llm_ref = blocking.clone();
        let llm: Arc<dyn LlmProvider> = blocking;

        let runtime = AgentBuilder::new(llm)
            .system_prompt("You are a helpful assistant")
            .build()
            .unwrap();

        let mut handle = AgentHandle::new(runtime);
        handle.send_input("long running task").await.unwrap();

        // Wait for the stream to be created (the LLM call happens)
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Send a chunk to start processing
        if let Some(tx) = llm_ref.tx.lock().unwrap().as_ref() {
            let _ = tx.send(Ok(StreamChunk::Text("processing...".to_string())));
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Cancel while stream is blocked waiting for more chunks
        handle.cancel();

        // Should receive RunCancelled
        let mut got_cancelled = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, handle.recv_event()).await {
            if matches!(event, RuntimeEvent::RunCancelled { .. }) {
                got_cancelled = true;
                break;
            }
        }

        assert!(
            got_cancelled,
            "should have received RunCancelled after cancel()"
        );
    }

    /// Scenario 3: error handling — error → RunFinished (not hang)
    #[tokio::test]
    async fn test_handle_error_recovery() {
        // Use an LLM that returns an error
        struct ErrorLlm;
        #[async_trait]
        impl LlmProvider for ErrorLlm {
            async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
                Err(LlmError::llm("simulated LLM failure"))
            }
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
                Err(LlmError::llm("simulated LLM failure"))
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::default()
            }
            fn info(&self) -> agent_base::llm_trait::ProviderInfo {
                agent_base::llm_trait::ProviderInfo {
                    name: "error-mock".to_string(),
                    model: "error-model".to_string(),
                    version: None,
                }
            }
        }

        let runtime = AgentBuilder::new(Arc::new(ErrorLlm))
            .system_prompt("test")
            .build()
            .unwrap();

        let mut handle = AgentHandle::new(runtime);
        handle.send_input("trigger error").await.unwrap();

        // Should receive RunFinished (not hang forever)
        let mut got_finished = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, handle.recv_event()).await {
            if matches!(event, RuntimeEvent::RunFinished { .. }) {
                got_finished = true;
                break;
            }
        }

        assert!(
            got_finished,
            "should have received RunFinished after error, not hang"
        );
    }

    /// Scenario 2b: new commands can still be sent after a cancel
    #[tokio::test]
    async fn test_handle_cancel_then_continue() {
        let llm = Arc::new(MockLlmClient::new(vec![
            // First response: never stops (will be cancelled)
            vec![StreamChunk::Text("processing...".to_string())],
            // Second response: normal
            vec![
                StreamChunk::Text("done!".to_string()),
                StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                },
            ],
        ]));

        let runtime = AgentBuilder::new(llm)
            .system_prompt("test")
            .build()
            .unwrap();

        let mut handle = AgentHandle::new(runtime);

        // First turn: send + cancel
        handle.send_input("long task").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        handle.cancel();

        // Drain events until RunCancelled
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, handle.recv_event()).await {
            if matches!(event, RuntimeEvent::RunCancelled { .. }) {
                break;
            }
        }

        // Second turn: should work normally
        handle.send_input("check disk").await.unwrap();
        let mut got_finished = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, handle.recv_event()).await {
            if matches!(event, RuntimeEvent::RunFinished { .. }) {
                got_finished = true;
                break;
            }
        }

        assert!(got_finished, "should be able to continue after cancel");
    }
}

// ---------------------------------------------------------------------------
// Multi-Agent integration tests
// ---------------------------------------------------------------------------

#[cfg(feature = "multi_agent")]
mod multi_agent_tests {
    use super::*;
    use agent_works::multi_agent::{
        AgentPath, ChildPermissionMode, MultiAgentConfig, MultiAgentRuntime,
    };

    /// Test that AgentPath parsing and navigation work correctly.
    #[test]
    fn test_agent_path_basics() {
        let root = AgentPath::root();
        assert!(root.is_root());
        assert_eq!(root.depth(), 0);
        assert_eq!(root.to_string(), "root");

        let child = root.join("searcher");
        assert!(!child.is_root());
        assert_eq!(child.depth(), 1);
        assert_eq!(child.name(), "searcher");
        assert_eq!(child.parent(), Some(root));

        let parsed: AgentPath = "root/searcher".parse().unwrap();
        assert_eq!(parsed, child);
    }

    /// Test that MultiAgentConfig defaults are correct.
    #[test]
    fn test_multi_agent_config_defaults() {
        let config = MultiAgentConfig::enabled();
        assert!(config.enabled);
        assert_eq!(config.max_sub_agents, 8);
        assert_eq!(config.max_agent_depth, 1);
    }

    /// Test building an agent with multi-agent enabled.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_build_with_multi_agent_enabled() {
        let llm = Arc::new(MockLlmClient::new(vec![vec![
            StreamChunk::Text("I'll analyze this in parallel.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let runtime = AgentBuilder::new(llm)
            .register_tool(EchoTool)
            .with_multi_agent(MultiAgentConfig::enabled())
            .build()
            .unwrap();

        let session_id = runtime.create_session().await;
        let result = runtime
            .run_turn_collect(session_id.clone(), "research topic")
            .await;

        assert!(result.is_ok(), "Expected ok, got: {result:?}");
        let (_events, outcome) = result.unwrap();
        assert_eq!(outcome, RunOutcome::Completed);
    }

    /// Test that multi-agent builder with custom limits works.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_agent_custom_limits() {
        let llm = Arc::new(MockLlmClient::new(vec![vec![
            StreamChunk::Text("ok".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let config = MultiAgentConfig {
            enabled: true,
            max_sub_agents: 4,
            max_agent_depth: 2,
            child_permission_mode: ChildPermissionMode::Full,
            child_excluded_tools: Vec::new(),
            child_reasoning_effort: None,
            child_read_only: true,
            control: Default::default(),
            ..Default::default()
        };

        let runtime = AgentBuilder::new(llm)
            .with_multi_agent(config)
            .build()
            .unwrap();

        let session_id = runtime.create_session().await;
        let result = runtime.run_turn_collect(session_id, "test").await;
        assert!(result.is_ok(), "Expected ok: {result:?}");
    }

    /// Test multi-agent spawn-and-task lifecycle via the runtime API.
    #[tokio::test]
    async fn test_runtime_spawn_child_and_wait() {
        use tokio_util::sync::CancellationToken;

        let llm = Arc::new(MockLlmClient::new(vec![vec![
            StreamChunk::Text("child response".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]));

        let cancel = CancellationToken::new();
        let config = MultiAgentConfig::enabled();

        let runtime = Arc::new(MultiAgentRuntime::new(
            config,
            llm,
            vec![],
            cancel,
            None,
            agent_base::Language::En,
            None,
            None,
        ));

        // Spawn a child
        let path = runtime
            .spawn_child(
                "test-worker",
                "You are a test worker.".to_string(),
                1,
                false,
                vec![],
            )
            .await
            .expect("spawn should succeed");

        assert_eq!(path, "root/test-worker");

        // Verify child is registered. Freshly spawned derives `done` (the
        // fact model has no `idle` — same facts as a finished agent).
        let agents = runtime.list_agents();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].agent_path, "root/test-worker");
        assert_eq!(agents[0].status, "done");
        assert_eq!(agents[0].tool_calls, 0);

        // Send a task
        runtime
            .send_task(&path, "do something".to_string(), true)
            .expect("send_task should succeed");

        // Wait for result
        let result = runtime.wait_for_result(Some(&path), 5000).await;
        assert_eq!(result.status, "ok");
        assert!(result.result.is_some());

        // Close the child
        let close_result = runtime.close_agent(&path).expect("close should succeed");
        assert!(close_result.closed);
        assert_eq!(close_result.message, "agent closed");

        // Verify closed. `close_agent` is "cancel + deferred cleanup"
        // (design doc §4/§5.2): the ChildCleanup drop-guard releases the
        // registry slot once the child loop returns, so the empty listing is
        // eventually — not synchronously — true.
        for _ in 0..200 {
            if runtime.list_agents().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let agents = runtime.list_agents();
        assert!(agents.is_empty());
    }

    /// Test spawn limit enforcement through the registry.
    #[tokio::test]
    async fn test_spawn_limit_enforcement() {
        use tokio_util::sync::CancellationToken;

        let llm = Arc::new(MockLlmClient::new(vec![]));
        let cancel = CancellationToken::new();
        let config = MultiAgentConfig {
            enabled: true,
            max_sub_agents: 2,
            max_agent_depth: 1,
            child_permission_mode: ChildPermissionMode::Full,
            child_excluded_tools: Vec::new(),
            child_reasoning_effort: None,
            child_read_only: true,
            control: Default::default(),
            ..Default::default()
        };

        let runtime = Arc::new(MultiAgentRuntime::new(
            config,
            llm,
            vec![],
            cancel,
            None,
            agent_base::Language::En,
            None,
            None,
        ));

        // Should be able to spawn up to max
        runtime
            .spawn_child("worker-1", "prompt".to_string(), 1, false, vec![])
            .await
            .expect("first spawn");
        runtime
            .spawn_child("worker-2", "prompt".to_string(), 1, false, vec![])
            .await
            .expect("second spawn");

        // Third spawn should fail
        let err = runtime
            .spawn_child("worker-3", "prompt".to_string(), 1, false, vec![])
            .await;
        assert!(err.is_err(), "third spawn should fail: {:?}", err);
        let msg = err.unwrap_err();
        assert!(msg.contains("max"), "expected 'max' in error: {}", msg);

        let agents = runtime.list_agents();
        assert_eq!(agents.len(), 2);
    }

    /// Test depth limit enforcement.
    #[tokio::test]
    async fn test_depth_limit_enforcement() {
        use tokio_util::sync::CancellationToken;

        let llm = Arc::new(MockLlmClient::new(vec![]));
        let cancel = CancellationToken::new();
        let config = MultiAgentConfig {
            enabled: true,
            max_sub_agents: 10,
            max_agent_depth: 1,
            child_permission_mode: ChildPermissionMode::Full,
            child_excluded_tools: Vec::new(),
            child_reasoning_effort: None,
            child_read_only: true,
            control: Default::default(),
            ..Default::default()
        };

        let runtime = Arc::new(MultiAgentRuntime::new(
            config,
            llm,
            vec![],
            cancel,
            None,
            agent_base::Language::En,
            None,
            None,
        ));

        // depth=1: allowed
        runtime
            .spawn_child("level1", "prompt".to_string(), 1, false, vec![])
            .await
            .expect("depth 1 should be allowed");

        // depth=2: should fail
        let err = runtime
            .spawn_child("level2", "prompt".to_string(), 2, false, vec![])
            .await;
        assert!(err.is_err(), "depth 2 should fail: {:?}", err);
        let msg = err.unwrap_err();
        assert!(msg.contains("depth"), "expected 'depth' in error: {}", msg);
    }

    /// Test close_agent on non-existent path returns error.
    #[tokio::test]
    async fn test_close_nonexistent_agent() {
        use tokio_util::sync::CancellationToken;

        let llm = Arc::new(MockLlmClient::new(vec![]));
        let cancel = CancellationToken::new();
        let config = MultiAgentConfig::enabled();

        let runtime = Arc::new(MultiAgentRuntime::new(
            config,
            llm,
            vec![],
            cancel,
            None,
            agent_base::Language::En,
            None,
            None,
        ));

        let result = runtime.close_agent("root/ghost");
        assert!(result.is_ok());
        assert!(!result.as_ref().unwrap().closed);
    }
}
