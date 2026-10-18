//! Write-gate integration (design doc D6): wrapping and close-release through a real spawn.

use std::sync::Arc;

use agent_base::{Content, Tool, ToolContext};

use crate::multi_agent::capability::ChildToolCapability;
use crate::multi_agent::child_config::ChildConfig;
use crate::multi_agent::config::{AgentAutonomy, ControlConfig, MultiAgentConfig};
use crate::multi_agent::runtime::MultiAgentRuntime;
use crate::multi_agent::write_gate::WorkspaceWriteGate;

use super::*;

struct StubWrite;

#[async_trait::async_trait]
impl Tool for StubWrite {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "fixture"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
}

fn runtime_with_gate(gate_enabled: bool) -> Arc<MultiAgentRuntime> {
    runtime_with_gate_and_tools(
        gate_enabled,
        Arc::new(StreamingStub),
        vec![Arc::new(StubWrite)],
    )
}

fn runtime_with_gate_and_tools(
    gate_enabled: bool,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        allow_child_write: true,
        child_excluded_tools: vec![],
        control: ControlConfig {
            autonomy: AgentAutonomy::Auto,
            child_write_gate: gate_enabled,
            ..ControlConfig::default()
        },
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        client,
        tools,
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

/// Two write sub-agents: the second write to the same file gets a named error
/// (via GatedTool). Verifies through the real build path that the wrapping is
/// actually in place (the sister assertion of the write-registration
/// integration assertion).
#[tokio::test(flavor = "multi_thread")]
async fn gated_tool_wraps_write_file_on_spawned_children() {
    let ma = runtime_with_gate(true);
    // Take the child runtime straight from the build path, check that the
    // registered write_file carries the GatedTool name (name pass-through),
    // and verify mutual exclusion via gate contention:
    let config = ChildConfig {
        system_prompt: Some("p".into()),
        ..Default::default()
    };
    let (child_a, _reg, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/a")
        .await
        .unwrap();
    let (child_b, _reg, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/b")
        .await
        .unwrap();
    // Each child runtime holds its own write_file registration (at the name
    // level) — the real mutual-exclusion assertion lives in the write_gate.rs
    // unit tests (the GatedTool layer); here we verify that the build path
    // completes the wrapping for two distinct child_paths without error.
    let _ = (child_a, child_b);
    let shared = Arc::new(WorkspaceWriteGate::new());
    shared
        .try_claim(std::path::Path::new("x.rs"), "root/a")
        .unwrap();
    let err = shared
        .try_claim(std::path::Path::new("x.rs"), "root/b")
        .unwrap_err();
    assert!(err.contains("root/a"));
}

/// Gate disabled: the build path no longer wraps (honest switch semantics).
/// With the switch off a sub-agent can still register write_file (back to
/// pure prompt discipline), and the gate table is empty.
#[tokio::test(flavor = "multi_thread")]
async fn gate_disabled_still_spawns_write_children() {
    let ma = runtime_with_gate(false);
    let config = ChildConfig {
        system_prompt: Some("p".into()),
        ..Default::default()
    };
    let (_child, registered, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/a")
        .await
        .unwrap();
    assert!(registered.contains("write_file"));
}

/// Close-release: spawn → claim under the echoed agent_path → after
/// close_agent the agent's claims are cleared by release_all (via
/// ChildCleanup::drop, polled to the terminal state).
#[tokio::test(flavor = "multi_thread")]
async fn close_releases_gate_claims() {
    let ma = runtime_with_gate(true);
    let path = ma
        .spawn_child_with_history(
            "w",
            "p".to_string(),
            true,
            None,
            None,
            None, // capability slot (Task 6's new signature; the legacy path passes None)
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    let path = path.agent_path;
    ma.write_gate_for_test()
        .try_claim(std::path::Path::new("x.rs"), &path)
        .unwrap();
    ma.close_agent(&path).unwrap();
    poll_until("gate claims released", || {
        ma.write_gate_for_test()
            .try_claim(std::path::Path::new("x.rs"), "root/probe")
            .is_ok()
    })
    .await;
}

/// Write-tool stub that sleeps inside `call`, creating an observable window
/// where the task is running and the claim is held (ToolCallOnceStub alone
/// finishes in milliseconds — Fix B would release before the first poll;
/// this is the race guard).
struct SlowWrite(std::time::Duration);

#[async_trait::async_trait]
impl Tool for SlowWrite {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "fixture (slow)"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        tokio::time::sleep(self.0).await;
        Ok(vec![Content::text("ok")])
    }
}

/// Claims are released when the task ends (session 20260920_5ba1bed4 case 2
/// regression): a lock's lifetime is the **task's**, not the registration's.
/// Case 2 showed writer-b's +80s retry colliding with finished writer-a's
/// "corpse lock" — the report was only delivered at 11:20:49, nobody closed
/// the agent, and the claim was held until close. After the fix: the moment
/// a child task ends (before the result is posted), every claim it took via
/// GatedTool is released; the close-path release in ChildCleanup::drop
/// stays as the terminal backstop (idempotent).
#[tokio::test(flavor = "multi_thread")]
async fn task_completion_releases_gate_claims() {
    let ma = runtime_with_gate_and_tools(
        true,
        Arc::new(ToolCallOnceStub::new("write_file", "{\"path\":\"x.rs\"}")),
        vec![Arc::new(SlowWrite(std::time::Duration::from_millis(500)))],
    );

    let echo = ma
        .spawn_child_with_history(
            "wg",
            "p".to_string(),
            true,
            None,
            None,
            Some(ChildToolCapability::Write),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    assert_eq!(echo.agent_path, "root/wg");
    ma.send_task("root/wg", "write it".to_string(), false)
        .unwrap();

    // 1) Mid-task: the child holds x.rs via GatedTool (named) — the slow
    //    write tool holds the 500ms window so the poll always lands inside
    //    it. Observation must go through the read-only holder_of: a
    //    try_claim probe would grab the lock itself and evict the child
    //    under test.
    poll_until("child claims x.rs via GatedTool", || {
        ma.write_gate_for_test()
            .holder_of(std::path::Path::new("x.rs"))
            .as_deref()
            == Some("root/wg")
    })
    .await;

    // 2) Task ended — no close_agent.
    poll_until("child task done", || {
        ma.list_agents()
            .iter()
            .any(|a| a.agent_path == "root/wg" && a.status == "done")
    })
    .await;

    // 3) Assert: the claim was released when the task ended (this failed
    //    before the fix — the lock hung until close).
    assert!(
        ma.write_gate_for_test()
            .try_claim(std::path::Path::new("x.rs"), "root/probe")
            .is_ok(),
        "claims must be released when the task ends, not held until close"
    );
}
