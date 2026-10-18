//! Stage-4 concurrent stress tests (design doc §12 stage 4: "concurrent
//! stress-test hardening").
//!
//! Unit-level cleanup tests (`runtime/tests/cleanup.rs`) prove each individual
//! teardown path is exact; these tests hammer the whole spawn→work→close loop
//! at scale and under contention, then assert *conservation*: the registry,
//! the mailbox, the concurrency limiter and the cumulative spawn count all
//! return to their expected values — no slot leak, no orphan mailbox, no
//! phantom reservation.
//!
//! Loop counts scale with the profile: the design doc's "1,000-run" target runs
//! under `cargo test --release`; debug keeps 10× fewer iterations so the
//! default CI run stays fast.

#![cfg(feature = "multi_agent")]

use std::sync::Arc;
use std::time::Duration;

use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{Content, Language, StreamChunk, Tool, ToolContext};
use agent_works::multi_agent::{ChildOutcome, ControlConfig, MultiAgentConfig, MultiAgentRuntime};
use tokio_util::sync::CancellationToken;

/// 100 rounds in debug (fast CI), 1000 in release (the doc's 1,000-run stress test).
const ITERS: usize = if cfg!(debug_assertions) { 100 } else { 1000 };

// ── fixtures ───────────────────────────────────────────────────────────────

struct StubLlm;

#[async_trait::async_trait]
impl LlmProvider for StubLlm {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(vec![
            Ok(StreamChunk::Text("done".to_string())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ]))))
    }
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        unreachable!("unused")
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".into(),
            model: "stub".into(),
            version: None,
        }
    }
}

struct StubTool(&'static str);

#[async_trait::async_trait]
impl Tool for StubTool {
    fn name(&self) -> &'static str {
        self.0
    }
    fn description(&self) -> &'static str {
        "stub"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
}

fn tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(StubTool("read_file")) as Arc<dyn Tool>,
        Arc::new(StubTool("list_files")) as Arc<dyn Tool>,
    ]
}

fn runtime_with(config: MultiAgentConfig) -> Arc<MultiAgentRuntime> {
    runtime_with_client(Arc::new(StubLlm), config)
}

fn runtime_with_client(
    client: Arc<dyn LlmProvider>,
    config: MultiAgentConfig,
) -> Arc<MultiAgentRuntime> {
    Arc::new(MultiAgentRuntime::new(
        config,
        client,
        tools(),
        CancellationToken::new(),
        None,
        Language::En,
        None,
        None,
    ))
}

/// `stream` never returns — keeps spawned children inside `run_turn`
/// (`running`) for the whole test, so a same-name re-spawn keeps hitting the
/// registry rejection instead of recycling a finished predecessor.
struct HangingLlm;

#[async_trait::async_trait]
impl LlmProvider for HangingLlm {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        unreachable!("the child is aborted long before this returns")
    }
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        unreachable!("unused")
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "hanging".into(),
            model: "hanging".into(),
            version: None,
        }
    }
}

/// Teardown (ChildCleanup) runs on the child task's exit, so post-close
/// conservation is eventually-true, never immediate.
async fn poll_until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..400 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("conservation not reached within 4s: {what}");
}

async fn spawn_child(rt: &Arc<MultiAgentRuntime>, name: &str) -> Child {
    let child = rt
        .child()
        .system_prompt("you are a worker; answer briefly")
        .spawn(name)
        .await
        .expect("spawn ok");
    Child(child)
}

/// Newtype so the handle keeps a name in test code.
struct Child(agent_works::multi_agent::ChildHandle);

// ── 1. thousand-round full lifecycle: nothing leaks ────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn thousand_round_spawn_work_close_conserves_everything() {
    let rt = runtime_with(MultiAgentConfig::enabled());

    for i in 0..ITERS {
        let child = spawn_child(&rt, &format!("w{i}")).await;
        child.0.task(format!("round {i}")).expect("task delivered");
        match child.0.wait(Duration::from_secs(5)).await {
            ChildOutcome::Ok { text, .. } => assert_eq!(text.as_deref(), Some("done")),
            other => panic!("round {i}: expected Ok, got {other:?}"),
        }
        child.0.close().expect("close ok");
        // Sequential rounds: only this child is live, so "registry empty" is
        // the exact per-round conservation check.
        poll_until("child left the registry", || rt.list_agents().is_empty()).await;
    }

    let st = rt.control().status();
    assert!(rt.list_agents().is_empty(), "registry must end empty");
    assert!(rt.mailbox().is_empty(), "mailbox hub must end empty");
    // P2 tombstones: each round's close posted one Closed notification and
    // nothing consumed it (this test runs no watcher). Conservation means
    // unregister preserves it as a readable tombstone — not that it vanishes.
    {
        use agent_works::multi_agent::mailbox::MailboxStatus;
        let mut closed = 0;
        while let Some(result) = rt.mailbox().try_recv_any() {
            assert_eq!(
                result.status,
                MailboxStatus::Closed,
                "only close notifications may remain"
            );
            closed += 1;
        }
        assert_eq!(
            closed, ITERS,
            "every close leaves exactly one readable Closed"
        );
    }
    assert_eq!(rt.mailbox().total_pending_results(), 0);
    assert_eq!(st.live_children, 0, "every concurrency slot returned");
    // Cumulative semantics (§7.2): committed spawns are never given back.
    assert_eq!(st.spawn_count, ITERS);
}

// ── 2. contention on the concurrency gate: exactly `cap` winners ──────────

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_storm_respects_cap_exactly() {
    const CAP: usize = 4;
    const RACERS: usize = 8;
    let rt = runtime_with(MultiAgentConfig {
        max_sub_agents: 64, // registry gate wide open — the limiter decides
        control: ControlConfig {
            max_concurrency: Some(CAP),
            ..Default::default()
        },
        ..MultiAgentConfig::enabled()
    });

    // All eight race to spawn at the same instant.
    let mut set = tokio::task::JoinSet::new();
    for i in 0..RACERS {
        let rt = Arc::clone(&rt);
        set.spawn(async move {
            rt.child()
                .system_prompt("worker")
                .spawn(format!("c{i}"))
                .await
                .map(|_| ())
                .map_err(|_| ())
        });
    }
    let mut ok = 0;
    let mut err = 0;
    while let Some(res) = set.join_next().await {
        match res.unwrap() {
            Ok(()) => ok += 1,
            Err(()) => err += 1,
        }
    }
    // CAS reservation: no overshoot, no undershoot — exactly CAP get a slot.
    assert_eq!(ok, CAP, "exactly cap spawns succeed under contention");
    assert_eq!(err, RACERS - CAP);
    assert_eq!(rt.control().status().live_children, CAP);

    // Release every slot, then a late spawn must succeed.
    for path in rt.list_agents() {
        rt.close_agent(&path.agent_path).ok();
    }
    poll_until("limiter drained", || {
        rt.control().status().live_children == 0
    })
    .await;
    spawn_child(&rt, "late").await;
    rt.close_agent("root/late").ok();
}

// ── 3. failed spawns roll their budget reservation back, at scale ─────────

#[tokio::test(flavor = "multi_thread")]
async fn failed_spawns_never_charge_the_spawn_budget() {
    // Same-name recycle (session 20260920_5ba1bed4) makes a re-spawn of a
    // finished predecessor succeed — and a freshly spawned child with no
    // task derives `done` immediately. The duplicate rejections this test
    // needs therefore require a predecessor that is verifiably mid-task:
    // u0 gets a hanging task and the loop waits for `running` before any
    // dupe attempt.
    const UNIQUE: usize = 50; // even attempts
    const DUPES: usize = 50; // odd attempts reuse the running u0
    let rt = runtime_with_client(
        Arc::new(HangingLlm),
        MultiAgentConfig {
            max_sub_agents: 128,
            control: ControlConfig {
                max_spawns: Some(UNIQUE + 10), // cap high enough to admit all spawns
                ..Default::default()
            },
            ..MultiAgentConfig::enabled()
        },
    );

    let u0 = rt
        .child()
        .system_prompt("worker")
        .spawn("u0")
        .await
        .expect("u0 spawn ok");
    u0.task("hang").unwrap();
    poll_until("u0 verifiably running", || {
        rt.list_agents()
            .iter()
            .any(|a| a.agent_path == "root/u0" && a.status == "running")
    })
    .await;

    let mut ok = 0;
    let mut err = 0;
    for i in 0..UNIQUE + DUPES {
        let name = if i % 2 == 0 {
            format!("v{}", i / 2)
        } else {
            "u0".to_string() // registry duplicate → spawn fails after reserve
        };
        match rt.child().system_prompt("worker").spawn(name).await {
            Ok(_) => ok += 1,
            Err(_) => err += 1,
        }
    }
    assert_eq!(ok, UNIQUE);
    assert_eq!(err, DUPES);
    // Ticket discipline (§7.2): only the committed spawns remain counted —
    // the 50 registry rejections rolled their reservation straight back.
    // UNIQUE + 1: u0 (the dupe target, spawned above the loop) plus the 50
    // fresh v-names; a hanging task charges no extra ticket.
    assert_eq!(
        rt.control().status().spawn_count,
        UNIQUE + 1,
        "rolled-back spawns must not charge the budget"
    );
}

// ── 4. mixed lifecycle chaos: every path conserves ────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn mixed_lifecycle_chaos_conserves_everything() {
    const CHAOS: usize = if cfg!(debug_assertions) { 40 } else { 200 };
    let rt = runtime_with(MultiAgentConfig::enabled());

    for i in 0..CHAOS {
        let name = format!("m{i}");
        let child = spawn_child(&rt, &name).await;
        match i % 4 {
            // Closed without ever working — result queue stays empty.
            0 => {
                child.0.close().ok();
            }
            // Straight work order, result consumed.
            1 => {
                child.0.task("work").unwrap();
                assert!(matches!(
                    child.0.wait(Duration::from_secs(5)).await,
                    ChildOutcome::Ok { .. }
                ));
                child.0.close().ok();
            }
            // Message parked first (no trigger), then a task folds it in.
            2 => {
                child.0.send("note").unwrap();
                child.0.task("work with the note").unwrap();
                assert!(matches!(
                    child.0.wait(Duration::from_secs(5)).await,
                    ChildOutcome::Ok { .. }
                ));
                child.0.close().ok();
            }
            // No task at all: wait must time out (not error), then close.
            _ => {
                assert!(matches!(
                    child.0.wait(Duration::from_millis(50)).await,
                    ChildOutcome::Timeout
                ));
                child.0.close().ok();
            }
        }
    }

    poll_until("all closed children gone", || rt.list_agents().is_empty()).await;
    // P2 tombstones: every chaos close posted a Closed notification; with no
    // watcher running they remain readable tombstones until drained here.
    {
        use agent_works::multi_agent::mailbox::MailboxStatus;
        let mut closed = 0;
        while let Some(result) = rt.mailbox().try_recv_any() {
            assert_eq!(result.status, MailboxStatus::Closed);
            closed += 1;
        }
        assert_eq!(
            closed, CHAOS,
            "every chaos close leaves one readable Closed"
        );
    }
    let st = rt.control().status();
    assert!(rt.list_agents().is_empty());
    assert!(rt.mailbox().is_empty(), "no orphan mailboxes after chaos");
    assert_eq!(rt.mailbox().total_pending_results(), 0);
    assert_eq!(st.live_children, 0);
    assert_eq!(st.spawn_count, CHAOS, "chaos committed every spawn");
}
