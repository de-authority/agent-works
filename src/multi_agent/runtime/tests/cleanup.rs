use super::*;

// ── ChildCleanup: all three exit paths release every resource
//    (§4 / §5.4, design doc §12 stage-1 acceptance) ──

fn make_ma_runtime_concurrent(
    max_concurrency: usize,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        control: ControlConfig {
            max_concurrency: Some(max_concurrency),
            ..Default::default()
        },
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        client,
        vec![],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

/// Provider whose `stream` panics — exercises the unwind path.
struct PanickingLlm;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for PanickingLlm {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        panic!("boom: provider stream explodes")
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
            name: "panicking".to_string(),
            model: "panicking".to_string(),
            version: None,
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn child_cleanup_normal_exit_releases_all_resources() {
    let ma = make_ma_runtime_concurrent(4, Arc::new(StreamingStub));
    let limiter = Arc::clone(&ma.limiter);
    let registry = Arc::clone(&ma.registry);
    let mailbox = Arc::clone(&ma.mailbox);
    let path = AgentPath::root().join("w");

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap();
    assert_eq!(limiter.current(), 1, "slot held while child is live");
    assert_eq!(registry.lock().unwrap().count(), 1);

    ma.send_task("root/w", "work".to_string(), false).unwrap();
    let r = ma.wait_for_result(Some("root/w"), 3000).await;
    assert_eq!(r.status, "ok");
    // Task finished but the child is still live (loop back at recv):
    // the slot belongs to the child's life, not the task's (§7.3).
    assert_eq!(limiter.current(), 1);

    // close = "cancel + deferred cleanup" (§5.2): it only cancels the
    // token; ChildCleanup::drop does the teardown once the loop returns.
    assert!(ma.close_agent("root/w").unwrap().closed);
    assert!(
        !ma.close_agent("root/w").unwrap().closed,
        "second close on a cancelled token reports not found"
    );

    poll_until("normal exit", || {
        limiter.current() == 0 && registry.lock().unwrap().count() == 0
    })
    .await;

    // close→wait contract (§5.2): after full teardown the waiter observes
    // the terminal state — the closed result was posted (K1 fix) and the
    // registry/mailbox misses synthesize "closed" (§9.2/K3).
    let r = ma.wait_for_result(Some("root/w"), 500).await;
    assert_eq!(r.status, "closed");
    assert_eq!(r.agent_path.as_deref(), Some("root/w"));
    assert!(!mailbox.contains(&path));
    assert_eq!(limiter.current(), 0);
    assert_eq!(registry.lock().unwrap().count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn child_cleanup_panic_releases_all_resources() {
    let ma = make_ma_runtime_concurrent(4, Arc::new(PanickingLlm));
    let limiter = Arc::clone(&ma.limiter);
    let registry = Arc::clone(&ma.registry);

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap();
    assert_eq!(limiter.current(), 1);

    ma.send_task("root/w", "work".to_string(), false).unwrap();
    // The provider panics inside run_turn; the panic unwinds through the
    // task future and ChildCleanup::drop fires during the unwind — the
    // same single path as normal exit, no explicit panic hook needed
    // (review B-2: three paths converge into one).
    poll_until("panic unwind", || {
        limiter.current() == 0 && registry.lock().unwrap().count() == 0
    })
    .await;

    // No result was ever posted (the panic beat the post), so wait must
    // still reach a terminal verdict via the registry synthesis — a
    // waiter must never hang on a dead child (§9.2 K3).
    let r = ma.wait_for_result(Some("root/w"), 500).await;
    assert_eq!(r.status, "closed");
}

#[tokio::test(flavor = "multi_thread")]
async fn child_cleanup_parent_drop_releases_all_resources() {
    let ma = make_ma_runtime_concurrent(2, Arc::new(HangingLlm));
    let limiter = Arc::clone(&ma.limiter);
    let registry = Arc::clone(&ma.registry);
    let mailbox = Arc::clone(&ma.mailbox);
    let path = AgentPath::root().join("w");

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap();
    ma.send_task("root/w", "work".to_string(), false).unwrap();
    // Let the child enter run_turn (it hangs in the provider's stream):
    // Drop(MultiAgentRuntime) then has no choice but to abort the future
    // mid-await — the JoinSet-abort path, the third cleanup route.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(limiter.current(), 1);

    drop(ma);

    // No runtime handle survives the drop (by construction — the last Arc
    // died), so "wait receives Closed" cannot be called here; it is
    // asserted in the normal-exit and panic tests above. What abort must
    // guarantee is identical conservation: the guard ran, everything
    // released.
    poll_until("parent-drop abort", || limiter.current() == 0).await;
    assert_eq!(registry.lock().unwrap().count(), 0);
    assert!(!mailbox.contains(&path));
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrency_gate_failure_rolls_back_registry() {
    let ma = make_ma_runtime_concurrent(1, Arc::new(StreamingStub));

    ma.spawn_child("a", "p".to_string(), 0, false, vec![])
        .await
        .unwrap();

    let err = ma
        .spawn_child("b", "p".to_string(), 0, false, vec![])
        .await
        .unwrap_err();
    assert!(
        err.contains("max concurrency reached"),
        "unexpected error: {err}"
    );

    // §3.3 "undo in reverse order on any failure": a rejected gate-3 slot must undo the
    // gate-2 registry entry and leak no mailbox.
    assert_eq!(ma.limiter.current(), 1, "failed spawn holds no extra slot");
    assert_eq!(ma.registry.lock().unwrap().count(), 1);
    assert!(!ma.mailbox.contains(&AgentPath::root().join("b")));
}
