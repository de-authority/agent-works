use super::*;

// ── child permission mode ──

fn make_runtime_full(
    mode: ChildPermissionMode,
    policy: Option<Arc<dyn ToolPolicy>>,
) -> Arc<MultiAgentRuntime> {
    make_runtime_full_with_approval(mode, policy, None)
}

fn make_runtime_full_with_approval(
    mode: ChildPermissionMode,
    policy: Option<Arc<dyn ToolPolicy>>,
    approval: Option<Arc<dyn ApprovalHandler>>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        child_permission_mode: mode,
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(StreamingStub),
        vec![],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        policy,
        approval,
    ))
}

#[test]
fn effective_permission_respects_mode() {
    let full = make_runtime_full(ChildPermissionMode::Full, None);
    assert!(full.effective_permission(false));
    assert!(full.effective_permission(true));

    let none = make_runtime_full(ChildPermissionMode::None, None);
    assert!(!none.effective_permission(false));
    assert!(!none.effective_permission(true));

    let per_spawn = make_runtime_full(ChildPermissionMode::PerSpawn, None);
    assert!(per_spawn.effective_permission(true));
    assert!(!per_spawn.effective_permission(false));
}

#[tokio::test]
async fn build_child_runtime_full_carries_no_policy() {
    let ma = make_ma_runtime();
    let child = ma
        .build_child_runtime("prompt".to_string(), true)
        .await
        .expect("build child");
    assert!(child.tool_policy().is_none());
}

#[tokio::test]
async fn build_child_runtime_none_falls_back_to_deny_all() {
    // Parent has no tool policy → child falls back to DenyAllToolPolicy.
    let ma = make_ma_runtime();
    let child = ma
        .build_child_runtime("prompt".to_string(), false)
        .await
        .expect("build child");
    assert!(child.tool_policy().is_some());
}

#[tokio::test]
async fn build_child_runtime_none_inherits_parent_policy() {
    // Parent has a tool policy → child inherits the same allocation.
    let parent_policy: Arc<dyn ToolPolicy> = Arc::new(DenyAllToolPolicy);
    let ma = make_runtime_full(ChildPermissionMode::None, Some(parent_policy.clone()));
    let child = ma
        .build_child_runtime("prompt".to_string(), false)
        .await
        .expect("build child");
    let child_policy = child.tool_policy().expect("child should carry a policy");
    assert!(Arc::ptr_eq(&parent_policy, child_policy));
}

#[tokio::test]
async fn build_child_runtime_none_delegates_to_parent_approval_handler() {
    // Codex-style: a restricted child routes the approval decision up to the
    // parent's handler (human-in-the-loop / auto) rather than hard-denying
    // locally. This is what makes `ask` mode coherent for sub-agents.
    //
    // D4: since the wrapping `SourcedApprovalHandler` sits between the child
    // and the parent handler, the delegation is asserted behaviourally —
    // approving through the child's handler must reach the parent handler
    // with the source stamped (child path), and the decision must come back.
    struct RecordingParent {
        seen: std::sync::Mutex<Vec<agent_base::ApprovalRequest>>,
    }

    #[async_trait::async_trait]
    impl ApprovalHandler for RecordingParent {
        async fn approve(
            &self,
            request: agent_base::ApprovalRequest,
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> agent_base::AgentResult<agent_base::ApprovalDecision> {
            self.seen.lock().unwrap().push(request);
            Ok(agent_base::ApprovalDecision::AllowOnce)
        }
    }

    let parent = Arc::new(RecordingParent {
        seen: Default::default(),
    });
    let parent_handler: Arc<dyn ApprovalHandler> = parent.clone();
    let ma = make_runtime_full_with_approval(
        ChildPermissionMode::None,
        None,
        Some(parent_handler.clone()),
    );
    let child = ma
        .build_child_runtime("prompt".to_string(), false)
        .await
        .expect("build child");
    let child_handler = child
        .approval_handler()
        .expect("child should carry an approval handler");
    let decision = child_handler
        .approve(
            agent_base::ApprovalRequest {
                title: "write_file".into(),
                message: "m".into(),
                action_key: None,
                risk_level: agent_base::RiskLevel::Sensitive,
                raw: None,
                source: None,
            },
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("delegation resolves");
    assert!(matches!(decision, agent_base::ApprovalDecision::AllowOnce));
    let seen = parent.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "request must reach the parent handler");
    assert_eq!(
        seen[0].source.as_deref(),
        Some("root/test-child"),
        "D4: wrapper stamps the child path as source"
    );
}

#[tokio::test]
async fn build_child_runtime_none_denies_when_parent_has_no_handler() {
    // Parent carries no handler → the child must remain read-only (DenyAll),
    // preserving the "no policy, no handler → nothing can happen" invariant.
    let ma = make_runtime_full(ChildPermissionMode::None, None);
    let child = ma
        .build_child_runtime("prompt".to_string(), false)
        .await
        .expect("build child");
    let child_handler = child
        .approval_handler()
        .expect("child should carry a fallback DenyAll handler");
    let decision = child_handler
        .approve(
            agent_base::ApprovalRequest {
                title: "x".into(),
                message: "x".into(),
                action_key: None,
                risk_level: agent_base::RiskLevel::Sensitive,
                raw: None,
                source: None,
            },
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("DenyAll approves without error");
    assert_eq!(decision, agent_base::ApprovalDecision::Deny);
}

#[tokio::test]
async fn build_child_runtime_excludes_root_level_tools() {
    struct NoopDecomposeTool;
    #[async_trait::async_trait]
    impl Tool for NoopDecomposeTool {
        fn name(&self) -> &'static str {
            "decompose"
        }

        fn description(&self) -> &'static str {
            "split a task into parallel slices"
        }

        fn schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }

        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &agent_base::ToolContext,
        ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
            Ok(vec![agent_base::Content::text("serial")])
        }
    }

    let config = MultiAgentConfig {
        child_excluded_tools: vec!["decompose".to_string()],
        ..MultiAgentConfig::default()
    };
    let ma = Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(StreamingStub),
        vec![Arc::new(NoopDecomposeTool), Arc::new(NoopReadFileTool)],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ));

    let child = ma
        .build_child_runtime("prompt".to_string(), true)
        .await
        .expect("build child");

    let registry = child.tools_mut();
    let registry = registry.read().await;
    assert!(
        registry.get("decompose").is_none(),
        "decompose must be excluded from child runtimes"
    );
    assert!(
        registry.get("read_file").is_some(),
        "non-excluded tools must still be inherited"
    );
}

// ── SourcedApprovalHandler (design 2026-09-19 D4) ──

/// D4: `SourcedApprovalHandler` fills in `source`, never overwrites an existing
/// value, and passes `Err` through.
mod sourced_handler {
    use std::sync::Arc;

    use agent_base::{ApprovalDecision, ApprovalHandler, ApprovalRequest, RiskLevel};

    /// Records the requests it receives; returns `Err` when `fail` is true
    /// (tests pass-through).
    struct RecordingHandler {
        seen: std::sync::Mutex<Vec<ApprovalRequest>>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl ApprovalHandler for RecordingHandler {
        async fn approve(
            &self,
            request: ApprovalRequest,
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> agent_base::AgentResult<ApprovalDecision> {
            self.seen.lock().unwrap().push(request);
            if self.fail {
                Err(agent_base::AgentError::internal("boom"))
            } else {
                Ok(ApprovalDecision::AllowOnce)
            }
        }
    }

    fn request() -> ApprovalRequest {
        ApprovalRequest {
            title: "write_file".into(),
            message: "m".into(),
            action_key: None,
            risk_level: RiskLevel::Sensitive,
            raw: None,
            source: None,
        }
    }

    fn handler() -> Arc<RecordingHandler> {
        Arc::new(RecordingHandler {
            seen: Default::default(),
            fail: false,
        })
    }

    #[tokio::test]
    async fn fills_source_when_absent() {
        let inner = handler();
        let h = crate::multi_agent::runtime::build::SourcedApprovalHandler {
            inner: inner.clone(),
            source: "root/coder-1".to_string(),
        };
        let d = h
            .approve(request(), tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        assert!(matches!(d, ApprovalDecision::AllowOnce));
        assert_eq!(
            inner.seen.lock().unwrap()[0].source.as_deref(),
            Some("root/coder-1")
        );
    }

    #[tokio::test]
    async fn never_overwrites_existing_source() {
        let inner = handler();
        let h = crate::multi_agent::runtime::build::SourcedApprovalHandler {
            inner: inner.clone(),
            source: "root/coder-1".to_string(),
        };
        let mut req = request();
        req.source = Some("root/other".into());
        h.approve(req, tokio_util::sync::CancellationToken::new())
            .await
            .unwrap();
        // The inner handler still sees the original value — the wrapper only
        // fills in when it is None.
        assert_eq!(
            inner.seen.lock().unwrap()[0].source.as_deref(),
            Some("root/other")
        );
    }

    #[tokio::test]
    async fn passes_through_inner_decision_and_error() {
        let inner = Arc::new(RecordingHandler {
            seen: Default::default(),
            fail: true,
        });
        let h = crate::multi_agent::runtime::build::SourcedApprovalHandler {
            inner,
            source: "root/x".to_string(),
        };
        let out = h
            .approve(request(), tokio_util::sync::CancellationToken::new())
            .await;
        assert!(out.is_err(), "inner Err must propagate");
    }
}
