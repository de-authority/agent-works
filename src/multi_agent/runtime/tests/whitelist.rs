use super::*;

// ── ChildConfig whitelist + spawn_with_config (§5.1 / §5.4) ──

/// A second business tool so the whitelist has something to select.
struct NoopDecomposeTool;

#[async_trait::async_trait]
impl Tool for NoopDecomposeTool {
    fn name(&self) -> &'static str {
        "decompose"
    }
    fn description(&self) -> &'static str {
        "split a task"
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

#[tokio::test(flavor = "multi_thread")]
async fn whitelist_selects_registered_subset() {
    // Case 1 (§5.4): every requested name resolves → spawn succeeds and
    // the echo reports exactly the requested subset.
    let ma = make_runtime_with_tools(
        vec![Arc::new(NoopReadFileTool), Arc::new(NoopDecomposeTool)],
        vec![],
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                tool_names: Some(BTreeSet::from(["read_file".to_string()])),
                ..Default::default()
            },
        )
        .await
        .expect("whitelist spawn");
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
    assert_eq!(spawned.agent_path(), &AgentPath::root().join("w"));
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelist_unknown_name_is_tool_not_found() {
    // Case 3 (§5.4:297): requested but neither registered nor excluded
    // (a typo / hallucinated tool) → hard ToolNotFound, fail loud.
    let ma = make_runtime_with_tools(vec![Arc::new(NoopReadFileTool)], vec![]);
    let err = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                tool_names: Some(BTreeSet::from(["grep".to_string()])),
                ..Default::default()
            },
        )
        .await
        .map(|_| ())
        .expect_err("unknown tool must fail");
    assert!(
        matches!(err, AgentError::ToolNotFound { ref name } if name == "grep"),
        "expected typed ToolNotFound, got {err:?}"
    );
    // rollback: a failed spawn leaves no residue.
    assert_eq!(ma.registry.lock().unwrap().count(), 0);
    assert_eq!(ma.limiter.current(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelist_intersection_with_excluded_warns_not_errors() {
    // Case 2 (§5.4:298, review M-3): a requested tool that the deployment
    // globally excluded → warn + drop it, spawn proceeds with the reduced
    // set (never a silent fake permission, but also not a hard error).
    let ma = make_runtime_with_tools(
        vec![Arc::new(NoopReadFileTool), Arc::new(NoopDecomposeTool)],
        vec!["decompose".to_string()],
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                tool_names: Some(BTreeSet::from([
                    "read_file".to_string(),
                    "decompose".to_string(),
                ])),
                ..Default::default()
            },
        )
        .await
        .expect("excluded-in-whitelist spawns with reduced set");
    // decompose was excluded → the echo reflects the REAL child tools.
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
    assert!(!spawned.spawned_tools().contains("decompose"));
}

#[tokio::test(flavor = "multi_thread")]
async fn spawn_with_config_requires_system_prompt() {
    // §5.1 fail-fast: empty prompt → ConfigError, before any gate.
    let ma = make_runtime_with_tools(vec![Arc::new(NoopReadFileTool)], vec![]);
    let err = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: None,
                ..Default::default()
            },
        )
        .await
        .map(|_| ())
        .expect_err("empty prompt must fail");
    assert!(
        err.to_string().contains("system_prompt is required"),
        "unexpected error: {err}"
    );
    assert_eq!(ma.limiter.current(), 0, "no gate reached before validation");
}

#[tokio::test(flavor = "multi_thread")]
async fn spawn_with_config_wires_max_turns_and_runs() {
    // max_turns / context_window are accepted by agent-base's builder
    // (execution_max_turns/context_window); smoke-test that a child
    // built with them still executes a task normally.
    let ma = make_runtime_with_tools(vec![Arc::new(NoopReadFileTool)], vec![]);
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                max_turns: Some(8),
                context_window: Some(4096),
                ..Default::default()
            },
        )
        .await
        .expect("spawn with overrides");
    let path = spawned.agent_path().to_string();
    ma.send_task(&path, "work".to_string(), false).unwrap();
    let r = ma.wait_for_result(Some(&path), 2000).await;
    assert_eq!(r.status, "ok");
}
