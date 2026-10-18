use super::*;

// ── tool-socket P3: three-prefix external-tool spawn exclusion (design "child agent" section) ──
//
// "Child agents must not reach external tools by default": `child_excluded_tools`
// is a startup-time static exact-name set, which cannot cover dynamic names
// (`mcp.{server}.{tool}` / `script.{name}` / `plugin.{p}.{t}`), so spawn-time
// exclusion goes by the three prefixes. The criterion is the same source as the
// approval-ledger matching and the fail-closed classification
// (`is_external_identity_key` — the single legal encoding point for the three
// prefixes). Explicit allow-listing comes later.

/// A business tool whose *name* is its identity key (the tool itself is a no-op).
struct NamedNoopTool(&'static str);

#[async_trait::async_trait]
impl Tool for NamedNoopTool {
    fn name(&self) -> &'static str {
        self.0
    }
    fn description(&self) -> &'static str {
        "noop external-shaped tool"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &agent_base::ToolContext,
    ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
        Ok(vec![agent_base::Content::text("ok")])
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn external_tools_never_reach_children() {
    // Registration loop: one tool per prefix is present in `business_tools`,
    // yet the child receives only the builtin — prefix match, not name list.
    let ma = make_runtime_with_tools(
        vec![
            Arc::new(NoopReadFileTool),
            Arc::new(NamedNoopTool("mcp.demo.search")),
            Arc::new(NamedNoopTool("script.deploy")),
            Arc::new(NamedNoopTool("plugin.pkg.tool")),
        ],
        vec![],
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("spawn");
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
    assert!(!spawned.spawned_tools().contains("mcp.demo.search"));
    assert!(!spawned.spawned_tools().contains("script.deploy"));
    assert!(!spawned.spawned_tools().contains("plugin.pkg.tool"));
}

#[tokio::test(flavor = "multi_thread")]
async fn full_permission_child_still_excluded() {
    // `full_permission` is about approvals, not the tool surface — the
    // external exclusion is unconditional.
    let ma = make_runtime_with_tools(
        vec![
            Arc::new(NoopReadFileTool),
            Arc::new(NamedNoopTool("script.deploy")),
        ],
        vec![],
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                full_permission: Some(true),
                ..Default::default()
            },
        )
        .await
        .expect("spawn");
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
    assert!(!spawned.spawned_tools().contains("script.deploy"));
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelist_external_name_warns_and_drops() {
    // Whitelist cannot smuggle an external tool in: the name is in
    // `business_tools`, but it is policy-excluded (case 2) → warn + drop,
    // spawn proceeds with the reduced set (echo is the truth).
    let ma = make_runtime_with_tools(
        vec![
            Arc::new(NoopReadFileTool),
            Arc::new(NamedNoopTool("script.deploy")),
        ],
        vec![],
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                tool_names: Some(BTreeSet::from([
                    "read_file".to_string(),
                    "script.deploy".to_string(),
                ])),
                ..Default::default()
            },
        )
        .await
        .expect("external-in-whitelist spawns with reduced set");
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
    assert!(!spawned.spawned_tools().contains("script.deploy"));
}

#[tokio::test(flavor = "multi_thread")]
async fn whitelist_unknown_external_name_is_policy_exclusion_not_error() {
    // Categorical rule: any external key in a whitelist is "excluded by
    // deployment", whether or not it resolves in the source list — the
    // boundary is the prefix, not the presence. (Non-external unknown names
    // still fail loud with `ToolNotFound` — see the whitelist scenario.)
    let ma = make_runtime_with_tools(vec![Arc::new(NoopReadFileTool)], vec![]);
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("prompt".to_string()),
                tool_names: Some(BTreeSet::from([
                    "read_file".to_string(),
                    "mcp.no.such.tool".to_string(),
                ])),
                ..Default::default()
            },
        )
        .await
        .expect("unknown external key is a policy exclusion, not ToolNotFound");
    assert_eq!(spawned.spawned_tools().len(), 1);
    assert!(spawned.spawned_tools().contains("read_file"));
}
