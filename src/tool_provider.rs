//! Tool supply contract (ToolProvider) — the socket that every tool source plugs into.
//!
//! **P0 is the contract layer**: only signatures and the identity-key spec, no implementation.
//! The four implementations (BuiltinProvider / WasmProvider / ScriptProvider / McpProvider)
//! land in later phases, all plugging into the same tool table (unified identity / approval / budget / switches).
//!
//! # Identity key contract (fixed in one place)
//!
//! | Source | Identity key | Example |
//! |---|---|---|
//! | Builtin tools (present at compile time) | bare name like `read_file` | `read_file` |
//! | MCP tools from `/connect` | `mcp.{server}.{tool}` | `mcp.figma.get_file` |
//! | Global script tools | `script.{name}` | `script.deploy` |
//! | All tools contributed by plugins (wasm/script/mcp forms unified) | `plugin.{plugin}.{tool}` | `plugin.my-tools.search` |
//!
//! - **Identity is determined by "ownership surface", not by "transport"**: an MCP server
//!   inside a plugin is an implementation detail of the plugin; the tool key remains
//!   `plugin.{plugin}.{tool}` (the tool segment = the server's tool name).
//! - The `plugin.` namespace is a single prefix — sub-agent exclusion, approval ledger
//!   matching, and fail-closed classification all go by the three prefixes
//!   `mcp.` / `script.` / `plugin.`.
//! - Name character set `[a-z0-9_-]{1,32}`; `mcp`/`script`/`plugin` are reserved words
//!   and must not be used as any name segment (hitting a reserved word → reject).
//!
//! # Approval priority table (normative MUST; for the implementation see the approval ledger phase)
//!
//! | Scenario | Behavior |
//! |---|---|
//! | Prefix `mcp.`/`script.`/`plugin.`, tier `ask every time` | **Prompt, even with Shift+Tab = auto** (ledger before mode) |
//! | Same as above, tier `always allow` | Allow |
//! | Same as above, tier `deny` | Deny |
//! | Shift+Tab = deny (global brake) | Deny everything, overrides the ledger |
//! | Builtin tools | Not in the ledger; go through the existing three-layer approval |
//!
//! Three hard rules:
//! 1. **fail-closed**: the classification fallback MUST return "needs confirmation" for the
//!    three prefixes, never auto-allow;
//! 2. **Single memory owner**: `always allow` is only recorded in approvals.json, never
//!    double-written with the builtin tools' action_key cache;
//! 3. **Test criterion**: an external tool call at tier `ask every time` still prompts under Shift+Tab=auto.
//!
//! # Contract invariants (every implementation MUST satisfy them)
//!
//! - `call` MUST reject identity keys outside its own namespace (fail-closed, no "helpful" forwarding);
//! - the host MUST validate `args` against the descriptor's `parameters` before entering `call` —
//!   wasm never sees unvalidated parameters;
//! - keep the trait surface small and narrow: better to give too little than to give the wrong thing.
//!   To add capability, change the contract first, then implement.

use agent_base::{AgentResult, Content, ToolContext, ToolExposure};
use async_trait::async_trait;
use serde_json::Value;

/// Max length of a name segment (`[a-z0-9_-]{1,32}`).
pub const NAME_SEGMENT_MAX: usize = 32;

/// Reserved words: must not be used as any identity-key name segment (plugin names / script names / builtin names are all rejected).
pub const RESERVED_NAMESPACES: [&str; 3] = ["mcp", "script", "plugin"];

/// Identity key error. Construction and parsing are always fail-closed: report an error when invalid, never settle for less.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityError {
    /// Name segment is empty, too long, or contains characters outside `[a-z0-9_-]`.
    BadSegment { segment: String },
    /// Name collides with a reserved word `mcp` / `script` / `plugin`.
    ReservedName { name: String },
    /// Free-form key does not match any of the four identity key syntaxes.
    MalformedKey { key: String },
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadSegment { segment } => write!(
                f,
                "identity segment `{segment}` must match [a-z0-9_-]{{1,{NAME_SEGMENT_MAX}}}"
            ),
            Self::ReservedName { name } => {
                write!(
                    f,
                    "`{name}` is a reserved identity namespace (mcp/script/plugin)"
                )
            }
            Self::MalformedKey { key } => write!(
                f,
                "`{key}` is not a valid tool identity key (builtin name, mcp.{{server}}.{{tool}}, script.{{name}}, or plugin.{{plugin}}.{{tool}})"
            ),
        }
    }
}

impl std::error::Error for IdentityError {}

/// Identity on a tool table. The four shapes cover all sources; serializes to the keys in the table above.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ToolIdentity {
    /// Builtin tool bare name (`read_file`). Not among the three prefixes, not in the approval ledger.
    Builtin(String),
    /// MCP tool from `/connect`: `mcp.{server}.{tool}`.
    Mcp { server: String, tool: String },
    /// Global script tool: `script.{name}`.
    Script { name: String },
    /// Tool contributed by a plugin (wasm/script/mcp forms unified): `plugin.{plugin}.{tool}`.
    Plugin { plugin: String, tool: String },
}

fn validate_segment(segment: &str) -> Result<(), IdentityError> {
    let ok = !segment.is_empty()
        && segment.len() <= NAME_SEGMENT_MAX
        && segment
            .chars()
            .all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(IdentityError::BadSegment {
            segment: segment.to_string(),
        })
    }
}

fn validate_name(name: &str) -> Result<(), IdentityError> {
    validate_segment(name)?;
    if RESERVED_NAMESPACES.contains(&name) {
        return Err(IdentityError::ReservedName {
            name: name.to_string(),
        });
    }
    Ok(())
}

impl ToolIdentity {
    /// Builtin tool: bare name, not in the ledger.
    pub fn builtin(name: &str) -> Result<Self, IdentityError> {
        validate_name(name)?;
        Ok(Self::Builtin(name.to_string()))
    }

    /// MCP tool key from `/connect`. The `tool` segment should be the normalized server
    /// tool name (see [`normalize_tool_segment`]).
    pub fn mcp(server: &str, tool: &str) -> Result<Self, IdentityError> {
        validate_segment(server)?;
        validate_segment(tool)?;
        Ok(Self::Mcp {
            server: server.to_string(),
            tool: tool.to_string(),
        })
    }

    /// Global script tool key.
    pub fn script(name: &str) -> Result<Self, IdentityError> {
        validate_name(name)?;
        Ok(Self::Script {
            name: name.to_string(),
        })
    }

    /// Plugin tool key (wasm/script/mcp forms unified). A `plugin` segment hitting a
    /// reserved word → reject (the same rule at install time).
    pub fn plugin(plugin: &str, tool: &str) -> Result<Self, IdentityError> {
        validate_name(plugin)?;
        validate_segment(tool)?;
        Ok(Self::Plugin {
            plugin: plugin.to_string(),
            tool: tool.to_string(),
        })
    }

    /// Parse a display key. Grammar (the number of dot-separated segments determines the
    /// ownership surface; segments contain no `.`):
    /// no dots = builtin; `mcp.s.t`; `script.n`; `plugin.p.t`.
    pub fn parse(key: &str) -> Result<Self, IdentityError> {
        match key.split('.').collect::<Vec<_>>().as_slice() {
            [name] => Self::builtin(name),
            ["mcp", server, tool] => Self::mcp(server, tool),
            ["script", name] => Self::script(name),
            ["plugin", plugin, tool] => Self::plugin(plugin, tool),
            _ => Err(IdentityError::MalformedKey {
                key: key.to_string(),
            }),
        }
    }

    /// Canonical display key (shared by LLM tool names / ledger keys / the `/tools` panel).
    pub fn to_key(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for ToolIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Builtin(name) => f.write_str(name),
            Self::Mcp { server, tool } => write!(f, "mcp.{server}.{tool}"),
            Self::Script { name } => write!(f, "script.{name}"),
            Self::Plugin { plugin, tool } => write!(f, "plugin.{plugin}.{tool}"),
        }
    }
}

/// Normalization of an MCP server tool name into the identity key's tool segment:
/// lowercase, each character outside `[a-z0-9_-]` replaced with `-`, truncated to 32 **characters**.
///
/// Numeric suffixes for post-normalization name collisions (`search` / `search-2` …) are a
/// load-time registration algorithm, not part of this contract. The result of an empty
/// string / all-invalid input cannot be used as an identity segment; rejected at load time.
pub fn normalize_tool_segment(raw: &str) -> String {
    raw.to_lowercase()
        .chars()
        .map(|c| match c {
            'a'..='z' | '0'..='9' | '_' | '-' => c,
            _ => '-',
        })
        .take(NAME_SEGMENT_MAX)
        .collect()
}

/// One row on the tool table: identity + model-visible description and parameter envelope.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    pub identity: ToolIdentity,
    /// Model-visible description.
    pub description: String,
    /// Parameter JSON Schema (the shape of MCP `inputSchema`). The plugin manifest's
    /// simplified `parameters` table is converted into this envelope at registration;
    /// it is what the host uses to validate `args`.
    pub parameters: Value,
    /// Hard timeout (milliseconds). `None` = framework default.
    pub timeout_ms: Option<u64>,
    /// Tool-table budget tier, default [`ToolExposure::Direct`].
    pub exposure: ToolExposure,
}

/// Tool supply abstraction: `list` / `call` / `health`; for the identity key spec see the module docs.
///
/// The four implementations plug into the same tool table; future new implementations such
/// as openapi won't break it — the dividing criterion is: "independently distributable
/// third-party capability / written by the users themselves / language-neutral" goes to the
/// plugin supply, "needs in-process state / high-frequency low-latency / security base" stays
/// builtin (but still implements this trait).
#[async_trait]
pub trait ToolProvider: Send + Sync {
    /// Enumerate the currently exposed tools. Identity keys MUST already be fully qualified ([`ToolIdentity`]).
    ///
    /// Dynamic sources (MCP discovery, plugin loading) may differ on each call; callers must
    /// not cache beyond one health cycle.
    async fn list(&self) -> AgentResult<Vec<ToolDescriptor>>;

    /// Execute a tool by identity key.
    ///
    /// - `identity` MUST be a key within this provider's own namespace (one that `list`
    ///   returned), otherwise reject (fail-closed, no "helpful" forwarding);
    /// - `args` MUST already have been validated by the host against [`ToolDescriptor::parameters`];
    /// - a cancelled/timed-out call only takes down the tool call, not the host.
    async fn call(
        &self,
        identity: &ToolIdentity,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<Vec<Content>>;

    /// Cheap liveness probe (MCP reconnection, wasm runtime, whether the script is still
    /// around). On failure, returns an error string explaining what is broken; never panics.
    async fn health(&self) -> AgentResult<()>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use serde_json::json;

    #[test]
    fn identity_display_and_parse_round_trip() {
        let cases = vec![
            ToolIdentity::builtin("read_file").unwrap(),
            ToolIdentity::mcp("figma", "get_file").unwrap(),
            ToolIdentity::script("deploy").unwrap(),
            ToolIdentity::plugin("my-tools", "search").unwrap(),
        ];
        let keys: Vec<String> = cases.iter().map(|c| c.to_key()).collect();
        assert_eq!(
            keys,
            vec![
                "read_file",
                "mcp.figma.get_file",
                "script.deploy",
                "plugin.my-tools.search"
            ]
        );
        for case in &cases {
            assert_eq!(ToolIdentity::parse(&case.to_key()).unwrap(), *case);
        }
    }

    #[test]
    fn identity_rejects_reserved_names() {
        for ctor in [
            ToolIdentity::builtin("mcp"),
            ToolIdentity::script("script"),
            ToolIdentity::plugin("plugin", "x"),
        ] {
            assert!(matches!(
                ctor,
                Err(IdentityError::ReservedName { name }) if name == "mcp" || name == "script" || name == "plugin"
            ));
        }
        // server / tool segments do not block reserved words: `mcp.mcp.get` is unambiguous syntax (not required by the design).
        assert!(ToolIdentity::mcp("mcp", "get").is_ok());
    }

    #[test]
    fn identity_rejects_bad_segments() {
        // Uppercase, empty segments, over 32 chars, containing dots/spaces/non-ASCII — all rejected.
        assert!(matches!(
            ToolIdentity::script("Deploy"),
            Err(IdentityError::BadSegment { .. })
        ));
        assert!(matches!(
            ToolIdentity::plugin("my tools", "x"),
            Err(IdentityError::BadSegment { .. })
        ));
        assert!(matches!(
            ToolIdentity::builtin(""),
            Err(IdentityError::BadSegment { .. })
        ));
        let long = "a".repeat(NAME_SEGMENT_MAX + 1);
        assert!(matches!(
            ToolIdentity::builtin(&long),
            Err(IdentityError::BadSegment { .. })
        ));
        assert!(matches!(
            ToolIdentity::mcp("notion", "get.file"),
            Err(IdentityError::BadSegment { .. })
        ));
        assert!(matches!(
            ToolIdentity::builtin("搜索"),
            Err(IdentityError::BadSegment { .. })
        ));
    }

    #[test]
    fn identity_segment_boundary_is_32() {
        let ok = "a".repeat(NAME_SEGMENT_MAX);
        assert!(ToolIdentity::builtin(&ok).is_ok());
    }

    #[test]
    fn parse_rejects_malformed_keys() {
        for key in [
            "mcp.onlyonesegment",
            "plugin.onlyonesegment",
            "mcp.a.b.c",
            "a.b",
            "script.a.b",
            "mcp..tool",
            "mcp.server.",
            ".server.tool",
        ] {
            assert!(
                matches!(
                    ToolIdentity::parse(key),
                    Err(IdentityError::MalformedKey { .. } | IdentityError::BadSegment { .. })
                ),
                "expected malformed/bad segment for {key:?}"
            );
        }
    }

    #[test]
    fn normalize_tool_segment_follows_three_rules() {
        // Lowercase, invalid chars → `-`, truncate to 32 characters.
        assert_eq!(normalize_tool_segment("Get File!"), "get-file-");
        assert_eq!(normalize_tool_segment("GET"), "get");
        assert_eq!(normalize_tool_segment("a.b:c"), "a-b-c");
        // Truncation counts characters; multibyte characters don't panic.
        let long = "x".repeat(40);
        assert_eq!(normalize_tool_segment(&long).len(), NAME_SEGMENT_MAX);
        let cjk = "搜".repeat(40);
        assert_eq!(normalize_tool_segment(&cjk), "-".repeat(NAME_SEGMENT_MAX));
    }

    // ── behavioral contract of the trait surface itself ─────────────────────────────────────────────

    struct EchoProvider;

    #[async_trait]
    impl ToolProvider for EchoProvider {
        async fn list(&self) -> AgentResult<Vec<ToolDescriptor>> {
            Ok(vec![ToolDescriptor {
                identity: ToolIdentity::script("echo").unwrap(),
                description: "echoes".into(),
                parameters: json!({"type": "object", "properties": {}}),
                timeout_ms: Some(1_000),
                exposure: ToolExposure::Direct,
            }])
        }

        async fn call(
            &self,
            identity: &ToolIdentity,
            args: &Value,
            _ctx: &ToolContext,
        ) -> AgentResult<Vec<Content>> {
            // Contract invariant: only accept keys in our own namespace, reject foreign keys.
            if identity.to_key() != "script.echo" {
                return Err(agent_base::AgentError::ToolNotFound {
                    name: identity.to_key(),
                });
            }
            Ok(vec![Content::text(args.to_string())])
        }

        async fn health(&self) -> AgentResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn provider_is_object_safe_and_round_trips() {
        let provider: Arc<dyn ToolProvider> = Arc::new(EchoProvider);
        let listed = provider.list().await.unwrap();
        assert_eq!(listed[0].identity.to_key(), "script.echo");

        let ctx = ToolContext::for_test();
        let out = provider
            .call(
                &ToolIdentity::script("echo").unwrap(),
                &json!({"x": 1}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        provider.health().await.unwrap();
    }

    #[tokio::test]
    async fn provider_call_rejects_foreign_identity() {
        let provider: Arc<dyn ToolProvider> = Arc::new(EchoProvider);
        let ctx = ToolContext::for_test();
        let err = provider
            .call(
                &ToolIdentity::plugin("my-tools", "search").unwrap(),
                &json!({}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, agent_base::AgentError::ToolNotFound { .. }));
    }
}
