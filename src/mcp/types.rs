use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// MCP transport.
///
/// The serde shape = the shape of `connections.json` (design "connections.json entry"):
/// `{"type":"stdio","command":…,"args":[…],"env":{…}}` /
/// `{"type":"http","url":…,"headers":{…}}`.
/// `env` / `headers` may be omitted (default empty); `${ENV:VAR}` in values is
/// pre-resolved to literals by the product layer at connection setup
/// (the framework accepts literals only).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpTransport {
    /// Remote HTTP. `headers` are extra request headers (Authorization etc.).
    Http {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
    /// Local stdio process. `env` is applied on top of the inherited process env
    /// (the MCP server's token port).
    Stdio {
        command: String,
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
}

impl McpTransport {
    /// HTTP transport (headers attached separately).
    pub fn http(url: impl Into<String>) -> Self {
        Self::Http {
            url: url.into(),
            headers: BTreeMap::new(),
        }
    }

    /// stdio transport (env attached separately).
    pub fn stdio(
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self::Stdio {
            command: command.into(),
            args: args.into_iter().map(Into::into).collect(),
            env: BTreeMap::new(),
        }
    }

    /// Attach a request header (only meaningful for [`McpTransport::Http`]).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        if let Self::Http { headers, .. } = &mut self {
            headers.insert(name.into(), value.into());
        }
        self
    }

    /// Attach an environment variable override (only meaningful for
    /// [`McpTransport::Stdio`]).
    pub fn with_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        if let Self::Stdio { env, .. } = &mut self {
            env.insert(name.into(), value.into());
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
    #[serde(default)]
    pub auto_reconnect: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn transport_serde_matches_connections_json_shape() {
        // stdio shape (design sample: npx + env port)
        let stdio: McpTransport = serde_json::from_value(json!({
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": { "NOTION_TOKEN": "tok-1" }
        }))
        .unwrap();
        match &stdio {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "npx");
                assert_eq!(args, &["-y", "@notionhq/notion-mcp-server"]);
                assert_eq!(env["NOTION_TOKEN"], "tok-1");
            }
            other => panic!("wrong variant: {other:?}"),
        }

        // http shape (headers attach an Authorization)
        let http: McpTransport = serde_json::from_value(json!({
            "type": "http",
            "url": "https://example.com/mcp",
            "headers": { "Authorization": "Bearer tok-2" }
        }))
        .unwrap();
        match &http {
            McpTransport::Http { url, headers } => {
                assert_eq!(url, "https://example.com/mcp");
                assert_eq!(headers["Authorization"], "Bearer tok-2");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn transport_serde_tolerates_missing_env_and_headers() {
        // Old samples have no env/headers field → empty map (must not blow up)
        let stdio: McpTransport = serde_json::from_value(json!({
            "type": "stdio", "command": "npx", "args": []
        }))
        .unwrap();
        assert!(matches!(stdio, McpTransport::Stdio { env, .. } if env.is_empty()));

        let http: McpTransport =
            serde_json::from_value(json!({"type": "http", "url": "https://x"})).unwrap();
        assert!(matches!(http, McpTransport::Http { headers, .. } if headers.is_empty()));
    }

    #[test]
    fn transport_serde_round_trips() {
        let t = McpTransport::stdio("npx", ["-y", "pkg"]).with_env("TOKEN", "v");
        let raw = serde_json::to_value(&t).unwrap();
        assert_eq!(raw["type"], "stdio");
        assert_eq!(raw["env"]["TOKEN"], "v");
        let back: McpTransport = serde_json::from_value(raw).unwrap();
        assert_eq!(back, t);

        let h = McpTransport::http("https://x").with_header("Authorization", "Bearer v");
        let back: McpTransport = serde_json::from_value(serde_json::to_value(&h).unwrap()).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn header_and_env_builders_only_touch_their_variant() {
        // with_header is a no-op for stdio, with_env is a no-op for http
        // (no blow-up, no cross-talk).
        let t = McpTransport::stdio("sh", ["-c", "x"]).with_header("X", "y");
        assert!(matches!(t, McpTransport::Stdio { env, .. } if env.is_empty()));
        let t = McpTransport::http("https://x").with_env("X", "y");
        assert!(matches!(t, McpTransport::Http { headers, .. } if headers.is_empty()));
    }
}
