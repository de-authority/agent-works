use std::sync::Arc;

use agent_base::{AgentResult, ToolRegistry};
use tracing::{info, warn};

use super::client::{McpClient, McpToolAdapter};
use super::enhanced_hub::ConnectionState;
use super::types::{McpServerConfig, McpToolInfo};

struct ServerEntry {
    config: McpServerConfig,
    client: Option<Arc<McpClient>>,
    tools: Vec<McpToolInfo>,
    state: ConnectionState,
}

pub struct McpHub {
    servers: Vec<ServerEntry>,
}

impl Default for McpHub {
    fn default() -> Self {
        Self::new()
    }
}

impl McpHub {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    pub fn add_server(&mut self, config: McpServerConfig) {
        self.servers.push(ServerEntry {
            config,
            client: None,
            tools: Vec::new(),
            state: ConnectionState::Disconnected,
        });
    }

    pub async fn connect_all(&mut self) -> AgentResult<()> {
        for entry in &mut self.servers {
            let client = Arc::new(McpClient::new(entry.config.transport.clone()).await?);
            entry.client = Some(client);
            entry.state = ConnectionState::Connected;
            info!("connected to MCP server: {}", entry.config.name);
        }
        Ok(())
    }

    pub async fn discover_all(&mut self) -> AgentResult<Vec<(String, Vec<McpToolInfo>)>> {
        let mut results = Vec::new();
        for entry in &mut self.servers {
            let Some(client) = &entry.client else {
                warn!("server {} not connected", entry.config.name);
                continue;
            };
            match client.list_tools().await {
                Ok(tools) => {
                    info!(
                        "discovered {} tools from {}",
                        tools.len(),
                        entry.config.name
                    );
                    let cloned = tools.clone();
                    entry.tools = tools;
                    results.push((entry.config.name.clone(), cloned));
                }
                Err(e) => {
                    warn!(
                        server_name = %entry.config.name,
                        error = %e,
                        "failed to discover tools from MCP server"
                    );
                    entry.state = ConnectionState::Failed(format!("{e}"));
                }
            }
        }
        Ok(results)
    }

    pub fn register_all(&self, registry: &mut ToolRegistry) {
        for entry in &self.servers {
            let Some(client) = &entry.client else {
                continue;
            };
            for tool_info in &entry.tools {
                let adapter =
                    McpToolAdapter::new(tool_info.clone(), client.clone(), &entry.config.name);
                registry.register(adapter);
            }
        }
    }

    pub async fn disconnect_all(&mut self) {
        for entry in &mut self.servers {
            entry.client = None;
            entry.state = ConnectionState::Disconnected;
            info!(server_name = %entry.config.name, "disconnected from MCP server");
        }
    }

    pub async fn health_check(&self) -> AgentResult<()> {
        for entry in &self.servers {
            let Some(client) = &entry.client else {
                warn!("server {} not connected", entry.config.name);
                continue;
            };
            match client.list_tools().await {
                Ok(_) => {
                    info!(server_name = %entry.config.name, "health check passed");
                }
                Err(e) => {
                    warn!(server_name = %entry.config.name, error = %e, "health check failed");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::McpTransport;
    use agent_base::ToolRegistry;
    use serde_json::json;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn http_config(name: &str, url: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::http(url),
            auto_reconnect: false,
        }
    }

    #[test]
    fn test_new_empty() {
        let hub = McpHub::new();
        assert!(hub.servers.is_empty());
    }

    #[tokio::test]
    async fn test_add_and_connect_all() {
        let mut hub = McpHub::new();
        hub.add_server(http_config("a", "http://localhost:1"));
        hub.add_server(http_config("b", "http://localhost:2"));
        hub.connect_all().await.unwrap();

        assert_eq!(hub.servers.len(), 2);
        assert!(hub.servers[0].client.is_some());
        assert!(hub.servers[1].client.is_some());
        assert!(matches!(hub.servers[0].state, ConnectionState::Connected));
    }

    #[tokio::test]
    async fn test_discover_all_and_register_all() {
        let server = MockServer::start().await;
        let mut hub = McpHub::new();
        hub.add_server(http_config("brave", &format!("{}/mcp", server.uri())));
        // Never connected — exercises the "not connected" warn branch.
        hub.add_server(http_config("unconnected", "http://localhost:1"));
        hub.connect_all().await.unwrap();

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"tools": [{"name": "search", "description": "d", "inputSchema": {}}]}
            })))
            .mount(&server)
            .await;

        let results = hub.discover_all().await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "brave");
        assert_eq!(results[0].1.len(), 1);
        assert_eq!(results[0].1[0].name, "search");

        let mut registry = ToolRegistry::default();
        hub.register_all(&mut registry);
        assert!(registry.get("mcp.brave.search").is_some());
    }

    #[tokio::test]
    async fn test_disconnect_all() {
        let mut hub = McpHub::new();
        hub.add_server(http_config("a", "http://localhost:1"));
        hub.connect_all().await.unwrap();
        hub.disconnect_all().await;

        assert!(hub.servers[0].client.is_none());
        assert!(matches!(
            hub.servers[0].state,
            ConnectionState::Disconnected
        ));
    }

    #[tokio::test]
    async fn test_health_check() {
        let server = MockServer::start().await;
        let mut hub = McpHub::new();
        hub.add_server(http_config("ok", &format!("{}/mcp", server.uri())));
        hub.add_server(http_config("noconn", "http://localhost:1"));
        hub.connect_all().await.unwrap();

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": {"tools": []}
            })))
            .mount(&server)
            .await;

        // Should not error even when one server is not connected / one responds.
        hub.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_discover_all_with_connection_error() {
        // A connected client whose list_tools fails (server returns error) should
        // be marked Failed and skipped in results, without bubbling the error.
        let server = MockServer::start().await;
        let mut hub = McpHub::new();
        hub.add_server(http_config("bad", &format!("{}/mcp", server.uri())));
        hub.connect_all().await.unwrap();

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let results = hub.discover_all().await.unwrap();
        assert!(results.is_empty());
        assert!(matches!(hub.servers[0].state, ConnectionState::Failed(_)));
    }
}
