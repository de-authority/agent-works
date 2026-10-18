//! plugin.json schema v1 sample validation tests (tool-socket contract P0 acceptance exit).
//!
//! Criteria: the samples under schemas/examples/ pass validation; each
//! counter-example from the design doc is rejected.
//! The validation engine consumes `schemas/plugin.schema.json` directly — the
//! test targets **that file**, not a reimplementation of it.

use serde_json::{Value, json};

fn schema() -> Value {
    serde_json::from_str(include_str!("../schemas/plugin.schema.json"))
        .expect("plugin.schema.json must be valid JSON")
}

fn example() -> Value {
    serde_json::from_str(include_str!("../schemas/examples/plugin-example.json"))
        .expect("plugin-example.json must be valid JSON")
}

fn assert_valid(instance: &Value) {
    let validator = jsonschema::validator_for(&schema()).expect("schema must compile");
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "expected valid, got errors: {errors:#?}");
}

fn assert_invalid(instance: &Value) {
    let validator = jsonschema::validator_for(&schema()).expect("schema must compile");
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| e.to_string())
        .collect();
    assert!(
        !errors.is_empty(),
        "expected invalid, but schema accepted: {instance}"
    );
}

/// Minimal wasm form (design: wasm runs with zero dependencies; module +
/// description are required).
fn minimal_wasm() -> Value {
    json!({
        "schemaVersion": 1,
        "name": "my-tools",
        "version": "0.1.0",
        "tools": [{
            "name": "search",
            "form": "wasm",
            "module": "wasm/search.wasm",
            "description": "search things"
        }]
    })
}

#[test]
fn schema_is_valid_draft07_meta_schema() {
    jsonschema::meta::validate(&schema()).expect("plugin.schema.json must satisfy draft-07");
}

#[test]
fn design_example_validates() {
    assert_valid(&example());
}

#[test]
fn minimal_per_form_manifests_validate() {
    // wasm + capabilities (fs:read carries paths)
    assert_valid(&minimal_wasm());

    // script: run argv + simple-form parameters
    assert_valid(&json!({
        "schemaVersion": 1,
        "name": "ops",
        "version": "1.2.3",
        "tools": [{
            "name": "deploy",
            "form": "script",
            "run": ["scripts/deploy.sh", "--env", "${env}"],
            "description": "deploy",
            "parameters": { "env": { "type": "string" } }
        }]
    }));

    // mcp stdio: command + env (${ENV:VAR} references)
    assert_valid(&json!({
        "schemaVersion": 1,
        "name": "wiki-plugin",
        "version": "0.1.0",
        "tools": [{
            "name": "wiki",
            "form": "mcp",
            "server": {
                "command": ["python", "server/main.py"],
                "env": { "WIKI_TOKEN": "${ENV:WIKI_TOKEN}" }
            }
        }]
    }));

    // mcp remote URL + headers
    assert_valid(&json!({
        "schemaVersion": 1,
        "name": "remote",
        "version": "0.1.0",
        "tools": [{
            "name": "remote",
            "form": "mcp",
            "server": {
                "url": "https://example.com/mcp",
                "headers": { "Authorization": "Bearer ${ENV:TOK}" }
            }
        }]
    }));

    // skills optional; exposure optional (defaults to direct)
    assert_valid(&json!({
        "schemaVersion": 1,
        "name": "budgeted",
        "version": "0.1.0",
        "tools": [{
            "name": "slow",
            "form": "wasm",
            "module": "wasm/slow.wasm",
            "description": "d",
            "exposure": "deferred",
            "timeout_secs": 5
        }],
        "skills": ["skills"]
    }));
}

#[test]
fn rejects_wrong_schema_version() {
    let mut v = minimal_wasm();
    v["schemaVersion"] = json!(2);
    assert_invalid(&v);

    let mut v = minimal_wasm();
    v.as_object_mut().unwrap().remove("schemaVersion");
    assert_invalid(&v);
}

#[test]
fn rejects_bad_plugin_names() {
    let long = "a".repeat(33);
    for name in [
        "My-Tools",
        "mcp",
        "script",
        "plugin",
        "",
        long.as_str(),
        "my tools",
    ] {
        let mut v = minimal_wasm();
        v["name"] = json!(name);
        assert_invalid(&v);
    }
}

#[test]
fn rejects_bad_tool_names() {
    let long = "a".repeat(33);
    for name in ["Search", "my.tool", long.as_str(), ""] {
        let mut v = minimal_wasm();
        v["tools"][0]["name"] = json!(name);
        assert_invalid(&v);
    }
    // Reserved words only block the "name" fields (plugin name/script name/
    // builtin name); tool names are not blocked.
    let mut v = minimal_wasm();
    v["tools"][0]["name"] = json!("plugin");
    assert_valid(&v);
}

#[test]
fn rejects_form_specific_field_mixups() {
    // wasm missing module
    let mut v = minimal_wasm();
    v["tools"][0].as_object_mut().unwrap().remove("module");
    assert_invalid(&v);

    // wasm missing description (the model-visible description is required)
    let mut v = minimal_wasm();
    v["tools"][0].as_object_mut().unwrap().remove("description");
    assert_invalid(&v);

    // wasm with run mixed in
    let mut v = minimal_wasm();
    v["tools"][0]["run"] = json!(["x.sh"]);
    assert_invalid(&v);

    // script missing run
    let mut v = minimal_wasm();
    v["tools"][0]["form"] = json!("script");
    assert_invalid(&v);

    // script with capabilities mixed in (the capability allowlist is a wasm
    // sandbox concept)
    let mut v = minimal_wasm();
    v["tools"][0]["form"] = json!("script");
    v["tools"][0]["run"] = json!(["x.sh"]);
    v["tools"][0]["capabilities"] = json!([{ "type": "fs:read", "paths": ["./data"] }]);
    assert_invalid(&v);

    // mcp missing server
    let mut v = minimal_wasm();
    v["tools"][0]["form"] = json!("mcp");
    assert_invalid(&v);

    // mcp with parameters mixed in (parameters come from the server's
    // inputSchema)
    let mut v = minimal_wasm();
    v["tools"][0] = json!({
        "name": "wiki",
        "form": "mcp",
        "server": { "command": ["python", "s.py"] },
        "parameters": { "q": { "type": "string" } }
    });
    assert_invalid(&v);
}

#[test]
fn rejects_capability_violations() {
    // fs:read must come with paths
    let mut v = minimal_wasm();
    v["tools"][0]["capabilities"] = json!([{ "type": "fs:read" }]);
    assert_invalid(&v);

    // net:http must come with hosts, and must not carry paths
    let mut v = minimal_wasm();
    v["tools"][0]["capabilities"] = json!([{ "type": "net:http", "paths": ["./data"] }]);
    assert_invalid(&v);

    let mut v = minimal_wasm();
    v["tools"][0]["capabilities"] = json!([{ "type": "net:http", "hosts": ["api.example.com"] }]);
    assert_valid(&v);

    // unknown capability type
    let mut v = minimal_wasm();
    v["tools"][0]["capabilities"] = json!([{ "type": "fs:write", "paths": ["./data"] }]);
    assert_invalid(&v);
}

#[test]
fn rejects_parameter_violations() {
    // scalars only: object/array types are rejected
    for ty in ["object", "array", "null", "any"] {
        let mut v = minimal_wasm();
        v["tools"][0]["parameters"] = json!({ "q": { "type": ty } });
        assert_invalid(&v);
    }

    // simple-form entries accept only type/description
    let mut v = minimal_wasm();
    v["tools"][0]["parameters"] = json!({ "q": { "type": "string", "required": true } });
    assert_invalid(&v);
}

#[test]
fn rejects_server_violations() {
    // command and url are mutually exclusive
    let mut v = minimal_wasm();
    v["tools"][0] = json!({
        "name": "wiki",
        "form": "mcp",
        "server": { "command": ["python", "s.py"], "url": "https://x.com/mcp" }
    });
    assert_invalid(&v);

    // the http form carries no env
    let mut v = minimal_wasm();
    v["tools"][0] = json!({
        "name": "wiki",
        "form": "mcp",
        "server": { "url": "https://x.com/mcp", "env": { "A": "1" } }
    });
    assert_invalid(&v);

    // url must be http(s)
    let mut v = minimal_wasm();
    v["tools"][0] = json!({
        "name": "wiki",
        "form": "mcp",
        "server": { "url": "ftp://x.com/mcp" }
    });
    assert_invalid(&v);
}

#[test]
fn rejects_unknown_fields_and_empty_tools() {
    // unknown top-level fields (typos must not pass silently)
    let mut v = minimal_wasm();
    v["tool"] = json!([]);
    assert_invalid(&v);

    // unknown fields in a tool entry
    let mut v = minimal_wasm();
    v["tools"][0]["timout_secs"] = json!(30);
    assert_invalid(&v);

    // tools must have at least one entry
    let mut v = minimal_wasm();
    v["tools"] = json!([]);
    assert_invalid(&v);

    let mut v = minimal_wasm();
    v.as_object_mut().unwrap().remove("tools");
    assert_invalid(&v);
}
