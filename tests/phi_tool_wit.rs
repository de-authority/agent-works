//! `phi:tool@0.1.0` WIT contract parsing tests (tool-socket contract P0).
//!
//! Criteria: `wit/phi-tool.wit` is valid WIT, and the package name/version/
//! interface/function signature headers match the contract.
//! The tests feed that file directly to wit-parser (the same parser family as
//! wit-bindgen) — guarding against contract drift that "looks right to the
//! eye but the parser rejects".

use std::path::PathBuf;

fn wit_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wit/phi-tool.wit")
}

#[test]
fn phi_tool_wit_parses_with_contract_identity() {
    let mut resolve = wit_parser::Resolve::default();
    let (pkg_id, _) = resolve
        .push_path(wit_path())
        .expect("wit/phi-tool.wit must be valid WIT");

    let pkg = &resolve.packages[pkg_id];
    assert_eq!(pkg.name.namespace, "phi");
    assert_eq!(pkg.name.name, "tool");
    assert_eq!(
        pkg.name.version.as_ref().map(|v| v.to_string()).as_deref(),
        Some("0.1.0"),
        "package version is part of the contract (phi:tool@0.1.0)"
    );

    let iface_id = pkg
        .interfaces
        .get("executor")
        .expect("interface `executor` must exist");
    let iface = &resolve.interfaces[*iface_id];

    let func = iface
        .functions
        .get("invoke")
        .expect("function `invoke` must exist");
    let params: Vec<&str> = func.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(params, ["tool", "args"], "invoke(tool, args) signature");
}
