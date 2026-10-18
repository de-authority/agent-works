//! wasm-form test fixture (dispatcher component) — one module covering all of
//! the loader's scenario criteria.
//!
//! Build (the artifact is committed to `../fixtures/`; rebuild and re-copy
//! after changing this file):
//! ```sh
//! cargo build --release --target wasm32-wasip2
//! cp target/wasm32-wasip2/release/plugin_wasm_fixtures.wasm ../fixtures/dispatcher.wasm
//! ```
//!
//! Scenario surface (`invoke(tool, args)` dispatch):
//! - `echo`     — echoes args back (happy path / parameter pass-through);
//! - `read_file` — `std::fs::read_to_string(args)` (capability criterion: no
//!   reading without an fs preopen);
//! - `spin`     — infinite loop (timeout criterion: epoch interruption cuts it
//!   off — the tool hangs, not the host);
//! - `boom`     — panic (trap criterion: the tool traps, not the host).

wit_bindgen::generate!({
    path: "../../wit",
    world: "tool-guest",
});

struct Guest;

impl exports::phi::tool::executor::Guest for Guest {
    fn invoke(tool: String, args: String) -> Result<String, String> {
        match tool.as_str() {
            "echo" => Ok(args),
            "read_file" => std::fs::read_to_string(&args).map_err(|e| e.to_string()),
            "spin" => loop {
                std::hint::spin_loop();
            },
            "boom" => panic!("fixture panic: {args}"),
            other => Err(format!("unknown tool `{other}`")),
        }
    }
}

export!(Guest);
