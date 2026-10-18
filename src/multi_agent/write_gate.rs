//! File-level write mutual exclusion between sub-agents (design 2026-09-19 D6).
//!
//! In-process Mutex table: CanonicalPath → owning agent_path. **Held for the
//! task duration** — a claim is valid for the sub-agent's lifetime (not a
//! single call), and the second sub-agent that tries to write gets a named
//! error, reports it to the parent, and the parent re-partitions. `try_claim`
//! never blocks: no waiting means no deadlock surface. The release hook hangs
//! off `ChildCleanup::drop` (normal close / panic / abort all go through this
//! one spot); the timeout reaper does not release (the agent survives, its
//! claims stay valid).
//!
//! Honest boundary: `execute_command`'s shell redirection is outside the
//! gate — the gate only wraps the `write_file`/`edit_file` main channels
//! (framework-known registered names); the shell channel relies on
//! task-partitioning discipline + parent prompt guidance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_base::{Content, Tool, ToolContext};

/// Tools intercepted by the gate (framework-known registered names, design D6).
pub const WRITE_GATE_TOOLS: &[&str] = &["write_file", "edit_file"];

/// Claim key: a stable spelling that does not depend on whether the file
/// exists at claim time.
///
/// If the file exists → `canonicalize` (resolves `..`, relative spellings,
/// symlinks); if it does not yet exist → `canonicalize` the nearest existing
/// ancestor and append the remaining components verbatim. Falls back to the
/// raw path only when nothing along the way exists.
///
/// Why "on failure return the raw path" is not enough: on macOS `/tmp` is a
/// symlink to `/private/tmp` (session 20260920_b979a4f7). The first writer
/// claims before the file exists and gets the raw-path key; once the file
/// exists, every later writer canonicalizes successfully and gets the
/// resolved key — two keys for one file, mutual exclusion structurally
/// bypassed.
fn stable_key(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path.to_path_buf();
    while let Some(name) = cur.file_name() {
        suffix.push(name.to_os_string());
        let parent = match cur.parent() {
            Some(p) if p != cur => p.to_path_buf(),
            _ => break,
        };
        if let Ok(canonical) = parent.canonicalize() {
            let mut key = canonical;
            for part in suffix.iter().rev() {
                key.push(part);
            }
            return key;
        }
        cur = parent;
    }
    path.to_path_buf()
}

/// In-process write gate. One
/// [`MultiAgentRuntime`](crate::multi_agent::runtime::MultiAgentRuntime)
/// instance per runtime (it must be shared across sub-agents for mutual
/// exclusion); the parent agent is exempt (the gate only wraps sub-agents'
/// tool instances).
#[derive(Default)]
pub struct WorkspaceWriteGate {
    holders: Mutex<HashMap<PathBuf, String>>,
}

impl WorkspaceWriteGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Free → claim it and Ok; already held by self → idempotent Ok; held by
    /// another → immediate Err (named, no waiting).
    pub fn try_claim(&self, path: &Path, agent: &str) -> Result<(), String> {
        let canonical = stable_key(path);
        let mut holders = self.holders.lock().unwrap();
        match holders.get(&canonical) {
            Some(owner) if owner == agent => Ok(()),
            Some(owner) => Err(format!("file locked by {owner}")),
            None => {
                holders.insert(canonical, agent.to_string());
                Ok(())
            }
        }
    }

    /// Called on the agent's close path: releases every claim it holds.
    pub fn release_all(&self, agent: &str) {
        self.holders
            .lock()
            .unwrap()
            .retain(|_, owner| owner != agent);
    }

    /// Read-only observation: the current holder of a path (None when
    /// unclaimed). For tests/diagnostics — unlike `try_claim` it does
    /// **not** create a claim (lesson from the session 20260920_5ba1bed4
    /// regression test: a try_claim probe would grab the lock itself and
    /// starve the child under test).
    pub fn holder_of(&self, path: &Path) -> Option<String> {
        let canonical = stable_key(path);
        self.holders.lock().unwrap().get(&canonical).cloned()
    }
}

/// Wraps a write tool: `try_claim(args.path, agent_path)` before the call; on
/// failure a named error (the sub-agent reports the conflict to the parent,
/// which re-partitions — fail loud, non-blocking).
pub struct GatedTool {
    inner: Arc<dyn Tool>,
    gate: Arc<WorkspaceWriteGate>,
    agent_path: String,
}

impl GatedTool {
    pub fn new(inner: Arc<dyn Tool>, gate: Arc<WorkspaceWriteGate>, agent_path: String) -> Self {
        Self {
            inner,
            gate,
            agent_path,
        }
    }
}

#[async_trait::async_trait]
impl Tool for GatedTool {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn description(&self) -> &'static str {
        self.inner.description()
    }
    fn schema(&self) -> serde_json::Value {
        self.inner.schema()
    }
    async fn call(
        &self,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        let path = args
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if path.is_empty() {
            return Err(agent_base::AgentError::internal(format!(
                "{}: missing `path` argument; write gate cannot claim",
                self.inner.name()
            )));
        }
        match self.gate.try_claim(Path::new(path), &self.agent_path) {
            Ok(()) => self.inner.call(args, ctx).await,
            Err(why) => Err(agent_base::AgentError::internal(format!(
                "{}: {why}",
                self.inner.name()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTool {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &'static str {
            "write_file"
        }
        fn description(&self) -> &'static str {
            "fixture"
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> agent_base::AgentResult<Vec<Content>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Content::text("written")])
        }
    }

    fn gated() -> (Arc<WorkspaceWriteGate>, Arc<CountingTool>, GatedTool) {
        let gate = Arc::new(WorkspaceWriteGate::new());
        let tool = Arc::new(CountingTool {
            calls: AtomicUsize::new(0),
        });
        let wrapped = GatedTool::new(tool.clone(), gate.clone(), "root/a".to_string());
        (gate, tool, wrapped)
    }

    fn args(path: &str) -> serde_json::Value {
        serde_json::json!({ "path": path })
    }

    /// Read-only observation creates no claim: the gate table is unchanged
    /// across holder_of calls — the probe never grabs the lock.
    #[test]
    fn holder_of_observes_without_claiming() {
        let gate = WorkspaceWriteGate::new();
        assert_eq!(gate.holder_of(Path::new("a.txt")), None);
        gate.try_claim(Path::new("a.txt"), "root/x").unwrap();
        assert_eq!(
            gate.holder_of(Path::new("a.txt")).as_deref(),
            Some("root/x")
        );
    }

    #[tokio::test]
    async fn claim_then_delegate() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        let out = wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        assert!(
            matches!(&out[0], Content::Text { text } if text == "written"),
            "gated tool must delegate to the inner tool"
        );
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn second_agent_gets_named_error() {
        let (gate, _t, wrapped_a) = gated();
        let ctx = ToolContext::for_test();
        wrapped_a.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        let wrapped_b = GatedTool::new(
            Arc::new(CountingTool {
                calls: AtomicUsize::new(0),
            }),
            gate,
            "root/b".to_string(),
        );
        let err = wrapped_b.call(&args("tmp/x.txt"), &ctx).await.unwrap_err();
        assert!(err.to_string().contains("file locked by root/a"), "{err}");
    }

    #[tokio::test]
    async fn self_reclaim_is_idempotent() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        assert_eq!(
            tool.calls.load(Ordering::SeqCst),
            2,
            "self re-entry passes idempotently"
        );
    }

    #[tokio::test]
    async fn release_all_frees_the_file() {
        let (gate, _t, wrapped_a) = gated();
        let ctx = ToolContext::for_test();
        wrapped_a.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        gate.release_all("root/a");
        let wrapped_b = GatedTool::new(
            Arc::new(CountingTool {
                calls: AtomicUsize::new(0),
            }),
            gate,
            "root/b".to_string(),
        );
        wrapped_b.call(&args("tmp/x.txt"), &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn missing_path_argument_fails_loud() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        let err = wrapped
            .call(&serde_json::json!({ "other": 1 }), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing `path`"), "{err}");
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0, "no claim, no call");
    }

    #[test]
    fn distinct_spellings_of_same_file_collide() {
        let gate = WorkspaceWriteGate::new();
        let dir = std::env::temp_dir().join("phimint_gate_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f.txt");
        std::fs::write(&file, "x").unwrap();
        gate.try_claim(&file, "root/a").unwrap();
        let err = gate.try_claim(&dir.join("./f.txt"), "root/b").unwrap_err();
        assert!(err.contains("root/a"));
        gate.release_all("root/a");
        gate.try_claim(&file, "root/b").unwrap();
    }

    /// Session 20260920_b979a4f7 regression (root cause of the round-2
    /// acceptance failure): under a symlinked directory (macOS
    /// `/tmp` → `/private/tmp`), the first writer claims while the file
    /// does not yet exist — canonicalize fails and the key falls back to
    /// the raw path. After that writer creates the file, later writers
    /// canonicalize the same spelling successfully, resolving the symlink
    /// → a different key → mutual exclusion bypassed (writer-b-v2 wrote
    /// inside writer-a-v2's sleep-60 hold window and the final file was
    /// "B"). The claim key must not depend on whether the file existed at
    /// claim time.
    #[cfg(unix)]
    #[test]
    fn claim_key_survives_file_creation_through_symlinked_dir() {
        let tmp = std::env::temp_dir().join(format!("phimint_gate_link_{}", std::process::id()));
        let real = tmp.join("real");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, tmp.join("link")).unwrap();
        let gate = WorkspaceWriteGate::new();
        let via_link = tmp.join("link").join("gate.txt");

        // writer-a: claims while the file does not exist (round-2
        // 10:21:55 "Created file").
        gate.try_claim(&via_link, "root/a")
            .expect("first claim on not-yet-existing file");

        // writer-a creates the file.
        std::fs::write(real.join("gate.txt"), b"A").unwrap();

        // writer-b: same spelling, file now exists — must hit root/a's
        // lock (round-2 10:22:20 bypassed exactly here; final file "B").
        let err = gate
            .try_claim(&via_link, "root/b")
            .expect_err("second writer must be blocked after the file exists");
        assert!(err.contains("root/a"), "named-owner error, got {err:?}");

        gate.release_all("root/a");
        gate.try_claim(&via_link, "root/b")
            .expect("released file is claimable");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
