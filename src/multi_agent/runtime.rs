//! Multi-agent runtime — coordinates sub-agent lifecycle, event bridging, and
//! cancellation.
//!
//! The [`MultiAgentRuntime`] is the central coordinator. It is created once during
//! builder setup and shared via `Arc` to all 5 multi-agent tools.
//!
//! This file holds the struct itself, its construction, the parent-facing
//! surface (messaging, waiting, closing, listing, accessors), and the result
//! types. The heavier machinery lives in sibling modules of this one: `spawn`
//! (the gate-ordered spawn chain, the `ChildCleanup` drop-guard, and the
//! child event loop), `build` (child runtime assembly and permission
//! resolution, §5.4/§7.5), `fork` (the `fork_history` context bridge), and
//! `outcome` (result formatting). All child modules see this file's items via
//! `use super::*` — the external path `multi_agent::runtime::*` is unchanged.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_base::llm_trait::LlmProvider;
use agent_base::{
    AgentBuilder, AgentError, AgentResult, AgentRuntime, AllowAllApprovalHandler, ApprovalHandler,
    DenyAllApprovalHandler, DenyAllToolPolicy, Language, ReasoningEffort, RuntimeEvent, SessionId,
    Tool, ToolPolicy,
};
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::budget::{SpawnTicket, usage_total};
use super::child_builder::ChildBuilder;
use super::child_config::ChildConfig;
use super::config::{AgentAutonomy, ChildPermissionMode, MultiAgentConfig};
use super::control::AgentControl;
use super::limiter::{AgentExecutionLimiter, ExecutionSlot};
use super::mailbox::{ChildMailbox, MailboxHub, MailboxResult, MailboxStatus};
use super::path::AgentPath;
use super::registry::{AgentLifecycleEvent, AgentRegistry, RegistrySnapshot};
pub use super::runtime::watcher::{ChildReport, ChildResultEvent};

mod build;
mod fork;
mod outcome;
mod spawn;
pub mod watcher;

// ---------------------------------------------------------------------------
// MultiAgentRuntime
// ---------------------------------------------------------------------------

/// Coordinates sub-agent lifecycle, event bridging, and cancellation.
///
/// Created once during builder setup and shared via `Arc` to all 5 multi-agent
/// tools. Each tool calls methods on the runtime to spawn, communicate with, or
/// close sub-agents.
pub struct MultiAgentRuntime {
    /// Agent lifecycle registry (spawn/close/query).
    ///
    /// `Arc`-wrapped so the child task's `ChildCleanup` (in `spawn`) can hold
    /// the same registry the runtime does — without holding an
    /// `Arc<MultiAgentRuntime>`
    /// itself (a task-side strong ref would keep the runtime alive forever and
    /// `Drop`/`cancel_all` could never run — the reference cycle would defeat
    /// the very abort path §4 of the design doc relies on).
    registry: Arc<Mutex<AgentRegistry>>,

    /// Inter-agent message hub.
    mailbox: Arc<MailboxHub>,

    /// Session control plane (design §7.1): the rollout budget plus the
    /// live-concurrency limiter, as one `Arc`. The [`limiter`](Self::limiter)
    /// field below is an `Arc` **mirror** of `control.limiter()` — one gate
    /// object, two names (the spawn chain and status reads keep reaching it
    /// directly).
    control: Arc<AgentControl>,

    /// Live-concurrency gate (design doc §7.3). Unlimited by default
    /// (`ControlConfig::max_concurrency = None`); still tracks `current`
    /// so the slot conservation is observable either way.
    limiter: Arc<AgentExecutionLimiter>,

    /// Shared LLM client (from parent agent).
    client: Arc<dyn LlmProvider>,

    /// Business tools to register on child agents (NOT the 5 multi-agent tools).
    business_tools: Vec<Arc<dyn Tool>>,

    /// Business tool names to exclude from child agents (root-level orchestration
    /// tools like `decompose`/`merge` that a leaf agent cannot actually use).
    child_excluded_tools: Vec<String>,

    /// Optional reasoning-effort override applied to every child agent (see
    /// [`MultiAgentConfig::child_reasoning_effort`]).
    child_reasoning_effort: Option<ReasoningEffort>,

    /// Whether to nudge children toward read-only behaviour (see
    /// [`MultiAgentConfig::child_read_only`]).
    child_read_only: bool,

    /// Whether children may request write capability via the spawn `tools`
    /// argument (design 2026-09-19 D3; see
    /// [`MultiAgentConfig::allow_child_write`]).
    allow_child_write: bool,

    /// Write gate (D6): process-wide file-level write-mutual-exclusion table
    /// shared by all children of this runtime (parent exempt — only child
    /// tool instances get wrapped).
    write_gate: Arc<super::write_gate::WorkspaceWriteGate>,

    /// Write-gate switch (`ControlConfig.child_write_gate`). `false` wraps
    /// nothing.
    write_gate_enabled: bool,

    /// Per-child echo of the actually-registered tool set (design
    /// 2026-09-19 T8 wiring): filled at spawn, surfaced through
    /// [`list_agents`](Self::list_agents), removed when the child leaves
    /// (same drop-guard as the write gate).
    spawned_tools: Arc<Mutex<HashMap<String, Vec<String>>>>,

    /// Fork-history policy applied to every spawn (see
    /// [`MultiAgentConfig::child_fork_history`]).
    child_fork_history: Option<String>,

    /// Deployment autonomy mode (design §7.5). Drives the `Manual`
    /// three-layer expansion in `spawn_permission`, the per-child
    /// `capability::read_only_nudge` gate (both in `build`), and
    /// the exclusion merge inside `resolve_capability`. `Auto`
    /// (default) keeps every layer exactly as configured.
    autonomy: AgentAutonomy,

    /// Mutating-tool set excluded from every child under `Manual` (§7.5).
    /// Read from [`super::config::ControlConfig::write_tools`]; unused in
    /// `Auto` mode.
    write_tools: Vec<String>,

    /// Per-task execution timeout for child `run_turn` (§9.2). `None` = no
    /// timeout (current behaviour).
    task_timeout: Option<Duration>,

    /// Channel to the bridge task that emits events on parent's event bus.
    event_tx: Mutex<Option<tokio::sync::mpsc::UnboundedSender<RuntimeEvent>>>,

    /// Root cancellation token (propagates to all children).
    root_cancel: CancellationToken,

    /// JoinSet tracking all child agent tasks.
    join_set: Mutex<JoinSet<()>>,

    /// Per-child cancellation tokens (`Arc` for the same reason as
    /// [`Self::registry`]: `ChildCleanup` removes the entry on every exit
    /// path without borrowing the runtime).
    child_cancels: Arc<Mutex<HashMap<AgentPath, CancellationToken>>>,

    /// Per-path spawn generation (same-name recycle, session
    /// 20260920_5ba1bed4): bumped on every recycle so a superseded
    /// predecessor's `ChildCleanup` — which unwinds asynchronously, possibly
    /// AFTER the successor is fully registered — can detect it no longer
    /// owns the path and skip the path-scoped teardown.
    child_generations: Arc<Mutex<HashMap<AgentPath, u64>>>,

    /// Error recovery strategy (inherited from parent).
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,

    /// Language preference.
    language: Language,

    /// Permission mode for spawned child agents (resolved per spawn).
    child_permission_mode: ChildPermissionMode,

    /// Parent's tool policy, inherited by "no permission" children.
    tool_policy: Option<Arc<dyn ToolPolicy>>,

    /// Parent's approval handler, delegated to by "no permission" children.
    ///
    /// This is the codex-style behaviour (see `build_child_runtime`): a
    /// restricted sub-agent does not hard-deny its own approval requests — it
    /// routes the decision up to the parent's handler (human-in-the-loop or
    /// auto), so `ask`/`deny` semantics stay coherent across the agent tree.
    approval_handler: Option<Arc<dyn ApprovalHandler>>,

    /// Parent session manager — for fork_history (child context inheritance).
    session_manager: Mutex<Option<Arc<agent_base::engine::SessionManager>>>,
}

impl MultiAgentRuntime {
    /// Create a new multi-agent runtime.
    ///
    /// This is called internally by the builder. Tools receive an `Arc<Self>`.
    #[allow(clippy::too_many_arguments)] // runtime fields are naturally positional
    pub fn new(
        config: MultiAgentConfig,
        client: Arc<dyn LlmProvider>,
        business_tools: Vec<Arc<dyn Tool>>,
        root_cancel: CancellationToken,
        error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
        language: Language,
        tool_policy: Option<Arc<dyn ToolPolicy>>,
        approval_handler: Option<Arc<dyn ApprovalHandler>>,
    ) -> Self {
        let child_permission_mode = config.child_permission_mode;
        let child_excluded_tools = config.child_excluded_tools.clone();
        let child_reasoning_effort = config.child_reasoning_effort.clone();
        let child_read_only = config.child_read_only;
        let allow_child_write = config.allow_child_write;
        let write_gate = Arc::new(super::write_gate::WorkspaceWriteGate::new());
        let write_gate_enabled = config.control.child_write_gate;
        let spawned_tools: Arc<Mutex<HashMap<String, Vec<String>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let child_fork_history = config.child_fork_history.clone();
        // Control plane: the config knobs land in `AgentControl` (budget +
        // limiter). `None` knobs → unlimited gates that still count (§7.2).
        let control = Arc::new(AgentControl::new(&config.control));
        let limiter = Arc::clone(control.limiter());
        let autonomy = config.control.autonomy;
        let write_tools = config.control.write_tools.clone();
        let task_timeout = config.control.task_timeout;
        let registry = Arc::new(Mutex::new(AgentRegistry::new(config)));
        Self {
            registry,
            mailbox: Arc::new(MailboxHub::new()),
            control,
            limiter,
            client,
            business_tools,
            child_excluded_tools,
            child_reasoning_effort,
            child_read_only,
            allow_child_write,
            write_gate,
            write_gate_enabled,
            spawned_tools,
            child_fork_history,
            autonomy,
            write_tools,
            task_timeout,
            event_tx: Mutex::new(None),
            root_cancel,
            join_set: Mutex::new(JoinSet::new()),
            child_cancels: Arc::new(Mutex::new(HashMap::new())),
            child_generations: Arc::new(Mutex::new(HashMap::new())),
            error_recovery,
            language,
            child_permission_mode,
            tool_policy,
            approval_handler,
            session_manager: Mutex::new(None),
        }
    }

    /// Get the shared LLM client.
    ///
    /// Used by tools that need to make LLM calls (e.g., Focus-based prompt generation).
    pub fn client(&self) -> &Arc<dyn LlmProvider> {
        &self.client
    }

    /// Get the configured fork-history policy for spawns (see
    /// [`MultiAgentConfig::child_fork_history`]).
    ///
    /// Used by the `spawn_agent` tool, which reads this instead of trusting
    /// an LLM-supplied argument.
    pub fn child_fork_history(&self) -> Option<&str> {
        self.child_fork_history.as_deref()
    }

    /// Test-only view of the write gate (D6): integration tests probe
    /// claims directly.
    #[cfg(test)]
    pub(crate) fn write_gate_for_test(&self) -> &Arc<super::write_gate::WorkspaceWriteGate> {
        &self.write_gate
    }

    /// Set the event sender for bridging child events to parent.
    ///
    /// Called by the builder after creating the bridge channel.
    pub fn set_event_sender(&self, tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>) {
        *self.event_tx.lock().unwrap() = Some(tx);
    }

    /// Set the parent session manager for fork_history support.
    ///
    /// Called by the builder after creating the runtime.
    pub fn set_session_manager(&self, session_manager: Arc<agent_base::engine::SessionManager>) {
        *self.session_manager.lock().unwrap() = Some(session_manager);
    }

    /// Send a message to a child agent (no execution trigger).
    ///
    /// Called by `send_message` tool.
    pub fn send_message(&self, agent_path: &str, message: String) -> Result<bool, String> {
        let path = self.parse_path(agent_path)?;
        Ok(self.mailbox.send_message(&path, message))
    }

    /// Send a task to a child agent (triggers execution).
    ///
    /// Called by `send_message` with `trigger=true` (and the deprecated
    /// `followup_task` shim). Records the enqueue fact — the derived status
    /// becomes `Queued` (or stays `Running` if a task is already executing);
    /// the child loop's dequeue turns it `Running`. No marker to forget.
    ///
    /// Ordering is load-bearing: the enqueue fact is recorded **before**
    /// `mailbox.send_task` makes the task receivable. The send bumps the
    /// mailbox sequence and wakes the child loop, so noting afterwards would
    /// let the child's `note_dequeued` win the registry lock first — the
    /// late enqueue note would resurrect an already-consumed task as a
    /// phantom `queue_len`, blocking quiescence forever. A failed send rolls
    /// the note back (`note_send_failed`) so the fact cannot outlive the
    /// task it describes.
    pub fn send_task(
        &self,
        agent_path: &str,
        task: String,
        interrupt: bool,
    ) -> Result<bool, String> {
        let path = self.parse_path(agent_path)?;
        if !self.mailbox.contains(&path) {
            return Err("agent not found".to_string());
        }
        let mut registry = self.registry.lock().unwrap();
        registry.set_task(&path, task.clone());
        registry.touch(&path);
        let tracked = registry.note_enqueued(&path);
        drop(registry);
        let sent = self.mailbox.send_task(&path, task, interrupt);
        if !sent && tracked {
            self.registry.lock().unwrap().note_send_failed(&path);
        }
        Ok(sent)
    }

    /// Wait for a result from any or a specific child agent.
    ///
    /// Called by `wait_agent` tool. Blocks until a result arrives or timeout.
    ///
    /// For a specific path, once the agent is gone from **both** the mailbox
    /// and the registry it reports `"closed"` immediately instead of spinning
    /// to the timeout — the registry entry's removal is the terminal state
    /// (design §9.2, K3), and `ChildCleanup` releases the registry slot before
    /// the mailbox entry disappears, which makes the close→wait contract
    /// (§5.2: after `close`, `wait` gets `Closed`) race-free for late pollers.
    pub async fn wait_for_result(&self, agent_path: Option<&str>, timeout_ms: u64) -> WaitResult {
        let filter_path = match agent_path {
            Some(s) => match AgentPath::parse(s) {
                Some(p) => Some(p),
                None => {
                    return WaitResult {
                        status: "error".to_string(),
                        result: Some(format!("invalid agent path: {}", s)),
                        agent_path: None,
                        has_more: false,
                        denied_tools: vec![],
                    };
                }
            },
            None => None,
        };

        let mut seq = self.mailbox.subscribe_seq();
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(timeout_ms);

        loop {
            // Check for existing results
            let result = match &filter_path {
                Some(path) => self.mailbox.try_recv_result(path),
                None => self.mailbox.try_recv_any(),
            };

            if let Some(r) = result {
                let has_more = self.mailbox.total_pending_results() > 0;
                let (status_str, result_text) = match r.status {
                    MailboxStatus::Ok => ("ok".to_string(), r.result),
                    MailboxStatus::Error => ("error".to_string(), r.result),
                    MailboxStatus::Closed => ("closed".to_string(), r.result),
                };
                return WaitResult {
                    status: status_str,
                    result: result_text,
                    agent_path: Some(r.agent_path.to_string()),
                    has_more,
                    denied_tools: r.denied_tools,
                };
            }

            // Terminal-state synthesis for a filtered path: gone from the
            // mailbox AND the registry ⇒ closed (see method docs). Checked
            // after draining real results so a posted Closed / Ok result still
            // wins; checked before the deadline so waiters don't have to race
            // the drop-guard's lock sequence.
            if let Some(path) = &filter_path {
                let gone =
                    !self.mailbox.contains(path) && !self.registry.lock().unwrap().contains(path);
                if gone {
                    return WaitResult {
                        status: "closed".to_string(),
                        result: None,
                        agent_path: Some(path.to_string()),
                        has_more: false,
                        denied_tools: vec![],
                    };
                }
            }

            // Wait for sequence number change or timeout
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return WaitResult {
                    status: "timeout".to_string(),
                    result: None,
                    agent_path: None,
                    has_more: false,
                    denied_tools: vec![],
                };
            }

            let remaining = deadline - now;
            tokio::select! {
                _ = seq.changed() => {
                    // Sequence changed — loop back to check results
                    continue;
                }
                _ = tokio::time::sleep(remaining) => {
                    return WaitResult {
                        status: "timeout".to_string(),
                        result: None,
                        agent_path: None,
                        has_more: false,
                        denied_tools: vec![],
                    };
                }
            }
        }
    }

    /// Non-blocking check for a result from any or a specific child agent.
    ///
    /// Returns the result immediately if one is available, or `"pending"` status
    /// if no result has arrived yet. Called by `wait_agent` with `blocking=false`.
    pub fn try_wait(&self, agent_path: Option<&str>) -> WaitResult {
        let filter_path = match agent_path {
            Some(s) => match AgentPath::parse(s) {
                Some(p) => Some(p),
                None => {
                    return WaitResult {
                        status: "error".to_string(),
                        result: Some(format!("invalid agent path: {}", s)),
                        agent_path: None,
                        has_more: false,
                        denied_tools: vec![],
                    };
                }
            },
            None => None,
        };

        let result = match &filter_path {
            Some(path) => self.mailbox.try_recv_result(path),
            None => self.mailbox.try_recv_any(),
        };

        if let Some(r) = result {
            let has_more = self.mailbox.total_pending_results() > 0;
            let (status_str, result_text) = match r.status {
                MailboxStatus::Ok => ("ok".to_string(), r.result),
                MailboxStatus::Error => ("error".to_string(), r.result),
                MailboxStatus::Closed => ("closed".to_string(), r.result),
            };
            return WaitResult {
                status: status_str,
                result: result_text,
                agent_path: Some(r.agent_path.to_string()),
                has_more,
                denied_tools: r.denied_tools,
            };
        }

        // Terminal-state synthesis for a filtered path (same as wait_for_result)
        if let Some(path) = &filter_path {
            let gone =
                !self.mailbox.contains(path) && !self.registry.lock().unwrap().contains(path);
            if gone {
                return WaitResult {
                    status: "closed".to_string(),
                    result: None,
                    agent_path: Some(path.to_string()),
                    has_more: false,
                    denied_tools: vec![],
                };
            }
        }

        WaitResult {
            status: "pending".to_string(),
            result: None,
            agent_path: filter_path.map(|p| p.to_string()),
            has_more: false,
            denied_tools: vec![],
        }
    }

    /// Start the background watcher task: the fan-in coordinator.
    ///
    /// The watcher watches the mailbox and emits two kinds of events on a
    /// single channel: per-result `Progress` (user-facing notice, never wakes
    /// the parent) and one `Batch` per generation of children (all full
    /// reports — this is what wakes the parent for its synthesis turn).
    ///
    /// Uses a watchdog wrapper that automatically restarts the watcher if it
    /// panics. The task exits when the `root_cancel` token is fired.
    pub fn start_watcher(
        self: &Arc<Self>,
    ) -> (
        tokio::task::JoinHandle<()>,
        mpsc::UnboundedReceiver<ChildResultEvent>,
    ) {
        let (child_result_tx, child_result_rx) = mpsc::unbounded_channel();

        // Focus summarizes each result for the user on the main agent's
        // behalf (10 s timeout, fail-open to a plain notice).
        let summarizer = Arc::new(crate::focus::ProgressSummarizer::new(
            Arc::clone(&self.client),
            crate::focus::DEFAULT_SUMMARY_TIMEOUT,
        ));

        let handle = watcher::spawn_watcher_with_watchdog(
            Arc::clone(&self.mailbox),
            Arc::clone(&self.registry),
            Some(summarizer),
            Some(child_result_tx),
            self.root_cancel.clone(),
            watcher::HELD_BATCH_REAP_AFTER,
        );

        (handle, child_result_rx)
    }

    /// Close a child agent — cancel now, clean up on the child's exit.
    ///
    /// Called by `close_agent` tool. Performs only the cancellation; the
    /// actual teardown (post Closed → registry release → mailbox unregister →
    /// concurrency slot return) runs in the child task's `ChildCleanup::drop`
    /// when its future ends (design §4/§5.2 — same exit as panic and abort).
    ///
    /// Returns `closed == true` if an agent was live at this moment (idempotent:
    /// a second close while the first child is still winding down reports
    /// `false`). A `wait_for_result` on this path afterwards yields `"closed"`.
    pub fn close_agent(&self, agent_path: &str) -> Result<CloseResult, String> {
        let path = self.parse_path(agent_path)?;

        // Get previous status (before the cancellation, while the entry is
        // still in its last live state).
        let previous_status = {
            let registry = self.registry.lock().unwrap();
            registry
                .get(&path)
                .map(|e| e.status().name().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };

        // Cancel only — do NOT remove the token or touch the mailbox:
        // ChildCleanup is the single removal point on all exit paths. The
        // registry DOES get a terminal fact now (`note_closing`): the child
        // may take arbitrarily long to unwind (cancellation lands at its
        // next await point), and until it does its stale `in_flight=true,
        // results_posted=0` facts would block fan-in quiescence forever —
        // a held sibling report would never fire (session
        // 20260914_50cf809d). Closing also derives the status to `Closed`
        // so list_agents / the task panel reflect the close immediately.
        let closed = {
            let cancels = self.child_cancels.lock().unwrap();
            match cancels.get(&path) {
                Some(token) => {
                    if token.is_cancelled() {
                        // Already cancelled and the future hasn't unwound to
                        // its guard yet — a close is in flight, not "live".
                        false
                    } else {
                        token.cancel();
                        true
                    }
                }
                None => false,
            }
        };
        if closed {
            // Lock ordering: child_cancels (above, released) → registry.
            // Idempotent beyond the first close (closing is sticky).
            self.registry.lock().unwrap().note_closing(&path);
        }

        Ok(CloseResult {
            closed,
            previous_status,
            message: if closed {
                "agent closed".to_string()
            } else {
                "agent not found".to_string()
            },
        })
    }

    /// List all active sub-agents.
    ///
    /// Called by `list_agents` tool. Reads the registry's derived snapshot —
    /// the same view the lifecycle watch channel publishes — so the tool and
    /// any UI consumer always see one identical truth.
    pub fn list_agents(&self) -> Vec<AgentInfo> {
        let tools = self.spawned_tools.lock().unwrap();
        self.registry
            .lock()
            .unwrap()
            .snapshot()
            .agents
            .into_iter()
            .map(|a| AgentInfo {
                agent_path: a.path.clone(),
                status: a.status,
                tool_calls: a.tool_calls,
                running_secs: a.running_secs,
                last_activity_secs: a.last_activity_secs,
                task: a.task,
                pending_results: a.pending_results,
                spawned_tools: tools.get(&a.path).cloned().unwrap_or_default(),
            })
            .collect()
    }

    /// Number of agents with work outstanding (executing or queued).
    ///
    /// Derived from facts, not markers. The fan-in coordinator's actual
    /// quiescence signal is [`AgentRegistry::quiescent`] (via the watcher).
    pub fn busy_count(&self) -> usize {
        self.registry.lock().unwrap().busy_count()
    }

    /// Subscribe to lifecycle snapshots (watch channel; latest only).
    ///
    /// Phase 5 hook: the phimint event bridge maps these onto its UI panel.
    /// Metric fields in the snapshot are transition-frozen (republishes
    /// happen on status transitions only) — derive status from this stream
    /// and poll `list_agents` for live metrics.
    pub fn subscribe_lifecycle(&self) -> watch::Receiver<Arc<RegistrySnapshot>> {
        self.registry.lock().unwrap().subscribe()
    }

    /// The last `max` lifecycle transitions, oldest first (diagnostics).
    pub fn recent_lifecycle_events(&self, max: usize) -> Vec<AgentLifecycleEvent> {
        self.registry.lock().unwrap().recent_events(max)
    }

    /// Get the mailbox hub (for tools that need it directly).
    pub fn mailbox(&self) -> &Arc<MailboxHub> {
        &self.mailbox
    }

    /// Get reference to the registry.
    pub fn registry(&self) -> &Mutex<AgentRegistry> {
        &self.registry
    }

    /// The session control plane (§7.1): rollout budget + concurrency
    /// limiter + status reads. Tools/operators reach gates through here
    /// without touching runtime internals.
    pub fn control(&self) -> &Arc<AgentControl> {
        &self.control
    }

    /// Cancel all child agents.
    pub fn cancel_all(&self) {
        let cancelled: Vec<AgentPath> = {
            let mut cancels = self.child_cancels.lock().unwrap();
            cancels
                .drain()
                .map(|(path, token)| {
                    token.cancel();
                    path
                })
                .collect()
        };
        // Same terminal-fact rationale as `close_agent`: every cancelled
        // child stops counting against fan-in quiescence immediately.
        let mut registry = self.registry.lock().unwrap();
        for path in &cancelled {
            registry.note_closing(path);
        }
    }
}

impl Drop for MultiAgentRuntime {
    fn drop(&mut self) {
        self.cancel_all();
        // Drain any already-completed join handles to detect panics
        let mut js = self.join_set.lock().unwrap();
        while let Some(result) = js.try_join_next() {
            if let Err(e) = result
                && e.is_panic()
            {
                tracing::error!(
                    error = %e,
                    "child agent task panicked"
                );
            }
        }
    }
}

impl MultiAgentRuntime {
    fn parse_path(&self, s: &str) -> Result<AgentPath, String> {
        AgentPath::parse(s).ok_or_else(|| format!("invalid agent path: '{}'", s))
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result from `wait_for_result()`.
#[derive(Clone, Debug)]
pub struct WaitResult {
    pub status: String,
    pub result: Option<String>,
    pub agent_path: Option<String>,
    pub has_more: bool,
    /// Tools the child attempted but was denied permission to call.
    pub denied_tools: Vec<String>,
}

/// Result from `close_agent()`.
#[derive(Clone, Debug)]
pub struct CloseResult {
    pub closed: bool,
    pub previous_status: String,
    pub message: String,
}

/// Agent info for `list_agents()`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AgentInfo {
    pub agent_path: String,
    /// Derived status name: `queued` (task waiting in queue), `running`
    /// (executing), `done` (no pending work — result delivered, or the
    /// agent has not been tasked yet; a freshly spawned agent reads `done`
    /// until its first `send_task` succeeds). `closed` never appears:
    /// unregistered agents leave this listing entirely.
    pub status: String,
    /// Tool calls actually executed (grows while the agent works). A frozen
    /// count together with a stale `last_activity_secs` is the stall signal.
    pub tool_calls: usize,
    /// Seconds spent in the current `Running` period; present only while
    /// running. Feeds the Phase 6 stall reaper.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub running_secs: Option<u64>,
    /// Seconds since the agent's last observed activity (task start or tool
    /// call); `None` until the agent receives its first task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity_secs: Option<u64>,
    /// What the agent was asked to do (None until the first task is sent).
    pub task: Option<String>,
    /// Results the agent has posted that no fired batch has handed to the
    /// parent yet. `done` + nonzero = the report exists and is en route in a
    /// future batch — do not redo the work, end the turn to receive it.
    #[serde(skip_serializing_if = "is_zero")]
    pub pending_results: usize,
    /// The tools this child was **actually** registered with (post-exclusion,
    /// post-whitelist; design 2026-09-19 external-voice finding 9 / T8 wiring). Empty
    /// for read-only children — the omission IS the fact.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spawned_tools: Vec<String>,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
