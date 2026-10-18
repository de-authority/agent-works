//! Background watcher task — the fan-in coordinator for child results.
//!
//! Session 20260903_0cf95e79 redesign: instead of waking the parent for
//! every single result (which pushed the "are we done yet?" decision into
//! the parent's LLM — it misread a frozen inventory `tool_count` as a stall
//! and started doing the children's work itself), the watcher now owns the
//! coordination:
//!
//! - **Progress** — each result is announced for the *user* immediately
//!   (Ok: Focus summary once wired, plain notice otherwise; Error: the
//!   error's first line, ANSI/control sanitized; Closed: plain notice).
//!   Progress never
//!   wakes the parent agent.
//! - **Batch** — once the registry is quiescent (see
//!   [`AgentRegistry::quiescent`]: nobody executing, nothing queued, and
//!   every registered agent has delivered ≥1 result) and at least one
//!   non-Closed report is pending, the watcher emits one `Batch` carrying
//!   every full report. The parent wakes exactly once per generation of
//!   children and runs one synthesis turn.
//!
//! Rules encoded here:
//! - A redundant `Closed` for an agent whose `Ok`/`Error` is already in the
//!   batch is dropped — close-after-complete is bookkeeping noise.
//! - A batch that contains only `Closed` results never wakes the parent
//!   (the user closed the children; there is nothing to synthesize). They
//!   were surfaced as Progress and the batch resets so stale notifications
//!   cannot leak into a later generation's batch.
//! - The Focus summary never gates the wake (session 20260903_d8fc41dc:
//!   awaited summaries serialized their timeouts in front of the batch).
//!   Ok results get a plain Progress notice synchronously, then the summary
//!   runs detached and follows as a second Progress event whenever Focus
//!   answers (or not at all on failure). A Progress event may therefore
//!   arrive after the Batch event, and the same agent may Progress twice —
//!   consumers must be idempotent (phimint's UI is: mark-finished is a
//!   no-op on an already-finished entry). Closed results need no summary
//!   and are announced synchronously. Error results skip Focus entirely:
//!   one synchronous Progress carrying the raw error's first line (session
//!   20260906_0f6d4341 — the user wants the real reason, not a paraphrase).
//! - Quiescence timing is **derived, not marked** (session
//!   20260904_c6559510): the child loop records facts at dequeue and before
//!   each post (`note_posted` precedes the post's seq bump), so "result
//!   drained ⇒ producer settled" holds structurally — no Done-before-post
//!   comment contract for callers to violate, and no phantom-idle window
//!   while a sibling task is still queued (`queue_len > 0` blocks the
//!   predicate directly). The delivery clause additionally seals the
//!   spawn→send window: a freshly spawned agent has zero deliveries, so a
//!   sibling's result cannot fire a batch that would exclude it.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::outcome::format_child_result;
use crate::focus::ProgressSummarizer;
use crate::multi_agent::mailbox::{MailboxHub, MailboxResult, MailboxStatus};
use crate::multi_agent::path::AgentPath;
use crate::multi_agent::registry::AgentRegistry;

/// One child's final report — the full conclusion, never the child's
/// working history (extraction happens in `outcome.rs` at post time).
#[derive(Clone, Debug)]
pub struct ChildReport {
    /// The child agent's path (e.g. "root/worker").
    pub agent_path: String,
    /// Status: "ok", "error", or "closed".
    pub status: String,
    /// The result text (if any).
    pub result: Option<String>,
    /// Formatted message suitable for injection into the parent's context.
    pub message: String,
}

/// An event delivered by the watcher task.
#[derive(Clone, Debug)]
pub enum ChildResultEvent {
    /// One child returned. User-facing progress only — never wakes the
    /// parent agent. Ok results with a wired summarizer are emitted
    /// **twice**: first synchronously with `summary: None` (the plain
    /// "returned" notice, the moment the child returns), then once more with
    /// the Focus summary when it lands (no second event on Focus failure).
    /// **Error results are emitted once**, synchronously, with `summary`
    /// carrying the raw first line of the child's error text — the user
    /// wants the real reason, not an LLM paraphrase of it (session
    /// 20260906_0f6d4341). Consumers must treat repeated Progress for the
    /// same agent as idempotent.
    Progress {
        /// The child agent's path.
        agent_path: String,
        /// Status: "ok", "error", or "closed".
        status: String,
        /// For Ok: the Focus-generated summary shown as a follow-up line.
        /// For Error: the error's first line, sanitized (ANSI stripped,
        /// control characters blanked) and truncated at 160 chars.
        /// `None` → the consumer shows a plain notice.
        summary: Option<String>,
    },
    /// Every child has returned — wake the parent once with all reports,
    /// including any `Closed` siblings of the same generation.
    Batch {
        /// Full reports, one per returned child, in arrival order.
        reports: Vec<ChildReport>,
    },
}

impl ChildResultEvent {
    /// The producing agent's path, for logging.
    pub fn agent_path(&self) -> &str {
        match self {
            Self::Progress { agent_path, .. } => agent_path,
            Self::Batch { reports } => reports
                .first()
                .map(|r| r.agent_path.as_str())
                .unwrap_or_default(),
        }
    }
}

/// Cap on the raw error reason surfaced in a Progress notice (fits a TUI
/// line or two; the full text still reaches the parent via the batch).
const MAX_ERROR_REASON_CHARS: usize = 160;

/// Strip ANSI escape sequences (CSI: ESC `[` params final byte). Provider
/// error bodies are gateway-controlled bytes (llm-providers embeds the whole
/// non-2xx response body in `LlmError::api`), and this reason line is headed
/// for the TUI transcript — never let it paint terminal state or leave
/// `[31m` debris behind a dropped ESC.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            while let Some(&n) = chars.peek() {
                chars.next();
                if ('@'..='~').contains(&n) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract the user-visible reason from a child's error text: the first
/// line with visible content — ANSI stripped, control characters (CR
/// splices, BEL, stray ESC…) blanked — truncated with a bounded check so a
/// multi-megabyte single-line body cannot stall the watcher drain loop.
/// `None` → the consumer's terse fallback notice.
fn error_reason(result_text: Option<&str>) -> Option<String> {
    let line = result_text
        .map(strip_ansi)
        .as_deref()?
        .lines()
        .map(|l| {
            l.chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect::<String>()
        })
        .map(|l| l.trim().to_string())
        .find(|l| !l.is_empty())?;
    let mut reason: String = line.chars().take(MAX_ERROR_REASON_CHARS).collect();
    if line.chars().nth(MAX_ERROR_REASON_CHARS).is_some() {
        reason.push('…');
    }
    Some(reason)
}

/// Spawn the background watcher (fan-in coordinator) task.
///
/// Returns a `JoinHandle` that the caller can store (or ignore — the task
/// Production held-batch reap delay: a batch held this long while no agent
/// can still post means quiescence failed on facts that will never settle
/// (force-closed child, unwind that never lands) — hand the batch over.
pub(crate) const HELD_BATCH_REAP_AFTER: Duration = Duration::from_secs(90);

/// exits when the `cancel` token is fired or the channel sender is dropped).
pub fn spawn_watcher(
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    summarizer: Option<Arc<ProgressSummarizer>>,
    child_result_tx: Option<mpsc::UnboundedSender<ChildResultEvent>>,
    cancel: CancellationToken,
    held_reap_after: Duration,
) -> tokio::task::JoinHandle<()> {
    let mut seq_rx = mailbox.subscribe_seq();

    tokio::spawn(async move {
        // Results held until the whole generation has returned.
        let mut batch: Vec<MailboxResult> = Vec::new();
        // When the oldest held report arrived — drives the held-batch reaper.
        let mut batch_first_at: Option<Instant> = None;
        let mut reap_tick = tokio::time::interval(held_reap_after);
        // Consume the interval's immediate first tick so reaping runs on a
        // steady cadence from now on, not at spawn time.
        reap_tick.tick().await;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Shutdown flush: reports already held must reach the
                    // parent — a watcher's death must never bury them.
                    force_handover(&mut batch, &registry, &child_result_tx);
                    break;
                }
                _ = reap_tick.tick() => {
                    // Held-batch reaper (liveness backstop): a batch held past
                    // `held_reap_after` while no agent can still post means
                    // quiescence failed on facts that will never settle —
                    // hand over anyway rather than strand the reports.
                    let held = batch_first_at.is_some_and(|t| t.elapsed() >= held_reap_after);
                    if !held {
                        continue;
                    }
                    let any_working = registry
                        .lock()
                        .unwrap()
                        .list()
                        .iter()
                        .any(|e| !e.closing && (e.in_flight || e.queue_len > 0));
                    if any_working {
                        continue;
                    }
                    tracing::info!(
                        batch_len = batch.len(),
                        held_for_secs = batch_first_at.map(|t| t.elapsed().as_secs()),
                        "watcher: held-batch reaper firing — quiescence unachievable, forcing handover"
                    );
                    force_handover(&mut batch, &registry, &child_result_tx);
                    batch_first_at = None;
                }
                result = seq_rx.changed() => {
                    if result.is_err() {
                        // Hub dropped — same contract as cancel: flush what
                        // the parent would otherwise never see.
                        force_handover(&mut batch, &registry, &child_result_tx);
                        break;
                    }

                    // Drain everything pending into the batch.
                    while let Some(r) = mailbox.try_recv_any() {
                        if matches!(r.status, MailboxStatus::Closed) {
                            // Redundant close notification: this agent's real
                            // result is already in the batch (or already
                            // flushed) — bookkeeping noise, drop it.
                            let already_accounted = batch.iter().any(|b| {
                                b.agent_path == r.agent_path
                                    && !matches!(b.status, MailboxStatus::Closed)
                            });
                            if already_accounted {
                                tracing::debug!(
                                    agent = %r.agent_path,
                                    "dropping redundant Closed notification"
                                );
                                continue;
                            }
                        }

                        if let Some(tx) = &child_result_tx {
                            let status = status_str(&r.status);
                            // Focus summarizes for the user, on the main
                            // agent's behalf. Closed needs no LLM call — a
                            // plain "closed" notice is enough. Any Focus
                            // failure degrades to `None` (plain notice).
                            //
                            // Session 20260903_d8fc41dc: the summarize call
                            // used to be awaited right here, which put every
                            // timeout (30 s) on the wake path — the batch
                            // could not be evaluated until all pending
                            // summaries had run, serializing N×timeout in
                            // front of the parent's wake. The summary only
                            // ever serves the user, so it now runs detached:
                            // the result enters the batch immediately and the
                            // Progress event follows whenever Focus answers.
                            // Consequence: a Progress event may arrive after
                            // the Batch event — consumers must be idempotent
                            // (phimint's UI is: mark-finished is a no-op on
                            // an already-finished entry).
                            match (&summarizer, &r.status) {
                                (_, MailboxStatus::Error) => {
                                    // Errors skip Focus entirely: the user
                                    // wants the real reason (e.g. "LLM call
                                    // failed: HTTP request failed: …"), and
                                    // an LLM paraphrase of an error adds a
                                    // second delayed notice plus a 30 s
                                    // timeout risk on a call that analyzes a
                                    // failure (session 20260906_0f6d4341:
                                    // both Error children surfaced as a
                                    // terse "execution error" followed by a vague
                                    // paraphrase while the raw text went
                                    // only to the parent). The raw first
                                    // line goes out synchronously instead.
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: error_reason(r.result.as_deref()),
                                    });
                                }
                                (Some(s), MailboxStatus::Ok) => {
                                    // Plain notice FIRST, synchronously: the
                                    // user learns the child returned the
                                    // moment it does (session
                                    // 20260904_e6612477: one summary's 30 s
                                    // Focus timeout delayed that child's only
                                    // notice by 30 s). The Focus summary —
                                    // when it lands — follows as a second
                                    // Progress event; on Focus failure no
                                    // second event is sent (the plain notice
                                    // already covers it).
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: None,
                                    });
                                    let task = registry
                                        .lock()
                                        .unwrap()
                                        .get(&r.agent_path)
                                        .and_then(|e| e.task.clone());
                                    let agent_name = r.agent_path.name().to_string();
                                    let agent_path = r.agent_path.to_string();
                                    let status = status.to_string();
                                    let result_text = r.result.clone();
                                    let summarizer = Arc::clone(s);
                                    let tx = tx.clone();
                                    tokio::spawn(async move {
                                        let summary = match summarizer
                                            .summarize(
                                                &agent_name,
                                                &status,
                                                task.as_deref(),
                                                result_text.as_deref(),
                                            )
                                            .await
                                        {
                                            Some(text) => Some(text),
                                            None => {
                                                tracing::debug!(
                                                    agent = %agent_path,
                                                    "progress summary unavailable — plain notice already sent"
                                                );
                                                return;
                                            }
                                        };
                                        let _ = tx.send(ChildResultEvent::Progress {
                                            agent_path,
                                            status,
                                            summary,
                                        });
                                    });
                                }
                                _ => {
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: None,
                                    });
                                }
                            }
                        }
                        if batch_first_at.is_none() {
                            batch_first_at = Some(Instant::now());
                        }
                        batch.push(r);
                    }

                    // Quiescence: nobody working, nothing queued, and every
                    // registered agent has delivered ≥1 result (delivery
                    // completeness seals the spawn→send window) — plus at
                    // least one real (non-Closed) report pending → wake the
                    // parent once with everything.
                    {
                        let reg = registry.lock().unwrap();
                        let q = reg.quiescent();
                        if !q {
                            let snapshot = reg.snapshot();
                            tracing::info!(
                                batch_len = batch.len(),
                                agent_count = snapshot.agents.len(),
                                agents = ?snapshot.agents.iter().map(|a| {
                                    format!("{}: status={}, tool_calls={}, pending={}", a.path, a.status, a.tool_calls, a.pending_results)
                                }).collect::<Vec<_>>(),
                                "watcher: quiescent=false, holding batch"
                            );
                        }
                    }
                    if batch.is_empty() || !registry.lock().unwrap().quiescent() {
                        continue;
                    }
                    let any_real = batch
                        .iter()
                        .any(|b| !matches!(b.status, MailboxStatus::Closed));
                    if !any_real {
                        // Only close notifications — surfaced as Progress
                        // above; the parent has nothing to synthesize. Reset
                        // so they cannot leak into a later generation.
                        batch.clear();
                        batch_first_at = None;
                        continue;
                    }
                    // Stamp the delivery fact BEFORE draining: everything in
                    // this batch is now handed over, so `pending_results` on
                    // list_agents drops to zero the moment the batch fires.
                    let batch_paths: Vec<AgentPath> = {
                        let mut seen = std::collections::BTreeSet::new();
                        batch
                            .iter()
                            .filter(|r| !matches!(r.status, MailboxStatus::Closed))
                            .filter(|r| seen.insert(r.agent_path.to_string()))
                            .map(|r| r.agent_path.clone())
                            .collect()
                    };
                    {
                        let mut reg = registry.lock().unwrap();
                        for path in &batch_paths {
                            reg.note_batch_handed_over(path);
                        }
                    }
                    let reports = batch
                        .drain(..)
                        .map(|r| format_child_result(&r))
                        .collect();
                    batch_first_at = None;
                    if let Some(tx) = &child_result_tx {
                        let _ = tx.send(ChildResultEvent::Batch { reports });
                    }
                }
            }
        }
    })
}

fn status_str(status: &MailboxStatus) -> &'static str {
    match status {
        MailboxStatus::Ok => "ok",
        MailboxStatus::Error => "error",
        MailboxStatus::Closed => "closed",
    }
}

/// Fire a held batch unconditionally — the liveness backstop shared by the
/// shutdown flush and the held-batch reaper.
///
/// The normal handover waits for registry quiescence, but registry facts can
/// settle into a state quiescence can never accept (a force-closed child
/// whose unwind never lands, an entry left mid-flight by an abnormal exit):
/// without this, held reports are stranded until process death and the
/// parent's final report is written blind (session 20260914_50cf809d — two
/// specialist reports lost). Semantics mirror the main handover path: drop
/// Closed-only batches, stamp `note_batch_handed_over` first, then drain.
fn force_handover(
    batch: &mut Vec<MailboxResult>,
    registry: &Arc<Mutex<AgentRegistry>>,
    child_result_tx: &Option<mpsc::UnboundedSender<ChildResultEvent>>,
) {
    if batch.is_empty() {
        return;
    }
    let any_real = batch
        .iter()
        .any(|b| !matches!(b.status, MailboxStatus::Closed));
    if !any_real {
        // Only close notifications — nothing to synthesize.
        batch.clear();
        return;
    }
    let batch_paths: Vec<AgentPath> = {
        let mut seen = std::collections::BTreeSet::new();
        batch
            .iter()
            .filter(|r| !matches!(r.status, MailboxStatus::Closed))
            .filter(|r| seen.insert(r.agent_path.to_string()))
            .map(|r| r.agent_path.clone())
            .collect()
    };
    {
        let mut reg = registry.lock().unwrap();
        for path in &batch_paths {
            reg.note_batch_handed_over(path);
        }
    }
    let reports = batch.drain(..).map(|r| format_child_result(&r)).collect();
    if let Some(tx) = child_result_tx {
        let _ = tx.send(ChildResultEvent::Batch { reports });
    }
}

/// Spawn a watchdog-wrapped watcher task.
///
/// The watchdog monitors the inner watcher task and restarts it if it panics.
/// This ensures the child result delivery mechanism remains operational even
/// in the face of unexpected panics. A panic counter is logged for diagnostics.
///
/// The watchdog itself exits when the `cancel` token is fired.
pub fn spawn_watcher_with_watchdog(
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    summarizer: Option<Arc<ProgressSummarizer>>,
    child_result_tx: Option<mpsc::UnboundedSender<ChildResultEvent>>,
    cancel: CancellationToken,
    held_reap_after: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut restart_count: u32 = 0;

        loop {
            if cancel.is_cancelled() {
                break;
            }

            let handle = spawn_watcher(
                mailbox.clone(),
                registry.clone(),
                summarizer.clone(),
                child_result_tx.clone(),
                cancel.clone(),
                held_reap_after,
            );

            // Wait for the watcher to finish.
            match handle.await {
                Ok(()) => {
                    // Normal exit (cancel or channel closed). No restart needed.
                    break;
                }
                Err(join_err) if join_err.is_panic() => {
                    restart_count += 1;
                    let panic_info = join_err
                        .try_into_panic()
                        .ok()
                        .and_then(|p| {
                            p.downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| p.downcast_ref::<String>().cloned())
                        })
                        .unwrap_or_else(|| "unknown panic".to_string());

                    tracing::warn!(
                        restart_count,
                        panic_info = %panic_info,
                        "watcher task panicked, restarting"
                    );

                    // Brief backoff to avoid tight restart loops on
                    // persistent panics (e.g. a bug in format_child_result).
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                Err(join_err) => {
                    // Cancelled or other JoinError — exit.
                    tracing::debug!(error = %join_err, "watcher task join error, exiting watchdog");
                    break;
                }
            }
        }

        if restart_count > 0 {
            tracing::info!(restart_count, "watcher watchdog exiting after restarts");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::mailbox::MailboxResult;
    use crate::multi_agent::path::AgentPath;

    struct Fixture {
        mailbox: Arc<MailboxHub>,
        registry: Arc<Mutex<AgentRegistry>>,
        rx: mpsc::UnboundedReceiver<ChildResultEvent>,
        cancel: CancellationToken,
        #[allow(dead_code)]
        handle: tokio::task::JoinHandle<()>,
    }

    /// Watcher + registry fixture. Children move through the same fact
    /// points (`note_enqueued` / `note_dequeued` / `note_posted`) the child
    /// loop uses, in the same order.
    fn fixture() -> Fixture {
        fixture_with_summarizer(None)
    }

    fn fixture_with_summarizer(summarizer: Option<Arc<ProgressSummarizer>>) -> Fixture {
        fixture_with(Config {
            summarizer,
            held_reap_after: std::time::Duration::from_secs(1),
        })
    }

    /// Fixture knobs that must differ per test (production uses
    /// `HELD_BATCH_REAP_AFTER`; tests keep the reap window above the
    /// `assert_no_event` windows except where the reaper is the subject).
    struct Config {
        summarizer: Option<Arc<ProgressSummarizer>>,
        held_reap_after: std::time::Duration,
    }

    fn fixture_with(cfg: Config) -> Fixture {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let handle = spawn_watcher_with_watchdog(
            mailbox.clone(),
            registry.clone(),
            cfg.summarizer,
            Some(tx),
            cancel.clone(),
            cfg.held_reap_after,
        );
        Fixture {
            mailbox,
            registry,
            rx,
            cancel,
            handle,
        }
    }

    /// Register a child in both the mailbox and the registry with a task
    /// dequeued and executing — the facts `send_task` + the child loop's
    /// dequeue produce.
    fn spawn_running(fx: &Fixture, name: &str) {
        let path = AgentPath::root().join(name);
        fx.mailbox.register(&path);
        fx.registry.lock().unwrap().register(&path, 1).unwrap();
        fx.registry.lock().unwrap().note_enqueued(&path);
        fx.registry.lock().unwrap().note_dequeued(&path);
    }

    /// The child loop's terminal order: record the delivery fact *before*
    /// posting the result (see `note_posted` in `spawn.rs`).
    fn finish_and_post(fx: &Fixture, name: &str, status: MailboxStatus, text: Option<&str>) {
        let path = AgentPath::root().join(name);
        fx.registry.lock().unwrap().note_posted(&path);
        fx.mailbox.register(&path);
        fx.mailbox.post_result(MailboxResult {
            agent_path: path,
            status,
            result: text.map(|s| s.to_string()),
            denied_tools: vec![],
        });
    }

    async fn next_event(fx: &mut Fixture) -> ChildResultEvent {
        tokio::time::timeout(std::time::Duration::from_secs(2), fx.rx.recv())
            .await
            .expect("event timeout")
            .expect("channel closed")
    }

    async fn assert_no_event(fx: &mut Fixture) {
        let got = tokio::time::timeout(std::time::Duration::from_millis(200), fx.rx.recv()).await;
        assert!(
            got.is_err(),
            "expected no event, got {:?}",
            got.ok().flatten()
        );
    }

    #[tokio::test]
    async fn lone_result_progress_then_batch() {
        let mut fx = fixture();
        finish_and_post(&fx, "worker", MailboxStatus::Ok, Some("done!"));

        match next_event(&mut fx).await {
            ChildResultEvent::Progress {
                agent_path, status, ..
            } => {
                assert_eq!(agent_path, "root/worker");
                assert_eq!(status, "ok");
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert_eq!(reports[0].agent_path, "root/worker");
                assert_eq!(reports[0].status, "ok");
                assert_eq!(reports[0].result.as_deref(), Some("done!"));
                assert!(reports[0].message.contains("done!"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn batch_fire_clears_pending_results() {
        // Session 20260904_841ed65b fix: the moment a batch fires, its
        // members are "handed over" — list_agents must read pending_results
        // == 0 from then on, or a mid-turn parent would keep seeing a
        // delivery gap that has already been closed. The stamp happens
        // BEFORE tx.send, so reading the registry after receiving the Batch
        // is deterministic.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }
        let pending = |fx: &Fixture, name: &str| {
            fx.registry
                .lock()
                .unwrap()
                .snapshot()
                .agents
                .into_iter()
                .find(|a| a.path == format!("root/{name}"))
                .expect("agent registered")
                .pending_results
        };

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        // Both Progress notices come first. (The pre-batch pending==1 state
        // is asserted in registry.rs — here the watcher may legitimately
        // have stamped+sent the batch before we get scheduled, so only the
        // post-Batch state is deterministic.)
        next_event(&mut fx).await;
        next_event(&mut fx).await;

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => assert_eq!(reports.len(), 2),
            other => panic!("expected Batch, got {other:?}"),
        }
        assert_eq!(pending(&fx, "a"), 0, "handed over with the batch");
        assert_eq!(pending(&fx, "b"), 0, "handed over with the batch");
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn batch_held_until_last_child_finishes() {
        // Two children; A finishes while B is still Running → only A's
        // Progress. When B finishes, one Batch carries both reports.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { agent_path, .. } => {
                assert_eq!(agent_path, "root/a");
            }
            other => panic!("expected Progress for a, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // b still Running — no Batch

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { agent_path, .. } => {
                assert_eq!(agent_path, "root/b");
            }
            other => panic!("expected Progress for b, got {other:?}"),
        }
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
                assert_eq!(paths, vec!["root/a", "root/b"]);
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_sibling_rides_along_in_batch() {
        // A completes Ok; B was closed (never delivered a real result).
        // The parent must learn B is gone in the same wake.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await;

        finish_and_post(&fx, "b", MailboxStatus::Closed, None);
        let _ = next_event(&mut fx).await; // Progress b
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2);
                assert_eq!(reports[0].status, "ok");
                assert_eq!(reports[1].status, "closed");
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_only_batch_never_wakes_parent() {
        // The user closed every child; no real result ever arrived.
        // Progress is fine, but no Batch may fire.
        let mut fx = fixture();
        spawn_running(&fx, "w");

        finish_and_post(&fx, "w", MailboxStatus::Closed, None);
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { status, .. } => assert_eq!(status, "closed"),
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // no Batch
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn redundant_closed_is_dropped_while_held() {
        // A completes, then the user closes it before B finishes. The close
        // notification must not duplicate A in the eventual batch.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a

        // Close B's sibling A again: post a redundant Closed (the registry
        // slot is still held by B, so the batch stays pending).
        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Closed,
            result: None,
            denied_tools: vec![],
        });
        assert_no_event(&mut fx).await; // dropped — no duplicate Progress

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        let _ = next_event(&mut fx).await; // Progress b
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2, "a + b, no duplicate of either");
                assert!(
                    reports.iter().all(|r| r.status == "ok"),
                    "redundant Closed must not ride along"
                );
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn stale_closed_after_flush_does_not_wake() {
        // A's Ok batch flushed; a late Closed for A arrives (cleanup race).
        // Lone Closed → Progress only, never a Batch.
        let mut fx = fixture();
        fx.mailbox.register(&AgentPath::root().join("a"));

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress
        let _ = next_event(&mut fx).await; // Batch

        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Closed,
            result: None,
            denied_tools: vec![],
        });
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { .. } => {}
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // lone Closed → no Batch
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn batch_flushed_results_are_not_repeated() {
        // After a flush, the batch is empty: a second generation of results
        // produces its own batch, without stale reports from the first.
        let mut fx = fixture();
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("gen1"));
        let _ = next_event(&mut fx).await;
        let _ = next_event(&mut fx).await; // Batch gen1

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("gen2"));
        let _ = next_event(&mut fx).await;
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert!(reports[0].message.contains("gen2"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    /// Variant-B regression (session 20260914_50cf809d): a held report must
    /// fire once the force-closed sibling is marked closing — even though it
    /// will never deliver a real result. Before the `closing` fact, the
    /// killed child's `in_flight=true, results_posted=0` blocked quiescence
    /// forever and the report was lost (the parent's final review was then
    /// written blind, claiming "both specialists completed").
    #[tokio::test]
    async fn force_closed_child_does_not_block_batch() {
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        // a finishes; held because b is in-flight with no result.
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await;

        // The parent force-closes b: terminal fact lands at cancel time,
        // then the child's unwind eventually posts Closed (the wake).
        fx.registry
            .lock()
            .unwrap()
            .note_closing(&AgentPath::root().join("b"));
        finish_and_post(&fx, "b", MailboxStatus::Closed, None);
        let _ = next_event(&mut fx).await; // Progress b (closed)

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
                assert!(
                    paths.contains(&"root/a"),
                    "a's held report must be delivered, got {paths:?}"
                );
            }
            other => panic!("expected Batch after close, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    /// A watcher's death must never bury held reports: on cancel the batch
    /// is flushed to the parent (session 20260914_50cf809d — the stranded
    /// batch died with the process).
    #[tokio::test]
    async fn watcher_shutdown_flush_delivers_held_batch() {
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        // a's report held: b is in-flight and (in this test) never settles —
        // no closing fact, no Closed post, no wake. Only the shutdown flush
        // can deliver it.
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await;

        fx.cancel.cancel();
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
                assert_eq!(paths, vec!["root/a"], "held report flushed on shutdown");
            }
            other => panic!("expected Batch on shutdown flush, got {other:?}"),
        }
    }

    /// The reaper backstop: a batch held past the reap window while no agent
    /// can still post is handed over even without any wake (here: b was
    /// closed and its unwind never posts anything).
    #[tokio::test]
    async fn held_batch_reaper_fires_when_nothing_can_post() {
        let mut fx = fixture_with(Config {
            summarizer: None,
            held_reap_after: std::time::Duration::from_millis(100),
        });
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await; // held — b in-flight

        // b force-closed; its Closed post NEVER arrives (unwind never lands).
        fx.registry
            .lock()
            .unwrap()
            .note_closing(&AgentPath::root().join("b"));

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
                assert_eq!(paths, vec!["root/a"], "reaper must deliver the held report");
            }
            other => panic!("expected reaper Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn watcher_exits_on_cancel() {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let handle = spawn_watcher(
            mailbox.clone(),
            registry,
            None,
            Some(tx),
            cancel.clone(),
            HELD_BATCH_REAP_AFTER,
        );
        cancel.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "watcher should exit on cancel");
    }

    #[tokio::test]
    async fn watchdog_exits_on_cancel() {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let handle = spawn_watcher_with_watchdog(
            mailbox.clone(),
            registry,
            None,
            Some(tx),
            cancel.clone(),
            HELD_BATCH_REAP_AFTER,
        );
        cancel.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle).await;
        assert!(result.is_ok(), "watchdog should exit on cancel");
    }

    #[tokio::test]
    async fn concurrent_results_all_land_in_one_batch() {
        // Two results posted before the watcher processes — both Progress
        // events fire, and one Batch carries both reports.
        let mut fx = fixture();
        fx.mailbox.register(&AgentPath::root().join("a"));
        fx.mailbox.register(&AgentPath::root().join("b"));

        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Ok,
            result: Some("a done".to_string()),
            denied_tools: vec![],
        });
        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("b"),
            status: MailboxStatus::Error,
            result: Some("b failed".to_string()),
            denied_tools: vec![],
        });

        let mut statuses = Vec::new();
        for _ in 0..2 {
            match next_event(&mut fx).await {
                ChildResultEvent::Progress {
                    agent_path, status, ..
                } => {
                    assert_ne!(agent_path, "root/none");
                    statuses.push(status);
                }
                other => panic!("expected Progress, got {other:?}"),
            }
        }
        assert!(statuses.contains(&"ok".to_string()));
        assert!(statuses.contains(&"error".to_string()));

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2);
                assert!(reports.iter().any(|r| r.status == "ok"));
                assert!(reports.iter().any(|r| r.status == "error"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    // ── Focus summarizer integration ──

    /// Minimal mock provider: `chat()` sleeps `delay`, then answers a
    /// canned summary JSON. The delay models slow providers (mimo took
    /// 21-30 s per one-sentence summary in session 20260903_d8fc41dc),
    /// scaled down for tests.
    struct SummaryStub {
        delay: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl agent_base::llm_trait::LlmProvider for SummaryStub {
        async fn stream(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
            Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::empty(),
            )))
        }

        async fn chat(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            Ok(agent_base::llm_trait::ChatResponse {
                content: r#"{"summary": "mock 摘要"}"#.to_string(),
                tool_calls: vec![],
                usage: Default::default(),
                finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
                raw: None,
                reasoning_content: None,
                thinking_signature: None,
            })
        }

        fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
            Default::default()
        }

        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "summary-stub".to_string(),
                model: "stub".to_string(),
                version: None,
            }
        }
    }

    #[tokio::test]
    async fn progress_summary_flows_from_summarizer() {
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "a");
        fx.registry
            .lock()
            .unwrap()
            .set_task(&AgentPath::root().join("a"), "分析任务".to_string());

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("report text"));

        // Three events, order-tolerant except for the plain-first rule: a
        // synchronous plain Progress (summary None) precedes everything, the
        // detached summary Progress and the Batch race each other, and each
        // must arrive with its payload.
        let mut saw_plain = false;
        let mut saw_summary = false;
        let mut batch_ok = false;
        for _ in 0..3 {
            match next_event(&mut fx).await {
                ChildResultEvent::Progress {
                    status, summary, ..
                } => {
                    assert_eq!(status, "ok");
                    match summary.as_deref() {
                        None => saw_plain = true,
                        Some("mock 摘要") => saw_summary = true,
                        other => panic!("unexpected summary {other:?}"),
                    }
                }
                ChildResultEvent::Batch { reports } => {
                    // The batch still carries the full report — Focus does
                    // not compress.
                    assert_eq!(reports.len(), 1);
                    assert_eq!(reports[0].result.as_deref(), Some("report text"));
                    batch_ok = true;
                }
            }
        }
        assert!(
            saw_plain,
            "synchronous plain Progress must arrive first-class"
        );
        assert!(saw_summary, "Progress with summary must arrive");
        assert!(batch_ok, "Batch with full report must arrive");
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn slow_summary_does_not_delay_batch() {
        // The regression for session 20260903_d8fc41dc: a slow summary used
        // to be awaited on the wake path, adding its full timeout to every
        // batch. Now the Batch must beat a slow summary by a wide margin.
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::from_millis(500),
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "a");
        let started = std::time::Instant::now();
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("report text"));

        // First: the synchronous plain Progress (summary None) — the child's
        // return is announced immediately, never held hostage by Focus.
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { summary, .. } => {
                assert!(summary.is_none(), "first notice must be the plain one");
            }
            other => panic!("expected plain Progress first, got {other:?}"),
        }

        // Second: the Batch, well under the summary's 500 ms delay.
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
            }
            ChildResultEvent::Progress {
                summary: Some(_), ..
            } => panic!("summary raced ahead of Batch"),
            ChildResultEvent::Progress { summary: None, .. } => {
                panic!("only one plain Progress expected")
            }
        }
        assert!(
            started.elapsed() < std::time::Duration::from_millis(400),
            "Batch must not wait for the summary (took {:?})",
            started.elapsed()
        );

        // The summary still lands afterwards for the user.
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { summary, .. } => {
                assert_eq!(summary.as_deref(), Some("mock 摘要"));
            }
            other => panic!("expected late Progress, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_progress_skips_summarizer() {
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "w");
        finish_and_post(&fx, "w", MailboxStatus::Closed, None);

        match next_event(&mut fx).await {
            ChildResultEvent::Progress {
                status, summary, ..
            } => {
                assert_eq!(status, "closed");
                assert!(summary.is_none(), "closed needs no LLM call");
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // lone Closed → no Batch
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn error_progress_carries_raw_reason_not_a_paraphrase() {
        // Session 20260906_0f6d4341: an Error child used to surface as a
        // terse "execution error" plus a detached Focus paraphrase, while the raw
        // error ("LLM call failed: HTTP request failed: …") went only to
        // the parent. Now the synchronous Progress must carry the raw
        // first line itself, and Focus must not be called for errors —
        // exactly one Progress, no follow-up.
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "e");
        finish_and_post(
            &fx,
            "e",
            MailboxStatus::Error,
            Some(
                "LLM call failed: HTTP request failed: error sending request for url (https://example.invalid/v1/messages)",
            ),
        );

        match next_event(&mut fx).await {
            ChildResultEvent::Progress {
                status, summary, ..
            } => {
                assert_eq!(status, "error");
                let reason = summary.expect("error Progress must carry the raw reason");
                assert!(reason.starts_with("LLM call failed"), "{reason}");
                assert!(reason.contains("error sending request"), "{reason}");
                assert!(!reason.contains("mock 摘要"), "paraphrase leaked: {reason}");
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        // The lone Error still wakes the parent (unlike lone Closed), with
        // the full raw error in the report — then nothing further (no
        // paraphrase Progress must follow).
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert_eq!(
                    reports[0].result.as_deref(),
                    Some(
                        "LLM call failed: HTTP request failed: error sending request for url (https://example.invalid/v1/messages)"
                    )
                );
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        assert_no_event(&mut fx).await;
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn error_reason_truncates_and_skips_blank_lines() {
        let long = format!("{}…tail", "x".repeat(300));
        // Blank lines are skipped; the FIRST content line wins over later
        // ones (real errors are multi-line: "LLM call failed:\n  caused by").
        assert_eq!(
            error_reason(Some("\n  \nboom: first line is the meat")),
            Some("boom: first line is the meat".to_string())
        );
        assert_eq!(
            error_reason(Some("first line\nsecond line\nthird")),
            Some("first line".to_string())
        );
        // Exact boundary: 160 chars passes through untouched, 161 gains the
        // ellipsis (pins off-by-one at MAX_ERROR_REASON_CHARS).
        let at = "x".repeat(MAX_ERROR_REASON_CHARS);
        let r = error_reason(Some(&at)).unwrap();
        assert_eq!(r.chars().count(), MAX_ERROR_REASON_CHARS);
        assert!(!r.ends_with('…'));
        let over = "x".repeat(MAX_ERROR_REASON_CHARS + 1);
        let r = error_reason(Some(&over)).unwrap();
        assert_eq!(r.chars().count(), MAX_ERROR_REASON_CHARS + 1);
        assert!(r.ends_with('…'));
        // Far over the limit: 300 chars → 160 + ellipsis, tail dropped.
        let truncated = error_reason(Some(&long)).unwrap();
        assert_eq!(truncated.chars().count(), 161, "160 chars + ellipsis");
        assert!(truncated.ends_with('…'));
        assert_eq!(error_reason(Some("  \n\n  ")), None);
        assert_eq!(error_reason(None), None);
    }

    #[tokio::test]
    async fn error_reason_neutralizes_control_chars_and_ansi() {
        // Provider error bodies are gateway-controlled bytes headed for the
        // TUI: escape sequences must not paint terminal state (or leave
        // "[31m" debris behind a dropped ESC), CR must not splice text, and
        // a pure-control first line must not shadow the real reason.
        assert_eq!(
            error_reason(Some("\x1b[31mError: boom\x1b[0m")),
            Some("Error: boom".to_string())
        );
        // CR mid-line becomes a space instead of a silent splice (A\rB → AB).
        assert_eq!(error_reason(Some("A\rB")), Some("A B".to_string()));
        // Tab stays readable rather than gluing words together.
        assert_eq!(
            error_reason(Some("error:\tboom")),
            Some("error: boom".to_string())
        );
        // A control-garbage first line is skipped, not shown invisibly.
        assert_eq!(
            error_reason(Some("\x07\nreal reason")),
            Some("real reason".to_string())
        );
        // Sanitization composes with truncation (still bounded at 160+1).
        let dirty = format!("\x1b[1m{}", "y".repeat(300));
        let r = error_reason(Some(&dirty)).unwrap();
        assert_eq!(r.chars().count(), 161);
        assert!(r.ends_with('…'));
    }

    #[tokio::test]
    async fn error_progress_with_none_result_falls_back_to_plain_notice() {
        // Error with no result text: error_reason(None) → the Progress
        // carries summary: None and the consumer shows its terse fallback
        // notice. Focus still must not run, and the parent still gets the
        // batch with the None report.
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "e");
        finish_and_post(&fx, "e", MailboxStatus::Error, None);

        match next_event(&mut fx).await {
            ChildResultEvent::Progress {
                status, summary, ..
            } => {
                assert_eq!(status, "error");
                assert!(
                    summary.is_none(),
                    "None error text must fall back to a plain notice"
                );
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert_eq!(reports[0].result, None);
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        assert_no_event(&mut fx).await;
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn fresh_spawn_without_delivery_blocks_the_batch() {
        // The spawn→send window: b is registered (spawn returned) but its
        // send_task is still in flight — zero deliveries. a's result must
        // Progress but NOT fire a batch that excludes b. Pins the delivery
        // clause of `quiescent()` at the layer that consumes it.
        let mut fx = fixture();
        spawn_running(&fx, "a");
        let b = AgentPath::root().join("b");
        fx.mailbox.register(&b);
        fx.registry.lock().unwrap().register(&b, 1).unwrap();

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await; // b never delivered — batch held

        // b's task lands and the child loop dequeues it.
        {
            let mut reg = fx.registry.lock().unwrap();
            reg.note_enqueued(&b);
            reg.note_dequeued(&b);
        }
        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        let _ = next_event(&mut fx).await; // Progress b

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => assert_eq!(
                reports
                    .iter()
                    .map(|r| r.agent_path.as_str())
                    .collect::<Vec<_>>(),
                vec!["root/a", "root/b"]
            ),
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn close_during_queued_task_drops_phantom_queue_from_quiescence() {
        // Close lands while a task sits queued (sent, never dequeued). The
        // registry entry — and its queue_len — must vanish with the close,
        // and the late notes the child loop may still issue must be no-ops.
        let mut fx = fixture();
        let b = AgentPath::root().join("b");
        fx.mailbox.register(&b);
        {
            let mut reg = fx.registry.lock().unwrap();
            reg.register(&b, 1).unwrap();
            reg.note_enqueued(&b);
        }
        assert!(!fx.registry.lock().unwrap().quiescent());

        fx.registry.lock().unwrap().close(&b); // close wins the race
        assert!(
            fx.registry.lock().unwrap().quiescent(),
            "phantom queue must not outlive the entry"
        );

        // Cleanup's Closed delivery; the child loop's note_posted is a no-op
        // on the removed entry.
        finish_and_post(&fx, "b", MailboxStatus::Closed, None);
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { status, .. } => assert_eq!(status, "closed"),
            other => panic!("expected Progress(closed), got {other:?}"),
        }
        assert_no_event(&mut fx).await;
        fx.cancel.cancel();
    }
}
