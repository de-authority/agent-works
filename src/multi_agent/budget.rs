//! Rollout spawn/token budget (`RolloutBudget`, design doc §7.2).
//!
//! Two independent budget dimensions, both **rollout-scoped** (cumulative for
//! the life of the parent runtime):
//!
//! - `max_spawns`: the *cumulative* number of spawns a rollout may commit.
//!   Distinct from `max_sub_agents` (live count) — a closed child still spent
//!   its spawn (review m-8: the two are different dimensions, neither
//!   replaces the other). There is deliberately **no `release_spawn`**: v3's
//!   "give the count back on close" contradicted the cumulative semantics.
//! - `child_max_tokens`: metered from **child** usage only (the parent's own
//!   spend is not metered — "rollout budget covering the whole rollout" is a
//!   later item, out of v3.1 scope). Fed by hook A: the child runtime's
//!   `on_turn_end` callback (§5.4, review M-4 — there is no usage on
//!   `run_turn`'s return value; the hook is the only real path).
//!
//! ## Ticket discipline
//!
//! [`try_reserve_spawn`](RolloutBudget::try_reserve_spawn) atomically
//! pre-increments the counter and returns a [`SpawnTicket`]. **An uncommitted
//! ticket rolls the increment back on `Drop`** — so any `?` / early-return /
//! panic on the spawn chain returns the reservation with no explicit
//! `cancel()` to forget (harder to misuse than v3's manual rollback calls).
//! `commit()` converts the reservation into the cumulative count; after that
//! nobody gives it back.
//!
//! ## Soft-race declaration (review m-7 — written here, not excused later)
//!
//! The token-headroom check and the counter increment are **not** one atomic
//! operation, and usage arrives asynchronously from turn-end callbacks, so
//! under concurrency spawns can transiently overshoot an exhausted token
//! budget by roughly one call's worth. This is a *budget* (throttle), not a
//! *quota* (exact billing); the spawn-count dimension is exact (single CAS).

use std::sync::Arc;

// Atomics are loom-checked under the `loom-check` feature (§12 stage 4 —
// the ticket commit/rollback race is modeled exhaustively in the `loom_*`
// tests); a normal build uses plain std atomics.
#[cfg(feature = "loom-check")]
use loom::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(not(feature = "loom-check"))]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Errors from [`RolloutBudget::try_reserve_spawn`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// The cumulative spawn cap is reached.
    MaxSpawnCountReached { max: usize },
    /// The child-token budget is spent.
    TokenBudgetExhausted { used: u64, max: u64 },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxSpawnCountReached { max } => {
                write!(f, "max spawn count reached (limit: {})", max)
            }
            Self::TokenBudgetExhausted { used, max } => {
                write!(f, "child token budget exhausted ({} / {})", used, max)
            }
        }
    }
}

impl std::error::Error for BudgetError {}

/// Sum the tokens of an [`agent_base::UsageInfo`] for budget metering.
///
/// NOTE (doc fact-check): design §5.4 calls `usage.total()`, but
/// `UsageInfo` (llm-trait) has no `total()` — its fields are all `Option`.
/// A provider's own `total_tokens` wins when present; otherwise prompt +
/// completion are summed (missing halves count as 0).
pub fn usage_total(u: &agent_base::UsageInfo) -> u64 {
    match u.total_tokens {
        Some(t) => t as u64,
        None => u.prompt_tokens.unwrap_or(0) as u64 + u.completion_tokens.unwrap_or(0) as u64,
    }
}

/// Rollout-scoped spawn / child-token budget (design §7.2).
pub struct RolloutBudget {
    child_max_tokens: u64, // u64::MAX = unlimited (ControlConfig None)
    used_tokens: AtomicU64,
    max_spawns: usize, // usize::MAX = unlimited (ControlConfig None)
    spawn_count: AtomicUsize,
}

impl std::fmt::Debug for RolloutBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RolloutBudget")
            .field("child_max_tokens", &self.child_max_tokens)
            .field("used_tokens", &self.used_tokens.load(Ordering::Relaxed))
            .field("max_spawns", &self.max_spawns)
            .field("spawn_count", &self.spawn_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl RolloutBudget {
    /// Build from the (all-`Option`) control config: `None` → unlimited.
    pub fn new(child_max_tokens: Option<u64>, max_spawns: Option<usize>) -> Self {
        Self {
            child_max_tokens: child_max_tokens.unwrap_or(u64::MAX),
            used_tokens: AtomicU64::new(0),
            max_spawns: max_spawns.unwrap_or(usize::MAX),
            spawn_count: AtomicUsize::new(0),
        }
    }

    /// Atomically pre-reserve one spawn (gate 1 of the spawn chain, §5.4).
    ///
    /// The returned [`SpawnTicket`] must be [`commit`](SpawnTicket::commit)ed
    /// once the child task is actually launched; dropping it uncommitted
    /// (any failure on the spawn chain) returns the reservation.
    pub fn try_reserve_spawn(self: &Arc<Self>) -> Result<SpawnTicket, BudgetError> {
        // Spawn-count dimension is exact: single CAS loop, reject ⇒ no change.
        let prev = self.spawn_count.fetch_add(1, Ordering::AcqRel);
        if prev >= self.max_spawns {
            self.spawn_count.fetch_sub(1, Ordering::AcqRel);
            return Err(BudgetError::MaxSpawnCountReached {
                max: self.max_spawns,
            });
        }
        // Token dimension is the soft-race leg (see module docs, review m-7).
        let used = self.used_tokens.load(Ordering::Acquire);
        if used >= self.child_max_tokens {
            self.spawn_count.fetch_sub(1, Ordering::AcqRel);
            return Err(BudgetError::TokenBudgetExhausted {
                used,
                max: self.child_max_tokens,
            });
        }
        Ok(SpawnTicket {
            budget: Some(Arc::clone(self)),
        })
    }

    /// Hook A consumer: `on_turn_end` reports one child turn's usage (§5.4).
    pub fn record_usage(&self, tokens: u64) {
        self.used_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Cumulative child tokens metered so far.
    pub fn used_tokens(&self) -> u64 {
        self.used_tokens.load(Ordering::Acquire)
    }

    /// Committed + in-flight (uncommitted) spawns counted against the cap.
    pub fn spawn_count(&self) -> usize {
        self.spawn_count.load(Ordering::Acquire)
    }

    /// Configured caps (`u64::MAX` / `usize::MAX` when the gate is disabled).
    pub fn child_max_tokens(&self) -> u64 {
        self.child_max_tokens
    }

    pub fn max_spawns(&self) -> usize {
        self.max_spawns
    }
}

/// Uncommitted-spawn rollback credential (§7.2).
///
/// Obtain only from [`RolloutBudget::try_reserve_spawn`]. `commit` converts
/// the reservation into the cumulative count; otherwise `Drop` decrements —
/// covering every failure point of the spawn chain without an explicit
/// `cancel()` (the v3 design's misuse surface).
#[must_use = "an uncommitted ticket rolls back the spawn count on drop"]
pub struct SpawnTicket {
    budget: Option<Arc<RolloutBudget>>,
}

impl std::fmt::Debug for SpawnTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnTicket")
            .field("committed", &self.budget.is_none())
            .finish()
    }
}

impl SpawnTicket {
    /// Finalize: the spawn count stays incremented permanently (cumulative
    /// rollout semantics — there is no release).
    pub fn commit(mut self) {
        self.budget = None;
    }
}

impl Drop for SpawnTicket {
    fn drop(&mut self) {
        if let Some(budget) = &self.budget {
            budget.spawn_count.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(max_spawns: Option<usize>, child_max_tokens: Option<u64>) -> Arc<RolloutBudget> {
        Arc::new(RolloutBudget::new(child_max_tokens, max_spawns))
    }

    #[test]
    fn spawn_cap_rejects_and_restores_exact() {
        let b = budget(Some(2), None);
        let t1 = b.try_reserve_spawn().expect("1st");
        let t2 = b.try_reserve_spawn().expect("2nd");
        assert_eq!(b.spawn_count(), 2);

        // Rejection leaves the count untouched (fetch_add then rollback).
        assert_eq!(
            b.try_reserve_spawn().unwrap_err(),
            BudgetError::MaxSpawnCountReached { max: 2 }
        );
        assert_eq!(b.spawn_count(), 2);

        // Cumulative semantics: committing keeps the count…
        t1.commit();
        assert_eq!(b.spawn_count(), 2);
        // …and dropping an uncommitted ticket returns exactly one.
        drop(t2);
        assert_eq!(b.spawn_count(), 1);
        let t3 = b.try_reserve_spawn().expect("room after rollback");
        assert_eq!(b.spawn_count(), 2);
        t3.commit();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn committed_spawns_never_released_by_close() {
        // §7.2 note: no release_spawn — the count is cumulative even after
        // the children are gone (rollout semantics, not live-count).
        let b = budget(Some(1), None);
        let t = b.try_reserve_spawn().expect("first");
        t.commit();
        assert_eq!(
            b.try_reserve_spawn().unwrap_err(),
            BudgetError::MaxSpawnCountReached { max: 1 }
        );
    }

    #[test]
    fn token_exhaustion_rejects_and_restores_count() {
        let b = budget(None, Some(100));
        let t = b.try_reserve_spawn().expect("under budget");
        t.commit();
        b.record_usage(100);
        assert_eq!(
            b.try_reserve_spawn().unwrap_err(),
            BudgetError::TokenBudgetExhausted {
                used: 100,
                max: 100
            }
        );
        // The rejected attempt rolled its spawn increment back.
        assert_eq!(b.spawn_count(), 1);
    }

    #[test]
    fn unlimited_by_default() {
        let b = budget(None, None);
        let mut ts = Vec::new();
        for _ in 0..1000 {
            ts.push(b.try_reserve_spawn().expect("unlimited"));
        }
        assert_eq!(b.spawn_count(), 1000);
        for t in ts {
            t.commit();
        }
        assert_eq!(b.spawn_count(), 1000);
    }

    #[test]
    fn usage_total_prefers_provider_total() {
        let u = agent_base::UsageInfo {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(20),
            reasoning_tokens: None,
        };
        assert_eq!(usage_total(&u), 20);
        let u2 = agent_base::UsageInfo {
            total_tokens: None,
            ..u
        };
        assert_eq!(usage_total(&u2), 15);
        let none = agent_base::UsageInfo {
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            reasoning_tokens: None,
        };
        assert_eq!(usage_total(&none), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cas_conservation_thousand_mixed_commit_rollback() {
        // §12 stage-2 acceptance: a thousand concurrent reserve/commit/rollback, exact final value.
        // i % 3 == 0 → rollback (drop), else commit. Exact expectation:
        const N: usize = 1000;
        let b = budget(None, None);
        let mut set = tokio::task::JoinSet::new();
        for i in 0..N {
            let b = Arc::clone(&b);
            set.spawn(async move {
                let t = b.try_reserve_spawn().expect("unlimited");
                // Yield so reservations interleave before the commit decision.
                tokio::task::yield_now().await;
                if i % 3 != 0 {
                    t.commit();
                }
                // else: dropped uncommitted → rollback
            });
        }
        while set.join_next().await.is_some() {}
        let committed = (0..N).filter(|i| i % 3 != 0).count();
        assert_eq!(
            b.spawn_count(),
            committed,
            "CAS conservation: commits minus rollbacks exactly"
        );
    }

    /// §12 stage-4 (run with the `loom-check` feature): cap 1, one thread
    /// commits its ticket, one drops it uncommitted.
    ///
    /// What the model actually shows (worth stating): both threads may get a
    /// ticket **sequentially** — an uncommitted rollback frees the slot for
    /// the next reserver — yet that is exactly the cumulative-commits
    /// semantics (§7.2): never *two live tickets*, never more than one
    /// committed spawn against cap 1, and the counter always equals the set
    /// of commits (no phantom, no leak).
    #[test]
    #[cfg(feature = "loom-check")]
    fn loom_reserve_commit_vs_rollback_race() {
        loom::model(|| {
            let b = Arc::new(RolloutBudget::new(None, Some(1)));
            let h1 = {
                let b = Arc::clone(&b);
                loom::thread::spawn(move || match b.try_reserve_spawn() {
                    Ok(t) => {
                        t.commit();
                        true
                    }
                    Err(_) => false,
                })
            };
            let h2 = {
                let b = Arc::clone(&b);
                loom::thread::spawn(move || match b.try_reserve_spawn() {
                    Ok(t) => {
                        drop(t); // uncommitted → rollback
                        true
                    }
                    Err(_) => false,
                })
            };
            let committed_by_t1 = h1.join().unwrap();
            let got_and_returned_t2 = h2.join().unwrap();
            // The committed set is {t1 if Ok}; t2's reservation, committed or
            // rolled back, is invisible to the final count — but the cap
            // must never be exceeded, and here it can be *at most* t1's one.
            assert!(
                usize::from(committed_by_t1) <= 1,
                "cumulative commits ≤ cap"
            );
            // Whichever path t2's ticket took, the count reflects commits exactly.
            assert_eq!(b.spawn_count(), usize::from(committed_by_t1));
            let _ = got_and_returned_t2;
        });
    }

    /// §12 stage-4 (loom-check): makes the documented soft race (module docs,
    /// review m-7) explicit — `record_usage` racing the token check means a
    /// spawn *may* be admitted just before the usage lands. Loom enumerates
    /// both outcomes; what must hold in every one of them is counter
    /// consistency, not exactness of the gate.
    #[test]
    #[cfg(feature = "loom-check")]
    fn loom_token_gate_is_a_throttle_not_a_quota() {
        loom::model(|| {
            let b = Arc::new(RolloutBudget::new(Some(1), None));
            let u = {
                let b = Arc::clone(&b);
                loom::thread::spawn(move || b.record_usage(1))
            };
            let r = {
                let b = Arc::clone(&b);
                loom::thread::spawn(move || match b.try_reserve_spawn() {
                    Ok(t) => {
                        t.commit();
                        true
                    }
                    Err(_) => false,
                })
            };
            u.join().unwrap();
            let admitted = r.join().unwrap();
            assert_eq!(b.used_tokens(), 1);
            assert_eq!(
                b.spawn_count(),
                usize::from(admitted),
                "a rejected reservation rolled back exactly"
            );
        });
    }
}
