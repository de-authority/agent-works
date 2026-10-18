//! Approval ledger + fail-closed gate (P1) — safety-line normalization for external tools.
//!
//! Normative text: see `docs/tool-contract.md` §2 (signed together with the P0 contract). Key points:
//!
//! - **Tiers** (persistent): `ask every time` (default) / `always allow` / `deny`.
//!   `allow once only` is a one-shot action, not a tier;
//! - **Ledger before mode**: three-prefix tools under tier `ask every time`
//!   **prompt for confirmation even with Shift+Tab=auto**; Shift+Tab=deny is a global
//!   brake that overrides the ledger;
//! - **fail-closed**: the three prefixes (`mcp.`/`script.`/`plugin.`) are never
//!   short-circuited by mode into automatic allow; unrecognized keys likewise fall to
//!   "needs confirmation";
//! - **Single memory owner**: `always allow` is only recorded in the ledger; the engine's
//!   action_key `AllowAlways` cache only serves exact actions of builtin tools — external
//!   tool decisions are downgraded to `AllowOnce` when handed to the engine via
//!   [`engine_decision_for_external`], never double-written.
//!
//! Storage injection: this module only honors the path given by the caller of
//! [`ApprovalLedger::load`] / [`ApprovalLedger::save`], **it does not sniff product paths**
//! (`~/.phimint/approvals.json` is the product layer's business).

use std::collections::BTreeMap;
use std::path::Path;

use agent_base::ApprovalDecision;
use serde::{Deserialize, Serialize};

/// Ledger tier. Default is [`ApprovalTier::Ask`] (ask every time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalTier {
    /// Ask every time (default).
    #[default]
    Ask,
    /// Always allow (set by the dialog's `[Always allow]` or `/tools`).
    AlwaysAllow,
    /// Deny (set via `/tools`; the dialog's `[Deny this time]` does not change the tier).
    Deny,
}

/// Shift+Tab mode switch (same vocabulary as the product's `RuntimeApprovalMode`, converted at the product layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    Auto,
    Ask,
    Deny,
}

/// The fail-closed gate's verdict for a single external tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalGate {
    /// Allow (no more asking the human).
    Allow,
    /// Deny.
    Deny,
    /// **Prompt** (goes through the ApprovalPrompt callback; a human gives the nod).
    Prompt,
}

/// Whether an identity key belongs to a three-prefix external tool (`mcp.`/`script.`/`plugin.`).
///
/// Prefix matching (including malformed long-tail keys): better to pull weird keys into
/// "needs confirmation" too, never auto-allow.
pub fn is_external_identity_key(key: &str) -> bool {
    key.starts_with("mcp.") || key.starts_with("script.") || key.starts_with("plugin.")
}

/// Decision table for external tools (the approval priority table, normative MUST).
///
/// | tier \\ mode | Auto | Ask | Deny |
/// |---|---|---|---|
/// | Ask | **Prompt** (ledger before mode) | Prompt | Deny (brake) |
/// | AlwaysAllow | Allow | Allow | Deny (brake) |
/// | Deny | Deny | Deny | Deny |
pub fn gate_decision(tier: ApprovalTier, mode: ApprovalMode) -> ExternalGate {
    if mode == ApprovalMode::Deny {
        return ExternalGate::Deny; // global brake overrides the ledger
    }
    match tier {
        ApprovalTier::Ask => ExternalGate::Prompt, // no exception even in auto mode
        ApprovalTier::AlwaysAllow => ExternalGate::Allow,
        ApprovalTier::Deny => ExternalGate::Deny,
    }
}

/// Single memory owner: `[Always allow]` for external tools is recorded in the ledger;
/// the engine's action_key `AllowAlways` cache must not be double-written — downgraded
/// to `AllowOnce` when handed to the engine.
/// Builtin tool decisions are passed through unchanged (the action_key cache keeps serving exact actions).
pub fn engine_decision_for_external(decision: ApprovalDecision) -> ApprovalDecision {
    match decision {
        ApprovalDecision::AllowAlways => ApprovalDecision::AllowOnce,
        other => other,
    }
}

/// Approval ledger: identity key → tier.
///
/// JSON (the shape of `~/.phimint/approvals.json`, machine-written):
/// ```json
/// { "schema_version": 1, "tiers": { "script.deploy": "always_allow" } }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalLedger {
    #[serde(default = "schema_version")]
    schema_version: u32,
    /// Identity key → tier. No entry = [`ApprovalTier::Ask`].
    #[serde(default)]
    tiers: BTreeMap<String, ApprovalTier>,
}

impl Default for ApprovalLedger {
    fn default() -> Self {
        Self {
            schema_version: schema_version(),
            tiers: BTreeMap::new(),
        }
    }
}

fn schema_version() -> u32 {
    1
}

impl ApprovalLedger {
    /// Empty ledger (everything defaults to `ask every time`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the tier: no entry means the default tier `ask every time` (fail-closed direction).
    pub fn tier_of(&self, identity_key: &str) -> ApprovalTier {
        self.tiers
            .get(identity_key)
            .copied()
            .unwrap_or(ApprovalTier::Ask)
    }

    /// Record a decision with three-button semantics:
    /// - `[Allow once only]` → allow this call, tier unchanged;
    /// - `[Always allow]` → raise the tier to `Always allow` + allow this call;
    /// - `[Deny this time]` → deny this call, tier unchanged (permanent deny goes through [`set_tier`]).
    ///
    /// Returns whether the tier changed.
    pub fn record_decision(&mut self, identity_key: &str, decision: &ApprovalDecision) -> bool {
        match decision {
            ApprovalDecision::AllowOnce | ApprovalDecision::Deny => false,
            ApprovalDecision::AllowAlways => {
                self.tiers
                    .insert(identity_key.to_string(), ApprovalTier::AlwaysAllow);
                true
            }
        }
    }

    /// Set the tier directly (`/tools` management surface; permanent deny = [`ApprovalTier::Deny`]).
    pub fn set_tier(&mut self, identity_key: &str, tier: ApprovalTier) {
        self.tiers.insert(identity_key.to_string(), tier);
    }

    /// Clear one ledger entry (`/tools remove <name>`: cleanup after deleting `script.{name}`,
    /// leaving no orphaned memory). Returns whether anything was removed.
    pub fn remove(&mut self, identity_key: &str) -> bool {
        self.tiers.remove(identity_key).is_some()
    }

    /// Sweep by prefix (`plugin remove`: clear all `plugin.{name}.*` entries).
    /// Returns the number of deleted entries.
    pub fn remove_prefix(&mut self, prefix: &str) -> usize {
        let keys: Vec<String> = self
            .tiers
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let n = keys.len();
        for k in keys {
            self.tiers.remove(&k);
        }
        n
    }

    /// Number of entries (for tests/panels).
    pub fn len(&self) -> usize {
        self.tiers.len()
    }

    /// Whether the ledger is empty.
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Load from the injected path. Missing file = brand-new ledger (Ok); parse failure = error
    /// (never silently pretend to be an empty ledger — the user's recorded "always allow" must
    /// not silently evaporate).
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(raw) => serde_json::from_str(&raw).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("approvals ledger {path:?} is corrupt: {e}"),
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(e),
        }
    }

    /// Write back to the injected path (machine-written, pure JSON).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let raw = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fail-closed decision matrix (SC6 test criteria live here) ─────────────────────────

    #[test]
    fn sc6_ask_tier_prompts_even_in_auto_mode() {
        // Tier `ask every time` + Shift+Tab auto → **still prompts**.
        assert_eq!(
            gate_decision(ApprovalTier::Ask, ApprovalMode::Auto),
            ExternalGate::Prompt
        );
    }

    #[test]
    fn gate_matrix_matches_priority_table() {
        use ApprovalMode as M;
        use ApprovalTier as T;
        let rows = [
            (T::Ask, M::Auto, ExternalGate::Prompt),
            (T::Ask, M::Ask, ExternalGate::Prompt),
            (T::Ask, M::Deny, ExternalGate::Deny),
            (T::AlwaysAllow, M::Auto, ExternalGate::Allow),
            (T::AlwaysAllow, M::Ask, ExternalGate::Allow),
            (T::AlwaysAllow, M::Deny, ExternalGate::Deny),
            (T::Deny, M::Auto, ExternalGate::Deny),
            (T::Deny, M::Ask, ExternalGate::Deny),
            (T::Deny, M::Deny, ExternalGate::Deny),
        ];
        for (tier, mode, want) in rows {
            assert_eq!(
                gate_decision(tier, mode),
                want,
                "tier={tier:?} mode={mode:?}"
            );
        }
    }

    #[test]
    fn deny_brake_overrides_standing_allow() {
        // Shift+Tab=deny global brake overrides the ledger's "always allow".
        assert_eq!(
            gate_decision(ApprovalTier::AlwaysAllow, ApprovalMode::Deny),
            ExternalGate::Deny
        );
    }

    // ── three-button tier migration ────────────────────────────────────────────────────────

    #[test]
    fn three_button_tier_migration() {
        let mut ledger = ApprovalLedger::new();
        let key = "mcp.figma.get_file";

        // ask every time by default
        assert_eq!(ledger.tier_of(key), ApprovalTier::Ask);

        // [Allow once only]: tier unchanged
        assert!(!ledger.record_decision(key, &ApprovalDecision::AllowOnce));
        assert_eq!(ledger.tier_of(key), ApprovalTier::Ask);

        // [Deny this time]: tier unchanged
        assert!(!ledger.record_decision(key, &ApprovalDecision::Deny));
        assert_eq!(ledger.tier_of(key), ApprovalTier::Ask);

        // [Always allow]: tier raised
        assert!(ledger.record_decision(key, &ApprovalDecision::AllowAlways));
        assert_eq!(ledger.tier_of(key), ApprovalTier::AlwaysAllow);

        // later [Allow once only]/[Deny this time] do not downgrade
        ledger.record_decision(key, &ApprovalDecision::AllowOnce);
        ledger.record_decision(key, &ApprovalDecision::Deny);
        assert_eq!(ledger.tier_of(key), ApprovalTier::AlwaysAllow);
    }

    #[test]
    fn set_tier_serves_permanent_deny() {
        let mut ledger = ApprovalLedger::new();
        ledger.set_tier("script.deploy", ApprovalTier::Deny);
        assert_eq!(ledger.tier_of("script.deploy"), ApprovalTier::Deny);
        // under the deny tier, [Always allow] can still raise the tier (keep the path for the user to change their mind)
        ledger.record_decision("script.deploy", &ApprovalDecision::AllowAlways);
        assert_eq!(ledger.tier_of("script.deploy"), ApprovalTier::AlwaysAllow);
    }

    // ── ledger cleanup (/tools remove, plugin remove) ────────────────────────────

    #[test]
    fn remove_clears_single_entry() {
        let mut ledger = ApprovalLedger::new();
        ledger.record_decision("script.deploy", &ApprovalDecision::AllowAlways);
        assert!(ledger.remove("script.deploy"));
        assert!(!ledger.remove("script.deploy"));
        assert_eq!(ledger.tier_of("script.deploy"), ApprovalTier::Ask);
    }

    #[test]
    fn remove_prefix_sweeps_plugin_entries() {
        let mut ledger = ApprovalLedger::new();
        for key in [
            "plugin.my-tools.search",
            "plugin.my-tools.deploy",
            "plugin.other.search",
            "script.deploy",
        ] {
            ledger.record_decision(key, &ApprovalDecision::AllowAlways);
        }
        // plugin remove my-tools → clears plugin.my-tools.*, leaves the rest alone
        assert_eq!(ledger.remove_prefix("plugin.my-tools."), 2);
        assert_eq!(ledger.len(), 2);
        assert_eq!(
            ledger.tier_of("plugin.my-tools.search"),
            ApprovalTier::Ask,
            "removed entries fall back to the default tier"
        );
        assert_eq!(
            ledger.tier_of("plugin.other.search"),
            ApprovalTier::AlwaysAllow
        );
    }

    // ── single memory owner ────────────────────────────────────────────────────────

    #[test]
    fn external_allow_always_never_reaches_engine_cache() {
        // ledger records "always allow"; the engine only sees AllowOnce (the action_key cache is not double-written).
        assert_eq!(
            engine_decision_for_external(ApprovalDecision::AllowAlways),
            ApprovalDecision::AllowOnce
        );
        assert_eq!(
            engine_decision_for_external(ApprovalDecision::AllowOnce),
            ApprovalDecision::AllowOnce
        );
        assert_eq!(
            engine_decision_for_external(ApprovalDecision::Deny),
            ApprovalDecision::Deny
        );
    }

    #[test]
    fn external_key_prefixes() {
        for key in [
            "mcp.figma.get_file",
            "script.deploy",
            "plugin.my-tools.search",
            // malformed long-tail keys also land on the external side (fail-closed, no auto-allow)
            "mcp.foo",
            "plugin.x",
        ] {
            assert!(is_external_identity_key(key), "{key}");
        }
        for key in [
            "read_file",
            "execute_command",
            "mcp",
            "script",
            "my-mcp.tool",
        ] {
            assert!(!is_external_identity_key(key), "{key}");
        }
    }

    // ── persistence (storage injection, no product path sniffing) ──────────────────────────

    #[test]
    fn load_save_roundtrip_via_injected_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("approvals.json");

        let mut ledger = ApprovalLedger::new();
        ledger.record_decision("script.deploy", &ApprovalDecision::AllowAlways);
        ledger.set_tier("mcp.notion.search", ApprovalTier::Deny);
        ledger.save(&path).unwrap();

        let back = ApprovalLedger::load(&path).unwrap();
        assert_eq!(back, ledger);
        assert_eq!(back.tier_of("script.deploy"), ApprovalTier::AlwaysAllow);
    }

    #[test]
    fn load_missing_file_is_fresh_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = ApprovalLedger::load(&dir.path().join("nope.json")).unwrap();
        assert!(ledger.is_empty());
    }

    #[test]
    fn load_corrupt_file_errors_instead_of_silent_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("approvals.json");
        std::fs::write(&path, "{not json").unwrap();
        let err = ApprovalLedger::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn json_shape_is_stable() {
        let mut ledger = ApprovalLedger::new();
        ledger.record_decision("script.deploy", &ApprovalDecision::AllowAlways);
        let raw = serde_json::to_value(&ledger).unwrap();
        assert_eq!(raw["schema_version"], 1);
        assert_eq!(raw["tiers"]["script.deploy"], "always_allow");
        // old files without schema_version still load (serde default)
        let back: ApprovalLedger = serde_json::from_str(r#"{"tiers":{"mcp.a.b":"deny"}}"#).unwrap();
        assert_eq!(back.tier_of("mcp.a.b"), ApprovalTier::Deny);
    }
}
