//! Child agent tool capability resolution (design doc 2026-09-19 D2/D3/D5).
//!
//! The LLM only selects the tool surface (what can be called), never the
//! approval semantics (D1 is orthogonal). This module is the **single encoding
//! point** for the exclusion rules: the exclusion-set construction from the
//! original `build_child_runtime_with_config` (including the Manual merge) is
//! moved into [`resolve_capability`], and its `excluded_tools` output is
//! consumed verbatim by the registration loop — the loop is untouched; what
//! changes is the roster inside the gate.
//!
//! Rules (carrier-independent): write capability = `write_tools` members
//! exempt their own entries in the exclusion table; `read_only` =
//! `write_tools` are forcibly merged into the exclusion set (a real read-only
//! with teeth).

use std::collections::BTreeSet;

use super::config::AgentAutonomy;
use super::preset::ChildPreset;

/// Semantic capability of a spawn request (the schema-level `tools` parameter
/// maps here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildToolCapability {
    /// Enforced true read-only: write_tools merged into the exclusion set.
    ReadOnly,
    /// Write capability requested: write_tools members exempted from the
    /// exclusion set.
    Write,
    /// Named preset: expands its whitelist + role prompt (unknown names are
    /// validated first at the tool layer; here it falls back to read-only,
    /// with no panic path).
    Preset(&'static str),
}

impl ChildToolCapability {
    /// Echo label (used in the spawn output message).
    pub fn label(&self) -> String {
        match self {
            Self::ReadOnly => "read_only".to_string(),
            Self::Write => "write".to_string(),
            Self::Preset(name) => format!("preset:{name}"),
        }
    }
}

/// Capability resolution result. Degradation does not go through the Result
/// error channel — degradation is a note attached to a successful resolution.
#[derive(Debug, Clone, Default)]
pub struct CapabilityResolution {
    /// Per-spawn adjusted exclusion set (consumed verbatim by the registration
    /// loop).
    pub excluded_tools: BTreeSet<String>,
    /// Preset whitelist (None = no whitelist set; keep ChildConfig.tool_names).
    pub tool_names: Option<BTreeSet<String>>,
    /// Preset role prompt (prepended before the scaffold).
    pub role_prompt: Option<String>,
    /// The preset's max_turns.
    pub max_turns: Option<u32>,
    /// Degradation reason (Some = this write request was degraded to
    /// read-only).
    pub degraded_reason: Option<String>,
}

/// Capability resolution (pure function, unit-testable).
///
/// `requested = None` is the legacy programmatic path (ChildBuilder / old
/// spawn_*): it skips the capability layer entirely, and the exclusion set is
/// byte-for-byte identical to before the upgrade. The LLM-facing spawn passes
/// `Some` (schema default is read_only).
pub fn resolve_capability(
    requested: Option<&ChildToolCapability>,
    autonomy: AgentAutonomy,
    allow_child_write: bool,
    child_excluded_tools: &[String],
    write_tools: &[String],
) -> CapabilityResolution {
    // First gate: the Manual merge (identical to before the upgrade; the
    // capability layer cannot bypass it).
    let base: BTreeSet<String> = if autonomy == AgentAutonomy::Manual {
        child_excluded_tools
            .iter()
            .chain(write_tools)
            .cloned()
            .collect()
    } else {
        child_excluded_tools.iter().cloned().collect()
    };

    let Some(cap) = requested else {
        // Legacy programmatic path: no capability layer, base as-is
        // (Auto does not merge write_tools).
        return CapabilityResolution {
            excluded_tools: base,
            ..Default::default()
        };
    };

    // Preset expansion: whitelist + role prompt + max_turns.
    let (tool_names, role_prompt, max_turns, preset_wants_write) = match cap {
        ChildToolCapability::Preset(name) => match ChildPreset::by_name(name) {
            Some(preset) => {
                let wants = preset
                    .config
                    .tool_names
                    .as_ref()
                    .map(|set| set.iter().any(|t| write_tools.contains(t)))
                    .unwrap_or(false);
                (
                    preset.config.tool_names.clone(),
                    preset.config.system_prompt.clone(),
                    preset.config.max_turns,
                    wants,
                )
            }
            // The tool layer already validated the name; fall back to
            // read-only here (no panic path).
            None => (None, None, None, false),
        },
        _ => (None, None, None, false),
    };

    let wants_write = match cap {
        ChildToolCapability::Write => true,
        ChildToolCapability::Preset(_) => preset_wants_write,
        ChildToolCapability::ReadOnly => false,
    };
    // Manual never approves; under Auto, consult the deployment switch.
    let permitted = wants_write && autonomy == AgentAutonomy::Auto && allow_child_write;

    let excluded_tools = if permitted {
        // Write exemption: remove write_tools members from the exclusion set.
        base.into_iter()
            .filter(|t| !write_tools.contains(t))
            .collect()
    } else {
        // Read-only enforcement: merge write_tools into the exclusion set.
        let mut set = base;
        for t in write_tools {
            set.insert(t.clone());
        }
        set
    };

    let degraded_reason = if !wants_write {
        None
    } else if autonomy == AgentAutonomy::Manual {
        Some("deployment autonomy is Manual; children are hard read-only".to_string())
    } else if !allow_child_write {
        Some("allow_child_write=false; request degraded to read-only".to_string())
    } else {
        None // wants_write and Auto and allow ⇒ permitted, unreachable
    };

    CapabilityResolution {
        excluded_tools,
        tool_names,
        role_prompt,
        max_turns,
        degraded_reason,
    }
}

/// The three read-only nudge conditions (design doc D3.1, per-child pure
/// function):
/// `autonomy == Manual` (forced, overrides everything)
/// || `child_read_only` (paranoid switch that forces the nudge on everyone,
///    default on — the framework's default deployment is byte-for-byte
///    identical to today's behavior)
/// || the resolved exclusion set still contains all write_tools (the child
///    agent effectively has no write tools).
pub fn read_only_nudge(
    autonomy: AgentAutonomy,
    child_read_only: bool,
    excluded_tools: &BTreeSet<String>,
    write_tools: &[String],
) -> bool {
    autonomy == AgentAutonomy::Manual
        || child_read_only
        || write_tools.iter().all(|t| excluded_tools.contains(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn excluded(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    const WRITE_TOOLS: &[&str] = &["write_file", "edit_file", "execute_command"];

    fn write_tools() -> Vec<String> {
        excluded(WRITE_TOOLS)
    }

    /// None (legacy path): Auto does not merge write_tools — byte-for-byte
    /// identical to before the upgrade (the precondition of the preset
    /// fixture regression sentinel).
    #[test]
    fn none_capability_keeps_legacy_base_set() {
        let res = resolve_capability(None, AgentAutonomy::Auto, false, &[], &write_tools());
        assert!(res.excluded_tools.is_empty());
        assert!(res.tool_names.is_none());
        assert!(res.degraded_reason.is_none());
    }

    #[test]
    fn none_capability_manual_merges_write_tools() {
        let res = resolve_capability(None, AgentAutonomy::Manual, false, &[], &write_tools());
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
    }

    /// read_only enforcement (v3): write_tools merged in even under Auto.
    #[test]
    fn read_only_forces_write_tools_into_exclusion() {
        let res = resolve_capability(
            Some(&ChildToolCapability::ReadOnly),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t), "{t} must be excluded");
        }
    }

    /// Write capability + allowed: write_tools members exempted from the
    /// exclusion set (including ones mistakenly listed in the deployment
    /// table).
    #[test]
    fn write_permitted_exempt_write_tools() {
        let table = excluded(&["write_file", "task_output"]);
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Auto,
            true,
            &table,
            &write_tools(),
        );
        assert!(res.excluded_tools.contains("task_output"));
        assert!(!res.excluded_tools.contains("write_file"));
        assert!(!res.excluded_tools.contains("edit_file"));
        assert!(!res.excluded_tools.contains("execute_command"));
        assert!(res.degraded_reason.is_none());
        assert!(
            res.tool_names.is_none(),
            "write must not produce a whitelist"
        );
    }

    /// Write capability + deployment disabled: degraded to read-only +
    /// reason.
    #[test]
    fn write_denied_by_flag_degrades_with_reason() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Auto,
            false,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        let why = res.degraded_reason.expect("must carry degraded reason");
        assert!(
            why.contains("allow_child_write"),
            "reason should name the flag: {why}"
        );
    }

    /// Manual has the highest priority: a write request is pushed back to
    /// read-only + reason.
    #[test]
    fn write_denied_by_manual_autonomy() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Manual,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        let why = res.degraded_reason.expect("Manual denial must explain");
        assert!(why.contains("Manual"), "{why}");
    }

    /// Preset (containing write tools) + allowed: the whitelist expands, and
    /// the exclusion set exempts write members.
    #[test]
    fn preset_with_write_tools_permitted_expands() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("coder")),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        let whitelist = res.tool_names.expect("coder preset carries whitelist");
        assert!(whitelist.contains("write_file"));
        assert!(whitelist.contains("read_file"));
        assert!(!res.excluded_tools.contains("write_file"));
        assert!(res.role_prompt.is_some());
        assert_eq!(res.max_turns, Some(64));
        assert!(res.degraded_reason.is_none());
    }

    /// Preset (containing write tools) + disabled: the whitelist keeps the
    /// role (its write entries are blocked by the §5.4 warn+drop).
    #[test]
    fn preset_with_write_tools_denied_degrades() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("tester")),
            AgentAutonomy::Auto,
            false,
            &[],
            &write_tools(),
        );
        assert!(res.tool_names.is_some(), "preset keeps its role");
        assert!(res.degraded_reason.is_some());
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
    }

    /// Read-only preset (researcher): exclusion-set semantics equivalent to
    /// read_only.
    #[test]
    fn read_only_preset_forces_read_only_exclusion() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("researcher")),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        assert!(
            res.degraded_reason.is_none(),
            "a read-only request never degrades"
        );
        assert_eq!(res.max_turns, Some(32));
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(ChildToolCapability::ReadOnly.label(), "read_only");
        assert_eq!(ChildToolCapability::Write.label(), "write");
        assert_eq!(ChildToolCapability::Preset("coder").label(), "preset:coder");
    }

    /// D3.1: the nudge's three conditions (per-child, based on the resolver's
    /// output exclusion set).
    #[test]
    fn nudge_three_conditions() {
        let read_only_excluded: BTreeSet<String> =
            WRITE_TOOLS.iter().map(|s| s.to_string()).collect();
        let write_exempted: BTreeSet<String> = ["task_output".to_string()].into_iter().collect();

        // Condition 1: Manual forces it.
        assert!(super::read_only_nudge(
            AgentAutonomy::Manual,
            false,
            &write_exempted,
            &write_tools()
        ));
        // Condition 2: the child_read_only paranoid switch (default
        // deployment path).
        assert!(super::read_only_nudge(
            AgentAutonomy::Auto,
            true,
            &write_exempted,
            &write_tools()
        ));
        // Condition 3: the resolved exclusion set still contains all
        // write_tools (the child agent effectively has no write tools).
        assert!(super::read_only_nudge(
            AgentAutonomy::Auto,
            false,
            &read_only_excluded,
            &write_tools()
        ));
        // Write child agent (after exemption): no nudge.
        assert!(!super::read_only_nudge(
            AgentAutonomy::Auto,
            false,
            &write_exempted,
            &write_tools()
        ));
    }
}
