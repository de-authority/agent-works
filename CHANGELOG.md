# Changelog

All notable changes to `agent-works` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-10-18

### Added
- **Tool provider contract** — `tool_provider.rs` plus the `phi-tool.wit`
  interface (WIT + Component Model) and `schemas/plugin.schema.json`: one
  contract behind three external tool shapes (MCP servers, user scripts, wasm
  plugin packs).
- **Plugin packs** (`src/plugin/`): wasmtime-based wasm loading, an integrity
  registry pinning installed packs, and an install/remove lifecycle.
- **Script tools** (`src/script_tool.rs`): user scripts run as tools;
  `${ENV:VAR}` references are pre-resolved at registration and a missing
  variable is named on the spot.
- **Approval ledger plumbing** (`src/approval_ledger.rs`): identity-keyed
  external tool tiers via `is_external_identity_key` (the `mcp.` / `script.` /
  `plugin.` prefixes), fail-closed.
- **Multi-agent**: child spawn sets exclude the external tool namespaces
  (warn + drop a reduced set), so external tools stay root-only.

### Fixed
- **MCP Streamable HTTP now works**: the full handshake runs once
  (initialize + notifications/initialized), `Mcp-Session-Id` is captured and
  replayed on every request, and SSE `data:` frames are parsed. Servers that
  refuse `tools/list` without a session (e.g. Taobao desktop) now expose their
  tools.
- A connection that discovers 0 tools reports ⚠ instead of a green check, and
  `attach` no longer swallows a discovery failure into a silent 0-tool success
  — the real cause is surfaced and the change rolled back.
- MCP child processes are spawned detached from the controlling terminal and
  stderr, fixing a TUI hard freeze when a child clobbered the tty.
- Script tools whose file is missing (or not executable on unix) are refused
  at install/mount time instead of installing silently.

## [0.10.0] - 2026-10-07

### Added
- **`DefaultGuard`** surfaces judge fail-open as a Warning notice

### Fixed
- `memory_read` paginates output instead of hitting the tool-output cap
- Memory shared-writer contract honored alongside Claude Code
- Judge fail-open messages carry the cause; one retry on parse failure before
  fail-open
- Compression compactor adapts to `CompactionOutcome` (#33)

## [0.9.0] - 2026-09-27

### Added
- **Child write capability framework** (multi-agent): `ChildToolCapability` +
  `resolve_capability` (presets / exclusion sets / role hints / degradation),
  `MultiAgentConfig::allow_child_write` (default `false`) + `child_write_gate`,
  `WorkspaceWriteGate` (`try_claim` never blocks, held for the task lifetime,
  released by `ChildCleanup::drop`, parent agent exempt), build-time tool
  registration filtering by the capability exclusion set with write-class tools
  wrapped in `GatedTool`, spawn echo carrying `SpawnEcho`
  (agent_path / registered_tools / degraded_reason), and `list_agents`
  exposing `spawned_tools`.
- `WorkspaceWriteGate::holder_of()` read-only observation.

### Fixed
- **Write-gate claim keys** no longer depend on whether the file existed at
  claim time — on macOS (`/tmp` -> `/private/tmp`) first-writer and later-writer
  canonicalization produced two keys for one file, structurally bypassing
  mutual exclusion.
- **Same-path spawn recycling**: spawning over a done/closed same-path
  predecessor takes over its registration (cancel token, registry, mailbox,
  generation bump, claim/echo clear) instead of failing with `AlreadyExists`;
  `SpawnEcho` carries `recycled: true`. Running predecessors still reject;
  per-path generation counters guard cleanup races.
- **Write-gate claims released at task end** (before `note_posted`) instead of
  held until close — a finished child no longer holds a lock a sibling's later
  retry collides with. The close-path release remains as an idempotent backstop.
- Skill scanning silently skips `_-prefixed` dirs (parked/disabled skills)
  instead of WARN-logging a name mismatch.

### Changed
- Bump `agent-base` to 0.8.0.

## [0.8.0] - 2026-09-18

### Added
- **Auto-memory system** (behind `memory` feature): persistent project memory
  compatible with Claude Code storage (`~/.claude/projects/<slug>/memory/`).
  Includes `MemoryStore`, `MemoryConfig`, memory tools (write/read/list/delete),
  frontmatter serde, atomic writes, line-level index management, and builder
  integration across all 4 build paths. 66 unit + 6 integration tests.
- **Session-scope slash commands**: slash-activated skills get `Session` scope
  (stays in effect for the whole session instead of single-turn). Catalog
  trigger copy distinguishes session-scope from single-turn skills.
- **Active Skills bake sanitization** (`demote_h2_headings`): prevents skill
  body `##` headings from colliding with prompt-surgery anchors.
- **`closing` terminal fact** on `AgentEntry`: force-killed children count as
  settled for fan-in quiescence immediately; `note_closing()` method +
  `status()` derives to `Closed` on close.
- **Held-batch reaper** (90s liveness backstop): hands over stranded reports
  when quiescence is unachievable rather than deadlocking.
- **Shutdown flush**: watcher force-hands-over held reports on cancel/hub-drop.

### Changed
- Default work budget increased from 96K → 210K (aligned with ~256K context
  window × 90% minus base overhead).

## [0.7.0] - 2026-09-11

### Added
- **`rotation_policy` module** (renamed from `token_budget`): `TokenBudgetConfig`,
  `TokenBudgetCore`, `TokenBudgetState`, `TokenBudgetAction`,
  `DEFAULT_SEED_MESSAGE`, `build_context_window_info`, `token_budget_base_overhead`.
- **Skill catalog discovery** (behind `prompt_skill`): filesystem-based skill
  resolver, catalog refresh middleware, telemetry, and `SkillTool` — enables
  runtime skill injection via prompt without dedicated tools.
- **`agent_instructions_paths`** on `AgentBuilder`: read instruction files at
  build time (e.g. CLAUDE.md, INSTRUCTIONS.md) and append to the system prompt;
  supports multiple paths with priority ordering.
- **Child `model` field now functional**: `ChildConfig.model` is applied via
  `AgentRuntime::set_model_override()` so sub-agents can use a different model
  tier (e.g. "lite").

### Changed
- Error results now emit a single synchronous Progress carrying the error's
  first line (ANSI/control sanitized, truncated at 160 chars) instead of a terse
  notice plus a detached Focus paraphrase — the user sees the real reason in the
  TUI the moment the child fails.

### Removed
- `token_budget` module (renamed to `rotation_policy`).

### Changed
- Error results now emit a single synchronous Progress carrying the error's
  first line (ANSI/control sanitized, truncated at 160 chars) instead of a
  terse notice plus a detached Focus paraphrase — the user sees the real
  reason in the TUI the moment the child fails (session 20260906_0f6d4341).
  Ok results are unchanged (plain notice, then Focus summary).

## [0.6.0] - 2026-09-06

### Added
- Full multi-agent orchestration layer (spawn / track / fan-in) per design doc
  stages 1-4, built on the push-based delivery model.
- `spawn_agent` gains a `task` field with Focus-based prompt expansion.
- Push-based child results with fan-in batch injection: child results are
  handed to the parent LLM as a single intact batch instead of trickling in.
- Fact-derived child status machine; the delivery gap is surfaced as facts
  (`results_handed_over` / `pending_results`).
- The `multi_agent` feature now implies `focus` (the fan-in coordinator uses
  Focus for user-facing progress summaries).
- `loom-check` feature to model-check the multi_agent atomic gates with loom
  (off by default; run `cargo test --features multi_agent,loom-check --lib loom`).
- `multi_agent_control` benchmark and `multi_agent` example.

### Fixed
- Child agent now returns only the last assistant message.
- Watcher emits the plain progress notice first and the Focus summary as a
  follow-up.
- Guard never judges completion after rejected tool calls.
