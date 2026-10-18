//! Persistent auto-memory (Phase 9b, L2) — Claude Code compatible.
//!
//! Memories live as Markdown files with YAML frontmatter under a
//! per-project directory (default: `~/.claude/projects/<slug>/memory/`),
//! indexed by `MEMORY.md`. The format, slug algorithm, and index layout are
//! byte-compatible with Claude Code's auto-memory, so both tools read and
//! write the same directory with zero migration.
//!
//! Layering (docs/phase9-plan.md "Framework Layering Design"):
//! - This module owns the *mechanism*: frontmatter (de)serialization, atomic
//!   file writes, row-level index sync, and a `std::sync::Mutex` serializing
//!   index read-modify-write within the process (file IO inside the critical
//!   section is synchronous and millisecond-scale, so no async lock needed).
//! - The consumer owns the *policy* via [`MemoryConfig`]: root path, index
//!   filename, and the prompt template injected into the system prompt.
//!
//! Concurrency model: in-process writes are serialized per [`MemoryStore`];
//! cross-process concurrency (phimint + Claude Code) relies on atomic
//! rename-based writes (last-write-wins, no torn files) plus [`rebuild_index`]
//! as the repair path.
//!
//! Shared-writer contract (the directory has a second writer — Claude Code —
//! and neither side can stop the other from writing):
//! - **Tolerant read**: any frontmatter this module doesn't know — missing
//!   stamps (`node_type`, `originSessionId`, …), unknown keys — still parses.
//!   Claude Code's documented format is a supported fixture, not an error.
//! - **Faithful write**: updates merge into the existing frontmatter instead of
//!   re-rendering it, so keys owned by the other writer survive (see
//!   [`MemoryStore::write_memory`]).
//! - **Conservative repair**: a row is identified structurally — by its own
//!   link target `- [title](name.md)`, never by a substring of another line.
//!   An update refreshes only the row's `— description` tail; a rebuild
//!   re-emits existing rows byte-identical and in their curated order
//!   ([`index_add`], [`rebuild_index`]) — human titles live only in the index.
//!   A rebuild never erases a row it cannot re-derive.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use agent_base::{AgentError, AgentResult};
use serde::{Deserialize, Serialize};

/// Default index file name (Claude Code compatible).
pub const DEFAULT_INDEX_FILENAME: &str = "MEMORY.md";

/// Fixed `metadata.node_type` value for every memory file.
pub const MEMORY_NODE_TYPE: &str = "memory";

/// Allowed `metadata.type` values.
pub const MEMORY_TYPES: [&str; 4] = ["user", "feedback", "project", "reference"];

// ─────────────────────────────────────────────────────────────────────────────
// Project slug
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a workspace path into a Claude Code style project slug.
///
/// `/Users/xxx/my.project` -> `-Users-xxx-my-project`: trailing `/` stripped,
/// every non-alphanumeric character becomes `-` (matching Claude Code).
/// Slugs longer than 200 chars are truncated. The slug embeds the user name and the absolute path, so
/// memories are naturally isolated per machine and per workspace. Bound to the
/// startup cwd (not the git root), matching Claude Code's behavior.
pub fn project_slug(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let trimmed = s.trim_end_matches('/');
    // Match Claude Code: every non-alphanumeric char becomes `-`
    let slug: String = trimmed
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    // Claude Code truncates at 200 chars
    if slug.len() > 200 {
        slug[..200].to_string()
    } else {
        slug
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Name validation (path-escape guard)
// ─────────────────────────────────────────────────────────────────────────────

/// Check a memory name against `^[a-z0-9][a-z0-9-]*$`.
///
/// The name is joined into a file path (`<name>.md`), so anything outside this
/// charset (slashes, dots, underscores, uppercase) is rejected *before* any
/// path is built — `../../etc/passwd` cannot escape the memory directory.
pub fn is_valid_memory_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Validate a memory name, returning a descriptive error for the LLM.
pub fn validate_memory_name(name: &str) -> AgentResult<()> {
    if is_valid_memory_name(name) {
        Ok(())
    } else {
        Err(AgentError::internal(format!(
            "invalid memory name `{name}`: must match ^[a-z0-9][a-z0-9-]*$ (kebab-case slug, no slashes/dots/underscores)"
        )))
    }
}

/// Validate a memory type against the four allowed values.
pub fn validate_memory_type(memory_type: &str) -> AgentResult<()> {
    if MEMORY_TYPES.contains(&memory_type) {
        Ok(())
    } else {
        Err(AgentError::internal(format!(
            "invalid memory type `{memory_type}`: must be one of {}",
            MEMORY_TYPES.join(" | ")
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frontmatter
// ─────────────────────────────────────────────────────────────────────────────

/// YAML frontmatter `metadata:` block (Claude Code compatible field names).
///
/// Every field is optional on read: Claude Code's documented format carries
/// only `type`, and files written before a field existed must keep parsing
/// (tolerant read — the shared directory has a second writer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryFrontmatterMetadata {
    #[serde(default)]
    pub node_type: String,
    #[serde(rename = "type", default)]
    pub memory_type: String,
    #[serde(
        rename = "originSessionId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub origin_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified: Option<String>,
}

/// YAML frontmatter of a memory file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub metadata: Option<MemoryFrontmatterMetadata>,
}

/// A parsed memory file: frontmatter fields plus the free-form Markdown body.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDoc {
    pub name: String,
    pub description: String,
    pub metadata: Option<MemoryFrontmatterMetadata>,
    pub body: String,
}

/// Split `---\n<yaml>\n---\n<body>` into `(yaml, body)`.
///
/// Returns `None` when the file has no (complete) frontmatter block. The body
/// keeps everything after the closing `---` line with leading blank lines
/// trimmed (the conventional single blank separator is not part of the body).
fn split_frontmatter(raw: &str) -> Option<(String, String)> {
    // Both line-ending conventions: a foreign writer on Windows must not be
    // treated as "no frontmatter" — the merge would then drop every key.
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;
    let mut cursor = 0usize;
    let mut close = None;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\n', '\r']) == "---" {
            close = Some(cursor);
            break;
        }
        cursor += line.len();
    }
    let close = close?;
    let yaml = rest[..close].to_string();
    let after = &rest[close + 3..]; // skip the closing "---"
    let after = after
        .strip_prefix("\r\n")
        .or_else(|| after.strip_prefix('\n'))
        .unwrap_or(after);
    Some((yaml, after.trim_start_matches('\n').to_string()))
}

/// Parse a memory file into a [`MemoryDoc`].
///
/// Used by listing/index-rebuild paths; callers treat `Err` as "skip this
/// file" (fault tolerance for files Claude Code may have written badly).
pub fn parse_memory_file(path: &Path) -> AgentResult<MemoryDoc> {
    let raw = fs::read_to_string(path)?;
    parse_memory_str(&raw).ok_or_else(|| {
        AgentError::internal(format!(
            "{}: missing or malformed frontmatter",
            path.display()
        ))
    })
}

/// Parse memory file content (frontmatter + body).
pub fn parse_memory_str(raw: &str) -> Option<MemoryDoc> {
    let (yaml, body) = split_frontmatter(raw)?;
    let fm: MemoryFrontmatter = match serde_yaml::from_str(&yaml) {
        Ok(fm) => fm,
        Err(e) => {
            tracing::warn!(error = %e, "skipping memory file with invalid frontmatter");
            return None;
        }
    };
    if fm.name.is_empty() || fm.description.is_empty() {
        tracing::warn!("skipping memory file with empty name or description");
        return None;
    }
    Some(MemoryDoc {
        name: fm.name,
        description: fm.description,
        metadata: fm.metadata,
        body,
    })
}

/// Render a full memory file (frontmatter + body) for writing.
///
/// `origin_session_id` and `modified` are auto-filled by `memory_write` and
/// omitted when `None` (matching Claude Code's optional fields).
pub fn render_memory_file(
    name: &str,
    description: &str,
    memory_type: &str,
    origin_session_id: Option<&str>,
    modified: Option<&str>,
    body: &str,
) -> AgentResult<String> {
    render_memory_file_merged(
        None,
        name,
        description,
        memory_type,
        origin_session_id,
        modified,
        body,
    )
}

/// Render a memory file, merging into `original_raw` when updating.
///
/// Write-fidelity contract (shared directory, second writer = Claude Code):
/// frontmatter keys this module doesn't own — top-level or inside `metadata` —
/// are preserved across an update. Known keys are upserted; the session stamps
/// are set-or-keep (`None` never clears a stamp the other writer set). Two
/// hard limits, both lossy by necessity: YAML comments are not preserved, and
/// a `metadata` value that is not a mapping is left untouched (never destroy
/// what we cannot merge into). Best-effort merge: if the original frontmatter
/// won't parse as YAML the file is re-rendered from the known fields alone —
/// warned, because that drops every foreign key.
fn render_memory_file_merged(
    original_raw: Option<&str>,
    name: &str,
    description: &str,
    memory_type: &str,
    origin_session_id: Option<&str>,
    modified: Option<&str>,
    body: &str,
) -> AgentResult<String> {
    use serde_yaml::{Mapping, Value};

    fn upsert(map: &mut Mapping, key: &str, value: &str) {
        map.insert(
            Value::String(key.to_string()),
            Value::String(value.to_string()),
        );
    }

    let mut map: Mapping = match original_raw.and_then(split_frontmatter) {
        Some((yaml, _)) => serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
            tracing::warn!(
                error = %e,
                "memory frontmatter not parseable; re-rendering from known fields (foreign keys dropped)"
            );
            Mapping::new()
        }),
        None => Mapping::new(),
    };
    upsert(&mut map, "name", name);
    upsert(&mut map, "description", description);
    // Merge into the original metadata mapping so unknown keys survive. A
    // non-mapping `metadata` is left untouched — faithful write: we never
    // destroy a value we can't merge into.
    let meta = match map.get(Value::String("metadata".to_string())) {
        Some(Value::Mapping(m)) => Some(m.clone()),
        None => Some(Mapping::new()),
        Some(other) => {
            tracing::warn!(
                value = ?other,
                "metadata is not a mapping; leaving it untouched (memory stamps not applied)"
            );
            None
        }
    };
    if let Some(mut meta) = meta {
        upsert(&mut meta, "node_type", MEMORY_NODE_TYPE);
        upsert(&mut meta, "type", memory_type);
        // Set-or-keep: `None` never clears a stamp — it may record the other
        // writer's provenance.
        if let Some(v) = origin_session_id {
            upsert(&mut meta, "originSessionId", v);
        }
        if let Some(v) = modified {
            upsert(&mut meta, "modified", v);
        }
        map.insert(Value::String("metadata".to_string()), Value::Mapping(meta));
    }

    // serde_yaml (never string concatenation) so descriptions with special
    // characters are quoted/escaped correctly.
    let yaml = serde_yaml::to_string(&map).map_err(|e| {
        AgentError::internal(format!("failed to serialize memory frontmatter: {e}"))
    })?;
    // `to_string` output always ends with '\n'.
    Ok(format!(
        "---\n{yaml}---\n\n{}",
        body.trim_start_matches('\n')
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestamps (ISO 8601 without a chrono dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Current UTC time as ISO 8601 with milliseconds, e.g. `2026-09-13T08:15:30.123Z`.
pub fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() as i64;
    let millis = d.subsec_millis();
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (y, m, dd) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (y, m, d).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic write
// ─────────────────────────────────────────────────────────────────────────────

/// Atomic file write: write to a unique sibling `.tmp` file, then `fs::rename`
/// over the target. A crash mid-write leaves the tmp remnant behind (dot-
/// prefixed, never scanned as a memory) but the target file always either has
/// its old content or its full new content — never a torn mix.
///
/// The tmp name is unique per call (monotonic counter): two threads writing
/// the same target concurrently must not share a tmp path, or one thread's
/// rename would steal the other's tmp file mid-flight.
pub(crate) fn atomic_write(path: &Path, content: &str) -> AgentResult<()> {
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "memory-index".to_string());
    let tmp = path.with_file_name(format!(".{file_name}.{unique}.tmp"));
    fs::write(&tmp, content)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Index sync (row-level, never a full rewrite)
// ─────────────────────────────────────────────────────────────────────────────
//
// Index lines look like `- [Human Title](name.md) — short description`. The
// link text is a human-readable title (possibly Chinese) that only exists in
// the index — NOT in any frontmatter. Therefore add/remove must locate rows by
// the link target `(name.md)` and touch only those rows, leaving every other
// row (and its title) byte-identical. A refresh of a row replaces only its
// `— description` tail — the human title survives updates too (shared-index
// contract with Claude Code). New rows use the memory name as link text, the
// same shape [`rebuild_index`] generates.

/// Sanitize a description for single-line index rows.
///
/// The write path already rejects line breaks in descriptions; this scrub is
/// defense-in-depth for descriptions that reach a row from any other path
/// (e.g. regenerated rows for files a foreign writer created).
fn index_line_desc(desc: &str) -> String {
    desc.replace(['\n', '\r', '\u{2028}', '\u{2029}', '\u{85}'], " ")
        .trim()
        .to_string()
}

/// The link target of an index row line, if the line is a row at all:
/// `- [title](name.md)` → `name`.
///
/// Row identity is structural — the link target must be the row's first
/// `](…)` — so a description embedding another memory's `](x.md)` link can
/// never hijack that memory's row (descriptions are free text written by LLMs
/// and foreign tools).
fn row_link_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("- [")?;
    let close = rest.find("](")?;
    let target = rest[close + 2..].split(')').next()?;
    target.strip_suffix(".md")
}

/// `Some(head)` — everything up to and including `](name.md)` — when `line` is
/// `name`'s row. The head is preserved byte-identical on refresh so human
/// titles in the link text survive.
fn row_head<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    if row_link_name(line) != Some(name) {
        return None;
    }
    let close = line.find("](")?;
    line.get(..close + 2 + name.len() + ".md)".len())
}

/// Render a new index row: link text is the memory name — the shape
/// [`rebuild_index`] generates. Human titles exist only in rows the other
/// writer created and are never re-rendered here.
fn render_index_row(name: &str, desc: &str) -> String {
    format!("- [{name}]({name}.md) — {}", index_line_desc(desc))
}

/// Add (or update) the row for `name` in the index.
///
/// Existing rows for other names — including their human titles — are
/// preserved verbatim.
pub(crate) fn index_add(
    memory_root: &Path,
    index_filename: &str,
    name: &str,
    desc: &str,
) -> AgentResult<()> {
    // Self-healing: the very first write may race ahead of ensure_dir.
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    let index_path = memory_root.join(index_filename);
    // P2-2 fix: distinguish NotFound (empty index) from real I/O errors.
    let content = match fs::read_to_string(&index_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                index_path.display()
            )));
        }
    };
    // Row identity is structural (the row's own link target), so a description
    // that embeds another `](name.md)` cannot hijack the refresh.
    let desc_line = index_line_desc(desc);

    let updated = if content.lines().any(|line| row_head(line, name).is_some()) {
        // Row exists: refresh only its `— description` tail. Everything up to
        // and including `](name.md)` — the human title in the link text — is
        // preserved byte-identical (Claude Code writes those titles).
        let mut out = content
            .lines()
            .map(|line| match row_head(line, name) {
                Some(head) => format!("{head} — {desc_line}"),
                None => line.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if content.ends_with('\n') {
            out.push('\n');
        }
        out
    } else {
        // New row: append after the existing rows.
        let trimmed = content.trim_end();
        let mut out = String::new();
        if !trimmed.is_empty() {
            out.push_str(trimmed);
            out.push('\n');
        }
        out.push_str(&render_index_row(name, desc));
        out.push('\n');
        out
    };
    atomic_write(&index_path, &updated)
}

/// Remove the row for `name` from the index. Other rows are untouched.
pub(crate) fn index_remove(
    memory_root: &Path,
    index_filename: &str,
    name: &str,
) -> AgentResult<()> {
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    let index_path = memory_root.join(index_filename);
    // P2-2 fix: distinguish NotFound (empty index) from real I/O errors.
    let content = match fs::read_to_string(&index_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                index_path.display()
            )));
        }
    };
    // Row identity is structural (the row's own link target), never a bare
    // substring of some other row's description.
    let mut updated = content
        .lines()
        .filter(|line| row_link_name(line) != Some(name))
        .collect::<Vec<_>>()
        .join("\n");
    if !updated.is_empty() && content.ends_with('\n') {
        updated.push('\n');
    }
    atomic_write(&index_path, &updated)
}

/// Find the existing index row for `name`, if any (same structural row
/// identity as [`index_add`] / [`index_remove`]). Returns the line
/// byte-identical — the human title in the link text must survive rebuilds.
fn find_row(index: &str, name: &str) -> Option<String> {
    index
        .lines()
        .find(|line| row_head(line, name).is_some())
        .map(str::to_string)
}

/// Rebuild the index from the memory directory (fault-tolerance path only).
///
/// Triggered when the index is lost or unreadable. Conservative repair
/// (shared-index contract with Claude Code): a file's existing row is re-emitted
/// byte-identical whenever one exists — human titles live only in the index and
/// cannot be recovered from the file. Rows are regenerated from frontmatter
/// only for files that have no row yet. Files that cannot be indexed at all
/// (corrupt frontmatter, name ≠ filename) keep their existing row rather than
/// being erased; only rows whose file is gone are dropped (dangling links).
pub fn rebuild_index(memory_root: &Path, index_filename: &str) -> AgentResult<()> {
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    // Only "no index yet" is empty. Any other read error must abort: treating
    // it as empty would regenerate every human title away — exactly the loss
    // conservative repair exists to prevent.
    let old_index = match fs::read_to_string(memory_root.join(index_filename)) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                memory_root.join(index_filename).display()
            )));
        }
    };
    // Encounter order of the old index: kept rows survive in their curated
    // order (the prompt tells the LLM to scan top-down); rows generated for
    // files that have no row yet are appended after them, sorted by name.
    let old_pos: Vec<&str> = old_index.lines().filter_map(row_link_name).collect();
    let mut entries: Vec<(usize, String, String)> = Vec::new(); // (order, name, row)
    for entry in fs::read_dir(memory_root)? {
        let path = entry?.path();
        if path.extension() != Some("md".as_ref())
            || path.file_name() == Some(index_filename.as_ref())
        {
            continue;
        }
        let Some(name) = path.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            continue;
        };
        let regenerated = parse_memory_file(&path)
            .ok()
            // Only index files whose own name matches their filename, so the
            // rebuilt index links resolve.
            .filter(|doc| doc.name == name)
            .map(|doc| render_index_row(&doc.name, &doc.description));
        let existing = find_row(&old_index, &name).map(|row| {
            let order = old_pos
                .iter()
                .position(|n| *n == name)
                .unwrap_or(usize::MAX);
            (order, row)
        });
        match existing.or_else(|| regenerated.map(|row| (usize::MAX, row))) {
            Some((order, row)) => entries.push((order, name, row)),
            None => tracing::warn!(
                file = %path.display(),
                "skipping unindexable memory file with no existing index row"
            ),
        }
    }
    entries.sort();
    let content = if entries.is_empty() {
        String::new()
    } else {
        entries
            .into_iter()
            .map(|(_, _, row)| row)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    atomic_write(&memory_root.join(index_filename), &content)
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Memory configuration (consumer-injected policy, Phase 9b).
///
/// The framework owns format + tools + index sync; the consumer decides where
/// memories live and how they are pitched in the system prompt.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Memory root directory, e.g. `~/.claude/projects/<slug>/memory/`.
    pub memory_root: PathBuf,
    /// Index file name (default `"MEMORY.md"`).
    pub index_filename: String,
    /// System prompt template. Placeholders substituted at build time:
    /// `{memory_root}`, `{index_content}`, `{tools_description}`.
    pub prompt_template: String,
}

/// Default prompt template for Claude Code compatible deployments.
pub const CLAUDE_COMPATIBLE_TEMPLATE: &str = r#"## Memory

You have a persistent memory directory at `{memory_root}`. Memories survive across sessions and are shared with Claude Code (same directory, same format — what you write here, Claude Code reads, and vice versa).

### Current memory index

The snapshot below was taken at startup and may be stale — call `memory_list` to refresh whenever unsure.

{index_content}

### Tools

{tools_description}

### How to remember

- `memory_write` stores one memory as `<name>.md` and syncs the index automatically. `name` must match `^[a-z0-9][a-z0-9-]*$`; writing to an existing `name` overwrites it in place (update instead of duplicating; prefer overwriting rolling `project` memories like status notes over creating new ones).
- The `description` is the recall key — one line (no newlines) that lets you tell at a glance whether the memory matters for the current task.
- Types: `user` (who the user is: role, preferences, working style), `feedback` (rules and corrections the user gave you), `project` (ongoing work, decisions, status), `reference` (pointers to external resources).
- Link related memories with `[[memory-name]]` in the body.

### When to use memory

- The user explicitly asks to remember something ("remember this", "save that").
- You learn a durable preference, correction, or project fact worth carrying into future sessions.

### When NOT to use memory

- Transient details that only matter within this session.
- Facts already recorded in the codebase (code structure, git history, config files).

### How to recall

Scan the index above by `description` and call `memory_read(name)` only for the relevant entries. Recall is selective by design — do not read everything."#;

impl MemoryConfig {
    /// Claude Code compatible configuration for `workspace`.
    ///
    /// Storage: `~/.claude/projects/<slug>/memory/` with the slug computed
    /// from the workspace path (same algorithm as Claude Code), so both tools
    /// share one memory directory.
    pub fn claude_compatible(workspace: &Path) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::custom(
            home.join(".claude")
                .join("projects")
                .join(project_slug(workspace))
                .join("memory"),
            CLAUDE_COMPATIBLE_TEMPLATE.to_string(),
        )
    }

    /// Custom configuration: the consumer computes `memory_root` itself
    /// (different product, different interop target) and supplies its own
    /// prompt template.
    pub fn custom(memory_root: PathBuf, prompt_template: String) -> Self {
        Self {
            memory_root,
            index_filename: DEFAULT_INDEX_FILENAME.to_string(),
            prompt_template,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryStore
// ─────────────────────────────────────────────────────────────────────────────

/// One memory directory + its in-process index lock.
///
/// All four memory tools share one `Arc<MemoryStore>`. File IO is synchronous
/// and bounded by the critical section (a read, one write, one rename), so a
/// `std::sync::Mutex` is the right primitive — never hold it across an `await`
/// (the tools don't).
pub struct MemoryStore {
    memory_root: PathBuf,
    index_filename: String,
    /// Serializes index read-modify-write within this process, so concurrent
    /// turns appending rows cannot lose updates.
    index_lock: Mutex<()>,
}

/// A row in the memory listing.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryListEntry {
    pub name: String,
    pub description: String,
    pub memory_type: Option<String>,
    pub modified: Option<String>,
}

impl MemoryStore {
    pub fn new(memory_root: PathBuf, index_filename: impl Into<String>) -> Self {
        Self {
            memory_root,
            index_filename: index_filename.into(),
            index_lock: Mutex::new(()),
        }
    }

    pub fn memory_root(&self) -> &Path {
        &self.memory_root
    }

    pub fn index_filename(&self) -> &str {
        &self.index_filename
    }

    pub fn index_path(&self) -> PathBuf {
        self.memory_root.join(&self.index_filename)
    }

    fn memory_path(&self, name: &str) -> PathBuf {
        // Caller must have run validate_memory_name first.
        self.memory_root.join(format!("{name}.md"))
    }

    /// Read the index file, or the empty string when it doesn't exist.
    pub fn read_index(&self) -> String {
        fs::read_to_string(self.index_path()).unwrap_or_default()
    }

    /// Ensure the memory directory exists (first-use auto-init).
    pub fn ensure_dir(&self) -> AgentResult<()> {
        fs::create_dir_all(&self.memory_root).map_err(|e| {
            AgentError::internal(format!(
                "failed to create memory dir {}: {e}",
                self.memory_root.display()
            ))
        })
    }

    /// Create or overwrite the memory `name` and sync the index row.
    ///
    /// Order: file first, index second — a crash between the two leaves an
    /// unindexed file (recoverable via [`rebuild_index`]), never a dangling
    /// index row.
    pub fn write_memory(
        &self,
        name: &str,
        description: &str,
        memory_type: &str,
        body: &str,
        origin_session_id: Option<&str>,
    ) -> AgentResult<()> {
        validate_memory_name(name)?;
        validate_memory_type(memory_type)?;
        // Single-line descriptions only — and not merely `\n`: Unicode line/
        // paragraph separators (U+2028/U+2029/U+0085) read as line breaks in
        // a prompt, so they could forge extra index rows.
        if description
            .chars()
            .any(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}' | '\u{85}'))
        {
            return Err(AgentError::internal(
                "invalid memory description: must be a single line (no newlines)",
            ));
        }
        self.ensure_dir()?;
        // Merge into the file's existing frontmatter so keys owned by the
        // other writer (Claude Code) survive the update.
        let file_path = self.memory_path(name);
        let original = fs::read_to_string(&file_path).ok();
        let file = render_memory_file_merged(
            original.as_deref(),
            name,
            description,
            memory_type,
            origin_session_id,
            Some(&now_iso8601()),
            body,
        )?;
        atomic_write(&file_path, &file)?;
        let _guard = self.index_lock.lock().unwrap();
        index_add(&self.memory_root, &self.index_filename, name, description)
    }

    /// Read one memory's full raw file content (frontmatter + body).
    pub fn read_memory(&self, name: &str) -> AgentResult<String> {
        validate_memory_name(name)?;
        let path = self.memory_path(name);
        fs::read_to_string(&path).map_err(|_| {
            AgentError::internal(format!(
                "memory `{name}` not found (looked at {})",
                path.display()
            ))
        })
    }

    /// Delete the memory `name`: remove its index row first, then the file.
    /// Order matches write's crash invariant: after a crash mid-delete, the
    /// worst case is an orphaned file (no dangling index row).
    pub fn delete_memory(&self, name: &str) -> AgentResult<()> {
        validate_memory_name(name)?;
        let path = self.memory_path(name);
        if !path.exists() {
            return Err(AgentError::internal(format!(
                "memory `{name}` not found (looked at {})",
                path.display()
            )));
        }
        // P2-3 fix: index first, file second — crash-safe (orphan file, no dangling row).
        {
            let _guard = self.index_lock.lock().unwrap();
            index_remove(&self.memory_root, &self.index_filename, name)?;
        }
        fs::remove_file(&path)
            .map_err(|e| AgentError::internal(format!("failed to delete memory `{name}`: {e}")))
    }

    /// List all parseable memories, sorted by name. Corrupt files are skipped
    /// with a warning (fault tolerance) and never break the listing.
    pub fn list_memories(&self) -> Vec<MemoryListEntry> {
        let mut entries = Vec::new();
        let Ok(dir) = fs::read_dir(&self.memory_root) else {
            return entries; // directory missing → no memories yet
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension() != Some("md".as_ref())
                || path.file_name() == Some(self.index_filename.as_ref())
            {
                continue;
            }
            if let Ok(doc) = parse_memory_file(&path) {
                entries.push(MemoryListEntry {
                    name: doc.name,
                    description: doc.description,
                    memory_type: doc.metadata.as_ref().map(|m| m.memory_type.clone()),
                    modified: doc.metadata.as_ref().and_then(|m| m.modified.clone()),
                });
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (modules 1 & 2)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tmp_root() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("memory");
        (dir, root)
    }

    // ── Claude Code compatibility fixtures ──
    //
    // The memory directory is shared with Claude Code (Phase 9c contract).
    // These fixtures pin the cross-writer format: if a fixture stops parsing,
    // phimint silently loses memories Claude Code wrote.

    /// Claude Code's documented format: `metadata` carries only `type` —
    /// no `node_type`, no origin/modified stamps.
    const CC_FORMAT_FILE: &str = "---\n\
name: cc-note\ndescription: written by claude code\n\
metadata:\n  type: feedback\n\
---\n\n来自 Claude Code 的事实\n";

    /// A file carrying frontmatter keys neither writer owns (today): a
    /// top-level `tags` and a `metadata.created`. Updates must not drop them.
    const EXTRA_FIELDS_FILE: &str = "---\n\
name: extra\ndescription: has unknown fields\ntags: [a, b]\n\
metadata:\n  node_type: memory\n  type: project\n  created: 2026-01-01\n\
---\n\nbody\n";

    // ── project_slug ──

    #[test]
    fn slug_encodes_absolute_path() {
        assert_eq!(
            project_slug(Path::new("/Users/xxx/project")),
            "-Users-xxx-project"
        );
    }

    #[test]
    fn slug_matches_claude_code_example() {
        assert_eq!(
            project_slug(Path::new(
                "/Users/kangzengchen/source/buka/buka-works/phimint"
            )),
            "-Users-kangzengchen-source-buka-buka-works-phimint"
        );
    }

    #[test]
    fn slug_strips_trailing_slash() {
        assert_eq!(project_slug(Path::new("/a/b/")), "-a-b");
    }

    #[test]
    fn slug_is_empty_for_root() {
        assert_eq!(project_slug(Path::new("/")), "");
    }

    #[test]
    fn slug_replaces_all_non_alphanumeric() {
        // Dots, underscores, spaces all become `-`
        assert_eq!(
            project_slug(Path::new("/Users/alice/my.project_v2")),
            "-Users-alice-my-project-v2"
        );
    }

    // ── name validation ──

    #[test]
    fn name_validation_accepts_kebab_case() {
        // `^[a-z0-9][a-z0-9-]*$` permits a trailing dash — harmless, and the
        // plan pins the contract to this exact pattern.
        for ok in [
            "a",
            "9",
            "valid-name",
            "a1-b2-c3",
            "9lives",
            "mem0ry",
            "trailing-",
        ] {
            assert!(is_valid_memory_name(ok), "`{ok}` should be valid");
        }
    }

    #[test]
    fn name_validation_rejects_path_escape_and_bad_shapes() {
        for bad in [
            "../../etc/passwd",
            "..",
            ".",
            "-leading-dash",
            "Upper",
            "has space",
            "has_underscore",
            "has.dot",
            "",
            "a/b",
            "中文",
            "a\nb",
        ] {
            assert!(!is_valid_memory_name(bad), "`{bad}` should be invalid");
        }
    }

    #[test]
    fn validate_memory_name_error_mentions_rule() {
        let err = validate_memory_name("../escape").unwrap_err().to_string();
        assert!(err.contains("^[a-z0-9][a-z0-9-]*$"), "{err}");
    }

    #[test]
    fn memory_type_validation() {
        for t in MEMORY_TYPES {
            validate_memory_type(t).unwrap();
        }
        assert!(validate_memory_type("secret").is_err());
        assert!(validate_memory_type("").is_err());
    }

    // ── frontmatter round trip ──

    #[test]
    fn frontmatter_round_trip_preserves_fields() {
        let file = render_memory_file(
            "cargo-commit-discipline",
            "绝不主动 commit；commit 永不带 Cargo 文件",
            "feedback",
            Some("8ee16870-4dc8-4dd1-9d13-4fc7a9980f79"),
            Some("2026-09-04T11:33:28.939Z"),
            "**规则：**\n\n0. 绝不主动 commit\n",
        )
        .unwrap();

        let doc = parse_memory_str(&file).unwrap();
        assert_eq!(doc.name, "cargo-commit-discipline");
        assert_eq!(doc.description, "绝不主动 commit；commit 永不带 Cargo 文件");
        let md = doc.metadata.unwrap();
        assert_eq!(md.node_type, "memory");
        assert_eq!(md.memory_type, "feedback");
        assert_eq!(
            md.origin_session_id.as_deref(),
            Some("8ee16870-4dc8-4dd1-9d13-4fc7a9980f79")
        );
        assert_eq!(md.modified.as_deref(), Some("2026-09-04T11:33:28.939Z"));
        assert_eq!(doc.body, "**规则：**\n\n0. 绝不主动 commit\n");
    }

    #[test]
    fn frontmatter_omits_optional_fields_when_none() {
        let file = render_memory_file("t", "d", "project", None, None, "b").unwrap();
        assert!(!file.contains("originSessionId"));
        assert!(!file.contains("modified"));
        // Required fields still present, in Claude Code layout.
        assert!(file.contains("node_type: memory"));
        assert!(file.contains("type: project"));
    }

    #[test]
    fn frontmatter_escapes_special_yaml_characters() {
        let tricky = "has: colon, \"quotes\", #hash and 中文";
        let file = render_memory_file("t", tricky, "user", None, None, "b").unwrap();
        let doc = parse_memory_str(&file).unwrap();
        assert_eq!(doc.description, tricky);
    }

    #[test]
    fn parse_rejects_missing_frontmatter() {
        assert!(parse_memory_str("just some text\n").is_none());
        assert!(parse_memory_str("---\nname: x\n").is_none()); // no closing ---
    }

    #[test]
    fn parse_rejects_empty_name_or_description() {
        let bad = "---\nname: \"\"\ndescription: d\n---\n\nbody\n";
        assert!(parse_memory_str(bad).is_none());
        let bad = "---\nname: x\ndescription: \"\"\n---\n\nbody\n";
        assert!(parse_memory_str(bad).is_none());
    }

    #[test]
    fn parse_tolerates_missing_metadata_block() {
        let raw = "---\nname: legacy\ndescription: no metadata\n---\n\nbody\n";
        let doc = parse_memory_str(raw).unwrap();
        assert_eq!(doc.name, "legacy");
        assert!(doc.metadata.is_none());
    }

    #[test]
    fn parse_tolerates_claude_code_metadata_without_node_type() {
        let doc = parse_memory_str(CC_FORMAT_FILE).expect("Claude Code format must parse");
        assert_eq!(doc.name, "cc-note");
        let meta = doc.metadata.expect("metadata present");
        assert_eq!(meta.memory_type, "feedback");
    }

    #[test]
    fn list_memories_includes_claude_code_format_files() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("cc-note.md"), CC_FORMAT_FILE).unwrap();
        std::fs::write(
            root.join("ours.md"),
            "---\nname: ours\ndescription: ours\nmetadata:\n  node_type: memory\n  type: project\n---\n\nb\n",
        )
        .unwrap();
        let names: Vec<_> = store.list_memories().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["cc-note".to_string(), "ours".to_string()]);
        drop(dir);
    }

    // ── timestamps ──

    #[test]
    fn civil_from_days_known_values() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    }

    #[test]
    fn now_iso8601_has_canonical_shape() {
        let ts = now_iso8601();
        // 2026-09-13T08:15:30.123Z
        assert_eq!(ts.len(), 24, "{ts}");
        assert!(ts.ends_with('Z'));
        let bytes = ts.as_bytes();
        for (i, expected) in b"dddd-dd-ddTdd:dd:dd.dddZ".iter().enumerate() {
            let actual = bytes[i];
            match expected {
                b'd' => assert!(actual.is_ascii_digit(), "{ts}"),
                b'-' | b'T' | b':' | b'.' | b'Z' => assert_eq!(actual, *expected, "{ts}"),
                _ => unreachable!(),
            }
        }
        // Year is plausible (this code was written in 2026).
        let year: i32 = ts[..4].parse().unwrap();
        assert!((2024..=2100).contains(&year), "{ts}");
    }

    // ── atomic write ──

    #[test]
    fn atomic_write_writes_content_and_leaves_no_tmp() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("file.md");
        atomic_write(&target, "hello").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello");
        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(leftovers, vec!["file.md".to_string()]);
        drop(dir);
    }

    #[test]
    fn atomic_write_overwrites_previous_content() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("file.md");
        atomic_write(&target, "old").unwrap();
        atomic_write(&target, "new").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        drop(dir);
    }

    // ── index_add ──

    #[test]
    fn index_add_creates_index_on_first_write() {
        let (dir, root) = tmp_root();
        index_add(&root, "MEMORY.md", "alpha", "first memory").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "- [alpha](alpha.md) — first memory\n");
        drop(dir);
    }

    #[test]
    fn index_add_appends_without_touching_other_rows() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        // Claude Code style rows with human titles (Chinese link text).
        std::fs::write(
            root.join("MEMORY.md"),
            "- [提交纪律](cargo-commit-discipline.md) — 绝不主动 commit\n- [工作原则](working-principles.md) — 高内聚低耦合\n",
        )
        .unwrap();
        index_add(&root, "MEMORY.md", "new-entry", "brand new").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index,
            "- [提交纪律](cargo-commit-discipline.md) — 绝不主动 commit\n\
             - [工作原则](working-principles.md) — 高内聚低耦合\n\
             - [new-entry](new-entry.md) — brand new\n"
        );
        drop(dir);
    }

    #[test]
    fn index_add_update_preserves_human_title() {
        // Shared-index contract: a refresh touches only the `— desc` tail.
        // The link text (often a human title written by Claude Code) survives.
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [旧标题](target.md) — 旧描述\n- [工作原则](working-principles.md) — 高内聚低耦合\n",
        )
        .unwrap();
        index_add(&root, "MEMORY.md", "target", "new description").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index,
            "- [旧标题](target.md) — new description\n\
             - [工作原则](working-principles.md) — 高内聚低耦合\n"
        );
        drop(dir);
    }

    #[test]
    fn index_add_is_name_scoped_not_prefix_scoped() {
        // `test.md` must not match the row for `test-long.md`.
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [Long](test-long.md) — long desc\n",
        )
        .unwrap();
        index_add(&root, "MEMORY.md", "test", "short desc").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index,
            "- [Long](test-long.md) — long desc\n- [test](test.md) — short desc\n"
        );
        drop(dir);
    }

    #[test]
    fn index_add_sanitizes_newlines_in_description() {
        let (dir, root) = tmp_root();
        index_add(&root, "MEMORY.md", "multi", "line one\nline two").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index.lines().count(), 1, "{index}");
        assert!(index.contains("line one line two"), "{index}");
        drop(dir);
    }

    // ── index_remove ──

    #[test]
    fn index_remove_deletes_only_target_row() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [A](a.md) — a\n- [B](b.md) — b\n- [C](c.md) — c\n",
        )
        .unwrap();
        index_remove(&root, "MEMORY.md", "b").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "- [A](a.md) — a\n- [C](c.md) — c\n");
        drop(dir);
    }

    #[test]
    fn index_remove_last_row_leaves_empty_index() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("MEMORY.md"), "- [A](a.md) — a\n").unwrap();
        index_remove(&root, "MEMORY.md", "a").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "");
        drop(dir);
    }

    #[test]
    fn index_remove_missing_name_is_noop() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("MEMORY.md"), "- [A](a.md) — a\n").unwrap();
        index_remove(&root, "MEMORY.md", "ghost").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "- [A](a.md) — a\n");
        drop(dir);
    }

    #[test]
    fn index_ops_tolerate_missing_index_file() {
        let (dir, root) = tmp_root();
        index_add(&root, "MEMORY.md", "a", "a desc").unwrap();
        index_remove(&root, "MEMORY.md", "a").unwrap();
        assert_eq!(std::fs::read_to_string(root.join("MEMORY.md")).unwrap(), "");
        drop(dir);
    }

    // ── rebuild_index ──

    #[test]
    fn rebuild_index_recovers_from_memory_files() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("beta.md"),
            "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
        )
        .unwrap();
        std::fs::write(
            root.join("alpha.md"),
            "---\nname: alpha\ndescription: first\nmetadata:\n  node_type: memory\n  type: user\n---\n\nbody\n",
        )
        .unwrap();
        // Corrupt file must be skipped, not fatal.
        std::fs::write(root.join("broken.md"), "not frontmatter at all\n").unwrap();
        // Index file itself must be excluded.
        std::fs::write(root.join("MEMORY.md"), "garbage").unwrap();

        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index,
            "- [alpha](alpha.md) — first\n- [beta](beta.md) — second\n"
        );
        drop(dir);
    }

    #[test]
    fn rebuild_index_skips_filename_name_mismatch() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("mismatch.md"),
            "---\nname: other-name\ndescription: d\n---\n\nbody\n",
        )
        .unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index, "",
            "mismatched file must not produce a dangling link: {index}"
        );
        drop(dir);
    }

    #[test]
    fn rebuild_index_preserves_human_title_rows() {
        // Conservative repair: a row we can regenerate is still re-emitted
        // byte-identical when one exists — the human title lives only in the
        // index and cannot be recovered from the file.
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("beta.md"),
            "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
        )
        .unwrap();
        std::fs::write(root.join("MEMORY.md"), "- [人类标题](beta.md) — second\n").unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "- [人类标题](beta.md) — second\n");
        drop(dir);
    }

    #[test]
    fn rebuild_index_keeps_rows_for_files_it_cannot_index() {
        // Files that fail our parse (corrupt, foreign format, name mismatch)
        // keep their existing rows — a rebuild never erases what it doesn't
        // understand. Rows whose file is gone are still dropped (dangling).
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("broken.md"), "no frontmatter here\n").unwrap();
        std::fs::write(
            root.join("mismatch.md"),
            "---\nname: other-name\ndescription: d\n---\n\nbody\n",
        )
        .unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [破](broken.md) — 尽量保留\n\
             - [改名](mismatch.md) — 也要保留\n\
             - [stale](gone.md) — 文件已删\n",
        )
        .unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index, "- [破](broken.md) — 尽量保留\n- [改名](mismatch.md) — 也要保留\n",
            "unparseable rows survive; stale rows go: {index}"
        );
        drop(dir);
    }

    // ── MemoryStore: write / update / read / delete / list ──

    #[test]
    fn store_first_write_creates_dir_file_and_index() {
        let (dir, root) = tmp_root();
        assert!(!root.exists());
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        store
            .write_memory("my-note", "a note", "project", "the body", Some("sess-1"))
            .unwrap();
        assert!(root.is_dir(), "memory dir must be auto-created");
        let raw = std::fs::read_to_string(root.join("my-note.md")).unwrap();
        assert!(raw.contains("name: my-note"));
        assert!(raw.contains("originSessionId: sess-1"));
        assert!(raw.contains("modified: "));
        assert!(raw.ends_with("the body"));
        let index = store.read_index();
        assert!(
            index.contains("- [my-note](my-note.md) — a note"),
            "{index}"
        );
        drop(dir);
    }

    #[test]
    fn store_same_name_update_writes_single_file_with_second_content() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        store
            .write_memory("dup", "first desc", "project", "first body", None)
            .unwrap();
        store
            .write_memory("dup", "second desc", "feedback", "second body", None)
            .unwrap();

        let files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".md") && n != "MEMORY.md")
            .collect();
        assert_eq!(files, vec!["dup.md".to_string()], "no duplicate files");

        let raw = std::fs::read_to_string(root.join("dup.md")).unwrap();
        assert!(raw.contains("second body"));
        assert!(!raw.contains("first body"));

        let index = store.read_index();
        assert_eq!(index.matches("(dup.md)").count(), 1, "{index}");
        assert!(index.contains("second desc"), "{index}");
        drop(dir);
    }

    #[test]
    fn store_update_preserves_unknown_frontmatter_fields() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("extra.md"), EXTRA_FIELDS_FILE).unwrap();

        store
            .write_memory("extra", "updated desc", "project", "new body", None)
            .unwrap();

        let raw = std::fs::read_to_string(root.join("extra.md")).unwrap();
        let doc = parse_memory_str(&raw).unwrap();
        assert_eq!(doc.description, "updated desc");
        assert_eq!(doc.body, "new body");
        // Keys neither writer owns must survive the update (round-trip).
        assert!(raw.contains("tags:"), "top-level `tags` dropped:\n{raw}");
        assert!(
            raw.contains("[a, b]") || raw.contains("- a"),
            "`tags` values dropped:\n{raw}"
        );
        assert!(
            raw.contains("created: 2026-01-01"),
            "metadata.created dropped:\n{raw}"
        );
        drop(dir);
    }

    #[test]
    fn store_read_returns_raw_file_content() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        store
            .write_memory("note", "d", "user", "remember this", None)
            .unwrap();
        let raw = store.read_memory("note").unwrap();
        assert!(raw.starts_with("---\n"));
        assert!(raw.contains("remember this"));
        drop(dir);
    }

    #[test]
    fn store_read_missing_is_error() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root, "MEMORY.md");
        let err = store.read_memory("ghost").unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
        drop(dir);
    }

    #[test]
    fn store_delete_removes_file_and_index_row() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        store
            .write_memory("gone", "d1", "user", "b1", None)
            .unwrap();
        store
            .write_memory("kept", "d2", "user", "b2", None)
            .unwrap();

        store.delete_memory("gone").unwrap();
        assert!(!root.join("gone.md").exists());
        assert!(root.join("kept.md").exists());
        let index = store.read_index();
        assert!(!index.contains("(gone.md)"), "{index}");
        assert!(index.contains("(kept.md)"), "{index}");
        drop(dir);
    }

    #[test]
    fn store_delete_missing_is_error() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root, "MEMORY.md");
        let err = store.delete_memory("ghost").unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
        drop(dir);
    }

    #[test]
    fn store_write_rejects_invalid_name_and_type() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        assert!(store.write_memory("../x", "d", "user", "b", None).is_err());
        assert!(
            store
                .write_memory("ok-name", "d", "wrong-type", "b", None)
                .is_err()
        );
        assert!(
            store
                .write_memory("ok-name", "multi\nline", "user", "b", None)
                .is_err()
        );
        assert!(
            !root.join("ok-name.md").exists(),
            "nothing may be written on rejection"
        );
        drop(dir);
    }

    #[test]
    fn store_list_skips_corrupt_and_sorts() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        store
            .write_memory("zeta", "last", "project", "b", None)
            .unwrap();
        store
            .write_memory("alpha", "first", "user", "b", None)
            .unwrap();
        std::fs::write(root.join("corrupt.md"), "garbage without frontmatter").unwrap();

        let entries = store.list_memories();
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
        assert_eq!(entries[0].memory_type.as_deref(), Some("user"));
        drop(dir);
    }

    #[test]
    fn store_list_missing_dir_is_empty() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root, "MEMORY.md");
        assert!(store.list_memories().is_empty());
        assert_eq!(store.read_index(), "");
        drop(dir);
    }

    // ── concurrency (in-process) ──

    #[test]
    fn concurrent_writes_keep_index_complete() {
        let (dir, root) = tmp_root();
        let store = Arc::new(MemoryStore::new(root.clone(), "MEMORY.md"));
        const N: usize = 32;

        let mut handles = Vec::new();
        for i in 0..N {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let name = format!("mem-{i:03}");
                store
                    .write_memory(
                        &name,
                        &format!("desc {i}"),
                        "project",
                        &format!("body {i}"),
                        None,
                    )
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let index = store.read_index();
        for i in 0..N {
            let name = format!("mem-{i:03}");
            assert!(
                index.contains(&format!("({name}.md)")),
                "index is missing `{name}`:\n{index}"
            );
        }
        assert_eq!(index.lines().count(), N, "no rows may be lost:\n{index}");

        let files = store.list_memories();
        assert_eq!(files.len(), N);
        drop(dir);
    }

    #[test]
    fn concurrent_same_name_writes_leave_consistent_state() {
        let (dir, root) = tmp_root();
        let store = Arc::new(MemoryStore::new(root.clone(), "MEMORY.md"));

        let mut handles = Vec::new();
        for i in 0..16 {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                store
                    .write_memory(
                        "shared",
                        &format!("desc {i}"),
                        "user",
                        &format!("body {i}"),
                        None,
                    )
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // One file, one row, and the file parses back to whatever row exists.
        let files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".md"))
            .collect();
        assert_eq!(files.len(), 2, "index + exactly one memory file: {files:?}");
        let index = store.read_index();
        assert_eq!(index.matches("(shared.md)").count(), 1, "{index}");
        drop(dir);
    }

    // ── shared-index anchor safety ──
    //
    // Row identity is the row's own link target (`- [title](name.md)`), never a
    // substring of some other row's description — descriptions are free text
    // and may embed links to other memories.

    #[test]
    fn index_add_anchor_inside_other_row_description_does_not_hijack() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [beta](beta.md) — see [x](alpha.md)\n",
        )
        .unwrap();
        index_add(&root, "MEMORY.md", "alpha", "new").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert!(
            index.contains("- [alpha](alpha.md) — new"),
            "alpha row must be created despite the anchor in beta's description: {index}"
        );
        assert!(
            index.contains("- [beta](beta.md) — see [x](alpha.md)"),
            "beta's row must be untouched: {index}"
        );
        drop(dir);
    }

    #[test]
    fn index_remove_ignores_anchor_inside_other_rows_description() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [alpha](alpha.md) — mine\n- [beta](beta.md) — see [x](alpha.md)\n",
        )
        .unwrap();
        index_remove(&root, "MEMORY.md", "alpha").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index, "- [beta](beta.md) — see [x](alpha.md)\n",
            "only alpha's own row may go: {index}"
        );
        drop(dir);
    }

    #[test]
    fn rebuild_binds_rows_by_row_link_not_description_substring() {
        // beta's description mentions alpha's link and sits ABOVE alpha's
        // human-titled row. Each file must bind to its own row.
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        for name in ["alpha", "beta"] {
            std::fs::write(
                root.join(format!("{name}.md")),
                format!("---\nname: {name}\ndescription: d\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n"),
            )
            .unwrap();
        }
        std::fs::write(
            root.join("MEMORY.md"),
            "- [beta](beta.md) — see [x](alpha.md)\n- [人类](alpha.md) — 旧\n",
        )
        .unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index.lines().count(), 2, "no duplication, no loss: {index}");
        assert!(index.contains("- [人类](alpha.md) — 旧"), "{index}");
        assert!(
            index.contains("- [beta](beta.md) — see [x](alpha.md)"),
            "{index}"
        );
        drop(dir);
    }

    // ── merge fallbacks and stamps (shared-writer contract) ──

    #[test]
    fn update_of_crlf_file_preserves_foreign_keys() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        // A foreign writer on Windows: CRLF line endings throughout.
        std::fs::write(
            root.join("crlf.md"),
            "---\r\nname: crlf\ndescription: old\r\ntags: [a, b]\r\nmetadata:\r\n  type: feedback\r\n---\r\n\r\nbody\r\n",
        )
        .unwrap();
        store
            .write_memory("crlf", "new desc", "feedback", "new body", None)
            .unwrap();
        let raw = std::fs::read_to_string(root.join("crlf.md")).unwrap();
        assert!(
            raw.contains("tags:"),
            "CRLF file lost foreign keys:\n{raw:?}"
        );
        assert!(raw.contains("new body"), "{raw:?}");
        drop(dir);
    }

    #[test]
    fn update_of_unparseable_yaml_frontmatter_falls_back_to_known_fields() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        // Valid delimiters, invalid YAML inside — exercises the documented
        // best-effort fallback (foreign keys cannot be merged and are lost).
        std::fs::write(
            root.join("bad.md"),
            "---\nname: [unclosed\ntags: keep-me\n---\n\nbody\n",
        )
        .unwrap();
        store
            .write_memory("bad", "d", "project", "new body", None)
            .unwrap();
        let raw = std::fs::read_to_string(root.join("bad.md")).unwrap();
        let doc = parse_memory_str(&raw).unwrap();
        assert_eq!(doc.description, "d");
        assert_eq!(doc.body, "new body");
        assert!(
            !raw.contains("keep-me"),
            "fallback contract changed (update the module doc if so): {raw}"
        );
        drop(dir);
    }

    #[test]
    fn update_with_none_origin_session_id_preserves_existing_stamp() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        store
            .write_memory("s", "d", "user", "b", Some("sess-1"))
            .unwrap();
        store.write_memory("s", "d2", "user", "b2", None).unwrap();
        let raw = std::fs::read_to_string(root.join("s.md")).unwrap();
        assert!(
            raw.contains("originSessionId: sess-1"),
            "stamp is shared provenance — set-or-keep, never clear on None:\n{raw}"
        );
        assert!(raw.contains("modified: "), "{raw}");
        drop(dir);
    }

    #[test]
    fn update_of_metadata_less_file_adds_metadata_keeps_unknown_keys() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("legacy.md"),
            "---\nname: legacy\ndescription: old\ntags: [a, b]\n---\n\nbody\n",
        )
        .unwrap();
        store
            .write_memory("legacy", "new desc", "project", "new body", None)
            .unwrap();
        let raw = std::fs::read_to_string(root.join("legacy.md")).unwrap();
        let doc = parse_memory_str(&raw).unwrap();
        assert_eq!(doc.description, "new desc");
        let md = doc.metadata.expect("metadata block must be created");
        assert_eq!(md.node_type, "memory");
        assert_eq!(md.memory_type, "project");
        assert!(raw.contains("tags:"), "unknown key dropped:\n{raw}");
        drop(dir);
    }

    // ── rebuild conservativeness ──

    #[test]
    fn rebuild_index_preserves_curated_row_order() {
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        for name in ["zeta", "alpha"] {
            std::fs::write(
                root.join(format!("{name}.md")),
                format!("---\nname: {name}\ndescription: {name} desc\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n"),
            )
            .unwrap();
        }
        std::fs::write(
            root.join("MEMORY.md"),
            "- [Z](zeta.md) — zeta desc\n- [A](alpha.md) — alpha desc\n",
        )
        .unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(
            index, "- [Z](zeta.md) — zeta desc\n- [A](alpha.md) — alpha desc\n",
            "rebuild must not reorder a curated index: {index}"
        );
        drop(dir);
    }

    #[test]
    fn rebuild_index_does_not_refresh_stale_row_tail() {
        // File description has moved on; the row tail is stale. Conservative
        // rebuild re-emits the row byte-identical (unlike index_add, which
        // refreshes the tail) — the row may be human-curated.
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("beta.md"),
            "---\nname: beta\ndescription: file says this now\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
        )
        .unwrap();
        std::fs::write(
            root.join("MEMORY.md"),
            "- [人类标题](beta.md) — stale tail\n",
        )
        .unwrap();
        rebuild_index(&root, "MEMORY.md").unwrap();
        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert_eq!(index, "- [人类标题](beta.md) — stale tail\n", "{index}");
        drop(dir);
    }

    #[cfg(unix)]
    #[test]
    fn rebuild_index_fails_when_index_cannot_be_read() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, root) = tmp_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("beta.md"),
            "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
        )
        .unwrap();
        std::fs::write(root.join("MEMORY.md"), "- [人类标题](beta.md) — second\n").unwrap();
        std::fs::set_permissions(
            root.join("MEMORY.md"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        if std::fs::read_to_string(root.join("MEMORY.md")).is_ok() {
            // e.g. running as root — cannot simulate an unreadable file.
            return;
        }
        // Unreadable (not missing): a conservative rebuild must refuse rather
        // than treat the index as empty and regenerate human titles away.
        let result = rebuild_index(&root, "MEMORY.md");
        std::fs::set_permissions(
            root.join("MEMORY.md"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert!(result.is_err(), "must not proceed on a read error");
        drop(dir);
    }

    #[test]
    fn description_with_unicode_line_separators_is_rejected() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        // U+2028 passes a naive `contains('\n')` check and visually forges
        // extra rows when the index is injected into the prompt.
        let err = store
            .write_memory(
                "trap",
                "ok\u{2028}- [victim](victim.md) — forged",
                "user",
                "b",
                None,
            )
            .unwrap_err();
        assert!(err.to_string().contains("single line"), "{err}");
        drop(dir);
    }

    #[test]
    fn update_preserves_non_mapping_metadata_value() {
        let (dir, root) = tmp_root();
        let store = MemoryStore::new(root.clone(), "MEMORY.md");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("odd.md"),
            "---\nname: odd\ndescription: old\nmetadata: some-string\n---\n\nbody\n",
        )
        .unwrap();
        store
            .write_memory("odd", "new desc", "project", "new body", None)
            .unwrap();
        let raw = std::fs::read_to_string(root.join("odd.md")).unwrap();
        assert!(
            raw.contains("some-string"),
            "foreign metadata value must survive: {raw}"
        );
        assert!(raw.contains("new desc"), "{raw}");
        drop(dir);
    }

    // ── serde key names vs the merge writer's string literals ──

    #[test]
    fn frontmatter_serde_keys_match_merge_upsert_names() {
        // The merge writer re-types key names as string literals; these serde
        // renames are the other half. If either side drifts, updates start
        // dropping or duplicating keys owned by the other writer.
        let fm = MemoryFrontmatter {
            name: "n".to_string(),
            description: "d".to_string(),
            metadata: Some(MemoryFrontmatterMetadata {
                node_type: "memory".to_string(),
                memory_type: "project".to_string(),
                origin_session_id: Some("s".to_string()),
                modified: Some("m".to_string()),
            }),
        };
        let yaml = serde_yaml::to_string(&fm).unwrap();
        for key in [
            "name:",
            "description:",
            "metadata:",
            "node_type:",
            "type:",
            "originSessionId:",
            "modified:",
        ] {
            assert!(yaml.contains(key), "serde must emit `{key}`:\n{yaml}");
        }
    }
}
