//! Filesystem-backed skill discovery and resolution.
//!
//! `SkillResolver` scans SKILL.md directories (later dirs override earlier
//! ones on name collision), resolves `/name args` slash input with fuzzy
//! matching, and serves exact-name lookups for the model-facing `skill` tool.
//! Host apps decide *which* directories to scan (policy); this module only
//! knows *how* to scan and resolve (mechanism).

use std::collections::HashMap;
use std::path::PathBuf;

use super::Skill;
use super::prompt_skill::{PromptSkill, SkillScope};

/// Scanned skill set backing slash lookup and the `skill` tool.
pub struct SkillResolver {
    skills: Vec<PromptSkill>,
}

/// A matched `/name args` slash invocation, with the skill's lifetime scope.
///
/// Hosts dispatch on [`scope`](ResolvedSkill::scope): `Session` skills go to
/// the system prompt (persist for the session), `Turn` skills are injected as
/// ephemeral turn input (one-shot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSkill {
    /// Canonical name of the matched skill.
    pub name: String,
    /// Body with `$ARGUMENTS` substituted.
    pub body: String,
    /// Lifetime scope (frontmatter `scope:`; missing/unknown → `session`).
    pub scope: SkillScope,
}

/// Failure reasons for `resolve_by_name` (the `skill` tool maps them to
/// different error messages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillLookupError {
    /// No skill with this name.
    NotFound,
    /// The skill is flagged `disable-model-invocation`: denied for the model
    /// (the user's slash path is unaffected — that flag only constrains
    /// model-side triggering).
    ModelInvocationDenied,
}

impl SkillResolver {
    /// Scan skills from multiple directories (later-scanned dirs win on name
    /// collision).
    pub fn from_dirs(dirs: &[PathBuf]) -> Self {
        let mut by_name: HashMap<String, PromptSkill> = HashMap::new();

        for dir in dirs {
            match PromptSkill::scan_dir(dir) {
                Ok(skills) => {
                    for skill in skills {
                        let name = skill.name().to_string();
                        by_name.insert(name, skill); // later entry overrides earlier one
                    }
                }
                Err(e) => {
                    tracing::warn!(dir = %dir.display(), error = %e, "failed to scan skill directory");
                }
            }
        }

        let mut skills: Vec<PromptSkill> = by_name.into_values().collect();
        skills.sort_by(|a, b| a.name().cmp(b.name()));
        Self { skills }
    }

    /// Parse user input; if it is `/skill-name args`, return the resolved skill body.
    ///
    /// Returns `None` when the input is not a slash command (or no skill
    /// matched); it should then be submitted as-is.
    pub fn resolve(&self, input: &str) -> Option<String> {
        self.resolve_with_meta(input).map(|r| r.body)
    }

    /// Like [`resolve`](Self::resolve), but also returns the matched skill's
    /// canonical name and lifetime scope — hosts need both to dispatch
    /// session-scope skills into their system prompt.
    pub fn resolve_with_meta(&self, input: &str) -> Option<ResolvedSkill> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        // Split: `/skill-name arg1 arg2 ...` → name="skill-name", rest="arg1 arg2 ..."
        let without_slash = &trimmed[1..];
        let (name, raw_args) = match without_slash.split_once(char::is_whitespace) {
            Some((n, rest)) => (n, rest.trim()),
            None => (without_slash, ""),
        };

        // Name must be non-empty and contain no whitespace (prevents "/ " from
        // being treated as a skill command)
        if name.is_empty() || name.contains(char::is_whitespace) {
            return None;
        }

        // Fuzzy match: exact → suffix → contains → word-overlap; within a tier
        // pick the shortest name
        let skill = self.fuzzy_find(name)?;

        if !skill.is_user_invocable() {
            tracing::info!(
                name = name,
                "skill is not user-invocable, passing through as text"
            );
            return None;
        }

        // Named parameters are not parsed yet; only raw_arguments are passed through
        let params = HashMap::new();
        let body = skill.resolve_body(&params, raw_args);

        tracing::info!(
            input = name,
            matched = skill.name(),
            args = raw_args,
            body_len = body.len(),
            scope = ?skill.scope(),
            "resolved /skill command"
        );

        Some(ResolvedSkill {
            name: skill.name().to_string(),
            body,
            scope: skill.scope(),
        })
    }

    /// Fuzzy skill lookup by priority: exact → suffix → contains →
    /// word-overlap. Within a tier, take the shortest name (most specific).
    /// Returns `None` when nothing matches at all.
    fn fuzzy_find(&self, query: &str) -> Option<&PromptSkill> {
        // Candidates: only user_invocable skills participate in matching
        let candidates: Vec<&PromptSkill> = self
            .skills
            .iter()
            .filter(|s| s.is_user_invocable())
            .collect();

        // 1. Exact match
        if let Some(s) = candidates.iter().find(|s| s.name() == query) {
            return Some(s);
        }

        // 2. Suffix match: skill name ends with query (e.g. requesting-code-review
        //    ends with code-review)
        let mut suffix_hits: Vec<&&PromptSkill> = candidates
            .iter()
            .filter(|s| s.name().ends_with(query) && s.name() != query)
            .collect();
        if !suffix_hits.is_empty() {
            suffix_hits.sort_by_key(|s| s.name().len());
            return Some(suffix_hits[0]);
        }

        // 3. Substring match: skill name contains query
        let mut contains_hits: Vec<&&PromptSkill> = candidates
            .iter()
            .filter(|s| s.name().contains(query))
            .collect();
        if !contains_hits.is_empty() {
            contains_hits.sort_by_key(|s| s.name().len());
            return Some(contains_hits[0]);
        }

        // 4. Word-level match: query split into words is a subset of the skill
        //    name's split words
        //    e.g. "review-pr" → ["review", "pr"] ⊆ ["pre", "landing", "pr", "review"]
        let query_words: Vec<&str> = query.split('-').collect();
        let mut word_hits: Vec<(&&PromptSkill, usize)> = candidates
            .iter()
            .filter_map(|s| {
                let name_words: Vec<&str> = s.name().split('-').collect();
                let all_match = query_words.iter().all(|qw| name_words.contains(qw));
                if all_match {
                    Some((s, name_words.len()))
                } else {
                    None
                }
            })
            .collect();
        if !word_hits.is_empty() {
            // Take the one with the fewest words (most specific)
            word_hits.sort_by_key(|(_, len)| *len);
            return Some(word_hits[0].0);
        }

        None
    }

    /// Number of loaded skills (for catalog building and tests).
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether empty (for catalog-injection emptiness checks and tests).
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Read-only access to the loaded skills (sorted by name).
    ///
    /// For same-crate consumers such as catalog rendering to iterate; no
    /// `&mut` is exposed — the resolver is a shared snapshot, changes go
    /// through rebuilding (hot reload).
    pub fn entries(&self) -> &[PromptSkill] {
        &self.skills
    }

    /// List the names of all loaded skills (including `user_invocable: false`;
    /// currently only used for host log lines).
    pub fn skill_names(&self) -> Vec<&str> {
        self.skills.iter().map(|s| s.name()).collect()
    }

    /// List (name, description) summaries of all loaded skills (for the `/`
    /// picker display).
    pub fn skill_summaries(&self) -> Vec<(String, String)> {
        self.skills
            .iter()
            .map(|s| (s.name().to_string(), s.brief_description()))
            .collect()
    }

    /// Exact-match by name (skill tool uses this; `/` slash path uses `resolve`).
    ///
    /// Unlike `resolve`, this does NOT filter on `user_invocable` — model tools
    /// can load every skill, and the catalog lists every skill regardless.
    /// Skills flagged `disable-model-invocation` are refused: the trait contract
    /// ("the LLM cannot auto-trigger this skill") is enforced at both model
    /// surfaces — the catalog omits them and this lookup denies them.
    /// Returns `(body, name)` so the caller can trace which skill was served.
    pub fn resolve_by_name(
        &self,
        name: &str,
        args: &str,
    ) -> Result<(String, &str), SkillLookupError> {
        let skill = self.skills.iter().find(|s| s.name() == name);
        let Some(skill) = skill else {
            return Err(SkillLookupError::NotFound);
        };
        if skill.disable_model_invocation() {
            return Err(SkillLookupError::ModelInvocationDenied);
        }
        let params = HashMap::new();
        let body = skill.resolve_body(&params, args);
        Ok((body, skill.name()))
    }

    /// Absolute path of the `SKILL.md` that backs a named skill (for the
    /// overflow fallback: tool returns truncated body + path for read_file).
    pub fn source_path_for(&self, name: &str) -> Option<&std::path::Path> {
        self.skills.iter().find(|s| s.name() == name)?.source_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Create a temporary skill directory structure:
    ///   tmp_dir/skills/test-skill/SKILL.md
    fn make_skill_dir(tmp: &Path, name: &str, body: &str, user_invocable: bool) {
        let skill_dir = tmp.join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        let invocable = if user_invocable { "true" } else { "false" };
        let content = format!(
            "---\nname: {name}\ndescription: test skill\nuser-invocable: {invocable}\n---\n\n{body}"
        );
        fs::write(skill_dir.join("SKILL.md"), content).unwrap();
    }

    #[test]
    fn resolve_slash_command() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "Run tests then commit.", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert_eq!(resolver.len(), 1);

        let result = resolver.resolve("/commit");
        assert!(result.is_some());
        assert!(result.unwrap().contains("Run tests then commit."));
    }

    #[test]
    fn resolve_slash_command_with_args() {
        let tmp = tempfile::tempdir().unwrap();
        let skill_dir = tmp.path().join("skills").join("commit");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nRun tests then commit.\nArgs: $ARGUMENTS",
        )
        .unwrap();

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        let result = resolver.resolve("/commit --amend");
        assert!(result.is_some());
        let body = result.unwrap();
        assert!(
            body.contains("--amend"),
            "body should contain raw arguments"
        );
    }

    #[test]
    fn non_slash_input_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("hello world").is_none());
        assert!(resolver.resolve("no slash").is_none());
    }

    #[test]
    fn unknown_skill_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/nonexistent").is_none());
    }

    #[test]
    fn not_user_invocable_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "internal", "body", false);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/internal").is_none());
    }

    #[test]
    fn bare_slash_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/").is_none());
        assert!(resolver.resolve("/ ").is_none());
    }

    #[test]
    fn project_level_overrides_user_level() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("user_skills");
        let proj_dir = tmp.path().join("proj_skills");

        // Same skill name, different body
        let sd = user_dir.join("commit").join("SKILL.md");
        fs::create_dir_all(sd.parent().unwrap()).unwrap();
        fs::write(
            &sd,
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nuser body",
        )
        .unwrap();

        let sd = proj_dir.join("commit").join("SKILL.md");
        fs::create_dir_all(sd.parent().unwrap()).unwrap();
        fs::write(
            &sd,
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nproject body",
        )
        .unwrap();

        // Later-scanned dirs take priority
        let resolver = SkillResolver::from_dirs(&[user_dir, proj_dir]);
        assert_eq!(resolver.len(), 1);
        let body = resolver.resolve("/commit").unwrap();
        assert!(
            body.contains("project body"),
            "project-level should override user-level"
        );
    }

    #[test]
    fn empty_dir_yields_empty_resolver() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::from_dirs(&[tmp.path().join("nonexistent")]);
        assert!(resolver.is_empty());
        assert!(resolver.resolve("/anything").is_none());
    }

    // ── Fuzzy matching tests ──

    #[test]
    fn fuzzy_suffix_match() {
        // Simulate the Claude Code scenario: skill named requesting-code-review,
        // user types /code-review
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(
            tmp.path(),
            "requesting-code-review",
            "review body here",
            true,
        );

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        let result = resolver.resolve("/code-review");
        assert!(
            result.is_some(),
            "suffix match should find requesting-code-review"
        );
        assert!(result.unwrap().contains("review body here"));
    }

    #[test]
    fn fuzzy_contains_match() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "design-review", "design body", true);
        make_skill_dir(tmp.path(), "requesting-code-review", "code body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // "review" is a substring but not a suffix → takes the contains branch
        // Both contain "review"; pick the shortest (design-review)
        let result = resolver.resolve("/review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("design body"),
            "contains match should pick shortest name"
        );
    }

    #[test]
    fn fuzzy_exact_takes_priority() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "review", "exact body", true);
        make_skill_dir(tmp.path(), "code-review", "suffix body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // Exact match "review" should win over suffix match "code-review"
        let result = resolver.resolve("/review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("exact body"),
            "exact match should win over suffix"
        );
    }

    #[test]
    fn fuzzy_suffix_picks_shortest() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "requesting-code-review", "long body", true);
        make_skill_dir(tmp.path(), "code-review", "short body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // Both end with "code-review"; pick the shortest
        let result = resolver.resolve("/code-review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("short body"),
            "suffix match should pick shortest name"
        );
    }

    #[test]
    fn fuzzy_word_overlap() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(
            tmp.path(),
            "finishing-a-development-branch",
            "finish body",
            true,
        );

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // "dev-branch" → ["dev", "branch"]; skill splits to
        // ["finishing","a","development","branch"].
        // "dev" is not among the skill's words (the skill uses "development"),
        // so no match
        let result = resolver.resolve("/dev-branch");
        assert!(result.is_none(), "partial word should not match");

        // "development-branch" → full word match
        let result = resolver.resolve("/development-branch");
        assert!(result.is_some());
        assert!(result.unwrap().contains("finish body"));
    }

    // ── resolve_by_name (`skill` tool path) ──

    #[test]
    fn resolve_by_name_denies_disable_model_invocation() {
        let tmp = tempfile::tempdir().unwrap();
        let deploy_dir = tmp.path().join("skills").join("deploy");
        fs::create_dir_all(&deploy_dir).unwrap();
        fs::write(
            deploy_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: d\nuser-invocable: true\ndisable-model-invocation: true\n---\n\nship it",
        )
        .unwrap();
        make_skill_dir(tmp.path(), "commit", "commit body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        assert_eq!(
            resolver.resolve_by_name("deploy", ""),
            Err(SkillLookupError::ModelInvocationDenied),
            "disable-model-invocation skill must be denied on the model path"
        );
        assert_eq!(
            resolver.resolve_by_name("nope", ""),
            Err(SkillLookupError::NotFound)
        );
        let (body, name) = resolver.resolve_by_name("commit", "").unwrap();
        assert_eq!(name, "commit");
        assert!(body.contains("commit body"));
    }

    #[test]
    fn skill_summaries_cover_all_skills_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "zeta", "z body", true);
        make_skill_dir(tmp.path(), "internal-helper", "h body", false);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        let summaries = resolver.skill_summaries();
        let names: Vec<&str> = summaries.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["internal-helper", "zeta"],
            "summaries sorted by name"
        );
        // Non-user-invocable skills must also be listed (only the slash path
        // filters; the picker wants everything).
        assert!(names.contains(&"internal-helper"));
        assert!(summaries.iter().all(|(_, d)| !d.is_empty()));
    }

    #[test]
    #[cfg(unix)]
    fn unreadable_dir_does_not_block_later_dirs() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let bad_dir = tmp.path().join("bad_skills");
        let good_dir = tmp.path().join("good_skills");
        fs::create_dir_all(bad_dir.join("broken")).unwrap();
        fs::create_dir_all(bad_dir.join("locked")).unwrap();
        // Put a real skill in the good directory
        let sd = good_dir.join("commit");
        fs::create_dir_all(&sd).unwrap();
        fs::write(
            sd.join("SKILL.md"),
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nlater dir body",
        )
        .unwrap();

        // Put a skill dir that will end up locked under bad_dir, then chmod
        // 000 bad_dir itself
        fs::write(
            bad_dir.join("broken").join("SKILL.md"),
            "---\nname: broken\ndescription: d\nuser-invocable: true\n---\n\nshould be unreadable",
        )
        .unwrap();
        fs::set_permissions(&bad_dir, fs::Permissions::from_mode(0o000)).unwrap();

        // Regardless of the scan outcome, restore permissions first so tempdir
        // cleanup leaves nothing behind.
        let result = std::panic::catch_unwind(|| {
            SkillResolver::from_dirs(&[bad_dir.clone(), good_dir.clone()])
        });
        fs::set_permissions(&bad_dir, fs::Permissions::from_mode(0o755)).unwrap();

        let resolver = result.expect("from_dirs must not panic on unreadable dir");
        assert_eq!(
            resolver.len(),
            1,
            "later readable dir must still be scanned"
        );
        assert!(resolver.resolve("/commit").is_some());
    }

    // ── scope resolution (skill-lifetime v3) ──

    /// Write a skill with arbitrary extra frontmatter lines.
    fn make_skill_with_frontmatter(tmp: &Path, name: &str, extra: &str) {
        let skill_dir = tmp.join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        let content = format!(
            "---\nname: {name}\ndescription: d\nuser-invocable: true\n{extra}\n---\n\nbody of {name}"
        );
        fs::write(skill_dir.join("SKILL.md"), content).unwrap();
    }

    #[test]
    fn scope_defaults_to_session_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_with_frontmatter(tmp.path(), "plain", "");

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        let resolved = resolver.resolve_with_meta("/plain").unwrap();
        assert_eq!(resolved.scope, SkillScope::Session);
        assert_eq!(resolved.name, "plain");
        assert!(resolved.body.contains("body of plain"));
    }

    #[test]
    fn scope_turn_is_parsed() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_with_frontmatter(tmp.path(), "oneshot", "scope: turn");

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert_eq!(
            resolver.resolve_with_meta("/oneshot").unwrap().scope,
            SkillScope::Turn
        );
    }

    #[test]
    fn scope_session_explicit_and_unknown_fall_back_to_session() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_with_frontmatter(tmp.path(), "explicit", "scope: session");
        // A typo must never fail the whole SKILL.md load — falls back to session.
        make_skill_with_frontmatter(tmp.path(), "typo", "scope: sesion");

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert_eq!(resolver.len(), 2, "typo in scope must not fail the load");
        assert_eq!(
            resolver.resolve_with_meta("/explicit").unwrap().scope,
            SkillScope::Session
        );
        assert_eq!(
            resolver.resolve_with_meta("/typo").unwrap().scope,
            SkillScope::Session
        );
    }

    #[test]
    fn resolve_delegates_to_resolve_with_meta_body() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_with_frontmatter(tmp.path(), "commit", "scope: turn");

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        let via_meta = resolver.resolve_with_meta("/commit").unwrap();
        assert_eq!(Some(via_meta.body), resolver.resolve("/commit"));
    }
}
