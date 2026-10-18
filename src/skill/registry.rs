use std::collections::{HashMap, HashSet};
use std::path::Path;
#[cfg(feature = "hot-reload")]
use std::path::PathBuf;
use std::sync::Arc;

use agent_base::{AgentError, AgentResult, UpdatePlanArgs};
use tokio::sync::RwLock;

use super::{Skill, SkillParam};

/// Lightweight summary of a skill — returned by `list()`.
/// Excludes `detailed_description` and `plan_steps` to save tokens.
#[derive(Clone, serde::Serialize)]
pub struct SkillSummary {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub param_defs: Vec<SkillParam>,
    pub has_plan: bool,
    pub version: String,
    pub category: String,
    pub author: String,
}

/// Runtime registry for skills. Supports dynamic registration,
/// runtime enable/disable, and optional hot-reloading.
///
/// Unlike `AgentBuilder` which only registers at build time, the registry
/// allows skills to be added, removed, enabled, or disabled at runtime
/// without restarting the agent.
pub struct SkillRegistry {
    skills: RwLock<Vec<Arc<dyn Skill>>>,
    /// Names of disabled skills. A skill is enabled if its name is NOT in this set.
    disabled: RwLock<HashSet<String>>,
    /// Hot-reload watcher handle (feature-gated).
    #[cfg(feature = "hot-reload")]
    watcher_handle: RwLock<Option<notify::RecommendedWatcher>>,
    /// Directories being watched for hot-reload.
    #[cfg(feature = "hot-reload")]
    watched_dirs: RwLock<Vec<PathBuf>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: RwLock::new(Vec::new()),
            disabled: RwLock::new(HashSet::new()),
            #[cfg(feature = "hot-reload")]
            watcher_handle: RwLock::new(None),
            #[cfg(feature = "hot-reload")]
            watched_dirs: RwLock::new(Vec::new()),
        }
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillRegistry {
    /// Register a skill. If a skill with the same name exists, it's replaced.
    pub async fn register(&self, skill: Arc<dyn Skill>) {
        let mut skills = self.skills.write().await;
        skills.retain(|s| s.name() != skill.name());
        skills.push(skill);
    }

    /// List all enabled skills as summaries (brief descriptions only).
    /// Disabled skills are excluded.
    pub async fn list(&self) -> Vec<SkillSummary> {
        let skills = self.skills.read().await;
        let disabled = self.disabled.read().await;
        skills
            .iter()
            .filter(|s| !disabled.contains(s.name()))
            .map(|s| SkillSummary {
                name: s.name().to_string(),
                description: s.brief_description(),
                tags: s.tags().iter().map(|t| t.to_string()).collect(),
                param_defs: s.parameters().to_vec(),
                has_plan: s.plan_steps(&HashMap::new()).is_some(),
                version: s.version().to_string(),
                category: s.category().to_string(),
                author: s.author().to_string(),
            })
            .collect()
    }

    /// Get an enabled skill by name (full info including detailed_description).
    /// Returns `None` if the skill doesn't exist or is disabled.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Skill>> {
        if !self.is_enabled(name).await {
            return None;
        }
        let skills = self.skills.read().await;
        skills.iter().find(|s| s.name() == name).cloned()
    }

    /// Apply a template skill: substitute parameters → generate plan args.
    /// Returns `None` if the skill doesn't exist or is not a template skill.
    pub async fn apply(
        &self,
        name: &str,
        params: &HashMap<String, String>,
    ) -> AgentResult<Option<UpdatePlanArgs>> {
        let skill = match self.get(name).await {
            Some(s) => s,
            None => return Ok(None),
        };

        // Validate required parameters
        for param in skill.parameters() {
            if param.required && !params.contains_key(&param.name) {
                return Err(AgentError::internal(format!(
                    "Skill '{}': missing required parameter '{}'",
                    name, param.name
                )));
            }
        }

        // Generate plan items via template expansion
        let steps = match skill.plan_steps(params) {
            Some(s) => s,
            None => return Ok(None), // Not a template skill
        };

        if steps.is_empty() {
            return Err(AgentError::internal(format!(
                "Skill '{}' generated empty plan steps",
                name
            )));
        }

        let objective = format!("{}: {}", name, skill.brief_description());

        let plan = UpdatePlanArgs {
            objective: Some(objective),
            explanation: Some(format!("generated from skill template '{}'", name)),
            plan: steps,
        };

        Ok(Some(plan))
    }

    /// Remove a skill by name.
    pub async fn remove(&self, name: &str) {
        let mut skills = self.skills.write().await;
        skills.retain(|s| s.name() != name);
        // Also clean up disabled tracking
        self.disabled.write().await.remove(name);
    }

    // ── Runtime enable/disable ──

    /// Enable a previously disabled skill.
    ///
    /// Returns `true` if the skill was disabled and is now enabled,
    /// `false` if it was already enabled or doesn't exist.
    pub async fn enable(&self, name: &str) -> bool {
        self.disabled.write().await.remove(name)
    }

    /// Disable a skill at runtime without removing it.
    ///
    /// A disabled skill is hidden from `list()`, `get()`, and `apply()`,
    /// but its tools remain registered (they are managed separately by
    /// the ToolRegistry).
    ///
    /// Returns `true` if the skill was enabled and is now disabled,
    /// `false` if it was already disabled or doesn't exist.
    pub async fn disable(&self, name: &str) -> bool {
        // Only disable if the skill actually exists
        let skills = self.skills.read().await;
        if skills.iter().any(|s| s.name() == name) {
            self.disabled.write().await.insert(name.to_string());
            true
        } else {
            false
        }
    }

    /// Check whether a skill is currently enabled.
    pub async fn is_enabled(&self, name: &str) -> bool {
        !self.disabled.read().await.contains(name)
    }

    // ── Directory loading ──

    /// Load skills from a directory containing skill subdirectories.
    ///
    /// Each subdirectory with a `SKILL.md` file is loaded as a skill.
    /// Skills with names already in the registry are replaced.
    ///
    /// Returns the number of skills loaded.
    ///
    /// Requires the `prompt_skill` feature.
    #[cfg(feature = "prompt_skill")]
    pub async fn load_from_dir(&self, path: &Path) -> AgentResult<usize> {
        let loaded = super::prompt_skill::PromptSkill::scan_dir(path)
            .map_err(|e| AgentError::internal(format!("Failed to scan skills dir: {}", e)))?;

        let count = loaded.len();
        for skill in loaded {
            self.register(Arc::new(skill)).await;
        }
        Ok(count)
    }

    /// Number of registered skills (including disabled).
    pub async fn count(&self) -> usize {
        self.skills.read().await.len()
    }

    /// Number of enabled skills.
    pub async fn enabled_count(&self) -> usize {
        let skills = self.skills.read().await;
        let disabled = self.disabled.read().await;
        skills
            .iter()
            .filter(|s| !disabled.contains(s.name()))
            .count()
    }

    /// Get all enabled skills as full `Arc<dyn Skill>` for use with
    /// `SkillDetailTool` and `SkillPrompter`.
    pub async fn all_skills(&self) -> Vec<Arc<dyn Skill>> {
        let skills = self.skills.read().await;
        let disabled = self.disabled.read().await;
        skills
            .iter()
            .filter(|s| !disabled.contains(s.name()))
            .cloned()
            .collect()
    }

    /// Get all registered skills including disabled ones.
    /// Used internally for management operations.
    pub async fn all_skills_including_disabled(&self) -> Vec<Arc<dyn Skill>> {
        self.skills.read().await.clone()
    }

    // ── Hot-reload (feature-gated) ──

    /// Start watching directories for skill changes and auto-reload.
    ///
    /// When a `SKILL.md` file is created, modified, or deleted in any watched
    /// directory, the corresponding skill is reloaded or removed.
    ///
    /// Spawns a background task that listens for file-system events and
    /// reloads skills from all watched directories on any change.
    #[cfg(feature = "hot-reload")]
    pub async fn start_watcher(&self, dirs: Vec<PathBuf>) -> AgentResult<()> {
        use notify::{EventKind, RecursiveMode, Watcher};

        // Stop existing watcher if any
        self.stop_watcher().await;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let mut watcher =
            notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    // Only react to file modifications that could affect SKILL.md
                    if matches!(
                        event.kind,
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                    ) {
                        let _ = tx.send(event);
                    }
                }
            })
            .map_err(|e| AgentError::internal(format!("Failed to create file watcher: {}", e)))?;

        for dir in &dirs {
            if dir.is_dir() {
                watcher.watch(dir, RecursiveMode::Recursive).map_err(|e| {
                    AgentError::internal(format!("Failed to watch {}: {}", dir.display(), e))
                })?;
            }
        }

        *self.watcher_handle.write().await = Some(watcher);
        *self.watched_dirs.write().await = dirs.clone();

        // Spawn the event loop that consumes file-system events and reloads skills.
        // The loop exits when the watcher is dropped (stop_watcher), which closes
        // the channel and causes `rx.recv()` to return `None`.
        //
        // SAFETY: We cast the registry pointer to `usize` for `Send`-ability.
        // The registry is guaranteed to outlive the spawned task: `stop_watcher()`
        // drops the watcher (closing the channel and ending the loop) before the
        // registry is dropped, and `Drop` for the registry also drops the watcher.
        let registry_addr = self as *const SkillRegistry as usize;
        let reload_dirs = dirs.clone();

        tokio::spawn(async move {
            // Debounce: collect events over a short window, then reload once
            while rx.recv().await.is_some() {
                // Drain any additional events that arrived in quick succession
                while rx.try_recv().is_ok() {}
                // Reload from all watched directories
                // SAFETY: registry outlives this task (see above).
                let registry = unsafe { &*(registry_addr as *const SkillRegistry) };
                for dir in &reload_dirs {
                    if dir.is_dir()
                        && let Err(e) = registry.load_from_dir(dir).await
                    {
                        tracing::warn!(dir = %dir.display(), error = %e, "hot-reload: failed to reload skills");
                    }
                }
                tracing::debug!("hot-reload: skills reloaded from watched dirs");
            }
            tracing::info!("hot-reload watcher event loop stopped");
        });

        tracing::info!(dirs = ?dirs, "started skill hot-reload watcher");
        Ok(())
    }

    /// Stop the hot-reload watcher if one is active.
    ///
    /// Drops the watcher handle, which closes the internal channel and
    /// causes the background event loop to exit.
    #[cfg(feature = "hot-reload")]
    pub async fn stop_watcher(&self) {
        *self.watcher_handle.write().await = None;
        self.watched_dirs.write().await.clear();
    }

    /// Check whether hot-reload is currently active.
    #[cfg(feature = "hot-reload")]
    pub async fn is_watching(&self) -> bool {
        self.watcher_handle.read().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::Skill;
    use std::sync::Arc;

    struct TestSkill {
        name: &'static str,
        desc: &'static str,
    }

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            self.name
        }
        fn brief_description(&self) -> String {
            self.desc.to_string()
        }
        fn detailed_description(&self) -> String {
            format!("Detailed: {}", self.desc)
        }
        fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
            vec![]
        }
    }

    #[tokio::test]
    async fn test_register_and_list() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "test-skill",
                desc: "A test skill",
            }))
            .await;

        let list = registry.list().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "test-skill");
        assert_eq!(list[0].description, "A test skill");
        assert!(!list[0].has_plan);
    }

    #[tokio::test]
    async fn test_get() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "test-skill",
                desc: "A test skill",
            }))
            .await;

        let skill = registry.get("test-skill").await;
        assert!(skill.is_some());
        assert_eq!(skill.unwrap().name(), "test-skill");

        assert!(registry.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_register_replace() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "test-skill",
                desc: "v1",
            }))
            .await;
        registry
            .register(Arc::new(TestSkill {
                name: "test-skill",
                desc: "v2",
            }))
            .await;

        let list = registry.list().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].description, "v2");
    }

    #[tokio::test]
    async fn test_remove() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "test-skill",
                desc: "test",
            }))
            .await;
        assert_eq!(registry.count().await, 1);
        registry.remove("test-skill").await;
        assert_eq!(registry.count().await, 0);
    }

    #[tokio::test]
    async fn test_apply_nonexistent() {
        let registry = SkillRegistry::new();
        let result = registry
            .apply("nonexistent", &HashMap::new())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_apply_knowledge_skill_returns_none() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "knowledge-skill",
                desc: "A knowledge skill",
            }))
            .await;
        let result = registry
            .apply("knowledge-skill", &HashMap::new())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    // ── enable/disable tests ──

    #[tokio::test]
    async fn test_disable_and_enable() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "toggle-skill",
                desc: "Toggle test",
            }))
            .await;

        assert!(registry.is_enabled("toggle-skill").await);

        // Disable
        assert!(registry.disable("toggle-skill").await);
        assert!(!registry.is_enabled("toggle-skill").await);

        // list() excludes disabled
        assert!(registry.list().await.is_empty());

        // get() returns None for disabled
        assert!(registry.get("toggle-skill").await.is_none());

        // Enable
        assert!(registry.enable("toggle-skill").await);
        assert!(registry.is_enabled("toggle-skill").await);
        assert_eq!(registry.list().await.len(), 1);
    }

    #[tokio::test]
    async fn test_disable_nonexistent_returns_false() {
        let registry = SkillRegistry::new();
        assert!(!registry.disable("no-such-skill").await);
    }

    #[tokio::test]
    async fn test_enable_nonexistent_returns_false() {
        let registry = SkillRegistry::new();
        assert!(!registry.enable("no-such-skill").await);
    }

    #[tokio::test]
    async fn test_enabled_count() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "skill-a",
                desc: "A",
            }))
            .await;
        registry
            .register(Arc::new(TestSkill {
                name: "skill-b",
                desc: "B",
            }))
            .await;

        assert_eq!(registry.enabled_count().await, 2);
        assert_eq!(registry.count().await, 2);

        registry.disable("skill-a").await;
        assert_eq!(registry.enabled_count().await, 1);
        assert_eq!(registry.count().await, 2); // total unchanged
    }

    #[tokio::test]
    async fn test_remove_cleans_up_disabled() {
        let registry = SkillRegistry::new();
        registry
            .register(Arc::new(TestSkill {
                name: "rm-skill",
                desc: "test",
            }))
            .await;
        registry.disable("rm-skill").await;
        registry.remove("rm-skill").await;

        // After remove + re-register, skill should be enabled by default
        registry
            .register(Arc::new(TestSkill {
                name: "rm-skill",
                desc: "test2",
            }))
            .await;
        assert!(registry.is_enabled("rm-skill").await);
    }

    // ── Hot-reload integration test (Phase 4) ──
    // This test verifies the end-to-end flow: file change → notify event →
    // spawned task consumes event → load_from_dir → skill appears in registry.

    #[cfg(feature = "hot-reload")]
    #[tokio::test]
    async fn test_hot_reload_detects_new_skill() {
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        // Use Arc so the raw pointer in the spawned task stays valid
        // even if the test function's stack frame is unwound early.
        let registry = Arc::new(SkillRegistry::new());

        // Pre-populate with one skill
        let init_dir = dir.path().join("init-skill");
        std::fs::create_dir(&init_dir).unwrap();
        std::fs::write(
            init_dir.join("SKILL.md"),
            "---\nname: init-skill\ndescription: Initial\n---\nBody",
        )
        .unwrap();
        registry.load_from_dir(dir.path()).await.unwrap();
        assert_eq!(registry.count().await, 1);

        // Start the hot-reload watcher
        registry
            .start_watcher(vec![dir.path().to_path_buf()])
            .await
            .unwrap();
        assert!(registry.is_watching().await);

        // Create a new skill directory — the watcher should pick this up
        let new_dir = dir.path().join("new-skill");
        std::fs::create_dir(&new_dir).unwrap();
        std::fs::write(
            new_dir.join("SKILL.md"),
            "---\nname: new-skill\ndescription: New\nauthor: test\n---\nBody",
        )
        .unwrap();

        // Wait for the hot-reload to detect and load the new skill.
        // File-system events are asynchronous, so we retry with a deadline.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut found = false;
        while tokio::time::Instant::now() < deadline {
            if registry.get("new-skill").await.is_some() {
                found = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Clean up before asserting (stops the spawned task)
        registry.stop_watcher().await;
        // Give the spawned task a moment to observe the closed channel
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(found, "hot-reload did not detect new skill within 5s");
        assert!(!registry.is_watching().await);
    }

    #[cfg(feature = "hot-reload")]
    #[tokio::test]
    async fn test_hot_reload_updates_existing_skill() {
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(SkillRegistry::new());

        // Create initial skill
        let skill_dir = dir.path().join("update-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: update-skill\ndescription: v1\n---\nBody",
        )
        .unwrap();
        registry.load_from_dir(dir.path()).await.unwrap();
        assert_eq!(registry.count().await, 1);

        // Verify initial description
        let skill = registry.get("update-skill").await.unwrap();
        assert_eq!(skill.brief_description(), "v1");

        // Start watcher
        registry
            .start_watcher(vec![dir.path().to_path_buf()])
            .await
            .unwrap();

        // Modify the SKILL.md file
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: update-skill\ndescription: v2-updated\nauthor: test\n---\nUpdated body",
        )
        .unwrap();

        // Wait for the reload with retry
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut updated = false;
        while tokio::time::Instant::now() < deadline {
            if let Some(s) = registry.get("update-skill").await
                && s.brief_description() == "v2-updated"
            {
                updated = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        registry.stop_watcher().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(updated, "hot-reload did not update skill within 5s");
    }
}
