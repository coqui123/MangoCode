/// Plugin registry — holds all loaded plugins and provides queries.
///
/// Tracks which plugins are enabled (global + per-project lists).
use crate::hooks::{register_plugin_hooks, HookRegistry};
use crate::manifest::normalize_manifest_relative_path;
use crate::plugin::{LoadedPlugin, PluginCommandDef, PluginError, ReloadDiff};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// PluginRegistry
// ---------------------------------------------------------------------------

/// Central store for all discovered plugins in a session.
///
/// `enabled()` returns only enabled plugins; `all()` returns every plugin
/// including disabled ones.
#[derive(Debug, Default, Clone)]
pub struct PluginRegistry {
    /// All plugins keyed by name.
    plugins: HashMap<String, LoadedPlugin>,
    /// Names of plugins that are currently enabled.
    enabled_names: std::collections::HashSet<String>,
    /// Accumulated load errors.
    pub errors: Vec<PluginError>,
}

impl PluginRegistry {
    // ---- Construction & population ----------------------------------------

    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a loaded plugin.  Emits a duplicate error if
    /// a different path already holds a plugin with the same name.
    pub fn insert(&mut self, plugin: LoadedPlugin) {
        let name = plugin.name.clone();
        let enabled = plugin.enabled;

        if let Some(existing) = self.plugins.get(&name) {
            if existing.path != plugin.path {
                self.errors.push(PluginError::DuplicateName {
                    name: name.clone(),
                    first: existing.path.to_string_lossy().into_owned(),
                    second: plugin.path.to_string_lossy().into_owned(),
                });
                // Keep the first one (first-wins).
                return;
            }
        }

        self.plugins.insert(name.clone(), plugin);
        if enabled {
            self.enabled_names.insert(name);
        }
    }

    /// Append multiple plugins at once, updating errors inline.
    pub fn extend(&mut self, plugins: Vec<LoadedPlugin>, errors: Vec<PluginError>) {
        self.errors.extend(errors);
        for p in plugins {
            self.insert(p);
        }
    }

    // ---- Queries ----------------------------------------------------------

    /// All loaded plugins (enabled + disabled).
    pub fn all(&self) -> Vec<&LoadedPlugin> {
        self.plugins.values().collect()
    }

    /// Only the enabled plugins.
    pub fn enabled(&self) -> Vec<&LoadedPlugin> {
        self.plugins
            .values()
            .filter(|p| self.enabled_names.contains(&p.name))
            .collect()
    }

    /// Look up a plugin by name.
    pub fn get(&self, name: &str) -> Option<&LoadedPlugin> {
        self.plugins.get(name)
    }

    /// Whether a plugin is enabled.
    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled_names.contains(name)
    }

    // ---- Enable / disable -------------------------------------------------

    /// Enable a plugin by name.  Returns `false` if the plugin is not loaded.
    pub fn enable(&mut self, name: &str) -> bool {
        if self.plugins.contains_key(name) {
            self.enabled_names.insert(name.to_string());
            if let Some(p) = self.plugins.get_mut(name) {
                p.enabled = true;
            }
            true
        } else {
            false
        }
    }

    /// Disable a plugin by name.  Returns `false` if the plugin is not loaded.
    pub fn disable(&mut self, name: &str) -> bool {
        if self.plugins.contains_key(name) {
            self.enabled_names.remove(name);
            if let Some(p) = self.plugins.get_mut(name) {
                p.enabled = false;
            }
            true
        } else {
            false
        }
    }

    // ---- Derived collections from enabled plugins -------------------------

    /// Collect all `PluginCommandDef` items from enabled plugins.
    pub fn all_command_defs(&self) -> Vec<PluginCommandDef> {
        let mut defs: Vec<PluginCommandDef> = Vec::new();
        for plugin in self.enabled() {
            let mut plugin_defs = crate::loader::collect_command_defs(plugin);
            // Keep the registry's loaded plugin identity authoritative.
            for d in &mut plugin_defs {
                d.plugin_source_id = plugin.source_id.clone();
            }
            defs.extend(plugin_defs);
        }
        defs
    }

    /// Collect skill search roots contributed by enabled plugins.
    ///
    /// Manifest entries may point at a root directory, a single skill
    /// directory (`SKILL.md` inside), or a flat `.md` skill file. Consumers like
    /// `discover_skills` and `SkillTool` search roots, so concrete skill paths
    /// are normalized to their parent directory.
    pub fn all_skill_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        let mut seen = HashSet::new();
        for plugin in self.enabled() {
            if let Some(ref p) = plugin.skills_path {
                push_unique_path(&mut paths, &mut seen, p.clone());
            }
            for rel_path in &plugin.manifest.skills {
                if let Ok(rel_path) = normalize_manifest_relative_path(rel_path) {
                    push_skill_search_path(&mut paths, &mut seen, plugin.path.join(rel_path));
                }
            }
        }
        paths
    }

    /// Collect markdown agent definition files contributed by enabled plugins.
    pub fn all_agent_files(&self) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        let mut seen = HashSet::new();
        for plugin in self.enabled() {
            if let Some(ref p) = plugin.agents_path {
                collect_agent_files_from_path(p, &mut files, &mut seen);
            }
            for rel_path in &plugin.manifest.agents {
                if let Ok(rel_path) = normalize_manifest_relative_path(rel_path) {
                    collect_agent_files_from_path(
                        &plugin.path.join(rel_path),
                        &mut files,
                        &mut seen,
                    );
                }
            }
        }
        files
    }

    /// Collect paths to all `output-styles/` directories contributed by enabled plugins.
    pub fn all_output_style_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for plugin in self.enabled() {
            if let Some(ref p) = plugin.output_styles_path {
                paths.push(p.clone());
            }
        }
        paths
    }

    /// Collect all output styles contributed by enabled plugins.
    pub fn all_output_styles(&self) -> Vec<mangocode_core::output_styles::OutputStyleDef> {
        let mut styles = Vec::new();
        let mut seen = HashSet::new();
        let mut plugins = self.enabled();
        plugins.sort_by(|a, b| a.name.cmp(&b.name));

        for plugin in plugins {
            if let Some(ref path) = plugin.output_styles_path {
                collect_output_styles_from_path(path, &mut styles, &mut seen);
            }

            for rel_path in &plugin.manifest.output_styles {
                if let Ok(rel_path) = normalize_manifest_relative_path(rel_path) {
                    let path = plugin.path.join(rel_path);
                    collect_output_styles_from_path(&path, &mut styles, &mut seen);
                }
            }
        }

        styles
    }

    /// Build the `HookRegistry` from all enabled plugins.
    pub fn build_hook_registry(&self) -> HookRegistry {
        let mut registry: HookRegistry = HashMap::new();
        for plugin in self.enabled() {
            if let Some(ref hooks_config) = plugin.hooks_config {
                register_plugin_hooks(
                    hooks_config,
                    &plugin.path.to_string_lossy(),
                    &plugin.name,
                    &plugin.source_id,
                    &mut registry,
                );
            }
        }
        registry
    }

    /// Collect all MCP server configs contributed by enabled plugins.
    pub fn all_mcp_servers(&self) -> Vec<mangocode_core::config::McpServerConfig> {
        let mut servers: Vec<mangocode_core::config::McpServerConfig> = Vec::new();
        for plugin in self.enabled() {
            for mcp in &plugin.manifest.mcp_servers {
                servers.push(mangocode_core::config::McpServerConfig {
                    name: mcp.name.clone(),
                    command: mcp.command.clone(),
                    args: mcp.args.clone(),
                    env: mcp.env.clone(),
                    url: mcp.url.clone(),
                    headers: std::collections::HashMap::new(),
                    pipedream: None,
                    server_type: mcp.server_type.clone(),
                });
            }
        }
        servers
    }

    /// Collect all LSP server configs contributed by enabled plugins.
    pub fn all_lsp_servers(&self) -> Vec<crate::manifest::PluginLspServer> {
        let mut servers: Vec<crate::manifest::PluginLspServer> = Vec::new();
        for plugin in self.enabled() {
            for lsp in &plugin.manifest.lsp_servers {
                servers.push(lsp.clone());
            }
        }
        servers
    }

    // ---- Statistics -------------------------------------------------------

    /// Total number of plugins (enabled + disabled).
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Number of enabled plugins.
    pub fn enabled_count(&self) -> usize {
        self.enabled_names.len()
    }

    /// Number of load errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    // ---- Reload diff ------------------------------------------------------

    /// Compare this registry against `old` and produce a diff report.
    pub fn diff_against(&self, old: &PluginRegistry) -> ReloadDiff {
        let old_names: std::collections::HashSet<&str> =
            old.plugins.keys().map(|s| s.as_str()).collect();
        let new_names: std::collections::HashSet<&str> =
            self.plugins.keys().map(|s| s.as_str()).collect();

        let added: Vec<String> = new_names
            .difference(&old_names)
            .map(|&s| s.to_string())
            .collect();
        let removed: Vec<String> = old_names
            .difference(&new_names)
            .map(|&s| s.to_string())
            .collect();
        let updated: Vec<String> = new_names
            .intersection(&old_names)
            .filter(|&&name| {
                let new_ver = self
                    .plugins
                    .get(name)
                    .and_then(|p| p.manifest.version.as_deref());
                let old_ver = old
                    .plugins
                    .get(name)
                    .and_then(|p| p.manifest.version.as_deref());
                new_ver != old_ver
            })
            .map(|&s| s.to_string())
            .collect();

        ReloadDiff {
            added,
            removed,
            updated,
            error_count: self.errors.len(),
        }
    }
}

fn push_unique_path(
    paths: &mut Vec<std::path::PathBuf>,
    seen: &mut HashSet<std::path::PathBuf>,
    path: std::path::PathBuf,
) {
    if seen.insert(path.clone()) {
        paths.push(path);
    }
}

fn push_skill_search_path(
    paths: &mut Vec<std::path::PathBuf>,
    seen: &mut HashSet<std::path::PathBuf>,
    path: std::path::PathBuf,
) {
    let search_path =
        if is_markdown_file(&path) || (path.is_dir() && skill_index_file(&path).is_some()) {
            path.parent().map(std::path::Path::to_path_buf)
        } else {
            Some(path)
        };

    if let Some(path) = search_path {
        push_unique_path(paths, seen, path);
    }
}

fn is_markdown_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
}

fn skill_index_file(dir: &Path) -> Option<std::path::PathBuf> {
    for filename in ["SKILL.md", "skill.md"] {
        let candidate = dir.join(filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                path = %dir.display(),
                error = %err,
                "failed to read plugin skill directory"
            );
            return None;
        }
    };

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!(
                    path = %dir.display(),
                    error = %err,
                    "failed to read plugin skill directory entry"
                );
                continue;
            }
        };
        let path = entry.path();
        let is_file = match entry.file_type() {
            Ok(file_type) => file_type.is_file(),
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to inspect plugin skill path"
                );
                continue;
            }
        };
        if is_file
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("skill.md"))
        {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates.into_iter().next()
}

fn collect_output_styles_from_path(
    path: &Path,
    styles: &mut Vec<mangocode_core::output_styles::OutputStyleDef>,
    seen: &mut HashSet<String>,
) {
    let loaded = if path.is_dir() {
        mangocode_core::output_styles::load_output_styles_dir(path)
    } else if path.is_file() {
        mangocode_core::output_styles::load_output_style_file(path)
            .into_iter()
            .collect()
    } else {
        Vec::new()
    };

    for mut style in loaded {
        style.name = style.name.trim().to_string();
        if !style.name.is_empty() && seen.insert(style.name.to_lowercase()) {
            styles.push(style);
        }
    }
}

fn collect_agent_files_from_path(
    path: &Path,
    files: &mut Vec<std::path::PathBuf>,
    seen: &mut HashSet<std::path::PathBuf>,
) {
    if path.is_file() {
        if is_markdown_file(path) {
            push_unique_path(files, seen, path.to_path_buf());
        }
        return;
    }

    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "failed to read plugin agent directory"
            );
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to read plugin agent directory entry"
                );
                continue;
            }
        };
        let candidate = entry.path();
        let is_file = match entry.file_type() {
            Ok(file_type) => file_type.is_file(),
            Err(err) => {
                tracing::warn!(
                    path = %candidate.display(),
                    error = %err,
                    "failed to inspect plugin agent path"
                );
                continue;
            }
        };
        if is_file && is_markdown_file(&candidate) {
            push_unique_path(files, seen, candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::PluginManifest;
    use crate::plugin::PluginSource;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn make_plugin(name: &str) -> LoadedPlugin {
        LoadedPlugin {
            name: name.to_string(),
            path: PathBuf::from(format!("/tmp/{}", name)),
            source: PluginSource::User,
            source_id: format!("{}@user", name),
            manifest: PluginManifest {
                name: name.to_string(),
                ..Default::default()
            },
            enabled: true,
            commands_path: None,
            agents_path: None,
            skills_path: None,
            output_styles_path: None,
            hooks_config: None,
        }
    }

    #[test]
    fn enable_disable() {
        let mut reg = PluginRegistry::new();
        reg.insert(make_plugin("alpha"));
        assert!(reg.is_enabled("alpha"));

        reg.disable("alpha");
        assert!(!reg.is_enabled("alpha"));
        assert_eq!(reg.enabled().len(), 0);

        reg.enable("alpha");
        assert!(reg.is_enabled("alpha"));
        assert_eq!(reg.enabled().len(), 1);
    }

    #[test]
    fn duplicate_name_kept_first() {
        let mut reg = PluginRegistry::new();
        reg.insert(make_plugin("beta"));
        let mut dup = make_plugin("beta");
        dup.path = PathBuf::from("/tmp/beta2");
        reg.insert(dup);
        assert_eq!(reg.plugin_count(), 1);
        assert_eq!(reg.error_count(), 1);
    }

    #[test]
    fn all_command_defs_uses_enabled_plugins_and_patches_source_id() {
        let tmp = TempDir::new().unwrap();
        let enabled_dir = tmp.path().join("enabled-plugin");
        let enabled_commands_dir = enabled_dir.join("commands");
        std::fs::create_dir_all(enabled_commands_dir.join("build")).unwrap();
        std::fs::write(
            enabled_commands_dir.join("build").join("deploy.md"),
            "# Deploy\nDeploy the current build.",
        )
        .unwrap();

        let disabled_dir = tmp.path().join("disabled-plugin");
        let disabled_commands_dir = disabled_dir.join("commands");
        std::fs::create_dir_all(&disabled_commands_dir).unwrap();
        std::fs::write(
            disabled_commands_dir.join("hidden.md"),
            "# Hidden\nThis command should not load.",
        )
        .unwrap();

        let mut enabled = make_plugin("enabled-plugin");
        enabled.path = enabled_dir;
        enabled.source_id = "enabled-plugin@project".to_string();
        enabled.commands_path = Some(enabled_commands_dir);
        enabled.manifest.capabilities = Some(vec!["read_files".to_string()]);

        let mut disabled = make_plugin("disabled-plugin");
        disabled.path = disabled_dir;
        disabled.source_id = "disabled-plugin@project".to_string();
        disabled.enabled = false;
        disabled.commands_path = Some(disabled_commands_dir);

        let mut reg = PluginRegistry::new();
        reg.insert(enabled);
        reg.insert(disabled);

        let defs = reg.all_command_defs();

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "enabled-plugin:build:deploy");
        assert_eq!(defs[0].plugin_source_id, "enabled-plugin@project");
        assert_eq!(
            defs[0].plugin_capabilities.as_deref(),
            Some(&["read_files".to_string()][..])
        );
        assert!(!defs.iter().any(|def| def.name.contains("hidden")));
    }

    #[test]
    fn all_output_styles_collects_plugin_dirs_and_manifest_entries() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let default_styles_dir = plugin_dir.join("output-styles");
        let extra_styles_dir = plugin_dir.join("extra-styles");
        std::fs::create_dir_all(&default_styles_dir).unwrap();
        std::fs::create_dir_all(&extra_styles_dir).unwrap();
        std::fs::write(
            default_styles_dir.join("terse.md"),
            "# Terse\nShort answers.\n\nAnswer briefly.",
        )
        .unwrap();
        std::fs::write(
            extra_styles_dir.join("friendly.md"),
            "# Friendly\nWarm answers.\n\nUse a warm tone.",
        )
        .unwrap();
        std::fs::write(
            extra_styles_dir.join("Friendly.md"),
            "# Friendly Duplicate\nDuplicate style.\n\nDo not load twice.",
        )
        .unwrap();
        std::fs::write(
            plugin_dir.join("direct.json"),
            r#"{"name":"direct","label":"Direct","description":"Direct answers.","prompt":"Be direct."}"#,
        )
        .unwrap();

        let mut plugin = make_plugin("style-plugin");
        plugin.path = plugin_dir;
        plugin.output_styles_path = Some(default_styles_dir);
        plugin.manifest.output_styles = vec![
            "./extra-styles".to_string(),
            "./direct.json".to_string(),
            "./ignore.txt".to_string(),
        ];

        let mut reg = PluginRegistry::new();
        reg.insert(plugin);

        let names: Vec<String> = reg
            .all_output_styles()
            .into_iter()
            .map(|style| style.name)
            .collect();
        assert!(names.contains(&"terse".to_string()));
        assert!(names.contains(&"friendly".to_string()));
        assert!(names.contains(&"direct".to_string()));
        assert!(!names.contains(&"ignore".to_string()));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.eq_ignore_ascii_case("friendly"))
                .count(),
            1
        );
    }

    #[test]
    fn all_output_styles_skips_manifest_paths_outside_plugin_root() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(
            tmp.path().join("outside.md"),
            "# Outside\nShould not load.\n\nThis is outside the plugin root.",
        )
        .unwrap();

        let mut plugin = make_plugin("style-plugin");
        plugin.path = plugin_dir;
        plugin.manifest.output_styles = vec!["../outside.md".to_string()];

        let mut reg = PluginRegistry::new();
        reg.insert(plugin);

        assert!(reg.all_output_styles().is_empty());
    }

    #[test]
    fn manifest_agent_and_skill_paths_are_exposed() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("agent-plugin");
        let extra_agents = plugin_dir.join("extra-agents");
        let extra_skills = plugin_dir.join("extra-skills");
        let direct_skill = extra_skills.join("helper");
        std::fs::create_dir_all(&extra_agents).unwrap();
        std::fs::create_dir_all(&direct_skill).unwrap();
        std::fs::write(
            plugin_dir.join("direct-agent.md"),
            "# Direct Agent\nUse this direct agent.",
        )
        .unwrap();
        std::fs::write(
            extra_agents.join("reviewer.md"),
            "# Reviewer\nReview carefully.",
        )
        .unwrap();
        std::fs::write(direct_skill.join("Skill.MD"), "# Helper\nUse this skill.").unwrap();
        std::fs::write(extra_skills.join("flat.md"), "# Flat\nUse this flat skill.").unwrap();

        let mut plugin = make_plugin("agent-plugin");
        plugin.path = plugin_dir;
        plugin.manifest.agents = vec![
            "./direct-agent.md".to_string(),
            "./extra-agents".to_string(),
            r".\extra-agents".to_string(),
            "../outside-agent.md".to_string(),
        ];
        plugin.manifest.skills = vec![
            "./extra-skills".to_string(),
            "./extra-skills/helper".to_string(),
            "./extra-skills/flat.md".to_string(),
            r".\extra-skills".to_string(),
            "../outside-skills".to_string(),
        ];

        let mut reg = PluginRegistry::new();
        reg.insert(plugin);

        let agent_files: Vec<String> = reg
            .all_agent_files()
            .into_iter()
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(agent_files.contains(&"direct-agent.md".to_string()));
        assert!(agent_files.contains(&"reviewer.md".to_string()));
        assert!(!agent_files.contains(&"outside-agent.md".to_string()));
        assert_eq!(
            agent_files
                .iter()
                .filter(|name| name.as_str() == "reviewer.md")
                .count(),
            1
        );

        let skill_paths = reg.all_skill_paths();
        assert!(skill_paths
            .iter()
            .any(|path| path.ends_with("extra-skills")));
        assert!(!skill_paths
            .iter()
            .any(|path| path.ends_with("outside-skills")));
        assert!(!skill_paths.iter().any(|path| path.ends_with("helper")));
        assert!(!skill_paths.iter().any(|path| path.ends_with("flat.md")));
        assert_eq!(
            skill_paths
                .iter()
                .filter(|path| path.ends_with("extra-skills"))
                .count(),
            1
        );
    }

    #[test]
    fn diff_detects_added_removed() {
        let mut old_reg = PluginRegistry::new();
        old_reg.insert(make_plugin("kept"));
        old_reg.insert(make_plugin("gone"));

        let mut new_reg = PluginRegistry::new();
        new_reg.insert(make_plugin("kept"));
        new_reg.insert(make_plugin("new-plugin"));

        let diff = new_reg.diff_against(&old_reg);
        assert_eq!(diff.added, vec!["new-plugin"]);
        assert_eq!(diff.removed, vec!["gone"]);
        assert!(diff.updated.is_empty());
    }
}
