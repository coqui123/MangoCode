/// Plugin discovery and loading.
///
/// Scan order:
/// 1. `~/.mangocode/plugins/<name>/`  — user-global plugins
/// 2. `<project>/.mangocode/plugins/<name>/`  — project-local plugins
/// 3. Extra paths from `settings.plugin_paths` (if the field exists)
///
/// Each plugin directory must contain a `plugin.json` or `plugin.toml`
/// manifest file.  A bare manifest file (no containing directory) is also
/// accepted.
use crate::manifest::{normalize_manifest_relative_path, PluginHooksConfig, PluginManifest};
use crate::plugin::{LoadedPlugin, PluginError, PluginSource};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Return the default user-level plugins directory: `~/.mangocode/plugins`.
pub fn default_user_plugins_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mangocode").join("plugins"))
}

/// Return the project-level plugins directory: `<project>/.mangocode/plugins`.
pub fn project_plugins_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(".mangocode").join("plugins")
}

// ---------------------------------------------------------------------------
// Core loader
// ---------------------------------------------------------------------------

/// Discover and load all plugins from the given root directories.
///
/// Each directory in `search_dirs` is scanned at depth 1: every immediate
/// subdirectory (or manifest file) is treated as a candidate plugin.
pub async fn discover_plugins(
    search_dirs: &[PathBuf],
    source: PluginSource,
) -> (Vec<LoadedPlugin>, Vec<PluginError>) {
    let mut plugins: Vec<LoadedPlugin> = Vec::new();
    let mut errors: Vec<PluginError> = Vec::new();

    for dir in search_dirs {
        if !dir.exists() {
            continue;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                errors.push(PluginError::Io {
                    path: dir.to_string_lossy().into_owned(),
                    message: e.to_string(),
                });
                continue;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            match try_load_from_path(&path, source.clone()) {
                Ok(Some(plugin)) => plugins.push(plugin),
                Ok(None) => {}
                Err(e) => errors.push(e),
            }
        }
    }

    (plugins, errors)
}

/// Try to load a plugin from a filesystem path.
///
/// `path` can be:
/// - A directory containing `plugin.json` or `plugin.toml`
/// - A direct `plugin.json` or `plugin.toml` file
///
/// Returns `Ok(None)` if the path does not look like a plugin (no manifest
/// found) without adding an error.
pub fn try_load_from_path(
    path: &Path,
    source: PluginSource,
) -> Result<Option<LoadedPlugin>, PluginError> {
    let (plugin_dir, manifest_path) = if path.is_dir() {
        // Look for manifest inside the directory.
        let json_path = path.join("plugin.json");
        let toml_path = path.join("plugin.toml");

        if json_path.exists() {
            (path.to_path_buf(), json_path)
        } else if toml_path.exists() {
            (path.to_path_buf(), toml_path)
        } else {
            // Directory with no manifest — not a plugin, skip silently.
            return Ok(None);
        }
    } else if path.is_file() {
        // Accept a bare manifest file.
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "plugin.json" || name == "plugin.toml" {
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            (parent, path.to_path_buf())
        } else {
            return Ok(None);
        }
    } else {
        return Ok(None);
    };

    let manifest = load_manifest(&manifest_path)?;

    // Resolve sub-paths.
    let commands_path = {
        let p = plugin_dir.join("commands");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let agents_path = {
        let p = plugin_dir.join("agents");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let skills_path = {
        let p = plugin_dir.join("skills");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let output_styles_path = {
        let p = plugin_dir.join("output-styles");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };

    // Load hooks config (hooks/hooks.json takes priority over inline manifest field).
    let hooks_config = load_hooks_config(&plugin_dir, &manifest);

    let plugin_name = manifest.name.clone();
    let plugin_source_id = format!("{}@{}", plugin_name, source.label());

    Ok(Some(LoadedPlugin {
        name: plugin_name,
        path: plugin_dir,
        source: source.clone(),
        source_id: plugin_source_id,
        manifest,
        enabled: true,
        commands_path,
        agents_path,
        skills_path,
        output_styles_path,
        hooks_config,
    }))
}

// ---------------------------------------------------------------------------
// Manifest loading
// ---------------------------------------------------------------------------

fn load_manifest(path: &Path) -> Result<PluginManifest, PluginError> {
    let bytes = std::fs::read(path).map_err(|e| PluginError::Io {
        path: path.to_string_lossy().into_owned(),
        message: e.to_string(),
    })?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("json");

    let manifest = match ext {
        "toml" => PluginManifest::from_toml(&bytes).map_err(|e| PluginError::InvalidManifest {
            path: path.to_string_lossy().into_owned(),
            message: e.to_string(),
        })?,
        _ => PluginManifest::from_json(&bytes).map_err(|e| PluginError::InvalidManifest {
            path: path.to_string_lossy().into_owned(),
            message: e.to_string(),
        })?,
    };

    Ok(manifest)
}

// ---------------------------------------------------------------------------
// Hooks loading
// ---------------------------------------------------------------------------

/// Load hooks for a plugin.
///
/// Priority:
/// 1. `hooks/hooks.json` inside the plugin directory
/// 2. Inline `hooks` field in the manifest
pub fn load_hooks_config(
    plugin_dir: &Path,
    manifest: &PluginManifest,
) -> Option<PluginHooksConfig> {
    // 1. File-based hooks.
    let hooks_file = plugin_dir.join("hooks").join("hooks.json");
    if hooks_file.exists() {
        if let Ok(bytes) = std::fs::read(&hooks_file) {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(config) = crate::hooks::parse_hooks_value(&value) {
                    return Some(config);
                }
            }
        }
    }

    // 2. Inline hooks in manifest.
    if let Some(ref inline) = manifest.hooks {
        if let Some(config) = crate::hooks::parse_hooks_value(inline) {
            return Some(config);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Command definitions collected from a plugin
// ---------------------------------------------------------------------------

/// Scan a plugin's commands directory and return all `PluginCommandDef` items.
pub fn collect_command_defs(plugin: &LoadedPlugin) -> Vec<crate::plugin::PluginCommandDef> {
    let mut defs: Vec<crate::plugin::PluginCommandDef> = Vec::new();
    let mut seen_files = HashSet::new();
    let mut seen_command_names = HashSet::new();

    {
        let mut collection = MarkdownCommandCollection {
            plugin_root: &plugin.path,
            plugin_source_id: &plugin.source_id,
            plugin_name: &plugin.name,
            capabilities: plugin.manifest.capabilities.clone(),
            defs: &mut defs,
            seen_files: &mut seen_files,
            seen_command_names: &mut seen_command_names,
        };

        // Commands from the `commands/` directory.
        if let Some(ref cmd_dir) = plugin.commands_path {
            collect_markdown_commands(cmd_dir, cmd_dir, &mut collection);
        }

        // Extra commands declared in the manifest.
        for rel_path in &plugin.manifest.commands {
            let Ok(rel_path) = normalize_manifest_relative_path(rel_path) else {
                continue;
            };
            let abs = plugin.path.join(rel_path);
            if abs.is_file() && has_markdown_extension(&abs) {
                if !mark_command_file_seen(&abs, collection.seen_files) {
                    continue;
                }
                let root = abs.parent().unwrap_or(&plugin.path);
                let cmd_name = command_name_from_markdown_file(&abs, root, &plugin.name);
                if !mark_command_name_seen(&cmd_name, collection.seen_command_names) {
                    continue;
                }
                collection.defs.push(crate::plugin::PluginCommandDef {
                    name: cmd_name,
                    description: extract_description_from_markdown_file(&abs)
                        .unwrap_or_else(|| "Plugin command".to_string()),
                    plugin_name: plugin.name.clone(),
                    plugin_source_id: plugin.source_id.clone(),
                    run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                        file_path: abs.to_string_lossy().into_owned(),
                        plugin_root: plugin.path.to_string_lossy().into_owned(),
                    },
                    plugin_capabilities: collection.capabilities.clone(),
                });
            } else if abs.is_dir() {
                collect_markdown_commands(&abs, &abs, &mut collection);
            }
        }
    }

    defs
}

struct MarkdownCommandCollection<'a> {
    plugin_root: &'a Path,
    plugin_source_id: &'a str,
    plugin_name: &'a str,
    capabilities: Option<Vec<String>>,
    defs: &'a mut Vec<crate::plugin::PluginCommandDef>,
    seen_files: &'a mut HashSet<PathBuf>,
    seen_command_names: &'a mut HashSet<String>,
}

/// Recursively collect .md files from `dir` into `PluginCommandDef` items.
fn collect_markdown_commands(
    dir: &Path,
    root: &Path,
    collection: &mut MarkdownCommandCollection<'_>,
) {
    use walkdir::WalkDir;

    let mut paths: Vec<PathBuf> = WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file())
        .collect();
    paths.sort_by_key(|path| command_file_sort_key(path));

    for path in paths {
        let path = path.as_path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // SKILL.md — use parent directory name as command name.
        if file_name.eq_ignore_ascii_case("skill.md") {
            if !mark_command_file_seen(path, collection.seen_files) {
                continue;
            }
            let cmd_name = command_name_from_skill_index_file(path, root, collection.plugin_name);
            if !mark_command_name_seen(&cmd_name, collection.seen_command_names) {
                continue;
            }
            collection.defs.push(crate::plugin::PluginCommandDef {
                name: cmd_name,
                description: extract_description_from_markdown_file(path)
                    .unwrap_or_else(|| "Plugin skill".to_string()),
                plugin_name: collection.plugin_name.to_string(),
                plugin_source_id: collection.plugin_source_id.to_string(),
                run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                    file_path: path.to_string_lossy().into_owned(),
                    plugin_root: collection.plugin_root.to_string_lossy().into_owned(),
                },
                plugin_capabilities: collection.capabilities.clone(),
            });
            continue;
        }

        if has_markdown_extension(path) {
            if !mark_command_file_seen(path, collection.seen_files) {
                continue;
            }
            let cmd_name = command_name_from_markdown_file(path, root, collection.plugin_name);
            if !mark_command_name_seen(&cmd_name, collection.seen_command_names) {
                continue;
            }
            collection.defs.push(crate::plugin::PluginCommandDef {
                name: cmd_name,
                description: extract_description_from_markdown_file(path)
                    .unwrap_or_else(|| "Plugin command".to_string()),
                plugin_name: collection.plugin_name.to_string(),
                plugin_source_id: collection.plugin_source_id.to_string(),
                run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                    file_path: path.to_string_lossy().into_owned(),
                    plugin_root: collection.plugin_root.to_string_lossy().into_owned(),
                },
                plugin_capabilities: collection.capabilities.clone(),
            });
        }
    }
}

fn command_file_sort_key(path: &Path) -> (u8, PathBuf) {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let is_skill_index = file_name.eq_ignore_ascii_case("skill.md");
    (u8::from(is_skill_index), path.to_path_buf())
}

fn mark_command_file_seen(path: &Path, seen_files: &mut HashSet<PathBuf>) -> bool {
    let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    seen_files.insert(key)
}

fn mark_command_name_seen(name: &str, seen_command_names: &mut HashSet<String>) -> bool {
    let key = name.trim().trim_start_matches('/').trim().to_lowercase();
    !key.is_empty() && seen_command_names.insert(key)
}

/// Derive a slash-command name from a markdown file path.
///
/// e.g. `<plugin_dir>/commands/build/deploy.md` → `myplugin:build:deploy`
fn has_markdown_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
}

fn command_name_from_markdown_file(path: &Path, root: &Path, plugin_name: &str) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if file_name.eq_ignore_ascii_case("skill.md") {
        command_name_from_skill_index_file(path, root, plugin_name)
    } else {
        command_name_from_file(path, root, plugin_name)
    }
}

fn command_name_from_skill_index_file(path: &Path, root: &Path, plugin_name: &str) -> String {
    let skill_dir = path.parent().unwrap_or(root);
    let parts = relative_path_parts(skill_dir, root);
    if parts.is_empty() {
        let base_name = skill_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("skill");
        format!("{}:{}", plugin_name, base_name)
    } else {
        format!("{}:{}", plugin_name, parts.join(":"))
    }
}

fn command_name_from_file(path: &Path, root: &Path, plugin_name: &str) -> String {
    let parts = command_path_parts(path, root);
    if parts.is_empty() {
        format!("{}:cmd", plugin_name)
    } else {
        format!("{}:{}", plugin_name, parts.join(":"))
    }
}

fn command_path_parts(path: &Path, root: &Path) -> Vec<String> {
    let mut parts = relative_path_parts(path, root);
    if let Some(last) = parts.last_mut() {
        if let Some(stem) = Path::new(last).file_stem().and_then(|s| s.to_str()) {
            *last = stem.to_string();
        }
    }
    parts
}

fn relative_path_parts(path: &Path, root: &Path) -> Vec<String> {
    use std::path::Component;

    let relative = path.strip_prefix(root).unwrap_or(path);
    relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .filter(|part| !part.is_empty())
        .collect()
}

/// Pull the first non-empty line from a markdown file as a description.
fn extract_description_from_markdown_file(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let trimmed = line.trim_start_matches('#').trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::PluginManifest;
    use crate::plugin::PluginSource;
    use tempfile::TempDir;

    fn make_plugin(plugin_dir: PathBuf, commands: Vec<String>) -> LoadedPlugin {
        LoadedPlugin {
            name: "style-plugin".to_string(),
            path: plugin_dir,
            source: PluginSource::Project,
            source_id: "style-plugin@project".to_string(),
            manifest: PluginManifest {
                name: "style-plugin".to_string(),
                commands,
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
    fn collect_command_defs_skips_manifest_paths_outside_plugin_root() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(
            tmp.path().join("outside.md"),
            "# Escaped\nThis command should not load.",
        )
        .unwrap();

        let plugin = make_plugin(plugin_dir, vec!["../outside.md".to_string()]);

        assert!(collect_command_defs(&plugin).is_empty());
    }

    #[test]
    fn collect_command_defs_accepts_normalized_manifest_paths() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("extra");
        std::fs::create_dir_all(&commands_dir).unwrap();
        std::fs::write(
            commands_dir.join("review.md"),
            "# Review\nReview the current changes.",
        )
        .unwrap();

        let plugin = make_plugin(plugin_dir, vec!["./extra/review.md".to_string()]);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:review");
    }

    #[test]
    fn collect_command_defs_preserves_nested_manifest_directory_path() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("extra");
        let nested_dir = commands_dir.join("build");
        std::fs::create_dir_all(&nested_dir).unwrap();
        std::fs::write(
            nested_dir.join("deploy.md"),
            "# Deploy\nDeploy the current build.",
        )
        .unwrap();

        let plugin = make_plugin(plugin_dir.clone(), vec!["./extra".to_string()]);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:build:deploy");
        assert_eq!(commands[0].plugin_source_id, "style-plugin@project");
        match &commands[0].run_action {
            crate::plugin::CommandRunAction::MarkdownPrompt { plugin_root, .. } => {
                assert_eq!(plugin_root.as_str(), plugin_dir.to_string_lossy().as_ref());
            }
            other => panic!("expected markdown prompt, got {other:?}"),
        }
    }

    #[test]
    fn collect_command_defs_accepts_case_insensitive_markdown_extensions() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("extra");
        std::fs::create_dir_all(&commands_dir).unwrap();
        std::fs::write(commands_dir.join("Build.MD"), "# Build\nBuild the project.").unwrap();

        let plugin = make_plugin(plugin_dir, vec!["./extra/Build.MD".to_string()]);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:Build");
    }

    #[test]
    fn collect_command_defs_names_manifest_skill_file_from_parent_dir() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let skill_dir = plugin_dir.join("extra").join("deploy");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "# Deploy Skill\nDeploy with the skill runner.",
        )
        .unwrap();

        let plugin = make_plugin(plugin_dir, vec!["./extra/deploy/SKILL.md".to_string()]);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:deploy");
        assert_eq!(commands[0].description, "Deploy Skill");
    }

    #[test]
    fn collect_command_defs_preserves_nested_commands_dir_path() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("commands");
        let nested_dir = commands_dir.join("build");
        std::fs::create_dir_all(&nested_dir).unwrap();
        std::fs::write(
            nested_dir.join("deploy.md"),
            "# Deploy\nDeploy the current build.",
        )
        .unwrap();

        let mut plugin = make_plugin(plugin_dir.clone(), vec![]);
        plugin.commands_path = Some(commands_dir);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:build:deploy");
        assert_eq!(commands[0].plugin_source_id, "style-plugin@project");
        match &commands[0].run_action {
            crate::plugin::CommandRunAction::MarkdownPrompt { plugin_root, .. } => {
                assert_eq!(plugin_root.as_str(), plugin_dir.to_string_lossy().as_ref());
            }
            other => panic!("expected markdown prompt, got {other:?}"),
        }
    }

    #[test]
    fn collect_command_defs_dedupes_manifest_commands_already_in_commands_dir() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("commands");
        let nested_dir = commands_dir.join("build");
        std::fs::create_dir_all(&nested_dir).unwrap();
        std::fs::write(
            nested_dir.join("deploy.md"),
            "# Deploy\nDeploy the current build.",
        )
        .unwrap();

        let mut plugin = make_plugin(
            plugin_dir,
            vec![
                "./commands".to_string(),
                "./commands/build/deploy.md".to_string(),
            ],
        );
        plugin.commands_path = Some(commands_dir);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:build:deploy");
    }

    #[test]
    fn collect_command_defs_dedupes_command_name_collisions() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("commands");
        let skill_dir = commands_dir.join("deploy");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            commands_dir.join("deploy.md"),
            "# File Deploy\nDeploy from the direct command file.",
        )
        .unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "# Skill Deploy\nDeploy from the skill command directory.",
        )
        .unwrap();

        let mut plugin = make_plugin(plugin_dir, vec![]);
        plugin.commands_path = Some(commands_dir);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:deploy");
        assert_eq!(commands[0].description, "File Deploy");
    }

    #[test]
    fn collect_command_defs_preserves_nested_skill_command_path() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join("style-plugin");
        let commands_dir = plugin_dir.join("commands");
        let nested_skill_dir = commands_dir.join("build").join("deploy");
        std::fs::create_dir_all(&nested_skill_dir).unwrap();
        std::fs::write(
            nested_skill_dir.join("SKILL.md"),
            "# Deploy Skill\nDeploy with the skill runner.",
        )
        .unwrap();

        let mut plugin = make_plugin(plugin_dir, vec![]);
        plugin.commands_path = Some(commands_dir);
        let commands = collect_command_defs(&plugin);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "style-plugin:build:deploy");
        assert_eq!(commands[0].plugin_source_id, "style-plugin@project");
    }
}
