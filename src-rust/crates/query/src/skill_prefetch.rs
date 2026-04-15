//! Skill prefetch — mirrors src/services/skillSearch/prefetch.js
//!
//! Reads all bundled and user-defined skill definitions in the background
//! and builds a searchable index. The query loop injects the skill listing
//! as a tool-context attachment when the index is ready.
//!
//! ## Extended indexing (Phase 1 / 2 additions)
//!
//! `SkillDefinition.tags` is now populated from the skill's `triggers` /
//! `when_to_use` frontmatter keys in addition to the explicit `tags:` list.
//! This means `SkillIndex::search` doubles as the intent-matching backend
//! for Phase 1 auto-loading with no additional code.
//!
//! Folder-based skills (directories with a `SKILL.md`) are indexed along
//! with their sub-file names, so the model can discover them via search.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A single skill definition held in the prefetch index.
#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    /// Searchable tags — includes explicit `tags:` values **and** all
    /// `triggers:` / `when_to_use:` phrases from the skill frontmatter.
    pub tags: Vec<String>,
    /// Source: "bundled" | "user" | "plugin:{name}"
    pub source: String,
    /// Path to the skill file (`.md`) or directory (`SKILL.md` container) on disk.
    pub path: Option<std::path::PathBuf>,
    /// Names of sub-files available for folder-based skills (e.g. `["security", "performance"]`).
    pub sub_file_names: Vec<String>,
    /// Whether this skill declares mandatory QA steps.
    pub qa_required: bool,
}

/// In-memory skill search index.
#[derive(Debug, Default)]
pub struct SkillIndex {
    /// All skills, keyed by name (lowercase).
    skills: HashMap<String, SkillDefinition>,
}

impl SkillIndex {
    /// Add a skill to the index.
    pub fn insert(&mut self, skill: SkillDefinition) {
        self.skills.insert(skill.name.to_lowercase(), skill);
    }

    /// Query by partial name, description, or tag match (case-insensitive).
    ///
    /// Because `tags` now contains trigger phrases, this also serves as the
    /// intent-matching query for Phase 1 auto-loading.
    pub fn search(&self, query: &str) -> Vec<&SkillDefinition> {
        let q = query.to_lowercase();
        self.skills
            .values()
            .filter(|s| {
                s.name.to_lowercase().contains(&q)
                    || s.description.to_lowercase().contains(&q)
                    || s.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// Return all skills.
    pub fn all(&self) -> Vec<&SkillDefinition> {
        self.skills.values().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }
}

/// Shared handle to the skill index (populated in the background).
pub type SharedSkillIndex = Arc<RwLock<SkillIndex>>;

/// Scan `project_root` for skill definitions in `.mangocode/skills/` and the
/// bundled skill list, then write the completed index into `index`.
///
/// This runs as a `tokio::task::spawn` parallel to model streaming.
pub async fn prefetch_skills(project_root: &Path, index: SharedSkillIndex) {
    let mut local = SkillIndex::default();

    // 1. User-defined skills: ~/.mangocode/skills/ + {project_root}/.mangocode/skills/
    let search_dirs: Vec<std::path::PathBuf> = {
        let mut dirs = Vec::new();
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join(".mangocode").join("skills"));
            dirs.push(home.join(".mangocode").join("commands"));
        }
        dirs.push(project_root.join(".mangocode").join("skills"));
        dirs.push(project_root.join(".mangocode").join("commands"));
        dirs.push(project_root.join(".agents").join("skills"));
        dirs
    };

    for dir in &search_dirs {
        index_dir(&mut local, dir);
    }

    // 2. Bundled skills: check if we ship any in a `skills/` directory next to the binary.
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        let bundled = exe_dir.join("skills");
        if bundled.is_dir() {
            let mut bundled_skills = SkillIndex::default();
            index_dir(&mut bundled_skills, &bundled);
            for mut skill in bundled_skills.skills.into_values() {
                skill.source = "bundled".to_string();
                local.insert(skill);
            }
        }
    }

    // Write the index once fully loaded.
    let mut guard = index.write().await;
    *guard = local;
}

/// Index all skills (flat `.md` files and folder-based `SKILL.md` directories)
/// found inside `dir` into `index`.
fn index_dir(index: &mut SkillIndex, dir: &Path) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Folder-based skill: look for SKILL.md
            let skill_md = path.join("SKILL.md");
            if skill_md.is_file() {
                if let Some(skill) = load_skill_from_file(&skill_md) {
                    // Override name to use directory stem when SKILL.md has no `name:` field
                    // (load_skill_from_file handles this via the path stem logic, but we
                    // double-check using the directory name for folder skills).
                    let mut skill = skill;
                    if skill.name == "SKILL" {
                        if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                            skill.name = dir_name.to_string();
                        }
                    }
                    // Collect sub-file names from sibling .md files
                    let mut sub_file_names: Vec<String> = Vec::new();
                    if let Ok(sub_entries) = std::fs::read_dir(&path) {
                        for se in sub_entries.flatten() {
                            let sp = se.path();
                            if sp.is_file()
                                && sp.extension().and_then(|e| e.to_str()) == Some("md")
                                && sp.file_name().and_then(|s| s.to_str()) != Some("SKILL.md")
                            {
                                if let Some(stem) = sp.file_stem().and_then(|s| s.to_str()) {
                                    sub_file_names.push(stem.to_string());
                                }
                            }
                        }
                    }
                    skill.sub_file_names = sub_file_names;
                    skill.path = Some(path);
                    index.insert(skill);
                }
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Some(skill) = load_skill_from_file(&path) {
                index.insert(skill);
            }
        }
    }
}

/// Parse a skill Markdown file into a `SkillDefinition`.
///
/// Supports the full frontmatter spec:
/// ```text
/// ---
/// name: my-skill
/// description: Does something useful
/// tags: [tag1, tag2]
/// triggers:
///   - keyword phrase one
///   - keyword phrase two
/// when_to_use: single phrase shorthand
/// qa_required: true
/// ---
///
/// Skill instructions here...
/// ```
///
/// `tags` in the returned `SkillDefinition` is the union of explicit `tags:`
/// values, `triggers:` items, and the `when_to_use:` value.
fn load_skill_from_file(path: &std::path::Path) -> Option<SkillDefinition> {
    let content = std::fs::read_to_string(path).ok()?;
    let stem = path.file_stem()?.to_string_lossy().to_string();

    if let Some(after_open) = content.strip_prefix("---") {
        let end = after_open.find("\n---")?;
        let front = &after_open[..end];

        let name = extract_yaml_str(front, "name").unwrap_or_else(|| stem.clone());
        let description = extract_yaml_str(front, "description").unwrap_or_default();
        let qa_required = extract_yaml_str(front, "qa_required")
            .map(|v| matches!(v.as_str(), "true" | "yes" | "1"))
            .unwrap_or(false);

        // Build tags = explicit tags + trigger phrases + when_to_use
        let mut tags = extract_yaml_list(front, "tags");
        let triggers = extract_yaml_block_list(front, "triggers");
        tags.extend(triggers);
        if let Some(wtu) = extract_yaml_str(front, "when_to_use") {
            if !wtu.is_empty() {
                tags.push(wtu);
            }
        }
        tags.dedup();

        Some(SkillDefinition {
            name,
            description,
            tags,
            source: "user".to_string(),
            path: Some(path.to_path_buf()),
            sub_file_names: Vec::new(),
            qa_required,
        })
    } else {
        // No frontmatter: use filename as name, first line as description
        Some(SkillDefinition {
            name: stem,
            description: content.lines().next().unwrap_or("").to_string(),
            tags: Vec::new(),
            source: "user".to_string(),
            path: Some(path.to_path_buf()),
            sub_file_names: Vec::new(),
            qa_required: false,
        })
    }
}

fn extract_yaml_str(front: &str, key: &str) -> Option<String> {
    for line in front.lines() {
        if let Some(rest) = line.trim().strip_prefix(&format!("{key}:")) {
            let val = rest.trim().trim_matches('"').trim_matches('\'').to_string();
            if !val.is_empty() && val != "~" {
                return Some(val);
            }
        }
    }
    None
}

/// Parse an inline YAML list: `key: [a, b, c]`
fn extract_yaml_list(front: &str, key: &str) -> Vec<String> {
    for line in front.lines() {
        if let Some(rest) = line.trim().strip_prefix(&format!("{key}:")) {
            let rest = rest.trim().trim_start_matches('[').trim_end_matches(']');
            return rest
                .split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Parse a YAML block list:
/// ```yaml
/// key:
///   - item one
///   - item two
/// ```
fn extract_yaml_block_list(front: &str, key: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut in_block = false;
    for line in front.lines() {
        let trimmed = line.trim();
        if trimmed == format!("{key}:") {
            in_block = true;
            continue;
        }
        if in_block {
            if let Some(val) = trimmed.strip_prefix("- ") {
                result.push(val.trim().trim_matches('"').trim_matches('\'').to_string());
            } else if !trimmed.is_empty() && !trimmed.starts_with('-') {
                // Hit a new key — stop collecting
                in_block = false;
            }
        }
    }
    result
}

/// Format a skill listing attachment for injection into the conversation.
pub fn format_skill_listing(index: &SkillIndex) -> String {
    if index.is_empty() {
        return String::new();
    }
    let mut out = String::from("Available skills:\n");
    let mut skills: Vec<_> = index.all();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    for skill in skills {
        let mut meta_parts: Vec<String> = Vec::new();
        if !skill.sub_file_names.is_empty() {
            meta_parts.push(format!("sub-files: {}", skill.sub_file_names.join(", ")));
        }
        if skill.qa_required {
            meta_parts.push("qa-enforced".to_string());
        }
        let meta = if meta_parts.is_empty() {
            String::new()
        } else {
            format!(" [{}]", meta_parts.join("; "))
        };
        out.push_str(&format!(
            "  /{} — {}{}\n",
            skill.name, skill.description, meta
        ));
    }
    out
}
