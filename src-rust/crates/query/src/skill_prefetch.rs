//! Skill prefetch.
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
    /// Source: "discovered" | "bundled"
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
    pub fn insert(&mut self, mut skill: SkillDefinition) {
        let Some(name) = normalized_skill_name(&skill.name) else {
            return;
        };
        skill.name = name;
        let key = skill.name.to_lowercase();
        self.skills.entry(key).or_insert(skill);
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

fn normalized_skill_name(name: &str) -> Option<String> {
    let name = strip_markdown_suffix(name.trim().trim_start_matches('/').trim()).trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn strip_markdown_suffix(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() >= 3 && bytes[bytes.len() - 3..].eq_ignore_ascii_case(b".md") {
        &name[..name.len() - 3]
    } else {
        name
    }
}

/// Shared handle to the skill index (populated in the background).
pub type SharedSkillIndex = Arc<RwLock<SkillIndex>>;

/// Scan `project_root`, configured paths, plugin paths, and bundled skills,
/// then write the completed index into `index`.
///
/// This runs as a `tokio::task::spawn` parallel to model streaming.
pub async fn prefetch_skills(
    project_root: &Path,
    skills_config: &mangocode_core::config::SkillsConfig,
    index: SharedSkillIndex,
) {
    let mut local = SkillIndex::default();

    // 1. Plugin, configured, URL-backed, and user-defined skills. Use the same
    // core discovery path as slash commands so names, priority, dependencies,
    // sub-files, and git URL caches stay wired consistently.
    let skills_config = mangocode_plugins::skills_config_with_plugin_paths(skills_config);
    for skill in mangocode_core::discover_skills(project_root, &skills_config).into_values() {
        local.insert(skill_definition_from_discovered(skill));
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

fn skill_definition_from_discovered(skill: mangocode_core::DiscoveredSkill) -> SkillDefinition {
    let mut sub_file_names: Vec<String> = skill.sub_files.keys().cloned().collect();
    sub_file_names.sort();

    SkillDefinition {
        name: skill.name,
        description: skill.description,
        tags: skill.triggers,
        source: "discovered".to_string(),
        path: Some(skill.source_path),
        sub_file_names,
        qa_required: skill.qa_required,
    }
}

/// Index all skills (flat `.md` files and folder-based `SKILL.md` directories)
/// found inside `dir` into `index`.
fn index_dir(index: &mut SkillIndex, dir: &Path) {
    if !dir.is_dir() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                path = %dir.display(),
                error = %err,
                "failed to read skill directory"
            );
            return;
        }
    };

    let mut paths = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!(
                    path = %dir.display(),
                    error = %err,
                    "failed to read skill directory entry"
                );
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to inspect skill path"
                );
                continue;
            }
        };
        paths.push((path, file_type));
    }
    paths.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (path, file_type) in paths {
        if file_type.is_dir() {
            // Folder-based skill: look for SKILL.md
            if let Some(skill_md) = skill_index_file(&path) {
                if let Some(skill) = load_skill_from_file(&skill_md) {
                    // Override name to use directory stem when SKILL.md has no `name:` field
                    // (load_skill_from_file handles this via the path stem logic, but we
                    // double-check using the directory name for folder skills).
                    let mut skill = skill;
                    if skill.name.eq_ignore_ascii_case("skill") {
                        if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                            skill.name = dir_name.to_string();
                        }
                    }
                    // Collect sub-file names from sibling .md files
                    let mut sub_file_names: Vec<String> = Vec::new();
                    match std::fs::read_dir(&path) {
                        Ok(sub_entries) => {
                            let mut sub_paths = Vec::new();
                            for sub_entry in sub_entries {
                                let sub_entry = match sub_entry {
                                    Ok(sub_entry) => sub_entry,
                                    Err(err) => {
                                        tracing::warn!(
                                            path = %path.display(),
                                            error = %err,
                                            "failed to read skill sub-file directory entry"
                                        );
                                        continue;
                                    }
                                };
                                let sub_path = sub_entry.path();
                                let is_file = match sub_entry.file_type() {
                                    Ok(file_type) => file_type.is_file(),
                                    Err(err) => {
                                        tracing::warn!(
                                            path = %sub_path.display(),
                                            error = %err,
                                            "failed to inspect skill sub-file path"
                                        );
                                        continue;
                                    }
                                };
                                sub_paths.push((sub_path, is_file));
                            }
                            sub_paths.sort_by(|(left, _), (right, _)| left.cmp(right));
                            for (sp, is_file) in sub_paths {
                                if is_file
                                    && has_markdown_extension(&sp)
                                    && !is_skill_index_file_name(
                                        sp.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                                    )
                                {
                                    if let Some(stem) = sp.file_stem().and_then(|s| s.to_str()) {
                                        sub_file_names.push(stem.to_string());
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                path = %path.display(),
                                error = %err,
                                "failed to read skill sub-file directory"
                            );
                        }
                    }
                    sub_file_names.sort();
                    skill.sub_file_names = sub_file_names;
                    skill.path = Some(path);
                    index.insert(skill);
                }
            }
        } else if file_type.is_file() && has_markdown_extension(&path) {
            if let Some(skill) = load_skill_from_file(&path) {
                index.insert(skill);
            }
        }
    }
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
                "failed to read skill directory for index file"
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
                    "failed to read skill directory entry for index file"
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
                    "failed to inspect skill index path"
                );
                continue;
            }
        };
        if is_file
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_skill_index_file_name)
        {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates.into_iter().next()
}

fn is_skill_index_file_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("skill.md")
}

fn has_markdown_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
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
    match load_skill_from_file_result(path) {
        Ok(skill) => skill,
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "failed to load skill definition"
            );
            None
        }
    }
}

fn load_skill_from_file_result(path: &std::path::Path) -> Result<Option<SkillDefinition>, String> {
    let content = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let stem = path
        .file_stem()
        .ok_or_else(|| "missing file stem".to_string())?
        .to_string_lossy()
        .to_string();

    if let Some(after_open) = content.strip_prefix("---") {
        let end = after_open
            .find("\n---")
            .ok_or_else(|| "frontmatter is missing closing marker".to_string())?;
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

        Ok(Some(SkillDefinition {
            name,
            description,
            tags,
            source: "user".to_string(),
            path: Some(path.to_path_buf()),
            sub_file_names: Vec::new(),
            qa_required,
        }))
    } else {
        // No frontmatter: accept plain markdown skills (common in MangoCode skill packs).
        // - For folder-based skills, SKILL.md should map to the parent directory name.
        // - Prefer a human-readable description derived from the first meaningful markdown line.
        let name = if path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(is_skill_index_file_name)
        {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or(&stem)
                .to_string()
        } else {
            stem
        };

        let description = derive_description_from_markdown(&content);
        Ok(Some(SkillDefinition {
            name,
            description,
            tags: Vec::new(),
            source: "user".to_string(),
            path: Some(path.to_path_buf()),
            sub_file_names: Vec::new(),
            qa_required: false,
        }))
    }
}

fn derive_description_from_markdown(content: &str) -> String {
    for line in content.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(rest) = l.strip_prefix('#') {
            // Strip all leading '#' (H1/H2/etc) and whitespace.
            let mut r = rest;
            while let Some(next) = r.strip_prefix('#') {
                r = next;
            }
            let r = r.trim();
            if !r.is_empty() {
                return r.to_string();
            }
        }
        return l.to_string();
    }
    String::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_dir_accepts_case_insensitive_skill_marker_and_markdown_extensions() {
        let tmp = tempfile::tempdir().unwrap();
        let skill_dir = tmp.path().join("rust-review");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("Skill.MD"),
            "# Rust Review\nReview carefully.",
        )
        .unwrap();
        std::fs::write(skill_dir.join("Security.MD"), "# Security\nCheck security.").unwrap();
        std::fs::write(
            skill_dir.join("Performance.md"),
            "# Performance\nCheck performance.",
        )
        .unwrap();

        let mut index = SkillIndex::default();
        index_dir(&mut index, tmp.path());

        let skill = index.search("rust-review").pop().unwrap();
        assert_eq!(skill.name, "rust-review");
        assert_eq!(
            skill.sub_file_names,
            vec!["Performance".to_string(), "Security".to_string()]
        );
    }

    #[test]
    fn skill_index_normalizes_slash_prefixed_names() {
        let mut index = SkillIndex::default();
        index.insert(SkillDefinition {
            name: " /ProjectReview ".to_string(),
            description: "Review project".to_string(),
            tags: vec!["review".to_string()],
            source: "discovered".to_string(),
            path: None,
            sub_file_names: Vec::new(),
            qa_required: false,
        });

        let listing = format_skill_listing(&index);

        assert!(listing.contains("/ProjectReview"));
        assert!(!listing.contains("//ProjectReview"));
        assert!(index
            .search("ProjectReview")
            .iter()
            .any(|skill| skill.name == "ProjectReview"));
    }

    #[test]
    fn skill_index_normalizes_markdown_suffix_case_insensitively() {
        let mut index = SkillIndex::default();
        index.insert(SkillDefinition {
            name: " /ProjectReview.MD ".to_string(),
            description: "Review project".to_string(),
            tags: vec!["review".to_string()],
            source: "discovered".to_string(),
            path: None,
            sub_file_names: Vec::new(),
            qa_required: false,
        });

        let listing = format_skill_listing(&index);

        assert!(listing.contains("/ProjectReview"));
        assert!(!listing.contains("/ProjectReview.MD"));
        assert!(index
            .search("ProjectReview")
            .iter()
            .any(|skill| skill.name == "ProjectReview"));
    }

    #[test]
    fn skill_index_keeps_first_normalized_definition() {
        let mut index = SkillIndex::default();
        index.insert(SkillDefinition {
            name: "ProjectReview".to_string(),
            description: "project definition".to_string(),
            tags: vec![],
            source: "discovered".to_string(),
            path: None,
            sub_file_names: Vec::new(),
            qa_required: false,
        });
        index.insert(SkillDefinition {
            name: "/projectreview".to_string(),
            description: "bundled definition".to_string(),
            tags: vec![],
            source: "bundled".to_string(),
            path: None,
            sub_file_names: Vec::new(),
            qa_required: false,
        });

        let matches = index.search("projectreview");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "ProjectReview");
        assert_eq!(matches[0].description, "project definition");
    }
}
