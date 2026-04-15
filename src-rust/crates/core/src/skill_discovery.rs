//! Skill discovery: load custom prompt-template skills from markdown files
//! on disk and (optionally) from git URLs.
//!
//! Search priority (first match wins for a given skill name):
//!   1. Project `.mangocode/skills/` and `.mangocode/commands/` — walk up from `cwd`
//!   2. Project `.agents/skills/` and `.agents/commands/` — walk up from `cwd`
//!   3. Global `~/.mangocode/skills/` and `~/.mangocode/commands/`
//!   4. Configured extra paths from `SkillsConfig.paths`
//!   5. Git-URL repos from `SkillsConfig.urls` (cloned once, then cached)
//!
//! ## Modular / folder-based skills
//!
//! A skill directory is recognised when a folder contains a `SKILL.md` file.
//! The directory layout mirrors the Perplexity Computer skill architecture:
//!
//! ```text
//! skills/
//!   rust-review/
//!     SKILL.md          ← primary content, injected on load
//!     security.md       ← sub-file, read on demand by the model
//!     performance.md    ← sub-file, read on demand by the model
//!     scripts/
//!       validate.py     ← bundled runnable helpers, copied into session workspace
//! ```
//!
//! ## Auto-loading triggers
//!
//! Skill frontmatter may declare `triggers:` — a list of keyword phrases.
//! The intent-resolver in the query layer matches incoming messages against
//! these triggers and injects matching skills into the system prompt *before*
//! the model starts generating, so expert context is always available.
//!
//! ## Dependency chains
//!
//! Skills may declare `dependencies:` listing other skill names. When a skill
//! is loaded, its dependencies are recursively loaded first (depth-first,
//! cycle-safe). This keeps shared reference material (e.g. design-foundations)
//! out of every skill file individually.
//!
//! ## QA enforcement
//!
//! Skills may declare `qa_required: true` and a `qa_steps:` list. The loader
//! surfaces these as a hard constraint injected into the system prompt section
//! for the current task.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A discovered skill loaded from a markdown file or skill directory.
#[derive(Debug, Clone)]
pub struct DiscoveredSkill {
    /// Skill name (from `name:` frontmatter or file stem / directory name).
    pub name: String,
    /// One-line description (from `description:` frontmatter or default).
    pub description: String,
    /// The prompt body — content of `SKILL.md` or the flat `.md` file after
    /// stripping frontmatter.
    pub template: String,
    /// Absolute path to the source file or directory.
    pub source_path: PathBuf,

    // ------------------------------------------------------------------
    // Extended fields (Phase 1 / 2 / 3 / 4 / 5 additions)
    // ------------------------------------------------------------------

    /// Keyword phrases that trigger auto-loading when matched against the
    /// user's message. Populated from `triggers:` / `when_to_use:` frontmatter.
    pub triggers: Vec<String>,

    /// Other skill names that must be loaded (depth-first) before this skill
    /// is injected. Populated from `dependencies:` frontmatter.
    pub dependencies: Vec<String>,

    /// Sub-files available within a folder-based skill.
    /// Key = stem (e.g. `"security"`), value = absolute path to the `.md` file.
    pub sub_files: HashMap<String, PathBuf>,

    /// Runnable scripts found in the skill's `scripts/` subdirectory.
    /// These are copied into the session workspace when the skill is loaded.
    pub scripts: Vec<PathBuf>,

    /// Whether this skill declares mandatory QA steps.
    pub qa_required: bool,

    /// Ordered list of QA steps the model must complete before delivery.
    /// Empty unless `qa_required` is true.
    pub qa_steps: Vec<String>,
}

// ---------------------------------------------------------------------------
// Frontmatter parsing
// ---------------------------------------------------------------------------

/// Parse a skill markdown file.
///
/// Accepts optional YAML frontmatter delimited by `---`.
/// Returns `None` when the file is empty after trimming.
pub fn parse_skill_file(content: &str, path: &Path) -> Option<DiscoveredSkill> {
    parse_skill_file_with_dir(content, path, None)
}

/// Internal: parse a skill file, optionally setting a parent directory for
/// sub-file and script discovery (used when parsing `SKILL.md` inside a
/// folder-based skill).
fn parse_skill_file_with_dir(
    content: &str,
    path: &Path,
    skill_dir: Option<&Path>,
) -> Option<DiscoveredSkill> {
    let content = content.trim();
    if content.is_empty() {
        return None;
    }

    let (name, description, template, triggers, dependencies, qa_required, qa_steps) =
        if let Some(after_open) = content.strip_prefix("---") {
            // Accept both `\n---` and `\r\n---` as the closing delimiter.
            if let Some(close_pos) = after_open.find("\n---") {
                let frontmatter = &after_open[..close_pos];
                let rest = after_open[close_pos + 4..].trim_start_matches(['\r', '\n']);

                let mut name: Option<String> = None;
                let mut description: Option<String> = None;
                let mut triggers: Vec<String> = Vec::new();
                let mut dependencies: Vec<String> = Vec::new();
                let mut qa_required = false;
                let mut qa_steps: Vec<String> = Vec::new();

                // Simple line-by-line YAML parser — handles the subset we care
                // about without pulling in a full YAML crate.
                let mut in_triggers = false;
                let mut in_deps = false;
                let mut in_qa_steps = false;

                for line in frontmatter.lines() {
                    let raw = line;
                    let trimmed = line.trim();

                    // Detect list-item continuation inside a block
                    if trimmed.starts_with("- ") {
                        let val = trimmed.trim_start_matches("- ").trim().trim_matches('"').trim_matches('\'').to_string();
                        if in_triggers {
                            triggers.push(val);
                            continue;
                        } else if in_deps {
                            dependencies.push(val);
                            continue;
                        } else if in_qa_steps {
                            qa_steps.push(val);
                            continue;
                        }
                    }

                    // Reset block flags on any non-list line
                    if !trimmed.starts_with("- ") && !raw.starts_with(' ') {
                        in_triggers = false;
                        in_deps = false;
                        in_qa_steps = false;
                    }

                    if let Some(v) = trimmed.strip_prefix("name:") {
                        name = Some(v.trim().trim_matches('"').trim_matches('\'').to_string());
                    } else if let Some(v) = trimmed.strip_prefix("description:") {
                        description = Some(v.trim().trim_matches('"').trim_matches('\'').to_string());
                    } else if let Some(v) = trimmed.strip_prefix("when_to_use:") {
                        // `when_to_use:` is a single-value shorthand for triggers
                        let val = v.trim().trim_matches('"').trim_matches('\'').to_string();
                        if !val.is_empty() {
                            triggers.push(val);
                        }
                    } else if trimmed == "triggers:" {
                        in_triggers = true;
                    } else if let Some(v) = trimmed.strip_prefix("triggers:") {
                        // Inline list: `triggers: [foo, bar]`
                        let v = v.trim().trim_start_matches('[').trim_end_matches(']');
                        for item in v.split(',') {
                            let item = item.trim().trim_matches('"').trim_matches('\'');
                            if !item.is_empty() {
                                triggers.push(item.to_string());
                            }
                        }
                    } else if trimmed == "dependencies:" {
                        in_deps = true;
                    } else if let Some(v) = trimmed.strip_prefix("dependencies:") {
                        let v = v.trim().trim_start_matches('[').trim_end_matches(']');
                        for item in v.split(',') {
                            let item = item.trim().trim_matches('"').trim_matches('\'');
                            if !item.is_empty() {
                                dependencies.push(item.to_string());
                            }
                        }
                    } else if let Some(v) = trimmed.strip_prefix("qa_required:") {
                        qa_required = matches!(v.trim(), "true" | "yes" | "1");
                    } else if trimmed == "qa_steps:" {
                        in_qa_steps = true;
                    }
                }

                (
                    name,
                    description,
                    rest.to_string(),
                    triggers,
                    dependencies,
                    qa_required,
                    qa_steps,
                )
            } else {
                // Malformed frontmatter — treat entire content as template.
                (None, None, content.to_string(), vec![], vec![], false, vec![])
            }
        } else {
            (None, None, content.to_string(), vec![], vec![], false, vec![])
        };

    let name = name.unwrap_or_else(|| {
        // For folder-based skills the path points to SKILL.md; use the parent
        // directory name as the skill name.
        if path.file_name().and_then(|s| s.to_str()) == Some("SKILL.md") {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed")
                .to_string()
        } else {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed")
                .to_string()
        }
    });

    let description = description.unwrap_or_else(|| derive_description_from_markdown(&template));

    if template.is_empty() && name.is_empty() {
        return None;
    }

    // ------------------------------------------------------------------
    // Discover sub-files and scripts when a skill directory is provided
    // ------------------------------------------------------------------
    let (sub_files, scripts) = if let Some(dir) = skill_dir {
        scan_skill_dir_extras(dir, &name)
    } else {
        (HashMap::new(), Vec::new())
    };

    Some(DiscoveredSkill {
        name,
        description,
        template,
        source_path: path.to_path_buf(),
        triggers,
        dependencies,
        sub_files,
        scripts,
        qa_required,
        qa_steps,
    })
}

fn derive_description_from_markdown(content: &str) -> String {
    for line in content.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        // Guard against malformed/unclosed frontmatter that begins with `---`
        // (parse_skill_file_with_dir treats this as template content).
        if l == "---" {
            continue;
        }
        // Also skip common frontmatter-like keys if present without a closing delimiter.
        if l.starts_with("name:") || l.starts_with("description:") {
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
    "Custom skill".to_string()
}

/// Scan a folder-based skill directory for sub-files and runnable scripts.
///
/// Sub-files: any `*.md` file in the directory other than `SKILL.md`.
/// Scripts: any file in a `scripts/` subdirectory.
fn scan_skill_dir_extras(
    dir: &Path,
    _skill_name: &str,
) -> (HashMap<String, PathBuf>, Vec<PathBuf>) {
    let mut sub_files: HashMap<String, PathBuf> = HashMap::new();
    let mut scripts: Vec<PathBuf> = Vec::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return (sub_files, scripts);
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Check for a `scripts/` subdirectory
            if path.file_name().and_then(|s| s.to_str()) == Some("scripts") {
                if let Ok(script_entries) = std::fs::read_dir(&path) {
                    for se in script_entries.flatten() {
                        scripts.push(se.path());
                    }
                }
            }
            continue;
        }
        // Collect non-SKILL.md markdown files as sub-files
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if filename != "SKILL.md" {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(filename)
                    .to_string();
                sub_files.insert(stem, path);
            }
        }
    }

    (sub_files, scripts)
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

/// Scan a single directory for `*.md` skill files **and** folder-based skills
/// (directories containing a `SKILL.md` file).
fn scan_dir(dir: &Path) -> Vec<DiscoveredSkill> {
    let mut skills = Vec::new();
    if !dir.is_dir() {
        return skills;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            tracing::debug!(dir = %dir.display(), error = %err, "skill_discovery: read_dir failed");
            return skills;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Folder-based skill: look for SKILL.md inside
            let skill_md = path.join("SKILL.md");
            if skill_md.is_file() {
                match std::fs::read_to_string(&skill_md) {
                    Ok(content) => {
                        if let Some(skill) =
                            parse_skill_file_with_dir(&content, &skill_md, Some(&path))
                        {
                            skills.push(skill);
                        }
                    }
                    Err(err) => {
                        tracing::debug!(
                            path = %skill_md.display(),
                            error = %err,
                            "skill_discovery: read failed"
                        );
                    }
                }
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    if let Some(skill) = parse_skill_file(&content, &path) {
                        skills.push(skill);
                    }
                }
                Err(err) => {
                    tracing::debug!(path = %path.display(), error = %err, "skill_discovery: read failed");
                }
            }
        }
    }

    skills
}

// ---------------------------------------------------------------------------
// Top-level discovery
// ---------------------------------------------------------------------------

/// Discover all skills from all configured sources.
///
/// Returns a `HashMap` of `skill_name → DiscoveredSkill` (first match wins;
/// duplicates from lower-priority sources are warned via `tracing::warn`).
pub fn discover_skills(
    cwd: &Path,
    config_skills: &crate::config::SkillsConfig,
) -> HashMap<String, DiscoveredSkill> {
    let mut all: HashMap<String, DiscoveredSkill> = HashMap::new();
    let mut warn_duplicates: Vec<String> = Vec::new();
    let mut scanned_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    // Inline closure: insert a batch, warning on duplicates.
    let mut add = |skills: Vec<DiscoveredSkill>| {
        for skill in skills {
            if let Some(existing) = all.get(&skill.name) {
                warn_duplicates.push(format!(
                    "Duplicate skill '{}' found at {} (keeping {})",
                    skill.name,
                    skill.source_path.display(),
                    existing.source_path.display()
                ));
            } else {
                all.insert(skill.name.clone(), skill);
            }
        }
    };

    let mut scan_once = |path: PathBuf| {
        if scanned_dirs.insert(path.clone()) {
            add(scan_dir(&path));
        }
    };

    // ---- 1. Project skills: walk up from cwd --------------------------------
    {
        let mut dir: &Path = cwd;
        loop {
            scan_once(dir.join(".mangocode").join("skills"));
            scan_once(dir.join(".mangocode").join("commands"));
            scan_once(dir.join(".agents").join("skills"));
            scan_once(dir.join(".agents").join("commands"));
            match dir.parent() {
                Some(parent) if parent != dir => dir = parent,
                _ => break,
            }
        }
    }

    // ---- 2. Global skills: ~/.mangocode/skills/ and commands/ ------------------
    if let Some(home) = dirs::home_dir() {
        scan_once(home.join(".mangocode").join("skills"));
        scan_once(home.join(".mangocode").join("commands"));
    }

    // ---- 3. Configured extra paths ------------------------------------------
    for path_str in &config_skills.paths {
        let path = Path::new(path_str);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        };
        scan_once(path);
    }

    // ---- 4. Git URL skills (cached) -----------------------------------------
    for url in &config_skills.urls {
        if let Some(git_skills) = fetch_git_skills(url) {
            add(git_skills);
        }
    }

    // Emit warnings for any duplicate skill names encountered.
    for w in &warn_duplicates {
        tracing::warn!("{}", w);
    }

    all
}

// ---------------------------------------------------------------------------
// Dependency resolution
// ---------------------------------------------------------------------------

/// Recursively load `skill_name` and all its declared dependencies into
/// `context`, depth-first and cycle-safe.
///
/// Returns the ordered list of skill templates to inject (dependencies first,
/// then the requested skill). Already-loaded skill names are tracked in
/// `loaded` to prevent cycles and duplicates.
pub fn load_skill_with_dependencies(
    skill_name: &str,
    index: &HashMap<String, DiscoveredSkill>,
    loaded: &mut HashSet<String>,
    context: &mut Vec<DiscoveredSkill>,
) {
    let key = skill_name.to_lowercase();
    if loaded.contains(&key) {
        return;
    }
    loaded.insert(key.clone());

    let skill = match index.get(&key).or_else(|| {
        // Fallback: case-insensitive linear scan
        index.values().find(|s| s.name.to_lowercase() == key)
    }) {
        Some(s) => s.clone(),
        None => {
            tracing::warn!(
                skill = skill_name,
                "skill_discovery: dependency not found, skipping"
            );
            return;
        }
    };

    // Load dependencies depth-first before the skill itself
    let deps = skill.dependencies.clone();
    for dep in &deps {
        load_skill_with_dependencies(dep, index, loaded, context);
    }

    context.push(skill);
}

// ---------------------------------------------------------------------------
// QA enforcement helpers
// ---------------------------------------------------------------------------

/// Format the mandatory QA block for injection into the system prompt dynamic
/// section when a skill declares `qa_required: true`.
///
/// The returned string uses emphatic language to signal that these steps are
/// not optional, mirroring the Perplexity Computer QA pattern.
pub fn format_qa_block(skill: &DiscoveredSkill) -> Option<String> {
    if !skill.qa_required || skill.qa_steps.is_empty() {
        return None;
    }
    let mut block = format!(
        "## Required QA for this task (skill: {})\n\
         You MUST complete ALL steps below before delivering output. \
         Skipping any step is a failure.\n\n",
        skill.name
    );
    for (i, step) in skill.qa_steps.iter().enumerate() {
        block.push_str(&format!("{}. {}\n", i + 1, step));
    }
    Some(block)
}

// ---------------------------------------------------------------------------
// Script installation helpers
// ---------------------------------------------------------------------------

/// Copy all scripts bundled with `skill` into `<session_workspace>/skills/<name>/scripts/`.
///
/// This makes scripts immediately executable without the model needing to know
/// their original location on disk.
pub fn install_skill_scripts(skill: &DiscoveredSkill, session_workspace: &Path) {
    if skill.scripts.is_empty() {
        return;
    }

    let dest = session_workspace
        .join("skills")
        .join(&skill.name)
        .join("scripts");

    if let Err(err) = std::fs::create_dir_all(&dest) {
        tracing::warn!(
            dir = %dest.display(),
            error = %err,
            "skill_discovery: could not create script destination"
        );
        return;
    }

    for script_path in &skill.scripts {
        if let Some(filename) = script_path.file_name() {
            let target = dest.join(filename);
            if let Err(err) = std::fs::copy(script_path, &target) {
                tracing::warn!(
                    src = %script_path.display(),
                    dst = %target.display(),
                    error = %err,
                    "skill_discovery: script copy failed"
                );
            } else {
                tracing::debug!(
                    script = %target.display(),
                    "skill_discovery: installed skill script"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intent-based trigger matching
// ---------------------------------------------------------------------------

/// Match a user message against all skill triggers in `index`.
///
/// Returns the matching skills in descending order of match confidence
/// (number of trigger phrases matched). Ties are broken alphabetically.
///
/// This is the core of Phase 1 auto-loading: call this before the model API
/// request and inject returned skills into the system prompt dynamic section.
pub fn resolve_skills_for_message<'a>(
    message: &str,
    index: &'a HashMap<String, DiscoveredSkill>,
) -> Vec<&'a DiscoveredSkill> {
    let msg = message.to_lowercase();
    let mut matched: Vec<(&'a DiscoveredSkill, usize)> = index
        .values()
        .filter_map(|skill| {
            let hits = skill
                .triggers
                .iter()
                .filter(|t| msg.contains(t.to_lowercase().as_str()))
                .count();
            if hits > 0 {
                Some((skill, hits))
            } else {
                None
            }
        })
        .collect();

    matched.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.name.cmp(&b.0.name)));
    matched.into_iter().map(|(s, _)| s).collect()
}

// ---------------------------------------------------------------------------
// Git URL support
// ---------------------------------------------------------------------------

/// Clone or reuse a cached git repo and return skills found in it.
///
/// Cache location: `<system-cache>/mangocode/skills/<repo-name>/`
/// On first access the repo is cloned with `--depth=1`.
/// Subsequent calls use the already-cloned cache directory as-is.
fn fetch_git_skills(url: &str) -> Option<Vec<DiscoveredSkill>> {
    let cache_dir = dirs::cache_dir()?.join("mangocode").join("skills");

    // Use the last path segment of the URL as the local directory name.
    let repo_name = url.split('/').next_back()?.trim_end_matches(".git");

    if repo_name.is_empty() {
        tracing::warn!(url, "skill_discovery: cannot derive repo name from git URL");
        return None;
    }

    let repo_dir = cache_dir.join(repo_name);

    if !repo_dir.exists() {
        tracing::info!(url, dest = %repo_dir.display(), "skill_discovery: cloning skills repo");

        // Ensure the parent cache directory exists.
        if let Err(err) = std::fs::create_dir_all(&cache_dir) {
            tracing::warn!(
                dir = %cache_dir.display(),
                error = %err,
                "skill_discovery: could not create cache dir"
            );
            return None;
        }

        let repo_dir_str = repo_dir.to_str()?;
        let status = std::process::Command::new("git")
            .args(["clone", "--depth=1", url, repo_dir_str])
            .status();

        match status {
            Ok(s) if s.success() => {
                tracing::info!(url, "skill_discovery: clone succeeded");
            }
            Ok(s) => {
                tracing::warn!(url, exit_code = ?s.code(), "skill_discovery: git clone failed");
                return None;
            }
            Err(err) => {
                tracing::warn!(url, error = %err, "skill_discovery: could not spawn git");
                return None;
            }
        }
    }

    // Scan repo root and optional `skills/` subdirectory.
    let mut skills = scan_dir(&repo_dir);
    skills.extend(scan_dir(&repo_dir.join("skills")));
    Some(skills)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    fn make_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn write_file(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    // ---- parse_skill_file ---------------------------------------------------

    #[test]
    fn test_parse_with_frontmatter() {
        let content =
            "---\nname: review\ndescription: Review code changes\n---\n\nPlease review $ARGUMENTS";
        let path = PathBuf::from("review.md");
        let skill = parse_skill_file(content, &path).unwrap();
        assert_eq!(skill.name, "review");
        assert_eq!(skill.description, "Review code changes");
        assert!(skill.template.contains("$ARGUMENTS"));
    }

    #[test]
    fn test_parse_no_frontmatter_uses_stem() {
        let content = "Do something useful.";
        let path = PathBuf::from("my-skill.md");
        let skill = parse_skill_file(content, &path).unwrap();
        assert_eq!(skill.name, "my-skill");
        assert_eq!(skill.description, "Do something useful.");
        assert_eq!(skill.template, "Do something useful.");
    }

    #[test]
    fn test_parse_missing_name_uses_stem() {
        let content = "---\ndescription: No name field\n---\n\nBody text.";
        let path = PathBuf::from("fallback.md");
        let skill = parse_skill_file(content, &path).unwrap();
        assert_eq!(skill.name, "fallback");
        assert_eq!(skill.description, "No name field");
    }

    #[test]
    fn test_parse_empty_returns_none() {
        let skill = parse_skill_file("   ", &PathBuf::from("empty.md"));
        assert!(skill.is_none());
    }

    #[test]
    fn test_parse_quoted_frontmatter_values() {
        let content = "---\nname: \"quoted name\"\ndescription: 'single quoted'\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("x.md")).unwrap();
        assert_eq!(skill.name, "quoted name");
        assert_eq!(skill.description, "single quoted");
    }

    // ---- triggers -----------------------------------------------------------

    #[test]
    fn test_parse_triggers_list() {
        let content = "---\nname: code-review\ndescription: Review\ntriggers:\n  - review my code\n  - audit this\n  - check for bugs\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("code-review.md")).unwrap();
        assert_eq!(skill.triggers, vec!["review my code", "audit this", "check for bugs"]);
    }

    #[test]
    fn test_parse_when_to_use_as_trigger() {
        let content =
            "---\nname: simplify\ndescription: Clean up\nwhen_to_use: After writing code\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("simplify.md")).unwrap();
        assert_eq!(skill.triggers, vec!["After writing code"]);
    }

    #[test]
    fn test_parse_inline_triggers_list() {
        let content =
            "---\nname: foo\ntriggers: [fix bug, broken, not working]\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("foo.md")).unwrap();
        assert!(skill.triggers.contains(&"fix bug".to_string()));
        assert!(skill.triggers.contains(&"broken".to_string()));
    }

    // ---- dependencies -------------------------------------------------------

    #[test]
    fn test_parse_dependencies() {
        let content = "---\nname: pptx\ndescription: Slides\ndependencies:\n  - design-foundations\n  - git-conventions\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("pptx.md")).unwrap();
        assert_eq!(skill.dependencies, vec!["design-foundations", "git-conventions"]);
    }

    #[test]
    fn test_dependency_chain_depth_first() {
        let mut index: HashMap<String, DiscoveredSkill> = HashMap::new();

        let make = |name: &str, deps: Vec<&str>| DiscoveredSkill {
            name: name.to_string(),
            description: String::new(),
            template: format!("template-{}", name),
            source_path: PathBuf::from(format!("{}.md", name)),
            triggers: vec![],
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            sub_files: HashMap::new(),
            scripts: vec![],
            qa_required: false,
            qa_steps: vec![],
        };

        index.insert("base".to_string(), make("base", vec![]));
        index.insert("mid".to_string(), make("mid", vec!["base"]));
        index.insert("top".to_string(), make("top", vec!["mid"]));

        let mut loaded = HashSet::new();
        let mut context: Vec<DiscoveredSkill> = Vec::new();
        load_skill_with_dependencies("top", &index, &mut loaded, &mut context);

        assert_eq!(context.len(), 3);
        assert_eq!(context[0].name, "base");
        assert_eq!(context[1].name, "mid");
        assert_eq!(context[2].name, "top");
    }

    #[test]
    fn test_dependency_cycle_safe() {
        let mut index: HashMap<String, DiscoveredSkill> = HashMap::new();
        let make = |name: &str, deps: Vec<&str>| DiscoveredSkill {
            name: name.to_string(),
            description: String::new(),
            template: format!("template-{}", name),
            source_path: PathBuf::from(format!("{}.md", name)),
            triggers: vec![],
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            sub_files: HashMap::new(),
            scripts: vec![],
            qa_required: false,
            qa_steps: vec![],
        };
        index.insert("a".to_string(), make("a", vec!["b"]));
        index.insert("b".to_string(), make("b", vec!["a"]));

        let mut loaded = HashSet::new();
        let mut context: Vec<DiscoveredSkill> = Vec::new();
        // Should not infinite-loop
        load_skill_with_dependencies("a", &index, &mut loaded, &mut context);
        assert_eq!(context.len(), 2);
    }

    // ---- QA steps -----------------------------------------------------------

    #[test]
    fn test_parse_qa_steps() {
        let content = "---\nname: pptx\ndescription: Slides\nqa_required: true\nqa_steps:\n  - Run markitdown output.pptx\n  - Convert to images\n  - Inspect slides\n---\nBody.";
        let skill = parse_skill_file(content, &PathBuf::from("pptx.md")).unwrap();
        assert!(skill.qa_required);
        assert_eq!(skill.qa_steps.len(), 3);
        assert_eq!(skill.qa_steps[0], "Run markitdown output.pptx");
    }

    #[test]
    fn test_format_qa_block() {
        let skill = DiscoveredSkill {
            name: "pptx".to_string(),
            description: String::new(),
            template: String::new(),
            source_path: PathBuf::from("pptx.md"),
            triggers: vec![],
            dependencies: vec![],
            sub_files: HashMap::new(),
            scripts: vec![],
            qa_required: true,
            qa_steps: vec!["Run check".to_string(), "Inspect visually".to_string()],
        };
        let block = format_qa_block(&skill).unwrap();
        assert!(block.contains("Required QA"));
        assert!(block.contains("pptx"));
        assert!(block.contains("1. Run check"));
        assert!(block.contains("2. Inspect visually"));
    }

    #[test]
    fn test_format_qa_block_none_when_not_required() {
        let skill = DiscoveredSkill {
            name: "foo".to_string(),
            description: String::new(),
            template: String::new(),
            source_path: PathBuf::from("foo.md"),
            triggers: vec![],
            dependencies: vec![],
            sub_files: HashMap::new(),
            scripts: vec![],
            qa_required: false,
            qa_steps: vec![],
        };
        assert!(format_qa_block(&skill).is_none());
    }

    // ---- trigger resolution -------------------------------------------------

    #[test]
    fn test_resolve_skills_for_message() {
        let mut index: HashMap<String, DiscoveredSkill> = HashMap::new();
        let make = |name: &str, triggers: Vec<&str>| DiscoveredSkill {
            name: name.to_string(),
            description: String::new(),
            template: String::new(),
            source_path: PathBuf::from(format!("{}.md", name)),
            triggers: triggers.iter().map(|s| s.to_string()).collect(),
            dependencies: vec![],
            sub_files: HashMap::new(),
            scripts: vec![],
            qa_required: false,
            qa_steps: vec![],
        };
        index.insert("code-review".to_string(), make("code-review", vec!["review my code", "audit"]));
        index.insert("debug".to_string(), make("debug", vec!["bug", "broken", "not working"]));

        let matched = resolve_skills_for_message("can you review my code please", &index);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "code-review");

        let matched2 = resolve_skills_for_message("something is broken and has a bug", &index);
        assert_eq!(matched2.len(), 1);
        assert_eq!(matched2[0].name, "debug");

        let matched3 = resolve_skills_for_message("unrelated task", &index);
        assert!(matched3.is_empty());
    }

    // ---- scan_dir -----------------------------------------------------------

    #[test]
    fn test_scan_dir_finds_flat_skills() {
        let tmp = make_temp_dir();
        write_file(
            tmp.path(),
            "review.md",
            "---\nname: review\n---\nReview $ARGUMENTS",
        );
        write_file(tmp.path(), "debug.md", "Debug help.");
        write_file(tmp.path(), "not-md.txt", "ignored");

        let skills = scan_dir(tmp.path());
        assert_eq!(skills.len(), 2);
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"review"));
        assert!(names.contains(&"debug"));
    }

    #[test]
    fn test_scan_dir_finds_folder_based_skill() {
        let tmp = make_temp_dir();
        let skill_dir = tmp.path().join("rust-review");
        std::fs::create_dir_all(&skill_dir).unwrap();
        write_file(
            &skill_dir,
            "SKILL.md",
            "---\nname: rust-review\ndescription: Review Rust code\ntriggers:\n  - review rust\n---\nMain instructions.",
        );
        write_file(&skill_dir, "security.md", "Security checks.");
        write_file(&skill_dir, "performance.md", "Perf checks.");

        let skills = scan_dir(tmp.path());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "rust-review");
        assert!(skills[0].sub_files.contains_key("security"));
        assert!(skills[0].sub_files.contains_key("performance"));
        assert!(skills[0].triggers.contains(&"review rust".to_string()));
    }

    #[test]
    fn test_scan_dir_nonexistent_returns_empty() {
        let skills = scan_dir(Path::new("/nonexistent/path/xyz"));
        assert!(skills.is_empty());
    }

    // ---- discover_skills ----------------------------------------------------

    #[test]
    fn test_discover_from_project_dir() {
        let tmp = make_temp_dir();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        write_file(
            &skills_dir,
            "myskill.md",
            "---\nname: myskill\ndescription: Test\n---\nDo it.",
        );

        let config = crate::config::SkillsConfig::default();
        let discovered = discover_skills(tmp.path(), &config);
        assert!(discovered.contains_key("myskill"));
        assert_eq!(discovered["myskill"].description, "Test");
    }

    #[test]
    fn test_discover_from_project_commands_dir() {
        let tmp = make_temp_dir();
        let commands_dir = tmp.path().join(".mangocode").join("commands");
        std::fs::create_dir_all(&commands_dir).unwrap();
        write_file(
            &commands_dir,
            "presentation.md",
            "---\nname: presentation\ndescription: Deck skill\n---\nMake slides.",
        );

        let config = crate::config::SkillsConfig::default();
        let discovered = discover_skills(tmp.path(), &config);
        assert!(discovered.contains_key("presentation"));
        assert_eq!(discovered["presentation"].description, "Deck skill");
    }

    #[test]
    fn test_discover_extra_paths() {
        let tmp = make_temp_dir();
        let extra = make_temp_dir();
        write_file(
            extra.path(),
            "extra.md",
            "---\nname: extra\n---\nExtra skill.",
        );

        let config = crate::config::SkillsConfig {
            paths: vec![extra.path().to_str().unwrap().to_string()],
            urls: vec![],
        };
        let discovered = discover_skills(tmp.path(), &config);
        assert!(discovered.contains_key("extra"));
    }

    #[test]
    fn test_discover_deduplicates_first_wins() {
        let tmp = make_temp_dir();
        let proj_skills = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&proj_skills).unwrap();
        write_file(
            &proj_skills,
            "dup.md",
            "---\nname: dup\ndescription: project\n---\nProject.",
        );

        let extra = make_temp_dir();
        write_file(
            extra.path(),
            "dup.md",
            "---\nname: dup\ndescription: extra\n---\nExtra.",
        );

        let config = crate::config::SkillsConfig {
            paths: vec![extra.path().to_str().unwrap().to_string()],
            urls: vec![],
        };
        let discovered = discover_skills(tmp.path(), &config);
        // Project-level wins over extra path.
        assert_eq!(discovered["dup"].description, "project");
    }
}
