// SkillTool: execute user-defined skill (prompt template) files programmatically.
//
// Skills are Markdown files stored in:
//   <project>/.mangocode/commands/<name>.md   (flat file)
//   <project>/.mangocode/skills/<name>/SKILL.md  (folder-based, with sub-files)
//   ~/.mangocode/commands/<name>.md
//
// Bundled skills (defined in bundled_skills.rs) are checked first before the
// disk directories, so they take precedence over same-named .md files.
//
// The model invokes this tool to expand a skill's prompt inline.
// Supports $ARGUMENTS placeholder substitution.
//
// Extended operations:
//   skill="list"              — enumerate all available skills
//   skill="<name>"            — load primary SKILL.md (or flat .md)
//   skill="<name>/<sub>"      — load a named sub-file from a folder-based skill
//                               e.g. skill="rust-review/security"
//   args="..."                — text substituted for $ARGUMENTS in the template

use crate::bundled_skills::{expand_prompt, find_bundled_skill, user_invocable_skills};
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use tracing::debug;

pub struct SkillTool;

#[derive(Debug, Deserialize)]
struct SkillInput {
    skill: String,
    #[serde(default)]
    args: Option<String>,
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "Skill"
    }

    fn description(&self) -> &str {
        "Execute a skill (custom prompt template) by name. \
         Skills are .md files in .mangocode/commands/ or ~/.mangocode/commands/. \
         Use skill=\"list\" to discover available skills. \
         The expanded skill prompt is returned for you to act on."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Skill name (without .md extension), or \"list\" to enumerate skills"
                },
                "args": {
                    "type": "string",
                    "description": "Arguments passed to the skill — replaces $ARGUMENTS in the template"
                }
            },
            "required": ["skill"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: SkillInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let dirs = skill_search_dirs(ctx);

        if params.skill == "list" {
            return list_skills(&dirs).await;
        }

        // Handle sub-file requests: skill="rust-review/security"
        let skill_ref = params.skill.trim_end_matches(".md");
        if let Some((skill_name, sub_name)) = skill_ref.split_once('/') {
            return load_sub_file(skill_name, sub_name, &dirs, params.args.as_deref()).await;
        }

        let skill_name = skill_ref;
        debug!(skill = skill_name, "Loading skill");

        // Check bundled skills first — they take precedence over disk files.
        if let Some(bundled) = find_bundled_skill(skill_name) {
            let args = params.args.as_deref().unwrap_or("");
            let prompt = expand_prompt(bundled, args);
            let prompt = prompt.trim().to_string();
            if prompt.is_empty() {
                return ToolResult::error(format!(
                    "Bundled skill '{}' expanded to empty content.",
                    skill_name
                ));
            }
            return ToolResult::success(prompt);
        }

        let raw = match find_and_read_skill(skill_name, &dirs).await {
            Some(c) => c,
            None => {
                return ToolResult::error(format!(
                    "Skill '{}' not found. Use skill=\"list\" to see available skills.",
                    skill_name
                ));
            }
        };

        // Strip YAML frontmatter if present (--- ... ---)
        let content = strip_frontmatter(&raw);

        // Substitute $ARGUMENTS
        let prompt = if let Some(args) = &params.args {
            content.replace("$ARGUMENTS", args)
        } else {
            content.replace("$ARGUMENTS", "")
        };

        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return ToolResult::error(format!("Skill '{}' expanded to empty content.", skill_name));
        }

        ToolResult::success(prompt)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return all directories that should be searched for skill files, in priority order.
///
/// Both `skills/` and `commands/` are included since MangoCode supports both
/// naming conventions. Folder-based skills live inside the `skills/` dirs.
fn skill_search_dirs(ctx: &ToolContext) -> Vec<PathBuf> {
    let mut dirs = vec![
        ctx.working_dir.join(".mangocode").join("commands"),
        ctx.working_dir.join(".mangocode").join("skills"),
        ctx.working_dir.join(".agents").join("skills"),
    ];
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".mangocode").join("commands"));
        dirs.push(home.join(".mangocode").join("skills"));
    }
    dirs
}

async fn list_skills(dirs: &[PathBuf]) -> ToolResult {
    // Start with the bundled skills.
    let mut lines: Vec<String> = Vec::new();
    let bundled = user_invocable_skills();
    for (name, desc) in &bundled {
        lines.push(format!("  {} — {} [bundled]", name, desc));
    }
    let bundled_names: Vec<&str> = bundled.iter().map(|(n, _)| *n).collect();

    // Then add disk skills (flat .md files + folder-based skills), skipping bundled names.
    let mut disk_skills: Vec<(String, PathBuf, Vec<String>)> = Vec::new();
    for dir in dirs {
        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();

                if path.is_dir() {
                    // Folder-based skill: look for SKILL.md
                    let skill_md = path.join("SKILL.md");
                    if skill_md.is_file() {
                        let dir_name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        if dir_name.is_empty() || bundled_names.contains(&dir_name.as_str()) {
                            continue;
                        }
                        if disk_skills.iter().any(|(n, _, _)| n == &dir_name) {
                            continue;
                        }
                        // Collect sub-file names
                        let mut sub_names: Vec<String> = Vec::new();
                        if let Ok(mut sub) = tokio::fs::read_dir(&path).await {
                            while let Ok(Some(se)) = sub.next_entry().await {
                                let sp = se.path();
                                if sp.is_file()
                                    && sp.extension().is_some_and(|e| e == "md")
                                    && sp.file_name().and_then(|s| s.to_str()) != Some("SKILL.md")
                                {
                                    if let Some(stem) = sp.file_stem().and_then(|s| s.to_str()) {
                                        sub_names.push(stem.to_string());
                                    }
                                }
                            }
                        }
                        sub_names.sort();
                        disk_skills.push((dir_name, skill_md, sub_names));
                    }
                } else if path.extension().is_some_and(|e| e == "md") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        let name = stem.to_string();
                        // Deduplicate: project-level shadows user-level;
                        // bundled skills shadow everything.
                        if !disk_skills.iter().any(|(n, _, _)| n == &name)
                            && !bundled_names.contains(&name.as_str())
                        {
                            disk_skills.push((name, path, vec![]));
                        }
                    }
                }
            }
        }
    }

    disk_skills.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, path, sub_files) in &disk_skills {
        let desc = read_skill_description(path).await;
        if sub_files.is_empty() {
            lines.push(format!("  {} — {}", name, desc));
        } else {
            lines.push(format!(
                "  {} — {} [sub-files: {}]",
                name,
                desc,
                sub_files.join(", ")
            ));
        }
    }

    let total = bundled.len() + disk_skills.len();
    if total == 0 {
        return ToolResult::success(
            "No skills found. Create .md files in .mangocode/commands/ to define skills.\n\
             Example: .mangocode/commands/review.md"
                .to_string(),
        );
    }

    ToolResult::success(format!(
        "Available skills ({}):\n{}",
        total,
        lines.join("\n")
    ))
}

/// Locate and read a skill's primary content (SKILL.md or flat .md file).
///
/// Search order per directory:
///  1. Flat file: `<dir>/<name>.md`
///  2. Folder-based: `<dir>/<name>/SKILL.md`
async fn find_and_read_skill(name: &str, dirs: &[PathBuf]) -> Option<String> {
    for dir in dirs {
        // Flat file
        let flat = dir.join(format!("{}.md", name));
        if let Ok(content) = tokio::fs::read_to_string(&flat).await {
            return Some(content);
        }
        // Folder-based skill
        let folder_skill = dir.join(name).join("SKILL.md");
        if let Ok(content) = tokio::fs::read_to_string(&folder_skill).await {
            return Some(content);
        }
    }
    None
}

/// Load a named sub-file from a folder-based skill.
///
/// Handles `skill="rust-review/security"` — reads `security.md` from the
/// `rust-review/` skill directory.
async fn load_sub_file(
    skill_name: &str,
    sub_name: &str,
    dirs: &[PathBuf],
    args: Option<&str>,
) -> ToolResult {
    let sub_name_clean = sub_name.trim_end_matches(".md");
    for dir in dirs {
        let candidate = dir.join(skill_name).join(format!("{}.md", sub_name_clean));
        if let Ok(raw) = tokio::fs::read_to_string(&candidate).await {
            let content = strip_frontmatter(&raw);
            let prompt = if let Some(a) = args {
                content.replace("$ARGUMENTS", a)
            } else {
                content.replace("$ARGUMENTS", "")
            };
            let prompt = prompt.trim().to_string();
            if prompt.is_empty() {
                return ToolResult::error(format!(
                    "Sub-file '{}/{}' expanded to empty content.",
                    skill_name, sub_name_clean
                ));
            }
            return ToolResult::success(prompt);
        }
    }
    ToolResult::error(format!(
        "Sub-file '{}/{}' not found. Use skill=\"list\" to discover available skills.",
        skill_name, sub_name_clean
    ))
}

async fn read_skill_description(path: &std::path::Path) -> String {
    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return "(no description)".to_string();
    };
    let body = strip_frontmatter(&content);
    // First non-empty, non-heading line
    for line in body.lines() {
        let t = line.trim().trim_start_matches('#').trim();
        if !t.is_empty() {
            let truncated = if t.len() > 80 { &t[..80] } else { t };
            return truncated.to_string();
        }
    }
    "(no description)".to_string()
}

/// Remove YAML frontmatter delimited by `---` at the start of the file.
fn strip_frontmatter(content: &str) -> String {
    if let Some(after_open) = content.strip_prefix("---") {
        // Find closing ---
        if let Some(close_pos) = after_open.find("\n---") {
            // Skip past the closing delimiter and any leading newline
            let rest = &after_open[close_pos + 4..];
            return rest.trim_start_matches('\n').to_string();
        }
    }
    content.to_string()
}
