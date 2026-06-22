// SkillTool: execute skill prompt templates programmatically.
//
// Skills are discovered through the same core loader used by slash commands:
// project/user .mangocode skill and command dirs, project .agents dirs,
// configured skills.paths / skills.urls entries, and enabled plugin skill paths.
//
// Bundled skills are checked first before discovered skills, so they take
// precedence over same-named files.
//
// The model invokes this tool to expand a skill's prompt inline.
// Supports $ARGUMENTS, $1, and $2 placeholder substitution.
//
// Extended operations:
//   skill="list"         - enumerate all available skills
//   skill="<name>"       - load a primary skill
//   skill="<name>/<sub>" - load a named sub-file from a folder-based skill
//   args="..."           - text substituted into placeholders

use crate::bundled_skills::{expand_prompt, find_bundled_skill, user_invocable_skills};
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::skill_discovery::{
    format_qa_block, install_skill_scripts, load_skill_with_dependencies, DiscoveredSkill,
};
use mangocode_core::split_command_words;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::{debug, warn};

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
         Skills are .md files or SKILL.md folders in project, user, configured URL/path, or plugin sources. \
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
                    "description": "Arguments passed to the skill; replaces $ARGUMENTS, $1, and $2 in the template"
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

        let skill_input = params.skill.trim();
        if skill_input.eq_ignore_ascii_case("list") {
            let discovered = discover_disk_skills(ctx);
            return list_skills(&discovered);
        }

        let skill_ref = normalize_skill_ref(skill_input);
        if let Some((skill_name, sub_name)) = skill_ref.split_once('/') {
            let discovered = discover_disk_skills(ctx);
            return load_sub_file(skill_name, sub_name, &discovered, params.args.as_deref()).await;
        }

        let skill_name = skill_ref;
        debug!(skill = skill_name, "Loading skill");

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

        let discovered = discover_disk_skills(ctx);
        let skill = match find_discovered_skill(skill_name, &discovered) {
            Some(skill) => skill,
            None => {
                return ToolResult::error(format!(
                    "Skill '{}' not found. Use skill=\"list\" to see available skills.",
                    skill_name
                ));
            }
        };

        let prompt = expand_discovered_skill_prompt(
            skill,
            &discovered,
            params.args.as_deref(),
            &ctx.working_dir,
        );
        if prompt.is_empty() {
            return ToolResult::error(format!("Skill '{}' expanded to empty content.", skill_name));
        }

        ToolResult::success(prompt)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn discover_disk_skills(ctx: &ToolContext) -> HashMap<String, DiscoveredSkill> {
    let skills_config = mangocode_plugins::skills_config_with_plugin_paths(&ctx.config.skills);
    mangocode_core::discover_skills(&ctx.working_dir, &skills_config)
}

fn normalize_skill_ref(skill: &str) -> &str {
    strip_markdown_suffix(skill.trim())
        .trim_start_matches('/')
        .trim()
}

fn strip_markdown_suffix(value: &str) -> &str {
    let bytes = value.as_bytes();
    if bytes.len() >= 3 && bytes[bytes.len() - 3..].eq_ignore_ascii_case(b".md") {
        &value[..value.len() - 3]
    } else {
        value
    }
}

fn normalized_skill_name(name: &str) -> Option<&str> {
    let name = normalize_skill_ref(name);
    (!name.is_empty()).then_some(name)
}

fn normalized_skill_lookup_key(name: &str) -> Option<String> {
    normalized_skill_name(name).map(str::to_lowercase)
}

fn list_skills(discovered: &HashMap<String, DiscoveredSkill>) -> ToolResult {
    let mut lines: Vec<String> = Vec::new();
    let bundled = user_invocable_skills();
    for (name, desc) in &bundled {
        lines.push(format!("  {} - {} [bundled]", name, desc));
    }

    let bundled_names: HashSet<String> = bundled.iter().map(|(n, _)| n.to_lowercase()).collect();
    let mut disk_skills: Vec<_> = discovered
        .values()
        .filter_map(|skill| {
            let name = normalized_skill_name(&skill.name)?;
            let lookup_key = name.to_lowercase();
            if bundled_names.contains(&lookup_key) {
                return None;
            }

            Some((
                lookup_key,
                skill.name.trim().starts_with('/'),
                name.to_string(),
                skill,
            ))
        })
        .collect();
    disk_skills.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.source_path.cmp(&b.3.source_path))
    });

    let mut seen_disk_skills = HashSet::new();
    let mut visible_disk_skills = Vec::new();
    for (lookup_key, _, name, skill) in disk_skills {
        if seen_disk_skills.insert(lookup_key) {
            visible_disk_skills.push((name, skill));
        }
    }

    for (name, skill) in &visible_disk_skills {
        let mut sub_files: Vec<String> = skill.sub_files.keys().cloned().collect();
        sub_files.sort();
        if sub_files.is_empty() {
            lines.push(format!("  {} - {}", name, skill.description));
        } else {
            lines.push(format!(
                "  {} - {} [sub-files: {}]",
                name,
                skill.description,
                sub_files.join(", ")
            ));
        }
    }

    let total = bundled.len() + visible_disk_skills.len();
    if total == 0 {
        return ToolResult::success(
            "No skills found. Create skills in .mangocode/skills/ or .mangocode/commands/, \
             configure skills.paths or skills.urls, or enable a plugin that provides skills."
                .to_string(),
        );
    }

    ToolResult::success(format!(
        "Available skills ({}):\n{}",
        total,
        lines.join("\n")
    ))
}

/// Handles `skill="rust-review/security"` by reading the discovered sub-file.
async fn load_sub_file(
    skill_name: &str,
    sub_name: &str,
    discovered: &HashMap<String, DiscoveredSkill>,
    args: Option<&str>,
) -> ToolResult {
    let sub_name_clean = strip_markdown_suffix(sub_name.trim()).trim();
    let Some(skill) = find_discovered_skill(skill_name, discovered) else {
        return ToolResult::error(format!(
            "Skill '{}' not found. Use skill=\"list\" to discover available skills.",
            skill_name
        ));
    };

    let candidate = skill.sub_files.get(sub_name_clean).or_else(|| {
        skill
            .sub_files
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(sub_name_clean))
            .map(|(_, path)| path)
    });

    if let Some(path) = candidate {
        if let Ok(raw) = tokio::fs::read_to_string(path).await {
            let prompt = expand_template(&strip_frontmatter(&raw), args);
            if !prompt.is_empty() {
                return ToolResult::success(prompt);
            }
            return ToolResult::error(format!(
                "Sub-file '{}/{}' expanded to empty content.",
                skill_name, sub_name_clean
            ));
        }
    }

    ToolResult::error(format!(
        "Sub-file '{}/{}' not found. Use skill=\"list\" to discover available skills.",
        skill_name, sub_name_clean
    ))
}

fn find_discovered_skill<'a>(
    name: &str,
    discovered: &'a HashMap<String, DiscoveredSkill>,
) -> Option<&'a DiscoveredSkill> {
    let key = normalized_skill_lookup_key(name)?;
    discovered.get(key.as_str()).or_else(|| {
        discovered
            .values()
            .filter(|s| normalized_skill_lookup_key(&s.name).as_deref() == Some(key.as_str()))
            .min_by(|a, b| {
                a.name
                    .trim()
                    .starts_with('/')
                    .cmp(&b.name.trim().starts_with('/'))
                    .then(a.name.cmp(&b.name))
                    .then(a.source_path.cmp(&b.source_path))
            })
    })
}

fn expand_discovered_skill_prompt(
    skill: &DiscoveredSkill,
    discovered: &HashMap<String, DiscoveredSkill>,
    args: Option<&str>,
    working_dir: &Path,
) -> String {
    let mut loaded = HashSet::new();
    let mut context = Vec::new();
    load_skill_with_dependencies(&skill.name, discovered, &mut loaded, &mut context);
    if context.is_empty() {
        context.push(skill.clone());
    }

    let session_scripts_root = working_dir.join(".mangocode").join("skill-scripts");
    for skill in &context {
        install_skill_scripts(skill, &session_scripts_root);
    }

    let mut parts: Vec<String> = context
        .iter()
        .map(|s| {
            let skill_args = if normalized_skill_lookup_key(&s.name)
                == normalized_skill_lookup_key(&skill.name)
            {
                args
            } else {
                None
            };
            expand_template(&s.template, skill_args)
        })
        .filter(|s| !s.is_empty())
        .collect();

    let qa_blocks: Vec<String> = context
        .iter()
        .filter_map(format_qa_block)
        .map(|block| block.trim().to_string())
        .filter(|block| !block.is_empty())
        .collect();
    if !qa_blocks.is_empty() {
        parts.push(qa_blocks.join("\n\n"));
    }

    parts.join("\n\n")
}

fn expand_template(template: &str, args: Option<&str>) -> String {
    let args = args.unwrap_or("");
    let words = match split_command_words(args) {
        Ok(words) => words,
        Err(err) => {
            warn!(
                error = %err,
                "failed to parse skill template arguments; preserving raw arguments"
            );
            let args = args.trim();
            if args.is_empty() {
                Vec::new()
            } else {
                vec![args.to_string()]
            }
        }
    };
    let arg1 = words.first().map(String::as_str).unwrap_or("");
    let arg2 = words.get(1).map(String::as_str).unwrap_or("");
    template
        .replace("$ARGUMENTS", args)
        .replace("$1", arg1)
        .replace("$2", arg2)
        .trim()
        .to_string()
}

/// Remove YAML frontmatter delimited by `---` at the start of the file.
fn strip_frontmatter(content: &str) -> String {
    mangocode_core::frontmatter::strip_frontmatter(content).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_skill(name: &str, template: &str, deps: Vec<&str>) -> DiscoveredSkill {
        DiscoveredSkill {
            name: name.to_string(),
            description: String::new(),
            template: template.to_string(),
            source_path: std::path::PathBuf::from(format!("{name}.md")),
            triggers: Vec::new(),
            dependencies: deps.into_iter().map(String::from).collect(),
            sub_files: HashMap::new(),
            scripts: Vec::new(),
            qa_required: false,
            qa_steps: Vec::new(),
        }
    }

    #[test]
    fn expand_template_preserves_quoted_args_for_placeholders() {
        let prompt = expand_template(
            "first=$1 second=$2 all=$ARGUMENTS",
            Some(r#""first arg" second"#),
        );

        assert_eq!(
            prompt,
            r#"first=first arg second=second all="first arg" second"#
        );
    }

    #[test]
    fn expand_template_preserves_raw_args_for_malformed_quotes() {
        let prompt = expand_template(
            "first=$1 second=$2 all=$ARGUMENTS",
            Some(r#"foo "unterminated"#),
        );

        assert_eq!(
            prompt,
            r#"first=foo "unterminated second= all=foo "unterminated"#
        );
    }

    #[test]
    fn discovered_skill_lookup_accepts_frontmatter_name_case_insensitively() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "review-code".to_string(),
            test_skill("Review-Code", "Review $ARGUMENTS", vec![]),
        );

        let skill = find_discovered_skill("review-code", &discovered).unwrap();

        assert_eq!(skill.name, "Review-Code");
    }

    #[test]
    fn discovered_skill_lookup_normalizes_slash_prefixed_names() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "/review-code".to_string(),
            test_skill("/Review-Code", "Review $ARGUMENTS", vec![]),
        );

        let skill = find_discovered_skill("/review-code", &discovered).unwrap();

        assert_eq!(skill.name, "/Review-Code");
    }

    #[test]
    fn discovered_skill_lookup_normalizes_markdown_suffix_case_insensitively() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "review-code".to_string(),
            test_skill("Review-Code", "Review $ARGUMENTS", vec![]),
        );

        let skill = find_discovered_skill("/review-code.MD", &discovered).unwrap();

        assert_eq!(skill.name, "Review-Code");
    }

    #[test]
    fn discovered_skill_lookup_prefers_unslashed_duplicate() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "/projectreview".to_string(),
            test_skill("/ProjectReview", "Slash review", vec![]),
        );
        discovered.insert(
            "projectreview".to_string(),
            test_skill("ProjectReview", "Plain review", vec![]),
        );

        let skill = find_discovered_skill("/projectreview", &discovered).unwrap();

        assert_eq!(skill.name, "ProjectReview");
    }

    #[test]
    fn list_skills_normalizes_and_dedupes_disk_names() {
        let mut discovered = HashMap::new();
        let mut plain = test_skill("ProjectReview.MD", "Plain review", vec![]);
        plain.description = "Plain review".to_string();
        let mut slash = test_skill("/ProjectReview", "Slash review", vec![]);
        slash.description = "Slash review".to_string();
        discovered.insert("projectreview".to_string(), plain);
        discovered.insert("/projectreview".to_string(), slash);

        let result = list_skills(&discovered);

        assert!(!result.is_error);
        assert_eq!(result.content.matches("ProjectReview -").count(), 1);
        assert!(result.content.contains("ProjectReview - Plain review"));
        assert!(!result.content.contains("ProjectReview.MD - Plain review"));
        assert!(!result.content.contains("/ProjectReview - Slash review"));
    }

    #[tokio::test]
    async fn load_sub_file_normalizes_slash_prefixed_skill_name() {
        let tmp = tempfile::tempdir().unwrap();
        let sub_file = tmp.path().join("security.md");
        std::fs::write(&sub_file, "Check security $ARGUMENTS").unwrap();
        let mut skill = test_skill("/ProjectReview", "Review", vec![]);
        skill.sub_files.insert("security".to_string(), sub_file);
        let mut discovered = HashMap::new();
        discovered.insert("/projectreview".to_string(), skill);

        let result = load_sub_file("/ProjectReview", "security", &discovered, Some("now")).await;

        assert!(!result.is_error);
        assert_eq!(result.content, "Check security now");
    }

    #[tokio::test]
    async fn load_sub_file_normalizes_markdown_suffix_case_insensitively() {
        let tmp = tempfile::tempdir().unwrap();
        let sub_file = tmp.path().join("security.md");
        std::fs::write(&sub_file, "Check security").unwrap();
        let mut skill = test_skill("ProjectReview", "Review", vec![]);
        skill.sub_files.insert("security".to_string(), sub_file);
        let mut discovered = HashMap::new();
        discovered.insert("projectreview".to_string(), skill);

        let result = load_sub_file("ProjectReview", "security.MD", &discovered, None).await;

        assert!(!result.is_error);
        assert_eq!(result.content, "Check security");
    }

    #[test]
    fn discovered_skill_expansion_includes_dependencies_first() {
        let mut discovered = HashMap::new();
        discovered.insert(
            "base".to_string(),
            test_skill("base", "Base $ARGUMENTS", vec![]),
        );
        discovered.insert(
            "review".to_string(),
            test_skill("review", "Review $ARGUMENTS", vec!["base"]),
        );

        let skill = find_discovered_skill("review", &discovered).unwrap();
        let prompt = expand_discovered_skill_prompt(
            skill,
            &discovered,
            Some("diff"),
            std::path::Path::new("."),
        );

        assert_eq!(prompt, "Base\n\nReview diff");
    }

    #[test]
    fn discovered_skill_expansion_includes_qa_block() {
        let mut discovered = HashMap::new();
        let mut review = test_skill("review", "Review $ARGUMENTS", vec![]);
        review.qa_required = true;
        review.qa_steps = vec!["Run focused tests".to_string()];
        discovered.insert("review".to_string(), review.clone());

        let skill = find_discovered_skill("review", &discovered).unwrap();
        let prompt = expand_discovered_skill_prompt(
            skill,
            &discovered,
            Some("diff"),
            std::path::Path::new("."),
        );

        assert!(prompt.contains("Review diff"));
        assert!(prompt.contains("Required QA"));
        assert!(prompt.contains("Run focused tests"));
    }
}
