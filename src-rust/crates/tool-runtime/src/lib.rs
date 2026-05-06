use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandlerKind {
    BuiltIn,
    Mcp,
    Plugin,
    Alias,
    Unavailable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallSource {
    Model,
    Agent,
    User,
    Hook,
    Proactive,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxPreference {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputPolicy {
    #[default]
    Auto,
    Raw,
    Summary,
    Structured,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    NotRequired,
    Allowed,
    Denied,
    Escalated,
    Cached,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolErrorKind {
    UnknownTool,
    InvalidInput,
    PermissionDenied,
    SandboxDenied,
    NetworkDenied,
    Timeout,
    Cancelled,
    ExecutionFailed,
    HookBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalKey {
    pub kind: String,
    pub value: String,
}

impl ApprovalKey {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCapabilities {
    pub mutating: bool,
    pub parallel_safe: bool,
    pub affected_paths: Vec<String>,
    pub network_targets: Vec<String>,
    pub approval_keys: Vec<ApprovalKey>,
    pub aliases: Vec<String>,
    pub sandbox_preference: SandboxPreference,
    pub supports_cancellation: bool,
    pub output_policy: OutputPolicy,
}

impl ToolCapabilities {
    pub fn read_only() -> Self {
        Self {
            mutating: false,
            parallel_safe: true,
            affected_paths: Vec::new(),
            network_targets: Vec::new(),
            approval_keys: Vec::new(),
            aliases: Vec::new(),
            sandbox_preference: SandboxPreference::ReadOnly,
            supports_cancellation: false,
            output_policy: OutputPolicy::Auto,
        }
    }

    pub fn mutating() -> Self {
        Self {
            mutating: true,
            parallel_safe: false,
            affected_paths: Vec::new(),
            network_targets: Vec::new(),
            approval_keys: Vec::new(),
            aliases: Vec::new(),
            sandbox_preference: SandboxPreference::WorkspaceWrite,
            supports_cancellation: false,
            output_policy: OutputPolicy::Auto,
        }
    }

    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }

    pub fn with_affected_paths(mut self, paths: Vec<String>) -> Self {
        self.affected_paths = paths;
        self
    }

    pub fn with_network_targets(mut self, targets: Vec<String>) -> Self {
        self.network_targets = targets;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub handler_kind: ToolHandlerKind,
    pub aliases: Vec<String>,
    pub capabilities: ToolCapabilities,
}

impl ToolSpec {
    pub fn canonical_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnavailableTool {
    pub requested_name: String,
    pub reason: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolRegistryPlan {
    pub specs: Vec<ToolSpec>,
    pub unavailable: Vec<UnavailableTool>,
    #[serde(skip)]
    alias_to_name: HashMap<String, String>,
}

impl ToolRegistryPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_specs(specs: Vec<ToolSpec>) -> Self {
        let mut plan = Self {
            specs,
            unavailable: Vec::new(),
            alias_to_name: HashMap::new(),
        };
        plan.rebuild_aliases();
        plan
    }

    pub fn add_spec(&mut self, spec: ToolSpec) {
        self.specs.push(spec);
        self.rebuild_aliases();
    }

    pub fn add_unavailable(&mut self, unavailable: UnavailableTool) {
        self.unavailable.push(unavailable);
    }

    pub fn rebuild_aliases(&mut self) {
        self.alias_to_name.clear();
        for spec in &self.specs {
            for name in spec.canonical_names() {
                self.alias_to_name
                    .entry(normalize_tool_name(name))
                    .or_insert_with(|| spec.name.clone());
            }
        }
    }

    pub fn canonical_name(&self, requested: &str) -> Option<&str> {
        self.alias_to_name
            .get(&normalize_tool_name(requested))
            .map(String::as_str)
    }

    pub fn spec_for(&self, requested: &str) -> Option<&ToolSpec> {
        let canonical = self.canonical_name(requested)?;
        self.specs.iter().find(|spec| spec.name == canonical)
    }

    pub fn suggestions_for(&self, requested: &str, limit: usize) -> Vec<String> {
        let requested = normalize_tool_name(requested);
        let mut scored = Vec::new();
        for spec in &self.specs {
            let mut best = score_name(&requested, &normalize_tool_name(&spec.name));
            for alias in &spec.aliases {
                best = best.max(score_name(&requested, &normalize_tool_name(alias)));
            }
            if best > 0 {
                scored.push((best, spec.name.clone()));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.dedup_by(|a, b| a.1 == b.1);
        scored
            .into_iter()
            .take(limit)
            .map(|(_, name)| name)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    pub requested_name: String,
    pub canonical_name: Option<String>,
    pub input: Value,
    pub source: ToolCallSource,
    pub parent_tool_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub kind: String,
    pub path: Option<String>,
    pub url: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutputEnvelope {
    pub success: bool,
    pub text: String,
    pub metadata: Option<Value>,
    pub duration_ms: Option<u64>,
    pub artifacts: Vec<ArtifactRef>,
    pub affected_paths: Vec<String>,
    pub raw_log_path: Option<String>,
    pub error_kind: Option<ToolErrorKind>,
}

impl ToolOutputEnvelope {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            success: true,
            text: text.into(),
            metadata: None,
            duration_ms: None,
            artifacts: Vec::new(),
            affected_paths: Vec::new(),
            raw_log_path: None,
            error_kind: None,
        }
    }

    pub fn error(text: impl Into<String>, error_kind: ToolErrorKind) -> Self {
        Self {
            success: false,
            text: text.into(),
            metadata: None,
            duration_ms: None,
            artifacts: Vec::new(),
            affected_paths: Vec::new(),
            raw_log_path: None,
            error_kind: Some(error_kind),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDispatchTrace {
    pub invocation: ToolInvocation,
    pub requester: Option<String>,
    pub tool_source: ToolHandlerKind,
    pub input_preview: String,
    pub approval_decision: ApprovalDecision,
    pub sandbox_policy: SandboxPreference,
    pub network_policy: Option<String>,
    pub duration_ms: Option<u64>,
    pub success: bool,
    pub affected_paths: Vec<String>,
    pub raw_log_path: Option<String>,
    pub output_preview: String,
    pub error_kind: Option<ToolErrorKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPlan {
    pub name: String,
    pub capabilities: ToolCapabilities,
    pub blocked: bool,
}

impl ToolCallPlan {
    pub fn new(name: impl Into<String>, capabilities: ToolCapabilities) -> Self {
        Self {
            name: name.into(),
            capabilities,
            blocked: false,
        }
    }

    pub fn blocked(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            capabilities: ToolCapabilities::mutating(),
            blocked: true,
        }
    }
}

pub fn plan_execution_batches(calls: &[ToolCallPlan]) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current_parallel = Vec::new();

    for (idx, call) in calls.iter().enumerate() {
        if call.blocked || call.capabilities.mutating || !call.capabilities.parallel_safe {
            if !current_parallel.is_empty() {
                batches.push(std::mem::take(&mut current_parallel));
            }
            batches.push(vec![idx]);
        } else {
            current_parallel.push(idx);
        }
    }

    if !current_parallel.is_empty() {
        batches.push(current_parallel);
    }

    batches
}

pub fn preview_json(value: &Value, max_chars: usize) -> String {
    preview_text(&value.to_string(), max_chars)
}

pub fn preview_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut preview = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    preview.push_str("...");
    preview
}

fn normalize_tool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

fn score_name(requested: &str, candidate: &str) -> usize {
    if requested == candidate {
        return 10_000;
    }
    if candidate.contains(requested) {
        return 5_000usize.saturating_sub(candidate.len().saturating_sub(requested.len()));
    }

    let mut chars = candidate.chars();
    let mut matched = 0usize;
    for ch in requested.chars() {
        if chars.any(|candidate_ch| candidate_ch == ch) {
            matched += 1;
        }
    }
    if requested.is_empty() || matched < requested.chars().count().saturating_sub(1) {
        0
    } else {
        matched
    }
}

pub fn dedupe_strings(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(normalize_tool_name(value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, aliases: &[&str], capabilities: ToolCapabilities) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            input_schema: Value::Null,
            handler_kind: ToolHandlerKind::BuiltIn,
            aliases: aliases.iter().map(|alias| alias.to_string()).collect(),
            capabilities,
        }
    }

    #[test]
    fn resolves_aliases_to_canonical_names() {
        let plan = ToolRegistryPlan::from_specs(vec![spec(
            "Bash",
            &["shell", "container.exec"],
            ToolCapabilities::mutating(),
        )]);

        assert_eq!(plan.canonical_name("shell"), Some("Bash"));
        assert_eq!(plan.canonical_name("container-exec"), Some("Bash"));
        assert_eq!(plan.spec_for("SHELL").unwrap().name, "Bash");
    }

    #[test]
    fn suggests_close_tool_names() {
        let plan = ToolRegistryPlan::from_specs(vec![
            spec(
                "ToolSearch",
                &["tool_search"],
                ToolCapabilities::read_only(),
            ),
            spec("ApplyPatch", &["apply_patch"], ToolCapabilities::mutating()),
        ]);

        assert_eq!(plan.suggestions_for("apply", 1), vec!["ApplyPatch"]);
    }

    #[test]
    fn batches_parallel_reads_and_serializes_mutations() {
        let calls = vec![
            ToolCallPlan::new("Read", ToolCapabilities::read_only()),
            ToolCallPlan::new("Grep", ToolCapabilities::read_only()),
            ToolCallPlan::new("Write", ToolCapabilities::mutating()),
            ToolCallPlan::new("Glob", ToolCapabilities::read_only()),
        ];

        assert_eq!(
            plan_execution_batches(&calls),
            vec![vec![0, 1], vec![2], vec![3]]
        );
    }
}
