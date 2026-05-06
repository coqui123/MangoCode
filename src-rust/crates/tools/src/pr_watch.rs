#![cfg_attr(not(feature = "tool-pr-watch"), allow(dead_code, unused_imports))]

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "tool-pr-watch")]
pub struct PrWatchTool;

#[derive(Debug, Deserialize)]
struct PrWatchInput {
    action: String,
    #[serde(default)]
    pr: Option<String>,
    #[serde(default)]
    auto_analyze: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WatchedPr {
    pr: u64,
    #[serde(default)]
    auto_analyze: bool,
    #[serde(default)]
    last_known_status: Option<String>,
    #[serde(default)]
    last_failure_key: Option<String>,
    #[serde(default)]
    last_checked_at: Option<String>,
}

#[derive(Debug, Clone)]
struct CheckStatus {
    name: String,
    state: String,
    conclusion: String,
}

#[derive(Debug, Clone)]
struct CheckRun {
    name: String,
    link: Option<String>,
}

#[derive(Debug, Clone)]
struct CheckSummary {
    checks: Vec<CheckStatus>,
}

impl CheckSummary {
    fn failed(&self) -> Vec<&CheckStatus> {
        self.checks.iter().filter(|c| is_failed(c)).collect()
    }

    fn status_line(&self) -> String {
        let total = self.checks.len();
        let failed = self.failed().len();
        let in_progress = self
            .checks
            .iter()
            .filter(|c| {
                let s = c.state.to_ascii_lowercase();
                s == "in_progress" || s == "queued" || s == "pending" || s == "waiting"
            })
            .count();
        let passed = total.saturating_sub(failed + in_progress);
        format!(
            "total: {}, passed: {}, failed: {}, in_progress: {}",
            total, passed, failed, in_progress
        )
    }

    fn failure_key(&self) -> Option<String> {
        let mut failed = self
            .failed()
            .into_iter()
            .map(|c| {
                format!(
                    "{}:{}:{}",
                    c.name,
                    c.state.to_ascii_lowercase(),
                    c.conclusion.to_ascii_lowercase()
                )
            })
            .collect::<Vec<_>>();
        if failed.is_empty() {
            return None;
        }
        failed.sort();
        Some(failed.join("|"))
    }
}

#[cfg(feature = "tool-pr-watch")]
#[async_trait]
impl Tool for PrWatchTool {
    fn name(&self) -> &str {
        "PrWatch"
    }

    fn description(&self) -> &str {
        "Watch a GitHub pull request for CI status changes. Alerts when checks fail and can auto-analyze failures."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "enum": ["watch", "unwatch", "list", "check"] },
                "pr": { "type": "string", "description": "PR number or URL" },
                "auto_analyze": { "type": "boolean", "default": false, "description": "Automatically analyze CI failures" }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: PrWatchInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match params.action.as_str() {
            "watch" => watch_pr(params, ctx).await,
            "unwatch" => unwatch_pr(params).await,
            "list" => list_watched_prs().await,
            "check" => check_pr_action(params, ctx).await,
            other => ToolResult::error(format!("Unknown action '{}'.", other)),
        }
    }
}

fn watched_prs_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mangocode").join("watched_prs.json"))
}

async fn load_watched_prs() -> Vec<WatchedPr> {
    let Some(path) = watched_prs_path() else {
        return Vec::new();
    };

    let text = match tokio::fs::read_to_string(path).await {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    serde_json::from_str(&text).unwrap_or_default()
}

async fn save_watched_prs(items: &[WatchedPr]) -> Result<(), String> {
    let path = watched_prs_path().ok_or_else(|| "Cannot determine home directory".to_string())?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create {}: {}", parent.display(), e))?;
    }

    let data = serde_json::to_string_pretty(items)
        .map_err(|e| format!("Failed to serialize watched PRs: {}", e))?;
    tokio::fs::write(&path, data)
        .await
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))
}

fn parse_pr_number(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if let Ok(n) = trimmed.parse::<u64>() {
        return Some(n);
    }

    let pull_re = Regex::new(r"/pull/(\d+)").ok()?;
    if let Some(cap) = pull_re.captures(trimmed) {
        return cap.get(1)?.as_str().parse::<u64>().ok();
    }

    let num_re = Regex::new(r"(\d+)").ok()?;
    num_re
        .captures_iter(trimmed)
        .last()
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
}

async fn watch_pr(params: PrWatchInput, _ctx: &ToolContext) -> ToolResult {
    let Some(pr_raw) = params.pr.as_deref() else {
        return ToolResult::error("'pr' is required for action=watch");
    };
    let Some(pr) = parse_pr_number(pr_raw) else {
        return ToolResult::error(format!("Could not parse PR number from '{}'.", pr_raw));
    };

    let mut items = load_watched_prs().await;
    let auto = params.auto_analyze.unwrap_or(false);

    if let Some(existing) = items.iter_mut().find(|w| w.pr == pr) {
        existing.auto_analyze = auto;
        if let Err(e) = save_watched_prs(&items).await {
            return ToolResult::error(e);
        }
        return ToolResult::success(format!(
            "Updated watch for PR #{} (auto_analyze={}).",
            pr, auto
        ));
    }

    items.push(WatchedPr {
        pr,
        auto_analyze: auto,
        last_known_status: None,
        last_failure_key: None,
        last_checked_at: None,
    });

    if let Err(e) = save_watched_prs(&items).await {
        return ToolResult::error(e);
    }

    ToolResult::success(format!("Watching PR #{} (auto_analyze={}).", pr, auto))
}

async fn unwatch_pr(params: PrWatchInput) -> ToolResult {
    let Some(pr_raw) = params.pr.as_deref() else {
        return ToolResult::error("'pr' is required for action=unwatch");
    };
    let Some(pr) = parse_pr_number(pr_raw) else {
        return ToolResult::error(format!("Could not parse PR number from '{}'.", pr_raw));
    };

    let mut items = load_watched_prs().await;
    let before = items.len();
    items.retain(|w| w.pr != pr);

    if items.len() == before {
        return ToolResult::error(format!("PR #{} is not currently watched.", pr));
    }

    if let Err(e) = save_watched_prs(&items).await {
        return ToolResult::error(e);
    }

    ToolResult::success(format!("Stopped watching PR #{}.", pr))
}

async fn list_watched_prs() -> ToolResult {
    let mut items = load_watched_prs().await;
    if items.is_empty() {
        return ToolResult::success("No watched PRs.");
    }

    items.sort_by_key(|w| w.pr);
    let mut out = String::new();
    out.push_str(&format!("Watched PRs ({}):\n", items.len()));
    for w in items {
        out.push_str(&format!(
            "- #{} | auto_analyze={} | status={} | checked={}\n",
            w.pr,
            w.auto_analyze,
            w.last_known_status
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            w.last_checked_at
                .clone()
                .unwrap_or_else(|| "never".to_string()),
        ));
    }

    ToolResult::success(out.trim_end().to_string())
}

async fn check_pr_action(params: PrWatchInput, ctx: &ToolContext) -> ToolResult {
    let Some(pr_raw) = params.pr.as_deref() else {
        return ToolResult::error("'pr' is required for action=check");
    };
    let Some(pr) = parse_pr_number(pr_raw) else {
        return ToolResult::error(format!("Could not parse PR number from '{}'.", pr_raw));
    };

    let summary = match query_pr_checks(&ctx.working_dir, pr).await {
        Ok(s) => s,
        Err(e) => return ToolResult::error(e),
    };

    let mut out = vec![
        format!("PR #{} checks", pr),
        format!("Status: {}", summary.status_line()),
    ];

    for check in &summary.checks {
        out.push(format!(
            "- {} | state={} | conclusion={}",
            check.name, check.state, check.conclusion
        ));
    }

    let should_analyze = params.auto_analyze.unwrap_or(false);
    if should_analyze {
        let failed = summary.failed();
        if !failed.is_empty() {
            out.push(String::new());
            out.push("Auto-analysis:".to_string());
            match analyze_failed_checks(&ctx.working_dir, &ctx.config, pr, &failed).await {
                Ok(analyses) => {
                    for a in analyses {
                        out.push(a);
                    }
                }
                Err(e) => out.push(format!("- Analysis failed: {}", e)),
            }
        }
    }

    // Update tracked status if this PR is in the watch list.
    let mut watched = load_watched_prs().await;
    if let Some(item) = watched.iter_mut().find(|w| w.pr == pr) {
        item.last_known_status = Some(summary.status_line());
        item.last_failure_key = summary.failure_key();
        item.last_checked_at = Some(Utc::now().to_rfc3339());
        let _ = save_watched_prs(&watched).await;
    }

    ToolResult::success(out.join("\n"))
}

async fn query_pr_checks(working_dir: &Path, pr: u64) -> Result<CheckSummary, String> {
    // Requirement: gh pr checks <number> --json name,state,conclusion
    let checks_json = run_gh_json(
        working_dir,
        &[
            "pr",
            "checks",
            &pr.to_string(),
            "--json",
            "name,state,conclusion",
        ],
    )
    .await?;

    let arr = checks_json
        .as_array()
        .ok_or_else(|| "Unexpected output from 'gh pr checks'".to_string())?;

    let mut checks = Vec::with_capacity(arr.len());
    for item in arr {
        checks.push(CheckStatus {
            name: item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("(unnamed)")
                .to_string(),
            state: item
                .get("state")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            conclusion: item
                .get("conclusion")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        });
    }

    Ok(CheckSummary { checks })
}

fn is_failed(check: &CheckStatus) -> bool {
    let state = check.state.to_ascii_lowercase();
    let conclusion = check.conclusion.to_ascii_lowercase();

    [
        "fail",
        "failure",
        "timed_out",
        "cancelled",
        "action_required",
        "error",
    ]
    .iter()
    .any(|needle| state.contains(needle) || conclusion.contains(needle))
}

async fn query_check_runs(
    working_dir: &Path,
    pr: u64,
) -> Result<HashMap<String, CheckRun>, String> {
    let checks_json = run_gh_json(
        working_dir,
        &[
            "pr",
            "checks",
            &pr.to_string(),
            "--json",
            "name,state,conclusion,link",
        ],
    )
    .await?;

    let arr = checks_json
        .as_array()
        .ok_or_else(|| "Unexpected output from 'gh pr checks --json ... link'".to_string())?;
    let mut out = HashMap::new();
    for item in arr {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(unnamed)")
            .to_string();
        let link = item
            .get("link")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.insert(name.clone(), CheckRun { name, link });
    }
    Ok(out)
}

fn extract_run_id(link: &str) -> Option<u64> {
    let re = Regex::new(r"/runs/(\d+)").ok()?;
    let cap = re.captures(link)?;
    cap.get(1)?.as_str().parse::<u64>().ok()
}

fn tail_lines(input: &str, max_lines: usize) -> String {
    let lines = input.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return input.to_string();
    }
    lines[lines.len() - max_lines..].join("\n")
}

async fn analyze_failed_checks(
    working_dir: &Path,
    config: &mangocode_core::config::Config,
    pr: u64,
    failed: &[&CheckStatus],
) -> Result<Vec<String>, String> {
    let by_name = query_check_runs(working_dir, pr).await?;
    let mut analyses = Vec::new();

    for check in failed {
        let Some(run) = by_name.get(&check.name) else {
            analyses.push(format!(
                "- {}: unable to locate run link for failed check.",
                check.name
            ));
            continue;
        };

        let Some(link) = run.link.as_deref() else {
            analyses.push(format!("- {}: missing run link.", run.name));
            continue;
        };

        let Some(run_id) = extract_run_id(link) else {
            analyses.push(format!(
                "- {}: could not parse run ID from {}",
                run.name, link
            ));
            continue;
        };

        let log = run_gh_text(
            working_dir,
            &["run", "view", &run_id.to_string(), "--log-failed"],
        )
        .await?;
        let log = tail_lines(&log, 2000);

        let analysis = match analyze_log_with_model(config, pr, &check.name, &log).await {
            Ok(t) => t,
            Err(e) => format!("analysis unavailable: {}", e),
        };

        analyses.push(format!(
            "- {} (run {}):\n{}",
            check.name,
            run_id,
            analysis.trim()
        ));
    }

    Ok(analyses)
}

async fn analyze_log_with_model(
    config: &mangocode_core::config::Config,
    pr: u64,
    check_name: &str,
    log: &str,
) -> Result<String, String> {
    let client = mangocode_api::AnthropicClient::from_config(config)
        .map_err(|e| format!("Cannot create model client: {}", e))?;

    let prompt = format!(
        "Analyze this CI failure. What failed, why, and what's the fix?\n\nPR: #{}\nCheck: {}\n\nFailed log (truncated to <=2000 lines):\n{}",
        pr, check_name, log
    );

    let request = mangocode_api::CreateMessageRequest::builder(config.effective_model(), 1200)
        .messages(vec![mangocode_api::ApiMessage {
            role: "user".to_string(),
            content: Value::String(prompt),
        }])
        .build();

    let response = client
        .create_message(request)
        .await
        .map_err(|e| format!("Model call failed: {}", e))?;

    for block in response.content {
        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                return Ok(text.to_string());
            }
        }
    }

    Err("Model response had no text block".to_string())
}

async fn run_gh_json(working_dir: &Path, args: &[&str]) -> Result<Value, String> {
    let output = tokio::process::Command::new("gh")
        .args(args)
        .current_dir(working_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to run gh {}: {}", args.join(" "), e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let out = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "gh {} failed: {}{}",
            args.join(" "),
            err.trim(),
            if out.trim().is_empty() {
                String::new()
            } else {
                format!(" | {}", out.trim())
            }
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    serde_json::from_str(&text).map_err(|e| format!("Failed to parse gh output as JSON: {}", e))
}

async fn run_gh_text(working_dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = tokio::process::Command::new("gh")
        .args(args)
        .current_dir(working_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to run gh {}: {}", args.join(" "), e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("gh {} failed: {}", args.join(" "), err.trim()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub async fn heartbeat_scan_watched_prs(
    working_dir: &Path,
    config: &mangocode_core::config::Config,
) -> Vec<String> {
    let mut watched = load_watched_prs().await;
    if watched.is_empty() {
        return Vec::new();
    }

    let mut alerts = Vec::new();

    for item in &mut watched {
        let summary = match query_pr_checks(working_dir, item.pr).await {
            Ok(s) => s,
            Err(e) => {
                alerts.push(format!("PR #{} check failed: {}", item.pr, e));
                continue;
            }
        };

        let failure_key = summary.failure_key();
        let had_new_failure = failure_key.is_some() && failure_key != item.last_failure_key;

        item.last_known_status = Some(summary.status_line());
        item.last_failure_key = failure_key.clone();
        item.last_checked_at = Some(Utc::now().to_rfc3339());

        if had_new_failure {
            let mut alert = format!(
                "Watched PR #{} has new failing checks ({}).",
                item.pr,
                summary.status_line()
            );

            if item.auto_analyze {
                let failed = summary.failed();
                match analyze_failed_checks(working_dir, config, item.pr, &failed).await {
                    Ok(lines) if !lines.is_empty() => {
                        alert.push_str("\n\nAuto-analysis:\n");
                        alert.push_str(&lines.join("\n"));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        alert.push_str(&format!("\n\nAuto-analysis failed: {}", e));
                    }
                }
            }

            alerts.push(alert);
        }
    }

    let _ = save_watched_prs(&watched).await;
    alerts
}
