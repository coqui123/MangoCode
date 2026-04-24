// PowerShell tool: execute PowerShell commands (Windows-native).
//
// On Windows, PowerShell provides richer scripting than cmd.exe.
// On non-Windows platforms, attempts to use `pwsh` (PowerShell Core).
//
// Security model
// ──────────────
// Before any execution the command is passed through `classify_ps_command`.
// The resulting `PsRiskLevel` drives the permission gate:
//
//   Critical → always blocked (hard error, never executed)
//   High     → requires explicit user approval (once / session / deny)
//   Medium   → requires approval only when ctx.require_confirmation is set
//   Low      → executes directly

use crate::{session_shell_state, PermissionLevel, ShellState, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::ps_classifier::{classify_ps_command, PsRiskLevel};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::debug;

pub struct PowerShellTool;

#[derive(Debug, Deserialize)]
struct PowerShellInput {
    command: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    /// When true, Medium-risk commands also prompt for approval.
    #[serde(default)]
    require_confirmation: bool,
}

fn default_timeout() -> u64 {
    120_000
}

/// Sentinel appended to the PowerShell wrapper script. Everything printed after
/// this marker is metadata (final pwd + env dump) rather than user-visible output.
const PS_STATE_SENTINEL: &str = "__CC_PS_STATE__";

/// Parse a PowerShell snapshot block (lines after `PS_STATE_SENTINEL`) into
/// `(new_cwd, env_delta)`.
///
/// The block format is:
/// ```text
/// __CC_PS_STATE__
/// C:\some\path          ← final cwd (first line after sentinel)
/// KEY=value             ← exported env vars (remaining lines)
/// ```
fn parse_ps_state_block(lines: &[String]) -> Option<(PathBuf, HashMap<String, String>)> {
    let mut iter = lines.iter();
    let cwd_line = iter.next()?;
    let cwd = PathBuf::from(cwd_line.trim());

    let mut env_vars = HashMap::new();
    for line in iter {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].to_string();
            let val = line[eq + 1..].to_string();
            // Filter out internal PowerShell variables we don't want to persist
            if !key.starts_with('_')
                && ![
                    "PSExecutionPolicyPreference",
                    "PSSessionOption",
                    "PSDefaultParameterValues",
                    "ErrorActionPreference",
                    "VerbosePreference",
                    "DebugPreference",
                    "InformationPreference",
                    "WarningPreference",
                    "ProgressPreference",
                    "ConfirmPreference",
                    "WhatIfPreference",
                ]
                .contains(&key.as_str())
            {
                env_vars.insert(key, val);
            }
        }
    }

    Some((cwd, env_vars))
}

/// Build the PowerShell wrapper script that:
/// 1. Restores saved cwd and env vars.
/// 2. Runs the user command.
/// 3. Prints the sentinel + final pwd + env dump so we can persist state.
fn build_ps_wrapper_script(command: &str, state: &ShellState, base_cwd: &PathBuf) -> String {
    let effective_cwd = state.cwd.as_ref().unwrap_or(base_cwd);

    // Escape the cwd for PowerShell embedding
    let cwd_escaped = effective_cwd.to_string_lossy().replace('\'', "''");

    // Build env variable restoration lines
    let mut env_lines = String::new();
    for (k, v) in &state.env_vars {
        let v_escaped = v.replace('\'', "''");
        env_lines.push_str(&format!("$env:{} = '{}'\n", k, v_escaped));
    }

    format!(
        r#"
Set-Location '{}'
{}
$ErrorActionPreference = 'Continue'
& {{ {} }}
if ($LASTEXITCODE) {{ $exitCode = $LASTEXITCODE }} else {{ $exitCode = if ($?) {{ 0 }} else {{ 1 }} }}
Write-Output '{}'
Get-Location | Select-Object -ExpandProperty Path
Get-ChildItem Env: | ForEach-Object {{ "$($_.Name)=$($_.Value)" }}
exit $exitCode
"#,
        cwd_escaped, env_lines, command, PS_STATE_SENTINEL
    )
}

// ---------------------------------------------------------------------------
// Risk-label helpers (used in messages shown to the user)
// ---------------------------------------------------------------------------

fn risk_label(level: PsRiskLevel) -> &'static str {
    match level {
        PsRiskLevel::Critical => "Critical",
        PsRiskLevel::High => "High",
        PsRiskLevel::Medium => "Medium",
        PsRiskLevel::Low => "Low",
    }
}

fn risk_explanation(level: PsRiskLevel, command: &str) -> String {
    match level {
        PsRiskLevel::Critical => format!(
            "PowerShell command classified as CRITICAL risk — execution blocked.\n\
             Reason: the command contains destructive or remote-code-execution patterns.\n\
             Command: {}",
            command
        ),
        PsRiskLevel::High => format!(
            "PowerShell wants to run a HIGH-risk command:\n  {}\n\n\
             This may modify system-wide security policy, the registry (HKLM), \
             user accounts, or firewall rules.",
            command
        ),
        PsRiskLevel::Medium => format!(
            "PowerShell wants to run a MEDIUM-risk command:\n  {}\n\n\
             This may delete files, control services, or make network requests.",
            command
        ),
        PsRiskLevel::Low => String::new(), // never shown
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Tool for PowerShellTool {
    fn name(&self) -> &str {
        "PowerShell"
    }

    fn description(&self) -> &str {
        "Execute a PowerShell command. IMPORTANT: Use PowerShell syntax, not bash/shell syntax. \
         Examples: 'Set-Location C:\\path' (not 'cd'), '$env:VAR = \"value\"' (not 'export VAR=value'), \
         'Get-ChildItem' (not 'ls'), 'Test-Path' (not 'test -f'). For env vars with special characters \
         like parentheses, use: Set-Item -Path \"env:VARNAME\" -Value \"value\". The working directory and \
         environment variables persist between commands. Use for Windows-native operations, .NET APIs, \
         registry access, and Windows-specific system administration."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PowerShell command or script to execute. Must use PowerShell syntax (e.g., 'Set-Location' not 'cd', '$env:VAR=value' not 'export VAR=value', 'Get-ChildItem' not 'ls')"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what this command does"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in ms (default 120000, max 600000)"
                },
                "require_confirmation": {
                    "type": "boolean",
                    "description": "When true, Medium-risk commands also prompt for approval"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: PowerShellInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // ── Step 1: classify the command ─────────────────────────────────────
        let risk = classify_ps_command(&params.command);

        // ── Step 2: apply the risk gate ──────────────────────────────────────
        match risk {
            PsRiskLevel::Critical => {
                // Hard block — never executed regardless of permission mode.
                return ToolResult::error(risk_explanation(PsRiskLevel::Critical, &params.command));
            }

            PsRiskLevel::High => {
                // Require explicit user permission (same once/session/deny
                // pattern as BashTool: delegate to ctx.check_permission which
                // in interactive mode shows the TUI dialog).
                let desc = format!(
                    "[{} risk] {}",
                    risk_label(risk),
                    params.description.as_deref().unwrap_or(&params.command)
                );
                let details = risk_explanation(PsRiskLevel::High, &params.command);
                if let Err(e) =
                    ctx.check_permission_with_details(self.name(), &desc, &details, false)
                {
                    return ToolResult::error(e.to_string());
                }
            }

            PsRiskLevel::Medium => {
                // Only gate if the caller set require_confirmation, or if the
                // context permission mode is Default (non-bypass, non-accept).
                let needs_gate = params.require_confirmation
                    || matches!(
                        ctx.permission_mode,
                        mangocode_core::config::PermissionMode::Default
                            | mangocode_core::config::PermissionMode::Plan
                    );

                if needs_gate {
                    let desc = format!(
                        "[{} risk] {}",
                        risk_label(risk),
                        params.description.as_deref().unwrap_or(&params.command)
                    );
                    let details = risk_explanation(PsRiskLevel::Medium, &params.command);
                    if let Err(e) =
                        ctx.check_permission_with_details(self.name(), &desc, &details, false)
                    {
                        return ToolResult::error(e.to_string());
                    }
                }
            }

            PsRiskLevel::Low => {
                // Standard (non-risk-gated) permission check — honours bypass
                // and plan-mode rules, but does not show a dialog.
                let desc = params.description.as_deref().unwrap_or(&params.command);
                if let Err(e) = ctx.check_permission(self.name(), desc, false) {
                    return ToolResult::error(e.to_string());
                }
            }
        }

        // ── Step 3: execute ──────────────────────────────────────────────────
        let (exe, args) = if cfg!(windows) {
            (
                "powershell",
                vec!["-NoProfile", "-NonInteractive", "-Command"],
            )
        } else {
            ("pwsh", vec!["-NoProfile", "-NonInteractive", "-Command"])
        };

        debug!(
            command = %params.command,
            risk    = ?risk,
            "Executing PowerShell command"
        );

        let timeout_ms = params.timeout.min(600_000);
        let timeout_dur = Duration::from_millis(timeout_ms);

        // Retrieve the persistent shell state for this session.
        let shell_state_arc = session_shell_state(&ctx.session_id);

        // Build a wrapper script that restores and then captures shell state.
        let script = {
            let state = shell_state_arc.lock();
            build_ps_wrapper_script(&params.command, &state, &ctx.working_dir)
        };

        let mut child = match Command::new(exe)
            .args(&args)
            .arg(&script)
            .current_dir(&ctx.working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to spawn PowerShell: {}", e)),
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let result = tokio::time::timeout(timeout_dur, async {
            let mut stdout_lines = Vec::new();
            let mut stderr_lines = Vec::new();

            if let Some(out) = stdout {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    stdout_lines.push(line);
                }
            }
            if let Some(err) = stderr {
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    stderr_lines.push(line);
                }
            }

            let status = child.wait().await;
            (stdout_lines, stderr_lines, status)
        })
        .await;

        match result {
            Ok((stdout_lines, stderr_lines, status)) => {
                let exit_code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);

                // Split stdout into user-visible output and the state block.
                let sentinel_pos = stdout_lines
                    .iter()
                    .rposition(|l| l.trim() == PS_STATE_SENTINEL);

                let (user_lines, state_lines) = match sentinel_pos {
                    Some(pos) => (&stdout_lines[..pos], &stdout_lines[pos + 1..]),
                    None => (stdout_lines.as_slice(), &[][..]),
                };

                // Update persistent shell state from the block.
                if !state_lines.is_empty() {
                    if let Some((new_cwd, env_delta)) = parse_ps_state_block(state_lines) {
                        let mut state = shell_state_arc.lock();
                        state.cwd = Some(new_cwd);
                        // Merge (not replace) so vars set in earlier calls survive
                        for (k, v) in env_delta {
                            state.env_vars.insert(k, v);
                        }
                    }
                }

                let mut output = user_lines.join("\n");
                if !stderr_lines.is_empty() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str("STDERR:\n");
                    output.push_str(&stderr_lines.join("\n"));
                }
                if output.is_empty() {
                    output = "(no output)".to_string();
                }

                // Truncate very long output (same limit as BashTool)
                const MAX_OUTPUT_LEN: usize = 100_000;
                if output.len() > MAX_OUTPUT_LEN {
                    let half = MAX_OUTPUT_LEN / 2;
                    let start = &output[..half];
                    let end = &output[output.len() - half..];
                    output = format!(
                        "{}\n\n... ({} characters truncated) ...\n\n{}",
                        start,
                        output.len() - MAX_OUTPUT_LEN,
                        end
                    );
                }

                if exit_code != 0 {
                    ToolResult::error(format!(
                        "PowerShell exited with code {}\n{}",
                        exit_code, output
                    ))
                } else {
                    ToolResult::success(output)
                }
            }
            Err(_) => {
                let _ = child.kill().await;
                ToolResult::error(format!(
                    "PowerShell command timed out after {}ms",
                    timeout_ms
                ))
            }
        }
    }
}
