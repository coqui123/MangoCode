# Tool System (Rust)

Source files:
- src-rust/crates/tools/src/lib.rs
- src-rust/crates/tools/src/*.rs

## Tool framework

Tool crate defines:
- Tool trait
- ToolContext
- ToolResult
- PermissionLevel

## Context and permissions

ToolContext carries:
- working directory
- permission mode/handler
- cost tracker
- session id
- file history
- mcp manager
- config

Permission checks are centralized through permission handler requests.

## Implemented tool families

Filesystem:
- file_read, file_write, file_edit, batch_edit, apply_patch, notebook_edit

Shell/process:
- bash, pty_bash, powershell, sleep

Search/navigation:
- glob_tool, grep_tool, tool_search

Web/network:
- web_fetch, web_search

Planning/tasks/agents:
- task tools, todo_write, team_tool, agent_tool, send_message

MCP/tools bridge:
- mcp_auth_tool, mcp_resources

Mode/state tools:
- enter_plan_mode, exit_plan_mode, config_tool, brief

Other:
- formatter hooks, remote_trigger, synthetic_output, skill_tool, repl_tool, worktree

## Session state

Tool crate maintains per-session:
- shell state registry (cwd/env persistence)
- snapshot registry (undo/snapshot support)