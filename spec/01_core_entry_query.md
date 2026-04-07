# Core Entry + Query (Rust)

Source files:
- src-rust/crates/cli/src/main.rs
- src-rust/crates/query/src/lib.rs

## CLI entrypoint

The binary entrypoint is in crates/cli/src/main.rs.

Key behavior:
- Clap-based argument parsing
- Headless print mode and interactive mode
- Settings/config loading from core
- Context construction (system + user/project)
- Tool + MCP + plugin wiring
- Dispatch into query loop

## Query loop crate

crates/query is the core turn engine.

Public modules include:
- compact
- command_queue
- session_memory
- auto_dream
- away_summary
- coordinator
- cron_scheduler
- skill_prefetch
- agent_tool

## Query outcomes

The loop returns structured outcomes:
- EndTurn
- MaxTokens
- Cancelled
- Error
- BudgetExceeded

## Config and controls

QueryConfig includes:
- model, max_tokens, max_turns
- system prompt fields
- thinking/effort controls
- tool result budget
- fallback model
- provider/model registry integration
- optional command queue and skill index

## Compaction + budget

Query crate includes:
- automatic compaction checks
- reactive/snip compaction paths
- context window warning states
- cost/budget termination guard