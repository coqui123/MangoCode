---
name: rust-review
description: Thorough Rust code review — correctness, idioms, safety, and MangoCode conventions
triggers:
  - review this rust
  - review my rust
  - check my code
  - audit this
  - look at this pr
  - check for bugs
dependencies:
  - git-conventions
qa_required: true
qa_steps:
  - "Run `cargo check` and confirm it passes cleanly (zero errors, zero warnings)"
  - "Run `cargo test` for any crates with changed files and confirm all tests pass"
  - "Run `cargo clippy -- -D warnings` and address every Clippy lint before reporting"
  - "Report exact file paths and line numbers for every finding"
---

# Rust Code Review

You are performing a thorough Rust code review. Work through three parallel
review agents covering different dimensions, then synthesise and report.

## Sub-files available

| Objective | Load with |
|---|---|
| Security-specific checks | `skill="rust-review/security"` |
| Performance-specific checks | `skill="rust-review/performance"` |

## Step 1: Understand the Scope

Run `git diff HEAD` to see what changed. If there is no diff, review files
mentioned or edited earlier in this conversation. Note the affected crates.

## Step 2: Launch Three Review Agents in Parallel

Use the Agent tool to launch all three concurrently in a single call.
Pass each agent the full diff and the affected file list.

### Agent 1: Correctness & Safety

- **Error handling**: Flag every `unwrap()` / `expect()` that should use `?`
  or be replaced with proper error propagation.
- **Panics**: Identify index accesses, integer arithmetic, and slice ops that
  could panic in production.
- **Lifetimes**: Check for lifetime elision that might hide borrow issues.
- **Async safety**: In async code, flag `blocking` calls on async threads,
  cancellation-unsafe sections, and missing `select!` arms.
- **Unsafe blocks**: Every `unsafe` block must have a `// SAFETY:` comment.
  Flag any that are missing or where the comment doesn't justify the invariant.

### Agent 2: Idiomatic Rust

- Flag `unwrap()` chains that should use `and_then` / `map` / `?`.
- Identify manual loops that should use iterator adaptors.
- Check that new error types implement `std::error::Error` + `Display`.
- Verify public API items have doc comments (`///`).
- Flag non-idiomatic string handling: `format!` where `push_str` is cleaner,
  `.to_string()` on string literals that should be `.to_owned()` or a `&str`.
- Check `Clone` derives that could be avoided with references.

### Agent 3: MangoCode Project Conventions

- **Tracing**: All `tracing::info!` / `tracing::debug!` calls must use
  structured fields — `tracing::info!(key = value, "message")` not string
  interpolation.
- **Tool trait**: Any new tool must implement `Tool` with all five methods
  (`name`, `description`, `permission_level`, `input_schema`, `execute`).
- **Skill discovery priority**: New skill directory scan paths must follow the
  priority order documented in `skill_discovery.rs`.
- **Error variants**: New error variants should use `thiserror::Error` derive
  where the crate already depends on it; do not add bare `String` errors to
  typed error enums.
- **Coordinator tools**: Verify no new tool is added to `COORDINATOR_ONLY_TOOLS`
  without a corresponding worker-side check.

## Step 3: Synthesise

Wait for all three agents to complete. Merge findings and deduplicate.
Group by severity:

- **Blocking** — must fix before merge (correctness/safety issues)
- **Should fix** — strong code-quality issues
- **Consider** — style, performance micro-opts, nice-to-have

For each finding: file path, line number, description, suggested fix.

If no issues found in a category, say so explicitly.
