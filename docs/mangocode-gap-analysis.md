# MangoCode Gap Analysis & Highest-Leverage Next Moves

_Date: 2026-06-17_

This document captures findings from a recent review of the MangoCode codebase against the current state of agentic-coding research and competing harnesses (Cursor, Aider, Devin/Cognition, SWE-agent, OpenHands, Claude Code, etc.).

---

## ProjectGraph observation

The `ProjectGraph` tool reports:

- `graphify-out/graph.json` is **missing**.
- `graphify-out/manifest.json` does not exist.
- Graph freshness: **missing**.
- Counts at scan time: `files_scanned=311, nodes=10701, edges=36811, communities=1`.

**Implication:** MangoCode already has a working code-knowledge-graph builder, but nothing is persisting or auto-injecting that graph back into the agent loop. The model is not benefiting from its own structural understanding of the repo between turns or between sessions.

**Fix shape:**

1. Add a post-commit git hook (or a `/graphify --persist` shortcut) that runs `ProjectGraph action=persist` after meaningful source changes.
2. Auto-inject the compact `context_pack` into every turn's system prompt, token-budgeted.
3. This is essentially the **Aider repomap pattern** — MangoCode is ~80% of the way there already.

---

## Three highest-leverage moves

Narrowed down from a longer gap list, prioritized by ROI given what already exists in the codebase.

### 1. Auto-persist + auto-inject the ProjectGraph context_pack

**Why it wins:** Biggest "free" improvement. The graph builder exists; the agent just isn't being fed its output.

**Surfaces:**

- `src-rust/crates/query/src/lib.rs` — inject context_pack into system prompt at turn start.
- `src-rust/crates/commands/src/lib.rs` — wire up `/graphify --persist` shortcut.
- Add a freshness check: rebuild on staleness or major file-count delta.
- Token-budget the injection so it doesn't blow context on small models.

**Behavior parity target:** Aider's repomap (ranked symbol skeleton fit to a token budget via tree-sitter + PageRank on the call graph).

### 2. Critic / verifier turn in the agent loop

**Why it wins:** Reflexion / Self-Refine-style verbal self-criticism reliably improves task success ~15-25% in published evals. One extra LLM call after a file-changing turn that reviews the diff against the original ask and either approves or sends back a correction.

**Surfaces:**

- `src-rust/crates/query/src/lib.rs` — add a post-edit verification phase.
- Add an opt-in config knob: `agent_verification_profile: critic` (in line with the existing `verification_policy` setting).
- Feed the verifier the original user ask, the captured turn diff (you already have `/changes`), and the test command output if configured.

**Behavior parity target:** Reflexion (Shinn et al. 2023), Self-Refine (Madaan et al. 2023), AlphaCodium flow engineering.

### 3. Sandboxed Bash backend

**Why it wins:** This is the production credibility gap. Codex CLI, Devin, OpenHands, and Cursor background agents all run in containers / microVMs. MangoCode runs Bash/PowerShell directly on the host. Worktrees give git isolation but **do not** isolate filesystem, network, or process.

**Surfaces:**

- `src-rust/crates/tools/` — Bash tool implementation.
- Add a backend trait: `HostBash` (current) vs `DockerBash` / `PodmanBash`.
- Gate via env var: `MANGOCODE_SANDBOX=docker|podman|host` (default `host` for backward compat).
- Mount the working directory read-write, deny network by default, add `--network=host` opt-in flag.
- Reuse the encrypted vault for any credentials passed into the sandbox.

**Behavior parity target:** OpenAI Codex CLI sandbox, OpenHands runtime container, Cursor background agents.

---

## Suggested sequencing

1. **ProjectGraph auto-inject** (Move 1) — smallest blast radius, biggest immediate win. Start here.
2. **Critic turn** (Move 2) — pure additive behind a feature flag. Ship after Move 1 so the verifier can leverage the injected graph context.
3. **Sandboxed Bash** (Move 3) — bigger change, touches the tools crate and CLI surface. Tackle after the agent-loop improvements land and the test surface is healthier.

---

## Next concrete patch target

`src-rust/crates/query/src/lib.rs` — sketch ProjectGraph context_pack auto-injection:

- Locate the system-prompt assembly path in the query loop.
- Add a call to `ProjectGraph action=context_pack compact=true` (or the in-crate equivalent) with a token budget derived from the model's context window.
- Persist the resulting pack to `graphify-out/graph.json` if missing or stale.
- Insert the pack as an `Untrusted content notice`-style data block in the system prompt so it follows existing trust hygiene.

---

## Longer wishlist (for reference)

Not prioritized above, but worth tracking:

- Incremental tree-sitter index with file-watcher for `CodeSearch`.
- `MANGO.md` auto-maintenance — agent appends learned project conventions after successful turns.
- Eval harness crate running SWE-bench Lite + Aider polyglot, reportable via `/doctor --bench`.
- CodeAct mode — Python action space alternative to JSON tool calls (great fit for the existing REPL tool).
- Plan-mode upgrade to show a structured patch plan with file/symbol targets before executing.
- Per-tool egress / path policies in `permissions.rs`.
- Inline diff approval UI (Cursor-style accept/reject hunk) — currently `/changes` is post-hoc only.
- Checkpoint timeline / rewind-to-step-N (Devin / Cline pattern).
- Cost / latency dashboard per provider per session.
- Curated library of specialist sub-agents (Test-Writer, Migration, Reviewer, Doc) on top of the existing `TeamCreate` primitive.

---

## Research / systems to crib from

| System | What to steal |
|---|---|
| Aider | Repomap (token-budgeted ranked symbol skeleton via tree-sitter + PageRank) |
| SWE-agent | Formal Agent-Computer Interface (ACI) spec |
| CodeAct (UIUC 2024) | Python code as universal action representation |
| OpenHands | Reference open-source agent stack (sandbox, editor, plan) |
| Reflexion / Self-Refine | Verbal self-criticism loop |
| Anthropic "Building effective agents" (Dec 2024) | Workflows over fully-autonomous agents for coding |
| AlphaCodium / Flow Engineering | TDD iterative code generation (spec → tests → code → debug) |
| MCP 2025 spec updates | `roots` (workspace boundaries) and `sampling` (server-initiated LLM calls) |
| Continue.dev | Per-project agent config files committed to repo |
