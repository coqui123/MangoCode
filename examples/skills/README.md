# MangoCode Example Skills

This directory contains example skills demonstrating the extended skill system
introduced in the `feat/skill-system-upgrade` branch.

## Installation

Copy any skill folder or file to one of the discovery paths:

```bash
# Project-level (checked first — highest priority)
cp -r examples/skills/rust-review .mangocode/skills/

# Global (available across all projects)
cp -r examples/skills/rust-review ~/.mangocode/skills/
cp -r examples/skills/git-conventions ~/.mangocode/skills/
cp -r examples/skills/design-foundations ~/.mangocode/skills/
```

---

## Skill Architecture

### Flat Skills (backward-compatible)

A single `.md` file with optional YAML frontmatter:

```
.mangocode/skills/
    my-skill.md
```

### Folder-Based Skills (new)

A directory containing `SKILL.md` as the primary entry point, with optional
sub-files and a `scripts/` subdirectory:

```
.mangocode/skills/
    rust-review/
        SKILL.md         ← primary content, loaded with skill="rust-review"
        security.md      ← sub-file, loaded with skill="rust-review/security"
        performance.md   ← sub-file, loaded with skill="rust-review/performance"
        scripts/
            check.sh     ← copied into session workspace when skill loads
```

---

## Frontmatter Reference

```yaml
---
name: my-skill
description: One-line description shown in skill list
triggers:
  - keyword phrase that auto-loads this skill
  - another phrase
when_to_use: Single-phrase shorthand (alias for one trigger entry)
dependencies:
  - other-skill-name    # loaded depth-first before this skill
qa_required: true
qa_steps:
  - "Step 1: Run validation"
  - "Step 2: Inspect output visually"
  - "Step 3: Fix issues and re-verify"
---

Your skill instructions here. Use $ARGUMENTS for user-supplied input.
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `name` | string | Skill name (defaults to filename stem) |
| `description` | string | Short description shown in `skill="list"` |
| `triggers` | list | Keyword phrases for auto-loading (Phase 1) |
| `when_to_use` | string | Single-value shorthand for `triggers` |
| `dependencies` | list | Skills loaded before this one (Phase 3) |
| `qa_required` | bool | Enforce mandatory QA checklist (Phase 5) |
| `qa_steps` | list | Ordered steps injected as hard constraints (Phase 5) |

---

## Invoking Skills

```bash
# List all available skills
/skill list

# Load a flat or folder-based skill
/skill rust-review

# Load a sub-file from a folder-based skill
/skill rust-review/security
/skill rust-review/performance
```

Or via the `Skill` tool in agentic sessions:

```json
{ "skill": "rust-review" }
{ "skill": "rust-review/security" }
{ "skill": "rust-review/security", "args": "focus on the auth module" }
```

---

## Auto-Loading (Phase 1)

Skills with `triggers:` are automatically injected into the system prompt when
the user's message matches a trigger phrase — before the model generates its
first token. No manual invocation required.

The resolver in `coordinator::resolve_skills_for_turn` handles matching.
Dependencies are expanded depth-first via `skill_discovery::load_skill_with_dependencies`.

---

## Included Examples

| Skill | Type | Description |
|---|---|---|
| `rust-review` | folder | Three-agent Rust code review with security + perf sub-files |
| `rust-review/security` | sub-file | Security checklist (memory safety, input handling, secrets) |
| `rust-review/performance` | sub-file | Performance checklist (allocations, concurrency, I/O) |
| `git-conventions` | folder | Commit format, branch naming, PR checklist |
| `design-foundations` | folder | Typography, color, layout rules — dependency of visual skills |
