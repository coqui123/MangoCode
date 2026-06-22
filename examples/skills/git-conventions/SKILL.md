---
name: git-conventions
description: MangoCode git commit format, branch naming, and PR checklist — auto-loaded as a dependency
triggers:
  - commit message
  - branch name
  - pull request
  - open a pr
  - squash and merge
---

# MangoCode Git Conventions

This skill is loaded automatically as a dependency by other skills (e.g. `rust-review`).
It defines the commit and branch standards to apply during reviews and when creating commits.

## Commit Message Format

Follow the Conventional Commits specification:

```
<type>(<scope>): <short summary>

[optional body — wrap at 72 chars]

[optional footer: BREAKING CHANGE, Closes #<issue>]
```

**Allowed types:**
- `feat` — new feature or capability
- `fix` — bug fix
- `refactor` — code restructure without behaviour change
- `perf` — performance improvement
- `test` — test additions or changes
- `docs` — documentation only
- `chore` — build, CI, dependency updates
- `style` — formatting, no logic change

**Scope:** the crate or module name, e.g. `skill-discovery`, `coordinator`, `tools`

**Summary:** imperative mood, lowercase, no trailing period, ≤ 72 chars

**Examples:**
```
feat(skill-discovery): add folder-based skill support with sub-files
fix(coordinator): prevent dependency cycle in skill resolver
refactor(system_prompt): inject skills into cacheable prompt section
```

## Branch Naming

```
<type>/<short-kebab-description>
```

Examples:
- `feat/skill-system-upgrade`
- `fix/coordinator-cycle-detection`
- `refactor/prompt-cache-alignment`

Feature branches target `main`. Use squash and merge on PR close.

## PR Checklist

Before marking a PR ready for review:

- [ ] `cargo check` passes with zero warnings
- [ ] `cargo test` passes for all affected crates
- [ ] `cargo clippy -- -D warnings` passes
- [ ] New public items have `///` doc comments
- [ ] CHANGELOG or PR description summarises the change
- [ ] No secrets, credentials, or personal information in any committed file
- [ ] Commit messages follow Conventional Commits format above

## Identity Note

When committing on behalf of a human, use their noreply GitHub email to avoid
leaking personal information. Never hardcode personal email addresses in commit
metadata.
