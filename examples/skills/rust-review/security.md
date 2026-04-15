---
name: rust-review/security
description: Security-focused sub-review for Rust code — injected by rust-review on demand
---

# Rust Security Review

Load this sub-file when you need security-specific analysis during a code review.
Reference: `skill="rust-review/security"`

## Checklist

### Memory Safety
- [ ] Every `unsafe` block has a `// SAFETY:` comment explaining invariants upheld by the caller
- [ ] No raw pointer arithmetic without bounds checking proven in the comment
- [ ] No `transmute` unless absolutely necessary; flag any usage for human review
- [ ] `Vec::from_raw_parts` / `Box::from_raw` — verify ownership transfer is correct

### Input Handling
- [ ] All user-supplied strings are validated before use as filesystem paths
- [ ] No `format!` calls that directly embed user input into shell commands passed to `Command`
- [ ] Deserialization of external data uses bounded types (no unbounded `Vec<u8>` from network)
- [ ] File paths are canonicalized before any sensitive operation (`.canonicalize()`)

### Secrets & Credentials
- [ ] No API keys, tokens, or passwords hardcoded in source or test fixtures
- [ ] No secrets written to log output via `tracing::debug!` or `println!`
- [ ] Temporary files containing sensitive data are removed on drop (consider `tempfile` crate)

### Dependency Surface
- [ ] No new `unsafe` dependency added without audit note in `Cargo.toml` comment
- [ ] Verify new dependencies don't pull in known-vulnerable crate versions (run `cargo audit` if available)

## Report Format

For each finding: file, line, severity (Critical / High / Medium / Low), description, remediation.
