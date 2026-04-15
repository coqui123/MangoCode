---
name: rust-review/performance
description: Performance-focused sub-review for Rust code — injected by rust-review on demand
---

# Rust Performance Review

Load this sub-file when you need performance analysis during a code review.
Reference: `skill="rust-review/performance"`

## Checklist

### Allocations
- [ ] No `String::from` / `.to_string()` / `format!` on hot paths where `&str` would work
- [ ] No `Vec` cloned in a loop where a reference would suffice
- [ ] No `HashMap::new()` inside a loop — reuse / pre-allocate with `HashMap::with_capacity`
- [ ] No `.collect::<Vec<_>>()` followed immediately by `.iter()` — chain the iterators directly

### Concurrency
- [ ] Independent async operations are spawned concurrently (`tokio::join!` / `FuturesUnordered`)
  not sequentially awaited
- [ ] No `Mutex` held across `.await` points (deadlock and starvation risk)
- [ ] No per-request `Arc::clone` of large objects — prefer passing references into spawned tasks

### I/O
- [ ] No synchronous blocking I/O (`std::fs::read`, `std::fs::write`) inside async tasks —
  use `tokio::fs` equivalents
- [ ] No un-buffered writes in a loop — wrap with `BufWriter`
- [ ] Large reads use streaming / chunked APIs, not `read_to_string` on unbounded input

### Startup
- [ ] No expensive computation (regex compilation, large file reads) in `fn main` or
  `impl Default` that runs before the first user interaction

## Report Format

For each finding: file, line, estimated impact (High / Medium / Low), description, suggested change.
