use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::clippy_diff::{introduced_findings, parse_clippy_findings, ClippyFinding};

/// Temporarily reverts a set of files to a "before" version on disk, restoring
/// each file's prior (current) content when the guard is dropped — including on
/// panic or early return. Best-effort: a file that can't be read or written is
/// skipped (and therefore not restored).
pub struct RevertGuard {
    restore: Vec<(PathBuf, Vec<u8>)>,
}

impl RevertGuard {
    /// Revert each `(path, before_content)` in place. The file's current bytes
    /// are saved first so they can be restored on drop. Returns the guard; hold
    /// it for as long as the reverted state is needed, then drop it to restore.
    pub fn revert(files: &[(PathBuf, String)]) -> Self {
        let mut restore = Vec::new();
        for (path, before) in files {
            let Ok(current) = std::fs::read(path) else {
                continue;
            };
            if std::fs::write(path, before.as_bytes()).is_ok() {
                restore.push((path.clone(), current));
            }
        }
        Self { restore }
    }

    /// Number of files actually reverted (and that will be restored on drop).
    pub fn reverted_count(&self) -> usize {
        self.restore.len()
    }
}

impl Drop for RevertGuard {
    fn drop(&mut self) {
        for (path, content) in &self.restore {
            let _ = std::fs::write(path, content);
        }
    }
}

/// Run `cargo clippy` for `package` in `working_dir`, returning its
/// `--message-format=json` stdout (newline-delimited diagnostics).
fn run_clippy_json(working_dir: &Path, package: &str) -> std::io::Result<String> {
    let output = std::process::Command::new("cargo")
        .args([
            "clippy",
            "-p",
            package,
            "--all-targets",
            "--message-format=json",
            "--quiet",
        ])
        .current_dir(working_dir)
        .output()?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Core of the baseline diff, with the clippy invocation injected so it can be
/// tested without spawning cargo. Runs `run_clippy` once with `changed_files`
/// reverted to their "before" content (the session-start baseline), then again
/// after the guard restores the current content, and returns the findings
/// present now but not in the baseline. The clippy spawn is injected so this
/// core is unit-testable without cargo; `compute_introduced_clippy` is the thin
/// production wrapper. The `?` on the baseline run still drops the guard (it is
/// in an inner scope), so the tree is always restored.
fn compute_introduced_clippy_with<F>(
    changed_files: &[(PathBuf, String)],
    run_clippy: F,
) -> std::io::Result<Vec<ClippyFinding>>
where
    F: Fn() -> std::io::Result<String>,
{
    let baseline_output = {
        let _guard = RevertGuard::revert(changed_files);
        run_clippy()?
    };
    let current_output = run_clippy()?;
    let baseline = parse_clippy_findings(&baseline_output);
    let current = parse_clippy_findings(&current_output);
    Ok(introduced_findings(&baseline, &current))
}

/// Compute the clippy findings INTRODUCED by this session's edits to `package`:
/// run clippy against the session-start state (files reverted to their "before"
/// content), then against the current state, and diff. The working tree is
/// restored before returning (the `RevertGuard` restores as it drops, even if
/// a clippy run errors). A spawn error propagates as `Err` so the caller can
/// fall back to advisory rather than mistaking it for "no new findings".
pub fn compute_introduced_clippy(
    working_dir: &Path,
    package: &str,
    changed_files: &[(PathBuf, String)],
) -> std::io::Result<Vec<ClippyFinding>> {
    compute_introduced_clippy_with(changed_files, || run_clippy_json(working_dir, package))
}

/// Group changed files by crate and collect the clippy findings introduced
/// across all of them. `package_of` maps a file path to its crate name; files
/// that don't resolve to a crate are skipped. `run_pkg` runs the baseline diff
/// for one crate — injected so this is testable without cargo. A crate whose
/// run errors contributes nothing (fail-safe: never invent a finding).
fn introduced_clippy_grouped<P, R>(
    changed_files: &[(PathBuf, String)],
    package_of: P,
    run_pkg: R,
) -> Vec<ClippyFinding>
where
    P: Fn(&Path) -> Option<String>,
    R: Fn(&str, &[(PathBuf, String)]) -> std::io::Result<Vec<ClippyFinding>>,
{
    let mut by_pkg: BTreeMap<String, Vec<(PathBuf, String)>> = BTreeMap::new();
    for (path, before) in changed_files {
        if let Some(pkg) = package_of(path) {
            by_pkg
                .entry(pkg)
                .or_default()
                .push((path.clone(), before.clone()));
        }
    }
    let mut all = Vec::new();
    for (pkg, files) in by_pkg {
        if let Ok(found) = run_pkg(&pkg, &files) {
            all.extend(found);
        }
    }
    all
}

/// The clippy findings introduced by this session across every changed crate.
/// `changed_files` are `(absolute path, session-start "before" content)` for the
/// edited `.rs` files; `package_of` resolves each to its crate. Runs clippy
/// against the reverted vs current state per crate (see `compute_introduced_clippy`).
pub fn introduced_clippy_findings<P>(
    workspace_dir: &Path,
    changed_files: &[(PathBuf, String)],
    package_of: P,
) -> Vec<ClippyFinding>
where
    P: Fn(&Path) -> Option<String>,
{
    introduced_clippy_grouped(changed_files, package_of, |pkg, files| {
        compute_introduced_clippy(workspace_dir, pkg, files)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn revert_writes_before_and_drop_restores_original() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("example.rs");
        fs::write(&file, b"after-content").unwrap();

        {
            let guard = RevertGuard::revert(&[(file.clone(), "before-content".to_string())]);
            assert_eq!(guard.reverted_count(), 1);
            // While the guard is alive, the file should contain the "before" content.
            let on_disk = fs::read_to_string(&file).unwrap();
            assert_eq!(on_disk, "before-content");
        }
        // Guard dropped — file should be restored to the original "after" content.
        let restored = fs::read(&file).unwrap();
        assert_eq!(restored, b"after-content");
    }

    #[test]
    fn reverted_count_matches_existing_files() {
        let dir = TempDir::new().unwrap();
        let f1 = dir.path().join("a.rs");
        let f2 = dir.path().join("b.rs");
        fs::write(&f1, b"a").unwrap();
        fs::write(&f2, b"b").unwrap();

        let guard =
            RevertGuard::revert(&[(f1.clone(), "x".to_string()), (f2.clone(), "y".to_string())]);
        assert_eq!(guard.reverted_count(), 2);
        drop(guard);
    }

    #[test]
    fn nonexistent_path_is_skipped() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("no_such_file.rs");

        let guard = RevertGuard::revert(&[(missing.clone(), "content".to_string())]);
        assert_eq!(guard.reverted_count(), 0);
        // The file should still not exist.
        assert!(!missing.exists());
        drop(guard);
    }

    /// Build one `compiler-message` JSON line for a clippy finding.
    fn clippy_msg(code: &str, message: &str, snippet: &str) -> String {
        serde_json::json!({
            "reason": "compiler-message",
            "message": {
                "code": { "code": code },
                "level": "warning",
                "message": message,
                "spans": [{ "is_primary": true, "text": [{ "text": snippet }] }],
            },
        })
        .to_string()
    }

    #[test]
    fn introduced_clippy_reports_only_new_findings_and_restores_tree() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("lib.rs");
        fs::write(&file, b"AFTER").unwrap();
        let changed = vec![(file.clone(), "BEFORE".to_string())];

        // Injected clippy reads the file: the baseline state ("BEFORE") has only
        // finding A; the current state ("AFTER") has A and B → B is introduced.
        let probe = file.clone();
        let run = move || -> std::io::Result<String> {
            let content = fs::read_to_string(&probe)?;
            if content == "BEFORE" {
                Ok(clippy_msg("clippy::a", "msg a", "snip a"))
            } else {
                Ok(format!(
                    "{}\n{}",
                    clippy_msg("clippy::a", "msg a", "snip a"),
                    clippy_msg("clippy::b", "msg b", "snip b"),
                ))
            }
        };

        let introduced = compute_introduced_clippy_with(&changed, run).unwrap();
        assert_eq!(introduced.len(), 1);
        assert_eq!(introduced[0].code, "clippy::b");
        // The tree is restored to the current ("AFTER") content afterward.
        assert_eq!(fs::read_to_string(&file).unwrap(), "AFTER");
    }

    #[test]
    fn grouped_collects_per_package_and_is_fail_safe() {
        let files = vec![
            (
                PathBuf::from("/w/crates/a/src/lib.rs"),
                "before-a".to_string(),
            ),
            (
                PathBuf::from("/w/crates/b/src/lib.rs"),
                "before-b".to_string(),
            ),
            (PathBuf::from("/w/README.md"), "x".to_string()), // resolves to no crate
        ];
        let package_of = |p: &Path| {
            let s = p.to_string_lossy();
            if s.contains("/crates/a/") {
                Some("a".to_string())
            } else if s.contains("/crates/b/") {
                Some("b".to_string())
            } else {
                None
            }
        };
        let run_pkg =
            |pkg: &str, _files: &[(PathBuf, String)]| -> std::io::Result<Vec<ClippyFinding>> {
                match pkg {
                    "a" => Ok(vec![ClippyFinding {
                        code: "clippy::a".into(),
                        message: "m".into(),
                        snippet: "s".into(),
                    }]),
                    // Crate "b" errors → fail-safe: it contributes no findings.
                    "b" => Err(std::io::Error::other("clippy failed")),
                    _ => Ok(vec![]),
                }
            };
        let found = introduced_clippy_grouped(&files, package_of, run_pkg);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].code, "clippy::a");
    }
}
