use chrono::Utc;
use mangocode_core::file_history::FileHistory;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use similar::TextDiff;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatchId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnPatch {
    pub id: PatchId,
    pub turn_index: usize,
    pub path: PathBuf,
    pub relative_path: String,
    pub tool_names: Vec<String>,
    pub binary: bool,
    pub before_text: Option<String>,
    pub after_text: Option<String>,
    pub before_exists: bool,
    pub after_exists: bool,
    pub unified_diff: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchBundle {
    pub session_id: String,
    pub turn_index: usize,
    pub patch_ids: Vec<PatchId>,
    pub file_list: Vec<String>,
    pub created_at: String,
    pub unified_diff_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackResult {
    pub reverted_paths: Vec<PathBuf>,
    pub errors: Vec<String>,
}

pub fn patches_for_turn(history: &FileHistory, turn_index: usize, root: &Path) -> Vec<TurnPatch> {
    let mut tools_by_path: HashMap<PathBuf, BTreeSet<String>> = HashMap::new();
    for entry in history.get_entries_for_turn(turn_index) {
        tools_by_path
            .entry(entry.path.clone())
            .or_default()
            .insert(entry.tool_name.clone());
    }

    history
        .snapshots_for_turn(turn_index)
        .into_iter()
        .map(|snapshot| {
            let relative_path = relative_path(&snapshot.path, root);
            let tool_names = tools_by_path
                .get(&snapshot.path)
                .map(|tools| tools.iter().cloned().collect())
                .unwrap_or_default();
            let unified_diff = if snapshot.binary {
                format!("Binary file changed: {relative_path}\n")
            } else {
                unified_diff_with_existence(
                    &relative_path,
                    snapshot.before_text.as_deref().unwrap_or_default(),
                    snapshot.after_text.as_deref().unwrap_or_default(),
                    snapshot.before_exists,
                    snapshot.after_exists,
                )
            };
            TurnPatch {
                id: patch_id(turn_index, &relative_path),
                turn_index,
                path: snapshot.path,
                relative_path,
                tool_names,
                binary: snapshot.binary,
                before_text: snapshot.before_text,
                after_text: snapshot.after_text,
                before_exists: snapshot.before_exists,
                after_exists: snapshot.after_exists,
                unified_diff,
            }
        })
        .collect()
}

pub fn export_patch_bundle(
    session_id: impl Into<String>,
    history: &FileHistory,
    turn_index: usize,
    root: &Path,
) -> PatchBundle {
    export_patch_bundle_from_patches(
        session_id,
        turn_index,
        patches_for_turn(history, turn_index, root),
    )
}

pub fn export_patch_bundle_from_patches(
    session_id: impl Into<String>,
    turn_index: usize,
    patches: Vec<TurnPatch>,
) -> PatchBundle {
    let patch_ids = patches.iter().map(|patch| patch.id.clone()).collect();
    let file_list = patches
        .iter()
        .map(|patch| patch.relative_path.clone())
        .collect::<Vec<_>>();
    let unified_diff_text = patches
        .iter()
        .map(|patch| patch.unified_diff.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    PatchBundle {
        session_id: session_id.into(),
        turn_index,
        patch_ids,
        file_list,
        created_at: Utc::now().to_rfc3339(),
        unified_diff_text,
    }
}

pub fn rollback_turn(history: &FileHistory, turn_index: usize) -> io::Result<RollbackResult> {
    let mut result = RollbackResult {
        reverted_paths: Vec::new(),
        errors: Vec::new(),
    };

    for snapshot in history.snapshots_for_turn(turn_index).into_iter().rev() {
        if snapshot.binary {
            result.errors.push(format!(
                "Cannot rollback binary file {} from text history",
                snapshot.path.display()
            ));
            continue;
        }

        if !snapshot.before_exists {
            if snapshot.path.exists() {
                match fs::remove_file(&snapshot.path) {
                    Ok(()) => result.reverted_paths.push(snapshot.path),
                    Err(err) => result.errors.push(format!(
                        "Failed to delete {}: {}",
                        snapshot.path.display(),
                        err
                    )),
                }
            } else {
                result.reverted_paths.push(snapshot.path);
            }
        } else {
            match snapshot.before_text {
                Some(before_text) => {
                    if let Some(parent) = snapshot.path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    match fs::write(&snapshot.path, before_text) {
                        Ok(()) => result.reverted_paths.push(snapshot.path),
                        Err(err) => result.errors.push(format!(
                            "Failed to restore {}: {}",
                            snapshot.path.display(),
                            err
                        )),
                    }
                }
                None => result.errors.push(format!(
                    "Cannot rollback {} because its previous text snapshot is unavailable",
                    snapshot.path.display()
                )),
            }
        }
    }

    Ok(result)
}

pub fn patch_id(turn_index: usize, relative_path: &str) -> PatchId {
    let mut hasher = Sha256::new();
    hasher.update(turn_index.to_string().as_bytes());
    hasher.update(b":");
    hasher.update(relative_path.as_bytes());
    let digest = hex::encode(hasher.finalize());
    PatchId(format!("turn-{turn_index}-{}", &digest[..12]))
}

pub fn unified_diff(relative_path: &str, before: &str, after: &str) -> String {
    unified_diff_with_existence(relative_path, before, after, true, true)
}

pub fn unified_diff_with_existence(
    relative_path: &str,
    before: &str,
    after: &str,
    before_exists: bool,
    after_exists: bool,
) -> String {
    let before_header = if before_exists {
        format!("a/{relative_path}")
    } else {
        "/dev/null".to_string()
    };
    let after_header = if after_exists {
        format!("b/{relative_path}")
    } else {
        "/dev/null".to_string()
    };

    TextDiff::from_lines(before, after)
        .unified_diff()
        .header(&before_header, &after_header)
        .to_string()
}

fn relative_path(path: &Path, root: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::file_history::FileHistory;

    #[test]
    fn creates_stable_patch_ids_and_diff_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src").join("lib.rs");
        fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut history = FileHistory::new();
        history.record_modification(
            path.clone(),
            b"fn old() {}\n",
            b"fn new() {}\n",
            3,
            "FileEdit",
        );

        let patches = patches_for_turn(&history, 3, dir.path());
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].relative_path, "src/lib.rs");
        assert_eq!(patches[0].id, patch_id(3, "src/lib.rs"));
        assert!(patches[0].before_exists);
        assert!(patches[0].after_exists);
        assert!(patches[0].unified_diff.contains("-fn old()"));
        assert!(patches[0].unified_diff.contains("+fn new()"));
    }

    #[test]
    fn exports_patch_bundle_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("README.md");
        let mut history = FileHistory::new();
        history.record_modification(path, b"old\n", b"new\n", 1, "FileWrite");

        let bundle = export_patch_bundle("session-1", &history, 1, dir.path());
        assert_eq!(bundle.session_id, "session-1");
        assert_eq!(bundle.turn_index, 1);
        assert_eq!(bundle.file_list, vec!["README.md"]);
        assert_eq!(bundle.patch_ids.len(), 1);
        assert!(bundle.unified_diff_text.contains("README.md"));
    }

    #[test]
    fn rolls_back_text_changes_for_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        fs::write(&path, "after").unwrap();

        let mut history = FileHistory::new();
        history.record_modification(path.clone(), b"before", b"after", 2, "FileEdit");

        let result = rollback_turn(&history, 2).unwrap();
        assert!(result.errors.is_empty());
        assert_eq!(result.reverted_paths, vec![path.clone()]);
        assert_eq!(fs::read_to_string(path).unwrap(), "before");
    }

    #[test]
    fn rolls_back_created_file_by_deleting_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("created.txt");
        fs::write(&path, "created").unwrap();

        let mut history = FileHistory::new();
        history.record_modification_with_existence(
            path.clone(),
            b"",
            b"created",
            (false, true),
            4,
            "FileWrite",
        );

        let result = rollback_turn(&history, 4).unwrap();
        assert!(result.errors.is_empty());
        assert_eq!(result.reverted_paths, vec![path.clone()]);
        assert!(!path.exists());
    }

    #[test]
    fn exported_diffs_use_dev_null_for_created_and_deleted_files() {
        let dir = tempfile::tempdir().unwrap();
        let created = dir.path().join("created.txt");
        let deleted = dir.path().join("deleted.txt");

        let mut history = FileHistory::new();
        history.record_modification_with_existence(
            created,
            b"",
            b"new\n",
            (false, true),
            9,
            "FileWrite",
        );
        history.record_modification_with_existence(
            deleted,
            b"old\n",
            b"",
            (true, false),
            9,
            "ApplyPatch",
        );

        let bundle = export_patch_bundle("session-1", &history, 9, dir.path());
        assert!(bundle.unified_diff_text.contains("--- /dev/null"));
        assert!(bundle.unified_diff_text.contains("+++ b/created.txt"));
        assert!(bundle.unified_diff_text.contains("--- a/deleted.txt"));
        assert!(bundle.unified_diff_text.contains("+++ /dev/null"));
    }
}
