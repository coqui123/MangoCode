//! Atomic file writes for tools that modify user files.
//!
//! A plain `fs::write` truncates the target before writing, so a crash,
//! power loss, or kill signal mid-write leaves the user's file empty or
//! partially written. Writing to a temp file in the same directory and
//! renaming it over the target makes the replacement all-or-nothing.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static THREAD_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn temp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tsn = THREAD_SEQ.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    });
    path.with_file_name(format!(
        ".{}.{}.{}.{}.mangotmp",
        file_name,
        std::process::id(),
        nonce,
        tsn,
    ))
}

/// Write `contents` to `path` atomically: write to a temp file in the same
/// directory, fsync it, then rename it over the target.
///
/// Falls back to a direct `fs::write` if the rename fails (e.g. the target
/// is held open without share-delete on Windows, or the filesystem does not
/// support rename-over), so no previously-writable path becomes unwritable.
pub async fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = temp_path_for(path);

    // Stage the new contents into a temp file in the same directory. An error
    // here (parent missing, disk full, permission) leaves the target intact, so
    // propagate it rather than truncating the target with a fallback write.
    let stage: io::Result<()> = async {
        let mut file = tokio::fs::File::create(&tmp).await?;
        file.write_all(contents).await?;
        file.sync_all().await?;
        drop(file);

        // The temp file is created with default permissions; carry over the
        // original file's mode so the rename doesn't change it.
        #[cfg(unix)]
        if let Ok(meta) = tokio::fs::metadata(path).await {
            tokio::fs::set_permissions(&tmp, meta.permissions()).await?;
        }
        Ok(())
    }
    .await;

    if let Err(e) = stage {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }

    match tokio::fs::rename(&tmp, path).await {
        Ok(()) => {
            // fsync the parent directory so the rename itself is durable across
            // a crash or power loss, not just the temp file's data.
            #[cfg(unix)]
            if let Some(parent) = path.parent() {
                if let Ok(dir) = tokio::fs::File::open(parent).await {
                    let _ = dir.sync_all().await;
                }
            }
            Ok(())
        }
        // Rename can fail when the target is held open without share-delete
        // (Windows). Fall back to a direct write; if that also fails, surface
        // the original rename error rather than the fallback's.
        Err(rename_err) => {
            let _ = tokio::fs::remove_file(&tmp).await;
            match tokio::fs::write(path, contents).await {
                Ok(()) => Ok(()),
                Err(_) => Err(rename_err),
            }
        }
    }
}

/// Synchronous version of [`write_atomic`] for callers outside async contexts.
pub fn write_atomic_sync(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = temp_path_for(path);

    let stage: io::Result<()> = (|| {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);

        #[cfg(unix)]
        if let Ok(meta) = std::fs::metadata(path) {
            std::fs::set_permissions(&tmp, meta.permissions())?;
        }
        Ok(())
    })();

    if let Err(e) = stage {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    match std::fs::rename(&tmp, path) {
        Ok(()) => {
            #[cfg(unix)]
            if let Some(parent) = path.parent() {
                if let Ok(dir) = std::fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
            Ok(())
        }
        Err(rename_err) => {
            let _ = std::fs::remove_file(&tmp);
            match std::fs::write(path, contents) {
                Ok(()) => Ok(()),
                Err(_) => Err(rename_err),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        write_atomic(&path, b"hello").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    }

    #[tokio::test]
    async fn replaces_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "old contents").unwrap();
        write_atomic(&path, b"new contents").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new contents");
    }

    #[tokio::test]
    async fn leaves_no_temp_files_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        write_atomic(&path, b"one").await.unwrap();
        write_atomic(&path, b"two").await.unwrap();
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["file.txt"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn preserves_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("script.sh");
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        write_atomic(&path, b"#!/bin/sh\necho hi\n").await.unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755);
    }

    #[tokio::test]
    async fn errors_when_parent_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("file.txt");
        assert!(write_atomic(&path, b"x").await.is_err());
    }

    #[test]
    fn sync_replaces_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "old").unwrap();
        write_atomic_sync(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }
}
