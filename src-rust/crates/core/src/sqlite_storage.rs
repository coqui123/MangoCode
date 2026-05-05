// sqlite_storage.rs — Optional SQLite-backed session storage.
//
// Provides `SqliteSessionStore` as a faster, queryable alternative to
// the default JSONL storage.  Enabled by adding `rusqlite` to the
// crate's dependencies (already done via `features = ["bundled"]`).

use std::path::{Path, PathBuf};

use crate::harness::{
    CheckpointBackend, CheckpointKind, HarnessCheckpoint, HarnessEvent, HarnessTurnStatus,
};

/// A persistent SQLite session + message store.
pub struct SqliteSessionStore {
    conn: rusqlite::Connection,
}

impl SqliteSessionStore {
    /// Open (or create) the database at `db_path` and ensure the schema exists.
    pub fn open(db_path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = rusqlite::Connection::open(db_path)?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT,
                model       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                cost_usd    REAL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at);

            CREATE TABLE IF NOT EXISTS harness_events (
                sequence      INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id      TEXT NOT NULL UNIQUE,
                session_id    TEXT NOT NULL,
                turn_id       TEXT,
                tool_call_id  TEXT,
                checkpoint_id TEXT,
                event_type    TEXT NOT NULL,
                payload_json  TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_harness_events_session_sequence
                ON harness_events(session_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_harness_events_turn
                ON harness_events(turn_id);

            CREATE TABLE IF NOT EXISTS harness_turns (
                turn_id      TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                status       TEXT NOT NULL,
                model        TEXT,
                cwd          TEXT,
                started_at   TEXT NOT NULL,
                completed_at TEXT,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_harness_turns_session
                ON harness_turns(session_id, updated_at);

            CREATE TABLE IF NOT EXISTS harness_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                turn_id       TEXT NOT NULL,
                kind          TEXT NOT NULL,
                backend       TEXT NOT NULL,
                repo_root     TEXT,
                git_ref       TEXT,
                snapshot_json TEXT,
                created_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_harness_checkpoints_session
                ON harness_checkpoints(session_id, created_at);
            ",
        )?;

        Ok(Self { conn })
    }

    /// Insert or replace a session record.  `created_at` is preserved on
    /// UPDATE so only `updated_at` changes.
    pub fn save_session(
        &self,
        session_id: &str,
        title: Option<&str>,
        model: &str,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO sessions (id, title, model, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(id) DO UPDATE SET
                 title      = excluded.title,
                 model      = excluded.model,
                 updated_at = excluded.updated_at",
            rusqlite::params![session_id, title, model, now],
        )?;
        Ok(())
    }

    /// Append a message to the given session (idempotent on `msg_id`).
    /// Also bumps `sessions.message_count` and `sessions.updated_at`.
    pub fn save_message(
        &self,
        session_id: &str,
        msg_id: &str,
        role: &str,
        content: &str,
        cost_usd: Option<f64>,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        // Insert the message; ignore if already stored.
        let inserted = self.conn.execute(
            "INSERT OR IGNORE INTO messages
             (id, session_id, role, content, created_at, cost_usd)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![msg_id, session_id, role, content, now, cost_usd],
        )?;
        // Only bump count when we actually inserted a new row.
        if inserted > 0 {
            self.conn.execute(
                "UPDATE sessions
                 SET updated_at    = ?1,
                     message_count = message_count + 1
                 WHERE id = ?2",
                rusqlite::params![now, session_id],
            )?;
        }
        Ok(())
    }

    /// Return the 100 most recently updated sessions.
    pub fn list_sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, model, created_at, updated_at, message_count
             FROM sessions
             ORDER BY updated_at DESC
             LIMIT 100",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(SessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                model: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                message_count: row.get::<_, Option<u32>>(5)?.unwrap_or(0),
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Full-text search across session titles and message content.
    /// Returns up to 50 matching sessions ordered by recency.
    pub fn search_sessions(&self, query: &str) -> anyhow::Result<Vec<SessionSummary>> {
        let like = format!("%{}%", query);
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT s.id, s.title, s.model,
                    s.created_at, s.updated_at, s.message_count
             FROM sessions s
             LEFT JOIN messages m ON m.session_id = s.id
             WHERE s.title LIKE ?1
                OR m.content LIKE ?1
             ORDER BY s.updated_at DESC
             LIMIT 50",
        )?;

        let rows = stmt.query_map(rusqlite::params![like], |row| {
            Ok(SessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                model: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                message_count: row.get::<_, Option<u32>>(5)?.unwrap_or(0),
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Delete a session and all of its messages.
    pub fn delete_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        self.conn.execute(
            "DELETE FROM sessions WHERE id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(())
    }

    pub fn append_harness_event(&self, event: &HarnessEvent) -> anyhow::Result<i64> {
        self.conn.execute(
            "INSERT OR IGNORE INTO harness_events
             (event_id, session_id, turn_id, tool_call_id, checkpoint_id,
              event_type, payload_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                event.event_id,
                event.session_id,
                event.turn_id,
                event.tool_call_id,
                event.checkpoint_id,
                event.event_type,
                serde_json::to_string(&event.payload)?,
                event.timestamp.to_rfc3339(),
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn list_harness_events(
        &self,
        session_id: &str,
        after_sequence: Option<i64>,
        limit: usize,
    ) -> anyhow::Result<Vec<(i64, HarnessEvent)>> {
        let mut stmt = self.conn.prepare(
            "SELECT sequence, event_id, session_id, turn_id, tool_call_id,
                    checkpoint_id, event_type, payload_json, created_at
             FROM harness_events
             WHERE session_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            rusqlite::params![session_id, after_sequence.unwrap_or(0), limit as i64],
            |row| {
                let payload_json: String = row.get(7)?;
                let created_at: String = row.get(8)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());
                Ok((
                    row.get::<_, i64>(0)?,
                    HarnessEvent {
                        event_id: row.get(1)?,
                        session_id: row.get(2)?,
                        turn_id: row.get(3)?,
                        tool_call_id: row.get(4)?,
                        checkpoint_id: row.get(5)?,
                        event_type: row.get(6)?,
                        timestamp,
                        payload: serde_json::from_str(&payload_json)
                            .unwrap_or(serde_json::Value::Null),
                    },
                ))
            },
        )?;
        Ok(rows.filter_map(|row| row.ok()).collect())
    }

    pub fn upsert_harness_turn(
        &self,
        session_id: &str,
        turn_id: &str,
        status: HarnessTurnStatus,
        model: Option<&str>,
        cwd: Option<&Path>,
        at: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<()> {
        let status = status_string(status);
        let at_s = at.to_rfc3339();
        let cwd_s = cwd.map(|p| p.display().to_string());
        self.conn.execute(
            "INSERT INTO harness_turns
             (turn_id, session_id, status, model, cwd, started_at, completed_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?6)
             ON CONFLICT(turn_id) DO UPDATE SET
                 status = excluded.status,
                 model = COALESCE(excluded.model, harness_turns.model),
                 cwd = COALESCE(excluded.cwd, harness_turns.cwd),
                 completed_at = CASE
                     WHEN excluded.status = 'started' THEN harness_turns.completed_at
                     ELSE excluded.updated_at
                 END,
                 updated_at = excluded.updated_at",
            rusqlite::params![turn_id, session_id, status, model, cwd_s, at_s],
        )?;
        Ok(())
    }

    pub fn save_harness_checkpoint(&self, checkpoint: &HarnessCheckpoint) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO harness_checkpoints
             (checkpoint_id, session_id, turn_id, kind, backend, repo_root,
              git_ref, snapshot_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                checkpoint.checkpoint_id,
                checkpoint.session_id,
                checkpoint.turn_id,
                checkpoint_kind_string(checkpoint.kind),
                checkpoint_backend_string(checkpoint.backend),
                checkpoint
                    .repo_root
                    .as_ref()
                    .map(|p| p.display().to_string()),
                checkpoint.git_ref,
                checkpoint
                    .snapshot
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?,
                checkpoint.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_harness_checkpoints(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<HarnessCheckpoint>> {
        let mut stmt = self.conn.prepare(
            "SELECT checkpoint_id, session_id, turn_id, kind, backend, repo_root,
                    git_ref, snapshot_json, created_at
             FROM harness_checkpoints
             WHERE session_id = ?1
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], checkpoint_from_row)?;
        Ok(rows.filter_map(|row| row.ok()).collect())
    }

    pub fn get_harness_checkpoint(&self, checkpoint_id: &str) -> anyhow::Result<HarnessCheckpoint> {
        let mut stmt = self.conn.prepare(
            "SELECT checkpoint_id, session_id, turn_id, kind, backend, repo_root,
                    git_ref, snapshot_json, created_at
             FROM harness_checkpoints
             WHERE checkpoint_id = ?1",
        )?;
        Ok(stmt.query_row(rusqlite::params![checkpoint_id], checkpoint_from_row)?)
    }
}

/// Summary row returned by `list_sessions` and `search_sessions`.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub title: Option<String>,
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: u32,
}

fn status_string(status: HarnessTurnStatus) -> &'static str {
    match status {
        HarnessTurnStatus::Started => "started",
        HarnessTurnStatus::Completed => "completed",
        HarnessTurnStatus::Cancelled => "cancelled",
        HarnessTurnStatus::Failed => "failed",
    }
}

fn checkpoint_kind_string(kind: CheckpointKind) -> &'static str {
    match kind {
        CheckpointKind::Before => "before",
        CheckpointKind::After => "after",
    }
}

fn checkpoint_backend_string(backend: CheckpointBackend) -> &'static str {
    match backend {
        CheckpointBackend::GitRef => "git_ref",
        CheckpointBackend::FileSnapshot => "file_snapshot",
        CheckpointBackend::MetadataOnly => "metadata_only",
    }
}

fn parse_checkpoint_kind(value: &str) -> CheckpointKind {
    match value {
        "after" => CheckpointKind::After,
        _ => CheckpointKind::Before,
    }
}

fn parse_checkpoint_backend(value: &str) -> CheckpointBackend {
    match value {
        "git_ref" => CheckpointBackend::GitRef,
        "file_snapshot" => CheckpointBackend::FileSnapshot,
        _ => CheckpointBackend::MetadataOnly,
    }
}

fn checkpoint_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<HarnessCheckpoint> {
    let kind: String = row.get(3)?;
    let backend: String = row.get(4)?;
    let repo_root: Option<String> = row.get(5)?;
    let snapshot_json: Option<String> = row.get(7)?;
    let created_at: String = row.get(8)?;
    let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());
    Ok(HarnessCheckpoint {
        checkpoint_id: row.get(0)?,
        session_id: row.get(1)?,
        turn_id: row.get(2)?,
        kind: parse_checkpoint_kind(&kind),
        backend: parse_checkpoint_backend(&backend),
        repo_root: repo_root.map(PathBuf::from),
        git_ref: row.get(6)?,
        snapshot: snapshot_json.and_then(|s| serde_json::from_str(&s).ok()),
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn append_and_replay_harness_events() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("sessions.db");
        let store = SqliteSessionStore::open(&db).unwrap();
        let event = HarnessEvent {
            event_id: "event-1".to_string(),
            session_id: "session-1".to_string(),
            turn_id: Some("turn-1".to_string()),
            tool_call_id: Some("tool-1".to_string()),
            checkpoint_id: None,
            event_type: "tool.started".to_string(),
            timestamp: Utc::now(),
            payload: json!({ "tool_name": "Read" }),
        };
        let sequence = store.append_harness_event(&event).unwrap();
        assert!(sequence > 0);

        let events = store.list_harness_events("session-1", None, 100).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].1.event_id, "event-1");
        assert_eq!(events[0].1.turn_id.as_deref(), Some("turn-1"));
        assert_eq!(events[0].1.tool_call_id.as_deref(), Some("tool-1"));
    }
}
