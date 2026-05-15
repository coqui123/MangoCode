//! Local multi-process coordination for MangoCode sessions.
//!
//! This module stores advisory session presence, work claims, and messages in a
//! user-local SQLite database so independent MangoCode processes can coordinate
//! without requiring network access.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::Settings;

const HEARTBEAT_STALE_SECONDS: i64 = 90;

pub fn process_session_id(session_id: &str) -> String {
    let suffix = format!("@{}", std::process::id());
    if session_id.ends_with(&suffix) {
        session_id.to_string()
    } else {
        format!("{session_id}{suffix}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionPresence {
    pub session_id: String,
    pub pid: u32,
    pub cwd: String,
    pub repo_root: String,
    pub model: String,
    pub title: Option<String>,
    pub started_at: String,
    pub heartbeat_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkClaim {
    pub claim_id: String,
    pub session_id: String,
    pub repo_root: String,
    pub claim_type: String,
    pub scope: String,
    pub summary: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoordinationMessage {
    pub message_id: String,
    pub from_session_id: String,
    pub to_session_id: Option<String>,
    pub repo_root: Option<String>,
    pub body: String,
    pub created_at: String,
    pub read_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConflictWarning {
    pub claim_id: String,
    pub session_id: String,
    pub repo_root: String,
    pub scope: String,
    pub summary: Option<String>,
    pub path: String,
    pub heartbeat_at: String,
}

pub struct CoordinationStore {
    conn: rusqlite::Connection,
}

#[derive(Clone)]
pub struct PresenceHeartbeat {
    state: Arc<Mutex<PresenceHeartbeatState>>,
}

#[derive(Clone)]
struct PresenceHeartbeatState {
    session_id: String,
    cwd: PathBuf,
    model: String,
    title: Option<String>,
}

pub fn spawn_presence_heartbeat(
    session_id: String,
    cwd: PathBuf,
    model: String,
    title: Option<String>,
) -> PresenceHeartbeat {
    let state = Arc::new(Mutex::new(PresenceHeartbeatState {
        session_id,
        cwd,
        model,
        title,
    }));
    let task_state = state.clone();
    tokio::spawn(async move {
        loop {
            if let Ok(store) = CoordinationStore::open_default() {
                if let Ok(heartbeat) = task_state.lock() {
                    let _ = store.register_session(
                        &heartbeat.session_id,
                        &heartbeat.cwd,
                        &heartbeat.model,
                        heartbeat.title.as_deref(),
                    );
                    let _ = store.heartbeat(&heartbeat.session_id);
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        }
    });
    PresenceHeartbeat { state }
}

impl PresenceHeartbeat {
    pub fn update(&self, session_id: String, cwd: PathBuf, model: String, title: Option<String>) {
        let (previous_session, previous_cwd, current_cwd) = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            let previous = state.session_id.clone();
            let previous_cwd = state.cwd.clone();
            state.session_id = session_id.clone();
            state.cwd = cwd.clone();
            state.model = model;
            state.title = title;
            (previous, previous_cwd, cwd)
        };
        if previous_session != session_id {
            if let Ok(store) = CoordinationStore::open_default() {
                let _ = store.unregister_session(&previous_session);
            }
        } else if canonical_project_root(&previous_cwd) != canonical_project_root(&current_cwd) {
            if let Ok(store) = CoordinationStore::open_default() {
                let _ = store.release_claims(&session_id, None);
            }
        }
    }
}

impl CoordinationStore {
    pub fn open_default() -> anyhow::Result<Self> {
        Self::open(&default_db_path())
    }

    pub fn open(db_path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = rusqlite::Connection::open(db_path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS coord_sessions (
                session_id   TEXT PRIMARY KEY,
                pid          INTEGER NOT NULL,
                cwd          TEXT NOT NULL,
                repo_root    TEXT NOT NULL,
                model        TEXT NOT NULL DEFAULT '',
                title        TEXT,
                started_at   TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_coord_sessions_repo
                ON coord_sessions(repo_root, heartbeat_at);

            CREATE TABLE IF NOT EXISTS coord_claims (
                claim_id    TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                repo_root   TEXT NOT NULL,
                claim_type  TEXT NOT NULL,
                scope       TEXT NOT NULL,
                summary     TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_coord_claims_repo
                ON coord_claims(repo_root, updated_at);
            CREATE INDEX IF NOT EXISTS idx_coord_claims_session
                ON coord_claims(session_id, updated_at);

            CREATE TABLE IF NOT EXISTS coord_messages (
                message_id       TEXT PRIMARY KEY,
                from_session_id  TEXT NOT NULL,
                to_session_id    TEXT,
                repo_root        TEXT,
                body             TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                read_at          TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_coord_messages_target
                ON coord_messages(to_session_id, repo_root, created_at);

            CREATE TABLE IF NOT EXISTS coord_message_reads (
                message_id  TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                read_at     TEXT NOT NULL,
                PRIMARY KEY(message_id, session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_coord_message_reads_session
                ON coord_message_reads(session_id, read_at);

            INSERT OR IGNORE INTO coord_message_reads (message_id, session_id, read_at)
            SELECT message_id, to_session_id, read_at
            FROM coord_messages
            WHERE read_at IS NOT NULL AND to_session_id IS NOT NULL;
            ",
        )?;
        Ok(Self { conn })
    }

    pub fn register_session(
        &self,
        session_id: &str,
        cwd: &Path,
        model: &str,
        title: Option<&str>,
    ) -> anyhow::Result<SessionPresence> {
        let now = now_string();
        let repo_root = canonical_project_root(cwd);
        let cwd_s = canonical_path_string(cwd);
        let repo_s = repo_root.to_string_lossy().to_string();
        let pid = std::process::id();
        self.conn.execute(
            "INSERT INTO coord_sessions
             (session_id, pid, cwd, repo_root, model, title, started_at, heartbeat_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
             ON CONFLICT(session_id) DO UPDATE SET
                pid = excluded.pid,
                cwd = excluded.cwd,
                repo_root = excluded.repo_root,
                model = COALESCE(NULLIF(excluded.model, ''), coord_sessions.model),
                title = COALESCE(excluded.title, coord_sessions.title),
                heartbeat_at = excluded.heartbeat_at",
            rusqlite::params![session_id, pid, cwd_s, repo_s, model, title, now],
        )?;
        Ok(SessionPresence {
            session_id: session_id.to_string(),
            pid,
            cwd: cwd_s,
            repo_root: repo_s,
            model: model.to_string(),
            title: title.map(str::to_string),
            started_at: now.clone(),
            heartbeat_at: now,
        })
    }

    pub fn heartbeat(&self, session_id: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE coord_sessions SET heartbeat_at = ?1 WHERE session_id = ?2",
            rusqlite::params![now_string(), session_id],
        )?;
        Ok(())
    }

    pub fn unregister_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "DELETE FROM coord_sessions WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        self.release_claims(session_id, None)?;
        Ok(())
    }

    pub fn list_sessions(&self, repo_root: Option<&Path>) -> anyhow::Result<Vec<SessionPresence>> {
        let cutoff = stale_cutoff();
        let repo = repo_root.map(|p| canonical_project_root(p).to_string_lossy().to_string());
        let sql = if repo.is_some() {
            "SELECT session_id, pid, cwd, repo_root, model, title, started_at, heartbeat_at
             FROM coord_sessions
             WHERE repo_root = ?1 AND heartbeat_at >= ?2
             ORDER BY heartbeat_at DESC"
        } else {
            "SELECT session_id, pid, cwd, repo_root, model, title, started_at, heartbeat_at
             FROM coord_sessions
             WHERE heartbeat_at >= ?1
             ORDER BY heartbeat_at DESC"
        };
        let mut stmt = self.conn.prepare(sql)?;
        if let Some(repo) = repo {
            let rows = stmt.query_map(rusqlite::params![repo, cutoff], session_from_row)?;
            collect_rows(rows)
        } else {
            let rows = stmt.query_map(rusqlite::params![cutoff], session_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn create_claim(
        &self,
        session_id: &str,
        cwd: &Path,
        claim_type: &str,
        scope: &str,
        summary: Option<&str>,
    ) -> anyhow::Result<WorkClaim> {
        let now = now_string();
        let repo_root_path = canonical_project_root(cwd);
        let repo_root = repo_root_path.to_string_lossy().to_string();
        let claim_id = Uuid::new_v4().to_string();
        let scope = normalize_claim_scope(&repo_root_path, scope);
        self.conn.execute(
            "INSERT INTO coord_claims
             (claim_id, session_id, repo_root, claim_type, scope, summary, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            rusqlite::params![claim_id, session_id, repo_root, claim_type, scope, summary, now],
        )?;
        Ok(WorkClaim {
            claim_id,
            session_id: session_id.to_string(),
            repo_root,
            claim_type: claim_type.to_string(),
            scope,
            summary: summary.map(str::to_string),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub fn release_claims(
        &self,
        session_id: &str,
        claim_id: Option<&str>,
    ) -> anyhow::Result<usize> {
        let changed = if let Some(claim_id) = claim_id {
            self.conn.execute(
                "DELETE FROM coord_claims WHERE session_id = ?1 AND claim_id = ?2",
                rusqlite::params![session_id, claim_id],
            )?
        } else {
            self.conn.execute(
                "DELETE FROM coord_claims WHERE session_id = ?1",
                rusqlite::params![session_id],
            )?
        };
        Ok(changed)
    }

    pub fn list_claims(&self, repo_root: Option<&Path>) -> anyhow::Result<Vec<WorkClaim>> {
        let cutoff = stale_cutoff();
        let repo = repo_root.map(|p| canonical_project_root(p).to_string_lossy().to_string());
        let sql = if repo.is_some() {
            "SELECT c.claim_id, c.session_id, c.repo_root, c.claim_type, c.scope,
                    c.summary, c.created_at, c.updated_at
             FROM coord_claims c
             JOIN coord_sessions s ON s.session_id = c.session_id
             WHERE c.repo_root = ?1 AND s.heartbeat_at >= ?2
             ORDER BY c.updated_at DESC"
        } else {
            "SELECT c.claim_id, c.session_id, c.repo_root, c.claim_type, c.scope,
                    c.summary, c.created_at, c.updated_at
             FROM coord_claims c
             JOIN coord_sessions s ON s.session_id = c.session_id
             WHERE s.heartbeat_at >= ?1
             ORDER BY c.updated_at DESC"
        };
        let mut stmt = self.conn.prepare(sql)?;
        if let Some(repo) = repo {
            let rows = stmt.query_map(rusqlite::params![repo, cutoff], claim_from_row)?;
            collect_rows(rows)
        } else {
            let rows = stmt.query_map(rusqlite::params![cutoff], claim_from_row)?;
            collect_rows(rows)
        }
    }

    pub fn find_conflicts(
        &self,
        session_id: &str,
        cwd: &Path,
        paths: &[PathBuf],
    ) -> anyhow::Result<Vec<ConflictWarning>> {
        let repo_root = canonical_project_root(cwd);
        let repo_s = repo_root.to_string_lossy().to_string();
        let cutoff = stale_cutoff();
        let mut stmt = self.conn.prepare(
            "SELECT c.claim_id, c.session_id, c.repo_root, c.claim_type, c.scope,
                    c.summary, c.created_at, c.updated_at, s.heartbeat_at
             FROM coord_claims c
             JOIN coord_sessions s ON s.session_id = c.session_id
             WHERE c.repo_root = ?1 AND c.session_id != ?2 AND s.heartbeat_at >= ?3",
        )?;
        let rows = stmt.query_map(rusqlite::params![repo_s, session_id, cutoff], |row| {
            Ok((claim_from_row(row)?, row.get::<_, String>(8)?))
        })?;
        let claims = collect_rows(rows)?;
        let relative_paths: Vec<String> = paths
            .iter()
            .map(|p| relative_scope(&repo_root, p))
            .collect();
        let mut conflicts = Vec::new();
        for (claim, heartbeat_at) in claims {
            for path in &relative_paths {
                if scopes_overlap(&claim.scope, path) {
                    conflicts.push(ConflictWarning {
                        claim_id: claim.claim_id.clone(),
                        session_id: claim.session_id.clone(),
                        repo_root: claim.repo_root.clone(),
                        scope: claim.scope.clone(),
                        summary: claim.summary.clone(),
                        path: path.clone(),
                        heartbeat_at: heartbeat_at.clone(),
                    });
                }
            }
        }
        conflicts.sort_by(|a, b| a.session_id.cmp(&b.session_id).then(a.path.cmp(&b.path)));
        conflicts.dedup();
        Ok(conflicts)
    }

    pub fn find_scope_conflicts(
        &self,
        session_id: &str,
        cwd: &Path,
        scope: &str,
    ) -> anyhow::Result<Vec<ConflictWarning>> {
        let repo_root = canonical_project_root(cwd);
        let repo_s = repo_root.to_string_lossy().to_string();
        let cutoff = stale_cutoff();
        let attempted_scope = normalize_claim_scope(&repo_root, scope);
        let mut stmt = self.conn.prepare(
            "SELECT c.claim_id, c.session_id, c.repo_root, c.claim_type, c.scope,
                    c.summary, c.created_at, c.updated_at, s.heartbeat_at
             FROM coord_claims c
             JOIN coord_sessions s ON s.session_id = c.session_id
             WHERE c.repo_root = ?1 AND c.session_id != ?2 AND s.heartbeat_at >= ?3",
        )?;
        let rows = stmt.query_map(rusqlite::params![repo_s, session_id, cutoff], |row| {
            Ok((claim_from_row(row)?, row.get::<_, String>(8)?))
        })?;
        let claims = collect_rows(rows)?;
        let mut conflicts = Vec::new();
        for (claim, heartbeat_at) in claims {
            if scopes_potentially_overlap(&claim.scope, &attempted_scope) {
                conflicts.push(ConflictWarning {
                    claim_id: claim.claim_id,
                    session_id: claim.session_id,
                    repo_root: claim.repo_root,
                    scope: claim.scope,
                    summary: claim.summary,
                    path: attempted_scope.clone(),
                    heartbeat_at,
                });
            }
        }
        conflicts.sort_by(|a, b| a.session_id.cmp(&b.session_id).then(a.path.cmp(&b.path)));
        conflicts.dedup();
        Ok(conflicts)
    }

    pub fn send_message(
        &self,
        from_session_id: &str,
        to_session_id: Option<&str>,
        repo_root: Option<&Path>,
        body: &str,
    ) -> anyhow::Result<CoordinationMessage> {
        let now = now_string();
        let message_id = Uuid::new_v4().to_string();
        let repo_s = repo_root.map(|p| canonical_project_root(p).to_string_lossy().to_string());
        self.conn.execute(
            "INSERT INTO coord_messages
             (message_id, from_session_id, to_session_id, repo_root, body, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                message_id,
                from_session_id,
                to_session_id,
                repo_s,
                body,
                now
            ],
        )?;
        Ok(CoordinationMessage {
            message_id,
            from_session_id: from_session_id.to_string(),
            to_session_id: to_session_id.map(str::to_string),
            repo_root: repo_s,
            body: body.to_string(),
            created_at: now,
            read_at: None,
        })
    }

    pub fn inbox(
        &self,
        session_id: &str,
        cwd: &Path,
        mark_read: bool,
    ) -> anyhow::Result<Vec<CoordinationMessage>> {
        let repo_root = canonical_project_root(cwd).to_string_lossy().to_string();
        let mut stmt = self.conn.prepare(
            "SELECT m.message_id, m.from_session_id, m.to_session_id, m.repo_root,
                    m.body, m.created_at, r.read_at
             FROM coord_messages m
             LEFT JOIN coord_message_reads r
                ON r.message_id = m.message_id AND r.session_id = ?1
             WHERE r.read_at IS NULL
               AND (
                    m.to_session_id = ?1
                    OR (m.to_session_id IS NULL AND m.repo_root = ?2 AND m.from_session_id != ?1)
               )
             ORDER BY m.created_at ASC
             LIMIT 50",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id, repo_root], message_from_row)?;
        let messages = collect_rows(rows)?;
        if mark_read && !messages.is_empty() {
            let now = now_string();
            for message in &messages {
                self.conn.execute(
                    "INSERT OR REPLACE INTO coord_message_reads
                     (message_id, session_id, read_at)
                     VALUES (?1, ?2, ?3)",
                    rusqlite::params![message.message_id, session_id, now],
                )?;
            }
        }
        Ok(messages)
    }
}

pub fn default_db_path() -> PathBuf {
    Settings::config_dir().join("coordination.db")
}

pub fn canonical_project_root(cwd: &Path) -> PathBuf {
    let start = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let mut cur = if start.is_file() {
        start.parent().unwrap_or(&start).to_path_buf()
    } else {
        start
    };
    loop {
        if cur.join(".git").exists() {
            return cur;
        }
        if !cur.pop() {
            return cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
        }
    }
}

pub fn relative_scope(repo_root: &Path, path: &Path) -> String {
    let raw_absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    let absolute = raw_absolute.canonicalize().unwrap_or(raw_absolute);
    let absolute_text = clean_path_text(&absolute);
    let repo_text = clean_path_text(repo_root);
    let absolute_match = match_scope_text(&absolute_text);
    let repo_match = match_scope_text(&repo_text);
    if absolute_match == repo_match {
        return ".".to_string();
    }
    let repo_prefix = format!("{repo_match}/");
    if absolute_match.starts_with(&repo_prefix) {
        return absolute_text[repo_text.len() + 1..].to_string();
    }
    absolute
        .strip_prefix(repo_root)
        .unwrap_or(&absolute)
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

fn canonical_path_string(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string()
}

fn normalize_scope(scope: &str) -> String {
    strip_windows_verbatim_prefix(&scope.trim().replace('\\', "/"))
        .trim_start_matches("./")
        .to_string()
}

fn normalize_claim_scope(repo_root: &Path, scope: &str) -> String {
    let normalized = normalize_scope(scope);
    let repo_text = clean_path_text(repo_root);
    let normalized_match = match_scope_text(&normalized);
    let repo_match = match_scope_text(&repo_text);
    if normalized_match == repo_match {
        return ".".to_string();
    }
    let repo_prefix = format!("{repo_match}/");
    if normalized_match.starts_with(&repo_prefix) {
        return normalized[repo_text.len() + 1..].to_string();
    }

    let scope_path = Path::new(scope);
    if scope_path.is_absolute() {
        return relative_scope(repo_root, scope_path);
    }

    normalized
}

fn scopes_overlap(claim_scope: &str, path: &str) -> bool {
    let claim = match_scope_text(&normalize_scope(claim_scope));
    let path = match_scope_text(&normalize_scope(path));
    if claim.is_empty() || claim == "." {
        return true;
    }
    if claim == path {
        return true;
    }
    if path.starts_with(claim.trim_end_matches('/'))
        && path[claim.trim_end_matches('/').len()..].starts_with('/')
    {
        return true;
    }
    if claim.ends_with("/**") {
        let prefix = claim.trim_end_matches("/**");
        return path == prefix || path.starts_with(&format!("{prefix}/"));
    }
    if claim.contains('*') || claim.contains('?') || claim.contains('[') {
        if let Ok(pattern) = glob::Pattern::new(&claim) {
            return pattern.matches_with(
                &path,
                glob::MatchOptions {
                    case_sensitive: !cfg!(windows),
                    require_literal_separator: true,
                    require_literal_leading_dot: false,
                },
            );
        }
    }
    false
}

fn scopes_potentially_overlap(left: &str, right: &str) -> bool {
    let left = match_scope_text(&normalize_scope(left));
    let right = match_scope_text(&normalize_scope(right));
    if left.is_empty() || left == "." || right.is_empty() || right == "." {
        return true;
    }
    if scopes_overlap(&left, &right) || scopes_overlap(&right, &left) {
        return true;
    }
    scope_static_prefix_overlaps(&left, &right)
}

fn scope_static_prefix_overlaps(left: &str, right: &str) -> bool {
    let left = static_scope_prefix(left);
    let right = static_scope_prefix(right);
    if left.is_empty() || right.is_empty() {
        return true;
    }
    left == right
        || left.starts_with(&format!("{right}/"))
        || right.starts_with(&format!("{left}/"))
}

fn static_scope_prefix(scope: &str) -> &str {
    let wildcard_idx = scope.find(['*', '?', '[']).unwrap_or(scope.len());
    scope[..wildcard_idx].trim_end_matches('/')
}

fn clean_path_text(path: &Path) -> String {
    strip_windows_verbatim_prefix(&path.to_string_lossy().replace('\\', "/"))
        .trim_end_matches('/')
        .to_string()
}

fn strip_windows_verbatim_prefix(path: &str) -> &str {
    path.strip_prefix("//?/").unwrap_or(path)
}

fn match_scope_text(scope: &str) -> String {
    if cfg!(windows) {
        scope.to_ascii_lowercase()
    } else {
        scope.to_string()
    }
}

fn now_string() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true)
}

fn stale_cutoff() -> String {
    (Utc::now() - Duration::seconds(HEARTBEAT_STALE_SECONDS))
        .to_rfc3339_opts(SecondsFormat::Nanos, true)
}

fn collect_rows<T>(rows: impl Iterator<Item = rusqlite::Result<T>>) -> anyhow::Result<Vec<T>> {
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn session_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionPresence> {
    let pid: i64 = row.get(1)?;
    Ok(SessionPresence {
        session_id: row.get(0)?,
        pid: pid.max(0) as u32,
        cwd: row.get(2)?,
        repo_root: row.get(3)?,
        model: row.get(4)?,
        title: row.get(5)?,
        started_at: row.get(6)?,
        heartbeat_at: row.get(7)?,
    })
}

fn claim_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkClaim> {
    Ok(WorkClaim {
        claim_id: row.get(0)?,
        session_id: row.get(1)?,
        repo_root: row.get(2)?,
        claim_type: row.get(3)?,
        scope: row.get(4)?,
        summary: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

fn message_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CoordinationMessage> {
    Ok(CoordinationMessage {
        message_id: row.get(0)?,
        from_session_id: row.get(1)?,
        to_session_id: row.get(2)?,
        repo_root: row.get(3)?,
        body: row.get(4)?,
        created_at: row.get(5)?,
        read_at: row.get(6)?,
    })
}

#[allow(dead_code)]
fn parse_time(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_overlap_handles_exact_prefix_and_glob() {
        assert!(scopes_overlap("src/lib.rs", "src/lib.rs"));
        assert!(scopes_overlap("src", "src/lib.rs"));
        assert!(scopes_overlap("src/**", "src/lib.rs"));
        assert!(scopes_overlap("src/*.rs", "src/lib.rs"));
        if cfg!(windows) {
            assert!(scopes_overlap("SRC", "src/lib.rs"));
            assert!(scopes_overlap("SRC/*.RS", "src/lib.rs"));
        }
        assert!(!scopes_overlap("src/*.rs", "src/bin/main.rs"));
        assert!(!scopes_overlap("src/*.md", "src/lib.rs"));
        assert!(!scopes_overlap("docs", "src/lib.rs"));
    }

    #[test]
    fn scope_conflict_detection_is_symmetric_for_claims() {
        assert!(scopes_potentially_overlap("src/**", "src/lib.rs"));
        assert!(scopes_potentially_overlap("src/lib.rs", "src/**"));
        assert!(scopes_potentially_overlap("src/*.rs", "src/lib.rs"));
        assert!(scopes_potentially_overlap("src/*.rs", "src/*.md"));
        assert!(!scopes_potentially_overlap("docs/**", "src/lib.rs"));
    }

    #[test]
    fn store_finds_live_claim_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .register_session("a", dir.path(), "model", Some("writer"))
            .unwrap();
        store
            .register_session("b", dir.path(), "model", Some("tester"))
            .unwrap();
        store
            .create_claim("a", dir.path(), "edit", "src/**", Some("rewrite"))
            .unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let conflicts = store
            .find_conflicts("b", dir.path(), &[dir.path().join("src/lib.rs")])
            .unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].session_id, "a");
    }

    #[test]
    fn store_finds_live_scope_claim_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .register_session("a", dir.path(), "model", Some("writer"))
            .unwrap();
        store
            .register_session("b", dir.path(), "model", Some("tester"))
            .unwrap();
        store
            .create_claim("a", dir.path(), "edit", "src/**", Some("rewrite"))
            .unwrap();

        let conflicts = store
            .find_scope_conflicts("b", dir.path(), "src/lib.rs")
            .unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].session_id, "a");

        assert!(store
            .find_scope_conflicts("b", dir.path(), "docs/**")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn absolute_claim_scopes_are_matched_repo_relative() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .register_session("a", dir.path(), "model", Some("writer"))
            .unwrap();
        store
            .register_session("b", dir.path(), "model", Some("tester"))
            .unwrap();
        let absolute_scope = dir.path().join("src").join("lib.rs");
        store
            .create_claim(
                "a",
                dir.path(),
                "edit",
                &absolute_scope.to_string_lossy(),
                Some("absolute"),
            )
            .unwrap();

        let conflicts = store
            .find_conflicts("b", dir.path(), &[dir.path().join("src/lib.rs")])
            .unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].scope, "src/lib.rs");
    }

    #[test]
    fn inbox_reads_direct_and_repo_broadcast_messages() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store.send_message("a", Some("b"), None, "direct").unwrap();
        store
            .send_message("a", None, Some(dir.path()), "repo")
            .unwrap();

        let inbox = store.inbox("b", dir.path(), true).unwrap();
        assert_eq!(inbox.len(), 2);
        assert!(store.inbox("b", dir.path(), false).unwrap().is_empty());
    }

    #[test]
    fn inbox_does_not_return_own_repo_broadcasts() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .send_message("a", None, Some(dir.path()), "repo")
            .unwrap();

        assert!(store.inbox("a", dir.path(), false).unwrap().is_empty());
        assert_eq!(store.inbox("b", dir.path(), false).unwrap().len(), 1);
    }

    #[test]
    fn broadcast_read_receipts_are_per_recipient() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .send_message("a", None, Some(dir.path()), "repo")
            .unwrap();

        assert_eq!(store.inbox("b", dir.path(), true).unwrap().len(), 1);
        assert!(store.inbox("b", dir.path(), false).unwrap().is_empty());
        assert_eq!(store.inbox("c", dir.path(), false).unwrap().len(), 1);
    }

    #[test]
    fn heartbeat_refresh_does_not_erase_existing_metadata() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let db = dir.path().join("coordination.db");
        let store = CoordinationStore::open(&db).unwrap();
        store
            .register_session("a", dir.path(), "model-a", Some("title-a"))
            .unwrap();
        store.register_session("a", dir.path(), "", None).unwrap();

        let sessions = store.list_sessions(Some(dir.path())).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].model, "model-a");
        assert_eq!(sessions[0].title.as_deref(), Some("title-a"));
    }

    #[test]
    fn process_session_id_distinguishes_local_process_identity() {
        let base = "conversation-session";
        let process_id = process_session_id(base);
        assert_ne!(process_id, base);
        assert!(process_id.starts_with(base));
        assert_eq!(process_session_id(&process_id), process_id);
    }
}
