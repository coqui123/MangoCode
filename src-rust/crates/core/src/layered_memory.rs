//! SQLite-backed layered memory primitives.
//!
//! This is the durable storage layer for the Claude-Mem/MemPalace-inspired
//! memory upgrade. Retrieval starts lexical and can later grow a vector index
//! without changing callers.

use once_cell::sync::Lazy;
#[cfg(feature = "fastembed")]
use once_cell::sync::OnceCell;
#[cfg(feature = "fastembed")]
use parking_lot::Mutex;
use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_FASTEMBED_MODEL: &str = "BAAI/bge-base-en-v1.5";

#[derive(Debug, Clone)]
struct EmbeddingRuntimeConfig {
    provider: String,
    model: String,
}

static EMBEDDING_RUNTIME_CONFIG: Lazy<RwLock<EmbeddingRuntimeConfig>> = Lazy::new(|| {
    RwLock::new(EmbeddingRuntimeConfig {
        provider: default_embeddings_provider().to_string(),
        model: DEFAULT_FASTEMBED_MODEL.to_string(),
    })
});

pub fn configure_embeddings(provider: &str, model: &str) {
    let mut config = EMBEDDING_RUNTIME_CONFIG.write();
    config.provider = if provider.trim().is_empty() {
        default_embeddings_provider().to_string()
    } else {
        provider.trim().to_ascii_lowercase()
    };
    config.model = if model.trim().is_empty() {
        DEFAULT_FASTEMBED_MODEL.to_string()
    } else {
        model.trim().to_string()
    };
}

#[derive(Debug, Clone, Copy)]
enum EmbeddingPurpose {
    Passage,
    Query,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClass {
    ProjectFact,
    UserPreference,
    Decision,
    ExternalDoc,
}

impl MemoryClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProjectFact => "project_fact",
            Self::UserPreference => "user_preference",
            Self::Decision => "decision",
            Self::ExternalDoc => "external_doc",
        }
    }

    fn from_str(value: &str) -> Self {
        match value {
            "user_preference" => Self::UserPreference,
            "decision" => Self::Decision,
            "external_doc" => Self::ExternalDoc,
            _ => Self::ProjectFact,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub class: MemoryClass,
    pub content: String,
    pub source: Option<String>,
    pub project: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct LayeredMemoryStore {
    conn: Connection,
}

impl LayeredMemoryStore {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    pub fn init(&self) -> anyhow::Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                project TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_class ON memories(class);
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
            "#,
        )?;
        let _ = self
            .conn
            .execute("ALTER TABLE memories ADD COLUMN vector TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE memories ADD COLUMN embedding TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE memories ADD COLUMN embedding_model TEXT", []);
        Ok(())
    }

    pub fn insert(
        &self,
        class: MemoryClass,
        content: &str,
        source: Option<&str>,
        project: Option<&str>,
    ) -> anyhow::Result<i64> {
        let cleaned = strip_private_sections(content);
        if cleaned.trim().is_empty() || looks_like_secret(&cleaned) {
            anyhow::bail!("memory skipped by privacy filter");
        }
        let now = chrono::Utc::now().to_rfc3339();
        let vector = encode_vector(&hashed_semantic_vector(&cleaned));
        let dense_embedding = local_embedding_for_text(&cleaned, EmbeddingPurpose::Passage)
            .ok()
            .filter(|embedding| !embedding.is_empty())
            .map(|embedding| encode_dense_embedding(&embedding));
        let embedding_model = dense_embedding
            .as_ref()
            .map(|_| local_embedding_model_name());
        if let Some(existing_id) = self
            .conn
            .query_row(
                "SELECT id FROM memories
                 WHERE class = ?1 AND content = ?2 AND COALESCE(project, '') = COALESCE(?3, '')
                 LIMIT 1",
                params![class.as_str(), cleaned, project],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
        {
            self.conn.execute(
                "UPDATE memories
                 SET source = COALESCE(?1, source),
                     vector = COALESCE(vector, ?2),
                     embedding = COALESCE(?3, embedding),
                     embedding_model = COALESCE(?4, embedding_model),
                     updated_at = ?5
                 WHERE id = ?6",
                params![
                    source,
                    vector,
                    dense_embedding,
                    embedding_model,
                    now,
                    existing_id
                ],
            )?;
            return Ok(existing_id);
        }
        self.conn.execute(
            "INSERT INTO memories
             (class, content, source, project, created_at, updated_at, vector, embedding, embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7, ?8)",
            params![
                class.as_str(),
                cleaned,
                source,
                project,
                now,
                vector,
                dense_embedding,
                embedding_model
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn manifest(&self, limit: usize) -> anyhow::Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT class, COUNT(*) FROM memories GROUP BY class ORDER BY class LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut lines = Vec::new();
        for row in rows {
            let (class, count) = row?;
            lines.push(format!("- {}: {} memories", class, count));
        }
        Ok(lines.join("\n"))
    }

    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryRecord>> {
        let terms = tokenize(query);
        let query_vector = hashed_semantic_vector(query);
        let query_embedding = local_embedding_for_text(query, EmbeddingPurpose::Query).ok();
        if terms.is_empty()
            && query_vector.is_empty()
            && query_embedding
                .as_ref()
                .is_none_or(|embedding| embedding.is_empty())
        {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT id, class, content, source, project, created_at, updated_at, vector, embedding
             FROM memories
             ORDER BY updated_at DESC
             LIMIT 500",
        )?;
        let rows = stmt.query_map([], read_scored_record)?;
        let mut scored = Vec::new();
        for row in rows {
            let (record, stored_vector, stored_embedding) = row?;
            let lower = record.content.to_ascii_lowercase();
            let lexical = terms
                .iter()
                .filter(|term| lower.contains(term.as_str()))
                .count() as f32;
            let semantic = stored_vector
                .as_deref()
                .map(decode_vector)
                .map(|vector| cosine_similarity(&query_vector, &vector))
                .unwrap_or_else(|| {
                    cosine_similarity(&query_vector, &hashed_semantic_vector(&record.content))
                });
            let dense = match (&query_embedding, stored_embedding.as_deref()) {
                (Some(query_embedding), Some(stored_embedding)) => cosine_similarity_dense(
                    query_embedding,
                    &decode_dense_embedding(stored_embedding),
                ),
                _ => 0.0,
            };
            let score = lexical * 2.0 + semantic + dense * 2.5;
            if score > 0.12 {
                scored.push((score, record));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(_, record)| record)
            .collect())
    }

    pub fn review(&self, limit: usize) -> anyhow::Result<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, class, content, source, project, created_at, updated_at
             FROM memories
             ORDER BY updated_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], read_record)?;
        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }
        Ok(records)
    }

    pub fn delete(&self, id: i64) -> anyhow::Result<bool> {
        let affected = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }
}

pub fn project_memory_db_path(project_root: &Path) -> PathBuf {
    project_root
        .join(".mangocode")
        .join("layered-memory.sqlite")
}

pub fn capture_explicit_memories(
    store: &LayeredMemoryStore,
    text: &str,
    source: Option<&str>,
    project: Option<&str>,
) -> Vec<i64> {
    let mut ids = Vec::new();
    for (class, content) in extract_explicit_memory_candidates(text) {
        if let Ok(id) = store.insert(class, &content, source, project) {
            ids.push(id);
        }
    }
    ids
}

pub fn format_memory_records(records: &[MemoryRecord]) -> String {
    records
        .iter()
        .map(|record| {
            format!(
                "- #{} [{}] {}{}",
                record.id,
                record.class.as_str(),
                record.content,
                record
                    .source
                    .as_ref()
                    .map(|source| format!(" (source: {})", source))
                    .unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn read_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    Ok(MemoryRecord {
        id: row.get(0)?,
        class: MemoryClass::from_str(&row.get::<_, String>(1)?),
        content: row.get(2)?,
        source: row.get(3)?,
        project: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

fn read_scored_record(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<(MemoryRecord, Option<String>, Option<String>)> {
    Ok((
        MemoryRecord {
            id: row.get(0)?,
            class: MemoryClass::from_str(&row.get::<_, String>(1)?),
            content: row.get(2)?,
            source: row.get(3)?,
            project: row.get(4)?,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        },
        row.get(7)?,
        row.get(8)?,
    ))
}

pub fn strip_private_sections(input: &str) -> String {
    let mut out = String::new();
    let mut rest = input;
    loop {
        let Some(start) = rest.find("<private>") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        let after_start = &rest[start + "<private>".len()..];
        let Some(end) = after_start.find("</private>") else {
            break;
        };
        rest = &after_start[end + "</private>".len()..];
    }
    out
}

pub fn looks_like_secret(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.contains("api_key=")
        || lower.contains("apikey=")
        || lower.contains("secret_key")
        || lower.contains("access_token")
        || lower.contains("refresh_token")
        || lower.contains("bearer ")
        || lower.contains("-----begin ")
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
        .filter(|part| part.len() >= 3)
        .map(str::to_string)
        .collect()
}

fn hashed_semantic_vector(text: &str) -> Vec<(u16, f32)> {
    const DIMS: usize = 384;
    let lower = text.to_ascii_lowercase();
    let tokens = tokenize(&lower);
    let mut weights = vec![0f32; DIMS];

    for token in tokens {
        add_hashed_feature(&mut weights, &token, 1.0);
        for window in token.as_bytes().windows(3) {
            if let Ok(trigram) = std::str::from_utf8(window) {
                add_hashed_feature(&mut weights, trigram, 0.45);
            }
        }
    }

    let norm = weights.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut weights {
            *value /= norm;
        }
    }

    weights
        .into_iter()
        .enumerate()
        .filter(|(_, value)| value.abs() > 0.0001)
        .map(|(idx, value)| (idx as u16, value))
        .collect()
}

fn add_hashed_feature(weights: &mut [f32], feature: &str, weight: f32) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    feature.hash(&mut h);
    let hash = h.finish();
    let idx = (hash as usize) % weights.len();
    let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
    weights[idx] += weight * sign;
}

fn cosine_similarity(a: &[(u16, f32)], b: &[(u16, f32)]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let b_map = b
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    a.iter()
        .filter_map(|(idx, av)| b_map.get(idx).map(|bv| av * bv))
        .sum::<f32>()
}

fn cosine_similarity_dense(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut a_norm = 0.0;
    let mut b_norm = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        dot += av * bv;
        a_norm += av * av;
        b_norm += bv * bv;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm.sqrt() * b_norm.sqrt())
    }
}

fn encode_vector(vector: &[(u16, f32)]) -> String {
    vector
        .iter()
        .map(|(idx, value)| format!("{}:{:.5}", idx, value))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_vector(value: &str) -> Vec<(u16, f32)> {
    value
        .split(',')
        .filter_map(|part| {
            let (idx, weight) = part.split_once(':')?;
            Some((idx.parse().ok()?, weight.parse().ok()?))
        })
        .collect()
}

fn encode_dense_embedding(embedding: &[f32]) -> String {
    embedding
        .iter()
        .map(|value| format!("{:.6}", value))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_dense_embedding(value: &str) -> Vec<f32> {
    value
        .split([',', ' ', '\n', '\t'])
        .filter_map(|part| part.trim().parse::<f32>().ok())
        .collect()
}

fn local_embedding_for_text(text: &str, purpose: EmbeddingPurpose) -> anyhow::Result<Vec<f32>> {
    if embeddings_provider() == "fastembed" {
        #[cfg(feature = "fastembed")]
        if let Ok(embedding) = fastembed_embedding_for_text(text, purpose) {
            return Ok(embedding);
        }
    }
    command_embedding_for_text(text)
}

fn command_embedding_for_text(text: &str) -> anyhow::Result<Vec<f32>> {
    let command = std::env::var("MANGOCODE_EMBEDDINGS_COMMAND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("MANGOCODE_EMBEDDINGS_COMMAND is not set"))?;
    let mut parts = command.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("MANGOCODE_EMBEDDINGS_COMMAND is empty"))?;
    let args = parts.collect::<Vec<_>>();
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(text.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!(
            "embedding command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_embedding_output(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(feature = "fastembed")]
struct FastEmbedRuntime {
    model_name: String,
    model: fastembed::TextEmbedding,
}

#[cfg(feature = "fastembed")]
static FASTEMBED_RUNTIME: OnceCell<Mutex<Option<FastEmbedRuntime>>> = OnceCell::new();

#[cfg(feature = "fastembed")]
fn fastembed_embedding_for_text(text: &str, purpose: EmbeddingPurpose) -> anyhow::Result<Vec<f32>> {
    let model_name = local_embedding_model_name();
    let runtime = FASTEMBED_RUNTIME.get_or_init(|| Mutex::new(None));
    let mut guard = runtime.lock();
    if guard
        .as_ref()
        .is_none_or(|runtime| runtime.model_name != model_name)
    {
        let model = parse_fastembed_model(&model_name)?;
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mangocode")
            .join("embeddings")
            .join("fastembed");
        let options = fastembed::InitOptions::new(model)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false);
        *guard = Some(FastEmbedRuntime {
            model_name: model_name.clone(),
            model: fastembed::TextEmbedding::try_new(options)?,
        });
    }
    let runtime = guard
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("fastembed runtime did not initialize"))?;
    let prefixed = match purpose {
        EmbeddingPurpose::Passage => format!("passage: {}", text),
        EmbeddingPurpose::Query => format!("query: {}", text),
    };
    let embeddings = runtime.model.embed(vec![prefixed], None)?;
    embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("fastembed returned no embedding"))
}

#[cfg(feature = "fastembed")]
fn parse_fastembed_model(model_name: &str) -> anyhow::Result<fastembed::EmbeddingModel> {
    match model_name.trim() {
        "" | "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" | "BGEBaseENV15" => {
            Ok(fastembed::EmbeddingModel::BGEBaseENV15)
        }
        "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" | "BGESmallENV15" => {
            Ok(fastembed::EmbeddingModel::BGESmallENV15)
        }
        "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" | "BGELargeENV15" => {
            Ok(fastembed::EmbeddingModel::BGELargeENV15)
        }
        other => other
            .parse::<fastembed::EmbeddingModel>()
            .map_err(|e| anyhow::anyhow!("unsupported fastembed model '{}': {}", other, e)),
    }
}

fn parse_embedding_output(output: &str) -> anyhow::Result<Vec<f32>> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        anyhow::bail!("embedding command returned no output");
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(array) = value.as_array() {
            let embedding = array
                .iter()
                .filter_map(|v| v.as_f64().map(|n| n as f32))
                .collect::<Vec<_>>();
            if !embedding.is_empty() {
                return Ok(embedding);
            }
        }
        if let Some(array) = value.get("embedding").and_then(|v| v.as_array()) {
            let embedding = array
                .iter()
                .filter_map(|v| v.as_f64().map(|n| n as f32))
                .collect::<Vec<_>>();
            if !embedding.is_empty() {
                return Ok(embedding);
            }
        }
    }
    let embedding = decode_dense_embedding(trimmed);
    if embedding.is_empty() {
        anyhow::bail!("embedding command output did not contain numeric vectors");
    }
    Ok(embedding)
}

fn local_embedding_model_name() -> String {
    std::env::var("MANGOCODE_EMBEDDINGS_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| EMBEDDING_RUNTIME_CONFIG.read().model.clone())
}

fn embeddings_provider() -> String {
    std::env::var("MANGOCODE_EMBEDDINGS_PROVIDER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| EMBEDDING_RUNTIME_CONFIG.read().provider.clone())
}

#[cfg(not(test))]
fn default_embeddings_provider() -> &'static str {
    "fastembed"
}

#[cfg(test)]
fn default_embeddings_provider() -> &'static str {
    "command"
}

fn extract_explicit_memory_candidates(text: &str) -> Vec<(MemoryClass, String)> {
    let mut out = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        let lower = line.to_ascii_lowercase();
        let candidate = if let Some(rest) = lower.strip_prefix("remember:") {
            Some((MemoryClass::ProjectFact, &line[line.len() - rest.len()..]))
        } else if let Some(rest) = lower.strip_prefix("remember ") {
            Some((MemoryClass::ProjectFact, &line[line.len() - rest.len()..]))
        } else if let Some(rest) = lower.strip_prefix("decision:") {
            Some((MemoryClass::Decision, &line[line.len() - rest.len()..]))
        } else if let Some(rest) = lower.strip_prefix("preference:") {
            Some((
                MemoryClass::UserPreference,
                &line[line.len() - rest.len()..],
            ))
        } else if lower.starts_with("i prefer ") || lower.starts_with("we prefer ") {
            Some((MemoryClass::UserPreference, line))
        } else if line.contains("://") {
            Some((MemoryClass::ExternalDoc, line))
        } else {
            None
        };

        if let Some((class, content)) = candidate {
            let cleaned = content.trim().trim_start_matches(':').trim();
            if !cleaned.is_empty() {
                out.push((class, cleaned.to_string()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_private_sections() {
        assert_eq!(
            strip_private_sections("keep <private>drop</private> keep2"),
            "keep  keep2"
        );
    }

    #[test]
    fn detects_common_secret_shapes() {
        assert!(looks_like_secret("access_token=abc"));
        assert!(!looks_like_secret("prefer concise summaries"));
    }

    #[test]
    fn stores_and_searches_memory() {
        let tmp = tempfile::tempdir().expect("tmp");
        let store = LayeredMemoryStore::open(&tmp.path().join("memory.db")).expect("open");
        store
            .insert(
                MemoryClass::ProjectFact,
                "MangoCode uses Rust workspaces",
                Some("test"),
                Some("mangocode"),
            )
            .expect("insert");
        let hits = store.search("rust workspace", 5).expect("search");
        assert_eq!(hits.len(), 1);
        assert!(store
            .manifest(10)
            .expect("manifest")
            .contains("project_fact"));
    }

    #[test]
    fn captures_only_explicit_memory_shapes() {
        let tmp = tempfile::tempdir().expect("tmp");
        let store = LayeredMemoryStore::open(&tmp.path().join("memory.db")).expect("open");
        let ids = capture_explicit_memories(
            &store,
            "hello\nremember: MangoCode prefers native features\nI prefer concise output",
            Some("test"),
            Some("mangocode"),
        );
        assert_eq!(ids.len(), 2);
        assert_eq!(store.review(10).expect("review").len(), 2);
    }

    #[test]
    fn dedupes_identical_memories() {
        let tmp = tempfile::tempdir().expect("tmp");
        let store = LayeredMemoryStore::open(&tmp.path().join("memory.db")).expect("open");
        let first = store
            .insert(
                MemoryClass::Decision,
                "Use native document extraction",
                Some("test-a"),
                Some("mangocode"),
            )
            .expect("insert first");
        let second = store
            .insert(
                MemoryClass::Decision,
                "Use native document extraction",
                Some("test-b"),
                Some("mangocode"),
            )
            .expect("insert second");
        assert_eq!(first, second);
        assert_eq!(store.review(10).expect("review").len(), 1);
    }

    #[test]
    fn semantic_vector_search_finds_related_text() {
        let tmp = tempfile::tempdir().expect("tmp");
        let store = LayeredMemoryStore::open(&tmp.path().join("memory.db")).expect("open");
        store
            .insert(
                MemoryClass::ProjectFact,
                "The attachment pipeline converts documents into markdown before prompting",
                Some("test"),
                Some("mangocode"),
            )
            .expect("insert");
        let hits = store.search("doc markdown ingestion", 5).expect("search");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn parses_dense_embedding_outputs() {
        assert_eq!(
            parse_embedding_output("[0.1, -0.2, 0.3]").expect("json array"),
            vec![0.1, -0.2, 0.3]
        );
        assert_eq!(
            parse_embedding_output("{\"embedding\":[1,2,3]}").expect("json object"),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            parse_embedding_output("0.5, 0.25 -0.75").expect("plain floats"),
            vec![0.5, 0.25, -0.75]
        );
    }
}
