//! Persistent cost ledger across sessions.
//! Stores cumulative usage at ~/.mangocode/usage.json.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const USAGE_FILE: &str = "usage.json";
const LEDGER_VERSION: u32 = 1;

/// A single session's cost record in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCostRecord {
    pub session_id: String,
    pub timestamp: String,
    pub model: String,
    pub cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub duration_ms: u64,
    pub working_dir: String,
}

/// The persistent usage ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLedger {
    pub version: u32,
    pub total_cost_usd: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_creation_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_sessions: u64,
    pub sessions: Vec<SessionCostRecord>,
}

impl Default for UsageLedger {
    fn default() -> Self {
        Self {
            version: LEDGER_VERSION,
            total_cost_usd: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_creation_tokens: 0,
            total_cache_read_tokens: 0,
            total_sessions: 0,
            sessions: vec![],
        }
    }
}

fn ledger_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mangocode").join(USAGE_FILE))
}

fn analytics_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mangocode").join("analytics").join("events.jsonl"))
}

impl UsageLedger {
    /// Load from disk, or return a fresh default if the file doesn't exist.
    pub fn load() -> Self {
        let mut ledger = ledger_path()
            .and_then(|path| std::fs::read_to_string(&path).ok())
            .and_then(|s| serde_json::from_str::<UsageLedger>(&s).ok())
            .unwrap_or_default();

        // One-time migration path for users who only have analytics JSONL.
        if ledger.sessions.is_empty() {
            ledger.backfill_from_analytics();
        }

        ledger
    }

    /// Append a session record and recompute totals. Writes atomically.
    pub fn record_session(&mut self, record: SessionCostRecord) {
        // Prevent duplicate writes from multiple exit paths.
        if self
            .sessions
            .iter()
            .any(|s| s.session_id == record.session_id)
        {
            return;
        }

        self.total_cost_usd += record.cost_usd;
        self.total_input_tokens += record.input_tokens;
        self.total_output_tokens += record.output_tokens;
        self.total_cache_creation_tokens += record.cache_creation_tokens;
        self.total_cache_read_tokens += record.cache_read_tokens;
        self.total_sessions += 1;
        self.sessions.push(record);
        let _ = self.save();
    }

    /// Persist to disk (atomic write via temp file + rename).
    fn save(&self) -> anyhow::Result<()> {
        let path = ledger_path().ok_or_else(|| anyhow::anyhow!("no home dir"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp = path.with_extension("tmp");
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, &content)?;

        // Windows rename cannot overwrite existing targets.
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        std::fs::rename(&tmp, &path)?;

        Ok(())
    }

    /// Return cost totals for the last N days.
    pub fn cost_since_days(&self, days: u64) -> f64 {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);

        self.sessions
            .iter()
            .filter_map(|s| {
                chrono::DateTime::parse_from_rfc3339(&s.timestamp)
                    .ok()
                    .map(|ts| (ts.with_timezone(&chrono::Utc), s.cost_usd))
            })
            .filter(|(ts, _)| *ts > cutoff)
            .map(|(_, cost)| cost)
            .sum()
    }

    /// Formatted history table for the /cost history command.
    pub fn format_history(&self, last_n: usize) -> String {
        if self.sessions.is_empty() {
            return "No session cost history recorded yet.".to_string();
        }

        let mut out = format!(
            "Cumulative Cost History - {} sessions, ${:.4} total\n\
             ======================================================\n\n\
             Last 24h: ${:.4}  |  Last 7d: ${:.4}  |  Last 30d: ${:.4}\n\n\
             Recent sessions:\n\
             {:<12} {:<22} {:<20} {:>12} {:>10}\n\
             {}\n",
            self.total_sessions,
            self.total_cost_usd,
            self.cost_since_days(1),
            self.cost_since_days(7),
            self.cost_since_days(30),
            "Session",
            "Timestamp",
            "Model",
            "Cost",
            "Tokens",
            "-".repeat(80),
        );

        let safe_n = if last_n == 0 { 20 } else { last_n };
        let start = self.sessions.len().saturating_sub(safe_n);

        for record in &self.sessions[start..] {
            let id_short = if record.session_id.len() > 8 {
                &record.session_id[..8]
            } else {
                &record.session_id
            };
            let ts_short = if record.timestamp.len() > 19 {
                &record.timestamp[..19]
            } else {
                &record.timestamp
            };
            let total_tok = record.input_tokens
                + record.output_tokens
                + record.cache_creation_tokens
                + record.cache_read_tokens;
            out.push_str(&format!(
                "{:<12} {:<22} {:<20} {:>11} {:>10}\n",
                id_short,
                ts_short,
                record.model,
                format!("${:.4}", record.cost_usd),
                format_tokens(total_tok),
            ));
        }

        out.push_str("\nUsage file: ~/.mangocode/usage.json");
        out
    }

    /// One-time migration: import historical session costs from events.jsonl.
    pub fn backfill_from_analytics(&mut self) {
        if !self.sessions.is_empty() {
            return;
        }

        let Some(path) = analytics_path() else {
            return;
        };
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };

        use std::collections::HashSet;
        let mut existing_ids: HashSet<String> =
            self.sessions.iter().map(|s| s.session_id.clone()).collect();
        let mut imported = 0_u64;

        for line in content.lines() {
            let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if val.get("event").and_then(|e| e.as_str()) != Some("session_ended") {
                continue;
            }

            let sid = val
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            if sid.is_empty() || existing_ids.contains(&sid) {
                continue;
            }

            let cost_usd = val.get("cost_usd").map(parse_f64_field).unwrap_or(0.0);

            let timestamp = val
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let model = val
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            self.total_cost_usd += cost_usd;
            self.total_sessions += 1;
            self.sessions.push(SessionCostRecord {
                session_id: sid.clone(),
                timestamp,
                model,
                cost_usd,
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_tokens: 0,
                cache_read_tokens: 0,
                duration_ms: 0,
                working_dir: String::new(),
            });
            existing_ids.insert(sid);
            imported += 1;
        }

        if imported > 0 {
            let _ = self.save();
        }
    }
}

fn parse_f64_field(value: &serde_json::Value) -> f64 {
    if let Some(v) = value.as_f64() {
        return v;
    }
    if let Some(v) = value.as_u64() {
        return v as f64;
    }
    value
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn format_tokens(t: u64) -> String {
    if t >= 1_000_000 {
        format!("{:.1}M", t as f64 / 1_000_000.0)
    } else if t >= 1_000 {
        format!("{:.1}K", t as f64 / 1_000.0)
    } else {
        format!("{}", t)
    }
}
