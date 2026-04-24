// Permission Critic: LLM-powered tool safety evaluator.
//
// An alternative to the static bash/ps classifiers.  Sends a lightweight
// API call to a cheap model (e.g. Haiku) asking whether a tool invocation
// is safe given the user's intent.  Results are cached for 60 seconds to
// avoid redundant calls.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::truncate::truncate_bytes_prefix;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Runtime configuration for the permission critic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticConfig {
    /// Whether the critic is active.
    pub enabled: bool,
    /// Model to use for evaluation (should be cheap/fast).
    pub model: String,
    /// Fall back to static classifiers when the API is unreachable.
    pub fallback_to_classifier: bool,
}

impl Default for CriticConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "claude-3-haiku-20240307".to_string(),
            fallback_to_classifier: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation result
// ---------------------------------------------------------------------------

/// The outcome of a single critic evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticEvaluation {
    pub tool_name: String,
    pub tool_input_summary: String,
    pub allowed: bool,
    pub reasoning: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Whether this result came from the cache.
    pub cached: bool,
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    allowed: bool,
    reasoning: String,
    created: Instant,
}

const CACHE_TTL: Duration = Duration::from_secs(60);
const MAX_HISTORY: usize = 50;

// ---------------------------------------------------------------------------
// PermissionCritic
// ---------------------------------------------------------------------------

/// LLM-powered permission evaluator.
///
/// Thread-safe: inner state is behind a `parking_lot::Mutex`.
pub struct PermissionCritic {
    config: Mutex<CriticConfig>,
    cache: Mutex<Vec<CacheEntry>>,
    /// Rolling history of recent evaluations (newest first).
    history: Mutex<VecDeque<CriticEvaluation>>,
}

impl PermissionCritic {
    pub fn new(config: CriticConfig) -> Self {
        Self {
            config: Mutex::new(config),
            cache: Mutex::new(Vec::new()),
            history: Mutex::new(VecDeque::new()),
        }
    }

    /// Whether the critic is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.lock().enabled
    }

    /// Toggle enabled state; returns new state.
    pub fn toggle(&self) -> bool {
        let mut cfg = self.config.lock();
        cfg.enabled = !cfg.enabled;
        cfg.enabled
    }

    /// Set enabled state explicitly.
    pub fn set_enabled(&self, enabled: bool) {
        self.config.lock().enabled = enabled;
    }

    /// Get a snapshot of the current config.
    pub fn get_config(&self) -> CriticConfig {
        self.config.lock().clone()
    }

    /// Update the config.
    pub fn update_config(&self, config: CriticConfig) {
        *self.config.lock() = config;
    }

    /// Return the most recent `n` evaluations (newest first).
    pub fn recent_evaluations(&self, n: usize) -> Vec<CriticEvaluation> {
        let history = self.history.lock();
        history.iter().take(n).cloned().collect()
    }

    /// Evaluate whether a tool invocation is safe.
    ///
    /// Returns `(allowed, reasoning)`.  Uses cache when possible; falls back
    /// to `Ok((true, "critic disabled"))` when the critic is off.
    pub async fn evaluate(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
        working_dir: &Path,
        user_intent: &str,
        api_key: &str,
        api_base: &str,
    ) -> anyhow::Result<(bool, String)> {
        let cfg = self.get_config();
        if !cfg.enabled {
            return Ok((true, "critic disabled".to_string()));
        }

        // Build cache key from tool name + serialized input.
        let input_str = serde_json::to_string(tool_input).unwrap_or_default();
        let cache_key = format!("{}:{}", tool_name, input_str);

        // Check cache.
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.iter().find(|e| e.key == cache_key) {
                if entry.created.elapsed() < CACHE_TTL {
                    let eval = CriticEvaluation {
                        tool_name: tool_name.to_string(),
                        tool_input_summary: summarize_input(tool_input),
                        allowed: entry.allowed,
                        reasoning: entry.reasoning.clone(),
                        timestamp: chrono::Utc::now(),
                        cached: true,
                    };
                    self.push_history(eval);
                    return Ok((entry.allowed, entry.reasoning.clone()));
                }
            }
        }

        // Build the evaluation prompt.
        let system = "You are a security evaluator for a coding assistant. \
            Assess whether the following tool call is safe and aligned with \
            the user's intent. Respond with ALLOW or DENY as the very first \
            word, followed by a one-sentence reason. Nothing else.";

        let user_msg = format!(
            "Tool: {tool_name}\n\
             Input: {input_str}\n\
             Working directory: {dir}\n\
             User's intent: {user_intent}\n\n\
             Is this tool call safe and appropriate?",
            dir = working_dir.display(),
        );

        // Make the API call.
        match self
            .call_api(&cfg.model, system, &user_msg, api_key, api_base)
            .await
        {
            Ok(response_text) => {
                let (allowed, reasoning) = parse_critic_response(&response_text);

                // Update cache.
                {
                    let mut cache = self.cache.lock();
                    // Evict expired entries.
                    cache.retain(|e| e.created.elapsed() < CACHE_TTL);
                    // Remove old entry for same key if present.
                    cache.retain(|e| e.key != cache_key);
                    cache.push(CacheEntry {
                        key: cache_key,
                        allowed,
                        reasoning: reasoning.clone(),
                        created: Instant::now(),
                    });
                }

                let eval = CriticEvaluation {
                    tool_name: tool_name.to_string(),
                    tool_input_summary: summarize_input(tool_input),
                    allowed,
                    reasoning: reasoning.clone(),
                    timestamp: chrono::Utc::now(),
                    cached: false,
                };
                self.push_history(eval);

                Ok((allowed, reasoning))
            }
            Err(e) => {
                tracing::warn!(error = %e, "Critic API call failed");
                if cfg.fallback_to_classifier {
                    Ok((true, format!("critic API error, falling back: {}", e)))
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Low-level API call to the evaluation model.
    async fn call_api(
        &self,
        model: &str,
        system: &str,
        user_msg: &str,
        api_key: &str,
        api_base: &str,
    ) -> anyhow::Result<String> {
        let url = format!("{}/v1/messages", api_base.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": model,
            "max_tokens": 100,
            "system": system,
            "messages": [
                { "role": "user", "content": user_msg }
            ]
        });

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let resp = client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Critic API returned {}: {}", status, text);
        }

        let json: serde_json::Value = resp.json().await?;

        // Extract text from the response content array.
        let text = json["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .unwrap_or("ALLOW no response text")
            .to_string();

        Ok(text)
    }

    fn push_history(&self, eval: CriticEvaluation) {
        let mut history = self.history.lock();
        history.push_front(eval);
        while history.len() > MAX_HISTORY {
            history.pop_back();
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse the critic model's response into (allowed, reasoning).
fn parse_critic_response(response: &str) -> (bool, String) {
    let trimmed = response.trim();
    let first_word = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_uppercase();

    let allowed = first_word == "ALLOW";
    let reasoning = trimmed
        .strip_prefix(&first_word)
        .unwrap_or(trimmed)
        .trim()
        .to_string();

    (allowed, reasoning)
}

/// Produce a short summary of tool input for the history log.
fn summarize_input(input: &serde_json::Value) -> String {
    // For bash/powershell, show the command.
    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        let truncated = if cmd.len() > 80 {
            format!("{}...", truncate_bytes_prefix(cmd, 77))
        } else {
            cmd.to_string()
        };
        return truncated;
    }
    // For file ops, show the path.
    if let Some(path) = input.get("file_path").and_then(|v| v.as_str()) {
        return path.to_string();
    }
    // For web fetch, show the URL.
    if let Some(url) = input.get("url").and_then(|v| v.as_str()) {
        return url.to_string();
    }
    // Fallback: first 80 chars of JSON.
    let s = serde_json::to_string(input).unwrap_or_default();
    if s.len() > 80 {
        format!("{}...", truncate_bytes_prefix(&s, 77))
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// Global critic instance, lazily initialized.
static GLOBAL_CRITIC: once_cell::sync::Lazy<Arc<PermissionCritic>> =
    once_cell::sync::Lazy::new(|| Arc::new(PermissionCritic::new(CriticConfig::default())));

/// Get the global critic instance.
pub fn global_critic() -> Arc<PermissionCritic> {
    GLOBAL_CRITIC.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_critic_response_allow() {
        let (allowed, reason) = parse_critic_response("ALLOW This is a safe read operation.");
        assert!(allowed);
        assert_eq!(reason, "This is a safe read operation.");
    }

    #[test]
    fn test_parse_critic_response_deny() {
        let (allowed, reason) =
            parse_critic_response("DENY This command deletes system files.");
        assert!(!allowed);
        assert_eq!(reason, "This command deletes system files.");
    }

    #[test]
    fn test_parse_critic_response_lowercase() {
        let (allowed, _) = parse_critic_response("allow seems fine");
        assert!(allowed);
    }

    #[test]
    fn test_parse_critic_response_garbage() {
        let (allowed, _) = parse_critic_response("maybe idk");
        assert!(!allowed); // anything that isn't ALLOW → deny
    }

    #[test]
    fn test_summarize_input_command() {
        let input = serde_json::json!({"command": "ls -la"});
        assert_eq!(summarize_input(&input), "ls -la");
    }

    #[test]
    fn test_summarize_input_file_path() {
        let input = serde_json::json!({"file_path": "/home/user/test.rs"});
        assert_eq!(summarize_input(&input), "/home/user/test.rs");
    }

    #[test]
    fn test_toggle() {
        let critic = PermissionCritic::new(CriticConfig::default());
        assert!(!critic.is_enabled());
        assert!(critic.toggle()); // now enabled
        assert!(!critic.toggle()); // now disabled
    }

    #[test]
    fn test_history_cap() {
        let critic = PermissionCritic::new(CriticConfig::default());
        for i in 0..60 {
            critic.push_history(CriticEvaluation {
                tool_name: format!("tool_{}", i),
                tool_input_summary: String::new(),
                allowed: true,
                reasoning: String::new(),
                timestamp: chrono::Utc::now(),
                cached: false,
            });
        }
        assert_eq!(critic.recent_evaluations(100).len(), MAX_HISTORY);
    }
}
