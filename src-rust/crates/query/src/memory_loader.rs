use crate::session_memory::migrate_legacy_agents_if_needed;
use anyhow::Result;
use std::path::PathBuf;
use tokio::fs;

pub struct MemoryLoader {
    pub memory_dir: PathBuf,
}

impl MemoryLoader {
    pub fn new(memory_dir: PathBuf) -> Self {
        Self { memory_dir }
    }

    /// Always called at session start. Returns the MEMORY.md index content.
    pub async fn load_index(&self) -> Result<String> {
        fs::create_dir_all(&self.memory_dir).await?;
        migrate_legacy_agents_if_needed(&self.memory_dir).await?;

        let index_path = self.memory_dir.join("MEMORY.md");
        let index = match fs::read_to_string(index_path).await {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e.into()),
        };

        Ok(index)
    }

    /// Given the current user query, decide which topic files are relevant
    /// and load them. Uses keyword matching against the index entries.
    pub async fn load_relevant_topics(&self, query: &str, index: &str) -> Result<Vec<(String, String)>> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Ok(vec![]);
        }

        let mut out = Vec::new();
        for line in index.lines() {
            let Some((link_path, searchable)) = parse_index_line(line) else {
                continue;
            };

            let search = searchable.to_lowercase();
            if !query_tokens.iter().any(|t| search.contains(t)) {
                continue;
            }

            let rel = link_path.strip_prefix("memory/").unwrap_or(&link_path);
            let topic_path = self.memory_dir.join(rel);
            let content = match fs::read_to_string(&topic_path).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            out.push((link_path, content));
        }

        Ok(out)
    }

    /// Search transcripts by grep (never load whole files).
    pub async fn grep_transcripts(&self, pattern: &str) -> Result<Vec<String>> {
        let conversations_dir = self
            .memory_dir
            .parent()
            .map(|p| p.join("conversations"))
            .unwrap_or_else(|| PathBuf::from("conversations"));

        let mut matches = Vec::new();
        let needle = pattern.to_lowercase();

        let mut dir = match fs::read_dir(&conversations_dir).await {
            Ok(d) => d,
            Err(_) => return Ok(matches),
        };

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }

            let content = match fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };

            for (idx, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    matches.push(format!("{}:{}:{}", path.display(), idx + 1, line));
                    if matches.len() >= 200 {
                        return Ok(matches);
                    }
                }
            }
        }

        Ok(matches)
    }
}

fn parse_index_line(line: &str) -> Option<(String, String)> {
    let open = line.find("(")?;
    let close = line[open + 1..].find(")")? + open + 1;
    let link = line[open + 1..close].trim().to_string();
    let title = line
        .split(']')
        .next()
        .and_then(|s| s.strip_prefix("- ["))
        .unwrap_or("")
        .trim();
    let hook = line
        .split('—')
        .nth(1)
        .map(str::trim)
        .unwrap_or("");

    Some((link, format!("{} {} {}", title, hook, line)))
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|t| t.len() >= 3)
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_relevant_topics_matches_index_keywords() {
        let dir = tempfile::tempdir().expect("tmp");
        let mem = dir.path().join("memory");
        fs::create_dir_all(&mem).await.expect("mkdir");
        fs::write(
            mem.join("MEMORY.md"),
            "- [Rust Patterns](memory/rust-patterns.md) — async traits and tokio\n",
        )
        .await
        .expect("write index");
        fs::write(mem.join("rust-patterns.md"), "use tokio::spawn(...)")
            .await
            .expect("write topic");

        let loader = MemoryLoader::new(mem);
        let index = loader.load_index().await.expect("index");
        let topics = loader
            .load_relevant_topics("Need tokio pattern", &index)
            .await
            .expect("topics");

        assert_eq!(topics.len(), 1);
        assert!(topics[0].0.contains("rust-patterns.md"));
    }
}
