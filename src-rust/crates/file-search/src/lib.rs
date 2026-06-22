use glob::Pattern;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchKind {
    Any,
    File,
    Folder,
    Symbol,
    Recent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchEntry {
    pub path: PathBuf,
    pub relative_path: String,
    pub kind: SearchKind,
    pub symbol_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHit {
    pub entry: FileSearchEntry,
    pub score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeChunk {
    pub content: String,
    pub path: PathBuf,
    pub relative_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeSearchHit {
    pub chunk: CodeChunk,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct FileSearchIndex {
    root: PathBuf,
    entries: Vec<FileSearchEntry>,
    chunks: Vec<CodeChunk>,
    recent: HashMap<String, i64>,
    next_recent_rank: i64,
}

impl FileSearchIndex {
    pub fn build(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::build_limited(root, usize::MAX)
    }

    pub fn build_limited(root: impl Into<PathBuf>, max_entries: usize) -> io::Result<Self> {
        let root = root.into();
        let ignore = IgnoreMatcher::load(&root)?;
        let mut entries = Vec::new();
        if max_entries == 0 {
            return Ok(Self {
                root,
                entries,
                chunks: Vec::new(),
                recent: HashMap::new(),
                next_recent_rank: 1,
            });
        }

        for entry in WalkDir::new(&root)
            .into_iter()
            .filter_entry(|entry| should_descend(entry, &root, &ignore))
        {
            let entry = entry?;
            if entry.depth() == 0 {
                continue;
            }
            if should_skip(&entry, &root, &ignore) {
                continue;
            }

            let path = entry.path().to_path_buf();
            let relative_path = normalize_path(path.strip_prefix(&root).unwrap_or(&path));
            let kind = if entry.file_type().is_dir() {
                SearchKind::Folder
            } else {
                SearchKind::File
            };
            entries.push(FileSearchEntry {
                path,
                relative_path,
                kind,
                symbol_name: None,
            });

            if entries.len() >= max_entries {
                break;
            }
        }

        Ok(Self {
            root,
            entries,
            chunks: Vec::new(),
            recent: HashMap::new(),
            next_recent_rank: 1,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn entries(&self) -> &[FileSearchEntry] {
        &self.entries
    }

    pub fn chunks(&self) -> &[CodeChunk] {
        &self.chunks
    }

    pub fn mark_recent(&mut self, path: impl AsRef<Path>) {
        let key = self.relative_key(path.as_ref());
        let rank = self.next_recent_rank;
        self.next_recent_rank += 1;
        self.recent.insert(key, rank);
    }

    pub fn with_recent(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        for path in paths {
            self.mark_recent(path);
        }
        self
    }

    pub fn add_symbol(&mut self, name: impl Into<String>, path: impl AsRef<Path>) {
        let name = name.into();
        let path = if path.as_ref().is_absolute() {
            path.as_ref().to_path_buf()
        } else {
            self.root.join(path.as_ref())
        };
        let relative_path = normalize_path(path.strip_prefix(&self.root).unwrap_or(&path));
        self.entries.push(FileSearchEntry {
            path,
            relative_path,
            kind: SearchKind::Symbol,
            symbol_name: Some(name),
        });
    }

    pub fn add_lightweight_symbols(&mut self, max_files: usize, max_symbols: usize) {
        let candidates = self
            .entries
            .iter()
            .filter(|entry| entry.kind == SearchKind::File && is_symbol_candidate(&entry.path))
            .take(max_files)
            .cloned()
            .collect::<Vec<_>>();

        let mut added = 0usize;
        for entry in candidates {
            let Ok(contents) = fs::read_to_string(&entry.path) else {
                continue;
            };
            for symbol in extract_symbols(&contents) {
                self.add_symbol(symbol, &entry.path);
                added += 1;
                if added >= max_symbols {
                    return;
                }
            }
        }
    }

    pub fn add_code_chunks(&mut self, max_files: usize, max_bytes_per_file: u64) {
        self.add_code_chunks_with_options(max_files, max_bytes_per_file, false);
    }

    pub fn add_code_chunks_with_options(
        &mut self,
        max_files: usize,
        max_bytes_per_file: u64,
        include_text_files: bool,
    ) {
        let candidates = self
            .entries
            .iter()
            .filter(|entry| {
                entry.kind == SearchKind::File
                    && is_indexable_search_file(&entry.path, include_text_files)
            })
            .take(max_files)
            .cloned()
            .collect::<Vec<_>>();

        for entry in candidates {
            let Ok(metadata) = fs::metadata(&entry.path) else {
                continue;
            };
            if metadata.len() > max_bytes_per_file {
                continue;
            }
            let Ok(contents) = fs::read_to_string(&entry.path) else {
                continue;
            };
            let language = detect_language(&entry.path).map(str::to_string);
            self.chunks.extend(chunk_source(
                &contents,
                entry.path,
                entry.relative_path,
                language,
                1_500,
            ));
        }
    }

    pub fn search(&self, query: &str, kind: SearchKind, limit: usize) -> Vec<SearchHit> {
        let query = query.trim().to_ascii_lowercase();
        let mut hits = Vec::new();

        for entry in &self.entries {
            if !kind_matches(kind, entry) {
                continue;
            }
            let Some(score) = self.score_entry(&query, entry, kind) else {
                continue;
            };
            hits.push(SearchHit {
                entry: entry.clone(),
                score,
            });
        }

        hits.sort_by_key(|hit| (Reverse(hit.score), hit.entry.relative_path.clone()));
        hits.truncate(limit);
        hits
    }

    pub fn search_code(&self, query: &str, limit: usize) -> Vec<CodeSearchHit> {
        search_chunks(&self.chunks, query, limit)
    }

    pub fn search_code_filtered(
        &self,
        query: &str,
        limit: usize,
        languages: &[String],
        paths: &[String],
    ) -> Vec<CodeSearchHit> {
        if languages.is_empty() && paths.is_empty() {
            return self.search_code(query, limit);
        }

        let language_set = languages
            .iter()
            .map(|language| language.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let path_set = paths
            .iter()
            .map(|path| normalize_path(Path::new(path)))
            .collect::<HashSet<_>>();
        let chunks = self
            .chunks
            .iter()
            .filter(|chunk| {
                (language_set.is_empty()
                    || chunk
                        .language
                        .as_ref()
                        .map(|language| language_set.contains(&language.to_ascii_lowercase()))
                        .unwrap_or(false))
                    && (path_set.is_empty() || path_set.contains(&chunk.relative_path))
            })
            .cloned()
            .collect::<Vec<_>>();

        search_chunks(&chunks, query, limit)
    }

    pub fn find_related(
        &self,
        path: impl AsRef<Path>,
        line: usize,
        limit: usize,
    ) -> Vec<CodeSearchHit> {
        let key = self.relative_key(path.as_ref());
        let Some(target) = self
            .chunks
            .iter()
            .find(|chunk| {
                chunk.relative_path == key && line >= chunk.start_line && line <= chunk.end_line
            })
            .or_else(|| self.chunks.iter().find(|chunk| chunk.relative_path == key))
        else {
            return Vec::new();
        };

        let target_tokens = tokenize(&target.content)
            .into_iter()
            .collect::<HashSet<_>>();
        if target_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores = HashMap::new();
        for (idx, chunk) in self.chunks.iter().enumerate() {
            if chunk.relative_path == target.relative_path
                && chunk.start_line == target.start_line
                && chunk.end_line == target.end_line
            {
                continue;
            }
            let tokens = tokenize(&chunk.content).into_iter().collect::<HashSet<_>>();
            if tokens.is_empty() {
                continue;
            }
            let shared = target_tokens.intersection(&tokens).count() as f64;
            if shared == 0.0 {
                continue;
            }
            let union = target_tokens.union(&tokens).count() as f64;
            scores.insert(idx, shared / union);
        }

        rerank_chunk_scores(&self.chunks, scores, limit, false)
    }

    fn score_entry(&self, query: &str, entry: &FileSearchEntry, kind: SearchKind) -> Option<i64> {
        if kind == SearchKind::Recent && !self.recent.contains_key(&entry.relative_path) {
            return None;
        }

        let display = entry
            .symbol_name
            .as_ref()
            .map(|name| format!("{name} {}", entry.relative_path))
            .unwrap_or_else(|| entry.relative_path.clone());
        let display_lower = display.to_ascii_lowercase();
        let file_name = Path::new(&entry.relative_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&entry.relative_path)
            .to_ascii_lowercase();

        let mut score = if query.is_empty() {
            1
        } else if file_name == query {
            10_000
        } else if display_lower == query {
            9_000
        } else if file_name.contains(query) {
            7_000 - file_name.len() as i64
        } else if display_lower.contains(query) {
            5_000 - display_lower.len() as i64
        } else {
            fuzzy_score(query, &display_lower)?
        };

        if let Some(rank) = self.recent.get(&entry.relative_path) {
            score += 1_000 + rank;
        }
        if entry.kind == SearchKind::Symbol {
            score += 150;
        }
        Some(score)
    }

    fn relative_key(&self, path: &Path) -> String {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        normalize_path(absolute.strip_prefix(&self.root).unwrap_or(&absolute))
    }
}

fn kind_matches(kind: SearchKind, entry: &FileSearchEntry) -> bool {
    match kind {
        SearchKind::Any => true,
        SearchKind::File => entry.kind == SearchKind::File,
        SearchKind::Folder => entry.kind == SearchKind::Folder,
        SearchKind::Symbol => entry.kind == SearchKind::Symbol,
        SearchKind::Recent => entry.kind == SearchKind::File || entry.kind == SearchKind::Folder,
    }
}

fn should_descend(entry: &DirEntry, root: &Path, ignore: &IgnoreMatcher) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    if is_default_ignored_dir(entry) {
        return false;
    }
    if is_hidden(entry.path(), root) {
        return false;
    }
    !ignore.matches(entry.path(), root)
}

fn should_skip(entry: &DirEntry, root: &Path, ignore: &IgnoreMatcher) -> bool {
    is_default_ignored_dir(entry)
        || is_hidden(entry.path(), root)
        || ignore.matches(entry.path(), root)
}

fn is_default_ignored_dir(entry: &DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    matches!(
        entry.file_name().to_str(),
        Some(
            ".git"
                | ".hg"
                | ".svn"
                | "__pycache__"
                | "node_modules"
                | ".venv"
                | "venv"
                | ".tox"
                | ".mypy_cache"
                | ".pytest_cache"
                | ".ruff_cache"
                | ".cache"
                | ".semble"
                | ".next"
                | "dist"
                | "build"
                | ".eggs"
        )
    )
}

fn is_hidden(path: &Path, root: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .map(|name| name.starts_with('.'))
            .unwrap_or(false)
    })
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

fn is_symbol_candidate(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "py"
                | "go"
                | "java"
                | "kt"
                | "swift"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
                | "cs"
        )
    )
}

fn extract_symbols(contents: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    for line in contents.lines().take(500) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
            continue;
        }
        if let Some(symbol) = extract_symbol_from_line(trimmed) {
            symbols.push(symbol);
        }
    }
    symbols
}

fn extract_symbol_from_line(line: &str) -> Option<String> {
    let mut tokens = line
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '(' | ')' | '{' | ':' | '<' | '='))
        .filter(|token| !token.is_empty());

    let first = tokens.next()?;
    let second = tokens.next();
    let third = tokens.next();

    let candidate = match first {
        "pub" | "export" | "async" => match second {
            Some("fn" | "function" | "class" | "struct" | "enum" | "trait" | "interface") => third,
            Some("const" | "let" | "var" | "type" | "static") => third,
            Some("async") => match third {
                Some("fn" | "function") => tokens.next(),
                _ => None,
            },
            Some(other) if is_declaration_keyword(other) => third,
            _ => None,
        },
        "fn" | "function" | "class" | "struct" | "enum" | "trait" | "interface" | "def"
        | "func" | "mod" => second,
        "const" | "let" | "var" | "type" | "static" => second,
        _ => None,
    }?;

    clean_symbol(candidate)
}

fn is_declaration_keyword(token: &str) -> bool {
    matches!(
        token,
        "fn" | "function"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "interface"
            | "def"
            | "func"
            | "mod"
            | "const"
            | "let"
            | "var"
            | "type"
            | "static"
    )
}

fn clean_symbol(token: &str) -> Option<String> {
    let symbol = token
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .to_string();
    if symbol.is_empty() {
        None
    } else {
        Some(symbol)
    }
}

fn is_indexable_search_file(path: &Path, include_text_files: bool) -> bool {
    detect_language(path)
        .map(|language| include_text_files || !DOC_LANGUAGES.contains(&language))
        .unwrap_or(false)
}

fn detect_language(path: &Path) -> Option<&'static str> {
    if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
        match name.to_ascii_lowercase().as_str() {
            "dockerfile" => return Some("dockerfile"),
            "makefile" | "gnumakefile" => return Some("make"),
            "justfile" => return Some("just"),
            "cmakelists.txt" => return Some("cmake"),
            "gemfile" | "rakefile" => return Some("ruby"),
            "build.gradle" | "settings.gradle" => return Some("groovy"),
            "build.gradle.kts" | "settings.gradle.kts" => return Some("kotlin"),
            _ => {}
        }
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some("rust"),
        "py" | "pyi" | "pyw" => Some("python"),
        "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
        "ts" | "tsx" | "mts" | "cts" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "kt" | "kts" => Some("kotlin"),
        "swift" => Some("swift"),
        "c" | "h" => Some("c"),
        "cc" | "cpp" | "cxx" | "hpp" | "hxx" => Some("cpp"),
        "cs" => Some("csharp"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "scala" => Some("scala"),
        "dart" => Some("dart"),
        "lua" => Some("lua"),
        "ex" | "exs" => Some("elixir"),
        "erl" | "hrl" => Some("erlang"),
        "hs" => Some("haskell"),
        "ml" | "mli" => Some("ocaml"),
        "sh" | "bash" | "zsh" | "fish" | "ps1" => Some("shell"),
        "sql" => Some("sql"),
        "html" | "htm" => Some("html"),
        "css" | "scss" | "sass" | "less" => Some("css"),
        "json" | "json5" | "yaml" | "yml" | "toml" | "xml" | "md" | "markdown" | "rst" | "txt"
        | "csv" | "tsv" | "lock" => Some("doc"),
        _ => None,
    }
}

const DOC_LANGUAGES: &[&str] = &["doc", "html", "css"];

fn chunk_source(
    source: &str,
    path: PathBuf,
    relative_path: String,
    language: Option<String>,
    desired_chars: usize,
) -> Vec<CodeChunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut start_line = 1usize;
    let mut current_line = 1usize;

    for line in source.lines() {
        if current.is_empty() {
            start_line = current_line;
        }
        if !current.is_empty() && current.len() + line.len() + 1 > desired_chars {
            chunks.push(CodeChunk {
                content: current.trim_end().to_string(),
                path: path.clone(),
                relative_path: relative_path.clone(),
                start_line,
                end_line: current_line.saturating_sub(1),
                language: language.clone(),
            });
            current.clear();
            start_line = current_line;
        }
        current.push_str(line);
        current.push('\n');
        current_line += 1;
    }

    if !current.trim().is_empty() {
        chunks.push(CodeChunk {
            content: current.trim_end().to_string(),
            path,
            relative_path,
            start_line,
            end_line: current_line.saturating_sub(1),
            language,
        });
    }

    chunks
}

fn search_chunks(chunks: &[CodeChunk], query: &str, limit: usize) -> Vec<CodeSearchHit> {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() || chunks.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    let tokenized_chunks = chunks
        .iter()
        .map(|chunk| {
            let tokens = tokenize(&format!("{} {}", chunk.relative_path, chunk.content));
            for token in tokens.iter().collect::<HashSet<_>>() {
                *doc_freq.entry(token.clone()).or_default() += 1;
            }
            tokens
        })
        .collect::<Vec<_>>();

    let mut raw_scores = HashMap::new();
    for (idx, tokens) in tokenized_chunks.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        let mut term_counts: HashMap<&str, usize> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token.as_str()).or_default() += 1;
        }

        let mut score = 0.0;
        for query_token in &query_tokens {
            let Some(count) = term_counts.get(query_token.as_str()) else {
                continue;
            };
            let df = *doc_freq.get(query_token).unwrap_or(&1) as f64;
            let idf = ((chunks.len() as f64 + 1.0) / (df + 0.5)).ln().max(0.1);
            score += (*count as f64).sqrt() * idf;
        }
        if score > 0.0 {
            raw_scores.insert(idx, score);
        }
    }

    let boosted = apply_query_boosts(chunks, raw_scores, query);
    rerank_chunk_scores(chunks, boosted, limit, true)
}

fn tokenize(text: &str) -> Vec<String> {
    if let Some(re) = token_regex() {
        return re
            .find_iter(text)
            .flat_map(|m| split_identifier(m.as_str()))
            .collect();
    }

    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| {
            token
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
        })
        .flat_map(split_identifier)
        .collect()
}

fn token_regex() -> Option<&'static Regex> {
    static TOKEN_RE: OnceLock<Result<Regex, regex::Error>> = OnceLock::new();
    TOKEN_RE
        .get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*"))
        .as_ref()
        .ok()
}

fn split_identifier(token: &str) -> Vec<String> {
    let lower = token.to_ascii_lowercase();
    let mut parts = Vec::new();
    if token.contains('_') {
        parts.extend(
            lower
                .split('_')
                .filter(|part| !part.is_empty())
                .map(str::to_string),
        );
    } else {
        let mut start = 0usize;
        let chars = token.char_indices().collect::<Vec<_>>();
        for idx in 1..chars.len() {
            let prev = chars[idx - 1].1;
            let current = chars[idx].1;
            let next = chars.get(idx + 1).map(|(_, ch)| *ch);
            let boundary = (prev.is_ascii_lowercase() && current.is_ascii_uppercase())
                || (prev.is_ascii_uppercase()
                    && current.is_ascii_uppercase()
                    && next.map(|ch| ch.is_ascii_lowercase()).unwrap_or(false));
            if boundary {
                parts.push(token[start..chars[idx].0].to_ascii_lowercase());
                start = chars[idx].0;
            }
        }
        parts.push(token[start..].to_ascii_lowercase());
    }

    if parts.len() >= 2 {
        let mut expanded = vec![lower];
        expanded.extend(parts);
        expanded
    } else {
        vec![lower]
    }
}

fn apply_query_boosts(
    chunks: &[CodeChunk],
    scores: HashMap<usize, f64>,
    query: &str,
) -> HashMap<usize, f64> {
    if scores.is_empty() {
        return scores;
    }

    let max_score = scores.values().copied().fold(0.0, f64::max);
    let mut boosted = scores;
    boost_multi_chunk_files(chunks, &mut boosted, max_score);

    if is_symbol_query(query) {
        boost_symbol_definitions(chunks, &mut boosted, query, max_score);
    } else {
        boost_stem_matches(chunks, &mut boosted, query, max_score);
        boost_embedded_symbols(chunks, &mut boosted, query, max_score);
    }

    boosted
}

fn boost_multi_chunk_files(chunks: &[CodeChunk], scores: &mut HashMap<usize, f64>, max_score: f64) {
    let mut file_sum: HashMap<&str, f64> = HashMap::new();
    let mut best_chunk: HashMap<&str, usize> = HashMap::new();
    for (&idx, &score) in scores.iter() {
        let path = chunks[idx].relative_path.as_str();
        *file_sum.entry(path).or_default() += score;
        if best_chunk
            .get(path)
            .map(|best| score > scores[best])
            .unwrap_or(true)
        {
            best_chunk.insert(path, idx);
        }
    }
    let max_file_sum = file_sum.values().copied().fold(0.0, f64::max);
    if max_file_sum == 0.0 {
        return;
    }
    for (path, idx) in best_chunk {
        let boost = max_score * 0.2 * file_sum[path] / max_file_sum;
        *scores.entry(idx).or_default() += boost;
    }
}

fn is_symbol_query(query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() || query.contains(' ') {
        return false;
    }
    query.starts_with('_')
        || query.contains("::")
        || query.contains("->")
        || query.contains('.')
        || query.chars().any(|ch| ch == '_' || ch.is_ascii_uppercase())
}

fn symbol_tail(query: &str) -> &str {
    query
        .rsplit_once("::")
        .or_else(|| query.rsplit_once("->"))
        .or_else(|| query.rsplit_once('.'))
        .map(|(_, tail)| tail)
        .unwrap_or(query)
        .trim()
}

fn chunk_defines_symbol(chunk: &CodeChunk, symbol: &str) -> bool {
    let patterns = [
        "class",
        "module",
        "defmodule",
        "def",
        "interface",
        "struct",
        "enum",
        "trait",
        "type",
        "func",
        "function",
        "object",
        "fn",
        "fun",
        "package",
        "namespace",
        "protocol",
        "record",
        "typedef",
        "CREATE TABLE",
        "CREATE VIEW",
        "CREATE FUNCTION",
        "CREATE PROCEDURE",
    ];
    for line in chunk.content.lines() {
        let trimmed = line.trim_start();
        for keyword in patterns {
            if trimmed.starts_with(keyword)
                && trimmed
                    .split(|ch: char| {
                        ch.is_whitespace() || matches!(ch, '<' | '(' | '{' | ':' | '[' | ';')
                    })
                    .any(|part| part == symbol)
            {
                return true;
            }
        }
    }
    false
}

fn boost_symbol_definitions(
    chunks: &[CodeChunk],
    scores: &mut HashMap<usize, f64>,
    query: &str,
    max_score: f64,
) {
    let symbol = symbol_tail(query);
    let boost = max_score * 3.0;
    let existing = scores.keys().copied().collect::<Vec<_>>();
    for idx in existing {
        if chunk_defines_symbol(&chunks[idx], symbol) {
            *scores.entry(idx).or_default() += boost;
        }
    }
    for (idx, chunk) in chunks.iter().enumerate() {
        if scores.contains_key(&idx) {
            continue;
        }
        if stem_matches(
            &file_stem(&chunk.relative_path),
            &symbol.to_ascii_lowercase(),
        ) && chunk_defines_symbol(chunk, symbol)
        {
            scores.insert(idx, boost);
        }
    }
}

fn boost_embedded_symbols(
    chunks: &[CodeChunk],
    scores: &mut HashMap<usize, f64>,
    query: &str,
    max_score: f64,
) {
    let names = query
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|word| word.chars().any(|ch| ch.is_ascii_uppercase()) && word.len() > 2)
        .collect::<Vec<_>>();
    if names.is_empty() {
        return;
    }
    let boost = max_score * 1.5;
    let existing = scores.keys().copied().collect::<Vec<_>>();
    for idx in existing {
        if names
            .iter()
            .any(|name| chunk_defines_symbol(&chunks[idx], name))
        {
            *scores.entry(idx).or_default() += boost;
        }
    }
}

fn boost_stem_matches(
    chunks: &[CodeChunk],
    scores: &mut HashMap<usize, f64>,
    query: &str,
    max_score: f64,
) {
    let keywords = tokenize(query)
        .into_iter()
        .filter(|word| word.len() > 2 && !STOPWORDS.contains(&word.as_str()))
        .collect::<HashSet<_>>();
    if keywords.is_empty() {
        return;
    }
    let existing = scores.keys().copied().collect::<Vec<_>>();
    for idx in existing {
        let path = Path::new(&chunks[idx].relative_path);
        let mut parts = split_identifier(path.file_stem().and_then(|s| s.to_str()).unwrap_or(""));
        if let Some(parent) = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|p| p.to_str())
        {
            parts.extend(split_identifier(parent));
        }
        let part_set = parts.into_iter().collect::<HashSet<_>>();
        let matches = keywords
            .iter()
            .filter(|keyword| {
                part_set.iter().any(|part| {
                    keyword == &part
                        || (keyword.len().min(part.len()) >= 3
                            && (keyword.starts_with(part) || part.starts_with(keyword.as_str())))
                })
            })
            .count();
        if matches > 0 {
            *scores.entry(idx).or_default() += max_score * matches as f64 / keywords.len() as f64;
        }
    }
}

const STOPWORDS: &[&str] = &[
    "the", "and", "are", "for", "from", "has", "have", "how", "what", "when", "where", "which",
    "who", "why", "with", "that", "this",
];

fn rerank_chunk_scores(
    chunks: &[CodeChunk],
    scores: HashMap<usize, f64>,
    limit: usize,
    penalize_paths: bool,
) -> Vec<CodeSearchHit> {
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(a_idx, a_score), (b_idx, b_score)| {
        let a = if penalize_paths {
            *a_score * file_path_penalty(&chunks[*a_idx].relative_path)
        } else {
            *a_score
        };
        let b = if penalize_paths {
            *b_score * file_path_penalty(&chunks[*b_idx].relative_path)
        } else {
            *b_score
        };
        b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    let mut file_counts: HashMap<&str, usize> = HashMap::new();
    for (idx, score) in ranked {
        let file_count = file_counts
            .entry(chunks[idx].relative_path.as_str())
            .or_default();
        let mut effective_score = score;
        if *file_count >= 1 {
            effective_score *= 0.5_f64.powi(*file_count as i32);
        }
        *file_count += 1;
        selected.push(CodeSearchHit {
            chunk: chunks[idx].clone(),
            score: effective_score,
        });
        if selected.len() >= limit {
            break;
        }
    }
    selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk.relative_path.cmp(&b.chunk.relative_path))
    });
    selected
}

fn file_path_penalty(file_path: &str) -> f64 {
    let normal = file_path.replace('\\', "/");
    let name = Path::new(file_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let mut penalty = 1.0;
    if is_test_path(&normal) {
        penalty *= 0.3;
    }
    if matches!(name, "__init__.py" | "package-info.java") {
        penalty *= 0.5;
    }
    if normal.contains("/compat/") || normal.contains("/legacy/") || normal.contains("/examples/") {
        penalty *= 0.3;
    }
    if normal.ends_with(".d.ts") {
        penalty *= 0.7;
    }
    penalty
}

fn is_test_path(path: &str) -> bool {
    path.contains("/test/")
        || path.contains("/tests/")
        || path.contains("/__tests__/")
        || path.contains("/spec/")
        || path.ends_with("_test.go")
        || path.ends_with("_test.py")
        || path.ends_with(".test.ts")
        || path.ends_with(".test.tsx")
        || path.ends_with(".spec.ts")
        || path.ends_with(".spec.tsx")
        || path.ends_with("_spec.rb")
        || path.ends_with("Test.java")
        || path.ends_with("Tests.java")
        || path.ends_with("Test.kt")
        || path.ends_with("Tests.kt")
        || path.ends_with("Tests.cs")
}

fn stem_matches(stem: &str, name: &str) -> bool {
    let stem_norm = stem.replace('_', "");
    stem == name || stem_norm == name || stem.trim_end_matches('s') == name
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn fuzzy_score(query: &str, candidate: &str) -> Option<i64> {
    if query.is_empty() {
        return Some(1);
    }
    let mut candidate_chars = candidate.chars();
    let mut matched = 0i64;
    let mut gaps = 0i64;

    for query_ch in query.chars() {
        let mut gap = 0i64;
        loop {
            match candidate_chars.next() {
                Some(candidate_ch) if candidate_ch == query_ch => {
                    matched += 1;
                    gaps += gap;
                    break;
                }
                Some(_) => gap += 1,
                None => return None,
            }
        }
    }

    Some(1_000 + matched * 25 - gaps)
}

#[derive(Debug, Clone)]
struct IgnoreMatcher {
    specs: Vec<IgnoreSpec>,
}

#[derive(Debug, Clone)]
struct IgnoreSpec {
    base: PathBuf,
    patterns: Vec<IgnorePattern>,
}

#[derive(Debug, Clone)]
struct IgnorePattern {
    pattern: Pattern,
    negated: bool,
}

impl IgnoreMatcher {
    fn load(root: &Path) -> io::Result<Self> {
        let mut specs = Vec::new();
        collect_ignore_specs(root, root, &mut specs)?;
        Ok(Self { specs })
    }

    fn matches(&self, path: &Path, _root: &Path) -> bool {
        path_ignored_by_specs(&self.specs, path)
    }
}

fn collect_ignore_specs(dir: &Path, root: &Path, specs: &mut Vec<IgnoreSpec>) -> io::Result<()> {
    if let Some(patterns) = load_ignore_patterns(dir)? {
        specs.push(IgnoreSpec {
            base: dir.to_path_buf(),
            patterns,
        });
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }
        let path = entry.path();
        if is_hidden(&path, root) || path_ignored_by_specs(specs, &path) {
            continue;
        }
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(
                ".git"
                    | ".hg"
                    | ".svn"
                    | "__pycache__"
                    | "node_modules"
                    | ".venv"
                    | "venv"
                    | ".tox"
                    | ".mypy_cache"
                    | ".pytest_cache"
                    | ".ruff_cache"
                    | ".cache"
                    | ".semble"
                    | ".next"
                    | "dist"
                    | "build"
                    | ".eggs"
            )
        ) {
            continue;
        }
        collect_ignore_specs(&path, root, specs)?;
    }
    Ok(())
}

fn path_ignored_by_specs(specs: &[IgnoreSpec], path: &Path) -> bool {
    let mut ignored = false;
    for spec in specs {
        let Ok(relative_path) = path.strip_prefix(&spec.base) else {
            continue;
        };
        let relative = normalize_path(relative_path);
        for pattern in &spec.patterns {
            if pattern.pattern.matches(&relative) {
                ignored = !pattern.negated;
            }
        }
    }
    ignored
}

fn load_ignore_patterns(base: &Path) -> io::Result<Option<Vec<IgnorePattern>>> {
    let mut contents = String::new();
    for name in [".gitignore", ".sembleignore"] {
        match fs::read_to_string(base.join(name)) {
            Ok(file_contents) => {
                contents.push_str(&file_contents);
                contents.push('\n');
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }

    if contents.trim().is_empty() {
        return Ok(None);
    }

    let mut patterns = Vec::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let negated = trimmed.starts_with('!');
        let pattern = trimmed.strip_prefix('!').unwrap_or(trimmed);
        patterns.extend(
            pattern_variants(pattern)
                .into_iter()
                .filter_map(|pattern| Pattern::new(&pattern).ok())
                .map(|pattern| IgnorePattern { pattern, negated }),
        );
    }
    Ok(Some(patterns))
}

fn pattern_variants(pattern: &str) -> Vec<String> {
    let normalized = pattern.trim_start_matches('/').replace('\\', "/");
    if normalized.ends_with('/') {
        let dir = normalized.trim_end_matches('/');
        vec![
            dir.to_string(),
            format!("{dir}/**"),
            format!("**/{dir}"),
            format!("**/{dir}/**"),
        ]
    } else if normalized.contains('/') {
        vec![normalized]
    } else {
        vec![normalized.clone(), format!("**/{normalized}")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn skips_hidden_and_gitignored_entries() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(".gitignore"),
            "target/\n*.log\n!important.log\n",
        )
        .unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::create_dir(dir.path().join(".hidden")).unwrap();
        fs::write(dir.path().join("src").join("main.rs"), "").unwrap();
        fs::write(dir.path().join("target").join("skip.rs"), "").unwrap();
        fs::write(dir.path().join("debug.log"), "").unwrap();
        fs::write(dir.path().join("important.log"), "").unwrap();
        fs::write(dir.path().join(".hidden").join("secret.rs"), "").unwrap();

        let index = FileSearchIndex::build(dir.path()).unwrap();
        let paths = index
            .entries()
            .iter()
            .map(|entry| entry.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"src/main.rs"));
        assert!(!paths.contains(&".gitignore"));
        assert!(!paths.contains(&"target"));
        assert!(!paths.contains(&"target/skip.rs"));
        assert!(!paths.contains(&"debug.log"));
        assert!(paths.contains(&"important.log"));
        assert!(!paths.contains(&".hidden"));
        assert!(!paths.contains(&".hidden/secret.rs"));
    }

    #[test]
    fn build_limited_caps_index_size() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..5 {
            fs::write(dir.path().join(format!("file-{i}.txt")), "").unwrap();
        }

        let index = FileSearchIndex::build_limited(dir.path(), 2).unwrap();

        assert_eq!(index.entries().len(), 2);
    }

    #[test]
    fn searches_files_folders_symbols_and_recent() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src").join("tool_runtime.rs"), "").unwrap();
        fs::write(dir.path().join("README.md"), "").unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.add_symbol("ToolRegistryPlan", "src/tool_runtime.rs");
        index.mark_recent("README.md");

        assert_eq!(
            index.search("runtime", SearchKind::File, 5)[0]
                .entry
                .relative_path,
            "src/tool_runtime.rs"
        );
        assert_eq!(
            index.search("src", SearchKind::Folder, 5)[0]
                .entry
                .relative_path,
            "src"
        );
        assert_eq!(
            index.search("registry", SearchKind::Symbol, 5)[0]
                .entry
                .symbol_name
                .as_deref(),
            Some("ToolRegistryPlan")
        );
        assert_eq!(
            index.search("", SearchKind::Recent, 5)[0]
                .entry
                .relative_path,
            "README.md"
        );
    }

    #[test]
    fn extracts_lightweight_symbols_from_common_declarations() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src").join("lib.rs"),
            "pub struct ToolRegistryPlan {}\nfn dispatch_tool() {}\n",
        )
        .unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.add_lightweight_symbols(10, 10);

        let symbols = index
            .search("dispatch", SearchKind::Symbol, 5)
            .into_iter()
            .filter_map(|hit| hit.entry.symbol_name)
            .collect::<Vec<_>>();

        assert!(symbols.contains(&"dispatch_tool".to_string()));
    }

    #[test]
    fn recent_search_prefers_latest_marked_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("old.md"), "").unwrap();
        fs::write(dir.path().join("new.md"), "").unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.mark_recent("old.md");
        index.mark_recent("new.md");

        assert_eq!(
            index.search("", SearchKind::Recent, 2)[0]
                .entry
                .relative_path,
            "new.md"
        );
    }

    #[test]
    fn skips_default_dependency_dirs() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("node_modules")).unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("node_modules").join("dep.rs"), "").unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "").unwrap();

        let index = FileSearchIndex::build(dir.path()).unwrap();
        let paths = index
            .entries()
            .iter()
            .map(|entry| entry.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.contains(&"node_modules"));
        assert!(!paths.contains(&"node_modules/dep.rs"));
    }

    #[test]
    fn honors_sembleignore_patterns() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(".sembleignore"), "generated/\n*.snap\n").unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::create_dir(dir.path().join("generated")).unwrap();
        fs::write(dir.path().join("src").join("lib.rs"), "").unwrap();
        fs::write(dir.path().join("generated").join("skip.rs"), "").unwrap();
        fs::write(dir.path().join("render.snap"), "").unwrap();

        let index = FileSearchIndex::build(dir.path()).unwrap();
        let paths = index
            .entries()
            .iter()
            .map(|entry| entry.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.contains(&"generated"));
        assert!(!paths.contains(&"generated/skip.rs"));
        assert!(!paths.contains(&"render.snap"));
    }

    #[test]
    fn honors_nested_ignore_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("packages")).unwrap();
        fs::create_dir(dir.path().join("packages").join("app")).unwrap();
        fs::write(
            dir.path()
                .join("packages")
                .join("app")
                .join(".sembleignore"),
            "local-only/\n*.tmp\n",
        )
        .unwrap();
        fs::create_dir(dir.path().join("packages").join("app").join("src")).unwrap();
        fs::create_dir(dir.path().join("packages").join("app").join("local-only")).unwrap();
        fs::write(
            dir.path()
                .join("packages")
                .join("app")
                .join("src")
                .join("lib.rs"),
            "",
        )
        .unwrap();
        fs::write(
            dir.path()
                .join("packages")
                .join("app")
                .join("local-only")
                .join("skip.rs"),
            "",
        )
        .unwrap();
        fs::write(
            dir.path().join("packages").join("app").join("scratch.tmp"),
            "",
        )
        .unwrap();

        let index = FileSearchIndex::build(dir.path()).unwrap();
        let paths = index
            .entries()
            .iter()
            .map(|entry| entry.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"packages/app/src/lib.rs"));
        assert!(!paths.contains(&"packages/app/local-only"));
        assert!(!paths.contains(&"packages/app/local-only/skip.rs"));
        assert!(!paths.contains(&"packages/app/scratch.tmp"));
    }

    #[test]
    fn parent_ignored_dirs_do_not_apply_nested_ignore_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(".sembleignore"), "ignored/\n").unwrap();
        fs::create_dir(dir.path().join("ignored")).unwrap();
        fs::write(
            dir.path().join("ignored").join(".sembleignore"),
            "!keep.rs\n",
        )
        .unwrap();
        fs::write(dir.path().join("ignored").join("keep.rs"), "").unwrap();
        fs::write(dir.path().join("visible.rs"), "").unwrap();

        let index = FileSearchIndex::build(dir.path()).unwrap();
        let paths = index
            .entries()
            .iter()
            .map(|entry| entry.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"visible.rs"));
        assert!(!paths.contains(&"ignored"));
        assert!(!paths.contains(&"ignored/keep.rs"));
    }

    #[test]
    fn code_chunks_include_common_extensionless_code_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Dockerfile"), "FROM alpine\nRUN echo ok\n").unwrap();
        fs::write(dir.path().join("Makefile"), "build:\n\tcargo build\n").unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.add_code_chunks(10, 64 * 1024);
        let paths = index
            .chunks()
            .iter()
            .map(|chunk| chunk.relative_path.as_str())
            .collect::<Vec<_>>();

        assert!(paths.contains(&"Dockerfile"));
        assert!(paths.contains(&"Makefile"));
    }

    #[test]
    fn code_search_expands_identifiers_and_boosts_definitions() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::create_dir(dir.path().join("tests")).unwrap();
        fs::write(
            dir.path().join("src").join("session_manager.rs"),
            "pub struct SessionManager {}\nimpl SessionManager { fn load_user_session(&self) {} }\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("tests").join("session_manager_test.rs"),
            "fn test_load_user_session() {}\n",
        )
        .unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.add_code_chunks(10, 64 * 1024);

        let hits = index.search_code("SessionManager", 3);

        assert_eq!(hits[0].chunk.relative_path, "src/session_manager.rs");
        assert_eq!(hits[0].chunk.start_line, 1);
        assert!(hits[0].chunk.content.contains("SessionManager"));
    }

    #[test]
    fn code_search_finds_related_chunks() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src").join("auth.rs"),
            "fn validate_token() {}\nfn refresh_token() {}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src").join("session.rs"),
            "fn validate_session_token() {}\nfn save_session() {}\n",
        )
        .unwrap();

        let mut index = FileSearchIndex::build(dir.path()).unwrap();
        index.add_code_chunks(10, 64 * 1024);

        let hits = index.find_related("src/auth.rs", 1, 3);

        assert!(hits
            .iter()
            .any(|hit| hit.chunk.relative_path == "src/session.rs"));
    }

    #[test]
    fn code_search_supports_text_inclusion_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::create_dir(dir.path().join("docs")).unwrap();
        fs::write(
            dir.path().join("src").join("auth.rs"),
            "fn validate_token() {}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("docs").join("auth.md"),
            "Authentication runbook mentions validate_token behavior.\n",
        )
        .unwrap();

        let mut code_only = FileSearchIndex::build(dir.path()).unwrap();
        code_only.add_code_chunks(10, 64 * 1024);
        assert!(code_only
            .search_code("runbook validate token", 5)
            .iter()
            .all(|hit| hit.chunk.relative_path != "docs/auth.md"));

        let mut with_text = FileSearchIndex::build(dir.path()).unwrap();
        with_text.add_code_chunks_with_options(10, 64 * 1024, true);
        let doc_hits =
            with_text.search_code_filtered("runbook validate token", 5, &["doc".into()], &[]);
        assert!(doc_hits
            .iter()
            .any(|hit| hit.chunk.relative_path == "docs/auth.md"));

        let file_hits =
            with_text.search_code_filtered("validate token", 5, &[], &["src/auth.rs".into()]);
        assert!(file_hits
            .iter()
            .all(|hit| hit.chunk.relative_path == "src/auth.rs"));
    }
}
