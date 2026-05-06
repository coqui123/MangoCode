use glob::Pattern;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
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

#[derive(Debug, Clone)]
pub struct FileSearchIndex {
    root: PathBuf,
    entries: Vec<FileSearchEntry>,
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
    if is_hidden(entry.path(), root) {
        return false;
    }
    !ignore.matches(entry.path(), root)
}

fn should_skip(entry: &DirEntry, root: &Path, ignore: &IgnoreMatcher) -> bool {
    is_hidden(entry.path(), root) || ignore.matches(entry.path(), root)
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
    patterns: Vec<IgnorePattern>,
}

#[derive(Debug, Clone)]
struct IgnorePattern {
    pattern: Pattern,
    negated: bool,
}

impl IgnoreMatcher {
    fn load(root: &Path) -> io::Result<Self> {
        let path = root.join(".gitignore");
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) if err.kind() == io::ErrorKind::NotFound => String::new(),
            Err(err) => return Err(err),
        };

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
        Ok(Self { patterns })
    }

    fn matches(&self, path: &Path, root: &Path) -> bool {
        let relative = normalize_path(path.strip_prefix(root).unwrap_or(path));
        let mut ignored = false;
        for pattern in &self.patterns {
            if pattern.pattern.matches(&relative) {
                ignored = !pattern.negated;
            }
        }
        ignored
    }
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
}
