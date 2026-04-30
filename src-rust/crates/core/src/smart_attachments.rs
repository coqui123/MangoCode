//! Smart attachment routing and native document-to-Markdown extraction.
//!
//! This module is intentionally provider-agnostic: callers classify an incoming
//! file, pass the selected model's modality capabilities, and receive the
//! representation MangoCode should send to the model.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_DOCUMENT_INPUT_BYTES: u64 = 50 * 1024 * 1024;
const EXTRACTOR_VERSION: &str = "native-markdown-v4";
const MAX_EXTRACTED_CHARS: usize = 250_000;
const TESSERACT_VAULT_KEYS: &[&str] = &["tesseract", "ocr-tesseract"];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentKind {
    Image,
    Pdf,
    OfficeDocument,
    Html,
    Text,
    Archive,
    Data,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum AttachmentRoutingDecision {
    RawImage,
    MarkdownDocument,
    RawPdf,
    TextInline,
    Rejected { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub id: String,
    pub source: String,
    pub media_type: Option<String>,
    pub kind: AttachmentKind,
    pub display_name: String,
    pub size_bytes: u64,
    pub hash: String,
    pub routing_decision: AttachmentRoutingDecision,
    pub raw_path: Option<PathBuf>,
    pub extracted_markdown_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelAttachmentCapabilities {
    pub image_input: bool,
    pub pdf_input: bool,
}

pub fn model_attachment_capabilities(
    provider: Option<&str>,
    model: &str,
) -> ModelAttachmentCapabilities {
    let provider = provider.unwrap_or("anthropic").to_ascii_lowercase();
    let model = model.to_ascii_lowercase();

    let image_input = match provider.as_str() {
        "anthropic" | "anthropic-max" | "amazon-bedrock" | "google" | "google-vertex" => true,
        "openai" | "azure" | "github-copilot" | "openai-codex" | "codex" => {
            openai_family_model_supports_images(&model)
        }
        "cohere" | "deepseek" | "cerebras" | "perplexity" => false,
        "mistral" => model_contains_any(&model, &["pixtral", "vision", "mistral-small-3.1"]),
        "xai" => model_contains_any(&model, &["vision", "grok-4", "grok-2-vision"]),
        "qwen" => model_contains_any(&model, &["vl", "omni", "vision", "qwen-vl"]),
        "openrouter" | "ollama" | "lm-studio" | "llama-cpp" | "deepinfra" | "groq"
        | "together-ai" | "togetherai" | "venice" | "sambanova" | "huggingface" | "nvidia"
        | "siliconflow" | "moonshotai" | "zhipuai" | "nebius" | "novita" | "ovhcloud"
        | "scaleway" | "vultr" | "baseten" | "friendli" | "upstage" | "stepfun" | "fireworks"
        | "minimax" => generic_model_supports_images(&model),
        _ => generic_model_supports_images(&model),
    };

    let pdf_input = match provider.as_str() {
        "anthropic" | "anthropic-max" | "amazon-bedrock" | "google" | "google-vertex" => true,
        // For coding workflows MangoCode still prefers Markdown extraction for
        // PDFs; this flag only prevents accidental raw PDF routing when native
        // Markdown extraction is disabled.
        _ => false,
    };

    ModelAttachmentCapabilities {
        image_input,
        pdf_input,
    }
}

fn openai_family_model_supports_images(model: &str) -> bool {
    model_contains_any(
        model,
        &[
            "gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4", "vision", "realtime",
        ],
    ) && !model_contains_any(model, &["embedding", "audio", "transcribe", "tts"])
}

fn generic_model_supports_images(model: &str) -> bool {
    model_contains_any(
        model,
        &[
            "vision",
            "vl",
            "omni",
            "multimodal",
            "gpt-4o",
            "gpt-4.1",
            "gpt-5",
            "o3",
            "o4",
            "gemini",
            "claude",
            "pixtral",
            "llava",
            "qwen-vl",
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen3-vl",
            "grok-vision",
        ],
    ) && !model_contains_any(
        model,
        &["embedding", "rerank", "moderation", "tts", "audio"],
    )
}

fn model_contains_any(model: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| model.contains(needle))
}

#[derive(Debug, Clone, Copy)]
pub struct AttachmentRoutingConfig {
    pub markdown_extraction_enabled: bool,
    pub images_raw_by_default: bool,
}

impl Default for AttachmentRoutingConfig {
    fn default() -> Self {
        Self {
            markdown_extraction_enabled: true,
            images_raw_by_default: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarkdownExtraction {
    pub markdown: String,
    pub cache_path: PathBuf,
    pub from_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentCacheMetadata {
    pub source_path: String,
    pub source_mtime_unix: Option<u64>,
    pub source_size_bytes: u64,
    pub source_hash: String,
    pub media_type: Option<String>,
    pub extractor_version: String,
    pub extracted_at: String,
}

pub fn classify_path(path: &Path) -> AttachmentKind {
    match extension(path).as_deref() {
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "svg") => {
            AttachmentKind::Image
        }
        Some("pdf") => AttachmentKind::Pdf,
        Some("doc" | "docx" | "ppt" | "pptx" | "xls" | "xlsx") => AttachmentKind::OfficeDocument,
        Some("html" | "htm") => AttachmentKind::Html,
        Some("json" | "xml" | "csv" | "jsonl" | "ndjson") => AttachmentKind::Data,
        Some(
            "txt" | "md" | "markdown" | "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "toml" | "yaml"
            | "yml",
        ) => AttachmentKind::Text,
        Some("zip") => AttachmentKind::Archive,
        _ => AttachmentKind::Unknown,
    }
}

pub fn media_type_for_path(path: &Path) -> Option<String> {
    let media_type = match extension(path).as_deref()? {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ppt" => "application/vnd.ms-powerpoint",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xls" => "application/vnd.ms-excel",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "json" | "jsonl" | "ndjson" => "application/json",
        "xml" => "application/xml",
        "zip" => "application/zip",
        "txt" | "md" | "markdown" | "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "toml" | "yaml"
        | "yml" => "text/plain",
        _ => return None,
    };
    Some(media_type.to_string())
}

pub fn image_markdown_fallback(path: &Path, label: Option<&str>) -> anyhow::Result<String> {
    let meta = std::fs::metadata(path)?;
    let hash = hash_file(path)?;
    let media_type = media_type_for_path(path).unwrap_or_else(|| "image/unknown".to_string());
    let dimensions = image_dimensions(path).map(|(w, h)| format!("{}x{}", w, h));
    let display_name = label
        .or_else(|| path.file_name().and_then(|s| s.to_str()))
        .unwrap_or("image");

    Ok(format!(
        "# Image Attachment: {}\n\n\
         > The selected provider/model does not support raw image input, so MangoCode converted the image attachment into a Markdown metadata fallback. \
         This is not OCR and does not describe visual contents; switch to a vision-capable model for full image understanding.\n\n\
         - Source path: `{}`\n\
         - Media type: `{}`\n\
         - Size: {} bytes\n\
         - Dimensions: {}\n\
         - SHA-256: `{}`\n",
        display_name,
        path.display(),
        media_type,
        meta.len(),
        dimensions.unwrap_or_else(|| "unknown".to_string()),
        hash
    ))
}

pub fn image_markdown_with_ocr_fallback(
    path: &Path,
    label: Option<&str>,
    config: Option<&crate::config::AttachmentConfig>,
) -> anyhow::Result<String> {
    let mut markdown = image_markdown_fallback(path, label)?;
    let enabled = config.map(|c| c.ocr_enabled).unwrap_or(true);
    if !enabled {
        markdown.push_str("\n## OCR\n\nOCR is disabled by configuration.\n");
        return Ok(markdown);
    }

    let lang = config
        .map(|c| c.tesseract_lang.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("eng");
    let configured_path = config.and_then(|c| c.tesseract_path.as_deref());
    match run_tesseract_ocr(path, lang, configured_path) {
        Ok(text) if !text.trim().is_empty() => {
            markdown.push_str("\n## OCR Text\n\n");
            markdown.push_str(&truncate_ocr_text(&text));
            markdown.push('\n');
        }
        Ok(_) => {
            markdown.push_str("\n## OCR\n\nTesseract ran but did not return readable text.\n");
        }
        Err(e) => {
            markdown.push_str(&format!(
                "\n## OCR\n\nTesseract OCR was not available or could not read this image: {}\n\
                 Configure `attachments.tesseract_path`, set `MANGOCODE_TESSERACT_PATH`, \
                 store vault key `tesseract`, or put `tesseract` on PATH.\n",
                e
            ));
        }
    }
    Ok(markdown)
}

pub fn resolve_tesseract_path(configured_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = configured_path
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
    {
        return Some(path);
    }
    if let Ok(path) = std::env::var("MANGOCODE_TESSERACT_PATH") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    if let Some(path) = tesseract_path_from_vault() {
        return Some(path);
    }
    Some(PathBuf::from("tesseract"))
}

pub fn tesseract_health(configured_path: Option<&str>) -> (bool, String) {
    let Some(path) = resolve_tesseract_path(configured_path) else {
        return (false, "not configured".to_string());
    };
    match Command::new(&path).arg("--version").output() {
        Ok(output) if output.status.success() => {
            let first = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("tesseract")
                .trim()
                .to_string();
            (true, format!("{} ({})", first, path.display()))
        }
        Ok(output) => (
            false,
            format!(
                "{} returned exit code {:?}",
                path.display(),
                output.status.code()
            ),
        ),
        Err(e) => (false, format!("{} ({})", path.display(), e)),
    }
}

pub fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    let bytes = std::fs::read(path).ok()?;
    match extension(path).as_deref() {
        Some("png") => png_dimensions(&bytes),
        Some("jpg" | "jpeg") => jpeg_dimensions(&bytes),
        Some("gif") => gif_dimensions(&bytes),
        Some("webp") => webp_dimensions(&bytes),
        Some("bmp") => bmp_dimensions(&bytes),
        _ => None,
    }
}

pub fn route_attachment(
    kind: AttachmentKind,
    caps: ModelAttachmentCapabilities,
    config: AttachmentRoutingConfig,
) -> AttachmentRoutingDecision {
    match kind {
        AttachmentKind::Image => {
            if caps.image_input && config.images_raw_by_default {
                AttachmentRoutingDecision::RawImage
            } else {
                AttachmentRoutingDecision::TextInline
            }
        }
        AttachmentKind::Pdf => {
            if config.markdown_extraction_enabled {
                AttachmentRoutingDecision::MarkdownDocument
            } else if caps.pdf_input {
                AttachmentRoutingDecision::RawPdf
            } else {
                AttachmentRoutingDecision::Rejected {
                    reason: "PDF input requires native Markdown extraction or a PDF-capable model"
                        .to_string(),
                }
            }
        }
        AttachmentKind::OfficeDocument | AttachmentKind::Html | AttachmentKind::Archive => {
            if config.markdown_extraction_enabled {
                AttachmentRoutingDecision::MarkdownDocument
            } else {
                AttachmentRoutingDecision::Rejected {
                    reason: "document conversion requires native Markdown extraction".to_string(),
                }
            }
        }
        AttachmentKind::Data => {
            if config.markdown_extraction_enabled {
                AttachmentRoutingDecision::MarkdownDocument
            } else {
                AttachmentRoutingDecision::TextInline
            }
        }
        AttachmentKind::Text => AttachmentRoutingDecision::TextInline,
        AttachmentKind::Unknown => AttachmentRoutingDecision::Rejected {
            reason: "unsupported attachment type".to_string(),
        },
    }
}

pub fn build_attachment_for_path(
    path: &Path,
    caps: ModelAttachmentCapabilities,
    config: AttachmentRoutingConfig,
) -> anyhow::Result<Attachment> {
    let meta = std::fs::metadata(path)?;
    let hash = hash_file(path)?;
    let kind = classify_path(path);
    let routing_decision = route_attachment(kind, caps, config);
    let display_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("attachment")
        .to_string();

    Ok(Attachment {
        id: hash.chars().take(16).collect(),
        source: path.display().to_string(),
        media_type: media_type_for_path(path),
        kind,
        display_name,
        size_bytes: meta.len(),
        hash,
        routing_decision,
        raw_path: Some(path.to_path_buf()),
        extracted_markdown_path: None,
    })
}

pub fn markdown_cache_path(hash: &str) -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".mangocode");
    dir.push("attachments");
    dir.push(EXTRACTOR_VERSION);
    dir.push(format!("{}.md", hash));
    dir
}

pub fn markdown_cache_metadata_path(hash: &str) -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".mangocode");
    dir.push("attachments");
    dir.push(EXTRACTOR_VERSION);
    dir.push(format!("{}.json", hash));
    dir
}

pub fn extract_markdown_native(path: &Path) -> anyhow::Result<MarkdownExtraction> {
    let meta = std::fs::metadata(path)?;
    if meta.len() > MAX_DOCUMENT_INPUT_BYTES {
        anyhow::bail!(
            "attachment is too large for native Markdown extraction ({} bytes > {} bytes)",
            meta.len(),
            MAX_DOCUMENT_INPUT_BYTES
        );
    }

    let hash = hash_file(path)?;
    let cache_path = markdown_cache_path(&hash);
    let metadata_path = markdown_cache_metadata_path(&hash);
    if cache_path.exists() && cache_metadata_matches(&metadata_path, path, &hash, &meta) {
        return Ok(MarkdownExtraction {
            markdown: std::fs::read_to_string(&cache_path)?,
            cache_path,
            from_cache: true,
        });
    }

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let markdown = match classify_path(path) {
        AttachmentKind::Text | AttachmentKind::Data | AttachmentKind::Html => {
            extract_text_like_markdown(path)?
        }
        AttachmentKind::Pdf => extract_pdf_markdown(path)?,
        AttachmentKind::OfficeDocument | AttachmentKind::Archive => {
            extract_zip_container_markdown(path)?
        }
        AttachmentKind::Image => anyhow::bail!("images are preserved as raw visual input"),
        AttachmentKind::Unknown => anyhow::bail!("unsupported attachment type"),
    };
    std::fs::write(&cache_path, &markdown)?;
    let metadata = AttachmentCacheMetadata {
        source_path: path.display().to_string(),
        source_mtime_unix: file_mtime_unix(&meta),
        source_size_bytes: meta.len(),
        source_hash: hash.clone(),
        media_type: media_type_for_path(path),
        extractor_version: EXTRACTOR_VERSION.to_string(),
        extracted_at: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    Ok(MarkdownExtraction {
        markdown,
        cache_path,
        from_cache: false,
    })
}

pub fn native_markdown_available() -> bool {
    true
}

pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn extract_text_like_markdown(path: &Path) -> anyhow::Result<String> {
    let content = std::fs::read_to_string(path)?;
    if matches!(classify_path(path), AttachmentKind::Html) {
        Ok(html_to_markdownish(&content))
    } else {
        Ok(format!(
            "# {}\n\n```{}\n{}\n```",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("document"),
            extension(path).unwrap_or_else(|| "text".to_string()),
            content
        ))
    }
}

fn extract_pdf_markdown(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut text = extract_pdf_text_objects(&bytes);
    if text.split_whitespace().count() < 20 {
        text.push('\n');
        text.push_str(&extract_pdf_stream_text(&bytes));
    }
    text = normalize_extracted_text(&text);
    let page_count = count_pdf_pages(&bytes);
    let image_count = count_pdf_images(&bytes);
    if text.split_whitespace().count() < 20 {
        if let Ok(ocr_text) = run_pdf_tesseract_ocr(path, &bytes, "eng", None) {
            let normalized_ocr = normalize_extracted_text(&ocr_text);
            if normalized_ocr.split_whitespace().count() >= 20 {
                return Ok(format!(
                    "# {}\n\n\
                     > Embedded PDF text was sparse, so MangoCode used local Tesseract OCR.\n\n\
                     {}\n\n\
                     ## PDF Structure\n\n\
                     - Pages detected: {}\n\
                     - Image objects detected: {}\n\
                     - OCR engine: Tesseract\n",
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("document.pdf"),
                    truncate_ocr_text(&normalized_ocr),
                    page_count,
                    image_count
                ));
            }
        }
        return Ok(format!(
            "# {}\n\n\
             > Native extraction found little embedded text in this PDF. \
             MangoCode detected {} page(s) and {} image object(s), so this is likely scanned or image-heavy. \
             Local Tesseract OCR was attempted or unavailable; use a vision-capable model with the original file/image pages when visual reading is needed.\n\n\
             ## Extraction Status\n\n\
             - Embedded text words detected: {}\n\
             - PDF pages detected: {}\n\
             - PDF image objects detected: {}\n",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("document.pdf"),
            page_count.max(1),
            image_count,
            text.split_whitespace().count(),
            page_count,
            image_count
        ));
    }
    Ok(format!(
        "# {}\n\n{}\n\n## PDF Structure\n\n- Pages detected: {}\n- Image objects detected: {}\n",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("document.pdf"),
        text,
        page_count,
        image_count
    ))
}

fn run_tesseract_ocr(
    path: &Path,
    lang: &str,
    configured_path: Option<&str>,
) -> anyhow::Result<String> {
    let tesseract = resolve_tesseract_path(configured_path)
        .ok_or_else(|| anyhow::anyhow!("no Tesseract path configured"))?;
    let output = Command::new(&tesseract)
        .arg(path)
        .arg("stdout")
        .arg("-l")
        .arg(if lang.trim().is_empty() { "eng" } else { lang })
        .arg("--psm")
        .arg("3")
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run {}: {}", tesseract.display(), e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "{} exited with {:?}: {}",
            tesseract.display(),
            output.status.code(),
            stderr.trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_pdf_tesseract_ocr(
    path: &Path,
    bytes: &[u8],
    lang: &str,
    configured_path: Option<&str>,
) -> anyhow::Result<String> {
    if let Ok(text) = run_tesseract_ocr(path, lang, configured_path) {
        if text.split_whitespace().count() >= 10 {
            return Ok(text);
        }
    }

    let mut out = Vec::new();
    for (idx, jpeg) in extract_embedded_jpegs(bytes)
        .into_iter()
        .take(20)
        .enumerate()
    {
        let tmp_path = std::env::temp_dir().join(format!(
            "mangocode-ocr-{}-{}.jpg",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
            idx
        ));
        std::fs::write(&tmp_path, jpeg)?;
        let result = run_tesseract_ocr(&tmp_path, lang, configured_path);
        let _ = std::fs::remove_file(&tmp_path);
        if let Ok(text) = result {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push(format!("## OCR page/image {}\n\n{}", idx + 1, trimmed));
            }
        }
    }

    if out.is_empty() {
        anyhow::bail!("Tesseract could not OCR the PDF or embedded page images");
    }
    Ok(out.join("\n\n"))
}

fn extract_embedded_jpegs(bytes: &[u8]) -> Vec<&[u8]> {
    let mut ranges = Vec::new();
    let mut cursor = 0usize;
    while cursor + 4 < bytes.len() && ranges.len() < 50 {
        let Some(start_rel) = bytes[cursor..].windows(2).position(|w| w == [0xFF, 0xD8]) else {
            break;
        };
        let start = cursor + start_rel;
        let Some(end_rel) = bytes[start + 2..]
            .windows(2)
            .position(|w| w == [0xFF, 0xD9])
        else {
            break;
        };
        let end = start + 2 + end_rel + 2;
        if end > start && end - start > 1024 {
            ranges.push(&bytes[start..end]);
        }
        cursor = end;
    }
    ranges
}

fn tesseract_path_from_vault() -> Option<PathBuf> {
    let passphrase = crate::vault::get_vault_passphrase()?;
    let vault = crate::vault::Vault::new();
    for key in TESSERACT_VAULT_KEYS {
        if let Ok(Some(secret)) = vault.get_secret(key, &passphrase) {
            let trimmed = secret.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

fn truncate_ocr_text(text: &str) -> String {
    let normalized = normalize_extracted_text(text);
    if normalized.len() <= MAX_EXTRACTED_CHARS {
        normalized
    } else {
        format!(
            "{}\n\n... (OCR text truncated, {} total characters)",
            crate::truncate::truncate_bytes_prefix(&normalized, MAX_EXTRACTED_CHARS),
            normalized.len()
        )
    }
}

fn extract_zip_container_markdown(path: &Path) -> anyhow::Result<String> {
    let snippets = match extension(path).as_deref() {
        Some("docx") => extract_docx_markdown(path)?,
        Some("pptx") => extract_pptx_markdown(path)?,
        Some("xlsx") => extract_xlsx_markdown(path)?,
        Some("zip") => extract_generic_zip_markdown(path)?,
        Some("doc" | "ppt" | "xls") => extract_legacy_office_strings(path)?,
        _ => extract_generic_zip_markdown(path)?,
    };
    if snippets.trim().is_empty() {
        anyhow::bail!("native ZIP/OOXML extraction could not find readable text");
    }
    Ok(format!(
        "# {}\n\n{}",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("document"),
        snippets
    ))
}

fn extract_docx_markdown(path: &Path) -> anyhow::Result<String> {
    let files = read_zip_text_files(path, |name| {
        name == "word/document.xml"
            || name.starts_with("word/header")
            || name.starts_with("word/footer")
            || name.starts_with("word/footnotes")
            || name.starts_with("word/endnotes")
            || name.starts_with("word/comments")
            || name.starts_with("word/charts/")
            || name.starts_with("word/embeddings/")
    })?;
    let mut out = Vec::new();
    for (name, xml) in files {
        let text = extract_ooxml_text(&xml);
        if !text.trim().is_empty() {
            out.push(format!("## {}\n\n{}", name, text));
        }
    }
    Ok(out.join("\n\n"))
}

fn extract_pptx_markdown(path: &Path) -> anyhow::Result<String> {
    let mut files = read_zip_text_files(path, |name| {
        (name.starts_with("ppt/slides/slide")
            || name.starts_with("ppt/notesSlides/notesSlide")
            || name.starts_with("ppt/comments/")
            || name.starts_with("ppt/charts/"))
            && name.ends_with(".xml")
    })?;
    files.sort_by_key(|(name, _)| slide_number(name).unwrap_or(usize::MAX));
    let mut out = Vec::new();
    for (idx, (name, xml)) in files.into_iter().enumerate() {
        let text = extract_ooxml_text(&xml);
        if !text.trim().is_empty() {
            out.push(format!("## Slide {} ({})\n\n{}", idx + 1, name, text));
        }
    }
    Ok(out.join("\n\n"))
}

fn extract_xlsx_markdown(path: &Path) -> anyhow::Result<String> {
    let shared_strings_xml = read_zip_text_files(path, |name| name == "xl/sharedStrings.xml")?
        .into_iter()
        .next()
        .map(|(_, xml)| xml)
        .unwrap_or_default();
    let shared_strings = extract_shared_strings(&shared_strings_xml);

    let mut sheets = read_zip_text_files(path, |name| {
        name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml")
    })?;
    sheets.sort_by_key(|(name, _)| sheet_number(name).unwrap_or(usize::MAX));

    let mut out = Vec::new();
    for (idx, (name, xml)) in sheets.into_iter().enumerate() {
        let rows = extract_worksheet_rows(&xml, &shared_strings);
        if !rows.is_empty() {
            let body = rows
                .into_iter()
                .take(300)
                .map(|row| format!("| {} |", row.join(" | ")))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(format!("## Sheet {} ({})\n\n{}", idx + 1, name, body));
        }
    }
    let mut charts = read_zip_text_files(path, |name| {
        name.starts_with("xl/charts/") && name.ends_with(".xml")
    })?;
    charts.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, xml) in charts.into_iter().take(50) {
        let text = extract_ooxml_text(&xml);
        if !text.trim().is_empty() {
            out.push(format!("## Chart ({})\n\n{}", name, text));
        }
    }
    Ok(out.join("\n\n"))
}

fn extract_legacy_office_strings(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = extract_printable_strings(&bytes);
    if text.split_whitespace().count() < 20 {
        anyhow::bail!(
            "legacy binary Office extraction found little readable text; save as .docx/.pptx/.xlsx for full native extraction"
        );
    }
    Ok(format!(
        "> Best-effort text recovered from a legacy binary Office file. \
         Layout, charts, comments, and embedded media may be incomplete; save as OOXML for higher fidelity.\n\n{}",
        text
    ))
}

fn extract_generic_zip_markdown(path: &Path) -> anyhow::Result<String> {
    let files = read_zip_text_files(path, |name| {
        let lower = name.to_ascii_lowercase();
        lower.ends_with(".txt")
            || lower.ends_with(".md")
            || lower.ends_with(".markdown")
            || lower.ends_with(".html")
            || lower.ends_with(".htm")
            || lower.ends_with(".xml")
            || lower.ends_with(".json")
            || lower.ends_with(".csv")
            || lower.ends_with(".toml")
            || lower.ends_with(".yaml")
            || lower.ends_with(".yml")
    })?;
    let mut out = Vec::new();
    for (name, content) in files.into_iter().take(50) {
        let body = if name.ends_with(".html") || name.ends_with(".htm") {
            html_to_markdownish(&content)
        } else if name.ends_with(".xml") {
            extract_ooxml_text(&content)
        } else {
            content
        };
        if !body.trim().is_empty() {
            out.push(format!("## {}\n\n{}", name, truncate_chars(&body, 20_000)));
        }
    }
    Ok(out.join("\n\n"))
}

fn read_zip_text_files<F>(path: &Path, mut include: F) -> anyhow::Result<Vec<(String, String)>>
where
    F: FnMut(&str) -> bool,
{
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut out = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().replace('\\', "/");
        if entry.is_dir() || !include(&name) || entry.size() > 10 * 1024 * 1024 {
            continue;
        }
        let mut buf = String::new();
        if entry.read_to_string(&mut buf).is_ok() {
            out.push((name, buf));
        }
        if out.len() >= 200 {
            break;
        }
    }
    Ok(out)
}

fn html_to_markdownish(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let lower = html.to_ascii_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if !in_tag && chars[i] == '<' {
            in_tag = true;
            let rest: String = lower_chars[i..].iter().take(20).collect();
            if rest.starts_with("<script") {
                in_script = true;
            } else if rest.starts_with("</script") {
                in_script = false;
            } else if rest.starts_with("<style") {
                in_style = true;
            } else if rest.starts_with("</style") {
                in_style = false;
            } else if rest.starts_with("<h1") {
                text.push_str("\n# ");
            } else if rest.starts_with("<h2") {
                text.push_str("\n## ");
            } else if rest.starts_with("<h3") {
                text.push_str("\n### ");
            } else if rest.starts_with("<p")
                || rest.starts_with("<li")
                || rest.starts_with("<br")
                || rest.starts_with("<tr")
                || rest.starts_with("<pre")
                || rest.starts_with("<code")
            {
                text.push('\n');
            }
            i += 1;
            continue;
        }
        if in_tag {
            if chars[i] == '>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }
        if !in_script && !in_style {
            text.push(chars[i]);
        }
        i += 1;
    }
    decode_entities(&text)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_extracted_text(text: &str) -> String {
    let normalized = text
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t");
    normalized
        .lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && line.chars().filter(|c| c.is_control()).count() < line.len().saturating_div(4)
                && !line.starts_with('%')
                && !line.contains(" obj")
                && !line.contains(" endobj")
                && !line.contains("stream")
                && !line.contains("endstream")
        })
        .take(20_000)
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_ooxml_text(text: &str) -> String {
    let runs = extract_xml_text_runs(text);
    runs.join(" ")
        .replace(" \n ", "\n")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_xml_text_runs(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut cursor = 0usize;
    while let Some(start_rel) = text[cursor..].find('>') {
        let start = cursor + start_rel + 1;
        let Some(end_rel) = text[start..].find('<') else {
            break;
        };
        let end = start + end_rel;
        let run = decode_entities(text[start..end].trim());
        if run.chars().filter(|c| c.is_alphanumeric()).count() >= 3 {
            runs.push(run);
        }
        cursor = end + 1;
        if runs.len() >= 2000 {
            break;
        }
    }
    runs
}

fn extract_shared_strings(xml: &str) -> Vec<String> {
    extract_xml_text_runs(xml)
}

fn extract_worksheet_rows(xml: &str, shared_strings: &[String]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut cursor = 0usize;
    while let Some(row_start_rel) = xml[cursor..].find("<row") {
        let row_start = cursor + row_start_rel;
        let Some(row_open_end_rel) = xml[row_start..].find('>') else {
            break;
        };
        let body_start = row_start + row_open_end_rel + 1;
        let Some(row_end_rel) = xml[body_start..].find("</row>") else {
            break;
        };
        let body_end = body_start + row_end_rel;
        let row_xml = &xml[body_start..body_end];
        let row = extract_cells(row_xml, shared_strings);
        if !row.is_empty() {
            rows.push(row);
        }
        cursor = body_end + "</row>".len();
        if rows.len() >= 500 {
            break;
        }
    }
    rows
}

fn count_pdf_pages(bytes: &[u8]) -> usize {
    String::from_utf8_lossy(bytes)
        .matches("/Type /Page")
        .count()
}

fn count_pdf_images(bytes: &[u8]) -> usize {
    let haystack = String::from_utf8_lossy(bytes).to_ascii_lowercase();
    haystack.matches("/subtype /image").count() + haystack.matches("/subtype/image").count()
}

fn extract_printable_strings(bytes: &[u8]) -> String {
    let mut runs = Vec::new();
    let mut current = Vec::new();
    for &b in bytes {
        if b.is_ascii_graphic() || b == b' ' || b == b'\t' {
            current.push(b);
        } else {
            if current.len() >= 4 {
                runs.push(String::from_utf8_lossy(&current).to_string());
            }
            current.clear();
        }
        if runs.len() >= 10_000 {
            break;
        }
    }
    if current.len() >= 4 {
        runs.push(String::from_utf8_lossy(&current).to_string());
    }
    runs.into_iter()
        .map(|line| decode_entities(line.trim()))
        .filter(|line| {
            let alnum = line.chars().filter(|c| c.is_alphanumeric()).count();
            alnum >= 3 && alnum.saturating_mul(2) >= line.len().min(80)
        })
        .take(5000)
        .collect::<Vec<_>>()
        .join("\n")
}

fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    ))
}

fn gif_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 10 || !bytes.starts_with(b"GIF8") {
        return None;
    }
    Some((
        u16::from_le_bytes(bytes[6..8].try_into().ok()?) as u32,
        u16::from_le_bytes(bytes[8..10].try_into().ok()?) as u32,
    ))
}

fn bmp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 26 || !bytes.starts_with(b"BM") {
        return None;
    }
    Some((
        u32::from_le_bytes(bytes[18..22].try_into().ok()?),
        u32::from_le_bytes(bytes[22..26].try_into().ok()?),
    ))
}

fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 30 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    if &bytes[12..16] == b"VP8X" && bytes.len() >= 30 {
        let w = 1 + u32::from_le_bytes([bytes[24], bytes[25], bytes[26], 0]);
        let h = 1 + u32::from_le_bytes([bytes[27], bytes[28], bytes[29], 0]);
        Some((w, h))
    } else if &bytes[12..16] == b"VP8 " && bytes.len() >= 30 {
        Some((
            (u16::from_le_bytes(bytes[26..28].try_into().ok()?) & 0x3fff) as u32,
            (u16::from_le_bytes(bytes[28..30].try_into().ok()?) & 0x3fff) as u32,
        ))
    } else {
        None
    }
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return None;
    }
    let mut i = 2usize;
    while i + 9 < bytes.len() {
        while i < bytes.len() && bytes[i] != 0xff {
            i += 1;
        }
        if i + 3 >= bytes.len() {
            return None;
        }
        let marker = bytes[i + 1];
        i += 2;
        if marker == 0xd8 || marker == 0xd9 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if i + 2 > bytes.len() {
            return None;
        }
        let len = u16::from_be_bytes(bytes[i..i + 2].try_into().ok()?) as usize;
        if len < 2 || i + len > bytes.len() {
            return None;
        }
        if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf) && len >= 7 {
            let h = u16::from_be_bytes(bytes[i + 3..i + 5].try_into().ok()?) as u32;
            let w = u16::from_be_bytes(bytes[i + 5..i + 7].try_into().ok()?) as u32;
            return Some((w, h));
        }
        i += len;
    }
    None
}

fn extract_cells(row_xml: &str, shared_strings: &[String]) -> Vec<String> {
    let mut cells = Vec::new();
    let mut cursor = 0usize;
    while let Some(c_start_rel) = row_xml[cursor..].find("<c") {
        let c_start = cursor + c_start_rel;
        let Some(c_open_end_rel) = row_xml[c_start..].find('>') else {
            break;
        };
        let attrs = &row_xml[c_start..c_start + c_open_end_rel];
        let body_start = c_start + c_open_end_rel + 1;
        let Some(c_end_rel) = row_xml[body_start..].find("</c>") else {
            break;
        };
        let body_end = body_start + c_end_rel;
        let body = &row_xml[body_start..body_end];
        let formula = tag_text(body, "f").map(|f| decode_entities(f.trim()));
        let mut value = tag_text(body, "v").unwrap_or_else(|| extract_ooxml_text(body));
        if attrs.contains("t=\"s\"") {
            if let Ok(idx) = value.trim().parse::<usize>() {
                if let Some(shared) = shared_strings.get(idx) {
                    value = shared.clone();
                }
            }
        }
        let mut value = decode_entities(value.trim());
        if let Some(formula) = formula.filter(|f| !f.is_empty()) {
            value = if value.is_empty() {
                format!("={}", formula)
            } else {
                format!("{} (formula: ={})", value, formula)
            };
        }
        if !value.is_empty() {
            cells.push(value);
        }
        cursor = body_end + "</c>".len();
        if cells.len() >= 200 {
            break;
        }
    }
    cells
}

fn tag_text(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}>", tag);
    let end_tag = format!("</{}>", tag);
    let start = xml.find(&start_tag)? + start_tag.len();
    let end = xml[start..].find(&end_tag)? + start;
    Some(xml[start..end].to_string())
}

fn slide_number(path: &str) -> Option<usize> {
    path.rsplit_once("slide")
        .and_then(|(_, rest)| rest.strip_suffix(".xml"))
        .and_then(|n| n.parse().ok())
}

fn sheet_number(path: &str) -> Option<usize> {
    path.rsplit_once("sheet")
        .and_then(|(_, rest)| rest.strip_suffix(".xml"))
        .and_then(|n| n.parse().ok())
}

fn extract_pdf_stream_text(bytes: &[u8]) -> String {
    let haystack = String::from_utf8_lossy(bytes);
    let mut out = String::new();
    let mut cursor = 0usize;
    while let Some(stream_rel) = haystack[cursor..].find("stream") {
        let stream_marker = cursor + stream_rel;
        let data_start = stream_marker + "stream".len();
        let data_start =
            if bytes.get(data_start) == Some(&b'\r') && bytes.get(data_start + 1) == Some(&b'\n') {
                data_start + 2
            } else if bytes.get(data_start) == Some(&b'\n') {
                data_start + 1
            } else {
                data_start
            };
        let Some(end_rel) = haystack[data_start..].find("endstream") else {
            break;
        };
        let data_end = data_start + end_rel;
        let header = &haystack[stream_marker.saturating_sub(500)..stream_marker];
        let data = &bytes[data_start..data_end.min(bytes.len())];
        let decoded = if header.contains("/FlateDecode") {
            let mut decoder = flate2::read::ZlibDecoder::new(Cursor::new(data));
            let mut decoded = Vec::new();
            if decoder.read_to_end(&mut decoded).is_ok() {
                decoded
            } else {
                data.to_vec()
            }
        } else {
            data.to_vec()
        };
        out.push_str(&extract_pdf_text_objects(&decoded));
        out.push('\n');
        cursor = data_end + "endstream".len();
        if out.len() > MAX_EXTRACTED_CHARS {
            break;
        }
    }
    out
}

fn extract_pdf_text_objects(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '(' => {
                let (literal, next) = read_pdf_literal_string(&chars, i + 1);
                let next_non_ws = skip_ws(&chars, next);
                if follows_pdf_text_operator(&chars, next_non_ws) {
                    out.push_str(&literal);
                    out.push('\n');
                }
                i = next;
            }
            '<' if chars.get(i + 1) != Some(&'<') => {
                let (hex, next) = read_pdf_hex_string(&chars, i + 1);
                let next_non_ws = skip_ws(&chars, next);
                if follows_pdf_text_operator(&chars, next_non_ws) {
                    out.push_str(&hex);
                    out.push('\n');
                }
                i = next;
            }
            _ => i += 1,
        }
        if out.len() > MAX_EXTRACTED_CHARS {
            break;
        }
    }
    out
}

fn read_pdf_literal_string(chars: &[char], mut i: usize) -> (String, usize) {
    let mut out = String::new();
    let mut depth = 1usize;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                if let Some(next) = chars.get(i + 1) {
                    match next {
                        'n' => out.push('\n'),
                        'r' => out.push('\n'),
                        't' => out.push('\t'),
                        'b' | 'f' => out.push(' '),
                        '(' | ')' | '\\' => out.push(*next),
                        _ => out.push(*next),
                    }
                    i += 2;
                    continue;
                }
            }
            '(' => {
                depth += 1;
                out.push('(');
            }
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return (decode_entities(&out), i + 1);
                }
                out.push(')');
            }
            c => out.push(c),
        }
        i += 1;
    }
    (decode_entities(&out), i)
}

fn read_pdf_hex_string(chars: &[char], mut i: usize) -> (String, usize) {
    let mut hex = String::new();
    while i < chars.len() && chars[i] != '>' {
        if chars[i].is_ascii_hexdigit() {
            hex.push(chars[i]);
        }
        i += 1;
    }
    let mut bytes = Vec::new();
    for chunk in hex.as_bytes().chunks(2) {
        if chunk.len() == 2 {
            if let Ok(part) = std::str::from_utf8(chunk) {
                if let Ok(byte) = u8::from_str_radix(part, 16) {
                    bytes.push(byte);
                }
            }
        }
    }
    (String::from_utf8_lossy(&bytes).to_string(), i + 1)
}

fn skip_ws(chars: &[char], mut i: usize) -> usize {
    while chars.get(i).is_some_and(|c| c.is_whitespace()) {
        i += 1;
    }
    i
}

fn follows_pdf_text_operator(chars: &[char], i: usize) -> bool {
    let op: String = chars.iter().skip(i).take(3).collect();
    op.starts_with("Tj") || op.starts_with("TJ") || op.starts_with("'") || op.starts_with("\"")
}

fn truncate_chars(text: &str, max: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max) {
        out.push(ch);
    }
    if text.chars().count() > max {
        out.push_str("\n\n... (truncated)");
    }
    out
}

fn file_mtime_unix(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

fn cache_metadata_matches(
    metadata_path: &Path,
    source_path: &Path,
    hash: &str,
    meta: &std::fs::Metadata,
) -> bool {
    let Ok(raw) = std::fs::read_to_string(metadata_path) else {
        return false;
    };
    let Ok(metadata) = serde_json::from_str::<AttachmentCacheMetadata>(&raw) else {
        return false;
    };
    metadata.extractor_version == EXTRACTOR_VERSION
        && metadata.source_hash == hash
        && metadata.source_size_bytes == meta.len()
        && metadata.source_mtime_unix == file_mtime_unix(meta)
        && metadata.source_path == source_path.display().to_string()
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_images_raw_for_vision_models() {
        let decision = route_attachment(
            AttachmentKind::Image,
            ModelAttachmentCapabilities {
                image_input: true,
                pdf_input: false,
            },
            AttachmentRoutingConfig::default(),
        );
        assert_eq!(decision, AttachmentRoutingDecision::RawImage);
    }

    #[test]
    fn routes_pdfs_to_markdown_by_default_even_when_pdf_capable() {
        let decision = route_attachment(
            AttachmentKind::Pdf,
            ModelAttachmentCapabilities {
                image_input: true,
                pdf_input: true,
            },
            AttachmentRoutingConfig::default(),
        );
        assert_eq!(decision, AttachmentRoutingDecision::MarkdownDocument);
    }

    #[test]
    fn routes_images_to_text_fallback_when_vision_is_unavailable() {
        let decision = route_attachment(
            AttachmentKind::Image,
            ModelAttachmentCapabilities::default(),
            AttachmentRoutingConfig::default(),
        );
        assert_eq!(decision, AttachmentRoutingDecision::TextInline);
    }

    #[test]
    fn classifies_common_document_types() {
        assert_eq!(classify_path(Path::new("a.pdf")), AttachmentKind::Pdf);
        assert_eq!(
            classify_path(Path::new("a.docx")),
            AttachmentKind::OfficeDocument
        );
        assert_eq!(classify_path(Path::new("a.png")), AttachmentKind::Image);
        assert_eq!(classify_path(Path::new("a.html")), AttachmentKind::Html);
    }

    #[test]
    fn provider_model_capabilities_are_conservative_for_text_only_models() {
        assert!(!model_attachment_capabilities(Some("cohere"), "command-a").image_input);
        assert!(!model_attachment_capabilities(Some("deepseek"), "deepseek-chat").image_input);
        assert!(!model_attachment_capabilities(Some("ollama"), "llama3.2").image_input);
        assert!(model_attachment_capabilities(Some("ollama"), "llava:latest").image_input);
        assert!(model_attachment_capabilities(Some("openai"), "gpt-4o").image_input);
        assert!(model_attachment_capabilities(Some("anthropic"), "claude-sonnet-4-6").pdf_input);
        assert!(!model_attachment_capabilities(Some("openai"), "gpt-4o").pdf_input);
    }

    #[test]
    fn image_markdown_fallback_includes_metadata() {
        let tmp = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .expect("tmp");
        let mut data = vec![0u8; 24];
        data[0..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        data[8..12].copy_from_slice(&13u32.to_be_bytes());
        data[12..16].copy_from_slice(b"IHDR");
        data[16..20].copy_from_slice(&320u32.to_be_bytes());
        data[20..24].copy_from_slice(&200u32.to_be_bytes());
        std::fs::write(tmp.path(), data).expect("write");

        let markdown = image_markdown_fallback(tmp.path(), Some("mock.png")).expect("fallback");
        assert!(markdown.contains("Image Attachment: mock.png"));
        assert!(markdown.contains("Dimensions: 320x200"));
        assert!(markdown.contains("SHA-256"));
        assert!(markdown.contains("not OCR"));
    }

    #[test]
    fn sparse_pdf_returns_scanned_status_markdown() {
        let markdown = extract_pdf_markdown(Path::new("missing")).err();
        assert!(markdown.is_some());
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        std::fs::write(
            tmp.path(),
            b"%PDF-1.7\n1 0 obj << /Type /Page /Resources << /XObject << /Im1 << /Subtype /Image >> >> >> >> endobj\n%%EOF",
        )
        .expect("write");
        let markdown = extract_pdf_markdown(tmp.path()).expect("extract");
        assert!(markdown.contains("likely scanned or image-heavy"));
        assert!(markdown.contains("PDF image objects detected"));
    }

    #[test]
    fn extracts_embedded_jpeg_ranges_from_pdf_like_bytes() {
        let bytes = [
            b"%PDF".as_slice(),
            &[0xFF, 0xD8],
            &[7u8; 1500],
            &[0xFF, 0xD9],
            b"trailer".as_slice(),
        ]
        .concat();
        let jpegs = extract_embedded_jpegs(&bytes);
        assert_eq!(jpegs.len(), 1);
        assert_eq!(&jpegs[0][..2], &[0xFF, 0xD8]);
        assert_eq!(&jpegs[0][jpegs[0].len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn legacy_office_strings_are_best_effort() {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        std::fs::write(
            tmp.path(),
            b"\0\0This legacy document contains enough printable project planning text to recover for MangoCode attachment routing research memory browser search output compression verification and implementation notes.\0\0",
        )
        .expect("write");
        let markdown = extract_legacy_office_strings(tmp.path()).expect("extract");
        assert!(markdown.contains("Best-effort text"));
        assert!(markdown.contains("legacy document"));
    }
}
