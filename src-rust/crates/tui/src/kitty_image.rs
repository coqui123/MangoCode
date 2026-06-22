// kitty_image.rs — Inline image rendering via Kitty graphics protocol or Sixel.
//
// Strategy:
//   1. Detect which image protocol the terminal supports:
//      - Kitty: $TERM contains "kitty" OR $TERM_PROGRAM is "WezTerm"
//      - iTerm: $TERM_PROGRAM is "iTerm.app"
//      - Sixel: $TERM contains xterm/screen/rxvt/mintty/iterm2
//      - Text: fallback to human-readable description
//   2. If base64 PNG/JPEG data is available:
//      - Kitty: emit APC escape sequence directly
//      - iTerm: emit OSC 1337 inline-image sequence
//      - Sixel: decode base64 → PNG/JPEG → convert to Sixel → emit escape sequence
//      - Text: return a placeholder string
//   3. For URL sources: always fall back to text (no remote fetching)
//
// Kitty graphics protocol (APC sequence):
//   ESC _ G a=T,f=<fmt>,m=0,q=2,C=1 ; <base64-data> ESC \
//
// Sixel escape sequence:
//   ESC P q ... ESC \

use mangocode_core::ImageSource;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum bytes per Kitty APC chunk.
const KITTY_CHUNK_SIZE: usize = 4096;

/// Maximum bytes per Sixel line (conservative limit for terminal compatibility).
const SIXEL_LINE_SIZE: usize = 1024;

/// Reserved Kitty image ID for the mascot (see rustle.rs).
/// Mascot re-emissions reuse this ID so the terminal replaces the stored image
/// in place instead of accumulating one entry per position change.
pub const MASCOT_IMAGE_ID: u32 = 1;

/// First image ID handed out to transcript images. The mascot owns ID 1.
const FIRST_TRANSCRIPT_IMAGE_ID: u32 = 2;

/// Hard cap on the number of distinct transcript images cached in terminal
/// memory simultaneously. When exceeded, the least-recently-emitted image is
/// evicted via `a=d,i=<id>` so memory stays bounded over long sessions.
const MAX_CACHED_TRANSCRIPT_IMAGES: usize = 256;

/// Sequential image-ID generator for transcript images.
static NEXT_TRANSCRIPT_IMAGE_ID: AtomicU32 = AtomicU32::new(FIRST_TRANSCRIPT_IMAGE_ID);

/// Content-hash → (kitty image ID, emit-counter) map. Lets the same image
/// content reuse one terminal-memory entry across many render passes instead
/// of allocating a fresh entry every frame.
static TRANSCRIPT_IMAGE_CACHE: Mutex<Option<TranscriptImageCache>> = Mutex::new(None);

struct TranscriptImageCache {
    /// content hash → (image id, last-emit tick)
    by_hash: HashMap<u64, (u32, u64)>,
    /// monotonically increasing counter — drives LRU eviction
    tick: u64,
}

impl TranscriptImageCache {
    fn new() -> Self {
        Self {
            by_hash: HashMap::new(),
            tick: 0,
        }
    }
}

/// Look up an image ID for the given content hash, or assign a new one.
///
/// Returns `(image_id, is_new)`:
///   - `is_new = true` means this hash hasn't been seen; the caller must emit
///     the full image data (the terminal doesn't have it yet).
///   - `is_new = false` means the terminal already has this image stored; the
///     caller can emit just a "display" sequence (`a=p,i=<id>`).
///
/// When the cache exceeds `MAX_CACHED_TRANSCRIPT_IMAGES`, the least-recently-
/// emitted image's ID is evicted with `a=d,i=<id>` so the terminal can free it.
pub fn intern_transcript_image_id(content_hash: u64) -> (u32, bool) {
    let mut guard = TRANSCRIPT_IMAGE_CACHE.lock();
    let cache = guard.get_or_insert_with(TranscriptImageCache::new);
    cache.tick = cache.tick.wrapping_add(1);
    let now = cache.tick;
    if let Some(entry) = cache.by_hash.get_mut(&content_hash) {
        entry.1 = now;
        return (entry.0, false);
    }
    // Evict if we'd exceed the cap.
    if cache.by_hash.len() >= MAX_CACHED_TRANSCRIPT_IMAGES {
        if let Some((&evict_hash, &(evict_id, _))) =
            cache.by_hash.iter().min_by_key(|(_, &(_, t))| t)
        {
            cache.by_hash.remove(&evict_hash);
            emit_kitty_delete_id(evict_id);
        }
    }
    let id = NEXT_TRANSCRIPT_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
    cache.by_hash.insert(content_hash, (id, now));
    (id, true)
}

/// Clear the in-process image cache and reset the ID counter. Call this when
/// the terminal-side image store has been wiped (`reset_transcript`) so we
/// don't keep stale mappings around.
pub fn reset_transcript_image_cache() {
    if let Some(cache) = TRANSCRIPT_IMAGE_CACHE.lock().as_mut() {
        cache.by_hash.clear();
        cache.tick = 0;
    }
    NEXT_TRANSCRIPT_IMAGE_ID.store(FIRST_TRANSCRIPT_IMAGE_ID, Ordering::Relaxed);
}

/// Delete the mascot's on-screen Kitty placement(s) (image ID 1) while keeping
/// the image data cached. Used when the mascot is hidden or repositioned so a
/// stale placement doesn't linger behind a re-blit. No-op on non-Kitty terminals.
pub fn delete_mascot_placement() {
    if detect_image_protocol() == ImageProtocol::Kitty {
        emit_kitty_delete_id(MASCOT_IMAGE_ID);
    }
}

/// Emit `a=d,i=<id>` to evict one specific image from terminal memory.
fn emit_kitty_delete_id(id: u32) {
    // Skip the actual escape-sequence emission in tests so we don't spew
    // terminal control codes into `cargo test` output.
    if cfg!(test) {
        return;
    }
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\x1b_Ga=d,d=i,i={}\x1b\\", id);
    let _ = stdout.flush();
}

// ---------------------------------------------------------------------------
// Image Protocol Detection
// ---------------------------------------------------------------------------

/// Supported image protocols in order of preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    Kitty,
    Iterm,
    Sixel,
    Text,
}

/// Detect which image protocol the running terminal supports.
///
/// Returns:
/// - `ImageProtocol::Kitty` if $TERM contains "kitty" or $TERM_PROGRAM is "WezTerm"
/// - `ImageProtocol::Iterm` if $TERM_PROGRAM is "iTerm.app"
/// - `ImageProtocol::Sixel` if $TERM contains xterm/screen/rxvt/mintty/iterm
/// - `ImageProtocol::Text` as fallback
pub fn detect_image_protocol() -> ImageProtocol {
    // Explicit override for testing/debugging protocol behavior.
    if let Ok(force) = std::env::var("MANGOCODE_IMAGE_PROTOCOL") {
        match force.trim().to_ascii_lowercase().as_str() {
            "kitty" => return ImageProtocol::Kitty,
            "iterm" | "osc1337" | "imgcat" => return ImageProtocol::Iterm,
            "sixel" => return ImageProtocol::Sixel,
            "text" | "none" => return ImageProtocol::Text,
            _ => {}
        }
    }

    // Check for Kitty protocol (highest priority)
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("kitty") {
            return ImageProtocol::Kitty;
        }
    }

    if let Ok(prog) = std::env::var("TERM_PROGRAM") {
        if prog.eq_ignore_ascii_case("WezTerm") {
            return ImageProtocol::Kitty;
        }
        if prog.eq_ignore_ascii_case("iTerm.app") {
            return ImageProtocol::Iterm;
        }
    }

    // SSH sessions: fall back to text — most SSH terminals don't support
    // inline image protocols, and sixel data will appear as garbage.
    if std::env::var("SSH_CONNECTION").is_ok() || std::env::var("SSH_TTY").is_ok() {
        return ImageProtocol::Text;
    }

    // Check for Sixel protocol (medium priority)
    // Windows Terminal (v1.22+) supports Sixel; detected via WT_SESSION env var.
    if std::env::var("WT_SESSION").is_ok() {
        return ImageProtocol::Sixel;
    }

    // Only enable sixel for terminals known to support it.
    // Plain "xterm-256color" does NOT imply sixel — most terminal emulators
    // set TERM=xterm-256color but don't support sixel at all.
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("mintty") || term.contains("mlterm") || term.contains("foot") {
            return ImageProtocol::Sixel;
        }
    }

    // Fallback to text
    ImageProtocol::Text
}

// Kept for backward compatibility
pub fn supports_kitty_graphics() -> bool {
    detect_image_protocol() == ImageProtocol::Kitty
}

// ---------------------------------------------------------------------------
// Core Rendering
// ---------------------------------------------------------------------------

/// Attempt to render `source` as an inline image.
///
/// * If an image protocol is available and the source carries base64 data,
///   the appropriate escape sequence is written to `stdout` and `None` is
///   returned (caller should skip adding a text line).
/// * Otherwise a human-readable fallback string is returned for display
///   as a normal text span.
///
/// The caller must flush stdout after this call when `None` is returned.
pub fn render_image(source: &ImageSource) -> Option<String> {
    // URL-type sources: never fetch remote URLs — fall back to text
    if source.source_type == "url" {
        let url = source.url.as_deref().unwrap_or("(no url)");
        return Some(format!("[Image: {}]", url));
    }

    // base64 data source
    if let Some(data) = &source.data {
        let protocol = detect_image_protocol();

        match protocol {
            ImageProtocol::Kitty => {
                // Dedup by content hash: identical images reuse the same
                // terminal-side image ID instead of allocating a new one.
                let hash = content_hash(data);
                let (image_id, is_new) = intern_transcript_image_id(hash);
                let ok = if is_new {
                    emit_kitty_apc(data, source.media_type.as_deref(), image_id)
                } else {
                    emit_kitty_redisplay(image_id)
                };
                if ok {
                    return None; // successfully emitted — caller skips text line
                }
            }
            ImageProtocol::Iterm => {
                if emit_iterm_osc1337(data, source.media_type.as_deref()) {
                    return None;
                }
            }
            ImageProtocol::Sixel => {
                if emit_sixel(data, source.media_type.as_deref()) {
                    return None; // successfully emitted — caller skips text line
                }
                // Fall through to text if Sixel conversion fails
            }
            ImageProtocol::Text => {
                // Fall through to generate fallback text
            }
        }

        // Fallback: describe the type and rough size
        let media = source.media_type.as_deref().unwrap_or("image");
        let size_kb = (data.len() * 3 / 4) / 1024; // rough decoded byte count
        if size_kb > 0 {
            return Some(format!("[Image: {} ~{}KB]", media, size_kb));
        }
        return Some(format!("[Image: {}]", media));
    }

    // No data, no URL
    Some("[Image: embedded image]".to_string())
}

// ---------------------------------------------------------------------------
// Kitty Graphics Protocol (APC)
// ---------------------------------------------------------------------------

/// Returns the Kitty format parameter for a base64-encoded image payload.
///
/// Kitty `f=100` accepts a base64-encoded PNG or JPEG directly; the terminal
/// auto-detects the image type from the data header.  We always use `f=100`.
fn kitty_format(_media_type: Option<&str>) -> u8 {
    100
}

/// Emit a Kitty graphics "delete all images" APC sequence.
///
/// This frees image-related memory held by the terminal emulator. Useful
/// after `Clear(All)` to also evict images from the terminal's offscreen
/// store — otherwise long sessions accumulate every transcript image in
/// terminal memory even after the screen is cleared.
///
/// No-op on terminals that don't support the Kitty graphics protocol;
/// terminals that don't recognize the APC sequence ignore it silently.
pub fn emit_kitty_delete_all() {
    use std::io::Write;
    let mut stdout = std::io::stdout();
    // a=d (delete action), d=A (apply to All images regardless of ID).
    let _ = write!(stdout, "\x1b_Ga=d,d=A\x1b\\");
    let _ = stdout.flush();
}

/// Compute a stable content hash for image dedup. Uses the std `DefaultHasher`
/// — this only needs in-process stability, not cross-process or across-version.
fn content_hash(data: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

/// Emit `a=p,i=<id>` to display an already-uploaded image at the current
/// cursor position. Used when the same image content has been transmitted
/// earlier in the session — saves both the upload bandwidth and a fresh
/// terminal-memory allocation.
fn emit_kitty_redisplay(image_id: u32) -> bool {
    let mut stdout = std::io::stdout();
    if write!(stdout, "\x1b_Ga=p,i={},q=2,C=1\x1b\\\r\n", image_id).is_err() {
        return false;
    }
    stdout.flush().is_ok()
}

/// Emit the full Kitty graphics APC sequence for a base64-encoded image,
/// tagging it with `image_id` so the terminal can replace or evict it later.
///
/// The base64 string is split into `KITTY_CHUNK_SIZE`-byte chunks and each
/// chunk is wrapped in the appropriate APC escape. Everything is written
/// directly to `stdout`.
fn emit_kitty_apc(base64_data: &str, media_type: Option<&str>, image_id: u32) -> bool {
    let fmt = kitty_format(media_type);
    let mut stdout = std::io::stdout();

    // Strip any whitespace/newlines that may have been inserted into the
    // base64 string (the API sometimes line-wraps it).
    let clean: String = base64_data.chars().filter(|c| !c.is_whitespace()).collect();

    let chunks: Vec<&str> = clean
        .as_bytes()
        .chunks(KITTY_CHUNK_SIZE)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect();

    if chunks.is_empty() {
        return false;
    }

    let total = chunks.len();
    for (i, chunk) in chunks.iter().enumerate() {
        let first = i == 0;
        let last = i == total - 1;
        let more = if last { 0u8 } else { 1 };

        let params = if first {
            format!("a=T,f={},i={},m={},q=2,C=1", fmt, image_id, more)
        } else {
            format!("a=T,i={},m={},q=2", image_id, more)
        };

        // Write the APC sequence: ESC _ G <params> ; <base64-chunk> ESC \
        if let Err(err) = write!(stdout, "\x1b_G{};{}\x1b\\", params, chunk) {
            tracing::warn!(error = %err, "failed to write Kitty image escape");
            return false;
        }
    }

    // Move to a new line so subsequent ratatui output begins cleanly.
    if let Err(err) = write!(stdout, "\r\n") {
        tracing::warn!(error = %err, "failed to terminate Kitty image escape");
        return false;
    }
    if let Err(err) = stdout.flush() {
        tracing::warn!(error = %err, "failed to flush Kitty image escape");
        return false;
    }
    true
}

/// Emit an iTerm OSC 1337 inline image sequence.
///
/// Sequence shape:
///   ESC ] 1337 ; File=inline=1;size=<bytes>;width=auto;height=auto : <base64> BEL
fn emit_iterm_osc1337(base64_data: &str, _media_type: Option<&str>) -> bool {
    let mut stdout = std::io::stdout();

    // Strip any whitespace/newlines that may have been inserted into the
    // base64 string.
    let clean: String = base64_data.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.is_empty() {
        return false;
    }

    let approx_size = clean.len() * 3 / 4;
    if let Err(err) = write!(
        stdout,
        "\x1b]1337;File=inline=1;size={};width=auto;height=auto:{}\x07",
        approx_size, clean
    ) {
        tracing::warn!(error = %err, "failed to write iTerm image escape");
        return false;
    }

    // Move to a new line so subsequent ratatui output begins cleanly.
    if let Err(err) = write!(stdout, "\r\n") {
        tracing::warn!(error = %err, "failed to terminate iTerm image escape");
        return false;
    }
    if let Err(err) = stdout.flush() {
        tracing::warn!(error = %err, "failed to flush iTerm image escape");
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Sixel Protocol
// ---------------------------------------------------------------------------

/// Decode base64-encoded image data and convert to Sixel protocol.
///
/// Returns `true` if successful and the Sixel escape sequence was written to stdout.
/// Returns `false` if decoding or conversion fails.
fn emit_sixel(base64_data: &str, _media_type: Option<&str>) -> bool {
    // Decode base64
    let decoded = match decode_base64(base64_data) {
        Ok(bytes) => bytes,
        Err(_) => {
            return false;
        }
    };

    // Decode image data (PNG or JPEG)
    let img_data = match decode_image_data(&decoded) {
        Ok(data) => data,
        Err(_) => {
            return false;
        }
    };

    // Convert to Sixel using icy_sixel
    let sixel_bytes = match convert_to_sixel(&img_data) {
        Ok(data) => data,
        Err(_) => {
            return false;
        }
    };

    // Emit Sixel escape sequence
    emit_sixel_sequence(&sixel_bytes)
}

/// Decode base64 string, stripping whitespace.
fn decode_base64(base64_data: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use base64::Engine;

    // Strip whitespace
    let clean: String = base64_data.chars().filter(|c| !c.is_whitespace()).collect();
    // Decode using base64 crate
    let decoded = base64::engine::general_purpose::STANDARD.decode(&clean)?;
    Ok(decoded)
}

/// Decoded image data with dimensions in RGBA format.
#[derive(Debug)]
struct ImageData {
    pixels: Vec<u8>, // RGBA format, 4 bytes per pixel
    width: u32,
    height: u32,
}

/// Decode PNG or JPEG image data into RGBA pixels.
fn decode_image_data(data: &[u8]) -> Result<ImageData, Box<dyn std::error::Error>> {
    // Try to detect PNG or JPEG by magic bytes and decode accordingly.

    // PNG magic: 89 50 4E 47
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return decode_png(data);
    }

    // JPEG magic: FF D8 FF
    if data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
        return decode_jpeg(data);
    }

    // Try PNG first, then JPEG as fallbacks
    decode_png(data).or_else(|_| decode_jpeg(data))
}

/// Decode PNG image data into RGBA pixels.
///
/// Uses the `image` crate to decode PNG data and convert to RGBA8 format.
/// Returns an error if decoding fails.
fn decode_png(data: &[u8]) -> Result<ImageData, Box<dyn std::error::Error>> {
    use image::ImageReader;
    use std::io::Cursor;

    // Decode the PNG using the image crate with explicit format hint
    let reader = ImageReader::new(Cursor::new(data)).with_guessed_format()?;
    let image = reader
        .decode()
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;

    // Convert to RGBA8 format
    let rgba_image = image.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    let pixels = rgba_image.into_raw();

    Ok(ImageData {
        pixels,
        width,
        height,
    })
}

/// Decode JPEG image data into RGBA pixels.
///
/// Uses the `image` crate to decode JPEG data and convert to RGBA8 format.
/// Returns an error if decoding fails.
fn decode_jpeg(data: &[u8]) -> Result<ImageData, Box<dyn std::error::Error>> {
    use image::ImageReader;
    use std::io::Cursor;

    // Decode the JPEG using the image crate with explicit format hint
    let reader = ImageReader::new(Cursor::new(data)).with_guessed_format()?;
    let image = reader
        .decode()
        .map_err(|e| format!("Failed to decode JPEG: {}", e))?;

    // Convert to RGBA8 format
    let rgba_image = image.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    let pixels = rgba_image.into_raw();

    Ok(ImageData {
        pixels,
        width,
        height,
    })
}

/// Convert RGBA image data to Sixel format using the icy_sixel library.
///
/// The icy_sixel crate provides high-quality color quantization and dithering
/// to convert true-color images down to the 256-color palette supported by Sixel.
fn convert_to_sixel(img_data: &ImageData) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use icy_sixel::encoder::EncodeOptions;

    // Use the simpler sixel_encode function with default options
    let sixel_string = icy_sixel::encoder::sixel_encode(
        &img_data.pixels,
        img_data.width as usize,
        img_data.height as usize,
        &EncodeOptions::default(),
    )?;

    // Convert the Sixel string to bytes for output
    Ok(sixel_string.into_bytes())
}

/// Emit Sixel escape sequence to stdout.
///
/// `icy_sixel::encoder::sixel_encode` already returns the full DCS-wrapped
/// sequence (`ESC P ... ESC \`), so we emit it verbatim.
fn emit_sixel_sequence(sixel_data: &[u8]) -> bool {
    let mut stdout = std::io::stdout();

    // Write sixel data in chunks to respect terminal line limits
    for chunk in sixel_data.chunks(SIXEL_LINE_SIZE) {
        if let Err(err) = stdout.write_all(chunk) {
            tracing::warn!(error = %err, "failed to write Sixel image escape");
            return false;
        }
    }

    // Move to a new line so subsequent output begins cleanly
    if let Err(err) = write!(stdout, "\r\n") {
        tracing::warn!(error = %err, "failed to terminate Sixel image escape");
        return false;
    }
    if let Err(err) = stdout.flush() {
        tracing::warn!(error = %err, "failed to flush Sixel image escape");
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transcript-image cache is process-wide static state. These tests
    /// mutate it, so they must run one at a time even when `cargo test` runs
    /// the test binary multi-threaded. This lock serializes them.
    static CACHE_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    fn intern_same_hash_returns_same_id_and_marks_cached() {
        let _g = CACHE_TEST_LOCK.lock();
        reset_transcript_image_cache();
        let h = content_hash("AAAA");
        let (id1, new1) = intern_transcript_image_id(h);
        let (id2, new2) = intern_transcript_image_id(h);
        assert!(new1, "first call should report new");
        assert!(!new2, "second call should report cached");
        assert_eq!(id1, id2, "same hash must reuse the id");
        assert!(id1 >= FIRST_TRANSCRIPT_IMAGE_ID);
        assert_ne!(id1, MASCOT_IMAGE_ID, "mascot id is reserved");
    }

    #[test]
    fn intern_different_hash_gets_different_id() {
        let _g = CACHE_TEST_LOCK.lock();
        reset_transcript_image_cache();
        let (id_a, _) = intern_transcript_image_id(content_hash("A"));
        let (id_b, _) = intern_transcript_image_id(content_hash("B"));
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn reset_image_cache_lets_id_be_reissued() {
        let _g = CACHE_TEST_LOCK.lock();
        reset_transcript_image_cache();
        let (id1, new1) = intern_transcript_image_id(content_hash("X"));
        assert!(new1);
        reset_transcript_image_cache();
        let (id2, new2) = intern_transcript_image_id(content_hash("X"));
        assert!(new2, "after reset the same hash must be treated as new");
        assert_eq!(id1, id2, "ID counter should restart, reissuing the first id");
    }

    #[test]
    fn cache_evicts_least_recently_used_when_full() {
        let _g = CACHE_TEST_LOCK.lock();
        reset_transcript_image_cache();
        // Fill the cache to capacity with unique hashes.
        for i in 0..MAX_CACHED_TRANSCRIPT_IMAGES as u64 {
            let (_, is_new) = intern_transcript_image_id(i);
            assert!(is_new);
        }
        // Touch hash 0 so it becomes most recently used.
        let _ = intern_transcript_image_id(0);
        // Add one more entry — this must trigger eviction. Hash 1 (now LRU)
        // should be the victim; hash 0 (recently touched) must still be cached.
        let new_hash = MAX_CACHED_TRANSCRIPT_IMAGES as u64 + 1;
        let (_, new_is_new) = intern_transcript_image_id(new_hash);
        assert!(new_is_new);
        // Hash 0 still cached (recently used)
        let (_, hit0) = intern_transcript_image_id(0);
        assert!(!hit0, "recently used entry must survive eviction");
        // Hash 1 was evicted — reinserting reports it as new
        let (_, hit1) = intern_transcript_image_id(1);
        assert!(hit1, "least-recently-used entry must have been evicted");
    }

    /// Test that we can decode a minimal valid PNG image.
    /// This is a 1x1 transparent PNG (smallest possible valid PNG).
    #[test]
    fn test_decode_minimal_png() {
        // Minimal 1x1 transparent PNG created with:
        // echo -ne '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > test.png
        let png_data = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];

        let result = decode_png(&png_data);
        assert!(result.is_ok(), "PNG decoding should succeed");

        let img = result.unwrap();
        assert_eq!(img.width, 1, "PNG width should be 1");
        assert_eq!(img.height, 1, "PNG height should be 1");
        assert_eq!(img.pixels.len(), 4, "RGBA8 1x1 image should have 4 bytes");
    }

    /// Test that decode_image_data correctly identifies and decodes PNG by magic bytes.
    #[test]
    fn test_decode_image_data_detects_png() {
        let png_data = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];

        let result = decode_image_data(&png_data);
        assert!(
            result.is_ok(),
            "decode_image_data should detect and decode PNG"
        );

        let img = result.unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
    }

    /// Test that invalid image data produces an error.
    #[test]
    fn test_decode_invalid_image() {
        let invalid_data = vec![0x00, 0x00, 0x00, 0x00];
        let result = decode_image_data(&invalid_data);
        assert!(
            result.is_err(),
            "Invalid image data should produce an error"
        );
    }

    /// Test that decode_image_data rejects empty data.
    #[test]
    fn test_decode_empty_data() {
        let empty_data = vec![];
        let result = decode_image_data(&empty_data);
        assert!(result.is_err(), "Empty data should produce an error");
    }
}
