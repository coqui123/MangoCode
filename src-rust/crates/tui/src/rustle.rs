//! Mango mascot rendering for ratatui.
//!
//! Rasterizes the mangoMan.svg using `resvg`, converts the pixmap to Unicode
//! block-art for high-quality terminal rendering.
//!
//! The active text fallback uses Braille patterns (2x4 dots per cell), which
//! provides much higher apparent resolution than half-block or quadrant glyphs.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

/// The pose / expression of the mango mascot.
/// Kept for API compatibility; the SVG rendering is static.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustlePose {
    Default,
    ArmsUp,
    LookLeft,
    LookRight,
}

/// Embedded SVG bytes from the repo root.
const SVG_DATA: &[u8] = include_bytes!("../../../mangoMan.svg");

/// Background color (warm dark brown #1A140F) for transparent pixels.
const BG_R: u8 = 26;
const BG_G: u8 = 20;
const BG_B: u8 = 15;

/// Foreground threshold against the warm background used by the Braille fallback.
/// Higher values shrink thin details; lower values include more anti-aliased edge pixels.
const BRAILLE_FG_THRESHOLD_SQ: i32 = 64;

/// Default raster target widths for inline-image protocols.
/// Higher widths preserve more SVG detail at the cost of larger escape payloads.
const INLINE_TARGET_W_SIXEL: u32 = 220;
const INLINE_TARGET_W_KITTY: u32 = 260;
const INLINE_TARGET_W_ITERM: u32 = 260;

/// Horizontal stretch factor used for text-cell rendering fallback.
/// Many Windows terminal fonts are effectively taller than the ideal 2:1 cell
/// model, so a wider fit improves mascot recognizability.
const FALLBACK_CELL_ASPECT_X: f32 = 3.0;

/// Parsed SVG tree, cached once.
static SVG_TREE: Lazy<Option<resvg::usvg::Tree>> =
    Lazy::new(|| resvg::usvg::Tree::from_data(SVG_DATA, &resvg::usvg::Options::default()).ok());

/// Cached rendered result: (max_rows, lines).
static CACHED_RENDER: Lazy<Mutex<(u16, Vec<Line<'static>>)>> =
    Lazy::new(|| Mutex::new((0, Vec::new())));

#[derive(Debug, Clone, Copy)]
struct SvgCropRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// Tight SVG content bounds in SVG coordinate space.
/// Computed once from alpha coverage so transparent margins do not shrink
/// the mascot during fallback rasterization.
static SVG_CONTENT_BOUNDS: Lazy<Option<SvgCropRect>> = Lazy::new(|| {
    let tree = SVG_TREE.as_ref()?;
    detect_svg_content_bounds(tree)
});

/// Quadrant block characters indexed by 4-bit pattern.
/// Bit layout: bit0=top-left, bit1=top-right, bit2=bottom-left, bit3=bottom-right.
/// A set bit means that quadrant uses the foreground color.
const QUADRANT_CHARS: [char; 16] = [
    ' ', // 0b0000 - empty (all bg)
    '▘', // 0b0001 - TL
    '▝', // 0b0010 - TR
    '▀', // 0b0011 - TL+TR (upper half)
    '▖', // 0b0100 - BL
    '▌', // 0b0101 - TL+BL (left half)
    '▞', // 0b0110 - TR+BL (diagonal)
    '▛', // 0b0111 - TL+TR+BL
    '▗', // 0b1000 - BR
    '▚', // 0b1001 - TL+BR (diagonal)
    '▐', // 0b1010 - TR+BR (right half)
    '▜', // 0b1011 - TL+TR+BR
    '▄', // 0b1100 - BL+BR (lower half)
    '▙', // 0b1101 - TL+BL+BR
    '▟', // 0b1110 - TR+BL+BR
    '█', // 0b1111 - full block (all fg)
];

fn detect_svg_content_bounds(tree: &resvg::usvg::Tree) -> Option<SvgCropRect> {
    let svg_size = tree.size();
    let svg_w = svg_size.width();
    let svg_h = svg_size.height();
    let svg_aspect = svg_h / svg_w;

    let probe_h: u32 = 768;
    let probe_w = ((probe_h as f32 / svg_aspect).round() as u32).max(64);

    // Keep probe transparent so alpha gives us a reliable content mask.
    let mut probe = resvg::tiny_skia::Pixmap::new(probe_w, probe_h)?;
    let sx = probe_w as f32 / svg_w;
    let sy = probe_h as f32 / svg_h;
    resvg::render(
        tree,
        resvg::tiny_skia::Transform::from_scale(sx, sy),
        &mut probe.as_mut(),
    );

    let data = probe.data();
    let mut min_x = probe_w;
    let mut min_y = probe_h;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..probe_h {
        let row_off = (y * probe_w * 4) as usize;
        for x in 0..probe_w {
            let a = data[row_off + (x * 4) as usize + 3];
            if a > 8 {
                found = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if !found {
        return None;
    }

    let pad = 2u32;
    min_x = min_x.saturating_sub(pad);
    min_y = min_y.saturating_sub(pad);
    max_x = (max_x + pad).min(probe_w.saturating_sub(1));
    max_y = (max_y + pad).min(probe_h.saturating_sub(1));

    let to_svg_x = svg_w / probe_w as f32;
    let to_svg_y = svg_h / probe_h as f32;

    Some(SvgCropRect {
        x: min_x as f32 * to_svg_x,
        y: min_y as f32 * to_svg_y,
        w: ((max_x - min_x + 1) as f32 * to_svg_x).max(1.0),
        h: ((max_y - min_y + 1) as f32 * to_svg_y).max(1.0),
    })
}

fn render_svg_cropped_to_pixmap(
    tree: &resvg::usvg::Tree,
    crop: SvgCropRect,
    target_w: u32,
    target_h: u32,
    fill_bg: bool,
) -> Option<resvg::tiny_skia::Pixmap> {
    let mut pixmap = resvg::tiny_skia::Pixmap::new(target_w, target_h)?;
    if fill_bg {
        pixmap.fill(resvg::tiny_skia::Color::from_rgba8(BG_R, BG_G, BG_B, 255));
    }

    let sx = target_w as f32 / crop.w;
    let sy = target_h as f32 / crop.h;
    let tx = -crop.x * sx;
    let ty = -crop.y * sy;

    resvg::render(
        tree,
        resvg::tiny_skia::Transform::from_row(sx, 0.0, 0.0, sy, tx, ty),
        &mut pixmap.as_mut(),
    );

    Some(pixmap)
}

fn rasterize_svg_rgba_for_inline(target_w: u32) -> Option<(Vec<u8>, u32, u32)> {
    let tree = SVG_TREE.as_ref()?;
    let svg_size = tree.size();
    let crop = SVG_CONTENT_BOUNDS.unwrap_or(SvgCropRect {
        x: 0.0,
        y: 0.0,
        w: svg_size.width(),
        h: svg_size.height(),
    });
    let aspect = crop.h / crop.w;
    let target_h = (target_w as f32 * aspect).round() as u32;

    // Keep inline protocols transparent so mascot overlays don't paint a full
    // opaque rectangle over the TUI during resize/redraw.
    let pixmap = render_svg_cropped_to_pixmap(tree, crop, target_w, target_h, false)?;
    Some((pixmap.take(), target_w, target_h))
}

fn inline_target_width(default_w: u32) -> u32 {
    std::env::var("MANGOCODE_MASCOT_INLINE_WIDTH")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map(|w| w.clamp(96, 2048))
        .unwrap_or(default_w)
}

/// Render the mango mascot sized to fit `max_rows` terminal rows.
/// Returns the rendered lines (content + 1 trailing blank).
/// Caches the result so repeated calls at the same size are free.
pub fn rustle_lines_sized(max_rows: u16) -> Vec<Line<'static>> {
    let max_rows = max_rows.max(2);

    {
        let cache = CACHED_RENDER.lock();
        if cache.0 == max_rows {
            return cache.1.clone();
        }
    }

    let lines = match std::env::var("MANGOCODE_MASCOT_RENDERER").ok().as_deref() {
        Some("quadrant") => render_svg_quadrants(max_rows),
        _ => render_svg_braille(max_rows),
    };

    {
        let mut cache = CACHED_RENDER.lock();
        *cache = (max_rows, lines.clone());
    }

    lines
}

/// API-compatible wrapper.
pub fn rustle_lines(_pose: &RustlePose) -> Vec<Line<'static>> {
    rustle_lines_sized(5)
}

/// Rasterize the SVG and convert to quadrant-block terminal lines.
///
/// Each terminal cell encodes a 2×2 pixel grid using quadrant block characters.
/// The rasterizer produces a pixel grid of (cols*2) × (rows*2) pixels, then
/// for each 2×2 block picks the best quadrant character + fg/bg color pair.
fn render_svg_quadrants(max_rows: u16) -> Vec<Line<'static>> {
    let tree = match SVG_TREE.as_ref() {
        Some(t) => t,
        None => return fallback_lines(),
    };

    let svg_size = tree.size();
    let svg_w = svg_size.width();
    let svg_h = svg_size.height();
    let crop = SVG_CONTENT_BOUNDS.unwrap_or(SvgCropRect {
        x: 0.0,
        y: 0.0,
        w: svg_w,
        h: svg_h,
    });
    let aspect = crop.h / crop.w;

    let content_rows = max_rows.saturating_sub(1).max(2) as u32;

    // Each terminal row = 2 pixel rows, each terminal col = 2 pixel cols (quadrant).
    // Terminal chars are ~2× taller than wide, so we apply a 2:1 aspect correction.
    let pixel_h = content_rows * 2;
    let pixel_w = ((pixel_h as f32 / aspect) * FALLBACK_CELL_ASPECT_X).round() as u32;
    let pixel_w = pixel_w.max(4) & !1; // ensure even
    let pixel_h = pixel_h.max(4) & !1;

    let pixmap = match render_svg_cropped_to_pixmap(tree, crop, pixel_w, pixel_h, true) {
        Some(p) => p,
        None => return fallback_lines(),
    };

    let data = pixmap.data();
    let cols = pixel_w / 2;
    let rows = pixel_h / 2;

    let mut lines: Vec<Line<'static>> = Vec::new();

    for row in 0..rows {
        let mut spans: Vec<Span<'static>> = Vec::new();

        // Track runs of identical (char, fg, bg) for compression.
        let mut run_ch: char = ' ';
        let mut run_fg = (0u8, 0u8, 0u8);
        let mut run_bg = (0u8, 0u8, 0u8);
        let mut run_len = 0usize;

        for col in 0..cols {
            let x = col * 2;
            let y = row * 2;
            let tl = pixel_rgb(data, pixel_w, x, y);
            let tr = pixel_rgb(data, pixel_w, x + 1, y);
            let bl = pixel_rgb(data, pixel_w, x, y + 1);
            let br = pixel_rgb(data, pixel_w, x + 1, y + 1);

            let (ch, fg, bg) = best_quadrant([tl, tr, bl, br]);

            if run_len > 0 && ch == run_ch && fg == run_fg && bg == run_bg {
                run_len += 1;
            } else {
                if run_len > 0 {
                    spans.push(make_span(run_ch, run_fg, run_bg, run_len));
                }
                run_ch = ch;
                run_fg = fg;
                run_bg = bg;
                run_len = 1;
            }
        }
        if run_len > 0 {
            spans.push(make_span(run_ch, run_fg, run_bg, run_len));
        }

        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    lines
}

#[inline]
fn braille_dot_bit(dx: u32, dy: u32) -> u8 {
    match (dx, dy) {
        (0, 0) => 0x01, // dot 1
        (0, 1) => 0x02, // dot 2
        (0, 2) => 0x04, // dot 3
        (0, 3) => 0x40, // dot 7
        (1, 0) => 0x08, // dot 4
        (1, 1) => 0x10, // dot 5
        (1, 2) => 0x20, // dot 6
        (1, 3) => 0x80, // dot 8
        _ => 0,
    }
}

#[inline]
fn is_foreground_pixel((r, g, b): (u8, u8, u8)) -> bool {
    let dr = r as i32 - BG_R as i32;
    let dg = g as i32 - BG_G as i32;
    let db = b as i32 - BG_B as i32;
    (dr * dr + dg * dg + db * db) >= BRAILLE_FG_THRESHOLD_SQ
}

fn make_braille_span(text: String, fg: (u8, u8, u8)) -> Span<'static> {
    Span::styled(
        text,
        Style::default()
            .fg(Color::Rgb(fg.0, fg.1, fg.2))
            .bg(Color::Rgb(BG_R, BG_G, BG_B)),
    )
}

/// High-resolution text fallback using Unicode Braille patterns (2x4 dots/cell).
fn render_svg_braille(max_rows: u16) -> Vec<Line<'static>> {
    let tree = match SVG_TREE.as_ref() {
        Some(t) => t,
        None => return fallback_lines(),
    };

    let svg_size = tree.size();
    let crop = SVG_CONTENT_BOUNDS.unwrap_or(SvgCropRect {
        x: 0.0,
        y: 0.0,
        w: svg_size.width(),
        h: svg_size.height(),
    });
    let aspect = crop.h / crop.w;

    // Reserve one trailing blank line for spacing parity with other renderers.
    let content_rows = max_rows.saturating_sub(1).max(2) as u32;

    // Braille cell = 2x4 dots. With typical terminal cell geometry (~1:2),
    // this approximates square subpixels and gives much finer detail.
    let pixel_h = (content_rows * 4).max(8);
    let pixel_w = ((pixel_h as f32 / aspect).round() as u32).max(4) & !1;

    let pixmap = match render_svg_cropped_to_pixmap(tree, crop, pixel_w, pixel_h, true) {
        Some(p) => p,
        None => return fallback_lines(),
    };

    let data = pixmap.data();
    let cols = pixel_w / 2;
    let rows = pixel_h / 4;
    let mut lines: Vec<Line<'static>> = Vec::new();

    for row in 0..rows {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut run_fg = (BG_R, BG_G, BG_B);
        let mut run_text = String::new();

        for col in 0..cols {
            let x0 = col * 2;
            let y0 = row * 4;

            let mut bits: u8 = 0;
            let mut sum_r: u32 = 0;
            let mut sum_g: u32 = 0;
            let mut sum_b: u32 = 0;
            let mut n: u32 = 0;

            for dy in 0..4 {
                for dx in 0..2 {
                    let px = pixel_rgb(data, pixel_w, x0 + dx, y0 + dy);
                    if is_foreground_pixel(px) {
                        bits |= braille_dot_bit(dx, dy);
                        sum_r += px.0 as u32;
                        sum_g += px.1 as u32;
                        sum_b += px.2 as u32;
                        n += 1;
                    }
                }
            }

            let ch = if bits == 0 {
                ' '
            } else {
                char::from_u32(0x2800 + bits as u32).unwrap_or(' ')
            };

            let fg = if n > 0 {
                ((sum_r / n) as u8, (sum_g / n) as u8, (sum_b / n) as u8)
            } else {
                (BG_R, BG_G, BG_B)
            };

            if run_text.is_empty() {
                run_fg = fg;
                run_text.push(ch);
            } else if fg == run_fg {
                run_text.push(ch);
            } else {
                spans.push(make_braille_span(run_text, run_fg));
                run_text = String::new();
                run_fg = fg;
                run_text.push(ch);
            }
        }

        if !run_text.is_empty() {
            spans.push(make_braille_span(run_text, run_fg));
        }

        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    lines
}

/// For a 2×2 pixel block, find the best quadrant character and fg/bg colors.
///
/// Tries all 16 quadrant patterns. For each, computes the average fg and bg
/// colors and the total squared error. Returns the pattern with minimum error.
fn best_quadrant(px: [(u8, u8, u8); 4]) -> (char, (u8, u8, u8), (u8, u8, u8)) {
    // Preserve true empty background cells as spaces.
    if px
        .iter()
        .all(|&(r, g, b)| r == BG_R && g == BG_G && b == BG_B)
    {
        return (' ', (BG_R, BG_G, BG_B), (BG_R, BG_G, BG_B));
    }

    let mut best_err = u64::MAX;
    let mut best_pat = 15u8;
    let mut best_fg = px[0];
    let mut best_bg = px[0];

    // Skip 0b0000 so non-background regions don't collapse to blank spaces.
    for pat in 1u8..16 {
        let (fg, bg) = avg_for_pattern(&px, pat);
        let err = pattern_error(&px, pat, fg, bg);

        // In ties, prefer patterns with more filled quadrants so flat colors
        // stay visible instead of disappearing into background.
        if err < best_err || (err == best_err && pat.count_ones() > best_pat.count_ones()) {
            best_err = err;
            best_pat = pat;
            best_fg = fg;
            best_bg = bg;
        }
    }

    (QUADRANT_CHARS[best_pat as usize], best_fg, best_bg)
}

/// Compute average fg and bg colors for a given quadrant pattern.
/// Bit i of `pat` → pixel i is foreground; otherwise background.
fn avg_for_pattern(px: &[(u8, u8, u8); 4], pat: u8) -> ((u8, u8, u8), (u8, u8, u8)) {
    let mut fg_r = 0u32;
    let mut fg_g = 0u32;
    let mut fg_b = 0u32;
    let mut fg_n = 0u32;
    let mut bg_r = 0u32;
    let mut bg_g = 0u32;
    let mut bg_b = 0u32;
    let mut bg_n = 0u32;

    for i in 0..4u8 {
        let (r, g, b) = px[i as usize];
        if pat & (1 << i) != 0 {
            fg_r += r as u32;
            fg_g += g as u32;
            fg_b += b as u32;
            fg_n += 1;
        } else {
            bg_r += r as u32;
            bg_g += g as u32;
            bg_b += b as u32;
            bg_n += 1;
        }
    }

    let fg = if fg_n > 0 {
        (
            (fg_r / fg_n) as u8,
            (fg_g / fg_n) as u8,
            (fg_b / fg_n) as u8,
        )
    } else {
        (BG_R, BG_G, BG_B)
    };
    let bg = if bg_n > 0 {
        (
            (bg_r / bg_n) as u8,
            (bg_g / bg_n) as u8,
            (bg_b / bg_n) as u8,
        )
    } else {
        (BG_R, BG_G, BG_B)
    };

    (fg, bg)
}

/// Total squared color error for a pattern + fg/bg assignment.
fn pattern_error(px: &[(u8, u8, u8); 4], pat: u8, fg: (u8, u8, u8), bg: (u8, u8, u8)) -> u64 {
    let mut err = 0u64;
    for i in 0..4u8 {
        let (r, g, b) = px[i as usize];
        let (cr, cg, cb) = if pat & (1 << i) != 0 { fg } else { bg };
        let dr = r as i32 - cr as i32;
        let dg = g as i32 - cg as i32;
        let db = b as i32 - cb as i32;
        err += (dr * dr + dg * dg + db * db) as u64;
    }
    err
}

/// Build a styled Span for `n` repetitions of a quadrant character.
fn make_span(ch: char, fg: (u8, u8, u8), bg: (u8, u8, u8), n: usize) -> Span<'static> {
    let s: String = std::iter::repeat_n(ch, n).collect();
    let style = if ch == '█' {
        Style::default().fg(Color::Rgb(fg.0, fg.1, fg.2))
    } else if ch == ' ' {
        Style::default().bg(Color::Rgb(bg.0, bg.1, bg.2))
    } else {
        Style::default()
            .fg(Color::Rgb(fg.0, fg.1, fg.2))
            .bg(Color::Rgb(bg.0, bg.1, bg.2))
    };
    Span::styled(s, style)
}

/// Extract RGB from RGBA pixel data, compositing alpha over background.
fn pixel_rgb(data: &[u8], width: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let idx = ((y * width + x) * 4) as usize;
    let r = data[idx];
    let g = data[idx + 1];
    let b = data[idx + 2];
    let a = data[idx + 3];

    if a == 255 {
        (r, g, b)
    } else if a == 0 {
        (BG_R, BG_G, BG_B)
    } else {
        let af = a as f32 / 255.0;
        let inv = 1.0 - af;
        (
            (r as f32 * af + BG_R as f32 * inv) as u8,
            (g as f32 * af + BG_G as f32 * inv) as u8,
            (b as f32 * af + BG_B as f32 * inv) as u8,
        )
    }
}

/// Encode the SVG as a Sixel string for high-quality terminal rendering.
/// Returns `None` if the SVG fails to parse/rasterize or Sixel encoding fails.
pub fn encode_svg_as_sixel() -> Option<String> {
    encode_svg_as_sixel_with_width(INLINE_TARGET_W_SIXEL)
}

/// Encode the SVG as a Sixel string using an explicit raster width.
pub fn encode_svg_as_sixel_with_width(target_w: u32) -> Option<String> {
    use icy_sixel::{BackgroundMode, EncodeOptions, PixelAspectRatio, QuantizeMethod, SixelImage};

    // Render at a higher base resolution so the mascot keeps edge detail.
    let target_w = inline_target_width(target_w.clamp(96, 1024));
    let (rgba, target_w, target_h) = rasterize_svg_rgba_for_inline(target_w)?;

    // For flat-color SVG art, disabling diffusion keeps edges much crisper.
    let sixel_opts = EncodeOptions {
        max_colors: 256,
        diffusion: 0.0,
        quantize_method: QuantizeMethod::Wu,
    };

    let sixel_image = SixelImage::from_rgba(rgba, target_w as usize, target_h as usize)
        .with_aspect_ratio(PixelAspectRatio::Square)
        .with_background_mode(BackgroundMode::Transparent);

    // icy_sixel already returns the full DCS-wrapped SIXEL sequence.
    sixel_image.encode_with(&sixel_opts).ok()
}

fn encode_svg_png_base64(target_w: u32) -> Option<(String, usize)> {
    use base64::Engine;
    use image::codecs::png::PngEncoder;
    use image::{ColorType, ImageEncoder};

    let (rgba, width, height) = rasterize_svg_rgba_for_inline(target_w)?;

    let mut png = Vec::new();
    let encoder = PngEncoder::new(&mut png);
    encoder
        .write_image(&rgba, width, height, ColorType::Rgba8.into())
        .ok()?;

    let png_len = png.len();
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    Some((b64, png_len))
}

/// Encode the SVG as a Kitty graphics APC sequence for highest-resolution
/// terminals that support the Kitty protocol.
pub fn encode_svg_as_kitty_apc() -> Option<String> {
    encode_svg_as_kitty_apc_with_width(INLINE_TARGET_W_KITTY)
}

/// Encode the SVG as a Kitty graphics APC sequence using an explicit raster width.
pub fn encode_svg_as_kitty_apc_with_width(target_w: u32) -> Option<String> {
    // Kitty can comfortably handle a larger raster than Sixel.
    let target_w = inline_target_width(target_w.clamp(128, 2048));
    let (b64, _) = encode_svg_png_base64(target_w)?;
    let mut out = String::new();

    const KITTY_CHUNK_SIZE: usize = 4096;
    let total = b64.len().div_ceil(KITTY_CHUNK_SIZE);

    for (i, chunk) in b64.as_bytes().chunks(KITTY_CHUNK_SIZE).enumerate() {
        let more = if i + 1 == total { 0u8 } else { 1u8 };
        let params = if i == 0 {
            format!("a=T,f=100,m={},q=2,C=1", more)
        } else {
            format!("a=T,m={},q=2", more)
        };
        let chunk = std::str::from_utf8(chunk).ok()?;
        out.push_str("\x1b_G");
        out.push_str(&params);
        out.push(';');
        out.push_str(chunk);
        out.push_str("\x1b\\");
    }

    Some(out)
}

/// Encode the SVG as an iTerm/OSC 1337 inline image sequence.
/// Some terminals (and xterm-compatible emulators with image addons) support
/// this protocol even when Kitty/Sixel are unavailable.
pub fn encode_svg_as_iterm_osc1337() -> Option<String> {
    encode_svg_as_iterm_osc1337_with_width(INLINE_TARGET_W_ITERM)
}

/// Encode the SVG as an iTerm/OSC 1337 inline image sequence using an explicit raster width.
pub fn encode_svg_as_iterm_osc1337_with_width(target_w: u32) -> Option<String> {
    let target_w = inline_target_width(target_w.clamp(128, 2048));
    let (b64, png_len) = encode_svg_png_base64(target_w)?;

    Some(format!(
        "\x1b]1337;File=inline=1;size={};width=auto;height=auto:{}\x07",
        png_len, b64
    ))
}

/// Fallback if SVG parsing fails.
fn fallback_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "  🥭 MangoCode  ",
            Style::default().fg(Color::Rgb(255, 176, 32)),
        )),
        Line::from(""),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_rasterizes_at_various_sizes() {
        for max_rows in [5, 8, 12] {
            let lines = rustle_lines_sized(max_rows);
            let content_lines = lines.len().saturating_sub(1);
            eprintln!("max_rows={}: {} content lines", max_rows, content_lines);
            assert!(content_lines > 0, "No content at max_rows={}", max_rows);
            assert!(
                content_lines <= max_rows as usize,
                "Got {} lines for max_rows={}",
                content_lines,
                max_rows
            );
            let width: usize = lines[0]
                .spans
                .iter()
                .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
                .sum();
            eprintln!("  width={} cols, {} spans", width, lines[0].spans.len());
            assert!(width > 2, "Image too narrow at max_rows={}", max_rows);
        }
    }
}
