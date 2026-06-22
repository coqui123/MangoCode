//! Build script for MangoCode CLI
//!
//! Embeds build-time metadata (version, timestamp, git info) into the binary
//! for display and debugging purposes.

use std::process::Command;

fn main() {
    // Embed build timestamp (RFC 3339 format)
    let now = chrono::Utc::now().to_rfc3339();
    println!("cargo:rustc-env=BUILD_TIME={}", now);

    // Embed short git commit hash
    let commit = get_git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_COMMIT={}", commit);

    // Package/distribution metadata
    println!("cargo:rustc-env=PACKAGE_URL=mangocode-source-snapshot");
    println!("cargo:rustc-env=FEEDBACK_CHANNEL=github");
    println!("cargo:rustc-env=ISSUES_EXPLAINER=Report issues at the MangoCode GitHub repository.");

    // Trigger rebuild if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");

    // Embed the mango mascot as the Windows executable icon.
    #[cfg(windows)]
    embed_windows_icon();
}

/// Rasterize `mangoMan.svg` into a multi-resolution `.ico` and set it as the
/// executable icon via a Windows resource. Best-effort: any failure prints a
/// warning and leaves the binary iconless rather than failing the build.
#[cfg(windows)]
fn embed_windows_icon() {
    use std::path::PathBuf;

    // SVG lives at the repo's src-rust root: crates/cli/ -> ../../mangoMan.svg
    let svg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../mangoMan.svg");
    println!("cargo:rerun-if-changed={}", svg_path.display());

    let ico_path = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("mango.ico");

    if let Err(err) = build_ico(&svg_path, &ico_path) {
        println!("cargo:warning=skipping exe icon: {err}");
        return;
    }

    let mut res = winresource::WindowsResource::new();
    res.set_icon(ico_path.to_str().unwrap());
    if let Err(err) = res.compile() {
        println!("cargo:warning=failed to embed exe icon resource: {err}");
    }
}

/// Render the SVG at several icon sizes and pack them into a single `.ico`.
#[cfg(windows)]
fn build_ico(svg_path: &std::path::Path, ico_path: &std::path::Path) -> Result<(), String> {
    use resvg::{tiny_skia, usvg};

    let data = std::fs::read(svg_path).map_err(|e| format!("read svg: {e}"))?;
    let tree =
        usvg::Tree::from_data(&data, &usvg::Options::default()).map_err(|e| format!("parse svg: {e}"))?;

    let size = tree.size();
    let (svg_w, svg_h) = (size.width(), size.height());
    if !(svg_w > 0.0 && svg_h > 0.0) {
        return Err("svg has zero size".into());
    }

    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);

    // Standard Windows icon resolutions.
    for px in [16u32, 24, 32, 48, 64, 128, 256] {
        let mut pixmap =
            tiny_skia::Pixmap::new(px, px).ok_or_else(|| format!("alloc {px}x{px} pixmap"))?;

        // Fit the SVG into the square canvas, preserving aspect, centered, on a
        // transparent background.
        let scale = (px as f32 / svg_w).min(px as f32 / svg_h);
        let tx = (px as f32 - svg_w * scale) / 2.0;
        let ty = (px as f32 - svg_h * scale) / 2.0;
        let transform = tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, tx, ty);

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let image = ico::IconImage::from_rgba_data(px, px, pixmap.take());
        let entry =
            ico::IconDirEntry::encode(&image).map_err(|e| format!("encode {px}px entry: {e}"))?;
        icon_dir.add_entry(entry);
    }

    let file = std::fs::File::create(ico_path).map_err(|e| format!("create ico: {e}"))?;
    icon_dir.write(file).map_err(|e| format!("write ico: {e}"))?;
    Ok(())
}

/// Get the short git commit hash, or None if git is not available.
fn get_git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;

    if output.status.success() {
        let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !commit.is_empty() {
            return Some(commit);
        }
    }

    None
}
