//! Decode PNG screenshots / embedded CF checkbox template to grayscale luminance buffers.

use image::ImageReader;

pub fn png_bytes_to_gray_luma(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let img = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("PNG reader: {}", e))?
        .decode()
        .map_err(|e| format!("PNG decode: {}", e))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut gray = Vec::with_capacity((w * h) as usize);
    for pix in rgba.pixels() {
        let [r, g, b, _a] = pix.0;
        let y = (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64).round();
        gray.push(y.clamp(0.0, 255.0) as u8);
    }
    Ok((w, h, gray))
}

/// English CF checkbox template bundled from nodriver `get_cf_template()`.
pub fn embedded_cf_template_gray() -> Result<(u32, u32, Vec<u8>), String> {
    const EMBEDDED_PNG: &[u8] = include_bytes!("cf_template.png");
    png_bytes_to_gray_luma(EMBEDDED_PNG)
}

/// Load grayscale template from `MANGOCODE_CF_TEMPLATE_PATH` if set, else embedded PNG.
pub fn load_cf_checkbox_template_gray() -> Result<(u32, u32, Vec<u8>), String> {
    if let Ok(path) = std::env::var("MANGOCODE_CF_TEMPLATE_PATH") {
        let data =
            std::fs::read(&path).map_err(|e| format!("Failed to read template {}: {}", path, e))?;
        return png_bytes_to_gray_luma(&data);
    }
    embedded_cf_template_gray()
}
