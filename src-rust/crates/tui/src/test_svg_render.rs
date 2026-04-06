#[test]
fn test_mango_svg_renders() {
    let svg_data = include_bytes!("../../../mangoMan.svg");
    let tree = resvg::usvg::Tree::from_data(svg_data, &resvg::usvg::Options::default());
    assert!(tree.is_ok(), "SVG parsing failed: {:?}", tree.err());
    let tree = tree.expect("embedded rustle SVG literal should parse");
    let size = tree.size();
    eprintln!("SVG size: {}x{}", size.width(), size.height());
    
    let target_w = 16u32;
    let target_h = ((target_w as f32) * size.height() / size.width()).round() as u32;
    let target_h = if target_h % 2 == 1 { target_h + 1 } else { target_h };
    eprintln!("Target: {}x{}", target_w, target_h);
    
    let pixmap = resvg::tiny_skia::Pixmap::new(target_w, target_h);
    assert!(pixmap.is_some(), "Failed to create pixmap");
    let mut pixmap = pixmap.expect("pixmap allocation for SVG render should succeed");
    
    let bg = resvg::tiny_skia::Color::from_rgba8(26, 20, 15, 255);
    pixmap.fill(bg);
    
    let sx = target_w as f32 / size.width();
    let sy = target_h as f32 / size.height();
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(sx, sy), &mut pixmap.as_mut());
    
    let data = pixmap.data();
    // Check that we have non-background pixels
    let non_bg = (0..target_w*target_h).filter(|&i| {
        let idx = (i * 4) as usize;
        data[idx] != 26 || data[idx+1] != 20 || data[idx+2] != 15
    }).count();
    eprintln!("Non-background pixels: {} / {}", non_bg, target_w * target_h);
    assert!(non_bg > 0, "SVG rendered as all-background — no visible content");
}
