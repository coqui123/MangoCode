//! Normalized correlation matching similar to OpenCV TM_CCOEFF_NORMED (sliding-window).

#[derive(Clone, Copy, Debug)]
pub struct PeakMatch {
    pub center_px: (i32, i32),
    pub score: f64,
}

#[derive(Debug)]
pub enum TemplateMatchErr {
    Size,
    NoMatchAboveThreshold,
}

fn flat_index(w: usize, x: usize, y: usize) -> usize {
    y * w + x
}

/// Gray `hay` / `tmpl` row-major, dimensions `hay_w*hay`.
pub fn best_tm_ccoeff_normed(
    hay_w: usize,
    hay_h: usize,
    hay: &[u8],
    tmpl_w: usize,
    tmpl_h: usize,
    tmpl: &[u8],
    min_score: f64,
) -> Result<PeakMatch, TemplateMatchErr> {
    if tmpl_w > hay_w
        || tmpl_h > hay_h
        || hay.len() != hay_w * hay_h
        || tmpl.len() != tmpl_w * tmpl_h
    {
        return Err(TemplateMatchErr::Size);
    }

    let n = (tmpl_w * tmpl_h) as f64;
    let mut tmpl_sum = 0.0_f64;
    let mut tmpl_sum_sq = 0.0_f64;
    for &t in tmpl.iter() {
        let tv = t as f64;
        tmpl_sum += tv;
        tmpl_sum_sq += tv * tv;
    }

    let mut best = -2.0_f64;
    let mut best_ix = (0usize, 0usize);
    let denom_t = (n * tmpl_sum_sq - tmpl_sum * tmpl_sum).sqrt();
    if !denom_t.is_finite() || denom_t < 1e-9 {
        return Err(TemplateMatchErr::NoMatchAboveThreshold);
    }

    let y_max = hay_h - tmpl_h;
    let x_max = hay_w - tmpl_w;
    for y in 0..=y_max {
        for x in 0..=x_max {
            let mut win_sum = 0.0_f64;
            let mut win_sum_sq = 0.0_f64;
            let mut cross = 0.0_f64;
            for ty in 0..tmpl_h {
                let row = flat_index(hay_w, x, y + ty);
                for tx in 0..tmpl_w {
                    let p = hay[row + tx] as f64;
                    let tv = tmpl[ty * tmpl_w + tx] as f64;
                    win_sum += p;
                    win_sum_sq += p * p;
                    cross += p * tv;
                }
            }
            let denom_w = (n * win_sum_sq - win_sum * win_sum).sqrt();
            if !denom_w.is_finite() || denom_w < 1e-9 {
                continue;
            }
            let score = (n * cross - win_sum * tmpl_sum) / (denom_w * denom_t);
            if score > best {
                best = score;
                best_ix = (x, y);
            }
        }
    }
    if best < min_score {
        return Err(TemplateMatchErr::NoMatchAboveThreshold);
    }
    let cx = best_ix.0 + tmpl_w / 2;
    let cy = best_ix.1 + tmpl_h / 2;
    Ok(PeakMatch {
        center_px: (cx as i32, cy as i32),
        score: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_patch_scores_one() {
        let p: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let r = best_tm_ccoeff_normed(3, 3, &p, 3, 3, &p, 0.99).expect("match");
        assert!(r.score > 0.999);
        assert_eq!(r.center_px, (1, 1));
    }
}
