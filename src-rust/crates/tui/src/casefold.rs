#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CaseFoldMatch {
    pub original_start: usize,
    pub original_end: usize,
    pub lower_start: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LowerChar {
    pub ch: char,
    pub original_start: usize,
}

pub(crate) fn case_insensitive_first_match(text: &str, query: &str) -> Option<CaseFoldMatch> {
    case_insensitive_non_overlapping_matches(text, query)
        .into_iter()
        .next()
}

pub(crate) fn case_insensitive_non_overlapping_matches(
    text: &str,
    query: &str,
) -> Vec<CaseFoldMatch> {
    case_insensitive_matches(text, query, false)
}

pub(crate) fn case_insensitive_first_match_from(
    text: &str,
    query: &str,
    min_original_start: usize,
) -> Option<CaseFoldMatch> {
    case_insensitive_matches(text, query, true)
        .into_iter()
        .find(|m| m.original_start >= min_original_start)
}

pub(crate) fn case_insensitive_last_match(text: &str, query: &str) -> Option<CaseFoldMatch> {
    case_insensitive_matches(text, query, true)
        .into_iter()
        .last()
}

pub(crate) fn case_insensitive_last_match_before(
    text: &str,
    query: &str,
    max_original_end: usize,
) -> Option<CaseFoldMatch> {
    case_insensitive_matches(text, query, true)
        .into_iter()
        .take_while(|m| m.original_end <= max_original_end)
        .last()
}

pub(crate) fn original_char_starts_for_range(text: &str, start: usize, end: usize) -> Vec<usize> {
    text.char_indices()
        .filter_map(|(byte, _)| (byte >= start && byte < end).then_some(byte))
        .collect()
}

pub(crate) fn lowercase_chars_with_original_starts(text: &str) -> Vec<LowerChar> {
    let mut chars = Vec::new();
    for (original_start, ch) in text.char_indices() {
        for lower_ch in ch.to_lowercase() {
            chars.push(LowerChar {
                ch: lower_ch,
                original_start,
            });
        }
    }
    chars
}

fn case_insensitive_matches(text: &str, query: &str, overlapping: bool) -> Vec<CaseFoldMatch> {
    let query_lc = query.to_lowercase();
    if query_lc.is_empty() {
        return Vec::new();
    }

    let (lowered, lower_byte_to_original_range) = lowercase_with_original_ranges(text);
    let mut matches = Vec::new();
    let mut cursor = 0usize;

    while cursor <= lowered.len() {
        let Some(rel) = lowered[cursor..].find(query_lc.as_str()) else {
            break;
        };
        let lower_start = cursor + rel;
        let lower_end = lower_start + query_lc.len();

        if let Some(case_match) =
            build_match(text, &lower_byte_to_original_range, lower_start, lower_end)
        {
            matches.push(case_match);
        }

        cursor = if overlapping {
            next_char_boundary_after(&lowered, lower_start)
        } else {
            lower_end
        };
    }

    matches
}

fn lowercase_with_original_ranges(text: &str) -> (String, Vec<(usize, usize)>) {
    let mut lowered = String::with_capacity(text.len());
    let mut lower_byte_to_original_range = Vec::new();

    for (original_start, ch) in text.char_indices() {
        let original_end = original_start + ch.len_utf8();
        for lower_ch in ch.to_lowercase() {
            lowered.push(lower_ch);
            lower_byte_to_original_range
                .extend((0..lower_ch.len_utf8()).map(|_| (original_start, original_end)));
        }
    }

    (lowered, lower_byte_to_original_range)
}

fn build_match(
    text: &str,
    lower_byte_to_original_range: &[(usize, usize)],
    lower_start: usize,
    lower_end: usize,
) -> Option<CaseFoldMatch> {
    if lower_start >= lower_byte_to_original_range.len()
        || lower_end <= lower_start
        || lower_end > lower_byte_to_original_range.len()
    {
        return None;
    }

    let original_start = lower_byte_to_original_range[lower_start].0;
    let original_end = lower_byte_to_original_range[lower_end - 1].1;
    if original_start < original_end
        && text.is_char_boundary(original_start)
        && text.is_char_boundary(original_end)
    {
        Some(CaseFoldMatch {
            original_start,
            original_end,
            lower_start,
        })
    } else {
        None
    }
}

fn next_char_boundary_after(text: &str, byte: usize) -> usize {
    let mut next = byte.saturating_add(1);
    while next < text.len() && !text.is_char_boundary(next) {
        next += 1;
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_match_maps_expanding_lowercase_to_original_range() {
        let m = case_insensitive_first_match("x\u{0130}y", "i\u{0307}").unwrap();

        assert_eq!(m.original_start, 1);
        assert_eq!(m.original_end, 3);
        assert_eq!(
            original_char_starts_for_range("x\u{0130}y", m.original_start, m.original_end),
            vec![1]
        );
    }

    #[test]
    fn overlapping_search_finds_next_original_match() {
        let text = "aaa";

        let first = case_insensitive_first_match_from(text, "aa", 0).unwrap();
        let second = case_insensitive_first_match_from(text, "aa", 1).unwrap();

        assert_eq!(first.original_start, 0);
        assert_eq!(second.original_start, 1);
    }

    #[test]
    fn last_match_before_excludes_match_containing_cursor() {
        let text = "aa bb aa";

        let m = case_insensitive_last_match_before(text, "aa", 7).unwrap();

        assert_eq!(m.original_start, 0);
    }
}
