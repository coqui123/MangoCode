//! Fuzzy string matching via edit-distance algorithms.
//!
//! Used to rank "did you mean …?" suggestions for mistyped tool names.

/// Classic Levenshtein edit distance: minimum single-character insertions,
/// deletions, or substitutions to transform `a` into `b`.
///
/// Operates on Unicode scalar values (`char`s), not bytes.  Uses a two-row
/// rolling buffer for *O(min(n, m))* extra space.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    // Ensure b_chars is the shorter side so the rolling buffer is smaller.
    let (a_chars, b_chars) = if a_chars.len() < b_chars.len() {
        (b_chars, a_chars)
    } else {
        (a_chars, b_chars)
    };

    let n = a_chars.len();
    let m = b_chars.len();

    if m == 0 {
        return n;
    }

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Optimal String Alignment distance (restricted Damerau–Levenshtein).
///
/// Like [`levenshtein`] but also counts a transposition of two **adjacent**
/// characters as a single edit (`"ca"` → `"ac"` = 1).
///
/// Operates on Unicode scalar values (`char`s), not bytes.  Uses a full
/// matrix because the transposition recurrence needs `d[i-2][j-2]`.
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let n = a_chars.len();
    let m = b_chars.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    // Full (n+1) × (m+1) matrix.
    let mut d = vec![vec![0usize; m + 1]; n + 1];

    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }

    d[n][m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        assert_eq!(levenshtein("hello", "hello"), 0);
        assert_eq!(damerau_levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn both_empty() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(damerau_levenshtein("", ""), 0);
    }

    #[test]
    fn one_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(damerau_levenshtein("", "abc"), 3);
        assert_eq!(damerau_levenshtein("abc", ""), 3);
    }

    #[test]
    fn pure_insertion() {
        assert_eq!(levenshtein("abc", "abcdef"), 3);
        assert_eq!(damerau_levenshtein("abc", "abcdef"), 3);
    }

    #[test]
    fn pure_deletion() {
        assert_eq!(levenshtein("abcdef", "abc"), 3);
        assert_eq!(damerau_levenshtein("abcdef", "abc"), 3);
    }

    #[test]
    fn pure_substitution() {
        assert_eq!(levenshtein("abc", "xyz"), 3);
        assert_eq!(damerau_levenshtein("abc", "xyz"), 3);
    }

    #[test]
    fn kitten_sitting() {
        // kitten -> sitten (s/k/s) -> sittin (s/e/i) -> sitting (ins g) = 3
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(damerau_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn flaw_lawn() {
        // flaw -> law (del f) -> lawn (ins n) = 2
        assert_eq!(levenshtein("flaw", "lawn"), 2);
        assert_eq!(damerau_levenshtein("flaw", "lawn"), 2);
    }

    #[test]
    fn transposition_damerau_less_than_levenshtein() {
        // Adjacent transposition: damerau counts it as 1, levenshtein as 2.
        assert_eq!(levenshtein("ca", "ac"), 2);
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);

        // Transpose last two characters of a longer string.
        assert_eq!(levenshtein("abcdef", "abcdfe"), 2);
        assert_eq!(damerau_levenshtein("abcdef", "abcdfe"), 1);
    }

    #[test]
    fn symmetry() {
        assert_eq!(levenshtein("abc", "ca"), levenshtein("ca", "abc"));
        assert_eq!(
            damerau_levenshtein("abc", "ca"),
            damerau_levenshtein("ca", "abc")
        );
        assert_eq!(
            levenshtein("kitten", "sitting"),
            levenshtein("sitting", "kitten")
        );
        assert_eq!(
            damerau_levenshtein("kitten", "sitting"),
            damerau_levenshtein("sitting", "kitten")
        );
    }

    #[test]
    fn unicode_multibyte() {
        // 'é' is a single Unicode scalar value — counts as one char.
        // "café" vs "cafe": one substitution (é -> e).
        assert_eq!(levenshtein("café", "cafe"), 1);
        assert_eq!(damerau_levenshtein("café", "cafe"), 1);

        // CJK: removing one character.
        assert_eq!(levenshtein("日本語", "日本"), 1);
        assert_eq!(damerau_levenshtein("日本語", "日本"), 1);
    }
}
