pub fn split_command_words(command: &str) -> anyhow::Result<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut in_word = false;
    let mut chars = command.chars().peekable();

    while let Some(ch) = chars.next() {
        match quote {
            Some(active_quote) => {
                if ch == active_quote {
                    quote = None;
                } else if active_quote == '"' && ch == '\\' {
                    match chars.peek().copied() {
                        Some(next) if next == '"' || next == '\\' => {
                            current.push(next);
                            let _ = chars.next();
                        }
                        _ => current.push(ch),
                    }
                } else {
                    current.push(ch);
                }
            }
            None if ch.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            None if ch == '\'' || ch == '"' => {
                quote = Some(ch);
                in_word = true;
            }
            None if ch == '\\' => {
                match chars.peek().copied() {
                    Some(next)
                        if next.is_whitespace() || next == '\'' || next == '"' || next == '\\' =>
                    {
                        current.push(next);
                        let _ = chars.next();
                    }
                    _ => current.push(ch),
                }
                in_word = true;
            }
            None => {
                current.push(ch);
                in_word = true;
            }
        }
    }

    if quote.is_some() {
        anyhow::bail!("command has an unterminated quote");
    }
    if in_word {
        words.push(current);
    }

    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_command_words_preserves_quotes_and_windows_paths() -> anyhow::Result<()> {
        let words = split_command_words(
            r#""C:\Program Files\Embed\embed.exe" --model "bge base" "" C:\tmp\plain"#,
        )?;

        assert_eq!(
            words,
            vec![
                r"C:\Program Files\Embed\embed.exe".to_string(),
                "--model".to_string(),
                "bge base".to_string(),
                String::new(),
                r"C:\tmp\plain".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn split_command_words_handles_escaped_spaces_and_quotes() -> anyhow::Result<()> {
        let words = split_command_words(r#"tool one\ two "quoted \"value\"" 'raw value'"#)?;

        assert_eq!(
            words,
            vec![
                "tool".to_string(),
                "one two".to_string(),
                "quoted \"value\"".to_string(),
                "raw value".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn split_command_words_reports_unterminated_quote() {
        let err = match split_command_words(r#"embed "unterminated"#) {
            Ok(words) => panic!("expected unterminated quote error, got {words:?}"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("unterminated quote"));
    }
}
