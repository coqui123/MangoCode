// input.rs — Slash command helpers and input mode types.

/// Check whether a string looks like a slash command (e.g. "/help").
pub fn is_slash_command(input: &str) -> bool {
    input.starts_with('/') && !input.starts_with("//")
}

/// Parse a slash command into `(command_name, args)`.
/// Returns `("", "")` if the input is not a slash command.
pub fn parse_slash_command(input: &str) -> (&str, &str) {
    if !is_slash_command(input) {
        return ("", "");
    }
    let without_slash = &input[1..];
    if let Some((space_idx, whitespace)) = without_slash
        .char_indices()
        .find(|(_, ch)| ch.is_whitespace())
    {
        (
            &without_slash[..space_idx],
            without_slash[space_idx + whitespace.len_utf8()..].trim(),
        )
    } else {
        (without_slash, "")
    }
}

/// Return true for slash commands whose arg-bearing form must bypass the TUI
/// overlay/toggle layer so the CLI command can validate and apply the args.
pub fn slash_command_with_args_skips_tui_intercept(cmd_name: &str, cmd_args: &str) -> bool {
    let cmd_name = cmd_name.trim().trim_start_matches('/').to_lowercase();
    !cmd_args.trim().is_empty()
        && matches!(
            cmd_name.as_str(),
            "config"
                | "settings"
                | "model"
                | "theme"
                | "output-style"
                | "mcp"
                | "memory"
                | "hooks"
                | "agents"
                | "diff"
                | "review"
                | "search"
                | "find"
                | "feedback"
                | "survey"
                | "bug"
                | "report"
                | "plan"
                | "resume"
                | "session"
                | "rewind"
                | "rename"
                | "export"
                | "help"
                | "cost"
                | "copy"
                | "vim"
                | "vi"
                | "voice"
                | "fast"
                | "speed"
                | "effort"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_command_detection() {
        assert!(is_slash_command("/help"));
        assert!(is_slash_command("/compact args"));
        assert!(!is_slash_command("//comment"));
        assert!(!is_slash_command("hello"));
        assert!(!is_slash_command(""));
    }

    #[test]
    fn parse_no_args() {
        let (cmd, args) = parse_slash_command("/help");
        assert_eq!(cmd, "help");
        assert_eq!(args, "");
    }

    #[test]
    fn parse_with_args() {
        let (cmd, args) = parse_slash_command("/compact  --force ");
        assert_eq!(cmd, "compact");
        assert_eq!(args, "--force");
    }

    #[test]
    fn parse_with_non_space_whitespace_args() {
        let (cmd, args) = parse_slash_command("/effort\thigh");
        assert_eq!(cmd, "effort");
        assert_eq!(args, "high");
    }

    #[test]
    fn parse_with_unicode_whitespace_args() {
        let (cmd, args) = parse_slash_command("/help\u{2003}teleport");
        assert_eq!(cmd, "help");
        assert_eq!(args, "teleport");
    }

    #[test]
    fn parse_non_slash() {
        let (cmd, args) = parse_slash_command("hello world");
        assert_eq!(cmd, "");
        assert_eq!(args, "");
    }

    #[test]
    fn arg_bearing_commands_skip_tui_intercepts() {
        for cmd in [
            "model",
            "config",
            "settings",
            "theme",
            "output-style",
            "mcp",
            "memory",
            "hooks",
            "agents",
            "diff",
            "review",
            "search",
            "find",
            "feedback",
            "survey",
            "bug",
            "report",
            "plan",
            "resume",
            "session",
            "rewind",
            "rename",
            "export",
            "help",
            "cost",
            "copy",
            "vim",
            "vi",
            "voice",
            "fast",
            "speed",
            "effort",
        ] {
            assert!(slash_command_with_args_skips_tui_intercept(cmd, "value"));
        }
        assert!(slash_command_with_args_skips_tui_intercept(
            "/MODEL", "value"
        ));
        assert!(slash_command_with_args_skips_tui_intercept("Find", "value"));
        assert!(!slash_command_with_args_skips_tui_intercept("effort", ""));
        assert!(!slash_command_with_args_skips_tui_intercept(
            "effort", "   "
        ));
        assert!(!slash_command_with_args_skips_tui_intercept(
            "context",
            "unexpected"
        ));
    }
}
