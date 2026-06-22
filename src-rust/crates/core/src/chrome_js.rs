//! Chrome DevTools `Runtime.evaluate` helpers (browser-harness `helpers.js()` semantics).
//! Shared by the `/chrome` command and the Browser tool.

use serde_json::Value;

/// True when the expression has a top-level `return` (not inside strings/comments),
/// matching `browser_harness.helpers._has_return_statement`.
pub fn has_top_level_return(expression: &str) -> bool {
    let bytes = expression.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Code,
        LineComment,
        BlockComment,
        String { quote: u8 },
    }
    let mut state = State::Code;

    while i < n {
        let ch = bytes[i];
        let nxt = bytes.get(i + 1).copied().unwrap_or(b'\0');
        match state {
            State::Code => {
                match ch {
                    b'\'' | b'"' | b'`' => {
                        state = State::String { quote: ch };
                        i += 1;
                        continue;
                    }
                    b'/' if nxt == b'/' => {
                        state = State::LineComment;
                        i += 2;
                        continue;
                    }
                    b'/' if nxt == b'*' => {
                        state = State::BlockComment;
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
                if i + 6 <= n
                    && &expression[i..i + 6] == "return"
                    && !is_word_char(expression.as_bytes().get(i.wrapping_sub(1)).copied())
                    && !is_word_char(expression.as_bytes().get(i + 6).copied())
                {
                    return true;
                }
                i += 1;
            }
            State::LineComment => {
                if ch == b'\n' {
                    state = State::Code;
                }
                i += 1;
            }
            State::BlockComment => {
                if ch == b'*' && nxt == b'/' {
                    state = State::Code;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            State::String { quote } => {
                if ch == b'\\' {
                    i += 2.min(n.saturating_sub(i));
                    continue;
                }
                if ch == quote {
                    state = State::Code;
                }
                i += 1;
            }
        }
    }
    false
}

fn is_word_char(b: Option<u8>) -> bool {
    matches!(b, Some(c) if c.is_ascii_alphanumeric() || c == b'_')
}

fn js_snippet(expression: &str, limit: usize) -> String {
    let snippet = expression.trim().replace('\n', "\\n");
    if snippet.len() > limit {
        format!("{}...", &snippet[..limit.saturating_sub(3)])
    } else {
        snippet
    }
}

/// Wrap user expression for CDP `Runtime.evaluate` (IIFE when top-level `return`).
pub fn prepare_eval_expression(expression: &str) -> String {
    let trimmed = expression.trim();
    if has_top_level_return(expression) && !trimmed.starts_with('(') {
        return format!("(function(){{{}}})()", expression);
    }
    expression.to_string()
}

fn js_exception_description(result: &Value, details: Option<&Value>) -> String {
    if let Some(desc) = result.get("description").and_then(|v| v.as_str()) {
        if !desc.is_empty() {
            return desc.to_string();
        }
    }
    if let Some(details) = details {
        if let Some(exc) = details.get("exception") {
            if let Some(d) = exc.get("description").and_then(|v| v.as_str()) {
                return d.to_string();
            }
            if let Some(v) = exc.get("value") {
                return v.to_string();
            }
            if let Some(c) = exc.get("className").and_then(|v| v.as_str()) {
                return c.to_string();
            }
        }
        if let Some(t) = details.get("text").and_then(|v| v.as_str()) {
            return t.to_string();
        }
    }
    "JavaScript evaluation failed".to_string()
}

fn decode_unserializable(s: &str) -> String {
    match s {
        "NaN" => "NaN".to_string(),
        "Infinity" => "Infinity".to_string(),
        "-Infinity" => "-Infinity".to_string(),
        "-0" => "-0".to_string(),
        _ if s.ends_with('n') && s.len() > 1 => s[..s.len() - 1].to_string(),
        _ => s.to_string(),
    }
}

/// Turn CDP `Runtime.evaluate` **result** object (`resp["result"]`) into a display string or error.
pub fn format_evaluate_response(full_response: &Value, expression: &str) -> Result<String, String> {
    let result = full_response.get("result").ok_or_else(|| {
        full_response
            .to_string()
            .chars()
            .take(200)
            .collect::<String>()
    })?;
    let details = full_response.get("exceptionDetails");
    if details.is_some() || result.get("subtype").and_then(|s| s.as_str()) == Some("error") {
        let desc = js_exception_description(result, details);
        let loc = if let Some(d) = details {
            match (d.get("lineNumber"), d.get("columnNumber")) {
                (Some(ln), Some(cn)) if ln.is_number() && cn.is_number() => format!(
                    " at line {}, column {}",
                    ln.as_u64().unwrap_or(0),
                    cn.as_u64().unwrap_or(0)
                ),
                _ => String::new(),
            }
        } else {
            String::new()
        };
        return Err(format!(
            "JavaScript evaluation failed{}: {}; expression: {}",
            loc,
            desc,
            js_snippet(expression, 160)
        ));
    }

    if let Some(v) = result.get("value") {
        return Ok(match v {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });
    }

    if let Some(u) = result.get("unserializableValue").and_then(|v| v.as_str()) {
        return Ok(decode_unserializable(u));
    }

    if let Some(desc) = result.get("description").and_then(|v| v.as_str()) {
        if !desc.is_empty() {
            return Ok(desc.to_string());
        }
    }

    Ok(result.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn simple_expression_no_wrap() {
        assert_eq!(prepare_eval_expression("document.title"), "document.title");
    }

    #[test]
    fn return_statement_wrapped() {
        let out = prepare_eval_expression("const x = 1; return x");
        assert_eq!(out, "(function(){const x = 1; return x})()");
    }

    #[test]
    fn iife_with_return_not_double_wrapped() {
        let e = "(function(){ return document.title; })()";
        assert_eq!(prepare_eval_expression(e), e);
    }

    #[test]
    fn return_in_string_not_trigger() {
        let e = "document.body.innerText.includes(\"return \")";
        assert!(!has_top_level_return(e));
    }

    #[test]
    fn return_in_comment_not_trigger() {
        let e = "// return comment\n1 + 1";
        assert!(!has_top_level_return(e));
    }

    #[test]
    fn top_level_return_with_whitespace_wrapped() {
        let e = "return\t1";
        let out = prepare_eval_expression(e);
        assert_eq!(out, format!("(function(){{{}}})()", e));
    }

    #[test]
    fn syntax_error_maps_to_err() {
        let full = json!({
            "result": {
                "type": "object",
                "subtype": "error",
                "description": "SyntaxError: Invalid or unexpected token"
            },
            "exceptionDetails": {
                "text": "Uncaught",
                "lineNumber": 1,
                "columnNumber": 12
            }
        });
        assert!(format_evaluate_response(&full, "bad").is_err());
    }

    #[test]
    fn unserializable_nan() {
        let full = json!({
            "result": {
                "type": "number",
                "unserializableValue": "NaN",
                "description": "NaN"
            }
        });
        let s = format_evaluate_response(&full, "1").unwrap();
        assert_eq!(s, "NaN");
    }
}
