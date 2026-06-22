//! Human-friendly parsing and formatting for durations and byte sizes.

use std::time::Duration;

/// Parse a compact, possibly compound, human-readable duration string.
pub fn parse_duration(input: &str) -> Result<Duration, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty duration string".into());
    }

    let mut total_nanos: u128 = 0;
    let mut rest = input;
    let mut parsed_any = false;

    while !rest.is_empty() {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }

        let num_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());

        if num_end == 0 {
            return Err(format!("expected a number near '{}'", rest));
        }

        let num_str = &rest[..num_end];
        let value: f64 = num_str
            .parse()
            .map_err(|e| format!("invalid number '{}': {}", num_str, e))?;

        rest = &rest[num_end..];

        let unit_end = rest
            .find(|c: char| c.is_ascii_digit() || c == '.' || c == ' ')
            .unwrap_or(rest.len());

        if unit_end == 0 {
            return Err(format!("missing unit after '{}'", num_str));
        }

        let unit = &rest[..unit_end];
        rest = &rest[unit_end..];

        let nanos_per_unit: f64 = match unit {
            "ns" => 1.0,
            "us" | "\u{00b5}s" => 1_000.0,
            "ms" => 1_000_000.0,
            "s" => 1_000_000_000.0,
            "m" => 60.0 * 1_000_000_000.0,
            "h" => 3_600.0 * 1_000_000_000.0,
            "d" => 86_400.0 * 1_000_000_000.0,
            "w" => 604_800.0 * 1_000_000_000.0,
            _ => return Err(format!("unknown unit '{}'", unit)),
        };

        let component_nanos = value * nanos_per_unit;
        if component_nanos.is_infinite() || component_nanos.is_nan() || component_nanos < 0.0 {
            return Err(format!("overflow computing '{}{}'", num_str, unit));
        }

        total_nanos = total_nanos.saturating_add(component_nanos as u128);
        parsed_any = true;
    }

    if !parsed_any {
        return Err("no duration components found".into());
    }

    let secs = total_nanos / 1_000_000_000;
    let sub_nanos = (total_nanos % 1_000_000_000) as u32;

    if secs > u64::MAX as u128 {
        return Err("duration overflow".into());
    }

    Ok(Duration::new(secs as u64, sub_nanos))
}

/// Format a [`Duration`] as a compact human-readable string.
pub fn format_duration(d: Duration) -> String {
    let total_ms = d.as_millis();

    if total_ms == 0 {
        if d.is_zero() {
            return "0s".into();
        }
        let us = d.as_micros();
        if us > 0 {
            return format!("{}ms", format_decimal(us as f64 / 1000.0));
        }
        return format!("{}ns", d.as_nanos());
    }

    let mut remaining = total_ms;
    let mut parts = Vec::new();

    const WEEK_MS: u128 = 7 * 24 * 3600 * 1000;
    const DAY_MS: u128 = 24 * 3600 * 1000;
    const HOUR_MS: u128 = 3600 * 1000;
    const MIN_MS: u128 = 60 * 1000;
    const SEC_MS: u128 = 1000;

    let units: &[(u128, &str)] = &[
        (WEEK_MS, "w"),
        (DAY_MS, "d"),
        (HOUR_MS, "h"),
        (MIN_MS, "m"),
        (SEC_MS, "s"),
    ];

    for &(divisor, suffix) in units {
        if remaining >= divisor {
            let count = remaining / divisor;
            remaining %= divisor;
            parts.push(format!("{}{}", count, suffix));
        }
    }

    if remaining > 0 {
        parts.push(format!("{}ms", remaining));
    }

    if parts.is_empty() {
        "0s".into()
    } else {
        parts.join("")
    }
}

/// Parse a human-readable byte-size string.
pub fn parse_byte_size(input: &str) -> Result<u64, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty byte-size string".into());
    }

    let num_end = input
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(input.len());

    if num_end == 0 {
        return Err(format!("expected a number at start of '{}'", input));
    }

    let num_str = &input[..num_end];
    let value: f64 = num_str
        .parse()
        .map_err(|e| format!("invalid number '{}': {}", num_str, e))?;

    let suffix = input[num_end..].trim();

    let multiplier: f64 = if suffix.is_empty() {
        1.0
    } else {
        match suffix.to_ascii_lowercase().as_str() {
            "b" => 1.0,
            "kib" => 1024.0,
            "mib" => 1024.0 * 1024.0,
            "gib" => 1024.0 * 1024.0 * 1024.0,
            "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
            "pib" => 1024.0_f64.powi(5),
            "kb" => 1000.0,
            "mb" => 1_000_000.0,
            "gb" => 1_000_000_000.0,
            "tb" => 1_000_000_000_000.0,
            "pb" => 1_000_000_000_000_000.0,
            _ => return Err(format!("unknown byte-size suffix '{}'", suffix)),
        }
    };

    let bytes = value * multiplier;
    if bytes.is_infinite() || bytes.is_nan() || bytes < 0.0 {
        return Err("byte-size overflow".into());
    }
    if bytes > u64::MAX as f64 {
        return Err("byte-size overflow: exceeds u64".into());
    }

    Ok(bytes as u64)
}

/// Format a byte count as a compact binary-unit string.
pub fn format_byte_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    const PIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0;

    let (value, unit) = if bytes == 0 {
        (0.0, "B")
    } else if (bytes as f64) >= PIB {
        (bytes as f64 / PIB, "PiB")
    } else if (bytes as f64) >= TIB {
        (bytes as f64 / TIB, "TiB")
    } else if (bytes as f64) >= GIB {
        (bytes as f64 / GIB, "GiB")
    } else if (bytes as f64) >= MIB {
        (bytes as f64 / MIB, "MiB")
    } else if (bytes as f64) >= KIB {
        (bytes as f64 / KIB, "KiB")
    } else {
        (bytes as f64, "B")
    };

    format!("{} {}", format_decimal(value), unit)
}

fn format_decimal(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v as u64)
    } else {
        let s = format!("{:.2}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn parse_duration_single_units() {
        assert_eq!(parse_duration("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_duration("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_duration("3m").unwrap(), Duration::from_secs(180));
        assert_eq!(parse_duration("1h").unwrap(), Duration::from_secs(3600));
        assert_eq!(parse_duration("1d").unwrap(), Duration::from_secs(86400));
        assert_eq!(parse_duration("1w").unwrap(), Duration::from_secs(604800));
        assert_eq!(parse_duration("100ns").unwrap(), Duration::from_nanos(100));
        assert_eq!(parse_duration("50us").unwrap(), Duration::from_micros(50));
        assert_eq!(
            parse_duration("50\u{00b5}s").unwrap(),
            Duration::from_micros(50)
        );
    }

    #[test]
    fn parse_duration_compound() {
        assert_eq!(parse_duration("1h30m").unwrap(), Duration::from_secs(5400));
        assert_eq!(
            parse_duration("2d12h").unwrap(),
            Duration::from_secs(2 * 86400 + 12 * 3600)
        );
        assert_eq!(parse_duration("1h 30m").unwrap(), Duration::from_secs(5400));
    }

    #[test]
    fn parse_duration_decimal_fractions() {
        assert_eq!(parse_duration("1.5h").unwrap(), Duration::from_secs(5400));
        assert_eq!(parse_duration("0.25s").unwrap(), Duration::from_millis(250));
        assert_eq!(parse_duration("0.5m").unwrap(), Duration::from_secs(30));
    }

    #[test]
    fn parse_duration_zero() {
        assert_eq!(parse_duration("0s").unwrap(), Duration::ZERO);
        assert_eq!(parse_duration("0ms").unwrap(), Duration::ZERO);
    }

    #[test]
    fn parse_duration_errors() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("   ").is_err());
        assert!(parse_duration("42").is_err());
        assert!(parse_duration("5x").is_err());
        assert!(parse_duration("abc").is_err());
    }

    #[test]
    fn format_duration_cases() {
        assert_eq!(format_duration(Duration::ZERO), "0s");
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(5400)), "1h30m");
        assert_eq!(format_duration(Duration::from_secs(90061)), "1d1h1m1s");
        assert_eq!(format_duration(Duration::from_secs(604800)), "1w");
        assert_eq!(format_duration(Duration::from_secs(7)), "7s");
    }

    #[test]
    fn format_duration_roundtrip() {
        for input in &["1h30m", "2s", "500ms", "1w", "1d1h1m1s"] {
            let d = parse_duration(input).unwrap();
            assert_eq!(format_duration(d), *input);
        }
    }

    #[test]
    fn parse_byte_size_basic() {
        assert_eq!(parse_byte_size("0").unwrap(), 0);
        assert_eq!(parse_byte_size("1024").unwrap(), 1024);
        assert_eq!(parse_byte_size("512B").unwrap(), 512);
        assert_eq!(parse_byte_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_byte_size("1MiB").unwrap(), 1048576);
        assert_eq!(parse_byte_size("1GiB").unwrap(), 1073741824);
    }

    #[test]
    fn parse_byte_size_decimal_units() {
        assert_eq!(parse_byte_size("1KB").unwrap(), 1000);
        assert_eq!(parse_byte_size("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_byte_size("1GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_byte_size("1TB").unwrap(), 1_000_000_000_000);
    }

    #[test]
    fn parse_byte_size_fractions() {
        assert_eq!(parse_byte_size("1.5KiB").unwrap(), 1536);
        assert_eq!(parse_byte_size("2.5MiB").unwrap(), 2621440);
        assert_eq!(parse_byte_size("10 GiB").unwrap(), 10 * 1073741824);
    }

    #[test]
    fn parse_byte_size_case_insensitive() {
        assert_eq!(parse_byte_size("1kib").unwrap(), 1024);
        assert_eq!(parse_byte_size("1KIB").unwrap(), 1024);
        assert_eq!(parse_byte_size("1Kib").unwrap(), 1024);
    }

    #[test]
    fn parse_byte_size_errors() {
        assert!(parse_byte_size("").is_err());
        assert!(parse_byte_size("abc").is_err());
        assert!(parse_byte_size("5XB").is_err());
    }

    #[test]
    fn format_byte_size_cases() {
        assert_eq!(format_byte_size(0), "0 B");
        assert_eq!(format_byte_size(512), "512 B");
        assert_eq!(format_byte_size(1024), "1 KiB");
        assert_eq!(format_byte_size(1536), "1.5 KiB");
        assert_eq!(format_byte_size(1048576), "1 MiB");
        assert_eq!(format_byte_size(1073741824), "1 GiB");
        assert_eq!(format_byte_size(1099511627776), "1 TiB");
    }

    #[test]
    fn format_byte_size_roundtrip() {
        for bytes in &[0u64, 1024, 1536, 1048576, 1073741824] {
            let formatted = format_byte_size(*bytes);
            let parsed = parse_byte_size(&formatted).unwrap();
            assert_eq!(parsed, *bytes);
        }
    }
}
