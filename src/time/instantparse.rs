//!
//! Interaction of Instant class with strings
//!

use crate::time::InstantError;
use crate::Instant;
use std::fmt::Write;

/// Local result alias used by [`Instant`] string parsers.
type Result<T> = std::result::Result<T, InstantError>;

/// Collect characters from a `Peekable<Chars>` while `pred` holds, leaving the
/// first non-matching character available for the next read (unlike std's
/// `take_while`, which consumes and discards it).
fn take_while_peek(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    mut pred: impl FnMut(char) -> bool,
) -> String {
    let mut out = String::new();
    while let Some(&c) = chars.peek() {
        if !pred(c) {
            break;
        }
        out.push(c);
        chars.next();
    }
    out
}

/// Microseconds from the digits after a decimal point. Trailing zeros may
/// be omitted, so the digit count sets the scale; more than 6 digits carry
/// sub-microsecond precision that is not stored and are truncated (rather
/// than overflowing i32). No digits is 0.
fn parse_fraction_us(digits: &str) -> Result<i32> {
    if digits.is_empty() {
        return Ok(0);
    }
    // The digits are ASCII, so the byte slice is safe
    let n = digits.len().min(6);
    Ok(digits[..n].parse::<i32>()? * 10i32.pow((6 - n) as u32))
}

/// Name of month `month` (1–12) from `names`, or "Invalid" out of range
/// (`as_datetime` can return one for extreme or INVALID instants).
fn month_name(names: &[&'static str; 12], month: i32) -> &'static str {
    if (1..=12).contains(&month) {
        names[(month - 1) as usize]
    } else {
        "Invalid"
    }
}

/// Full month names
const MONTH_NAMES: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
];

/// Abbreviated month names
const MONTH_ABBRS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Abbreviated weekday names, indexed by `Weekday as i32` (0 = Sunday).
const WEEKDAY_ABBRS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

#[derive(PartialEq, Debug)]
enum ParseVal {
    Str(String),
    Num(i32),
}

impl Instant {
    /// Parse a string into an Instant object
    ///
    /// Attempts to guess the string format.
    /// Use sparingly and with caution.  This is
    /// probably not what you want.
    ///
    /// RFC 3339 is tried first. Otherwise the string is tokenized into numbers
    /// and month names, and numeric fields are consumed positionally in
    /// year, month, day, hour, minute, second, microsecond order. Separators
    /// are ignored, so ISO-ordered strings (e.g. `"2024-01-04 13:14:12.123"`)
    /// and month-name strings (e.g. `"March 4 2024"`) work, but locale-ordered
    /// numeric dates such as `MM/DD/YYYY` are ambiguous and are not supported;
    /// use [`strptime`](Self::strptime) with an explicit format for those.
    ///
    /// # Arguments:
    ///   s (str): The string to parse
    ///
    /// # Returns:
    ///  Instant: The instant object
    ///
    /// # Raises:
    /// SCErr: If the string cannot be parsed
    pub fn from_string(s: &str) -> Result<Self> {
        // Try RFC 3339 first — it's unambiguous and common
        if let Ok(r) = Self::from_rfc3339(s) {
            return Ok(r);
        }

        let mut chars = s.chars().peekable();
        let mut year = -1;
        let mut month = -1;
        let mut day = -1;
        let mut hour = -1;
        let mut minute = -1;
        let mut second = -1;
        let mut microsecond = 0i32;
        let mut microsecond_set = false;

        let mut thelist = Vec::<ParseVal>::new();

        let mut isperiod: bool = false;
        while let Some(c) = chars.peek() {
            if c.is_ascii_digit() {
                let cstr = take_while_peek(&mut chars, |c| c.is_ascii_digit());
                // Following a period: a fraction of a second, in microseconds
                let val = match isperiod {
                    true => parse_fraction_us(&cstr)?,
                    false => cstr.parse::<i32>()?,
                };
                thelist.push(ParseVal::Num(val));
            } else if c.is_alphabetic() {
                thelist.push(ParseVal::Str(take_while_peek(&mut chars, |c| {
                    c.is_alphabetic()
                })));
            } else if let Some(c) = chars.next() {
                isperiod = c == '.';
            }
        }

        // Look for month names (full or abbreviated)
        let mut to_remove = Vec::new();
        thelist.iter().enumerate().for_each(|(idx, x)| match x {
            ParseVal::Num(_) => {}
            ParseVal::Str(s) => {
                if month == -1 {
                    let found = MONTH_NAMES
                        .iter()
                        .position(|&m| m == *s)
                        .or_else(|| MONTH_ABBRS.iter().position(|&m| m == *s));
                    if let Some(m) = found {
                        if idx < thelist.len() - 1 {
                            if let ParseVal::Num(n) = thelist[idx + 1] {
                                day = n;
                                to_remove.push(idx + 1);
                            }
                        }
                        to_remove.push(idx);
                        month = m as i32 + 1;
                    }
                }
            }
        });
        // Remove in reverse order so indices stay valid
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            thelist.remove(idx);
        }

        // Fill the remaining fields positionally from the leftover numbers, in
        // year, month, day, hour, minute, second, microsecond order. Separators
        // (`:` `/` `-`) are not tokenized, so numeric fields must already appear
        // in that order — ISO-8601-ordered strings and month-name strings parse
        // correctly; ambiguous locale-ordered numeric dates (e.g. MM/DD/YYYY) do
        // not and should be parsed with `strptime` and an explicit format.
        // Fill remaining fields from leftover numbers
        thelist.iter().for_each(|x| match x {
            ParseVal::Num(x) => {
                if year == -1 {
                    year = *x;
                } else if month == -1 {
                    month = *x;
                } else if day == -1 {
                    day = *x;
                } else if hour == -1 {
                    hour = *x;
                } else if minute == -1 {
                    minute = *x;
                } else if second == -1 {
                    second = *x;
                } else if !microsecond_set {
                    microsecond = *x;
                    microsecond_set = true;
                }
            }
            ParseVal::Str(_) => {}
        });

        if year == -1 || month == -1 || day == -1 {
            return Err(InstantError::InvalidString(s.to_string()));
        }
        if hour == -1 || minute == -1 || second < 0 {
            hour = 0;
            minute = 0;
            second = 0;
            microsecond = 0;
        }
        // Integer microseconds: no float rounding on the way in
        Self::from_datetime_us(
            year,
            month,
            day,
            hour,
            minute,
            second as i64 * 1_000_000 + microsecond as i64,
        )
    }

    /// Parse a string into an Instant object
    ///
    /// # Notes:
    /// * The format string is a subset of the Python datetime module
    ///
    /// # Arguments:
    /// * s (str): The string to parse
    /// * format (str): The format string
    ///
    /// # Format Codes:
    /// * %Y - Year with century as a decimal number
    /// * %m - Month as a zero-padded decimal number [01, 12]
    /// * %B - Full month name (January, February, etc.)
    /// * %b - Abbreviated month name (Jan, Feb, etc.)
    /// * %d - Day of the month as a zero-padded decimal number [01, 31]
    /// * %H - Hour (24-hour clock) as a zero-padded decimal number
    /// * %M - Minute as a zero-padded decimal number
    /// * %S - Second as a zero-padded decimal number
    /// * %f - Microsecond as a decimal number, allowing for trailing zeros
    /// * %z - UTC offset in the form +HHMM or -HHMM (or +HH:MM), or 'Z' for UTC.
    ///   `+HHMM` means local time is ahead of UTC, so `12:00:00+0100` is
    ///   `11:00:00Z`; the offset shifts the calendar label, so it is exact
    ///   across a leap second
    ///
    /// # Returns:
    /// Instant: The instant object
    ///
    pub fn strptime(s: &str, format: &str) -> Result<Self> {
        let mut chars = format.chars();
        let mut s_chars = s.chars().peekable();
        let mut year = 0;
        let mut month: i32 = 0;
        let mut day = 0;
        let mut hour = 0;
        let mut minute = 0;
        let mut second = 0;
        let mut microsecond = 0;
        let mut offset = 0;

        while let Some(c) = chars.next() {
            match c {
                '%' => match chars.next() {
                    Some('Y') => year = s_chars.by_ref().take(4).collect::<String>().parse()?,
                    Some('m') => month = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some(code @ ('B' | 'b')) => {
                        let names = if code == 'B' {
                            &MONTH_NAMES
                        } else {
                            &MONTH_ABBRS
                        };
                        let name = take_while_peek(&mut s_chars, |c| c.is_alphabetic());
                        month = names
                            .iter()
                            .position(|&m| m == name)
                            .map(|m| m as i32 + 1)
                            .ok_or(InstantError::InvalidMonthString(name))?;
                    }
                    Some('d') => day = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('H') => hour = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('M') => minute = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('S') => second = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('f') => {
                        let smicro = take_while_peek(&mut s_chars, |c| c.is_ascii_digit());
                        microsecond = parse_fraction_us(&smicro)?;
                    }
                    Some('z') => {
                        let z = s_chars.by_ref().take(1).collect::<String>();
                        if z == "Z" {
                            // UTC
                        } else {
                            let sign = if z == "-" { -1 } else { 1 };
                            let h = s_chars
                                .by_ref()
                                .take(2)
                                .collect::<String>()
                                .parse::<i32>()?;
                            // take the colon if it is there (it appears to be optional)
                            if s_chars.peek() == Some(&':') {
                                s_chars.next();
                            }
                            let m = s_chars
                                .by_ref()
                                .take(2)
                                .collect::<String>()
                                .parse::<i32>()?;
                            offset = sign * (h * 60 + m);
                        }
                    }
                    Some(t) => {
                        return Err(InstantError::InvalidFormat(t));
                    }
                    None => {
                        return Err(InstantError::InvalidFormat('%'));
                    }
                },
                _ => {
                    let n = s_chars.next().unwrap_or('_');
                    if c != n {
                        return Err(InstantError::InvalidString(format!(
                            "{} doesn't match {}",
                            c, n
                        )));
                    }
                }
            }
        }

        // `offset` is local time minus UTC, in minutes. Local label minus
        // offset is the UTC label (applied to the label, not as elapsed
        // time, so it is exact across a leap second).
        Self::from_local_datetime_us(
            year,
            month,
            day,
            hour,
            minute,
            second as i64 * 1_000_000 + microsecond as i64,
            offset as i64 * 60_000_000,
        )
    }

    /// Parse a string in RFC3339 format
    ///
    /// # Arguments:
    ///    rfc3339 (str): The string in RFC3339 format
    ///
    /// # Notes:
    /// * Only allows a subset of the RFC3339 format: "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// # Returns:
    ///   Instant: The instant object
    pub fn from_rfc3339(rfc3339: &str) -> std::result::Result<Self, InstantError> {
        // Try formats ending with 'Z' (UTC) first
        if let Ok(r) = Self::strptime(rfc3339, "%Y-%m-%dT%H:%M:%S.%fZ") {
            return Ok(r);
        }
        if let Ok(r) = Self::strptime(rfc3339, "%Y-%m-%dT%H:%M:%SZ") {
            return Ok(r);
        }

        // Try formats with timezone offset (+HH:MM or -HH:MM)
        // RFC 3339 allows offsets like +00:00, -05:00, etc.
        let s = rfc3339.trim();
        // `s.len()` counts bytes, so the fixed-width slices below are only
        // panic-free when the 6-byte tail is ASCII and starts on a char
        // boundary; non-ASCII input must fall through to the error return.
        if s.len() >= 6 && s.is_char_boundary(s.len() - 6) {
            let offset_start = s.len() - 6;
            let maybe_offset = &s[offset_start..];
            if maybe_offset.is_ascii()
                && (maybe_offset.starts_with('+') || maybe_offset.starts_with('-'))
                && maybe_offset.chars().nth(3) == Some(':')
            {
                // `%z` takes `+HH:MM` and applies it to the label (exact
                // across a leap second), so hand the whole string over
                let r = Self::strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")
                    .or_else(|_| Self::strptime(s, "%Y-%m-%dT%H:%M:%S%z"));
                if let Ok(r) = r {
                    return Ok(r);
                }
            }
        }

        // Try bare formats (no timezone indicator — assume UTC)
        if let Ok(r) = Self::strptime(rfc3339, "%Y-%m-%dT%H:%M:%S.%f") {
            return Ok(r);
        }
        if let Ok(r) = Self::strptime(rfc3339, "%Y-%m-%dT%H:%M:%S") {
            return Ok(r);
        }
        Err(InstantError::InvalidString(rfc3339.to_string()))
    }

    /// Format the Instant object as a string in RFC3339 format
    ///
    /// # Returns:
    /// str: The formatted string in RFC3339 format: "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// # Notes:
    /// * This is the same as ISO8601 format
    pub fn as_rfc3339(&self) -> String {
        self.strftime("%Y-%m-%dT%H:%M:%S.%fZ").unwrap()
    }

    /// Format the Instant object as a string in ISO8601 format
    ///
    /// # Returns:
    /// str: The formatted string in ISO8601 format: "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// # Notes:
    /// * This is the same as RFC3339 format
    pub fn as_iso8601(&self) -> String {
        self.as_rfc3339()
    }

    /// Format the Instant object as a string
    ///
    /// # Notes:
    /// * The format string is a subset of the Python datetime module
    ///
    /// # Arguments:
    ///  format (str): The format string
    ///
    /// # Format Codes:
    /// * %Y - Year with century as a decimal number
    /// * %m - Month as a zero-padded decimal number [01, 12]
    /// * %B - Full month name (January, February, etc.)
    /// * %b - Abbreviated month name (Jan, Feb, etc.)
    /// * %d - Day of the month as a zero-padded decimal number [01, 31]
    /// * %H - Hour (24-hour clock) as a zero-padded decimal number
    /// * %M - Minute as a zero-padded decimal number
    /// * %S - Second as a zero-padded decimal number
    /// * %f - Microsecond as a decimal number
    /// * %A - Full weekday name (Sunday, Monday, etc.)
    /// * %a - Abbreviated weekday name (Sun, Mon, etc.)
    /// * %w - Weekday as a decimal number [0(Sunday), 6(Saturday)]
    ///
    /// # Returns:
    /// str: The formatted string
    ///
    pub fn strftime(&self, format: &str) -> std::result::Result<String, InstantError> {
        let mut result = String::new();
        let mut chars = format.chars();

        let (year, month, day, hour, minute, second_us) = self.as_datetime_us();
        let second = second_us / 1_000_000;
        let microsecond = second_us % 1_000_000;

        while let Some(c) = chars.next() {
            if c != '%' {
                result.push(c);
                continue;
            }
            // Writing to a String cannot fail
            let _ = match chars.next() {
                Some('Y') => write!(result, "{year}"),
                Some('m') => write!(result, "{month:02}"),
                Some('d') => write!(result, "{day:02}"),
                Some('H') => write!(result, "{hour:02}"),
                Some('M') => write!(result, "{minute:02}"),
                Some('S') => write!(result, "{second:02}"),
                Some('f') => write!(result, "{microsecond:06}"),
                Some('B') => write!(result, "{}", month_name(&MONTH_NAMES, month)),
                Some('b') => write!(result, "{}", month_name(&MONTH_ABBRS, month)),
                Some('A') => write!(result, "{}", self.day_of_week()),
                Some('a') => {
                    let idx = self.day_of_week() as i32;
                    let abbr = if (0..7).contains(&idx) {
                        WEEKDAY_ABBRS[idx as usize]
                    } else {
                        "Invalid"
                    };
                    write!(result, "{abbr}")
                }
                Some('w') => write!(result, "{:02}", self.day_of_week() as i32),
                Some(c) => {
                    return Err(InstantError::InvalidFormat(c));
                }
                None => {
                    return Err(InstantError::InvalidString(
                        "Expected a format character".to_string(),
                    ));
                }
            };
        }
        Ok(result)
    }
}
