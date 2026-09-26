//!
//! Interaction of Instant class with strings
//!

use super::instant::IsoYear;
use crate::time::{Duration, InstantError};
use crate::Instant;
use std::fmt::Write;

/// Local result alias used by [`Instant`] string parsers.
type Result<T> = std::result::Result<T, InstantError>;

/// Microseconds from the ASCII digits after a decimal point, rounded to the
/// nearest microsecond (halves up), and whether rounding went up. Trailing
/// zeros may be omitted, so the digit count sets the scale. The result can
/// be `1_000_000` (`.9999995`), which [`Fields::to_instant`] carries into
/// the seconds, as [`Instant::from_datetime`] does. No digits is 0.
fn parse_fraction_us(digits: &str) -> (i64, bool) {
    let n = digits.len().min(6);
    let us = digits[..n].parse::<i64>().unwrap_or(0) * 10i64.pow((6 - n) as u32);
    let up = digits.as_bytes().get(6).is_some_and(|&b| b >= b'5');
    (us + i64::from(up), up)
}

/// Check a UTC offset's fields and return it in minutes (local − UTC).
/// Hours 00–23 and minutes 00–59, as for any real zone (and Python's
/// `datetime`, which rejects |offset| ≥ 24 h).
fn offset_minutes(sign: i32, h: i32, m: i32, s: &str) -> Result<i32> {
    if h > 23 || m > 59 {
        return Err(InstantError::InvalidString(format!(
            "UTC offset {h:02}:{m:02} out of range (hours 00-23, minutes 00-59) in {s:?}"
        )));
    }
    Ok(sign * (h * 60 + m))
}

/// A read position in a string being parsed.
struct Cursor<'a> {
    s: &'a str,
    /// Byte index into `s`, always on a char boundary
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }

    /// The unread input
    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn is_empty(&self) -> bool {
        self.pos == self.s.len()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn eat(&mut self, c: char) -> bool {
        let hit = self.peek() == Some(c);
        if hit {
            self.pos += c.len_utf8();
        }
        hit
    }

    fn error(&self, what: &str) -> InstantError {
        InstantError::InvalidString(format!("{what} at {:?} in {:?}", self.rest(), self.s))
    }

    fn expect(&mut self, c: char) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.error(&format!("expected {c:?}")))
        }
    }

    /// The run of ASCII digits at the cursor (possibly empty)
    fn digit_run(&mut self) -> &'a str {
        let rest = self.rest();
        let n = rest.bytes().take_while(u8::is_ascii_digit).count();
        self.pos += n;
        &rest[..n]
    }

    /// The run of alphabetic characters at the cursor (possibly empty)
    fn word(&mut self) -> &'a str {
        let rest = self.rest();
        let n = rest
            .find(|c: char| !c.is_alphabetic())
            .unwrap_or(rest.len());
        self.pos += n;
        &rest[..n]
    }

    /// Exactly `n` (≤ 9) ASCII digits
    fn digits(&mut self, n: usize, field: &str) -> Result<i32> {
        let rest = self.rest();
        if rest.len() >= n && rest.as_bytes()[..n].iter().all(u8::is_ascii_digit) {
            self.pos += n;
            Ok(rest[..n].parse()?)
        } else {
            Err(self.error(&format!("expected {n}-digit {field}")))
        }
    }

    /// A year: exactly four digits, or a sign and at least four digits
    /// (the ISO 8601 expanded form that `strftime` writes outside 0–9999)
    fn year(&mut self) -> Result<i32> {
        let sign = match self.peek() {
            Some('+') => 1,
            Some('-') => -1,
            _ => return self.digits(4, "year"),
        };
        self.bump();
        let digits = self.digit_run();
        if digits.len() < 4 {
            return Err(self.error("expected at least 4 year digits after the sign"));
        }
        Ok(sign * digits.parse::<i32>()?)
    }

    /// Fraction-of-second digits (at least one), rounded to the nearest
    /// microsecond; see [`parse_fraction_us`]
    fn fraction_us(&mut self) -> Result<(i64, bool)> {
        let digits = self.digit_run();
        if digits.is_empty() {
            return Err(self.error("expected fraction-of-second digits"));
        }
        Ok(parse_fraction_us(digits))
    }

    /// A UTC offset in minutes (local − UTC): `Z` / `z`, `±HH:MM`, `±HHMM`
    /// or `±HH`, with exactly two digits per field
    fn utc_offset(&mut self) -> Result<i32> {
        let sign = match self.peek() {
            Some('Z' | 'z') => {
                self.bump();
                return Ok(0);
            }
            Some('+') => 1,
            Some('-') => -1,
            _ => return Err(self.error("expected a UTC offset (Z, ±HH:MM, ±HHMM or ±HH)")),
        };
        self.bump();
        let h = self.digits(2, "UTC offset hour")?;
        let m = if self.eat(':') || self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.digits(2, "UTC offset minute")?
        } else {
            0
        };
        offset_minutes(sign, h, m, self.s)
    }
}

/// Calendar fields read from a string.
#[derive(Default)]
struct Fields {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    /// Microseconds of the second, `0..=1_000_000`
    frac_us: i64,
    /// `frac_us` was rounded up from more than 6 digits
    rounded_up: bool,
    /// Local time minus UTC, in minutes
    offset_min: i32,
}

impl Fields {
    /// The instant, with the offset applied to the label (exact across a
    /// leap second). A fraction that rounded up past the end of its minute
    /// (or of its leap second) carries as in [`Instant::from_datetime`]:
    /// into the leap second where one follows, otherwise into the next
    /// minute, or from the end of a leap second into the next day.
    fn to_instant(&self) -> Result<Instant> {
        let build = |second_us| {
            Instant::from_local_datetime_us(
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
                second_us,
                self.offset_min as i64 * 60_000_000,
            )
        };
        let second_us = self.second as i64 * 1_000_000 + self.frac_us;
        let r = build(second_us);
        if r.is_err() && self.rounded_up {
            if let Ok(t) = build(second_us - 1) {
                return Ok(t + Duration::from_microseconds(1));
            }
        }
        r
    }
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

/// Month number (1–12) of a full or abbreviated month name.
fn month_number(name: &str) -> Option<i32> {
    MONTH_NAMES
        .iter()
        .position(|&m| m == name)
        .or_else(|| MONTH_ABBRS.iter().position(|&m| m == name))
        .map(|m| m as i32 + 1)
}

/// Abbreviated weekday names, indexed by `Weekday as i32` (0 = Sunday).
const WEEKDAY_ABBRS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// A token of the free-form [`Instant::from_string`] parser.
enum Token<'a> {
    /// A run of ASCII digits, and the last non-space separator character
    /// between it and the previous token (`None` if there was none)
    Num { digits: &'a str, sep: Option<char> },
    /// A run of alphabetic characters (month and weekday names, `T`, `Z`, ...)
    Word(&'a str),
}

/// Split `s` into digit runs and words; other characters are separators.
fn tokenize(s: &str) -> Vec<Token<'_>> {
    let mut c = Cursor::new(s);
    let mut tokens = Vec::new();
    let mut sep = None;
    while let Some(ch) = c.peek() {
        if ch.is_ascii_digit() {
            let digits = c.digit_run();
            tokens.push(Token::Num {
                digits,
                sep: sep.take(),
            });
        } else if ch.is_alphabetic() {
            tokens.push(Token::Word(c.word()));
            sep = None;
        } else {
            c.bump();
            if !ch.is_whitespace() {
                sep = Some(ch);
            }
        }
    }
    tokens
}

impl Instant {
    /// Parse a string into an Instant object
    ///
    /// Attempts to guess the string format.
    /// Use sparingly and with caution.  This is
    /// probably not what you want.
    ///
    /// RFC 3339 is tried first. Otherwise the string is tokenized into numbers
    /// and words, and numeric fields are consumed positionally in year,
    /// month, day, hour, minute, second order. ISO-ordered strings (e.g.
    /// `"2024-01-04 13:14:12.123"`) and month-name strings (e.g.
    /// `"March 4 2024"`) work, but locale-ordered numeric dates such as
    /// `MM/DD/YYYY` are ambiguous and are not supported; use
    /// [`strptime`](Self::strptime) with an explicit format for those.
    ///
    /// After the fields:
    /// * a number after `.` following the seconds is the fraction of a
    ///   second (rounded to the nearest microsecond);
    /// * a number after `+` or `-` following the minutes is a UTC offset,
    ///   `±HHMM`, `±HH:MM` or `±HH` (hours 00–23, minutes 00–59), applied as
    ///   in [`from_rfc3339`](Self::from_rfc3339);
    /// * any other number is an error.
    ///
    /// Seconds default to 0 (`"2024-01-04 13:14"`); an hour without minutes
    /// is an error, and with no time at all it is midnight. Words other than
    /// month names (weekday names, `T`, `Z`, `UTC`, ...) are ignored, so a
    /// zone *name* is not applied: without a numeric offset the time is UTC.
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

        let mut tokens = tokenize(s);
        let bad = |what: String| InstantError::InvalidString(format!("{what} in {s:?}"));
        // [year, month, day, hour, minute, second], filled in that order
        let mut fields: [Option<i32>; 6] = [None; 6];

        // The first month name (full or abbreviated) sets the month, and a
        // number right after it the day
        let month_word = tokens.iter().enumerate().find_map(|(i, t)| match t {
            Token::Word(w) => month_number(w).map(|m| (i, m)),
            Token::Num { .. } => None,
        });
        if let Some((i, month)) = month_word {
            fields[1] = Some(month);
            if let Some(Token::Num { digits, .. }) = tokens.get(i + 1) {
                fields[2] = Some(digits.parse()?);
                tokens.remove(i + 1);
            }
            tokens.remove(i);
        }

        let mut frac = None;
        let mut offset = None;
        let mut nums = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Num { digits, sep } => Some((*digits, *sep)),
                Token::Word(_) => None,
            })
            .peekable();
        while let Some((digits, sep)) = nums.next() {
            if sep == Some('.') && fields[5].is_some() && frac.is_none() && offset.is_none() {
                frac = Some(parse_fraction_us(digits));
            } else if matches!(sep, Some('+' | '-')) && fields[4].is_some() && offset.is_none() {
                let sign = if sep == Some('-') { -1 } else { 1 };
                let (h, m) = match digits.len() {
                    4 => (&digits[..2], &digits[2..]),
                    2 => match nums.peek() {
                        Some(&(mm, Some(':'))) if mm.len() == 2 => {
                            nums.next();
                            (digits, mm)
                        }
                        _ => (digits, "00"),
                    },
                    _ => {
                        return Err(bad(format!(
                            "UTC offset {digits:?} is not ±HHMM, ±HH:MM or ±HH"
                        )))
                    }
                };
                offset = Some(offset_minutes(sign, h.parse()?, m.parse()?, s)?);
            } else if fields[0].is_none() && frac.is_none() && offset.is_none() {
                // The year keeps a sign written before it ("-0044-03-15",
                // the ISO 8601 expanded form); it used to be dropped,
                // turning 44 BC into AD 44.
                let year: i32 = digits.parse()?;
                fields[0] = Some(if sep == Some('-') { -year } else { year });
            } else if let Some(slot) = fields
                .iter_mut()
                .find(|f| f.is_none())
                .filter(|_| frac.is_none() && offset.is_none())
            {
                *slot = Some(digits.parse()?);
            } else {
                return Err(bad(format!("unexpected number {digits:?}")));
            }
        }

        let [Some(year), Some(month), Some(day), hour, minute, second] = fields else {
            return Err(InstantError::InvalidString(s.to_string()));
        };
        let (hour, minute) = match (hour, minute) {
            (Some(h), Some(m)) => (h, m),
            (Some(_), None) => return Err(bad("an hour without minutes".to_string())),
            (None, _) => (0, 0),
        };
        let (frac_us, rounded_up) = frac.unwrap_or((0, false));
        Fields {
            year,
            month,
            day,
            hour,
            minute,
            second: second.unwrap_or(0),
            frac_us,
            rounded_up,
            offset_min: offset.unwrap_or(0),
        }
        .to_instant()
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
    /// * %Y - Year: exactly 4 digits, or a sign and at least 4 digits (the
    ///   ISO 8601 expanded form, e.g. `-0044`, `+10000`, which
    ///   [`strftime`](Self::strftime) writes for years outside 0000–9999)
    /// * %m - Month as a zero-padded decimal number [01, 12]
    /// * %B - Full month name (January, February, etc.)
    /// * %b - Abbreviated month name (Jan, Feb, etc.)
    /// * %d - Day of the month as a zero-padded decimal number [01, 31]
    /// * %H - Hour (24-hour clock) as a zero-padded decimal number
    /// * %M - Minute as a zero-padded decimal number
    /// * %S - Second as a zero-padded decimal number
    /// * %f - Fraction of a second: one or more digits (`.5` is 500 ms),
    ///   rounded to the nearest microsecond beyond 6 digits
    /// * %z - UTC offset `±HH:MM`, `±HHMM` or `±HH` (exactly two digits per
    ///   field, hours 00–23, minutes 00–59), or `Z` / `z` for UTC.
    ///   `+HHMM` means local time is ahead of UTC, so `12:00:00+0100` is
    ///   `11:00:00Z`; the offset shifts the calendar label, so it is exact
    ///   across a leap second
    /// * %% - A literal `%`
    ///
    /// The numeric fields take exactly the stated number of digits, other
    /// format characters must match literally, and the whole string must be
    /// consumed: leftover input is an error.
    ///
    /// # Returns:
    /// Instant: The instant object
    ///
    pub fn strptime(s: &str, format: &str) -> Result<Self> {
        let mut c = Cursor::new(s);
        let mut f = Fields::default();
        let mut fmt = format.chars();

        while let Some(fc) = fmt.next() {
            if fc != '%' {
                c.expect(fc)?;
                continue;
            }
            match fmt.next() {
                Some('Y') => f.year = c.year()?,
                Some('m') => f.month = c.digits(2, "month")?,
                Some(code @ ('B' | 'b')) => {
                    let names = if code == 'B' {
                        &MONTH_NAMES
                    } else {
                        &MONTH_ABBRS
                    };
                    let name = c.word();
                    f.month = names
                        .iter()
                        .position(|&m| m == name)
                        .map(|m| m as i32 + 1)
                        .ok_or_else(|| InstantError::InvalidMonthString(name.to_string()))?;
                }
                Some('d') => f.day = c.digits(2, "day")?,
                Some('H') => f.hour = c.digits(2, "hour")?,
                Some('M') => f.minute = c.digits(2, "minute")?,
                Some('S') => f.second = c.digits(2, "second")?,
                Some('f') => (f.frac_us, f.rounded_up) = c.fraction_us()?,
                Some('z') => f.offset_min = c.utc_offset()?,
                Some('%') => c.expect('%')?,
                Some(t) => return Err(InstantError::InvalidFormat(t)),
                None => return Err(InstantError::InvalidFormat('%')),
            }
        }
        if !c.is_empty() {
            return Err(c.error("unparsed input left over"));
        }
        f.to_instant()
    }

    /// Parse a string in RFC3339 format
    ///
    /// # Arguments:
    ///    rfc3339 (str): The string in RFC3339 format
    ///
    /// # Notes:
    /// * Format `YYYY-MM-DDTHH:MM:SS[.fff…][zone]`: `T` may be `t`; the
    ///   fraction has one or more digits, rounded to the nearest microsecond
    ///   beyond 6; surrounding whitespace is ignored, anything else left over
    ///   is an error
    /// * The zone is `Z` / `z`, or an offset `±HH:MM` (RFC 3339), `±HHMM` or
    ///   `±HH` (ISO 8601 forms, also accepted), with hours 00–23 and minutes
    ///   00–59. The offset shifts the calendar label, so it is exact across
    ///   a leap second, and a leap second written in local time
    ///   (`00:59:60+01:00`) is the UTC one
    /// * Without a zone the time is taken as UTC (not RFC 3339, which
    ///   requires one)
    /// * The year is 4 digits, or a sign and at least 4 digits (ISO 8601
    ///   expanded years, as [`as_rfc3339`](Self::as_rfc3339) writes them
    ///   outside 0000–9999)
    ///
    /// # Returns:
    ///   Instant: The instant object
    pub fn from_rfc3339(rfc3339: &str) -> std::result::Result<Self, InstantError> {
        let mut c = Cursor::new(rfc3339.trim());
        let mut f = Fields {
            year: c.year()?,
            ..Default::default()
        };
        c.expect('-')?;
        f.month = c.digits(2, "month")?;
        c.expect('-')?;
        f.day = c.digits(2, "day")?;
        if !(c.eat('T') || c.eat('t')) {
            return Err(c.error("expected 'T'"));
        }
        f.hour = c.digits(2, "hour")?;
        c.expect(':')?;
        f.minute = c.digits(2, "minute")?;
        c.expect(':')?;
        f.second = c.digits(2, "second")?;
        if c.eat('.') {
            (f.frac_us, f.rounded_up) = c.fraction_us()?;
        }
        // Once a zone is present it is used or the parse fails: an offset is
        // never dropped and the local label read as UTC
        if !c.is_empty() {
            f.offset_min = c.utc_offset()?;
        }
        if !c.is_empty() {
            return Err(c.error("unparsed input left over"));
        }
        f.to_instant()
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
                Some('Y') => write!(result, "{}", IsoYear(year)),
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
                Some('w') => write!(result, "{}", self.day_of_week() as i32),
                Some('%') => write!(result, "%"),
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
