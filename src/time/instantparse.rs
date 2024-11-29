//!
//! Interaction of Instant class with strings
//!

use crate::skerror;
use crate::Instant;
use crate::SKResult;
use itertools::Itertools;

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const MONTH_ABBRS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

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
    /// Args:
    ///   s (str): The string to parse
    ///
    /// Returns:
    ///  Instant: The instant object
    ///
    /// Raises:
    /// SKError: If the string cannot be parsed
    pub fn from_string(s: &str) -> SKResult<Instant> {
        let mut chars = s.chars().peekable();
        let mut year = -1;
        let mut month = -1;
        let mut day = -1;
        let mut hour = -1;
        let mut minute = -1;
        let mut second = -1;
        let mut microsecond = -1;

        let mut thelist = Vec::<ParseVal>::new();
        // Find numbers in the string

        while let Some(c) = chars.peek() {
            if c.is_ascii_digit() {
                thelist.push(ParseVal::Num(
                    chars
                        .take_while_ref(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse()?,
                ));
            } else if c.is_alphabetic() {
                thelist.push(ParseVal::Str(
                    chars
                        .take_while_ref(|c| c.is_alphabetic())
                        .collect::<String>(),
                ));
            } else {
                chars.next();
            }
        } // end of while

        let mut to_remove = Vec::new();
        thelist.iter().enumerate().for_each(|(idx, x)| match x {
            ParseVal::Num(_) => {}
            ParseVal::Str(s) => {
                if month == -1 {
                    month = match MONTH_NAMES.iter().position(|&m| m == *s) {
                        Some(m) => {
                            println!("match month name: {}", m);
                            if idx < thelist.len() - 1 {
                                if let ParseVal::Num(n) = thelist[idx + 1] {
                                    println!("day = {}", n);
                                    day = n;
                                    to_remove.push(idx + 1);
                                }
                            }
                            to_remove.push(idx);
                            m as i32 + 1
                        }
                        None => month,
                    };
                }
                if month == -1 {
                    month = match MONTH_ABBRS.iter().position(|&m| m == *s) {
                        Some(m) => {
                            if idx < thelist.len() - 1 {
                                if let ParseVal::Num(n) = thelist[idx + 1] {
                                    day = n;
                                    to_remove.push(idx + 1);
                                }
                            }
                            to_remove.push(idx);
                            m as i32 + 1
                        }
                        None => month,
                    };
                }
            }
        }); // look for month names

        for &idx in to_remove.iter() {
            thelist.remove(idx);
        }

        // Look for ??:??:?? for time
        if let Some(p) = thelist
            .iter()
            .position(|x| *x == ParseVal::Str(String::from(":")))
        {
            if (p > 0)
                && (p < thelist.len() - 4)
                && (thelist[p + 2] == ParseVal::Str(String::from(":")))
            {
                if let ParseVal::Num(h) = thelist[p - 1] {
                    hour = h;
                }
                if let ParseVal::Num(m) = thelist[p + 1] {
                    minute = m;
                }
                if let ParseVal::Num(s) = thelist[p + 3] {
                    second = s;
                }
                if let ParseVal::Num(m) = thelist[p + 5] {
                    microsecond = m;
                }
            }
        }

        // Look for ??/??/???? for date
        if let Some(p) = thelist
            .iter()
            .position(|x| *x == ParseVal::Str(String::from("/")))
        {
            if (p > 0)
                && (p < thelist.len() - 4)
                && (thelist[p + 2] == ParseVal::Str(String::from("/")))
            {
                if let ParseVal::Num(m) = thelist[p - 1] {
                    if m > 1900 {
                        year = m;
                    } else {
                        month = m;
                    }
                }
                if let ParseVal::Num(d) = thelist[p + 1] {
                    day = d;
                }
                if let ParseVal::Num(y) = thelist[p + 3] {
                    if year >= 0 {
                        month = y;
                    } else {
                        year = y;
                    }
                }
            }
        }

        // Look for ??-??-???? for date
        if let Some(p) = thelist
            .iter()
            .position(|x| *x == ParseVal::Str(String::from("-")))
        {
            if (p > 0)
                && (p < thelist.len() - 4)
                && (thelist[p + 2] == ParseVal::Str(String::from("-")))
            {
                if let ParseVal::Num(m) = thelist[p - 1] {
                    if m > 1900 {
                        year = m;
                    } else {
                        month = m;
                    }
                }
                if let ParseVal::Num(d) = thelist[p + 1] {
                    day = d;
                }
                if let ParseVal::Num(y) = thelist[p + 3] {
                    if year >= 0 {
                        month = y;
                    } else {
                        year = y;
                    }
                }
            }
        }

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
                } else if microsecond == -1 {
                    microsecond = *x;
                }
            }
            ParseVal::Str(_) => {}
        });

        if year == -1 || month == -1 || day == -1 {
            return skerror!("Invalid date string");
        }
        if hour == -1 || minute == -1 || second < 0 {
            hour = 0;
            minute = 0;
            second = 0;
            microsecond = 0;
        }
        Ok(Instant::from_datetime(
            year,
            month,
            day,
            hour,
            minute,
            second as f64 + microsecond as f64 / 1_000_000.0,
        ))
    }

    pub fn strptime(s: &str, format: &str) -> SKResult<Instant> {
        let mut chars = format.chars();
        let mut s_chars = s.chars();
        let mut year = 0;
        let mut month: i32 = 0;
        let mut day = 0;
        let mut hour = 0;
        let mut minute = 0;
        let mut second = 0;
        let mut microsecond = 0;

        while let Some(c) = chars.next() {
            match c {
                '%' => match chars.next() {
                    Some('Y') => year = s_chars.by_ref().take(4).collect::<String>().parse()?,
                    Some('m') => month = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('B') => {
                        let month_name = s_chars
                            .take_while_ref(|c| c.is_alphabetic())
                            .collect::<String>();
                        month = match MONTH_NAMES.iter().position(|&m| m == month_name) {
                            Some(m) => m as i32 + 1,
                            None => {
                                return skerror!("Invalid month name");
                            }
                        };
                    }
                    Some('b') => {
                        let month_abbr = s_chars
                            .take_while_ref(|c| c.is_alphabetic())
                            .collect::<String>();
                        month = match MONTH_ABBRS.iter().position(|&m| m == month_abbr) {
                            Some(m) => m as i32 + 1,
                            None => {
                                return skerror!("Invalid month abbreviation");
                            }
                        }
                    }
                    Some('d') => day = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('H') => hour = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('M') => minute = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('S') => second = s_chars.by_ref().take(2).collect::<String>().parse()?,
                    Some('f') => {
                        microsecond = s_chars
                            .take_while_ref(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse()?
                    }
                    Some(t) => {
                        return skerror!("Invalid format string: {}", t);
                    }
                    None => {
                        return skerror!("Expected a special character");
                    }
                },
                _ => {
                    let n = s_chars.next().unwrap();
                    if c != n {
                        return skerror!("Invalid format match: \"{}\" \"{}\"", c, n);
                    }
                }
            }
        }

        Ok(Instant::from_datetime(
            year,
            month,
            day,
            hour,
            minute,
            second as f64 + microsecond as f64 / 1_000_000.0,
        ))
    }

    /// Parse a string in RFC3339 format
    ///
    /// Args:
    ///    rfc3339 (str): The string in RFC3339 format
    ///
    /// Returns:
    ///   Instant: The instant object
    pub fn from_rfc3339(rfc3339: &str) -> crate::SKResult<Self> {
        if rfc3339.len() < 20 {
            return skerror!("Invalid RFC3339 string");
        }
        Self::strptime(rfc3339, "%Y-%m-%dT%H:%M:%S%.fZ")
    }

    pub fn strftime(&self, format: &str) -> SKResult<String> {
        let mut result = String::new();
        let mut chars = format.chars();

        let (year, month, day, hour, minute, fsecond) = self.as_datetime();
        let second = fsecond as i32;
        let nanosecond = (fsecond.fract() * 1_000_000_000.0) as u32;

        while let Some(c) = chars.next() {
            if c == '%' {
                match chars.next() {
                    Some('Y') => {
                        result.push_str(&year.to_string());
                    }
                    Some('m') => {
                        result.push_str(&format!("{:02}", month));
                    }
                    Some('d') => {
                        result.push_str(&format!("{:02}", day));
                    }
                    Some('H') => {
                        result.push_str(&format!("{:02}", hour));
                    }
                    Some('M') => {
                        result.push_str(&format!("{:02}", minute));
                    }
                    Some('S') => {
                        result.push_str(&format!("{:02}", second));
                    }
                    Some('f') => {
                        result.push_str(&format!("{:09}", nanosecond));
                    }
                    Some(_) => {
                        return skerror!("Invalid format string");
                    }
                    None => {
                        return skerror!("Expected a special character");
                    }
                }
            } else {
                result.push(c);
            }
        }
        Ok(result)
    }
}
