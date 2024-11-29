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

impl Instant {
    pub fn from_string(s: &str) -> SKResult<Instant> {
        let mut chars = s.chars().peekable();
        let mut year = -1;
        let mut month = -1;
        let mut day = -1;
        let mut hour = 0;
        let mut minute = 0;
        let mut second = 0;
        let mut microsecond = 0;
        let mut numlist = Vec::<i32>::new();
        let mut wordlist = Vec::<String>::new();
        // Find numbers in the string

        while let Some(c) = chars.peek() {
            if c.is_ascii_digit() {
                numlist.push(
                    chars
                        .take_while_ref(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse()?,
                );
            } else if c.is_alphabetic() {
                wordlist.push(
                    chars
                        .take_while_ref(|c| c.is_alphabetic())
                        .collect::<String>(),
                );
            } else {
                chars.next();
            }
        }

        // Find the month if a string
        for (idx, &month_name) in MONTH_NAMES.iter().enumerate() {
            if wordlist.contains(&month_name.to_string()) {
                month = idx as i32 + 1;
                wordlist.retain(|x| x != month_name);
                break;
            }
        }
        if month > 0 {
            // Look for abbreviated month names
            for (idx, &month_abbr) in MONTH_ABBRS.iter().enumerate() {
                if wordlist.contains(&month_abbr.to_string()) {
                    if month > 0 {
                        return skerror!("Ambiguous month name");
                    }
                    month = idx as i32 + 1;
                    wordlist.retain(|x| x != month_abbr);
                    break;
                }
            }
        }

        // Find the year
        for idx in 0..numlist.len() {
            if numlist[idx] > 1900 {
                year = numlist[idx];
                numlist.remove(idx);
                break;
            }
        }

        println!("numlist: {:?}", numlist);
        println!("wordlist: {:?}", wordlist);

        Ok(Instant::J2000)
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
