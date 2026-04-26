use std::num::ParseIntError;

use thiserror::Error;

/// Errors produced when constructing or parsing an [`Instant`](crate::Instant).
#[derive(Error, Debug)]
pub enum InstantError {
    #[error("Invalid Month String: {0}")]
    InvalidMonthString(String),
    #[error("Invalid Month: {0}")]
    InvalidMonth(i32),
    #[error("Invalid Day: {0}")]
    InvalidDay(i32),
    #[error("Invalid Hour: {0}")]
    InvalidHour(i32),
    #[error("Invalid Minute: {0}")]
    InvalidMinute(i32),
    #[error("Invalid Second: {0}")]
    InvalidSecond(i32),
    /// Raised when a fractional second falls outside `[0.0, 60.0)` and is
    /// not a valid leap second on the requested date.
    #[error("Invalid Second: {0}")]
    InvalidSecondF(f64),
    /// Raised when the seconds field is in the leap-second range
    /// `[60.0, 61.0)` but the supplied date does not correspond to an
    /// actual UTC leap second.
    #[error("Invalid leap second")]
    InvalidLeapSecond,
    #[error("Invalid Microsecond: {0}")]
    InvalidMicrosecond(i32),
    #[error("Invalid String: {0}")]
    InvalidString(String),
    #[error("Invalid Format Character: {0}")]
    InvalidFormat(char),
    #[error("Missing Format Character")]
    MissingFormat,
    /// Wraps a [`ParseIntError`] surfaced while parsing a numeric field
    /// (year, month, day, hour, minute, ...).
    #[error("Failed to parse integer field: {0}")]
    ParseInt(#[from] ParseIntError),
}
