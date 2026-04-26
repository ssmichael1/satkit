//! Errors produced by the `tle` module.

use thiserror::Error;

/// Errors that can occur while parsing, formatting, or fitting TLEs.
#[derive(Debug, Error)]
pub enum Error {
    #[error("Invalid TLE line lengths: line1 = {line1}, line2 = {line2}")]
    InvalidLineLengths { line1: usize, line2: usize },

    #[error("Line {line} too short: expected 69 characters, got {got}")]
    LineTooShort { line: u8, got: usize },

    /// Failed to parse a numeric/string field from a TLE line.
    #[error("Could not parse {field}: {message}")]
    ParseField {
        field: &'static str,
        message: String,
    },

    #[error("Year out of range for TLE: {0}")]
    YearOutOfRange(i32),

    #[error("Invalid sat num: {0}")]
    InvalidSatNum(String),

    #[error("Invalid first digit in sat num: {0}")]
    InvalidFirstDigit(char),

    #[error("Parse error")]
    EmptySatNum,

    #[error("Sat num >= 340000 cannot be represented in alpha5 format")]
    SatNumTooLargeForAlpha5,

    #[error("Invalid sat num value")]
    InvalidSatNumValue,

    /// Wraps an error from constructing an [`Instant`](crate::time::Instant)
    /// while assembling a TLE epoch.
    #[error("Invalid TLE epoch: {0}")]
    InvalidEpoch(String),

    #[error("States and times must have the same length")]
    StatesTimesLengthMismatch,

    #[error("States and times must not be empty")]
    EmptyStates,

    #[error("Epoch is out of range. Must be between {min} and {max}")]
    EpochOutOfRange { min: String, max: String },

    #[error("Could not convert state to Keplerian elements: {0}")]
    KeplerConversion(String),

    #[error("SGP4 evaluation failed: {0}")]
    Sgp4(String),

    #[error("Normal equations are singular: {0}")]
    SingularNormalEquations(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),
}

/// Convenient type alias used throughout the `tle` module.
pub type Result<T> = std::result::Result<T, Error>;
