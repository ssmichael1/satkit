//! Errors produced by the `tle` module.

use thiserror::Error;

/// Errors that can occur while parsing, formatting, or fitting TLEs.
///
/// `#[non_exhaustive]`: variants are added as the parser and fitter learn to
/// reject more malformed input, so downstream matches need a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Invalid TLE line lengths: line1 = {line1}, line2 = {line2}")]
    InvalidLineLengths { line1: usize, line2: usize },

    #[error("Line {line} too short: expected 69 characters, got {got}")]
    LineTooShort { line: u8, got: usize },

    #[error("Line {line} contains non-ASCII characters; TLE lines must be ASCII")]
    NonAscii { line: u8 },

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
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),

    #[error("States and times must have the same length")]
    StatesTimesLengthMismatch,

    #[error("States and times must not be empty")]
    EmptyStates,

    #[error("Epoch is out of range. Must be between {min} and {max}")]
    EpochOutOfRange { min: String, max: String },

    #[error(transparent)]
    Kepler(#[from] crate::kepler::Error),

    #[error("SGP4 evaluation failed: {0}")]
    Sgp4(String),

    /// The element set's ephemeris type selects a propagator satkit does
    /// not implement. Type 4 is SGP4-XP: its line 1 carries agom and a
    /// B term in the columns a classic TLE uses for nddot and B*, so
    /// running classic SGP4 on it would return a plausible but wrong state.
    #[error("Ephemeris type {0} (SGP4-XP) is not supported: satkit implements classic SGP4 only, and an SGP4-XP element set stores agom and a B term where a classic TLE stores nddot and B*")]
    UnsupportedEphemerisType(u8),

    #[error("Normal equations are singular: {0}")]
    SingularNormalEquations(String),

    /// [`TLE::fit_from_states`](crate::TLE::fit_from_states) terminated on
    /// an element set outside the domain a TLE can represent.
    #[error("Fitted TLE has {field} = {value}, outside the valid range {range}")]
    FitElementOutOfRange {
        field: &'static str,
        value: f64,
        range: &'static str,
    },

    /// [`TLE::to_2line`](crate::TLE::to_2line) refuses an eccentricity
    /// outside `[0, 1)`: the 7-digit field cannot hold it, and writing
    /// `|e|` for a negative value would silently describe a different orbit
    /// (the perigee direction reversed).
    #[error("Eccentricity {0} cannot be written to a TLE; it must be in [0, 1)")]
    EccentricityOutOfRange(f64),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),
    /// An HTTP status the server used to say "stop asking" — currently
    /// CelesTrak's 503/403 throttling of repeated identical GP queries. The
    /// message says how to avoid it.
    #[cfg(feature = "download")]
    #[error("{0}")]
    HttpThrottled(String),
}

/// Convenient type alias used throughout the `tle` module.
pub type Result<T> = std::result::Result<T, Error>;
