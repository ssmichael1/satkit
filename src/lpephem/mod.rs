//! Low-precision ephemerides for the sun, moon, and planets.

/// Lunar calculations
pub mod moon;
/// Solar calculations
pub mod sun;

// This part isn't working yet...
mod planets;
pub use planets::heliocentric_pos;

use thiserror::Error;

/// Errors produced by the [`lpephem`](crate::lpephem) module.
///
/// Shared across the [`sun`] and [`planets`] submodules; [`moon`] does
/// not currently surface fallible operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned by [`sun::riseset`] when the sun does not rise or set on
    /// the given date at the supplied location (e.g. polar regions in
    /// summer or winter).
    #[error(
        "Invalid position.  Sun doesn't rise/set on this day at this location \
         (e.g., Alaska in summer)"
    )]
    NoSunriseOrSunset,

    /// Returned by [`heliocentric_pos`] when the requested body is not
    /// represented in the low-precision Keplerian-element table.
    #[error("Invalid body")]
    InvalidBody,

    /// Returned by [`heliocentric_pos`] when the requested time falls
    /// outside the validity window of the low-precision tables
    /// (3000 BC – 3000 AD).
    #[error("Time out of range")]
    TimeOutOfRange,

    /// Wraps an error from constructing an [`Instant`](crate::Instant)
    /// for the validity-window endpoints.
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),
}

/// Convenient type alias used throughout the `lpephem` module.
pub type Result<T> = std::result::Result<T, Error>;
