//! Errors produced by the `sgp4` module.

use thiserror::Error;

/// Typed SGP4 error codes, following the legacy Vallado convention.
#[derive(Debug, Clone, Error, PartialEq, Eq, Copy)]
pub enum SGP4Error {
    #[error("Success")]
    SGP4Success = 0,
    #[error("Eccentricity > 1 or < 0")]
    SGP4ErrorEccen = 1,
    #[error("Mean motion < 0")]
    SGP4ErrorMeanMotion = 2,
    #[error("Perturbed Eccentricity > 1 or < 0")]
    SGP4ErrorPerturbEccen = 3,
    #[error("Semi-Latus Rectum < 0")]
    SGP4ErrorSemiLatusRectum = 4,
    #[error("Unused")]
    SGP4ErrorUnused = 5,
    #[error("Orbit Decayed")]
    SGP4ErrorOrbitDecay = 6,
}
impl From<i32> for SGP4Error {
    fn from(val: i32) -> Self {
        match val {
            0 => Self::SGP4Success,
            1 => Self::SGP4ErrorEccen,
            2 => Self::SGP4ErrorMeanMotion,
            3 => Self::SGP4ErrorPerturbEccen,
            4 => Self::SGP4ErrorSemiLatusRectum,
            6 => Self::SGP4ErrorOrbitDecay,
            _ => Self::SGP4ErrorUnused,
        }
    }
}

impl From<SGP4Error> for i32 {
    fn from(val: SGP4Error) -> Self {
        match val {
            SGP4Error::SGP4ErrorEccen => 1,
            SGP4Error::SGP4ErrorMeanMotion => 2,
            SGP4Error::SGP4ErrorOrbitDecay => 6,
            SGP4Error::SGP4ErrorPerturbEccen => 3,
            SGP4Error::SGP4ErrorSemiLatusRectum => 4,
            SGP4Error::SGP4ErrorUnused => -1,
            SGP4Error::SGP4Success => 0,
        }
    }
}

/// Errors that can occur while initialising or evaluating SGP4.
#[derive(Debug, Error)]
pub enum Error {
    /// `sgp4init` returned a non-zero error code while constructing the
    /// internal `SatRec`. Carries the typed [`SGP4Error`] describing the
    /// failure (eccentricity, mean motion, perturbed eccentricity,
    /// semi-latus rectum, or orbit decay).
    #[error("SGP4 init error: {0}")]
    SatRecInit(SGP4Error),

    /// Wraps an error surfaced by an [`SGP4Source`](super::SGP4Source)
    /// implementation while building [`SGP4InitArgs`](super::SGP4InitArgs)
    /// — for example an `OMM` with an unsupported mean-element theory or
    /// a malformed epoch.
    #[error(transparent)]
    Source(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wrap an arbitrary `std::error::Error` value as an
    /// [`Error::Source`] without an explicit `Box::new` at the call
    /// site.
    pub fn source<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Source(Box::new(e))
    }
}

/// Convenient type alias used throughout the `sgp4` module.
pub type Result<T> = std::result::Result<T, Error>;
