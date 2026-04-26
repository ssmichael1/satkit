//! Errors produced by the `sgp4` module.

use thiserror::Error;

/// Errors that can occur while initialising or evaluating SGP4.
#[derive(Debug, Error)]
pub enum Error {
    /// `sgp4init` returned a non-zero error code while constructing the
    /// internal `SatRec`. The numeric code matches the legacy Vallado
    /// convention (1 = eccentricity, 2 = mean motion, 3 = perturbed
    /// eccentricity, 4 = semi-latus rectum, 6 = orbit decay).
    #[error("SGP4 init error code {0}")]
    SatRecInit(i32),

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
