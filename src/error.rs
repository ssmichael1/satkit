//! Top-level error type for the satkit crate.
//!
//! **Deprecated as of 0.17.0.** Prefer the module-scoped error types
//! (e.g. [`tle::Error`](crate::tle::Error),
//! [`orbitprop::Error`](crate::orbitprop::Error)) directly, and either
//! define a downstream `enum AppError` with the variants you actually
//! use, or use `anyhow` / `color_eyre` for application-level error
//! aggregation. This façade will be removed in a future release.
//!
//! The original use case was a single result type that lets `?` work
//! across modules without an outer enum:
//!
//! ```rust,ignore
//! fn do_thing() -> Result<(), satkit::Error> {
//!     let tle = satkit::TLE::from_url(url)?;             // tle::Error
//!     let states = satkit::orbitprop::propagate(...)?;   // orbitprop::Error
//!     Ok(())
//! }
//! ```

use thiserror::Error;

/// Top-level satkit error covering every public module-scoped error.
#[deprecated(
    since = "0.17.0",
    note = "use the module-scoped error types (e.g. `tle::Error`) and define a downstream error enum, or use `anyhow`/`color_eyre`"
)]
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Tle(#[from] crate::tle::Error),

    #[error(transparent)]
    Omm(#[from] crate::omm::Error),

    #[error(transparent)]
    Frames(#[from] crate::frames::Error),

    #[error(transparent)]
    Itrfcoord(#[from] crate::itrfcoord::Error),

    #[error(transparent)]
    Orbitprop(#[from] crate::orbitprop::Error),

    #[error(transparent)]
    Time(#[from] crate::time::InstantError),

    #[error(transparent)]
    Kepler(#[from] crate::kepler::Error),

    #[error(transparent)]
    Sgp4(#[from] crate::sgp4::Error),

    #[error(transparent)]
    Frametransform(#[from] crate::frametransform::Error),

    #[error(transparent)]
    SpaceWeather(#[from] crate::spaceweather::Error),

    #[error(transparent)]
    SolarCycleForecast(#[from] crate::solar_cycle_forecast::Error),

    #[error(transparent)]
    EarthOrientationParams(#[from] crate::earth_orientation_params::Error),

    #[error(transparent)]
    JplEphem(#[from] crate::jplephem::Error),

    #[error(transparent)]
    EarthGravity(#[from] crate::earthgravity::Error),

    #[error(transparent)]
    LpEphem(#[from] crate::lpephem::Error),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    UpdateData(#[from] crate::utils::update_data::Error),
}

/// Convenient type alias used throughout satkit.
#[deprecated(
    since = "0.17.0",
    note = "use the module-scoped error types (e.g. `tle::Error`) and define a downstream error enum, or use `anyhow`/`color_eyre`"
)]
#[allow(deprecated)]
pub type Result<T> = std::result::Result<T, Error>;
