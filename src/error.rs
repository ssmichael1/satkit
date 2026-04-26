//! Top-level error type for the satkit crate.
//!
//! Most functions return module-scoped error types
//! (e.g. [`tle::Error`](crate::tle::Error),
//! [`orbitprop::Error`](crate::orbitprop::Error)). For downstream apps
//! that consume multiple modules and don't want to define their own
//! outer error, this façade has `From` impls for every public module
//! error and can be used as a single result type.
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
pub type Result<T> = std::result::Result<T, Error>;
