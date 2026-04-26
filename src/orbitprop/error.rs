//! Errors produced by the `orbitprop` module.

use numeris::ode;
use thiserror::Error;

use crate::Frame;

/// Errors that can occur while configuring or executing orbit propagation.
#[derive(Debug, Error)]
pub enum Error {
    // -- propagator-internal errors --------------------------------------
    /// Returned when the integrated state matrix has an unexpected
    /// number of columns.
    #[error("Invalid number of columns: {c}")]
    InvalidStateColumns { c: usize },

    /// Returned by the dense-output interp helpers when the underlying
    /// ODE solution does not carry interpolation data.
    #[error("No Dense Output in Solution")]
    NoDenseOutputInSolution,

    /// Wraps an [`ode::OdeError`] surfaced by the chosen integrator.
    /// `OdeError` does not implement `std::error::Error` (numeris keeps
    /// it as a plain `Display`-only enum), so this variant is built
    /// manually rather than via `#[from]`.
    #[error("ODE Error: {0}")]
    OdeError(ode::OdeError),

    /// RODAS4 does not support state transition matrix propagation
    /// (`C == 7`).
    #[error("RODAS4 does not support state transition matrix propagation")]
    RODAS4NoSTM,

    /// Gauss-Jackson 8 does not support state transition matrix
    /// propagation (`C == 7`).
    #[error("Gauss-Jackson 8 does not support state transition matrix propagation")]
    GaussJackson8NoSTM,

    // -- precomputed.rs --------------------------------------------------
    /// Returned by [`Precomputed::interp`](crate::orbitprop::Precomputed::interp)
    /// when the requested time falls outside the precomputed range.
    #[error(
        "Precomputed::interp: time {time} is outside of precomputed range : {begin} to {end}"
    )]
    PrecomputedOutOfRange {
        time: String,
        begin: String,
        end: String,
    },

    /// Wraps an error surfaced while building a
    /// [`Precomputed`](crate::orbitprop::Precomputed) interp table.
    /// Currently captures stringified errors from the still-anyhow
    /// `jplephem` module; will become a typed variant when that module
    /// is migrated in Phase 3.
    #[error("Cannot compute precomputed interpolation data: {0}")]
    Precompute(String),

    // -- satstate.rs -----------------------------------------------------
    /// Returned by [`SatState::set_pos_uncertainty`](crate::orbitprop::SatState::set_pos_uncertainty),
    /// [`SatState::set_vel_uncertainty`](crate::orbitprop::SatState::set_vel_uncertainty),
    /// and the internal `cov_frame_to_gcrf` helper when the supplied
    /// frame is not one of the supported orbital or inertial frames.
    #[error("Unsupported frame for uncertainty: {frame}. Must be GCRF, LVLH, RIC, or NTW")]
    UnsupportedUncertaintyFrame { frame: Frame },

    // -- settings.rs -----------------------------------------------------
    /// Returned by [`PropSettings::set_gravity`](crate::orbitprop::PropSettings::set_gravity)
    /// when `order > degree`.
    #[error("Gravity order ({order}) must be ≤ degree ({degree})")]
    InvalidGravityOrder { order: u16, degree: u16 },
}

impl From<ode::OdeError> for Error {
    fn from(e: ode::OdeError) -> Self {
        Self::OdeError(e)
    }
}

/// Convenient type alias used throughout the `orbitprop` module.
pub type Result<T> = std::result::Result<T, Error>;
