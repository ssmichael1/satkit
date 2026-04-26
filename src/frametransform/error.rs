//! Errors produced by the `frametransform` module.

use std::num::{ParseFloatError, ParseIntError};

use thiserror::Error;

use crate::Frame;

/// Errors produced by the
/// [`frametransform`](crate::frametransform) module.
#[derive(Debug, Error)]
pub enum Error {
    /// [`to_gcrf`](super::to_gcrf) and [`from_gcrf`](super::from_gcrf)
    /// only build rotation matrices for satellite-local orbital frames
    /// (`GCRF`, `LVLH`, `RTN`, `NTW`). Time-dependent inertial /
    /// Earth-fixed frames must use the dedicated quaternion helpers
    /// ([`qitrf2gcrf`](super::qitrf2gcrf),
    /// [`qteme2gcrf`](super::qteme2gcrf), …).
    #[error(
        "to_gcrf: frame {frame} is not a satellite-local orbital frame; use the \
         time-based quaternion helpers (qitrf2gcrf, qteme2gcrf, etc.) instead"
    )]
    UnsupportedFrame { frame: Frame },

    /// A `j = N` table-definition line in an IERS table file is
    /// malformed.
    #[error("Error parsing file {fname}, invalid table definition line")]
    InvalidIersTableDef { fname: String },

    /// Encountered a coefficient row in an IERS table file before any
    /// table dimension was declared.
    #[error("Error parsing file {fname}, table not initialized")]
    IersTableNotInitialized { fname: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseInt(#[from] ParseIntError),

    #[error(transparent)]
    ParseFloat(#[from] ParseFloatError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the `frametransform` module.
pub type Result<T> = std::result::Result<T, Error>;
