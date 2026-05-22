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

    /// [`rotation`](super::rotation) and friends do not handle the
    /// orbit-dependent frames ([`Frame::LVLH`], [`Frame::RTN`],
    /// [`Frame::NTW`]) — they require the satellite's position and velocity
    /// to define their axes. Use [`to_gcrf`](super::to_gcrf) /
    /// [`from_gcrf`](super::from_gcrf) for those.
    #[error(
        "rotation: frame pair ({from}, {to}) involves an orbit-dependent frame; \
         use to_gcrf / from_gcrf with pos and vel"
    )]
    OrbitFrameRequiresState { from: Frame, to: Frame },

    /// [`rotation_approx`](super::rotation_approx) is only valid between
    /// ITRF and the inertial cluster (GCRF, EME2000, ICRF, TEME). The
    /// intermediate frames [`Frame::TIRS`] and [`Frame::CIRS`] are defined
    /// by the IERS 2010 reduction and have no FK5 analogue.
    #[error(
        "rotation_approx: frame {frame} has no FK5 approximate-reduction \
         analogue; use rotation() for full IERS 2010"
    )]
    ApproxNotSupportedForFrame { frame: Frame },

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
