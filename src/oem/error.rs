//! Errors produced by the `oem` module.

use thiserror::Error;

use crate::Instant;

/// Errors that can occur while parsing, interpolating, building or writing
/// an OEM.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// KVN text is not an OEM, or one of its lines is malformed. `line` is
    /// the 1-based line number.
    #[error("OEM, line {line}: {message}")]
    Kvn { line: usize, message: String },

    /// A mandatory keyword is absent or empty.
    #[error("Missing required OEM field {0}")]
    MissingField(&'static str),

    /// A field is present but its text cannot be read as the expected type.
    #[error("Invalid OEM field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },

    /// The message holds no segment, or a segment holds no state vector.
    #[error("OEM has no {0}")]
    Empty(&'static str),

    /// The epochs of a segment's state vectors are not strictly increasing.
    #[error("OEM state vector epochs are not strictly increasing at {0}")]
    EpochOrder(Instant),

    /// `TIME_SYSTEM` is not one of UTC, TAI, TT, GPS, TDB or UT1 (for
    /// example a mission-elapsed or spacecraft-clock time).
    #[error("Unsupported OEM TIME_SYSTEM: {0} (supported: UTC, TAI, TT, GPS, TDB, UT1)")]
    UnsupportedTimeSystem(String),

    /// `INTERPOLATION` names a method other than Lagrange, Hermite or linear.
    #[error(
        "Unsupported OEM INTERPOLATION: {0} (supported: LAGRANGE, HERMITE, LINEAR); \
         choose a method explicitly instead"
    )]
    UnsupportedInterpolation(String),

    /// The interpolation degree is zero.
    #[error("OEM interpolation degree must be at least 1")]
    InvalidDegree,

    /// The requested time is outside the span a segment (or every segment
    /// of the message) can be interpolated over.
    #[error("{time} is outside the usable span of the OEM ({start} to {stop})")]
    OutOfRange {
        time: Instant,
        start: Instant,
        stop: Instant,
    },

    /// The sampling step of [`OEM::from_propagation`](super::OEM::from_propagation)
    /// is zero or negative.
    #[error("OEM sampling step must be positive")]
    InvalidStep,

    /// The propagation an OEM was to be built from cannot be interpolated.
    #[error(transparent)]
    Propagation(#[from] crate::orbitprop::Error),

    /// The text starts with neither `CCSDS_OEM_VERS` (KVN) nor an XML tag.
    #[error("Input is neither OEM KVN (starts with CCSDS_OEM_VERS) nor XML (starts with '<')")]
    UnrecognizedFormat,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Returned when input looks like XML but the `omm-xml` cargo feature is
    /// disabled.
    #[error("Input appears to be XML but the `omm-xml` feature is not enabled")]
    XmlFeatureDisabled,

    #[cfg(feature = "omm-xml")]
    #[error(transparent)]
    Xml(#[from] quick_xml::DeError),
}

/// Convenient type alias used throughout the `oem` module.
pub type Result<T> = std::result::Result<T, Error>;
