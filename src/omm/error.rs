//! Errors produced by the `omm` module.

use thiserror::Error;

/// Errors that can occur while parsing or using OMM messages.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A mandatory field is absent or empty.
    #[error("Missing required OMM field {0}")]
    MissingField(&'static str),

    /// A field is present but its text cannot be parsed as the expected type.
    #[error("Invalid OMM field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },

    /// Raised when the `EPOCH` string cannot be parsed as RFC 3339.
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),

    /// `MEAN_ELEMENT_THEORY` names a theory other than SGP4.
    #[error("Unsupported MEAN_ELEMENT_THEORY: {0}")]
    UnsupportedMeanElementTheory(String),

    /// `TIME_SYSTEM` is something other than UTC.
    #[error("Unsupported TIME_SYSTEM for SGP4: {0}")]
    UnsupportedTimeSystem(String),

    /// `EPHEMERIS_TYPE` denotes elements that classic SGP4 cannot propagate
    /// (currently type 4, SGP4-XP).
    #[error(
        "Unsupported EPHEMERIS_TYPE {0}: satkit implements classic SGP4 only (type 4 is SGP4-XP)"
    )]
    UnsupportedEphemerisType(u8),

    /// The JSON document is neither an OMM object nor an array of them.
    #[error("Expected a JSON OMM object or an array of OMM objects, found {0}")]
    UnexpectedJsonShape(&'static str),

    /// Text handed to [`OMM::from_text`](crate::omm::OMM::from_text) starts
    /// with neither a JSON bracket nor an XML tag.
    #[error("Input is neither JSON (starts with '[' or '{{') nor XML (starts with '<'); KVN is not supported")]
    UnrecognizedFormat,

    #[error(transparent)]
    Json(#[from] serde_json::Error),

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

    /// Returned when input looks like XML but the `omm-xml` cargo feature is
    /// disabled.
    #[error("Input appears to be XML but the `omm-xml` feature is not enabled")]
    XmlFeatureDisabled,

    #[cfg(feature = "omm-xml")]
    #[error(transparent)]
    Xml(#[from] quick_xml::DeError),
}

/// Convenient type alias used throughout the `omm` module.
pub type Result<T> = std::result::Result<T, Error>;
