//! Errors produced by the `omm` module.

use std::num::{ParseFloatError, ParseIntError};

use thiserror::Error;

/// Errors that can occur while parsing or using OMM messages.
#[derive(Debug, Error)]
pub enum Error {
    #[error("Missing required field {0}")]
    MissingField(&'static str),

    #[error("Invalid float for {field}: {source}")]
    InvalidFloatField {
        field: &'static str,
        #[source]
        source: ParseFloatError,
    },

    #[error("Invalid float value: {0}")]
    InvalidFloat(#[from] ParseFloatError),

    #[error("Invalid integer value: {0}")]
    InvalidInt(#[from] ParseIntError),

    /// Raised when [`OMM::epoch_instant`](crate::omm::OMM::epoch_instant)
    /// fails to parse `epoch` as RFC 3339.
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),

    #[error("Unsupported MEAN_ELEMENT_THEORY: {0}")]
    UnsupportedMeanElementTheory(String),

    #[error("Unsupported TIME_SYSTEM for SGP4: {0}")]
    UnsupportedTimeSystem(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),

    /// Returned by [`OMM::from_url`](crate::omm::OMM::from_url) when the
    /// response looks like XML but the `omm-xml` cargo feature is disabled.
    #[error("Response appears to be XML but the `omm-xml` feature is not enabled")]
    XmlFeatureDisabled,

    #[cfg(feature = "omm-xml")]
    #[error(transparent)]
    Xml(#[from] quick_xml::DeError),
}

/// Convenient type alias used throughout the `omm` module.
pub type Result<T> = std::result::Result<T, Error>;
