use thiserror::Error;

#[derive(Error, Debug)]
pub enum SKErr {
    #[error("Error: {0}")]
    Error(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Format error: {0}")]
    FormatError(#[from] std::fmt::Error),
    #[error("Invalid Instant: {0}")]
    InvalidInstant(crate::Instant),
    #[error("Time Error: {0}")]
    TimeError(#[from] crate::time::InstantError),
    #[error("Kepler Error: {0}")]
    KeplerError(#[from] crate::kepler::KeplerError),
    #[error("Propagation Error: {0}")]
    PropagationError(#[from] crate::orbitprop::PropagationError),
    #[error("Parse Int Error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("Parse Float Error: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    #[error("ODE Error: {0}")]
    ODEError(#[from] crate::ode::ODEError),
    #[error("UTF-8 Error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("From UTF-8 Error: {0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
    #[error("Array From Slice error: {0}")]
    ArrayError(#[from] std::array::TryFromSliceError),
    #[error("ureq error: {0}")]
    UreqError(#[from] ureq::Error),
    #[error("JSON Error: {0}")]
    JsonError(#[from] json::Error),
}

pub type SKResult<T> = Result<T, SKErr>;

impl<T> From<crate::time::InstantError> for SKResult<T> {
    fn from(e: crate::time::InstantError) -> Self {
        Err(SKErr::TimeError(e))
    }
}

impl<T> From<crate::ode::ODEError> for SKResult<T> {
    fn from(e: crate::ode::ODEError) -> Self {
        Err(SKErr::ODEError(e))
    }
}

impl<T> From<SKErr> for SKResult<T> {
    fn from(e: SKErr) -> Self {
        Err(e)
    }
}

macro_rules! skerror {
    ($($arg:tt)*) => {
        Err(crate::SKErr::Error(format!($($arg)*)))
    };
}

pub(crate) use skerror;
