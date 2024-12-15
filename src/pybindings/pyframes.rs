use crate::frames::Frame;
use pyo3::prelude::*;

#[derive(PartialEq, Eq)]
#[pyclass(name = "frame", module = "satkit", eq, eq_int)]
pub enum PyFrame {
    /// International Terrestrial Reference Frame
    #[allow(clippy::upper_case_acronyms)]
    ITRF,
    /// Terrestrial Intermediate Reference System
    #[allow(clippy::upper_case_acronyms)]
    TIRS,
    /// Celestial Intermediate Reference System
    #[allow(clippy::upper_case_acronyms)]
    CIRS,
    /// Geocentric Celestial Reference Frame
    #[allow(clippy::upper_case_acronyms)]
    GCRF,
    /// True Equator Mean Equinox
    #[allow(clippy::upper_case_acronyms)]
    TEME,
    /// Earth Mean Equator 2000
    EME2000,
    /// International Celestial Reference Frame
    #[allow(clippy::upper_case_acronyms)]
    ICRF,
    /// Local Vertical Local Horizontal
    #[allow(clippy::upper_case_acronyms)]
    LVLH,
}

impl From<Frame> for PyFrame {
    fn from(frame: Frame) -> Self {
        match frame {
            Frame::ITRF => Self::ITRF,
            Frame::TIRS => Self::TIRS,
            Frame::CIRS => Self::CIRS,
            Frame::GCRF => Self::GCRF,
            Frame::TEME => Self::TEME,
            Frame::EME2000 => Self::EME2000,
            Frame::ICRF => Self::ICRF,
            Frame::LVLH => Self::LVLH,
        }
    }
}

impl From<PyFrame> for Frame {
    fn from(frame: PyFrame) -> Self {
        match frame {
            PyFrame::ITRF => Self::ITRF,
            PyFrame::TIRS => Self::TIRS,
            PyFrame::CIRS => Self::CIRS,
            PyFrame::GCRF => Self::GCRF,
            PyFrame::TEME => Self::TEME,
            PyFrame::EME2000 => Self::EME2000,
            PyFrame::ICRF => Self::ICRF,
            PyFrame::LVLH => Self::LVLH,
        }
    }
}
