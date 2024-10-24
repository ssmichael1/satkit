use crate::frames::Frame;
use pyo3::prelude::*;

#[derive(PartialEq)]
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
            Frame::ITRF => PyFrame::ITRF,
            Frame::TIRS => PyFrame::TIRS,
            Frame::CIRS => PyFrame::CIRS,
            Frame::GCRF => PyFrame::GCRF,
            Frame::TEME => PyFrame::TEME,
            Frame::EME2000 => PyFrame::EME2000,
            Frame::ICRF => PyFrame::ICRF,
            Frame::LVLH => PyFrame::LVLH,
        }
    }
}

impl From<PyFrame> for Frame {
    fn from(frame: PyFrame) -> Self {
        match frame {
            PyFrame::ITRF => Frame::ITRF,
            PyFrame::TIRS => Frame::TIRS,
            PyFrame::CIRS => Frame::CIRS,
            PyFrame::GCRF => Frame::GCRF,
            PyFrame::TEME => Frame::TEME,
            PyFrame::EME2000 => Frame::EME2000,
            PyFrame::ICRF => Frame::ICRF,
            PyFrame::LVLH => Frame::LVLH,
        }
    }
}
