use satkit::Frame;
use pyo3::prelude::*;

#[derive(Clone, PartialEq, Eq)]
#[pyclass(name = "frame", module = "satkit", eq, eq_int, from_py_object)]
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
    ///
    /// z = -r (nadir), y = -h (opposite angular momentum), x completes right-handed system
    #[allow(clippy::upper_case_acronyms)]
    LVLH,
    /// Radial / In-track / Cross-track (a.k.a. RSW in Vallado, RTN in CCSDS).
    ///
    /// R = radial (outward), I = in-track (perpendicular to R in the orbit
    /// plane, **not** strictly along velocity for eccentric orbits),
    /// C = cross-track (along angular momentum). For "along velocity"
    /// semantics on eccentric orbits, use [`PyFrame::NTW`] instead.
    #[allow(clippy::upper_case_acronyms)]
    RIC,
    /// Velocity-aligned orbital frame (Vallado §3.3).
    ///
    /// N = in-plane normal to velocity, T = tangent (along v̂),
    /// W = cross-track (along angular momentum). The natural frame for
    /// prograde/retrograde maneuvers: a pure +T delta-v of magnitude Δv
    /// adds *exactly* Δv to |v|.
    #[allow(clippy::upper_case_acronyms)]
    NTW,
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
            Frame::RIC => Self::RIC,
            Frame::NTW => Self::NTW,
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
            PyFrame::RIC => Self::RIC,
            PyFrame::NTW => Self::NTW,
        }
    }
}
