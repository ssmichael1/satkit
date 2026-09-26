use pyo3::prelude::*;
use satkit::Frame;

#[derive(Clone, PartialEq, Eq)]
/// Coordinate reference frame
///
/// Used to specify the frame for thrust vectors and maneuvers.
///
/// Available frames:
///
/// - ``GCRF`` - Geocentric Celestial Reference Frame (inertial)
/// - ``ITRF`` - International Terrestrial Reference Frame (Earth-fixed)
/// - ``TEME`` - True Equator Mean Equinox (SGP4 output frame)
/// - ``CIRS`` - Celestial Intermediate Reference System
/// - ``TIRS`` - Terrestrial Intermediate Reference System
/// - ``EME2000`` - Earth Mean Equator 2000
/// - ``ICRF`` - International Celestial Reference Frame
/// - ``LVLH`` - Local Vertical Local Horizontal: z = -r (nadir), y = -h (opposite angular momentum), x completes right-handed system
/// - ``RTN`` - Radial / Tangential / Normal (CCSDS OEM convention; also
///   exposed as ``RSW`` and ``RIC`` aliases for Vallado / older-NASA naming):
///   R = radial (outward), T = tangential (in-track), N = normal (cross-track)
/// - ``NTW`` - Normal-to-velocity / Tangent / Cross-track (velocity-aligned):
///   T = along velocity, N = in-plane perpendicular to v, W = cross-track
///
/// Example:
///
/// ```python
/// import satkit as sk
///
/// # Use RTN frame for in-track thrust (RSW and RIC are aliases and work too)
/// t = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)
/// ```
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
    /// Radial / Tangential / Normal — CCSDS OEM/OMM/ODM convention.
    ///
    /// Also known as RSW (Vallado) or RIC (older NASA / Clohessy-Wiltshire
    /// literature). R = radial (outward), T = tangential (perpendicular
    /// to R in the orbit plane — **not** strictly along velocity for
    /// eccentric orbits), N = normal (along angular momentum). For
    /// "along velocity" semantics on eccentric orbits, use
    /// [`PyFrame::NTW`] instead. Python-level aliases ``frame.RSW`` and
    /// ``frame.RIC`` resolve to the same variant as ``frame.RTN``.
    #[allow(clippy::upper_case_acronyms)]
    RTN,
    /// Velocity-aligned orbital frame (Vallado §3.3).
    ///
    /// N = in-plane normal to velocity, T = tangent (along v̂),
    /// W = cross-track (along angular momentum). The natural frame for
    /// prograde/retrograde maneuvers: a pure +T delta-v of magnitude Δv
    /// adds *exactly* Δv to |v|.
    #[allow(clippy::upper_case_acronyms)]
    NTW,
}

#[pymethods]
impl PyFrame {
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (&'static str, String))> {
        crate::pyutils::enum_reduce(slf.as_any(), "frame")
    }

    /// Python-level alias for ``frame.RTN`` — Vallado's name for the
    /// same Radial / S=(W×R) / W=(R×V) orbital frame. Resolves to the
    /// same enum value as ``frame.RTN``, so ``frame.RSW == frame.RTN``
    /// is True.
    #[classattr]
    #[allow(non_upper_case_globals)]
    const RSW: PyFrame = PyFrame::RTN;

    /// Python-level alias for ``frame.RTN`` — older NASA / Clohessy-
    /// Wiltshire name (Radial / In-track / Cross-track). Resolves to the
    /// same enum value as ``frame.RTN``, so ``frame.RIC == frame.RTN``
    /// is True. Kept for backward compatibility with code written
    /// against earlier satkit versions where `RIC` was the canonical
    /// name.
    #[classattr]
    #[allow(non_upper_case_globals)]
    const RIC: PyFrame = PyFrame::RTN;
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
            Frame::RTN => Self::RTN,
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
            PyFrame::RTN => Self::RTN,
            PyFrame::NTW => Self::NTW,
        }
    }
}
