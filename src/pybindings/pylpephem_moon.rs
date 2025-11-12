use super::pyutils;
use crate::lpephem::moon;
use crate::Instant;
use pyo3::prelude::*;

#[derive(PartialEq, Eq)]
#[pyclass(name = "moonphase", eq, eq_int)]
pub enum MoonPhase {
    NewMoon = moon::MoonPhase::NewMoon as isize,
    WaxingCrescent = moon::MoonPhase::WaxingCrescent as isize,
    FirstQuarter = moon::MoonPhase::FirstQuarter as isize,
    WaxingGibbous = moon::MoonPhase::WaxingGibbous as isize,
    FullMoon = moon::MoonPhase::FullMoon as isize,
    WaningGibbous = moon::MoonPhase::WaningGibbous as isize,
    LastQuarter = moon::MoonPhase::LastQuarter as isize,
    WaningCrescent = moon::MoonPhase::WaningCrescent as isize,
}

impl From<&MoonPhase> for moon::MoonPhase {
    fn from(p: &MoonPhase) -> Self {
        match p {
            MoonPhase::NewMoon => Self::NewMoon,
            MoonPhase::WaxingCrescent => Self::WaxingCrescent,
            MoonPhase::FirstQuarter => Self::FirstQuarter,
            MoonPhase::WaxingGibbous => Self::WaxingGibbous,
            MoonPhase::FullMoon => Self::FullMoon,
            MoonPhase::WaningGibbous => Self::WaningGibbous,
            MoonPhase::LastQuarter => Self::LastQuarter,
            MoonPhase::WaningCrescent => Self::WaningCrescent,
        }
    }
}

impl From<moon::MoonPhase> for MoonPhase {
    fn from(p: moon::MoonPhase) -> Self {
        match p {
            moon::MoonPhase::NewMoon => Self::NewMoon,
            moon::MoonPhase::WaxingCrescent => Self::WaxingCrescent,
            moon::MoonPhase::FirstQuarter => Self::FirstQuarter,
            moon::MoonPhase::WaxingGibbous => Self::WaxingGibbous,
            moon::MoonPhase::FullMoon => Self::FullMoon,
            moon::MoonPhase::WaningGibbous => Self::WaningGibbous,
            moon::MoonPhase::LastQuarter => Self::LastQuarter,
            moon::MoonPhase::WaningCrescent => Self::WaningCrescent,
        }
    }
}

/// Approximate Moon position in the GCRF Frame
///
/// Notes:
///   * From Vallado Algorithm 31
///   * Valid with accuracy of 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude, and 1275 km in range
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element numpy array or Nx3 numpy array representing moon position in GCRF frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_gcrf(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_arr(&moon::pos_gcrf, time)
}

/// Approximate Moon phase angle
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute phase
///
/// Returns:
///     float|numpy.ndarray: scalar or numpy array representing moon phase at input time[s].  Units are radians
#[pyfunction]
pub fn phase(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(moon::phase, time)
}

/// Moon phase name
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute phase name
///
/// Returns:
///     str|list: phase name string or list of phase name strings (e.g., "New Moon", "Waxing Crescent", etc.)
#[pyfunction]
pub fn phase_name(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(|t: &Instant| MoonPhase::from(moon::phase_name(t)), time)
}

/// Fraction of moon illuminated
///
/// Args:
///    time (satkit.time|numpy.ndarray|list): time[s] at which to
///
/// Returns:
///    float|numpy.ndarray: scalar or numpy array representing fraction of moon illuminated at input time[s].  Range is 0.0 to 1.0
#[pyfunction]
pub fn illumination(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(moon::illumination, time)
}
