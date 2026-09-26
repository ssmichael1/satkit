use crate::pysolarsystem::SolarSystem;
use crate::pyutils;
use pyo3::prelude::*;
use satkit::lpephem;

use anyhow::Result;

/// Approximate Heliocentric position of a planet
/// in ICRF frame but centered on sun not solar system barycenter
///
/// Notes:
///  * See: <https://ssd.jpl.nasa.gov/?planet_pos>
///
/// Args:
///    planet (satkit.solarsystem): solar system body (the Sun or one of the 8 planets)
///    time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///   numpy.ndarray: 3-element numpy array for a single time, or Nx3 numpy
///   array for a list / array of N times, representing planet position in ICRF frame
///   at input time[s].  Units are meters
#[pyfunction]
pub fn heliocentric_pos(planet: &SolarSystem, time: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(
        &|t| lpephem::heliocentric_pos(planet.into(), t).map_err(anyhow::Error::from),
        time,
    )
}
