use super::pyutils;
use crate::lpephem;
use crate::pybindings::pysolarsystem::SolarSystem;
use pyo3::prelude::*;

use anyhow::Result;

/// Approximate Heliocentric position of a planets
/// in ICRF frame but centered on sun not solar system barycenter
///  
/// Notes:
///  * See: <https://ssd.jpl.nasa.gov/?planet_pos>
///
/// Args:
///    time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///    planet (str): planet name
///
/// Returns:
///   numpy.ndarray: 3-element numpy array or Nx3 numpy
///   array representing planet position in ICRF frame at input time[s].  Units are meters
#[pyfunction]
pub fn heliocentric_pos(planet: &SolarSystem, time: &Bound<'_, PyAny>) -> Result<PyObject> {
    pyutils::py_vec3_of_time_arr(
        &|t| lpephem::heliocentric_pos(planet.into(), t).unwrap(),
        time,
    )
}
