use pyo3::prelude::*;

use super::pysolarsystem;
use super::pyutils::*;
use crate::jplephem;
use crate::solarsystem::SolarSystem;
use crate::Instant;
use nalgebra as na;

use anyhow::Result;

/// Return the position and velocity of the given body in Geocentric coordinate system (GCRF)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|numpy.ndarray|list): Time(s) at which to return position
///
/// Returns:
///     tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays
#[pyfunction]
pub fn geocentric_state(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<(na::Vector3<f64>, na::Vector3<f64>)> {
        jplephem::geocentric_state(rbody, tm)
    };
    tuple_func_of_time_arr(f, tm)
}

/// Return the position & velocity the given body in the barycentric coordinate system (origin is solar system barycenter)
///
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|numpy.ndarray|list): Time(s) at which to return position
///
/// Returns:
///     tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays
///
/// Notes:
///  * Positions for all bodies are natively relative to solar system barycenter,
///    with exception of moon, which is computed in Geocentric system
///  * EMB (2) is the Earth-Moon barycenter
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
///
#[pyfunction]
pub fn barycentric_state(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<(na::Vector3<f64>, na::Vector3<f64>)> {
        jplephem::barycentric_state(rbody, tm)
    };
    tuple_func_of_time_arr(f, tm)
}

/// Return the position of the given body in the GCRF coordinate system (origin is Earth center)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|numpy.ndarray|list): Time(s) at which to return position
///
/// Returns:
///     numpy.ndarray: 3-vector of cartesian Geocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array
#[pyfunction]
pub fn geocentric_pos(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> Result<PyObject> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<na::Vector3<f64>> { jplephem::geocentric_pos(rbody, tm) };
    py_vec3_of_time_result_arr(&f, tm)
}

/// Return the position of the given body in the Barycentric coordinate system (origin is solarsystem barycenter)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|numpy.ndarray|list): Time(s) at which to return position
///
/// Returns:
///     numpy.ndarray: 3-vector of cartesian Heliocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array
///
/// Notes:
///  * Positions for all bodies are natively relative to solar system barycenter,
///    with exception of moon, which is computed in Geocentric system
///  * EMB (2) is the Earth-Moon barycenter
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
#[pyfunction]
pub fn barycentric_pos(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> Result<PyObject> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<na::Vector3<f64>> { jplephem::barycentric_pos(rbody, tm) };
    py_vec3_of_time_result_arr(&f, tm)
}
