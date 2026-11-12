use pyo3::prelude::*;

use crate::pysolarsystem;
use crate::pyutils::*;
use satkit::jplephem;
use satkit::mathtypes::*;
use satkit::Instant;
use satkit::SolarSystem;

use anyhow::Result;

/// Return the position and velocity of the given body in Geocentric coordinate system (GCRF)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|datetime.datetime|numpy.datetime64|list|tuple|numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position
///
/// Returns:
///     tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays
///
/// Raises:
///     RuntimeError: if a time is outside the span of the loaded ephemeris
///         ("Invalid Julian date"; about 1550 to 2650 for DE440), or the
///         ephemeris file is unavailable
///     TypeError: if tm is not a time or a list / array of times
#[pyfunction]
pub fn geocentric_state(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let rbody: SolarSystem = body.into();
    let f =
        |tm: &Instant| -> Result<(Vector3, Vector3)> { Ok(jplephem::geocentric_state(rbody, tm)?) };
    tuple_func_of_time_arr(f, tm)
}

/// Return the position & velocity the given body in the barycentric coordinate system (origin is solar system barycenter)
///
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|datetime.datetime|numpy.datetime64|list|tuple|numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position
///
/// Returns:
///     tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays
///
/// Raises:
///     RuntimeError: if a time is outside the span of the loaded ephemeris
///         ("Invalid Julian date"; about 1550 to 2650 for DE440), or the
///         ephemeris file is unavailable
///     TypeError: if tm is not a time or a list / array of times
///
/// Notes:
///  * Every body, the Moon included, is returned relative to the solar system
///    barycenter, in the ICRF axes. (The DE files tabulate the Moon
///    geocentrically; it is combined with the Earth-Moon barycenter here.
///    Releases up to 0.24.3 returned the geocentric Moon from this function:
///    use ``geocentric_pos`` / ``geocentric_state`` for that.)
///  * EMB is the Earth-Moon barycenter; there is no ``Earth`` member. The
///    Earth is ``barycentric(EMB) - geocentric(Moon) / (1 + EMRAT)``, with
///    ``EMRAT = satkit.jplephem.consts("EMRAT")``, or equivalently
///    ``barycentric(body) - geocentric(body)`` for any body
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
#[pyfunction]
pub fn barycentric_state(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<(Vector3, Vector3)> {
        Ok(jplephem::barycentric_state(rbody, tm)?)
    };
    tuple_func_of_time_arr(f, tm)
}

/// Return the position of the given body in the GCRF coordinate system (origin is Earth center)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|datetime.datetime|numpy.datetime64|list|tuple|numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position
///
/// Returns:
///     numpy.ndarray: 3-vector of cartesian Geocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array
///
/// Raises:
///     RuntimeError: if a time is outside the span of the loaded ephemeris
///         ("Invalid Julian date"; about 1550 to 2650 for DE440), or the
///         ephemeris file is unavailable
///     TypeError: if tm is not a time or a list / array of times
#[pyfunction]
pub fn geocentric_pos(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<Vector3> { Ok(jplephem::geocentric_pos(rbody, tm)?) };
    py_vec3_of_time_result_arr(&f, tm)
}

/// Return the position of the given body in the Barycentric coordinate system (origin is solarsystem barycenter)
///
/// Args:
///     body (satkit.solarsystem): Solar system body for which to return position
///     tm (satkit.time|datetime.datetime|numpy.datetime64|list|tuple|numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position
///
/// Returns:
///     numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the solar system barycenter. If input is list or numpy array of N times, then r will be Nx3 array
///
/// Raises:
///     RuntimeError: if a time is outside the span of the loaded ephemeris
///         ("Invalid Julian date"; about 1550 to 2650 for DE440), or the
///         ephemeris file is unavailable
///     TypeError: if tm is not a time or a list / array of times
///
/// Notes:
///  * Every body, the Moon included, is returned relative to the solar system
///    barycenter, in the ICRF axes. (The DE files tabulate the Moon
///    geocentrically; it is combined with the Earth-Moon barycenter here.
///    Releases up to 0.24.3 returned the geocentric Moon from this function:
///    use ``geocentric_pos`` / ``geocentric_state`` for that.)
///  * EMB is the Earth-Moon barycenter; there is no ``Earth`` member. The
///    Earth is ``barycentric(EMB) - geocentric(Moon) / (1 + EMRAT)``, with
///    ``EMRAT = satkit.jplephem.consts("EMRAT")``, or equivalently
///    ``barycentric(body) - geocentric(body)`` for any body
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
#[pyfunction]
pub fn barycentric_pos(
    body: &pysolarsystem::SolarSystem,
    tm: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let rbody: SolarSystem = body.into();
    let f = |tm: &Instant| -> Result<Vector3> { Ok(jplephem::barycentric_pos(rbody, tm)?) };
    py_vec3_of_time_result_arr(&f, tm)
}

/// Return a named constant from the loaded JPL ephemeris file
///
/// The DE ephemeris files carry a table of constants used in their
/// construction — e.g. "AU" (astronomical unit, km), "EMRAT" (Earth/Moon
/// mass ratio), "GM1".."GM9" and "GMS"/"GMB" (GM values in au^3/day^2).
///
/// Args:
///     name (str): Constant name (case-sensitive, as in the DE file header)
///
/// Returns:
///     float | None: The constant's value, or None if the name is not present
///     (or the ephemeris file could not be loaded)
#[pyfunction]
pub fn consts(name: &str) -> Option<f64> {
    jplephem::consts(name).copied()
}
