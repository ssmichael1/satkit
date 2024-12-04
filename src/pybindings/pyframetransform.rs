use super::pyutils::*;
use super::PyInstant;
use crate::frametransform as ft;
use pyo3::prelude::*;

/// Greenwich Mean Sidereal Time
///
/// Notes:
///     * Vallado algorithm 15:
///     * GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate GMST
///
/// Returns:
///     float|numpy.array: GMST at input time[s] in radians
#[pyfunction]
pub fn gmst(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::gmst, tm)
}

///
/// Equation of Equinoxes
///
#[pyfunction]
pub fn eqeq(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::eqeq, tm)
}

/// Greenwich apparant sidereal time, radians
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate GAST
///
/// Returns:
///     float|numpy.array: GAST at input time[s] in radians
#[pyfunction]
pub fn gast(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::gast, tm)
}

/// Earth Rotation Angle
///
///
/// Notes:
///     * See: IERS Technical Note 36, Chapter 5, Equation 5.15
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate Earth Rotation Angle
///
/// Returns:
///     float|numpy.array: Earth Rotation Angle at input time[s] in radians
///
/// Calculation Details
///
/// * Let t be UT1 Julian date
/// * let f be fractional component of t (fraction of day)
/// * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))
///
#[pyfunction]
pub fn earth_rotation_angle(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::earth_rotation_angle, tm)
}

/// Rotation from International Terrestrial Reference Frame (ITRF) to the Terrestrial Intermediate Reference System (TIRS)
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from ITRF to TIRS at input time[s]
#[pyfunction]
pub fn qitrf2tirs(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2tirs, tm)
}

/// Rotation from Terrestrial Intermediate Reference System to Celestial Intermediate Reference Systems
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from TIRS to CIRS at input time[s]
#[pyfunction]
pub fn qtirs2cirs(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qtirs2cirs, tm)
}

/// Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from CIRS to GCRF at input time[s]

#[pyfunction]
pub fn qcirs2gcrf(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qcirs2gcrs, tm)
}

///Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)
///
/// Notes:
///    * Uses full IAU2010 Reduction; See IERS Technical Note 36, Chapter 5
///    * Very computationally expensive
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from ITRF to GCRF at input time[s]

#[pyfunction]
pub fn qitrf2gcrf(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2gcrf, tm)
}

///Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)
///
/// Notes:
///     * Uses full IAU2010 Reduction; See IERS Technical Note 36, Chapter 5
///     * Very computationally expensive
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from GCRF to ITRF at input time[s]
#[pyfunction]
pub fn qgcrf2itrf(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qgcrf2itrf, tm)
}

/// Approximate rotation from Geocentric Celestrial Reference Frame to International Terrestrial Reference Frame
///
/// Notes:
///     * Uses an approximation of the IAU-76/FK5 Reduction; see Vallado section 3.7.3
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from GCRF to ITRF at input time[s]
#[pyfunction]
pub fn qgcrf2itrf_approx(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qgcrf2itrf_approx, tm)
}

/// Approximate rotation from International Terrestrial Reference Frame to Geocentric Celestrial Reference Frame
///
/// Notes:
///     * Uses an approximation of the IAU-76/FK5 Reduction; see Vallado section 3.7.3
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from ITRF to GCRF at input time[s]
#[pyfunction]
pub fn qitrf2gcrf_approx(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2gcrf_approx, tm)
}

/// Rotation from True Equator Mean Equinox (TEME) frame to International Terrestrial Reference Frame (ITRF)
///
/// Notes:
///     * TEME is output frame of SGP4 propagator
///     * This is Equation 3-90 in Vallado
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from TEME to ITRF at input time[s]
#[pyfunction]
pub fn qteme2itrf(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qteme2itrf, tm)
}

/// Rotation from True Equator Mean Equinox (TEME) frame to Geocentric Celestial Reference Frame (GCRF)
///
/// Notes:
///    * TEME is output frame of SGP4 propagator
///    * Approximate rotation from TEME to GCRF, accurate to 1 asec
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///    satkit.quaternion|list: Quaternion or list of quaternions representing rotation from TEME to GCRF at input time[s]
#[pyfunction]
pub fn qteme2gcrf(tm: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qteme2gcrf, tm)
}

///
/// Get Earth Orientation Parameters at given instant
///
/// Args:
///     tm (satkit.time):   Instant at which to query parameters
///
/// Returns:
///     (float, float, float, float, float, float): tuple with following elements:
///     * 0 : (UT1 - UTC) in seconds
///     * 1 : X polar motion in arcsecs
///     * 2 : Y polar motion in arcsecs
///     * 3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
///     * 4 : dX wrt IAU-2000A nutation, milli-arcsecs
///     * 5 : dY wrt IAU-2000A nutation, milli-arcsecs
#[pyfunction(name = "earth_orientation_params")]
pub fn pyeop(time: &PyInstant) -> Option<(f64, f64, f64, f64, f64, f64)> {
    crate::earth_orientation_params::get(&time.0).map(|r| (r[0], r[1], r[2], r[3], r[4], r[5]))
}
