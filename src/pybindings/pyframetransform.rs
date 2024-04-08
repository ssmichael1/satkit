use super::pyutils::*;
use super::PyAstroTime;
use crate::frametransform as ft;
use pyo3::prelude::*;

///
/// Greenwich Mean Sidereal Time
///
/// Vallado algorithm 15:
///
/// GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)
///
///
/// # Arguments
///
///   * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///           representing time at which to calculate output
/// 
/// # Returns
///
/// * Greenwich Mean Sideral Time, radians, at intput time(s)
///
#[pyfunction]
pub fn gmst(tm: &PyAny) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::gmst, tm)
}

///
/// Equation of Equinoxes
///
#[pyfunction]
pub fn eqeq(tm: &PyAny) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::eqeq, tm)
}

///
/// Greenwich apparant sidereal time, radians
///
/// # Arguments:
///
///   * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///           representing time at which to calculate output
///
/// # Returns:
///
///  * Greenwich apparant sidereal time, radians, at input time(s)
///
#[pyfunction]
pub fn gast(tm: &PyAny) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::gast, tm)
}

///
/// Earth Rotation Angle
///
/// See
/// [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf?__blob=publicationFile&v=1)
/// Equation 5.15
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
///
/// # Returns:
///
///  * Earth rotation angle, in radians, at input time(s)
///
/// # Calculation Details
///
/// * Let t be UT1 Julian date
/// * let f be fractional component of t (fraction of day)
/// * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t − 2451545.0))
///
#[pyfunction]
pub fn earth_rotation_angle(tm: &PyAny) -> PyResult<PyObject> {
    py_func_of_time_arr(ft::earth_rotation_angle, tm)
}

///
/// Rotation from International Terrestrial Reference Frame
/// (ITRF) to the Terrestrial Intermediate Reference System (TIRS)
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
/// 
/// # Returns:
///
///  * Quaternion representing rotation from ITRF to TIRS at input time(s)
///
#[pyfunction]
pub fn qitrf2tirs(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2tirs, tm)
}

///
/// Rotation from Terrestrial Intermediate Reference System to
/// Celestial Intermediate Reference Systems
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
/// 
/// # Returns:
///
///  * Quaternion representing rotation from TIRS to CIRS at input time(s)
///
#[pyfunction]
pub fn qtirs2cirs(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qtirs2cirs, tm)
}

///
/// Rotation from Celestial Intermediate Reference System
/// to Geocentric Celestial Reference Frame
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
/// 
/// # Returns:
///
///  * Quaternion representing rotation from CIRS to GCRF at input time(s)
///
#[pyfunction]
pub fn qcirs2gcrf(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qcirs2gcrs, tm)
}

///
/// Quaternion representing rotation from the
/// International Terrestrial Reference Frame (ITRF)
/// to the Geocentric Celestial Reference Frame (GCRF)
///
/// Uses full IAU2006 Reduction
/// See IERS Technical Note 36, Chapter 5
///
/// Note: Very computationally expensive
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
///
/// # Returns:
///
///  * Quaternion representing rotation from ITRF to GCRF at input time(s)
///
#[pyfunction]
pub fn qitrf2gcrf(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2gcrf, tm)
}

///
/// Quaternion representing rotation from the
/// Geocentric Celestial Reference Frame (GCRF)
/// to the International Terrestrial Reference Frame (ITRF)
///
/// Uses full IAU2006 Reduction
/// See IERS Technical Note 36, Chapter 5
///
/// Note: Very computationally expensive
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
///
/// # Returns:
///
///  * Quaternion representing rotation from GCRF to ITRF at input time(s)
///
#[pyfunction]
pub fn qgcrf2itrf(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qgcrf2itrf, tm)
}

///
/// Approximate rotation from
/// Geocentric Celestrial Reference Frame to
/// International Terrestrial Reference Frame
///
/// This uses an approximation of the IAU-76/FK5 Reduction
/// See Vallado section 3.7.3
///
/// # Arguments:
///
///  * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
///          representing time at which to calculate output
///
/// # Returns:
///
///  * Quaternion representing rotation from GCRF to ITRF at input time(s)
///
#[pyfunction]
pub fn qgcrf2itrf_approx(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qgcrf2itrf_approx, tm)
}

#[pyfunction]
pub fn qitrf2gcrf_approx(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qitrf2gcrf_approx, tm)
}

///
/// Rotation from True Equator Mean Equinox (TEME) frame
/// to International Terrestrial Reference Frame (ITRF)
///
/// Note: TEME is output frame of SGP4 propagator
///
/// This is Equation 3-90 in Vallado
///
/// # Arguments:
///
///  * tm: astro.time struct representing input time
///
/// # Returns:
///
///  * Quaternion representing rotation from TEME to ITRF
///
#[pyfunction]
pub fn qteme2itrf(tm: &PyAny) -> PyResult<PyObject> {
    py_quat_from_time_arr(ft::qteme2itrf, tm)
}

///
/// Get Earth Orientation Parameters at given instant
///
/// # Arguments:
///
/// * tm: Instant at which to query parameters
///
/// # Return:
///
/// * Vector [f64; 4] with following elements:
///   * 0 : (UT1 - UTC) in seconds
///   * 1 : X polar motion in arcsecs
///   * 2 : Y polar motion in arcsecs
///   * 3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
///   * 4 : dX wrt IAU-2000A nutation, milli-arcsecs
///   * 5 : dY wrt IAU-2000A nutation, milli-arcsecs
///
#[pyfunction(name = "earth_orientation_params")]
pub fn pyeop(time: &PyAstroTime) -> Option<(f64, f64, f64, f64, f64, f64)> {
    match crate::earth_orientation_params::get(&time.inner) {
        None => None,
        Some(r) => Some((r[0], r[1], r[2], r[3], r[4], r[5])),
    }
}
