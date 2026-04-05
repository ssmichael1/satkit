use crate::pyutils::*;
use crate::PyInstant;
use satkit::frametransform as ft;
use satkit::mathtypes::*;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use numpy as np;
use numpy::PyArrayMethods;

use anyhow::Result;

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
pub fn gmst(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    py_func_of_time_arr(ft::gmst, tm)
}

///
/// Equation of Equinoxes
///
#[pyfunction]
pub fn eqeq(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn gast(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn earth_rotation_angle(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qitrf2tirs(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qtirs2cirs(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qcirs2gcrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qitrf2gcrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qgcrf2itrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qgcrf2itrf_approx(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qitrf2gcrf_approx(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qteme2itrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
pub fn qteme2gcrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
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
///
///     Or None if the time is outside the range of available Earth Orientation Parameters (EOP)
///    (EOP are only available from 1962 to current, and predict to current + ~ 4 months)
///
#[pyfunction(name = "earth_orientation_params")]
pub fn pyeop(time: &PyInstant) -> Option<(f64, f64, f64, f64, f64, f64)> {
    satkit::earth_orientation_params::get(&time.0).map(|r| (r[0], r[1], r[2], r[3], r[4], r[5]))
}

///
/// Disable warning about out-of-range Earth Orientation Parameters (EOP)
///
/// Warning is shown only once, but to prevent it from being shown,
/// run this function.
///
/// # Example
/// ```python
/// import satkit
/// satkit.frametransform.disable_eop_warning()
/// ```
///
/// Return the DCM that transforms a 3-vector from the given satellite-
/// local frame into GCRF at the current state.
///
/// Supported frames:
///
/// * ``frame.GCRF`` — returns the 3x3 identity matrix
/// * ``frame.LVLH`` — Local Vertical / Local Horizontal
/// * ``frame.RTN``  — Radial / Tangential / Normal (= RSW = RIC)
/// * ``frame.NTW``  — Normal-to-velocity / Tangent / Cross-track
///
/// For arbitrary frame-to-frame rotation, compose with ``from_gcrf``::
///
///     # NTW -> RIC
///     dcm = sk.frametransform.from_gcrf(sk.frame.RTN, pos, vel) @ \
///           sk.frametransform.to_gcrf(sk.frame.NTW, pos, vel)
///
/// Args:
///     frame (satkit.frame): Source satellite-local frame
///     pos (numpy.ndarray): 3-element position vector in GCRF [m]
///     vel (numpy.ndarray): 3-element velocity vector in GCRF [m/s]
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix (frame -> GCRF)
///
/// Raises:
///     RuntimeError: if the frame is not a satellite-local orbital frame.
///         Inertial / Earth-fixed frames (ITRF, TEME, EME2000, etc.) need
///         the time-based quaternion helpers instead (qitrf2gcrf,
///         qteme2gcrf, ...).
#[pyfunction]
pub fn to_gcrf(
    frame: crate::pyframes::PyFrame,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let pos_vec: Vector3 = py_to_smatrix(pos)?;
    let vel_vec: Vector3 = py_to_smatrix(vel)?;
    let rust_frame: satkit::Frame = frame.into();
    let dcm = ft::to_gcrf(rust_frame, &pos_vec, &vel_vec)?;
    pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
        let arr = np::PyArray1::from_slice(py, dcm.as_slice());
        Ok(arr.reshape(vec![3, 3])?.into_py_any(py)?)
    })
}

/// Return the DCM that transforms a 3-vector from GCRF into the given
/// satellite-local frame at the current state.
///
/// Transpose of ``to_gcrf``. See that function's docs for supported
/// frames, error conditions, and composition examples.
///
/// Args:
///     frame (satkit.frame): Destination satellite-local frame
///     pos (numpy.ndarray): 3-element position vector in GCRF [m]
///     vel (numpy.ndarray): 3-element velocity vector in GCRF [m/s]
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix (GCRF -> frame)
///
/// Raises:
///     RuntimeError: if the frame is not a satellite-local orbital frame.
#[pyfunction]
pub fn from_gcrf(
    frame: crate::pyframes::PyFrame,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let pos_vec: Vector3 = py_to_smatrix(pos)?;
    let vel_vec: Vector3 = py_to_smatrix(vel)?;
    let rust_frame: satkit::Frame = frame.into();
    let dcm = ft::from_gcrf(rust_frame, &pos_vec, &vel_vec)?;
    pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
        let arr = np::PyArray1::from_slice(py, dcm.as_slice());
        Ok(arr.reshape(vec![3, 3])?.into_py_any(py)?)
    })
}

/// Rotation from the Mean-of-Date frame (MOD) to the Geocentric Celestial
/// Reference Frame (GCRF). Accounts for precession but not nutation.
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing
///     rotation from MOD to GCRF at input time[s]
#[pyfunction]
pub fn qmod2gcrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    py_quat_from_time_arr(ft::qmod2gcrf, tm)
}

/// Approximate rotation from True-of-Date (TOD) to Mean-of-Date (MOD).
/// Accounts for nutation only.
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing
///     rotation from TOD to MOD at input time[s]
#[pyfunction]
pub fn qtod2mod_approx(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    py_quat_from_time_arr(ft::qtod2mod_approx, tm)
}

/// Transform a satellite state (position + velocity) from ITRF to GCRF.
///
/// Unlike the raw :func:`qitrf2gcrf` quaternion, this function correctly
/// handles the Earth-rotation contribution to velocity: a point at rest
/// on Earth's surface has zero velocity in ITRF but ~465 m/s in GCRF
/// (at the equator), and this function accounts for that term.
///
/// The IAU 2010 ITRF → GCRF reduction decomposes into three stages:
/// polar motion (ITRF → TIRS), Earth rotation about the CIO polar axis
/// (TIRS → CIRS), and precession-nutation (CIRS → GCRF). The
/// Earth-rotation sweep term ``omega_earth x r`` is computed in **TIRS**
/// — not ITRF or GCRF — because TIRS is defined such that Earth's
/// rotation axis is exactly along its +z axis. Computing the sweep
/// anywhere else would introduce either a polar-motion-sized error
/// (~0.3 arcsec in ITRF) or a precession-sized error (tens of degrees
/// in GCRF).
///
/// Implementation steps:
///
/// 1. Rotate position and velocity from ITRF to TIRS via polar motion.
/// 2. Add ``omega_earth x r_tirs`` to the velocity in TIRS (where ``omega_earth``
///    is exactly ``(0, 0, OMEGA_EARTH)``).
/// 3. Rotate TIRS → CIRS → GCRF via the full IAU 2010 chain.
///
/// Uses the full IAU 2010 reduction (includes polar motion, Earth
/// rotation, precession-nutation with dX/dY corrections from Earth
/// orientation parameters).
///
/// Args:
///     pos_itrf (array-like): 3-element position vector in ITRF [m]
///     vel_itrf (array-like): 3-element velocity vector *as observed in
///         ITRF* [m/s] (zero for a point at rest on Earth)
///     time (satkit.time): Epoch of the state
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_gcrf, vel_gcrf)`` of
///     the state expressed in GCRF.
#[pyfunction]
pub fn itrf_to_gcrf_state(
    pos_itrf: &Bound<'_, PyAny>,
    vel_itrf: &Bound<'_, PyAny>,
    time: &PyInstant,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let pos_vec: Vector3 = py_to_smatrix(pos_itrf)?;
    let vel_vec: Vector3 = py_to_smatrix(vel_itrf)?;
    let (pos_gcrf, vel_gcrf) = ft::itrf_to_gcrf_state(&pos_vec, &vel_vec, &time.0);
    pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
        let p = np::PyArray1::from_slice(py, pos_gcrf.as_slice()).into_py_any(py)?;
        let v = np::PyArray1::from_slice(py, vel_gcrf.as_slice()).into_py_any(py)?;
        Ok((p, v))
    })
}

/// Transform a satellite state (position + velocity) from GCRF to ITRF.
///
/// Inverse of :func:`itrf_to_gcrf_state`. Rotates the state through
/// GCRF → CIRS → TIRS, subtracts the Earth-rotation ``omega_earth x r``
/// term **in TIRS** (where Earth's rotation axis is exactly along +z),
/// then applies inverse polar motion to reach ITRF. A geostationary
/// satellite (whose GCRF velocity is pure orbital motion) produces zero
/// velocity in ITRF. Uses the full IAU 2010 reduction.
///
/// Args:
///     pos_gcrf (array-like): 3-element position vector in GCRF [m]
///     vel_gcrf (array-like): 3-element velocity vector in GCRF [m/s]
///     time (satkit.time): Epoch of the state
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_itrf, vel_itrf)``.
///     ``vel_itrf`` is the velocity as observed in ITRF.
#[pyfunction]
pub fn gcrf_to_itrf_state(
    pos_gcrf: &Bound<'_, PyAny>,
    vel_gcrf: &Bound<'_, PyAny>,
    time: &PyInstant,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let pos_vec: Vector3 = py_to_smatrix(pos_gcrf)?;
    let vel_vec: Vector3 = py_to_smatrix(vel_gcrf)?;
    let (pos_itrf, vel_itrf) = ft::gcrf_to_itrf_state(&pos_vec, &vel_vec, &time.0);
    pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
        let p = np::PyArray1::from_slice(py, pos_itrf.as_slice()).into_py_any(py)?;
        let v = np::PyArray1::from_slice(py, vel_itrf.as_slice()).into_py_any(py)?;
        Ok((p, v))
    })
}

#[pyfunction(name = "disable_eop_time_warning")]
pub fn disable_eop_time_warning() {
    satkit::earth_orientation_params::disable_eop_time_warning();
}
