use crate::pyinstant::ToTimeVec;
use crate::pyutils::*;
use crate::PyInstant;
use numpy as np;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use satkit::frametransform as ft;
use satkit::mathtypes::*;
use satkit::Instant;

use anyhow::{bail, Result};

/// Greenwich Mean Sidereal Time
///
/// Notes:
///     * Vallado algorithm 15:
///     * GMST = 67310.54841 + (876600ʰ + 8640184.812866) tᵤₜ₁ + 0.093104 tᵤₜ₁² − 6.2e−6 tᵤₜ₁³ (seconds of time; tᵤₜ₁ = Julian centuries of UT1 from J2000.0)
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate GMST
///
/// Returns:
///     float|numpy.array: GMST at input time[s] in radians
#[pyfunction]
pub fn gmst(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    py_func_of_time_arr(ft::gmst, tm)
}

/// Equation of the Equinoxes
///
/// The difference between apparent and mean sidereal time (GAST - GMST),
/// arising from nutation of the Earth's axis.
///
/// Notes:
///     * Two-term approximation (Vallado 2013, §3.7.3); against the IAU 1994
///       equation of the equinoxes with the full IAU 1980 nutation (ERFA
///       ``eqeq94``) it is good to about 0.6" (0.65" max, 43 ms of time,
///       over 1950-2100)
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate output
///
/// Returns:
///     float|numpy.array: Equation of the equinoxes at input time[s] in radians
#[pyfunction]
pub fn eqeq(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    py_func_of_time_arr(ft::eqeq, tm)
}

/// Greenwich apparent sidereal time, radians
///
/// GMST (IAU 1982) plus the two-term equation of the equinoxes (``eqeq``),
/// so good to about 0.6" (0.65" max, 43 ms of time, over 1950-2100)
/// against ERFA ``gst94``.
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
    satkit::frametransform::ierstable::preload()?;
    py_quat_from_time_arr(ft::qcirs2gcrs, tm)
}

///Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)
///
/// Notes:
///    * Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation); see IERS Technical Note 36, Chapter 5
///    * Very computationally expensive
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from ITRF to GCRF at input time[s]

#[pyfunction]
pub fn qitrf2gcrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    satkit::frametransform::ierstable::preload()?;
    py_quat_from_time_arr(ft::qitrf2gcrf, tm)
}

///Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)
///
/// Notes:
///     * Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation); see IERS Technical Note 36, Chapter 5
///     * Very computationally expensive
///
/// Args:
///     tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate rotation
///
/// Returns:
///     satkit.quaternion|list: Quaternion or list of quaternions representing rotation from GCRF to ITRF at input time[s]
#[pyfunction]
pub fn qgcrf2itrf(tm: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    satkit::frametransform::ierstable::preload()?;
    py_quat_from_time_arr(ft::qgcrf2itrf, tm)
}

/// Approximate rotation from Geocentric Celestrial Reference Frame to International Terrestrial Reference Frame
///
/// Notes:
///     * Accurate to about 1 arcsec (1.0" max against the full IERS 2010
///       reduction, 1973-2026), of which up to 0.6" is polar motion, which
///       this chain neglects
///     * The chain is GAST (GMST82 + two-term equation of the equinoxes),
///       two-term nutation (``qtod2mod_approx``) and IAU 2006 precession
///       without frame bias (``qmod2gcrf``); see Vallado section 3.7.3. It is
///       often labelled "IAU-76/FK5", but it is neither the IAU 1976
///       precession nor the 106-term IAU 1980 nutation series
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
///     * Accurate to about 1 arcsec (1.0" max against the full IERS 2010
///       reduction, 1973-2026), of which up to 0.6" is polar motion, which
///       this chain neglects
///     * The chain is GAST (GMST82 + two-term equation of the equinoxes),
///       two-term nutation (``qtod2mod_approx``) and IAU 2006 precession
///       without frame bias (``qmod2gcrf``); see Vallado section 3.7.3. It is
///       often labelled "IAU-76/FK5", but it is neither the IAU 1976
///       precession nor the 106-term IAU 1980 nutation series
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
///     * This is Equation 3-90 in Vallado: GMST (IAU 1982) rotation TEME -> PEF,
///       then polar motion PEF -> ITRF. No precession-nutation, so it is exact
///       to the model and the same as rotation(TEME, ITRF) and
///       rotation_approx(TEME, ITRF)
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
///    * TEME is output frame of SGP4 propagator (quasi-inertial)
///    * **Approximate**: the same as rotation_approx(TEME, GCRF), not
///      rotation(TEME, GCRF). GMST82 to PEF, then the approximate chain of
///      qitrf2gcrf_approx; no polar motion. Accurate to 0.55" max against the
///      full IERS 2010 reduction (1973-2026): ~19 m at LEO, ~110 m at GEO
///    * rotation(TEME, GCRF) is the full reduction (matches ERFA to ~5 uas)
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
///     * 3 : LOD: excess length of day, -d(UT1-UTC)/dt, seconds per day
///     * 4 : dX wrt IAU-2000A nutation, milli-arcsecs
///     * 5 : dY wrt IAU-2000A nutation, milli-arcsecs
///
///     Or None if the time is outside the range of available Earth Orientation Parameters (EOP)
///    (EOP are available from 1973-01-02 with the default finals2000A.all, or from 1962 with
///    CelesTrak's EOP-All.csv in a data directory, to current, with predictions up to a year ahead)
///
#[pyfunction(name = "earth_orientation_params")]
pub fn pyeop(time: &PyInstant) -> Option<(f64, f64, f64, f64, f64, f64)> {
    satkit::earth_orientation_params::get(&time.0).map(|r| (r[0], r[1], r[2], r[3], r[4], r[5]))
}

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
///         Time-dependent frames (the Earth-fixed ITRF, the quasi-inertial
///         TEME, EME2000, etc.) need
///         the time-based quaternion helpers instead (qitrf2gcrf,
///         qteme2gcrf, ...).
#[pyfunction]
pub fn to_gcrf(
    py: Python,
    frame: crate::pyframes::PyFrame,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let pos_vec: Vector3 = py_to_smatrix(pos)?;
    let vel_vec: Vector3 = py_to_smatrix(vel)?;
    let rust_frame: satkit::Frame = frame.into();
    let dcm = ft::to_gcrf(rust_frame, &pos_vec, &vel_vec)?;
    // numeris matrices are column-major while numpy's reshape is
    // row-major; flatten the transpose so the numpy array has the same
    // element layout as the Rust matrix (previously this returned the
    // transposed, i.e. inverse, rotation).
    Ok(slice2py2d(py, dcm.transpose().as_slice(), 3, 3)?)
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
    py: Python,
    frame: crate::pyframes::PyFrame,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let pos_vec: Vector3 = py_to_smatrix(pos)?;
    let vel_vec: Vector3 = py_to_smatrix(vel)?;
    let rust_frame: satkit::Frame = frame.into();
    let dcm = ft::from_gcrf(rust_frame, &pos_vec, &vel_vec)?;
    // Column-major -> row-major via transpose; see `to_gcrf`.
    Ok(slice2py2d(py, dcm.transpose().as_slice(), 3, 3)?)
}

/// Rotation from the Mean-of-Date frame (MOD) to the Geocentric Celestial
/// Reference Frame (GCRF). Accounts for precession but not nutation.
///
/// Notes:
///     * Precession only: the IAU 2006 angles zeta_A, z_A, theta_A (Vallado
///       Eqs. 3-88, 3-89), not the IAU 1976 precession
///     * No frame bias: the target is the J2000 mean equator and equinox
///       (EME2000), 23 mas from GCRF
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
/// Notes:
///     * Two-term nutation (Vallado 2013, §3.7.3), good to 0.9" (0.88" max
///       over 1950-2100) against the IAU 2006/2000A nutation
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
/// The IERS 2010 ITRF → GCRF reduction decomposes into three stages:
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
/// 3. Rotate TIRS → CIRS → GCRF via the full IERS 2010 chain.
///
/// Uses the full IERS 2010 reduction (includes polar motion, Earth
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
///     the state expressed in GCRF: position in meters, velocity in m/s
///     (shape (3,) each, or (N, 3) for batched input).
#[pyfunction]
pub fn itrf_to_gcrf_state(
    pos_itrf: &Bound<'_, PyAny>,
    vel_itrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    satkit::frametransform::ierstable::preload()?;
    state_transform_batch(pos_itrf, vel_itrf, time, ft::itrf_to_gcrf_state)
}

/// Transform a satellite state (position + velocity) from GCRF to ITRF.
///
/// Inverse of :func:`itrf_to_gcrf_state`. Rotates the state through
/// GCRF → CIRS → TIRS, subtracts the Earth-rotation ``omega_earth x r``
/// term **in TIRS** (where Earth's rotation axis is exactly along +z),
/// then applies inverse polar motion to reach ITRF. A geostationary
/// satellite (whose GCRF velocity is pure orbital motion) produces zero
/// velocity in ITRF. Uses the full IERS 2010 reduction.
///
/// Args:
///     pos_gcrf (array-like): 3-element position vector in GCRF [m]
///     vel_gcrf (array-like): 3-element velocity vector in GCRF [m/s]
///     time (satkit.time): Epoch of the state
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_itrf, vel_itrf)``:
///     position in meters, velocity as observed in ITRF in m/s
///     (shape (3,) each, or (N, 3) for batched input).
#[pyfunction]
pub fn gcrf_to_itrf_state(
    pos_gcrf: &Bound<'_, PyAny>,
    vel_gcrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    satkit::frametransform::ierstable::preload()?;
    state_transform_batch(pos_gcrf, vel_gcrf, time, ft::gcrf_to_itrf_state)
}

/// Approximate ITRF → GCRF state transform using the approximate reduction
/// of :func:`qitrf2gcrf_approx`.
///
/// Faster alternative to :func:`itrf_to_gcrf_state` when the full IERS 2010
/// precision is not required; accurate to ~1 arcsec on position. Neglects
/// polar motion, so the Earth-rotation sweep ``omega_earth x r`` is
/// evaluated in ITRF directly. Accepts scalar or batched inputs like
/// :func:`itrf_to_gcrf_state`.
///
/// Args:
///     pos_itrf (array-like): (3,) or (N, 3) position vector in ITRF, meters
///     vel_itrf (array-like): (3,) or (N, 3) velocity vector as observed in ITRF, m/s
///     time (satkit.time): Epoch of the state (length-N array/list for batched input)
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_gcrf, vel_gcrf)``:
///     position in meters, velocity in m/s
#[pyfunction]
pub fn itrf_to_gcrf_state_approx(
    pos_itrf: &Bound<'_, PyAny>,
    vel_itrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    state_transform_batch(pos_itrf, vel_itrf, time, ft::itrf_to_gcrf_state_approx)
}

/// Approximate GCRF → ITRF state transform using the approximate reduction
/// of :func:`qgcrf2itrf_approx`.
///
/// Inverse of :func:`itrf_to_gcrf_state_approx`; accurate to ~1 arcsec on
/// position. Accepts scalar or batched inputs like
/// :func:`gcrf_to_itrf_state`.
///
/// Args:
///     pos_gcrf (array-like): (3,) or (N, 3) position vector in GCRF, meters
///     vel_gcrf (array-like): (3,) or (N, 3) velocity vector in GCRF, m/s
///     time (satkit.time): Epoch of the state (length-N array/list for batched input)
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_itrf, vel_itrf)``:
///     position in meters, velocity as observed in ITRF in m/s
#[pyfunction]
pub fn gcrf_to_itrf_state_approx(
    pos_gcrf: &Bound<'_, PyAny>,
    vel_gcrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    state_transform_batch(pos_gcrf, vel_gcrf, time, ft::gcrf_to_itrf_state_approx)
}

/// Apply a GCRF<->ITRF state transform to either a single state or a batch.
///
/// Scalar: ``pos``/``vel`` are 3-element vectors and ``time`` is a single
/// ``satkit.time`` or ``datetime.datetime``. Batch: ``pos``/``vel`` are
/// shape ``(N, 3)`` arrays and ``time`` is a length-``N`` time array.
fn state_transform_batch(
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
    cfunc: fn(&Vector3, &Vector3, &Instant) -> (Vector3, Vector3),
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    if pos.is_instance_of::<np::PyArray2<f64>>() {
        let parr = pos.extract::<np::PyReadonlyArray2<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid pos array: {}", e))
        })?;
        let varr = vel.extract::<np::PyReadonlyArray2<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid vel array: {}", e))
        })?;
        let pshape = parr.shape();
        let vshape = varr.shape();
        let n = pshape[0];
        if pshape[1] != 3 {
            bail!(
                "pos must have shape (N, 3), got ({}, {})",
                pshape[0],
                pshape[1]
            );
        }
        if vshape[0] != n || vshape[1] != 3 {
            bail!(
                "vel must have same shape as pos ({}, 3), got ({}, {})",
                n,
                vshape[0],
                vshape[1]
            );
        }
        let tm = time.to_time_vec()?;
        if tm.len() != n {
            bail!(
                "time array length ({}) must match number of states ({})",
                tm.len(),
                n
            );
        }
        let pa = parr.as_array();
        let va = varr.as_array();
        let mut pout = Vec::with_capacity(n * 3);
        let mut vout = Vec::with_capacity(n * 3);
        for i in 0..n {
            let p = Vector3::from_array([pa[(i, 0)], pa[(i, 1)], pa[(i, 2)]]);
            let v = Vector3::from_array([va[(i, 0)], va[(i, 1)], va[(i, 2)]]);
            let (po, vo) = cfunc(&p, &v, &tm[i]);
            pout.extend_from_slice(po.as_slice());
            vout.extend_from_slice(vo.as_slice());
        }
        let py = pos.py();
        return Ok((
            np::PyArray1::from_vec(py, pout)
                .reshape([n, 3])?
                .into_py_any(py)?,
            np::PyArray1::from_vec(py, vout)
                .reshape([n, 3])?
                .into_py_any(py)?,
        ));
    }

    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let t = instant_from_pyany(time)?;
    let (po, vo) = cfunc(&p, &v, &t);
    let py = time.py();
    Ok((vec2py(py, &po)?, vec2py(py, &vo)?))
}

// ───── Frame-enum dispatch (new in 0.17.0) ─────────────────────────────

/// Quaternion rotating a vector from ``from_frame`` to ``to_frame`` at time
/// ``tm``. Full IERS 2010 reduction.
///
/// Uses the shortest path through the frame graph for each pair (does not
/// always pivot through GCRF). Pairs involving orbit-dependent frames
/// (LVLH, RTN, NTW) require state and are not supported here — use
/// :func:`to_gcrf` / :func:`from_gcrf` for those.
///
/// Args:
///     from_frame (satkit.frame): Source frame
///     to_frame (satkit.frame): Destination frame
///     tm (satkit.time|datetime.datetime): Epoch
///
/// Returns:
///     satkit.quaternion: Rotation from ``from_frame`` to ``to_frame`` at ``tm``.
///
/// Raises:
///     RuntimeError: if the pair involves LVLH / RTN / NTW.
#[pyfunction]
pub fn rotation(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    satkit::frametransform::ierstable::preload()?;
    rotation_dispatch_batch(from_frame, to_frame, tm, /* approx = */ false)
}

/// Quaternion rotating a vector from ``from_frame`` to ``to_frame`` using
/// the approximate reduction of :func:`qitrf2gcrf_approx` (~1 arcsec;
/// TEME <-> GCRF / EME2000 / ICRF 0.55", as :func:`qteme2gcrf`; TEME <-> ITRF
/// is exact, as :func:`qteme2itrf`).
///
/// Only valid between ITRF and the inertial cluster (GCRF, EME2000, ICRF,
/// TEME). TIRS and CIRS are defined by the IERS 2010 reduction and have
/// no analogue in the approximate chain.
///
/// Args:
///     from_frame (satkit.frame): Source frame
///     to_frame (satkit.frame): Destination frame
///     tm (satkit.time|datetime.datetime): Epoch
///
/// Returns:
///     satkit.quaternion: Approximate rotation from ``from_frame`` to ``to_frame``.
///
/// Raises:
///     RuntimeError: if either frame is TIRS / CIRS, or if the pair involves
///         LVLH / RTN / NTW.
#[pyfunction]
pub fn rotation_approx(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    rotation_dispatch_batch(from_frame, to_frame, tm, /* approx = */ true)
}

/// Shared scalar/batch dispatch for [`rotation`] / [`rotation_approx`]:
/// a single quaternion for a scalar time, or a list of quaternions for an
/// array time, as the per-pair helpers return.
fn rotation_dispatch_batch(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
    approx: bool,
) -> Result<Py<PyAny>> {
    let from: satkit::Frame = from_frame.into();
    let to: satkit::Frame = to_frame.into();
    py_quat_from_time_result_arr(
        |t: &Instant| -> Result<Quaternion> {
            if approx {
                Ok(ft::rotation_approx(from, to, t)?)
            } else {
                Ok(ft::rotation(from, to, t)?)
            }
        },
        tm,
    )
}

/// State (position + velocity) transform from ``from_frame`` to ``to_frame``
/// at time ``tm``. Properly handles the Earth-rotation sweep term when
/// transitioning between rotating (ITRF) and inertial frames.
///
/// Currently supported pairs: identity, ITRF↔{GCRF, EME2000, ICRF, TEME},
/// and within-inertial pairs. Other pairs raise RuntimeError.
///
/// Args:
///     from_frame (satkit.frame): Source frame
///     to_frame (satkit.frame): Destination frame
///     tm (satkit.time|datetime.datetime): Epoch
///     pos (numpy.ndarray): 3-element position vector [m]
///     vel (numpy.ndarray): 3-element velocity vector [m/s]
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: (pos, vel) in ``to_frame``:
///     position in meters, velocity in m/s.
#[pyfunction]
pub fn transform_state(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    satkit::frametransform::ierstable::preload()?;
    let t = instant_from_pyany(tm)?;
    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let (po, vo) = ft::transform_state(from_frame.into(), to_frame.into(), &t, &p, &v)?;
    let py = tm.py();
    Ok((vec2py(py, &po)?, vec2py(py, &vo)?))
}

/// State transform using the approximate reduction of :func:`rotation_approx`.
/// Same supported-pair set as :func:`transform_state`; TEME <-> ITRF does not
/// use the approximate chain and equals the full transform.
///
/// Args:
///     from_frame (satkit.frame): Source frame
///     to_frame (satkit.frame): Destination frame
///     tm (satkit.time|datetime.datetime): Epoch
///     pos (numpy.ndarray): 3-element position vector, meters
///     vel (numpy.ndarray): 3-element velocity vector, m/s
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: (pos, vel) in ``to_frame``:
///     position in meters, velocity in m/s.
#[pyfunction]
pub fn transform_state_approx(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let t = instant_from_pyany(tm)?;
    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let (po, vo) = ft::transform_state_approx(from_frame.into(), to_frame.into(), &t, &p, &v)?;
    let py = tm.py();
    Ok((vec2py(py, &po)?, vec2py(py, &vo)?))
}

/// Quaternion rotating a vector from ``from_frame`` to ``to_frame`` — the
/// unified front door that supports **all** frames, both the time-parameterised
/// Earth chain (ITRF, TIRS, CIRS, GCRF, TEME, EME2000, ICRF) and the
/// orbit-dependent frames (LVLH, RTN, NTW), in a single call.
///
/// Unlike :func:`rotation` (which rejects the orbit frames) and :func:`to_gcrf`
/// (which rejects the Earth frames), this accepts any pair. It does **not**
/// always pivot through GCRF: a purely Earth-frame pair delegates to
/// :func:`rotation`, which takes the shortest path through the frame graph;
/// only pairs involving an orbit-dependent frame compose through GCRF. The
/// orbit state (``pos``, ``vel``, both in GCRF) is only consulted when an
/// orbit-dependent frame is involved.
///
/// Args:
///     from_frame (satkit.frame): Source frame
///     to_frame (satkit.frame): Destination frame
///     tm (satkit.time|datetime.datetime): Epoch
///     pos (numpy.ndarray): 3-element GCRF position vector [m]
///     vel (numpy.ndarray): 3-element GCRF velocity vector [m/s]
///
/// Returns:
///     satkit.quaternion: Rotation from ``from_frame`` to ``to_frame`` at ``tm``.
#[pyfunction]
pub fn rotation_with_state(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    satkit::frametransform::ierstable::preload()?;
    let t = instant_from_pyany(tm)?;
    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let q = ft::rotation_with_state(from_frame.into(), to_frame.into(), &t, &p, &v)?;
    Ok(crate::pyquaternion::PyQuaternion(q).into_py_any(tm.py())?)
}

/// Disable the warning about out-of-range Earth Orientation Parameters (EOP).
///
/// The warning is shown only once by default; call this function to suppress
/// it entirely.
///
/// Example:
///     >>> import satkit
///     >>> satkit.frametransform.disable_eop_time_warning()
#[pyfunction(name = "disable_eop_time_warning")]
pub fn disable_eop_time_warning() {
    satkit::earth_orientation_params::disable_eop_time_warning();
}

/// Time bounds of the loaded Earth Orientation Parameters (EOP) table.
///
/// Returns:
///     (satkit.time, satkit.time, satkit.time) | None: ``(first, last_observed, last)`` —
///     the first row, the last *observed* row (rows after it are IERS
///     predictions) and the last row of the table; ``None`` if no table is
///     loaded. Epochs after ``last`` use that row's values held constant
///     (see :func:`eop_status`); refresh with ``satkit.utils.update_datafiles()``.
///
/// Example:
///     >>> first, last_obs, last = satkit.frametransform.eop_coverage()
#[pyfunction(name = "eop_coverage")]
pub fn eop_coverage() -> Option<(PyInstant, PyInstant, PyInstant)> {
    satkit::earth_orientation_params::coverage().map(|c| {
        (
            PyInstant(c.first),
            PyInstant(c.last_observed),
            PyInstant(c.last),
        )
    })
}

/// Which file the loaded Earth Orientation Parameters (EOP) table came from.
///
/// satkit reads the IERS Bulletin A combined file ``finals2000A.all`` (primary;
/// fetched from the USNO and IERS mirrors, observed values from 1973 plus about a
/// year of predictions) and CelesTrak's ``EOP-All.csv`` (fallback when both mirrors
/// are unreachable; also read when present, e.g. a hand-provisioned data directory).
/// When both are present the one whose observed record runs later is used.
///
/// Returns:
///     str | None: ``"finals2000A"`` for the IERS file (a table with the CelesTrak
///     file's pre-1973 rows in front of it reports this too), ``"celestrak"`` for
///     ``EOP-All.csv``, ``None`` if no table is loaded.
#[pyfunction(name = "eop_source")]
pub fn eop_source() -> Option<&'static str> {
    use satkit::earth_orientation_params::EopSource;
    satkit::earth_orientation_params::source().map(|s| match s {
        EopSource::IersFinals2000A => "finals2000A",
        EopSource::CelesTrak => "celestrak",
    })
}

/// Classify an epoch against the loaded Earth Orientation Parameters (EOP) table.
///
/// Args:
///     tm (satkit.time): Epoch to classify
///
/// Returns:
///     str: one of
///
///     * ``"observed"`` — inside the table, on or before the last observed row
///     * ``"predicted"`` — inside the table, IERS prediction
///     * ``"extrapolated"`` — after the table end: the last row is held constant
///       (accuracy degrades by ~0.1 arcsec / ~10 ms per few months — refresh the
///       data files)
///     * ``"before_table"`` — before the table's first row (1973-01-02 for the default
///       ``finals2000A.all``, 1962 with ``EOP-All.csv``): no EOP, zeros are used
///     * ``"not_loaded"`` — no EOP table loaded at all: zeros are used
#[pyfunction(name = "eop_status")]
pub fn eop_status(tm: &PyInstant) -> &'static str {
    use satkit::earth_orientation_params::EopStatus::*;
    match satkit::earth_orientation_params::status(&tm.0) {
        Observed => "observed",
        Predicted => "predicted",
        Extrapolated => "extrapolated",
        BeforeTable => "before_table",
        NotLoaded => "not_loaded",
    }
}
