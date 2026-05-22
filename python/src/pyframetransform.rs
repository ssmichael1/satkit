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
///     the state expressed in GCRF.
#[pyfunction]
pub fn itrf_to_gcrf_state(
    pos_itrf: &Bound<'_, PyAny>,
    vel_itrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
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
///     (numpy.ndarray, numpy.ndarray): Tuple ``(pos_itrf, vel_itrf)``.
///     ``vel_itrf`` is the velocity as observed in ITRF.
#[pyfunction]
pub fn gcrf_to_itrf_state(
    pos_gcrf: &Bound<'_, PyAny>,
    vel_gcrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    state_transform_batch(pos_gcrf, vel_gcrf, time, ft::gcrf_to_itrf_state)
}

/// Approximate ITRF → GCRF state transform using the IAU-76/FK5 reduction.
///
/// Faster alternative to :func:`itrf_to_gcrf_state` when the full IERS 2010
/// precision is not required; accurate to ~1 arcsec on position. Neglects
/// polar motion, so the Earth-rotation sweep ``omega_earth x r`` is
/// evaluated in ITRF directly. Accepts scalar or batched inputs like
/// :func:`itrf_to_gcrf_state`.
#[pyfunction]
pub fn itrf_to_gcrf_state_approx(
    pos_itrf: &Bound<'_, PyAny>,
    vel_itrf: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    state_transform_batch(pos_itrf, vel_itrf, time, ft::itrf_to_gcrf_state_approx)
}

/// Approximate GCRF → ITRF state transform using the IAU-76/FK5 reduction.
///
/// Inverse of :func:`itrf_to_gcrf_state_approx`; accurate to ~1 arcsec on
/// position. Accepts scalar or batched inputs like
/// :func:`gcrf_to_itrf_state`.
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
        return pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
            let pout = np::PyArray2::<f64>::zeros(py, (n, 3), false);
            let vout = np::PyArray2::<f64>::zeros(py, (n, 3), false);
            for i in 0..n {
                let p = Vector3::from_array([pa[(i, 0)], pa[(i, 1)], pa[(i, 2)]]);
                let v = Vector3::from_array([va[(i, 0)], va[(i, 1)], va[(i, 2)]]);
                let (po, vo) = cfunc(&p, &v, &tm[i]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        po.as_slice().as_ptr(),
                        pout.as_raw_array_mut().as_mut_ptr().offset(i as isize * 3),
                        3,
                    );
                    std::ptr::copy_nonoverlapping(
                        vo.as_slice().as_ptr(),
                        vout.as_raw_array_mut().as_mut_ptr().offset(i as isize * 3),
                        3,
                    );
                }
            }
            Ok((pout.into_py_any(py)?, vout.into_py_any(py)?))
        });
    }

    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let t = instant_from_pyany(time)?;
    let (po, vo) = cfunc(&p, &v, &t);
    pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
        Ok((
            np::PyArray1::from_slice(py, po.as_slice()).into_py_any(py)?,
            np::PyArray1::from_slice(py, vo.as_slice()).into_py_any(py)?,
        ))
    })
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
    let t = instant_from_pyany(tm)?;
    let q = ft::rotation(from_frame.into(), to_frame.into(), &t)?;
    pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
        Ok(crate::pyquaternion::PyQuaternion(q).into_py_any(py)?)
    })
}

/// Quaternion rotating a vector from ``from_frame`` to ``to_frame`` using
/// the IAU-76/FK5 approximate reduction (~1 arcsec).
///
/// Only valid between ITRF and the inertial cluster (GCRF, EME2000, ICRF,
/// TEME). TIRS and CIRS are defined by the IERS 2010 reduction and have
/// no FK5 analogue.
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
    let t = instant_from_pyany(tm)?;
    let q = ft::rotation_approx(from_frame.into(), to_frame.into(), &t)?;
    pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
        Ok(crate::pyquaternion::PyQuaternion(q).into_py_any(py)?)
    })
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
///     tuple[numpy.ndarray, numpy.ndarray]: (pos, vel) in ``to_frame``.
#[pyfunction]
pub fn transform_state(
    from_frame: crate::pyframes::PyFrame,
    to_frame: crate::pyframes::PyFrame,
    tm: &Bound<'_, PyAny>,
    pos: &Bound<'_, PyAny>,
    vel: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let t = instant_from_pyany(tm)?;
    let p: Vector3 = py_to_smatrix(pos)?;
    let v: Vector3 = py_to_smatrix(vel)?;
    let (po, vo) = ft::transform_state(from_frame.into(), to_frame.into(), &t, &p, &v)?;
    pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
        Ok((
            np::PyArray1::from_slice(py, po.as_slice()).into_py_any(py)?,
            np::PyArray1::from_slice(py, vo.as_slice()).into_py_any(py)?,
        ))
    })
}

/// State transform using the IAU-76/FK5 approximate reduction. Same
/// supported-pair set as :func:`transform_state`.
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
    pyo3::Python::attach(|py| -> Result<(Py<PyAny>, Py<PyAny>)> {
        Ok((
            np::PyArray1::from_slice(py, po.as_slice()).into_py_any(py)?,
            np::PyArray1::from_slice(py, vo.as_slice()).into_py_any(py)?,
        ))
    })
}

#[pyfunction(name = "disable_eop_time_warning")]
pub fn disable_eop_time_warning() {
    satkit::earth_orientation_params::disable_eop_time_warning();
}
