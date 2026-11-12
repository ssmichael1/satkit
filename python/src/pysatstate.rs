use crate::pyframes::PyFrame;
use crate::pyinstant::{PyInstant, TimeArg};
use crate::pypropsettings::PyPropSettings;
use crate::pyquaternion::PyQuaternion;
use crate::pysatproperties::{satproperties_arg, PySatProperties};
use crate::pyutils::{slice2py1d, slice2py2d};
use crate::PyDuration;

use numpy as np;
use numpy::PyUntypedArrayMethods;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;
use satkit::orbitprop::{ImpulsiveManeuver, SatState, StateCov};
use satkit::{Frame, Instant};

use anyhow::{bail, Result};

/// A 3-vector from any real numeric array-like (numpy array of any numeric
/// dtype, list, tuple); `what` names it in the error message. A wrong length
/// is still the `RuntimeError` it was when only float64 arrays were accepted.
fn vec3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vector3> {
    crate::pyutils::to_vector3(obj, what).map_err(|e| {
        if e.is_instance_of::<pyo3::exceptions::PyValueError>(obj.py()) {
            pyo3::exceptions::PyRuntimeError::new_err(e.value(obj.py()).to_string())
        } else {
            e
        }
    })
}

/// A 6x6 state covariance from any real numeric array-like (element by
/// element, so non-contiguous input such as Fortran-order arrays and views
/// works)
fn cov6(cov: &Bound<'_, PyAny>) -> Result<Matrix6> {
    let arr = crate::pyutils::to_f64_ndarray(cov)?;
    if arr.shape() != [6, 6] {
        bail!(
            "Covariance must be a 6x6 matrix, got shape {:?}",
            arr.shape()
        );
    }
    let m = crate::pyutils::py_to_smatrix::<6, 6>(arr.as_any())?;
    check_symmetric(&m)?;
    Ok(m)
}

/// A covariance is symmetric; refuse one that is not (or that holds a NaN),
/// rather than store a matrix whose two triangles disagree. Round-off level
/// asymmetry, as left by `Phi @ P @ Phi.T`, is accepted and stored as given.
fn check_symmetric(m: &Matrix6) -> PyResult<()> {
    for i in 0..6 {
        for j in 0..i {
            let (a, b) = (m[(i, j)], m[(j, i)]);
            let tol =
                1.0e-9 * a.abs().max(b.abs()) + 1.0e-12 * (m[(i, i)] * m[(j, j)]).abs().sqrt();
            let diff = (a - b).abs();
            if diff.is_nan() || diff > tol {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "covariance must be symmetric: element [{i}, {j}] = {a} but [{j}, {i}] = {b}"
                )));
            }
        }
    }
    Ok(())
}

/// Maneuvers are resolved in the satellite-local frames or GCRF only
fn check_maneuver_frame(frame: Frame) -> PyResult<()> {
    match frame {
        Frame::GCRF | Frame::RTN | Frame::NTW | Frame::LVLH => Ok(()),
        Frame::ITRF | Frame::TIRS | Frame::CIRS | Frame::TEME | Frame::EME2000 | Frame::ICRF => {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported frame for maneuver: {frame}. Must be GCRF, RTN (RIC), NTW, or LVLH"
            )))
        }
    }
}

crate::arg_extractor!(propsettings_arg: Option<PyPropSettings>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid propsettings: {e}"))
});

/// Satellite state: GCRF position, velocity, optional covariance, and maneuvers
///
/// Args:
///     time (satkit.time | datetime.datetime | numpy.datetime64): Epoch of the state
///     pos (array-like): 3-element GCRF position, meters
///     vel (array-like): 3-element GCRF velocity, m/s
///     cov (array-like, optional): symmetric 6x6 GCRF state covariance
///         (position block m^2, velocity block (m/s)^2, cross blocks m^2/s). Default None.
///
/// Raises:
///     TypeError: if time is not a time, or pos, vel or cov is not a real numeric array-like
///     ValueError: if pos or vel does not have 3 elements, or cov is not a symmetric 6x6 matrix
#[pyclass(name = "satstate", module = "satkit", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySatState(SatState);

#[pymethods]
impl PySatState {
    #[new]
    #[pyo3(signature=(time, pos, vel, cov=None))]
    fn py_new(
        time: TimeArg,
        pos: &Bound<'_, PyAny>,
        vel: &Bound<'_, PyAny>,
        cov: Option<&Bound<'_, PyAny>>,
    ) -> Result<Self> {
        let mut state =
            SatState::from_pv(&time.0, &vec3(pos, "position")?, &vec3(vel, "velocity")?);

        if let Some(cov) = cov {
            state.set_cov(StateCov::PVCov(cov6(cov)?));
        }

        Ok(Self(state))
    }

    /// Create a state from Keplerian elements
    ///
    /// The two-body position and velocity of the elements (``kepler.to_pv()``)
    /// are taken as GCRF at ``time``; no covariance, no maneuvers. The
    /// elements are treated as osculating GCRF elements, so ``kepler.mu``
    /// should be Earth's.
    ///
    /// Args:
    ///     time (satkit.time): epoch of the state
    ///     kepler (satkit.kepler): osculating Keplerian elements
    ///
    /// Returns:
    ///     satkit.satstate: state at ``time``
    #[staticmethod]
    fn from_kepler(time: TimeArg, kepler: &crate::pykepler::PyKepler) -> Self {
        Self(SatState::from_kepler(&time.0, &kepler.0))
    }

    /// Set 1-sigma position uncertainty in a satellite-local or inertial frame.
    ///
    /// The uncertainty is interpreted as a diagonal 3x3 covariance in the
    /// given ``frame`` (axes aligned with the frame), rotated into GCRF,
    /// and stored in the position block of the 6x6 state covariance. Any
    /// existing velocity covariance is preserved; the position-velocity
    /// cross blocks are cleared.
    ///
    /// Args:
    ///     sigma (array-like): 3 1-sigma position
    ///         components along the frame's axes. Units: meters.
    ///     frame (satkit.frame): Coordinate frame — **required**, no
    ///         default (matching the Rust API). Supported values:
    ///         ``frame.GCRF``, ``frame.RTN`` (= RSW = RIC), ``frame.NTW``,
    ///         ``frame.LVLH``.
    ///
    /// Raises:
    ///     ValueError: if sigma does not have 3 elements, or the frame is
    ///         not one of the supported orbital / inertial frames above.
    ///     TypeError: if sigma is not a real numeric array-like
    fn set_pos_uncertainty(&mut self, sigma: &Bound<'_, PyAny>, frame: PyFrame) -> Result<()> {
        let sigma = vec3(sigma, "position uncertainty")?;
        self.0.set_pos_uncertainty(&sigma, frame.into())?;
        Ok(())
    }

    /// Set 1-sigma velocity uncertainty in a satellite-local or inertial frame.
    ///
    /// Analogous to :meth:`set_pos_uncertainty`, but for the velocity
    /// block of the 6x6 state covariance. Any existing position
    /// covariance is preserved; the position-velocity cross blocks are
    /// cleared.
    ///
    /// Args:
    ///     sigma (array-like): 3 1-sigma velocity
    ///         components along the frame's axes. Units: m/s.
    ///     frame (satkit.frame): Coordinate frame — **required**, no
    ///         default (matching the Rust API). Supported values:
    ///         ``frame.GCRF``, ``frame.RTN``, ``frame.NTW``, ``frame.LVLH``.
    ///
    /// Raises:
    ///     ValueError: if sigma does not have 3 elements, or the frame is
    ///         not one of the supported orbital / inertial frames above.
    ///     TypeError: if sigma is not a real numeric array-like
    fn set_vel_uncertainty(&mut self, sigma: &Bound<'_, PyAny>, frame: PyFrame) -> Result<()> {
        let sigma = vec3(sigma, "velocity uncertainty")?;
        self.0.set_vel_uncertainty(&sigma, frame.into())?;
        Ok(())
    }

    /// Set the full 6x6 state covariance matrix, or clear it with None
    ///
    /// Args:
    ///     cov (array-like | None): symmetric 6x6 GCRF state covariance for position (meters) and velocity (m/s):
    ///         position block m^2, velocity block (m/s)^2, cross blocks m^2/s. None removes the covariance
    ///
    /// Raises:
    ///     ValueError: if cov is not 6x6, or is not symmetric
    ///     TypeError: if cov is not a real numeric array-like or None
    #[setter]
    fn set_cov(&mut self, cov: Option<&Bound<'_, PyAny>>) -> Result<()> {
        self.0.cov = match cov {
            Some(cov) => StateCov::PVCov(cov6(cov)?),
            None => StateCov::None,
        };
        Ok(())
    }

    /// Epoch of this satellite state
    ///
    /// Returns:
    ///     satkit.time: Epoch of the state
    #[getter]
    fn get_time(&self) -> PyInstant {
        PyInstant(self.0.time)
    }

    /// GCRF position of the satellite
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element GCRF position, meters
    #[getter]
    fn get_pos_gcrf(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, &self.0.pv.as_slice()[0..3])
    }

    /// GCRF velocity of the satellite
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element GCRF velocity, m/s
    #[getter]
    fn get_vel_gcrf(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, &self.0.pv.as_slice()[3..6])
    }

    /// Full 6x6 state covariance matrix in GCRF, or None if not set
    ///
    /// Element [i, j] is the covariance of state elements i and j of
    /// [x, y, z, vx, vy, vz]: the upper-left 3x3 block is position (m^2), the
    /// lower-right velocity ((m/s)^2), the off-diagonal blocks the
    /// position-velocity cross-covariance (m^2/s). The getter returns a new
    /// array each time, exactly as assigned (or as propagated); modify the
    /// state by assigning a matrix, or None to clear it.
    ///
    /// Returns:
    ///     numpy.ndarray | None: 6x6 covariance, or None if not set
    #[getter]
    fn get_cov(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0.cov {
            StateCov::None => Ok(py.None()),
            // Storage is column-major and numpy reads the slice row-major,
            // hence the transpose (without it the getter returned the
            // transpose of what was set)
            StateCov::PVCov(cov) => slice2py2d(py, cov.transpose().as_slice(), 6, 6),
        }
    }

    /// Quaternion to go from gcrf (Geocentric Celestial Reference Frame) to lvlh (Local-Vertical, Local-Horizontal) frame
    ///
    /// Notes:
    ///     LVLH coordinate system:
    ///     * z axis = -r (nadir, pointing toward Earth center)
    ///     * y axis = -h (opposite angular momentum, h = r x v)
    ///     * x axis completes right-handed system (approximately velocity direction for circular orbits)
    ///
    /// Returns:
    ///     satkit.quaternion: quaternion to go from gcrf to lvlh frame
    #[getter]
    fn get_qgcrf2lvlh(&self) -> PyQuaternion {
        self.0.qgcrf2lvlh().into()
    }

    /// Alias for pos_gcrf: 3-element GCRF position, meters
    #[getter]
    fn get_pos(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_pos_gcrf(py)
    }

    /// Alias for vel_gcrf: 3-element GCRF velocity, m/s
    #[getter]
    fn get_vel(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_vel_gcrf(py)
    }

    /// Add an impulsive maneuver (instantaneous delta-v) to the state
    ///
    /// Args:
    ///     time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to apply the maneuver
    ///     delta_v (array-like): 3-element delta-v vector [m/s]
    ///     frame (satkit.frame): Coordinate frame — **required**, no
    ///         default (matching the Rust API). Supported frames:
    ///
    ///         * ``frame.GCRF`` — inertial Cartesian
    ///         * ``frame.RTN`` — radial / tangential / normal (a.k.a.
    ///           RSW, RIC). T is perpendicular to R in the orbit plane;
    ///           for eccentric orbits it is **not** strictly along velocity.
    ///         * ``frame.NTW`` — normal-to-velocity / tangent / cross-track.
    ///           T is along velocity, so a pure +T delta-v of magnitude Δv
    ///           adds *exactly* Δv to |v|. Use this for prograde/retrograde
    ///           burns, especially on eccentric orbits.
    ///         * ``frame.LVLH`` — Local Vertical / Local Horizontal (classical
    ///           crewed-spaceflight frame).
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     ValueError: if delta_v does not have 3 elements, or the frame is
    ///         not one of the four above
    ///     TypeError: if time is not a time, or delta_v is not numeric
    ///
    /// See also:
    ///     :meth:`add_prograde`, :meth:`add_retrograde`, :meth:`add_radial`,
    ///     :meth:`add_normal` for ergonomic scalar-magnitude alternatives.
    fn add_maneuver(
        &mut self,
        time: TimeArg,
        delta_v: &Bound<'_, PyAny>,
        frame: PyFrame,
    ) -> Result<()> {
        let dv: Vector3 = crate::pyutils::py_to_smatrix(delta_v)?;
        let rust_frame: Frame = frame.into();
        // Refused here rather than by a later `propagate` (which only
        // checked the maneuvers inside its span) or by `pickle.dumps`
        check_maneuver_frame(rust_frame)?;
        self.0
            .add_maneuver(ImpulsiveManeuver::new(time.0, dv, rust_frame));
        Ok(())
    }

    /// Add a prograde impulsive burn (NTW +T axis, along velocity).
    ///
    /// A positive ``dv_mps`` adds energy (raises semi-major axis). The burn
    /// adds exactly ``dv_mps`` to |v| regardless of orbit eccentricity.
    ///
    /// Args:
    ///     time (satkit.time): Time at which to apply the burn
    ///     dv_mps (float): Magnitude along velocity vector [m/s]
    fn add_prograde(&mut self, time: TimeArg, dv_mps: f64) {
        self.0
            .add_maneuver(ImpulsiveManeuver::prograde(time.0, dv_mps));
    }

    /// Add a retrograde impulsive burn (NTW -T axis, opposite velocity).
    ///
    /// Equivalent to ``add_prograde`` with a negated magnitude. ``dv_mps``
    /// should be positive; a positive value removes energy from the orbit.
    ///
    /// Args:
    ///     time (satkit.time): Time at which to apply the burn
    ///     dv_mps (float): Magnitude along anti-velocity vector [m/s]
    fn add_retrograde(&mut self, time: TimeArg, dv_mps: f64) {
        self.0
            .add_maneuver(ImpulsiveManeuver::retrograde(time.0, dv_mps));
    }

    /// Add a radial-outward impulsive burn (NTW +N axis).
    ///
    /// For circular orbits this is the outward radial direction. For
    /// eccentric orbits the N axis leans off the radial by the flight-path
    /// angle.
    ///
    /// Args:
    ///     time (satkit.time): Time at which to apply the burn
    ///     dv_mps (float): Magnitude along in-plane normal-to-velocity [m/s]
    fn add_radial(&mut self, time: TimeArg, dv_mps: f64) {
        self.0
            .add_maneuver(ImpulsiveManeuver::radial_out(time.0, dv_mps));
    }

    /// Add a cross-track ("normal") impulsive burn (NTW +W axis).
    ///
    /// Positive values push in the +angular-momentum direction. Changes
    /// orbit inclination without altering energy (at apsides).
    ///
    /// Args:
    ///     time (satkit.time): Time at which to apply the burn
    ///     dv_mps (float): Magnitude along angular momentum direction [m/s]
    fn add_normal(&mut self, time: TimeArg, dv_mps: f64) {
        self.0
            .add_maneuver(ImpulsiveManeuver::normal(time.0, dv_mps));
    }

    /// Remove every scheduled maneuver from this state
    ///
    /// Returns:
    ///     None
    fn clear_maneuvers(&mut self) {
        self.0.maneuvers.clear();
    }

    /// Number of impulsive maneuvers scheduled on this state
    ///
    /// Returns:
    ///     int: Number of impulsive maneuvers (``len(maneuvers)``)
    #[getter]
    fn get_num_maneuvers(&self) -> usize {
        self.0.maneuvers.len()
    }

    /// The scheduled impulsive maneuvers, in the order they were added
    ///
    /// A new list of new dicts on every access: changing it does not change
    /// the state. Use the ``add_*`` methods to add maneuvers and
    /// ``clear_maneuvers`` to remove them.
    ///
    /// Returns:
    ///     list[dict]: One dict per maneuver with keys "time" (satkit.time),
    ///     "delta_v" (3-element numpy array, m/s, in "frame"), and
    ///     "frame" (satkit.frame)
    #[getter]
    fn get_maneuvers(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.0
            .maneuvers
            .iter()
            .map(|m| -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                d.set_item("time", PyInstant(m.time))?;
                d.set_item("delta_v", crate::pyutils::vec2py(py, &m.delta_v)?)?;
                d.set_item("frame", crate::pyframes::PyFrame::from(m.frame))?;
                d.into_py_any(py)
            })
            .collect()
    }

    /// Propagate state to a new time
    ///
    /// The covariance, if set, is propagated with the state transition
    /// matrix. Propagation is split at each scheduled maneuver whose time t
    /// satisfies ``start <= t < target`` (``target <= t < start`` when
    /// propagating backward, which subtracts the delta-v): a state at a
    /// maneuver's own time is the state just before the burn, so a maneuver
    /// at the current time is applied and one at the target time is not. The
    /// maneuver list is carried over to the returned state unchanged.
    ///
    /// Args:
    ///     timedur (satkit.time | datetime.datetime | numpy.datetime64 | satkit.duration): Time for which to
    ///         compute the new state, or a duration to propagate from the current time
    ///     propsettings (satkit.propsettings, optional): Propagation settings. Default: ``propsettings()``
    ///     satproperties (satkit.satproperties, optional): Satellite properties (drag, SRP, thrust)
    ///
    /// Returns:
    ///     satkit.satstate: New state at input time
    ///
    /// Raises:
    ///     TypeError: if timedur is not a time or a satkit.duration (a float
    ///         or a datetime.timedelta is refused), or propsettings /
    ///         satproperties has the wrong type
    ///     ValueError: if the position is all zeros
    ///     RuntimeError: if the propagation fails (see ``satkit.propagate``).
    ///         A covariance needs the state transition matrix, which
    ///         ``integrator.rodas4`` and ``integrator.gauss_jackson8`` cannot
    ///         propagate
    #[pyo3(signature=(timedur, *, propsettings=None, satproperties=None))]
    fn propagate(
        &self,
        timedur: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = propsettings_arg)] propsettings: Option<PyPropSettings>,
        #[pyo3(from_py_with = satproperties_arg)] satproperties: Option<PySatProperties>,
    ) -> Result<Self> {
        let time: Instant = {
            if let Ok(dur) = timedur.cast::<PyDuration>() {
                self.0.time + dur.borrow().0
            } else if let Ok(t) = timedur.extract::<TimeArg>() {
                t.0
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "timedur must be satkit.time, datetime.datetime or satkit.duration, got {}",
                    timedur.get_type()
                ))
                .into());
            }
        };
        let propsettings = propsettings.map(|p| p.0);

        let satprops_ref = satproperties
            .as_ref()
            .map(|s| &s.0 as &dyn satkit::orbitprop::SatProperties);

        Ok(Self(self.0.propagate(
            &time,
            propsettings.as_ref(),
            satprops_ref,
        )?))
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tm = PyInstant(Instant::INVALID).into_py_any(py).unwrap();

        let pos = np::PyArray1::from_slice(py, &[0.0, 0.0, 0.0])
            .into_py_any(py)
            .unwrap();
        let vel = np::PyArray1::from_slice(py, &[0.0, 0.0, 0.0])
            .into_py_any(py)
            .unwrap();
        (PyTuple::new(py, vec![tm, pos, vel]).unwrap(), d)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let state = state.extract::<&[u8]>(py)?;
        // Self-describing format:
        //   [0]      version byte (1 or 2)
        //   [1..9]   time: v2 = Instant raw i64 microseconds (exact);
        //            v1 (satkit 0.20 – 0.23) = TAI MJD as f64, which lost a
        //            microsecond ~1% of the time, still read (rounded)
        //   [9..57]  position + velocity (6 f64)
        //   [57]     covariance flag (0 = none, 1 = 6x6 PVCov follows)
        //   [..]     if flag == 1: 36 f64 covariance (288 bytes)
        //   [..]     maneuver count (u32 little-endian)
        //   [..]     count * 33-byte maneuvers: 8 time (as above), 24 delta-v,
        //            1 frame tag
        const HEADER: usize = 1 + 8 + 48 + 1; // version + time + pv + cov flag
        if state.len() < HEADER {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid satstate pickle: truncated header",
            ));
        }
        let version = state[0];
        if version != 1 && version != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported satstate pickle version {} (expected 1 or 2)",
                version
            )));
        }
        let read_f64 = |at: usize| f64::from_le_bytes(state[at..at + 8].try_into().unwrap());
        let read_time = |at: usize| match version {
            1 => Instant::from_mjd_with_scale(read_f64(at), satkit::TimeScale::TAI),
            _ => Instant::new(i64::from_le_bytes(state[at..at + 8].try_into().unwrap())),
        };

        self.0.time = read_time(1);
        let mut pv = [0.0f64; 6];
        for (i, v) in pv.iter_mut().enumerate() {
            *v = read_f64(9 + i * 8);
        }
        self.0.pv = Vector6::from_slice(&pv);
        self.0.maneuvers.clear();

        let mut offset = HEADER;
        self.0.cov = match state[57] {
            0 => StateCov::None,
            1 => {
                if state.len() < offset + 288 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "invalid satstate pickle: truncated covariance",
                    ));
                }
                let mut cov = [0.0f64; 36];
                for (i, v) in cov.iter_mut().enumerate() {
                    *v = read_f64(offset + i * 8);
                }
                offset += 288;
                StateCov::PVCov(Matrix6::from_slice(&cov))
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid satstate pickle: unknown covariance flag {other}"
                )));
            }
        };

        if state.len() < offset + 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid satstate pickle: missing maneuver count",
            ));
        }
        let count = u32::from_le_bytes(state[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if state.len() != offset + count * 33 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid satstate pickle: maneuver block length mismatch",
            ));
        }
        for _ in 0..count {
            let t = read_time(offset);
            offset += 8;
            let mut dv = [0.0f64; 3];
            for (i, v) in dv.iter_mut().enumerate() {
                *v = read_f64(offset + i * 8);
            }
            offset += 24;
            let frame = crate::pyutils::maneuver_frame_from_u8(state[offset])?;
            offset += 1;
            self.0
                .add_maneuver(ImpulsiveManeuver::new(t, Vector3::from_slice(&dv), frame));
        }

        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // See `__setstate__` for the format. Values are written little-endian via
        // `to_le_bytes`, so there is no alignment assumption on the buffer.
        let cov_len: usize = match self.0.cov {
            StateCov::None => 0,
            StateCov::PVCov(_) => 288,
        };
        let len = (1 + 8 + 48 + 1) + cov_len + 4 + self.0.maneuvers.len() * 33;
        let mut buffer: Vec<u8> = Vec::with_capacity(len);

        buffer.push(2u8); // version
        buffer.extend_from_slice(&self.0.time.raw.to_le_bytes());
        for v in self.0.pv.as_slice() {
            buffer.extend_from_slice(&v.to_le_bytes());
        }
        match self.0.cov {
            StateCov::None => buffer.push(0),
            StateCov::PVCov(cov) => {
                buffer.push(1);
                for v in cov.as_slice() {
                    buffer.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        buffer.extend_from_slice(&(self.0.maneuvers.len() as u32).to_le_bytes());
        for m in &self.0.maneuvers {
            buffer.extend_from_slice(&m.time.raw.to_le_bytes());
            for v in m.delta_v.as_slice() {
                buffer.extend_from_slice(&v.to_le_bytes());
            }
            buffer.push(crate::pyutils::maneuver_frame_to_u8(m.frame)?);
        }
        pyo3::types::PyBytes::new(py, &buffer).into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}
