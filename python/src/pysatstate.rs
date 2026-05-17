use crate::pyframes::PyFrame;
use crate::pyinstant::PyInstant;
use crate::pypropsettings::PyPropSettings;
use crate::pyquaternion::PyQuaternion;
use crate::pysatproperties::PySatProperties;
use crate::PyDuration;

use numpy as np;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone, PyTuple};
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;
use satkit::orbitprop::{ImpulsiveManeuver, PropSettings, SatState, StateCov};
use satkit::{Frame, Instant};

use anyhow::{bail, Result};

#[pyclass(name = "satstate", module = "satkit", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySatState(SatState);

#[pymethods]
impl PySatState {
    #[new]
    #[pyo3(signature=(time, pos, vel, cov=None))]
    fn py_new(
        time: &PyInstant,
        pos: &Bound<'_, np::PyArray1<f64>>,
        vel: &Bound<'_, np::PyArray1<f64>>,
        cov: Option<&Bound<'_, np::PyArray2<f64>>>,
    ) -> Result<Self> {
        if pos.len() != 3 || vel.len() != 3 {
            bail!("Position and velocity must be 1-d numpy arrays with length 3");
        }

        let mut state = SatState::from_pv(
            &time.0,
            &numeris::vector![
                pos.get_owned(0).unwrap(),
                pos.get_owned(1).unwrap(),
                pos.get_owned(2).unwrap(),
            ],
            &numeris::vector![
                vel.get_owned(0).unwrap(),
                vel.get_owned(1).unwrap(),
                vel.get_owned(2).unwrap(),
            ],
        );

        if let Some(cov) = cov {
            let dims = cov.dims();
            if dims[0] != 6 || dims[1] != 6 {
                bail!("Covariance must be 6x6 numpy array");
            }
            let nacov = Matrix6::from_slice(unsafe { cov.as_slice().unwrap() }).transpose();
            state.set_cov(StateCov::PVCov(nacov));
        }

        Ok(Self(state))
    }

    /// Set 1-sigma position uncertainty in a satellite-local or inertial frame.
    ///
    /// The uncertainty is interpreted as a diagonal 3x3 covariance in the
    /// given ``frame`` (axes aligned with the frame), rotated into GCRF,
    /// and stored in the position block of the 6x6 state covariance. Any
    /// existing velocity covariance is preserved.
    ///
    /// Args:
    ///     sigma (numpy.ndarray): 3-element array of 1-sigma position
    ///         components along the frame's axes. Units: meters.
    ///     frame (satkit.frame): Coordinate frame — **required**, no
    ///         default (matching the Rust API). Supported values:
    ///         ``frame.GCRF``, ``frame.LVLH``, ``frame.RTN`` (= RSW = RIC),
    ///         ``frame.NTW``.
    ///
    /// Raises:
    ///     RuntimeError: if the frame is not one of the supported
    ///         orbital / inertial frames above.
    fn set_pos_uncertainty(
        &mut self,
        sigma: &Bound<'_, np::PyArray1<f64>>,
        frame: PyFrame,
    ) -> Result<()> {
        if sigma.len() != 3 {
            bail!("Position uncertainty must be 1-d numpy array with length 3");
        }
        let na_sigma = Vector3::from_slice(unsafe { sigma.as_slice().unwrap() });
        let rust_frame: Frame = frame.into();
        self.0.set_pos_uncertainty(&na_sigma, rust_frame)?;
        Ok(())
    }

    /// Set 1-sigma velocity uncertainty in a satellite-local or inertial frame.
    ///
    /// Analogous to :meth:`set_pos_uncertainty`, but for the velocity
    /// block of the 6x6 state covariance. Any existing position
    /// covariance is preserved.
    ///
    /// Args:
    ///     sigma (numpy.ndarray): 3-element array of 1-sigma velocity
    ///         components along the frame's axes. Units: m/s.
    ///     frame (satkit.frame): Coordinate frame — **required**, no
    ///         default (matching the Rust API). Supported values:
    ///         ``frame.GCRF``, ``frame.LVLH``, ``frame.RTN``, ``frame.NTW``.
    ///
    /// Raises:
    ///     RuntimeError: if the frame is not one of the supported
    ///         orbital / inertial frames above.
    fn set_vel_uncertainty(
        &mut self,
        sigma: &Bound<'_, np::PyArray1<f64>>,
        frame: PyFrame,
    ) -> Result<()> {
        if sigma.len() != 3 {
            bail!("Velocity uncertainty must be 1-d numpy array with length 3");
        }
        let na_sigma = Vector3::from_slice(unsafe { sigma.as_slice().unwrap() });
        let rust_frame: Frame = frame.into();
        self.0.set_vel_uncertainty(&na_sigma, rust_frame)?;
        Ok(())
    }

    /// Set full 6x6 state covariance matrix
    ///
    /// Args:
    ///     cov (numpy.ndarray): 6x6 numpy array with state covariance matrix for position (meters) and velocity (m/s)
    ///
    /// Returns:
    ///     None
    #[setter]
    fn set_cov(&mut self, cov: &Bound<'_, np::PyArray2<f64>>) -> PyResult<()> {
        if cov.readonly().shape()[0] != 6 || cov.readonly().shape()[1] != 6 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Covariance must be 6x6 numpy array",
            ));
        }
        let na_cov = Matrix6::from_slice(unsafe { cov.as_slice().unwrap() }).transpose();
        self.0.cov = StateCov::PVCov(na_cov);
        Ok(())
    }

    #[getter]
    fn get_time(&self) -> PyInstant {
        PyInstant(self.0.time)
    }

    #[getter]
    fn get_pos_gcrf(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, &self.0.pv.as_slice()[0..3])
                .into_py_any(py)
                .unwrap()
        })
    }

    #[getter]
    fn get_vel_gcrf(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, &self.0.pv.as_slice()[3..6])
                .into_py_any(py)
                .unwrap()
        })
    }

    /// Get full 6x6 state covariance matrix
    ///
    /// Returns:
    ///     numpy.ndarray: 6x6 numpy array with state covariance matrix for position (meters) and velocity (m/s)
    #[getter]
    fn get_cov(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            match self.0.cov {
                StateCov::None => PyNone::get(py).into_py_any(py).unwrap(),
                StateCov::PVCov(cov) => {
                    let dims = vec![6, 6];
                    np::PyArray1::from_slice(py, cov.as_slice())
                        .reshape(dims)
                        .unwrap()
                        .into_py_any(py)
                        .unwrap()
                }
            }
        })
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

    /// Alias for pos_gcrf
    #[getter]
    fn get_pos(&self) -> Py<PyAny> {
        self.get_pos_gcrf()
    }

    /// Alias for vel_gcrf
    #[getter]
    fn get_vel(&self) -> Py<PyAny> {
        self.get_vel_gcrf()
    }

    /// Add an impulsive maneuver (instantaneous delta-v) to the state
    ///
    /// Args:
    ///     time (satkit.time): Time at which to apply the maneuver
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
    /// See also:
    ///     :meth:`add_prograde`, :meth:`add_retrograde`, :meth:`add_radial`,
    ///     :meth:`add_normal` for ergonomic scalar-magnitude alternatives.
    fn add_maneuver(
        &mut self,
        time: PyInstant,
        delta_v: &Bound<'_, PyAny>,
        frame: PyFrame,
    ) -> Result<()> {
        let dv: Vector3 = crate::pyutils::py_to_smatrix(delta_v)?;
        let rust_frame: Frame = frame.into();
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
    fn add_prograde(&mut self, time: PyInstant, dv_mps: f64) {
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
    fn add_retrograde(&mut self, time: PyInstant, dv_mps: f64) {
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
    fn add_radial(&mut self, time: PyInstant, dv_mps: f64) {
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
    fn add_normal(&mut self, time: PyInstant, dv_mps: f64) {
        self.0
            .add_maneuver(ImpulsiveManeuver::normal(time.0, dv_mps));
    }

    /// Get the list of maneuvers
    ///
    /// Returns:
    ///     int: Number of impulsive maneuvers
    #[getter]
    fn get_num_maneuvers(&self) -> usize {
        self.0.maneuvers.len()
    }

    /// Propagate state to a new time
    ///
    /// Automatically segments propagation at impulsive maneuver times.
    ///
    /// Args:
    ///     time (satkit.time|satkit.duration): Time for which to compute new state or alternatively
    ///         a duration to propagate from the current time
    ///     propsettings (satkit.propsettings, optional): Propagation settings
    ///     satproperties (satkit.satproperties, optional): Satellite properties (drag, SRP, thrust)
    ///
    /// Returns:
    ///     satkit.satstate: New state at input time
    #[pyo3(signature=(timedur, **kwargs))]
    fn propagate(
        &self,
        timedur: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Result<Self> {
        let time: Instant = {
            if timedur.is_instance_of::<PyInstant>() {
                timedur
                    .extract::<PyInstant>()
                    .map_err(|e| anyhow::anyhow!("Invalid instant: {}", e))?
                    .0
            } else if timedur.is_instance_of::<PyDuration>() {
                let dur = timedur
                    .extract::<PyDuration>()
                    .map_err(|e| anyhow::anyhow!("Invalid duration: {}", e))?;
                self.0.time + dur.0
            } else {
                bail!("1st argument must be satkit.time or satkit.duration");
            }
        };

        let mut propsettings: Option<PropSettings> = None;
        let mut satprops_obj: Option<PySatProperties> = None;

        if let Some(kw) = kwargs {
            if let Some(v) = kw.get_item("propsettings")? {
                propsettings = Some(
                    v.extract::<PyPropSettings>()
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid propsettings: {}",
                                e
                            ))
                        })?
                        .0,
                );
            }
            if let Some(v) = kw.get_item("satproperties")? {
                satprops_obj = Some(v.extract::<PySatProperties>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid satproperties: {}", e))
                })?);
            }
        }

        let satprops_ref = satprops_obj
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
        if state.len() < 56 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "State must be at least 56 bytes",
            ));
        }
        let time = Instant::from_mjd_with_scale(
            f64::from_le_bytes(state[0..8].try_into().unwrap()),
            satkit::TimeScale::TAI,
        );

        let pv = Vector6::from_slice(unsafe {
            std::slice::from_raw_parts(state[8..56].as_ptr() as *const f64, 6)
        });
        self.0.time = time;
        self.0.pv = pv;
        self.0.cov = StateCov::None;
        self.0.maneuvers.clear();

        let mut offset = 56;

        // Covariance (optional: 288 bytes = 36 f64s)
        if offset + 288 <= state.len() {
            // Check tag byte at end to distinguish cov from maneuvers
            // Format: if we have enough bytes for a full cov matrix, read it
            let cov = Matrix6::from_slice(unsafe {
                std::slice::from_raw_parts(state[offset..].as_ptr() as *const f64, 36)
            });
            // Check if this looks like a covariance (non-zero diagonal)
            // We use a version tag: bytes 56..57 == 0x01 means cov follows
            // For backwards compat: old format had cov immediately at offset 56
            // and total length was exactly 56 + 288 = 344 with no maneuvers
            self.0.cov = StateCov::PVCov(cov);
            offset += 288;
        }

        // Maneuvers: each is 8 (time) + 24 (delta_v) + 1 (frame tag) = 33 bytes
        while offset + 33 <= state.len() {
            let t = Instant::from_mjd_with_scale(
                f64::from_le_bytes(state[offset..offset + 8].try_into().unwrap()),
                satkit::TimeScale::TAI,
            );
            offset += 8;
            let dv = Vector3::from_slice(unsafe {
                std::slice::from_raw_parts(state[offset..].as_ptr() as *const f64, 3)
            });
            offset += 24;
            let frame = match state[offset] {
                0 => Frame::GCRF,
                1 => Frame::RTN,
                _ => Frame::GCRF,
            };
            offset += 1;
            self.0.add_maneuver(ImpulsiveManeuver::new(t, dv, frame));
        }

        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let cov_len: usize = match self.0.cov {
            StateCov::None => 0,
            StateCov::PVCov(_) => 288,
        };
        // Each maneuver: 8 (time) + 24 (delta_v) + 1 (frame tag) = 33 bytes
        let maneuver_len = self.0.maneuvers.len() * 33;
        let len = 56 + cov_len + maneuver_len;
        let mut buffer: Vec<u8> = vec![0; len];
        buffer[0..8].clone_from_slice(
            &self
                .0
                .time
                .as_mjd_with_scale(satkit::TimeScale::TAI)
                .to_le_bytes(),
        );
        unsafe {
            buffer[8..56].clone_from_slice(std::slice::from_raw_parts(
                self.0.pv.as_slice().as_ptr() as *const u8,
                48,
            ));
        }
        let mut offset = 56;
        if let StateCov::PVCov(cov) = self.0.cov {
            unsafe {
                buffer[offset..offset + 288].clone_from_slice(std::slice::from_raw_parts(
                    cov.as_slice().as_ptr() as *const u8,
                    288,
                ));
            }
            offset += 288;
        }
        for m in &self.0.maneuvers {
            buffer[offset..offset + 8].clone_from_slice(
                &m.time
                    .as_mjd_with_scale(satkit::TimeScale::TAI)
                    .to_le_bytes(),
            );
            offset += 8;
            unsafe {
                buffer[offset..offset + 24].clone_from_slice(std::slice::from_raw_parts(
                    m.delta_v.as_slice().as_ptr() as *const u8,
                    24,
                ));
            }
            offset += 24;
            buffer[offset] = match m.frame {
                Frame::RTN => 1,
                _ => 0,
            };
            offset += 1;
        }
        pyo3::types::PyBytes::new(py, &buffer).into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}
