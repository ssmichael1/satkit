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

    /// Set position uncertainty (1-sigma, meters) in the lvlh (local-vertical, local-horizontal) frame
    ///
    /// Args:
    ///     sigma_lvlh (numpy.ndarray): 3-element numpy array with 1-sigma position uncertainty in LVLH frame.  Units are meters
    ///
    /// Returns:
    ///     None
    fn set_lvlh_pos_uncertainty(
        &mut self,
        sigma_pvh: &Bound<'_, np::PyArray1<f64>>,
    ) -> PyResult<()> {
        if sigma_pvh.len() != 3 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Position uncertainty must be 1-d numpy array with length 3",
            ));
        }
        let na_sigma_pvh = Vector3::from_slice(unsafe { sigma_pvh.as_slice().unwrap() });

        self.0.set_lvlh_pos_uncertainty(&na_sigma_pvh);
        Ok(())
    }

    /// Set position uncertainty (1-sigma, meters) in the gcrf (Geocentric Celestial Reference Frame)
    ///
    /// Args:
    ///     sigma_gcrf (numpy.ndarray): 3-element numpy array with 1-sigma position uncertainty in GCRF frame.  Units are meters
    ///
    /// Returns:
    ///     None
    fn set_gcrf_pos_uncertainty(
        &mut self,
        sigma_cart: &Bound<'_, np::PyArray1<f64>>,
    ) -> Result<()> {
        if sigma_cart.len() != 3 {
            bail!("Position uncertainty must be 1-d numpy array with length 3");
        }
        let na_sigma_cart = Vector3::from_slice(unsafe { sigma_cart.as_slice().unwrap() });

        self.0.set_gcrf_pos_uncertainty(&na_sigma_cart);
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
    ///     frame (satkit.frame, optional): Coordinate frame (default: frame.GCRF).
    ///         For frame.RIC, components are [R, I, C] where R = radial (outward),
    ///         I = in-track (along velocity), C = cross-track (along angular momentum)
    ///
    /// Returns:
    ///     None
    #[pyo3(signature=(time, delta_v, frame=PyFrame::GCRF))]
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

        self.0
            .propagate(&time, propsettings.as_ref(), satprops_ref)
            .map(Self)
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
                1 => Frame::RIC,
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
                &m.time.as_mjd_with_scale(satkit::TimeScale::TAI).to_le_bytes(),
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
                Frame::RIC => 1,
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
