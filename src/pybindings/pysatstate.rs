use super::pyinstant::PyInstant;
use super::pypropsettings::PyPropSettings;
use super::pyquaternion::PyQuaternion;

use numpy as np;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::mathtypes::*;
use crate::orbitprop::{PropSettings, SatState, StateCov};
use crate::pybindings::PyDuration;
use crate::Instant;

use anyhow::{bail, Result};

#[pyclass(name = "satstate", module = "satkit")]
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
            &nalgebra::vector![
                pos.get_owned(0).unwrap(),
                pos.get_owned(1).unwrap(),
                pos.get_owned(2).unwrap()
            ],
            &nalgebra::vector![
                vel.get_owned(0).unwrap(),
                vel.get_owned(1).unwrap(),
                vel.get_owned(2).unwrap()
            ],
        );

        if let Some(cov) = cov {
            let dims = cov.dims();
            if dims[0] != 6 || dims[1] != 6 {
                bail!("Covariance must be 6x6 numpy array");
            }
            let nacov = Matrix6::from_row_slice(unsafe { cov.as_slice().unwrap() });
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
        let na_sigma_pvh = Vector3::from_row_slice(unsafe { sigma_pvh.as_slice().unwrap() });

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
        let na_sigma_cart = Vector3::from_row_slice(unsafe { sigma_cart.as_slice().unwrap() });

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
        let na_cov = Matrix6::from_row_slice(unsafe { cov.as_slice().unwrap() });
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
            np::PyArray1::from_slice(py, self.0.pv.fixed_view::<3, 1>(0, 0).as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }

    #[getter]
    fn get_vel_gcrf(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, self.0.pv.fixed_view::<3, 1>(3, 0).as_slice())
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
    ///     lvlh coordinate system:
    ///     * z axis = -r (nadir)
    ///     * y axis = -h (h = p cross v)
    ///     * x axis such that x cross y = z
    ///
    /// Returns:
    ///     satkit.quaternion: quaternion to go from gcrf to lvlh frame
    #[getter]
    fn get_qgcrf2lvlh(&self) -> PyQuaternion {
        self.0.qgcrf2lvlh().into()
    }

    /// Return position (meters) in GCRF frame
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array with position in GCRF frame.  Units are meters
    #[getter]
    fn get_pos(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, self.0.pv.fixed_view::<3, 1>(0, 0).as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }
    #[getter]
    fn get_vel(&self) -> Py<PyAny> {
        pyo3::Python::attach(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, self.0.pv.fixed_view::<3, 1>(3, 0).as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }

    /// Propagate state to a new time
    ///
    /// Args:
    ///     time (satkit.time|satkit.duration): Time for which to compute new state or alternatively
    ///     a duration to propagate from the current time
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

        let propsettings: Option<PropSettings> = match kwargs.is_some() {
            true => {
                let kw = kwargs.unwrap();
                match kw.get_item("propsettings")? {
                    None => None,
                    Some(v) => Some(
                        v.extract::<PyPropSettings>()
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Invalid propsettings: {}",
                                    e
                                ))
                            })?
                            .0,
                    ),
                }
            }
            false => None,
        };

        self.0.propagate(&time, propsettings.as_ref()).map(Self)
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
            crate::TimeScale::TAI,
        );

        let pv = Vector6::from_row_slice(unsafe {
            std::slice::from_raw_parts(state[8..56].as_ptr() as *const f64, 6)
        });
        self.0.time = time;
        self.0.pv = pv;
        if state.len() >= 92 {
            let cov = Matrix6::from_row_slice(unsafe {
                std::slice::from_raw_parts(state[56..].as_ptr() as *const f64, 36)
            });
            self.0.cov = StateCov::PVCov(cov);
        }
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let len: usize = 56
            + match self.0.cov {
                StateCov::None => 0,
                StateCov::PVCov(_) => 36,
            };
        let mut buffer: Vec<u8> = vec![0; len];
        buffer[0..8].clone_from_slice(
            &self
                .0
                .time
                .as_mjd_with_scale(crate::TimeScale::TAI)
                .to_le_bytes(),
        );
        unsafe {
            buffer[8..56].clone_from_slice(std::slice::from_raw_parts(
                self.0.pv.as_ptr() as *const u8,
                48,
            ));
        }
        if let StateCov::PVCov(cov) = self.0.cov {
            unsafe {
                buffer[56..].clone_from_slice(std::slice::from_raw_parts(
                    cov.as_ptr() as *const u8,
                    36 * 8,
                ));
            }
        }
        pyo3::types::PyBytes::new(py, &buffer).into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}
