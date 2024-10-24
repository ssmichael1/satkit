use super::pyastrotime::PyAstroTime;
use super::pypropsettings::PyPropSettings;
use super::pyquaternion::Quaternion;

use nalgebra as na;
use numpy as np;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone, PyTuple};

use crate::orbitprop::{PropSettings, SatState, StateCov};
use crate::pybindings::PyDuration;
use crate::AstroTime;

#[pyclass(name = "satstate", module = "satkit")]
#[derive(Clone, Debug)]
pub struct PySatState {
    inner: SatState,
}

#[pymethods]
impl PySatState {
    #[new]
    #[pyo3(signature=(time, pos, vel, cov=None))]
    fn py_new(
        time: &PyAstroTime,
        pos: &Bound<'_, np::PyArray1<f64>>,
        vel: &Bound<'_, np::PyArray1<f64>>,
        cov: Option<&Bound<'_, np::PyArray2<f64>>>,
    ) -> PyResult<Self> {
        if pos.len() != 3 || vel.len() != 3 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Position and velocity must be 1-d numpy arrays with length 3",
            ));
        }

        let mut state = SatState::from_pv(
            &time.inner,
            &na::Vector3::<f64>::from_row_slice(unsafe { pos.as_slice().unwrap() }),
            &na::Vector3::<f64>::from_row_slice(unsafe { vel.as_slice().unwrap() }),
        );
        if cov.is_some() {
            let cov = cov.unwrap();
            let dims = cov.dims();
            if dims[0] != 6 || dims[1] != 6 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Covariance must be 6x6 matrix",
                ));
            }
            let nacov = na::Matrix6::<f64>::from_row_slice(unsafe { cov.as_slice().unwrap() });
            state.set_cov(StateCov::PVCov(nacov));
        }

        Ok(PySatState { inner: state })
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
        let na_sigma_pvh =
            na::Vector3::<f64>::from_row_slice(unsafe { sigma_pvh.as_slice().unwrap() });

        self.inner.set_lvlh_pos_uncertainty(&na_sigma_pvh);
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
    ) -> PyResult<()> {
        if sigma_cart.len() != 3 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Position uncertainty must be 1-d numpy array with length 3",
            ));
        }
        let na_sigma_cart =
            na::Vector3::<f64>::from_row_slice(unsafe { sigma_cart.as_slice().unwrap() });

        self.inner.set_gcrf_pos_uncertainty(&na_sigma_cart);
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
        let na_cov = na::Matrix6::from_row_slice(unsafe { cov.as_slice().unwrap() });
        self.inner.cov = StateCov::PVCov(na_cov);
        Ok(())
    }

    #[getter]
    fn get_time(&self) -> PyAstroTime {
        PyAstroTime {
            inner: self.inner.time,
        }
    }

    #[getter]
    fn get_pos_gcrf(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice_bound(py, self.inner.pv.fixed_view::<3, 1>(0, 0).as_slice())
                .to_object(py)
        })
    }

    #[getter]
    fn get_vel_gcrf(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice_bound(py, self.inner.pv.fixed_view::<3, 1>(3, 0).as_slice())
                .to_object(py)
        })
    }

    /// Get full 6x6 state covariance matrix
    ///
    /// Returns:
    ///     numpy.ndarray: 6x6 numpy array with state covariance matrix for position (meters) and velocity (m/s)
    #[getter]
    fn get_cov(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            match self.inner.cov {
                StateCov::None => PyNone::get_bound(py).to_object(py),
                StateCov::PVCov(cov) => {
                    let dims = vec![6, 6];
                    np::PyArray1::from_slice_bound(py, cov.as_slice())
                        .reshape(dims)
                        .unwrap()
                        .to_object(py)
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
    fn get_qgcrf2lvlh(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.qgcrf2lvlh(),
        }
    }

    /// Return position (meters) in GCRF frame
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array with position in GCRF frame.  Units are meters
    #[getter]
    fn get_pos(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice_bound(py, self.inner.pv.fixed_view::<3, 1>(0, 0).as_slice())
                .to_object(py)
        })
    }
    #[getter]
    fn get_vel(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice_bound(py, self.inner.pv.fixed_view::<3, 1>(3, 0).as_slice())
                .to_object(py)
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
    ) -> PyResult<Self> {
        let time: AstroTime = {
            if timedur.is_instance_of::<PyAstroTime>() {
                timedur.extract::<PyAstroTime>()?.inner
            } else if timedur.is_instance_of::<PyDuration>() {
                let dur = timedur.extract::<PyDuration>()?;
                self.inner.time + dur.inner
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "timedur must be satkit.time or satkit.duration",
                ));
            }
        };

        let propsettings: Option<PropSettings> = match kwargs.is_some() {
            true => {
                let kw = kwargs.unwrap();
                match kw.get_item("propsettings")? {
                    None => None,
                    Some(v) => Some(v.extract::<PyPropSettings>()?.inner),
                }
            }
            false => None,
        };

        match self.inner.propagate(&time, propsettings.as_ref()) {
            Ok(s) => Ok(PySatState { inner: s }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Error propagating state: {}",
                e
            ))),
        }
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new_bound(py);
        let tm = PyAstroTime {
            inner: AstroTime { mjd_tai: 0.0 },
        }
        .into_py(py);
        let pos = np::PyArray1::from_slice_bound(py, &[0.0, 0.0, 0.0]).to_object(py);
        let vel = np::PyArray1::from_slice_bound(py, &[0.0, 0.0, 0.0]).to_object(py);
        (PyTuple::new_bound(py, vec![tm, pos, vel]), d)
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let state = state.extract::<&[u8]>(py)?;
        if state.len() < 56 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "State must be at least 56 bytes",
            ));
        }
        let time = AstroTime::from_mjd(
            f64::from_le_bytes(state[0..8].try_into().unwrap()),
            crate::TimeScale::TAI,
        );

        let pv = na::Vector6::<f64>::from_row_slice(unsafe {
            std::slice::from_raw_parts(state[8..56].as_ptr() as *const f64, 6)
        });
        self.inner.time = time;
        self.inner.pv = pv;
        if state.len() >= 92 {
            let cov = na::Matrix6::<f64>::from_row_slice(unsafe {
                std::slice::from_raw_parts(state[56..].as_ptr() as *const f64, 36)
            });
            self.inner.cov = StateCov::PVCov(cov);
        }
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let len: usize = 56
            + match self.inner.cov {
                StateCov::None => 0,
                StateCov::PVCov(_) => 36,
            };
        let mut buffer: Vec<u8> = vec![0; len];
        buffer[0..8].clone_from_slice(&self.inner.time.to_mjd(crate::TimeScale::TAI).to_le_bytes());
        unsafe {
            buffer[8..56].clone_from_slice(std::slice::from_raw_parts(
                self.inner.pv.as_ptr() as *const u8,
                48,
            ));
        }
        if let StateCov::PVCov(cov) = self.inner.cov {
            unsafe {
                buffer[56..].clone_from_slice(std::slice::from_raw_parts(
                    cov.as_ptr() as *const u8,
                    36 * 8,
                ));
            }
        }
        Ok(pyo3::types::PyBytes::new_bound(py, &buffer).to_object(py))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
