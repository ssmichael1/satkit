use super::pyastrotime::PyAstroTime;
use super::pypropsettings::PyPropSettings;
use super::pyquaternion::Quaternion;

use nalgebra as na;
use numpy as np;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone};

use crate::orbitprop::{PropSettings, SatState, StateCov};

#[pyclass(name = "satstate")]
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
        pos: &np::PyArray1<f64>,
        vel: &np::PyArray1<f64>,
        cov: Option<&np::PyArray2<f64>>,
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

    /// Set position uncertainty (1-sigma, meters) in the
    /// lvlh (local-vertical, local-horizontal) frame
    ///
    /// # Arguments
    ///
    /// * `sigma_lvlh` - 3-vector with 1-sigma position uncertainty in LVLH frame
    fn set_lvlh_pos_uncertainty(&mut self, sigma_pvh: &np::PyArray1<f64>) -> PyResult<()> {
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

    /// Set position uncertainty (1-sigma, meters) in the
    /// gcrf (Geocentric Celestial Reference Frame)
    ///
    /// # Arguments
    ///
    /// * `sigma_gcrf` - 3-vector with 1-sigma position uncertainty in GCRF frame    
    ///
    fn set_gcrf_pos_uncertainty(&mut self, sigma_cart: &np::PyArray1<f64>) -> PyResult<()> {
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

    #[setter]
    fn set_cov(&mut self, cov: &np::PyArray2<f64>) -> PyResult<()> {
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
    fn get_cov(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            match self.inner.cov {
                StateCov::None => PyNone::get(py).to_object(py),
                StateCov::PVCov(cov) => {
                    let dims = vec![6 as usize, 6 as usize];
                    np::PyArray1::from_slice(py, cov.as_slice())
                        .reshape(dims)
                        .unwrap()
                        .to_object(py)
                }
            }
        })
    }

    /// Quaternion to go from gcrf (Geocentric Celestial Reference Frame)
    /// to lvlh (Local-Vertical, Local-Horizontal) frame
    ///
    /// Note: lvlh:
    ///       z axis = -r (nadir)
    ///       y axis = -h (h = p cross v)
    ///       x axis such that x cross y = z
    #[getter]
    fn get_qgcrf2lvlh(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.qgcrf2lvlh(),
        }
    }

    #[getter]
    fn get_pos(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, self.inner.pv.fixed_view::<3, 1>(0, 0).as_slice())
                .to_object(py)
        })
    }
    #[getter]
    fn get_vel(&self) -> Py<PyAny> {
        pyo3::Python::with_gil(|py| -> Py<PyAny> {
            np::PyArray1::from_slice(py, self.inner.pv.fixed_view::<3, 1>(3, 0).as_slice())
                .to_object(py)
        })
    }

    ///
    /// Propagate state to a new time
    ///
    /// # Arguments:
    ///
    /// * `time` - Time for which to compute new state
    /// * `propsettings` - optional Settings for the propagator
    ///
    #[pyo3(signature=(time, **kwargs))]
    fn propagate(&self, time: &PyAstroTime, kwargs: Option<&PyDict>) -> PyResult<Self> {
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

        match self.inner.propagate(&time.inner, propsettings.as_ref()) {
            Ok(s) => Ok(PySatState { inner: s }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Error propagating state: {}",
                e.to_string()
            ))),
        }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.to_string())
    }
}
