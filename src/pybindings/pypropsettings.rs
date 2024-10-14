use pyo3::prelude::*;

use crate::orbitprop::PropSettings;
use crate::pybindings::PyAstroTime;

#[pyclass(name = "propsettings")]
#[derive(Clone, Debug)]
pub struct PyPropSettings {
    pub inner: PropSettings,
}

#[pymethods]
impl PyPropSettings {
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(PyPropSettings {
            inner: PropSettings::default(),
        })
    }

    #[getter]
    fn get_abs_error(&self) -> f64 {
        self.inner.abs_error
    }

    #[setter(abs_error)]
    fn set_abs_error(&mut self, val: f64) -> PyResult<()> {
        self.inner.abs_error = val;
        Ok(())
    }

    #[getter]
    fn get_rel_error(&self) -> f64 {
        self.inner.rel_error
    }

    #[setter(rel_error)]
    fn set_rel_error(&mut self, val: f64) -> PyResult<()> {
        self.inner.rel_error = val;
        Ok(())
    }

    #[getter]
    fn get_gravity_order(&self) -> u16 {
        self.inner.gravity_order
    }

    #[setter(gravity_order)]
    fn set_gravity_order(&mut self, val: u16) -> PyResult<()> {
        self.inner.gravity_order = val;
        Ok(())
    }

    #[getter]
    fn get_enable_interp(&self) -> bool {
        self.inner.enable_interp
    }

    #[setter(enable_interp)]
    fn set_enable_interp(&mut self, val: bool) -> PyResult<()> {
        self.inner.enable_interp = val;
        Ok(())
    }

    #[getter]
    fn get_use_spaceweather(&self) -> bool {
        self.inner.use_spaceweather
    }

    #[setter(use_spaceweather)]
    fn set_use_spacewather(&mut self, val: bool) -> PyResult<()> {
        self.inner.use_spaceweather = val;
        Ok(())
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.to_string())
    }

    fn precompute_terms(&mut self, start: &PyAstroTime, stop: &PyAstroTime) -> PyResult<()> {
        match self.inner.precompute_terms(&start.inner, &stop.inner) {
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            Ok(_) => Ok(()),
        }
    }
}

impl From<&PyPropSettings> for PropSettings {
    fn from(item: &PyPropSettings) -> PropSettings {
        item.inner.clone()
    }
}

impl From<&PropSettings> for PyPropSettings {
    fn from(item: &PropSettings) -> PyPropSettings {
        PyPropSettings {
            inner: item.clone(),
        }
    }
}
