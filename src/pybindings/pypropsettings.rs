use pyo3::prelude::*;

use crate::orbitprop::PropSettings;
use crate::pybindings::PyAstroTime;

use pyo3::types::{PyDict, PyString};

#[pyclass(name = "propsettings")]
#[derive(Clone, Debug)]
pub struct PyPropSettings {
    pub inner: PropSettings,
}

#[pymethods]
impl PyPropSettings {
    #[new]
    #[pyo3(signature=(**kwargs))]
    fn py_new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut ps = PropSettings::default();
        if let Some(kw) = kwargs {
            if let Some(abserr) = kw.get_item("abs_error")? {
                ps.abs_error = abserr.extract::<f64>()?;
                kw.del_item("abs_error")?;
            }
            if let Some(relerr) = kw.get_item("rel_error")? {
                ps.rel_error = relerr.extract::<f64>()?;
                kw.del_item("rel_error")?;
            }
            if let Some(gravorder) = kw.get_item("gravity_order")? {
                ps.gravity_order = gravorder.extract::<u16>()?;
                kw.del_item("gravity_order")?;
            }
            if let Some(interp) = kw.get_item("enable_iterp")? {
                ps.enable_interp = interp.extract::<bool>()?;
                kw.del_item("enable_interp")?;
            }
            if let Some(sw) = kw.get_item("use_spaceweather")? {
                ps.use_spaceweather = sw.extract::<bool>()?;
                kw.del_item("use_spaceweather")?;
            }
            if !kw.is_empty() {
                let keystring: String = kw.iter().fold(String::from(""), |acc, (k, _v)| {
                    let mut a2 = acc.clone();
                    a2.push_str(k.downcast::<PyString>().unwrap().to_str().unwrap());
                    a2.push_str(", ");
                    a2
                });
                let s = format!("Invalid kwargs: {}", keystring);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(s));
            }
        }

        Ok(PyPropSettings { inner: ps })
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
        self.inner.to_string()
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
