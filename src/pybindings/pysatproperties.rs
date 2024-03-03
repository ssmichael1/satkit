use crate::orbitprop::SatPropertiesStatic;

use super::pyutils::kwargs_or_default;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

#[pyclass(name = "satproperties_static")]
#[derive(Clone, Debug)]
pub struct PySatProperties {
    pub inner: SatPropertiesStatic,
}

#[pymethods]
impl PySatProperties {
    ///
    /// Create a static sat properties object
    /// setting satellite susceptibility to
    /// drag & radiation pressure
    ///
    /// With Cr A / m (m^2/kg),  radiation pressure
    /// and Cd A / m (m^2/kg), drag pressure
    /// passed in as arguments in that order, or set explicitly
    /// via the "craoverm" and "cdaoverm" keyword arguments
    ///
    /// If these are not set, default is 0
    ///
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &PyTuple, mut kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut craoverm: f64 = 0.0;
        let mut cdaoverm: f64 = 0.0;

        if args.len() > 0 {
            craoverm = args[0].extract::<f64>()?;
        }
        if args.len() > 1 {
            cdaoverm = args[1].extract::<f64>()?;
        }

        if kwargs.is_some() {
            craoverm = kwargs_or_default(&mut kwargs, "craoverm", craoverm)?;
            cdaoverm = kwargs_or_default(&mut kwargs, "cdaoverm", cdaoverm)?;
            if !kwargs.unwrap().is_empty() {
                let keystring: String =
                    kwargs
                        .unwrap()
                        .iter()
                        .fold(String::from(""), |acc, (k, _v)| {
                            let mut a2 = acc.clone();
                            a2.push_str(k.downcast::<PyString>().unwrap().to_str().unwrap());
                            a2.push_str(", ");
                            a2
                        });
                let s = format!("Invalid keyword args: {}", keystring);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(s));
            }
        }

        Ok(PySatProperties {
            inner: SatPropertiesStatic::new(cdaoverm, craoverm),
        })
    }

    #[getter]
    fn get_craoverm(&self) -> f64 {
        self.inner.craoverm
    }

    #[getter]
    fn get_cdaoverm(&self) -> f64 {
        self.inner.cdaoverm
    }

    #[setter]
    fn set_craoverm(&mut self, craoverm: f64) -> PyResult<()> {
        self.inner.craoverm = craoverm;
        Ok(())
    }

    #[setter]
    fn set_cdaoverm(&mut self, cdaoverm: f64) -> PyResult<()> {
        self.inner.cdaoverm = cdaoverm;
        Ok(())
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.to_string())
    }
}
