use crate::orbitprop::SatPropertiesStatic;

use super::pyutils::kwargs_or_default;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString, PyTuple};

#[pyclass(name = "satproperties_static", module = "satkit")]
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
    fn new(args: &Bound<PyTuple>, mut kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut craoverm: f64 = 0.0;
        let mut cdaoverm: f64 = 0.0;

        if args.len() > 0 {
            craoverm = args.get_item(0)?.extract::<f64>()?;
        }
        if args.len() > 1 {
            cdaoverm = args.get_item(1)?.extract::<f64>()?;
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

    /// Get the satellite's susceptibility to radiation pressure
    ///
    /// Returns:
    ///     float: Cr A / m (m^2/kg)
    #[getter]
    fn get_craoverm(&self) -> f64 {
        self.inner.craoverm
    }

    /// Get the satellite's susceptibility to drag
    ///
    /// Returns:
    ///     float: Cd A / m (m^2/kg)
    #[getter]
    fn get_cdaoverm(&self) -> f64 {
        self.inner.cdaoverm
    }

    /// Set the satellite's susceptibility to radiation pressure
    ///
    /// Args:
    ///     craoverm (float): Cr A / m (m^2/kg)
    #[setter]
    fn set_craoverm(&mut self, craoverm: f64) -> PyResult<()> {
        self.inner.craoverm = craoverm;
        Ok(())
    }

    /// Set the satellite's susceptibility to drag
    ///
    /// Args:
    ///     cdaoverm (float): Cd A / m (m^2/kg)
    #[setter]
    fn set_cdaoverm(&mut self, cdaoverm: f64) -> PyResult<()> {
        self.inner.cdaoverm = cdaoverm;
        Ok(())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let state = state.as_bytes(py);
        if state.len() != 16 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let craoverm = f64::from_le_bytes(state[0..8].try_into()?);
        let cdaoverm = f64::from_le_bytes(state[8..16].try_into()?);
        self.inner.cdaoverm = cdaoverm;
        self.inner.craoverm = craoverm;
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        let mut raw = [0 as u8; 16];
        raw[0..8].clone_from_slice(&self.inner.craoverm.to_le_bytes());
        raw[8..16].clone_from_slice(&self.inner.cdaoverm.to_le_bytes());
        Ok(pyo3::types::PyBytes::new_bound(py, &raw).to_object(py))
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.to_string())
    }
}
