use crate::Duration;

use super::pyastrotime::PyAstroTime;
use pyo3::prelude::*;

#[pyclass(name = "duration")]
#[derive(Clone)]
pub struct PyDuration {
    pub inner: Duration,
}

#[pymethods]
impl PyDuration {
    #[staticmethod]
    fn from_days(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Days(d),
        }
    }

    #[staticmethod]
    fn from_seconds(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Seconds(d),
        }
    }

    #[staticmethod]
    fn from_minutes(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Minutes(d),
        }
    }

    #[staticmethod]
    fn from_hours(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Hours(d),
        }
    }

    fn __add__(&self, other: &PyAny) -> PyResult<PyObject> {
        if other.is_instance_of::<PyDuration>() {
            let dur = other.extract::<PyDuration>()?;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(PyDuration {
                    inner: self.inner.clone() + dur.inner.clone(),
                }
                .into_py(py))
            })
        } else if other.is_instance_of::<PyAstroTime>() {
            let tm = other.extract::<PyAstroTime>()?;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(PyAstroTime {
                    inner: tm.inner + self.inner.clone(),
                }
                .into_py(py))
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid right-hand side",
            ))
        }
    }

    fn __sub__(&self, other: &PyDuration) -> PyDuration {
        PyDuration {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn __mul__(&self, other: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Seconds(self.inner.seconds() * other),
        }
    }

    fn days(&self) -> f64 {
        self.inner.days()
    }

    fn seconds(&self) -> f64 {
        self.inner.seconds()
    }

    fn minutes(&self) -> f64 {
        self.inner.minutes()
    }

    fn hours(&self) -> f64 {
        self.inner.hours()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
