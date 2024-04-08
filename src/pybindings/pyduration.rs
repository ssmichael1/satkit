use crate::Duration;

use super::pyastrotime::PyAstroTime;
use pyo3::prelude::*;
use pyo3::types::PyDict;


/// Class representing durations of times, allowing for representation
/// via common measures of duration (years, days, hours, minutes, seconds)
///
/// This enum can be added to and subtracted from "satkit.time" objects to
/// represent new "satkit" objects, and is also returned when
/// two "satkit" objects are subtracted from one anothre
/// 
/// # Example
/// 
/// ```python
/// from satkit import duration
/// d = duration(seconds=3.0)
/// d2 = duration(minutes=4.0)
/// print(d + d2)
/// # Output: Duration: 4 minutes, 3.000 seconds
/// ```
/// 
#[pyclass(name = "duration")]
#[derive(Clone)]
pub struct PyDuration {
    pub inner: Duration,
}

#[pymethods]
impl PyDuration {

    ///
    /// Create a new Duration object.
    /// 
    /// The duration can be created by passing the number of days, seconds, minutes, and hours.
    /// as keyword arguments
    /// 
    /// they will be summed up to create the duration
    /// 
    /// If no arguments are passed, the duration will be 0
    /// 
    /// # Keyword Arguments
    /// 
    /// * `days` - The number of days
    /// * `seconds` - The number of seconds
    /// * `minutes` - The number of minutes
    /// 
    /// 
    /// # Example
    /// 
    /// ```python
    /// from satkit import duration
    /// 
    /// # Create a duration of 1 day
    /// dur = duration(days=1)
    /// 
    /// ```
    /// 
    #[new]
    #[pyo3(signature=(**kwargs))]
    fn py_new(kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut days = 0.0;
        let mut seconds = 0.0;
        let mut minutes = 0.0;
        let mut hours = 0.0;

        if let Some(kwargs) = kwargs {
            if let Some(d) = kwargs.get_item("days")? {
                days = d.extract::<f64>()?;
            }
            if let Some(s) = kwargs.get_item("seconds")? {
                seconds = s.extract::<f64>()?;
            }
            if let Some(m) = kwargs.get_item("minutes")? {
                minutes = m.extract::<f64>()?;
            }
            if let Some(h) = kwargs.get_item("hours")? {
                hours = h.extract::<f64>()?;
            }
        }

        Ok(PyDuration {
            inner: Duration::Seconds(seconds) + Duration::Days(days) + Duration::Minutes(minutes) + Duration::Hours(hours),
        })
    }


    ///
    /// Create a new Duration object from the number of days
    /// 
    /// # Arguments
    /// 
    /// * `d` - The number of days
    /// 
    /// # Example
    /// 
    /// ```python
    /// from satkit import duration
    /// dur = duration.from_days(1)
    /// ```
    /// 
    #[staticmethod]
    fn from_days(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Days(d),
        }
    }

    ///
    /// Create a new Duration object from the number of seconds
    /// 
    /// # Arguments
    /// 
    /// * `s` - The number of seconds
    ///
    /// # Example
    /// 
    /// ```python
    /// from satkit import duration
    /// dur = duration.from_seconds(1)
    /// ````
    /// 
    #[staticmethod]
    fn from_seconds(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Seconds(d),
        }
    }

    ///
    /// Create a new Duration object from the number of minutes
    /// 
    /// # Arguments
    /// 
    /// * `m` - The number of minutes
    /// 
    /// # Example
    /// 
    /// ```python
    /// from satkit import duration
    /// dur = duration.from_minutes(1)
    /// ```
    /// 
    #[staticmethod]
    fn from_minutes(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Minutes(d),
        }
    }

    ///
    /// Create a new Duration object from the number of hours
    /// 
    /// # Arguments
    /// 
    /// * `h` - The number of hours
    /// 
    /// # Example
    /// 
    /// ```python
    /// from satkit import duration
    /// dur = duration.from_hours(1)
    /// ```
    /// 
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

    /// Return duration in units of days, where 1 day = 86,400 seconds
    fn days(&self) -> f64 {
        self.inner.days()
    }

    /// Return duration in units of seconds
    fn seconds(&self) -> f64 {
        self.inner.seconds()
    }

    /// Return duration in units of minutes
    fn minutes(&self) -> f64 {
        self.inner.minutes()
    }

    /// Return duration in units of hours
    fn hours(&self) -> f64 {
        self.inner.hours()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}
