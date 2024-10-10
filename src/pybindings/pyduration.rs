use crate::Duration;

use super::pyastrotime::PyAstroTime;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;

/// Class representing durations of times, allowing for representation
/// via common measures of duration (years, days, hours, minutes, seconds)
///
/// This enum can be added to and subtracted from "satkit.time" objects to
/// represent new "satkit" objects, and is also returned when
/// two "satkit" objects are subtracted from one anothre
///
/// Keyword Arguments:
///     days (float): Duration in days
///     seconds (float): Duration in seconds
///     minutes (float): Duration in minutes
///     hours (float): Duration in hours
///
/// Example:
///
/// >>> from satkit import duration
/// >>> d = duration(seconds=3.0)
/// >>> d2 = duration(minutes=4.0)
/// >>> print(d + d2)
/// Duration: 4 minutes, 3.000 seconds
///
/// >>> from satkit import duration, time
/// >>> instant = satkit.time(2023, 3, 5)
/// >>> plus1day = instant + duration(days=1.0)
///
#[pyclass(name = "duration", module = "satkit")]
#[derive(Clone)]
pub struct PyDuration {
    pub inner: Duration,
}

#[pymethods]
impl PyDuration {
    /// Create a new Duration object.
    ///
    /// The duration can be created by passing the number of days, seconds, minutes, and hours.
    /// as keyword arguments
    ///
    /// they will be summed up to create the duration
    ///
    /// If no arguments are passed, the duration will be 0
    ///
    /// Keyword Arguments:
    ///     days (float): Duration in days
    ///     seconds (float): Duration in seconds
    ///     minutes (float): Duration in minutes
    ///     hours (float): Duration in hours
    ///
    /// Example:
    ///
    ///
    /// >>> from satkit import duration
    /// >>>
    /// >>> # Create a duration of 1 day
    /// >>> dur = duration(days=1)
    ///
    ///
    #[new]
    #[pyo3(signature=(**kwargs))]
    fn py_new(kwargs: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
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
            inner: Duration::Seconds(seconds)
                + Duration::Days(days)
                + Duration::Minutes(minutes)
                + Duration::Hours(hours),
        })
    }

    /// Create new duration object from the number of days
    ///
    /// Args:
    ///     d (float): The number of days
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_days(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Days(d),
        }
    }

    /// Create new duration object from the number of seconds
    ///
    /// Args:
    ///     d (float): The number of seconds
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_seconds(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Seconds(d),
        }
    }

    /// Create new duration object from the number of minutes
    ///
    /// Args:
    ///     d (float): The number of minutes
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_minutes(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Minutes(d),
        }
    }

    /// Create new duration object from number of hours
    ///
    /// Args:
    ///     d (float): The number of hours
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_hours(d: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Hours(d),
        }
    }

    /// Add durations or add duration to satkit.time
    ///
    /// Args:
    ///     other (duration|satkit.time): Duration or time object to add
    ///
    /// Returns:
    ///     duration|satkit.time: New duration or time object
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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

    /// Subtract durations
    ///
    /// Args:
    ///     other (duration): Duration to subtract
    ///
    /// Returns:
    ///     duration: New duration object representing the difference
    fn __sub__(&self, other: &PyDuration) -> PyDuration {
        PyDuration {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    /// Multiply duration by a scalar (scale duration)
    ///
    /// Args:
    ///     other (float): Scalar to multiply duration by
    ///
    /// Returns:
    ///     duration: New duration object representing the scaled duration
    fn __mul__(&self, other: f64) -> PyDuration {
        PyDuration {
            inner: Duration::Seconds(self.inner.seconds() * other),
        }
    }

    /// Duration in units of days, where 1 day = 86,400 seconds
    ///
    /// Returns:
    ///     float: Duration in days
    fn days(&self) -> f64 {
        self.inner.days()
    }

    /// Duration in units of seconds
    ///
    /// Returns:
    ///     float: Duration in seconds
    fn seconds(&self) -> f64 {
        self.inner.seconds()
    }

    /// Duration in units of minutes
    ///
    /// Returns:
    ///     float: Duration in minutes
    fn minutes(&self) -> f64 {
        self.inner.minutes()
    }

    /// Duration in units of hours
    ///
    /// Returns:
    ///     float: Duration in hours
    fn hours(&self) -> f64 {
        self.inner.hours()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn __setstate__(&mut self, py: Python, s: Py<PyBytes>) -> PyResult<()> {
        let s = s.as_bytes(py);
        if s.len() != 8 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let t = f64::from_le_bytes(s.try_into()?);
        self.inner = Duration::Days(t);
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new_bound(py, f64::to_le_bytes(self.inner.days()).as_slice()).to_object(py))
    }
}
