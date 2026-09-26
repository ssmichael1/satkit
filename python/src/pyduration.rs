use satkit::Duration;

use crate::pyinstant::PyInstant;
use crate::pyutils::invalid_value;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

const US_PER_SECOND: f64 = 1.0e6;
const US_PER_MINUTE: f64 = 60.0e6;
const US_PER_HOUR: f64 = 3_600.0e6;
const US_PER_DAY: f64 = 86_400.0e6;

crate::arg_extractor!(days_arg: f64, |_| invalid_value("days"));
crate::arg_extractor!(seconds_arg: f64, |_| invalid_value("seconds"));
crate::arg_extractor!(minutes_arg: f64, |_| invalid_value("minutes"));
crate::arg_extractor!(hours_arg: f64, |_| invalid_value("hours"));
crate::arg_extractor!(microseconds_arg: i64, |_| invalid_value("microseconds"));

/// Refuse a NaN or infinite count of `what` with `ValueError`, and one too
/// large for a duration (beyond ±2^63 microseconds, about ±292,000 years)
/// with `OverflowError`; `us_per_unit` converts the count to microseconds.
///
/// The core `Duration` constructors map NaN to zero and saturate the rest,
/// so without this `duration(days=nan)` was a zero duration and
/// `time + inf` a garbage date.
pub(crate) fn check_duration_value(value: f64, us_per_unit: f64, what: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be finite, got {value}"
        )));
    }
    // 2^63 is exact in f64; i64::MAX as f64 rounds up to it
    if (value * us_per_unit).abs() >= 9_223_372_036_854_775_808.0 {
        return Err(pyo3::exceptions::PyOverflowError::new_err(format!(
            "{what} out of range for a duration: {value}"
        )));
    }
    Ok(())
}

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
#[pyclass(name = "duration", module = "satkit", from_py_object)]
#[derive(Clone)]
pub struct PyDuration(pub Duration);

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
    ///     microseconds (int): Duration in microseconds
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
    #[pyo3(signature=(*, days=0.0, hours=0.0, minutes=0.0, seconds=0.0, microseconds=0))]
    fn py_new(
        #[pyo3(from_py_with = days_arg)] days: f64,
        #[pyo3(from_py_with = hours_arg)] hours: f64,
        #[pyo3(from_py_with = minutes_arg)] minutes: f64,
        #[pyo3(from_py_with = seconds_arg)] seconds: f64,
        #[pyo3(from_py_with = microseconds_arg)] microseconds: i64,
    ) -> PyResult<Self> {
        check_duration_value(days, US_PER_DAY, "days")?;
        check_duration_value(hours, US_PER_HOUR, "hours")?;
        check_duration_value(minutes, US_PER_MINUTE, "minutes")?;
        check_duration_value(seconds, US_PER_SECOND, "seconds")?;
        Ok(Self(
            Duration::from_seconds(seconds)
                + Duration::from_days(days)
                + Duration::from_minutes(minutes)
                + Duration::from_hours(hours)
                + Duration::from_microseconds(microseconds),
        ))
    }

    /// Create new duration object from the number of days
    ///
    /// Args:
    ///     d (float): The number of days
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_days(d: f64) -> PyResult<Self> {
        check_duration_value(d, US_PER_DAY, "days")?;
        Ok(Self(Duration::from_days(d)))
    }

    /// Create new duration object from the number of seconds
    ///
    /// Args:
    ///     seconds (float): The number of seconds
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_seconds(seconds: f64) -> PyResult<Self> {
        check_duration_value(seconds, US_PER_SECOND, "seconds")?;
        Ok(Self(Duration::from_seconds(seconds)))
    }

    /// Create new duration object from the number of minutes
    ///
    /// Args:
    ///     minutes (float): The number of minutes
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_minutes(minutes: f64) -> PyResult<Self> {
        check_duration_value(minutes, US_PER_MINUTE, "minutes")?;
        Ok(Self(Duration::from_minutes(minutes)))
    }

    /// Create new duration object from number of hours
    ///
    /// Args:
    ///     hours (float): The number of hours
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_hours(hours: f64) -> PyResult<Self> {
        check_duration_value(hours, US_PER_HOUR, "hours")?;
        Ok(Self(Duration::from_hours(hours)))
    }

    /// Create new duration object from the number of milliseconds
    ///
    /// Args:
    ///     d (float): The number of milliseconds
    ///
    /// Returns:
    ///     duration: New duration object
    #[staticmethod]
    fn from_milliseconds(d: f64) -> PyResult<Self> {
        check_duration_value(d, 1_000.0, "milliseconds")?;
        Ok(Self(Duration::from_milliseconds(d)))
    }

    /// Add durations or add duration to satkit.time
    ///
    /// Args:
    ///     other (duration|satkit.time): Duration or time object to add
    ///
    /// Returns:
    ///     duration|satkit.time: New duration or time object
    fn __add__(&self, other: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
        if other.is_instance_of::<Self>() {
            let dur = other
                .extract::<Self>()
                .map_err(|e| anyhow::anyhow!("Invalid duration: {}", e))?;
            Ok(Self(self.0 + dur.0).into_py_any(other.py())?)
        } else if other.is_instance_of::<PyInstant>() {
            let tm = other
                .extract::<PyInstant>()
                .map_err(|e| anyhow::anyhow!("Invalid time object: {}", e))?;
            Ok(PyInstant(tm.0 + self.0).into_py_any(other.py())?)
        } else {
            // Not a supported operand: let Python raise its standard
            // `TypeError: unsupported operand type(s)`. A bare number is
            // deliberately not accepted (its unit would be ambiguous).
            Ok(other.py().NotImplemented())
        }
    }

    /// Subtract durations
    ///
    /// Args:
    ///     other (duration): Duration to subtract
    ///
    /// Returns:
    ///     duration: New duration object representing the difference
    fn __sub__(&self, other: &Self) -> Self {
        Self(self.0 - other.0)
    }

    /// Multiply duration by a scalar (scale duration)
    ///
    /// Args:
    ///     other (float): Scalar to multiply duration by
    ///
    /// Returns:
    ///     duration: New duration object representing the scaled duration
    fn __mul__(&self, other: f64) -> PyResult<Self> {
        let secs = self.0.as_seconds() * other;
        check_duration_value(secs, US_PER_SECOND, "scaled duration (seconds)")?;
        Ok(Self(Duration::from_seconds(secs)))
    }

    /// Divide a duration by a real number (scale it) or by another duration
    ///
    /// Args:
    ///     other (float|int|duration): Scalar divisor, or a duration
    ///
    /// Returns:
    ///     duration|float: The scaled duration, or the dimensionless ratio
    ///     for a duration divisor
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<Self>() {
            let dur = other
                .extract::<Self>()
                .map_err(|e| anyhow::anyhow!("Invalid duration: {}", e))?;
            if dur.0.as_microseconds() == 0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by a zero duration",
                ));
            }
            (self.0.as_seconds() / dur.0.as_seconds()).into_py_any(other.py())
        } else if let Ok(scalar) = other.extract::<f64>() {
            // Any real number: Python float or int, or a numpy scalar
            if scalar == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "duration division by zero",
                ));
            }
            let secs = self.0.as_seconds() / scalar;
            check_duration_value(secs, US_PER_SECOND, "scaled duration (seconds)")?;
            PyDuration(Duration::from_seconds(secs)).into_py_any(other.py())
        } else {
            Ok(other.py().NotImplemented())
        }
    }

    // Backed by an exact microsecond count, so hashing the raw integer is
    // consistent with __eq__ (defining __eq__ alone would make duration unhashable).
    fn __hash__(&self) -> isize {
        self.0.usec as isize
    }

    // Comparison methods for duration objects
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.0 < other.0
    }

    fn __le__(&self, other: &Self) -> bool {
        self.0 <= other.0
    }

    fn __gt__(&self, other: &Self) -> bool {
        self.0 > other.0
    }

    fn __ge__(&self, other: &Self) -> bool {
        self.0 >= other.0
    }

    /// Duration in units of days, where 1 day = 86,400 seconds
    ///
    /// Returns:
    ///     float: Duration in days
    #[getter]
    fn days(&self) -> f64 {
        self.0.as_days()
    }

    /// Duration in units of seconds
    ///
    /// Returns:
    ///     float: Duration in seconds
    #[getter]
    fn seconds(&self) -> f64 {
        self.0.as_seconds()
    }

    /// Duration in units of minutes
    ///
    /// Returns:
    ///     float: Duration in minutes
    #[getter]
    fn minutes(&self) -> f64 {
        self.0.as_minutes()
    }

    /// Duration in units of hours
    ///
    /// Returns:
    ///     float: Duration in hours
    #[getter]
    fn hours(&self) -> f64 {
        self.0.as_hours()
    }

    /// Duration in units of whole microseconds
    ///
    /// Returns:
    ///     int: Duration in microseconds
    #[getter]
    fn microseconds(&self) -> i64 {
        self.0.as_microseconds()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }

    fn __setstate__(&mut self, py: Python, s: Py<PyBytes>) -> Result<()> {
        let s = s.as_bytes(py);
        if s.len() != 8 {
            bail!("Invalid serialization length");
        }
        let t = i64::from_le_bytes(s.try_into()?);
        self.0 = Duration { usec: t };
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        PyBytes::new(py, i64::to_le_bytes(self.0.usec).as_slice()).into_py_any(py)
    }
}
