use satkit::Duration;

use crate::pyinstant::PyInstant;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::{PyDelta, PyDeltaAccess};
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

const US_PER_SECOND: f64 = 1.0e6;
const US_PER_MINUTE: f64 = 60.0e6;
const US_PER_HOUR: f64 = 3_600.0e6;
const US_PER_DAY: f64 = 86_400.0e6;

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

/// The `OverflowError` for time or duration arithmetic whose result is
/// beyond the ±2^63-microsecond range (about ±292,000 years). The core
/// operators saturate there; the bindings use the checked forms and raise.
pub(crate) fn arithmetic_overflow(what: &str) -> PyErr {
    pyo3::exceptions::PyOverflowError::new_err(format!(
        "{what} out of range (beyond about ±292,000 years)"
    ))
}

/// Class representing durations of time, allowing for representation
/// via common measures of duration (days, hours, minutes, seconds,
/// microseconds)
///
/// A duration is an integer number of microseconds. It can be added to and
/// subtracted from "satkit.time" objects to give new "satkit.time" objects,
/// and is also returned when two "satkit.time" objects are subtracted from
/// one another
///
/// Keyword Arguments:
///     days (float): Duration in days (of 86400 seconds)
///     hours (float): Duration in hours
///     minutes (float): Duration in minutes
///     seconds (float): Duration in seconds
///     microseconds (int): Duration in microseconds
///
/// Example:
///
/// >>> from satkit import duration
/// >>> d = duration(seconds=3.0)
/// >>> d2 = duration(minutes=4.0)
/// >>> print(d + d2)
/// 4 minutes 3.000000 seconds
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
    /// Raises:
    ///     TypeError: If an argument is not a real number, or ``microseconds``
    ///         is not an integer
    ///     ValueError: If an argument is NaN or infinite
    ///     OverflowError: If an argument, or the sum, is too large for a
    ///         duration (beyond about ±292,000 years)
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
        days: f64,
        hours: f64,
        minutes: f64,
        seconds: f64,
        microseconds: i64,
    ) -> PyResult<Self> {
        // Plain typed parameters: an argument of the wrong type is PyO3's
        // TypeError naming the argument, as for the from_* constructors
        // (through 0.24 it was `ValueError: Invalid value for days`)
        check_duration_value(days, US_PER_DAY, "days")?;
        check_duration_value(hours, US_PER_HOUR, "hours")?;
        check_duration_value(minutes, US_PER_MINUTE, "minutes")?;
        check_duration_value(seconds, US_PER_SECOND, "seconds")?;
        // Each component fits; their sum may not (it used to wrap:
        // duration(days=1e8, seconds=1e12) came out negative)
        [
            Duration::from_days(days),
            Duration::from_minutes(minutes),
            Duration::from_hours(hours),
            Duration::from_microseconds(microseconds),
        ]
        .into_iter()
        .try_fold(Duration::from_seconds(seconds), |acc, d| acc.checked_add(d))
        .map(Self)
        .ok_or_else(|| arithmetic_overflow("duration"))
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

    /// Create new duration object from a whole number of microseconds (exact)
    ///
    /// Args:
    ///     microseconds (int): The number of microseconds
    ///
    /// Returns:
    ///     duration: New duration object
    ///
    /// Raises:
    ///     TypeError: If ``microseconds`` is not an integer
    ///     OverflowError: If ``microseconds`` does not fit a signed 64-bit
    ///         integer (about ±292,000 years)
    #[staticmethod]
    fn from_microseconds(microseconds: i64) -> Self {
        Self(Duration::from_microseconds(microseconds))
    }

    /// Create new duration object from a datetime.timedelta (exact: both
    /// count whole microseconds)
    ///
    /// Args:
    ///     td (datetime.timedelta): The time difference
    ///
    /// Returns:
    ///     duration: New duration object
    ///
    /// Raises:
    ///     OverflowError: If ``td`` is beyond the range of a duration (about
    ///         ±292,000 years; a timedelta reaches ±2.7 million)
    #[staticmethod]
    fn from_timedelta(td: &Bound<'_, PyDelta>) -> PyResult<Self> {
        timedelta_to_duration(td).map(Self)
    }

    /// Convert to a datetime.timedelta (exact: both count whole microseconds)
    ///
    /// Returns:
    ///     datetime.timedelta: The same duration as a timedelta
    fn to_timedelta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDelta>> {
        // Normalised as timedelta does: days of any sign, then seconds and
        // microseconds that are not negative. An i64 of microseconds is at
        // most 1.07e8 days, well inside timedelta's 1e9.
        const US_PER_DAY: i64 = 86_400_000_000;
        let rem = self.0.usec.rem_euclid(US_PER_DAY);
        PyDelta::new(
            py,
            self.0.usec.div_euclid(US_PER_DAY) as i32,
            (rem / 1_000_000) as i32,
            (rem % 1_000_000) as i32,
            false,
        )
    }

    /// Add durations or add duration to satkit.time
    ///
    /// Args:
    ///     other (duration|satkit.time): Duration or time object to add
    ///
    /// Returns:
    ///     duration|satkit.time: New duration or time object
    ///
    /// Raises:
    ///     OverflowError: if the result is out of range (beyond about ±292,000 years)
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<Self>() {
            let dur = other.extract::<Self>()?;
            let sum = self.0.checked_add(dur.0);
            Self(sum.ok_or_else(|| arithmetic_overflow("duration"))?).into_py_any(other.py())
        } else if other.is_instance_of::<PyInstant>() {
            let tm = other.extract::<PyInstant>()?;
            let t = tm.0.checked_add(self.0);
            PyInstant(t.ok_or_else(|| arithmetic_overflow("time"))?).into_py_any(other.py())
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
    ///
    /// Raises:
    ///     OverflowError: if the result is out of range (beyond about ±292,000 years)
    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.0
            .checked_sub(other.0)
            .map(Self)
            .ok_or_else(|| arithmetic_overflow("duration"))
    }

    /// Multiply duration by a scalar (scale duration)
    ///
    /// Args:
    ///     other (float): Scalar to multiply duration by
    ///
    /// Returns:
    ///     duration: New duration object representing the scaled duration
    ///
    /// Raises:
    ///     ValueError: if ``other`` is NaN or infinite
    ///     OverflowError: if the result is out of range (beyond about ±292,000 years)
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let usec = match Factor::extract(other)? {
            // Not a real number: Python raises its own TypeError
            None => return Ok(py.NotImplemented()),
            Some(Factor::Int(k)) => i64::try_from(self.0.usec as i128 * k as i128).ok(),
            Some(Factor::Float(f)) => {
                if !f.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "duration factor must be finite, got {f}"
                    )));
                }
                mul_exact(self.0.usec, f)
            }
        };
        let usec = usec.ok_or_else(|| arithmetic_overflow("scaled duration"))?;
        Self(Duration::from_microseconds(usec)).into_py_any(py)
    }

    /// Multiply a scalar by a duration: ``2 * d`` is ``d * 2``
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(other)
    }

    /// The duration with its sign reversed
    ///
    /// Raises:
    ///     OverflowError: for the most negative duration (-2^63 microseconds)
    fn __neg__(&self) -> PyResult<Self> {
        self.0
            .usec
            .checked_neg()
            .map(|us| Self(Duration::from_microseconds(us)))
            .ok_or_else(|| arithmetic_overflow("duration"))
    }

    /// The duration itself
    fn __pos__(&self) -> Self {
        self.clone()
    }

    /// The duration without its sign
    ///
    /// Raises:
    ///     OverflowError: for the most negative duration (-2^63 microseconds)
    fn __abs__(&self) -> PyResult<Self> {
        self.0
            .usec
            .checked_abs()
            .map(|us| Self(Duration::from_microseconds(us)))
            .ok_or_else(|| arithmetic_overflow("duration"))
    }

    /// Divide a duration by a real number (scale it) or by another duration
    ///
    /// Args:
    ///     other (float|int|duration): Scalar divisor, or a duration
    ///
    /// Returns:
    ///     duration|float: The scaled duration, or the dimensionless ratio
    ///     for a duration divisor
    ///
    /// Raises:
    ///     ZeroDivisionError: if ``other`` is zero or a zero duration
    ///     ValueError: if ``other`` is NaN
    ///     OverflowError: if the result is out of range (beyond about ±292,000 years)
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
            // The microsecond counts directly: one rounding, not three
            (self.0.usec as f64 / dur.0.usec as f64).into_py_any(other.py())
        } else {
            // Any real number: Python float or int, or a numpy scalar
            let zero =
                || pyo3::exceptions::PyZeroDivisionError::new_err("duration division by zero");
            let usec = match Factor::extract(other)? {
                None => return Ok(other.py().NotImplemented()),
                Some(Factor::Int(0)) => return Err(zero()),
                Some(Factor::Int(k)) => {
                    let (n, d) = (self.0.usec as i128, k as i128);
                    i64::try_from(div_round(n * d.signum(), d.abs())).ok()
                }
                Some(Factor::Float(f)) => {
                    if f == 0.0 {
                        return Err(zero());
                    }
                    if f.is_nan() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "duration divisor must not be NaN",
                        ));
                    }
                    div_exact(self.0.usec, f)
                }
            };
            let usec = usec.ok_or_else(|| arithmetic_overflow("scaled duration"))?;
            PyDuration(Duration::from_microseconds(usec)).into_py_any(other.py())
        }
    }

    /// Floor division: by a duration, the whole number of times it fits
    /// (an int); by a real number, the duration scaled and rounded down to
    /// a whole microsecond
    ///
    /// Exact integer-microsecond arithmetic, floored as Python's ``//``
    /// (towards negative infinity), so ``a == (a // b) * b + a % b``.
    ///
    /// Args:
    ///     other (duration|int|float): Duration or real-number divisor
    ///
    /// Returns:
    ///     int|duration: ``floor(self / other)`` as an int for a duration
    ///     divisor, else the duration ``floor(self / other)`` microseconds
    ///
    /// Raises:
    ///     ZeroDivisionError: if ``other`` is zero or a zero duration
    ///     ValueError: if ``other`` is NaN or infinite
    ///     OverflowError: if the result is out of range (beyond about ±292,000 years)
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let zero = || pyo3::exceptions::PyZeroDivisionError::new_err("duration division by zero");
        if let Ok(d) = other.cast::<Self>() {
            let b = d.borrow().0.usec as i128;
            if b == 0 {
                return Err(zero());
            }
            // i128: i64::MIN // -1 does not fit an i64, but is a fine int
            return floor_div(self.0.usec as i128, b).into_py_any(py);
        }
        let usec = match Factor::extract(other)? {
            None => return Ok(py.NotImplemented()),
            Some(Factor::Int(0)) => return Err(zero()),
            Some(Factor::Int(k)) => i64::try_from(floor_div(self.0.usec as i128, k as i128)).ok(),
            Some(Factor::Float(f)) => {
                if f == 0.0 {
                    return Err(zero());
                }
                if !f.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "duration divisor must be finite, got {f}"
                    )));
                }
                div_floor_exact(self.0.usec, f)
            }
        };
        let usec = usec.ok_or_else(|| arithmetic_overflow("scaled duration"))?;
        Self(Duration::from_microseconds(usec)).into_py_any(py)
    }

    /// Remainder of floor division by a duration: ``self - (self // other) * other``
    ///
    /// Exact; like Python's ``%``, the result has the sign of ``other``
    /// (or is zero).
    ///
    /// Args:
    ///     other (duration): Duration divisor
    ///
    /// Returns:
    ///     duration: The remainder
    ///
    /// Raises:
    ///     ZeroDivisionError: if ``other`` is a zero duration
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Ok(d) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let (a, b) = (self.0.usec as i128, d.borrow().0.usec as i128);
        if b == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "duration modulo by a zero duration",
            ));
        }
        // |remainder| < |b|, so it fits an i64
        let r = a - floor_div(a, b) * b;
        Self(Duration::from_microseconds(r as i64)).into_py_any(py)
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

    /// Duration in units of milliseconds (the whole duration, like
    /// ``seconds``; the float nearest the exact value)
    ///
    /// Returns:
    ///     float: Duration in milliseconds
    #[getter]
    fn milliseconds(&self) -> f64 {
        // Divided, not multiplied by 1e-3, so it is correctly rounded
        self.0.as_microseconds() as f64 / 1.0e3
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

/// A `datetime.timedelta` as a duration, exactly; `OverflowError` if it is
/// beyond the ±2^63-microsecond range (a timedelta reaches ±1e9 days)
pub(crate) fn timedelta_to_duration(td: &Bound<'_, PyDelta>) -> PyResult<Duration> {
    let us = (td.get_days() as i128 * 86_400 + td.get_seconds() as i128) * 1_000_000
        + td.get_microseconds() as i128;
    i64::try_from(us)
        .map(Duration::from_microseconds)
        .map_err(|_| arithmetic_overflow("datetime.timedelta"))
}

/// A real-number operand of `duration * x` / `duration / x`
enum Factor {
    /// Anything with `__index__` that fits an i64 (Python `int`, `bool`,
    /// numpy integer scalars): the arithmetic is exact
    Int(i64),
    /// Anything else with `__float__`
    Float(f64),
}

impl Factor {
    /// `None` if `obj` is not a real number (the operator then returns
    /// `NotImplemented`); errors other than a wrong type are passed on
    /// (`OverflowError` for a Python int too large for a float)
    fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if !obj.is_instance_of::<pyo3::types::PyFloat>() {
            if let Ok(k) = obj.extract::<i64>() {
                return Ok(Some(Self::Int(k)));
            }
        }
        match obj.extract::<f64>() {
            Ok(f) => Ok(Some(Self::Float(f))),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyTypeError>(obj.py()) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// `n / d` rounded to the nearest integer, halves away from zero (as
/// `f64::round`, which every float → microsecond conversion uses). `d > 0`.
fn div_round(n: i128, d: i128) -> i128 {
    let (q, r) = (n.div_euclid(d), n.rem_euclid(d));
    if 2 * r > d || (2 * r == d && n >= 0) {
        q + 1
    } else {
        q
    }
}

/// `floor(n / d)`, as Python's `//`. `d != 0`.
fn floor_div(n: i128, d: i128) -> i128 {
    let q = n / d;
    if n % d != 0 && ((n < 0) != (d < 0)) {
        q - 1
    } else {
        q
    }
}

/// `floor(usec / f)` from the exact quotient; `None` if it does not fit an
/// i64. `f` is finite and not zero.
fn div_floor_exact(usec: i64, f: f64) -> Option<i64> {
    // usec / (m 2^e), with the sign moved to the numerator
    let (m, e) = decompose(f);
    let n = usec as i128 * m.signum();
    let m = m.abs();
    let r = if e <= 0 {
        // n 2^-e / m. Beyond 2^120 the quotient exceeds 2^63 (m < 2^53),
        // as it does when the shifted numerator overflows an i128
        if e < -120 {
            return if n == 0 { Some(0) } else { None };
        }
        n.checked_mul(1i128 << -e)?.div_euclid(m)
    } else if e > 60 {
        // The divisor is beyond 2^112: |quotient| < 1, so the floor is 0
        // or -1
        if n < 0 {
            -1
        } else {
            0
        }
    } else {
        n.div_euclid(m << e)
    };
    i64::try_from(r).ok()
}

/// A finite `f` as `m * 2^e` with integer `|m| < 2^53`, exactly
fn decompose(f: f64) -> (i128, i32) {
    let bits = f.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & 0x000f_ffff_ffff_ffff) as i128;
    let (m, e) = if exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1 << 52), exp - 1075)
    };
    (if bits >> 63 == 0 { m } else { -m }, e)
}

/// `usec * f` rounded once to the nearest microsecond, from the exact
/// product; `None` if it does not fit an i64. `f` is finite.
///
/// Going through float seconds (`usec / 1e6 * f`, then back) rounded three
/// times and lost a microsecond above 2^53 µs (285 years), even for `f = 1`.
fn mul_exact(usec: i64, f: f64) -> Option<i64> {
    let (m, e) = decompose(f);
    // |usec| <= 2^63 and |m| < 2^53, so the product fits an i128
    let p = usec as i128 * m;
    let r = if p == 0 {
        0
    } else if e >= 0 {
        // p != 0, so anything beyond 2^64 is out of range anyway
        if e > 64 {
            return None;
        }
        p.checked_mul(1i128 << e)?
    } else if e <= -126 {
        // |p| < 2^116: less than half a microsecond
        0
    } else {
        div_round(p, 1i128 << -e)
    };
    i64::try_from(r).ok()
}

/// `usec / f` rounded once to the nearest microsecond, from the exact
/// quotient; `None` if it does not fit an i64. `f` is not zero or NaN
/// (an infinite `f` gives 0).
fn div_exact(usec: i64, f: f64) -> Option<i64> {
    if usec == 0 || f.is_infinite() {
        return Some(0);
    }
    // usec / (m 2^e), with the sign moved to the numerator
    let (m, e) = decompose(f);
    let n = usec as i128 * m.signum();
    let m = m.abs();
    let r = if e <= 0 {
        // n 2^-e / m. Beyond 2^120 the quotient exceeds 2^63 (m < 2^53),
        // as it does when the shifted numerator overflows an i128
        if e < -120 {
            return None;
        }
        div_round(n.checked_mul(1i128 << -e)?, m)
    } else if e > 60 {
        // e > 0 means a normal number, m >= 2^52: the divisor is beyond
        // 2^112 and the quotient less than half a microsecond
        0
    } else {
        div_round(n, m << e)
    };
    i64::try_from(r).ok()
}
