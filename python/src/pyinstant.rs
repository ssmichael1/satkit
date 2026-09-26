use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDateTime;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::types::PyTzInfo;
use pyo3::types::{PyDelta, PyDeltaAccess};
use pyo3::IntoPyObjectExt;

use satkit::{Instant, TimeScale, Weekday};

use crate::pyduration::PyDuration;
use crate::pyutils::warn_deprecated;

use anyhow::{bail, Result};

use numpy as np;

/// Specify time scale used to represent or convert between the "satkit.time"
/// representation of time
///
/// Most of the time, these are not needed directly, but various time scales
/// are needed to compute precise rotations between various inertial and
/// Earth-fixed coordinate frames
///
/// For an excellent overview, see:
/// https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter2-TimeScales.pdf
///
/// * UTC = Universal Time Coordinate
/// * TT = Terrestrial Time
/// * UT1 = Universal time, corrected for polar wandering
/// * TAI = International Atomic Time
/// * GPS = Global Positioning System Time (epoch = 1/6/1980 00:00:00)
/// * TDB = Barycentric Dynamical Time
///
#[derive(Clone, PartialEq, Eq)]
#[pyclass(name = "timescale", module = "satkit", eq, eq_int, from_py_object)]
pub enum PyTimeScale {
    /// Invalid time scale
    Invalid = TimeScale::Invalid as isize,
    /// Universal Time Coordinate
    #[allow(clippy::upper_case_acronyms)]
    UTC = TimeScale::UTC as isize,
    /// Terrestrial Time
    #[allow(clippy::upper_case_acronyms)]
    TT = TimeScale::TT as isize,
    /// UT1
    #[allow(clippy::upper_case_acronyms)]
    UT1 = TimeScale::UT1 as isize,
    /// International Atomic Time
    #[allow(clippy::upper_case_acronyms)]
    TAI = TimeScale::TAI as isize,
    /// Global Positioning System (GPS) Time
    #[allow(clippy::upper_case_acronyms)]
    GPS = TimeScale::GPS as isize,
    /// Barycentric Dynamical Time
    #[allow(clippy::upper_case_acronyms)]
    TDB = TimeScale::TDB as isize,
}

crate::enum_pickle!(PyTimeScale, "timescale");

#[derive(Clone, PartialEq, Eq)]
/// Represent the day of the week
///
/// Values:
/// - `Sunday`
/// - `Monday`
/// - `Tuesday`
/// - `Wednesday`
/// - `Thursday`
/// - `Friday`
/// - `Saturday`
#[pyclass(name = "weekday", module = "satkit", eq, eq_int, from_py_object)]
pub enum PyWeekday {
    Sunday = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6,
    Invalid = -1,
}

crate::enum_pickle!(PyWeekday, "weekday");

impl From<&PyWeekday> for Weekday {
    fn from(w: &PyWeekday) -> Self {
        match w {
            PyWeekday::Sunday => Self::Sunday,
            PyWeekday::Monday => Self::Monday,
            PyWeekday::Tuesday => Self::Tuesday,
            PyWeekday::Wednesday => Self::Wednesday,
            PyWeekday::Thursday => Self::Thursday,
            PyWeekday::Friday => Self::Friday,
            PyWeekday::Saturday => Self::Saturday,
            PyWeekday::Invalid => Self::Invalid,
        }
    }
}

impl From<Weekday> for PyWeekday {
    fn from(w: Weekday) -> Self {
        match w {
            Weekday::Sunday => Self::Sunday,
            Weekday::Monday => Self::Monday,
            Weekday::Tuesday => Self::Tuesday,
            Weekday::Wednesday => Self::Wednesday,
            Weekday::Thursday => Self::Thursday,
            Weekday::Friday => Self::Friday,
            Weekday::Saturday => Self::Saturday,
            Weekday::Invalid => Self::Invalid,
        }
    }
}
/// Convert a satkit::Weekday into a Python PyWeekday object
impl From<&PyTimeScale> for TimeScale {
    fn from(s: &PyTimeScale) -> Self {
        match s {
            PyTimeScale::Invalid => Self::Invalid,
            PyTimeScale::UTC => Self::UTC,
            PyTimeScale::TT => Self::TT,
            PyTimeScale::UT1 => Self::UT1,
            PyTimeScale::TAI => Self::TAI,
            PyTimeScale::GPS => Self::GPS,
            PyTimeScale::TDB => Self::TDB,
        }
    }
}

impl From<PyTimeScale> for TimeScale {
    fn from(s: PyTimeScale) -> Self {
        match s {
            PyTimeScale::Invalid => Self::Invalid,
            PyTimeScale::UTC => Self::UTC,
            PyTimeScale::TT => Self::TT,
            PyTimeScale::UT1 => Self::UT1,
            PyTimeScale::TAI => Self::TAI,
            PyTimeScale::GPS => Self::GPS,
            PyTimeScale::TDB => Self::TDB,
        }
    }
}

/// Representation of an instant in time
///
/// This has functionality similar to the "datetime" object, and in fact has
/// the ability to convert to an from the "datetime" object.  However, a separate
/// time representation is needed as the "datetime" object does not allow for
/// conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)
///
/// Note: If no arguments are passed in, the created object represents the current time
///
/// Note: UTC before 1972 follows the "rubber second" model of USNO
/// ``tai-utc.dat`` / ERFA ``dat`` from 1961-01-01 (TAI - UTC drifts and steps
/// by fractions of a second); before 1961, UTC is taken to equal TAI.
///
/// Args:
///     year (int): Gregorian year (e.g., 2024) (optional)
///     month (int): Gregorian month (1 = January, 2 = February, ...) (optional)
///     day (int): Day of month, beginning with 1 (optional)
///     hour (int): Hour of day, in range [0,23] (optional), default is 0
///     min (int): Minute of hour, in range [0,59] (optional), default is 0
///     sec (float): floating point second of minute, in range [0,60) (optional), defialt is 0
///     scale (satkit.timescale): Time scale (optional), default is satkit.timescale.UTC
///
/// Returns:
///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
#[pyclass(name = "time", module = "satkit", from_py_object)]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone, Debug)]
pub struct PyInstant(pub Instant);

// The Python API names conversions `to_*` (paired with `from_*`); PyInstant is
// Copy, so clippy would rather those took `self` by value. PyO3 methods take
// `&self`, and the name is fixed by the Python API, so the lint is silenced here.
#[allow(clippy::wrong_self_convention)]
#[pymethods]
impl PyInstant {
    /// Representation of an instant in time
    ///
    /// This has functionality similar to the "datetime" object, and in fact has
    /// the ability to convert to an from the "datetime" object.  However, a separate
    /// time representation is needed as the "datetime" object does not allow for
    /// conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)
    ///
    /// Args:
    ///     year (int, optional): Gregorian year (e.g., 2024) (optional)
    ///    month (int, optional): Gregorian month (1 = January, 2 = February, ...) (optional)
    ///     day (int, optional): Day of month, beginning with 1 (optional)
    ///    hour (int, optional): Hour of day, in range [0,23] (optional), default is 0
    ///     min (int, optional): Minute of hour, in range [0,59] (optional), default is 0
    ///     sec (float, optional): floating point second of minute, in range [0,60) (optional), defialt is 0
    ///     string (str, optional): If this is only argument, attempt to parse time from string
    ///
    /// Note: If no arguments are passed in, the created object represents the current time
    ///
    /// The J2000 epoch: 2000-01-01 12:00:00 TT
    #[classattr]
    #[allow(non_snake_case)]
    const fn J2000() -> Self {
        Self(Instant::J2000)
    }

    /// The GPS epoch: 1980-01-06 00:00:00 UTC
    #[classattr]
    #[allow(non_snake_case)]
    const fn GPS_EPOCH() -> Self {
        Self(Instant::GPS_EPOCH)
    }

    /// The Modified Julian Date epoch: 1858-11-17 00:00:00 UTC
    #[classattr]
    #[allow(non_snake_case)]
    const fn MJD_EPOCH() -> Self {
        Self(Instant::MJD_EPOCH)
    }

    /// The Unix epoch: 1970-01-01 00:00:00 UTC
    #[classattr]
    #[allow(non_snake_case)]
    const fn UNIX_EPOCH() -> Self {
        Self(Instant::UNIX_EPOCH)
    }

    /// Create a satkit.time object
    ///
    /// Args:
    ///     *args: Either no arguments (current time), a single string, or
    ///         Gregorian components (year, month, day) or
    ///         (year, month, day, hour, minute, second)
    ///     scale (satkit.timescale, optional): Time scale in which the Gregorian
    ///         components are interpreted. Default is satkit.timescale.UTC.
    ///         Ignored when constructing from a string or with no arguments.
    ///
    /// Returns:
    ///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
    #[new]
    #[pyo3(signature=(*py_args, scale=&PyTimeScale::UTC))]
    fn py_new(py_args: &Bound<'_, PyTuple>, scale: &PyTimeScale) -> Result<Self> {
        if py_args.is_empty() {
            Ok(Self(Instant::now()))
        } else if py_args.len() == 3 || py_args.len() == 6 {
            let int = |i: usize| py_args.get_item(i)?.extract::<i32>();
            let (year, month, day) = (int(0)?, int(1)?, int(2)?);
            let (hour, min, sec) = if py_args.len() == 6 {
                (int(3)?, int(4)?, py_args.get_item(5)?.extract::<f64>()?)
            } else {
                (0, 0, 0.0)
            };
            Ok(Self(Instant::from_datetime_with_scale(
                year,
                month,
                day,
                hour,
                min,
                sec,
                scale.into(),
            )?))
        } else if py_args.len() == 1 {
            let item = py_args.get_item(0)?;
            let s = item.extract::<&str>()?;

            // Input is a string, first try rfc3339 format
            match Instant::from_rfc3339(s) {
                Ok(v) => Ok(Self(v)),
                Err(_) => {
                    // Now try multiple formats
                    Self::from_string(s)
                }
            }
        } else {
            bail!("Must pass in year, month, day or year, month, day, hour, min, sec");
        }
    }

    /// Create satkit.time object from a string, guessing its format
    ///
    /// RFC 3339 is tried first (see ``from_rfc3339``). Otherwise the numbers
    /// in the string are read in year, month, day, hour, minute, second
    /// order, so ISO-ordered strings (``"2024-01-04 13:14:12.123"``) and
    /// month-name strings (``"March 4 2024"``) work, but locale-ordered dates
    /// such as ``MM/DD/YYYY`` are not supported: use ``strptime`` for those.
    ///
    /// Args:
    ///    string (str): String representing time
    ///
    /// Notes:
    ///    - A number after ``.`` following the seconds is the fraction of a
    ///      second, rounded to the nearest microsecond.
    ///    - A number after ``+`` or ``-`` following the minutes is a UTC
    ///      offset (``±HHMM``, ``±HH:MM`` or ``±HH``; hours 00-23, minutes
    ///      00-59) and is applied: ``"2024-01-04 13:14:12 +0100"`` is
    ///      ``12:14:12Z``. Any other extra number is an error.
    ///    - Seconds default to 0 (``"2024-01-04 13:14"``); an hour without
    ///      minutes is an error, and a date alone is midnight.
    ///    - Words other than month names (weekday names, ``T``, ``Z``,
    ///      ``UTC``, ...) are ignored, so a zone *name* is not applied:
    ///      without a numeric offset the time is UTC.
    ///    - This is probably not what you want. Use with caution, and prefer
    ///      ``from_rfc3339`` or ``strptime`` when the format is known.
    ///
    /// Returns:
    ///   satkit.time: Time object representing input time
    ///
    /// Raises:
    ///   RuntimeError: If input string cannot be parsed
    ///
    #[staticmethod]
    fn from_string(string: &str) -> Result<Self> {
        Ok(Instant::from_string(string).map(Self)?)
    }

    /// Create satkit.time object from string with given format
    ///
    /// Args:
    ///   date_string (str): String representing time
    ///   format (str): Format string
    ///
    /// Returns:
    ///   satkit.time: Time object representing input time
    ///
    /// Raises:
    ///   RuntimeError: If the string does not match the format
    ///
    /// The format string is a subset of the Python "datetime" strptime
    /// format. Characters other than format codes must match literally, and
    /// the whole string must be consumed: leftover input is an error.
    ///
    /// Format Codes:
    /// %Y: Year: exactly four digits, or a sign and at least four digits
    ///     (ISO 8601 expanded years such as -0044 or +10000, as strftime
    ///     writes them outside 0000-9999)
    /// %m: Month, exactly two digits (01-12)
    /// %d: Day of the month, exactly two digits (01-31)
    /// %H: Hour (24-hour clock), exactly two digits (00-23)
    /// %M: Minute, exactly two digits (00-59)
    /// %S: Second, exactly two digits (00-59, or 60 in a leap second)
    /// %f: Fraction of a second: one or more digits (5 is 500 ms), rounded
    ///     to the nearest microsecond beyond six
    /// %z: UTC offset ±HH:MM, ±HHMM or ±HH (exactly two digits per field,
    ///     hours 00-23, minutes 00-59), or Z / z for UTC. +HHMM means local
    ///     time is ahead of UTC, so 12:00:00+0100 is 11:00:00Z
    /// %b: Abbreviated month name (Jan, Feb, ...)
    /// %B: Full month name (January, February, ...)
    /// %%: A literal %
    #[staticmethod]
    fn strptime(date_string: &str, format: &str) -> Result<Self> {
        Ok(Instant::strptime(date_string, format).map(Self)?)
    }

    /// Format time object as string
    ///
    /// Args:
    ///  format (str): Format string
    ///
    /// Returns:
    /// str: String representing time in given format
    ///
    /// Raises:
    /// ValueError: If input string cannot be formatted
    ///
    /// Format Codes:
    /// %Y: Year with century as a decimal number
    /// %m: Month as a zero-padded decimal number
    /// %d: Day of the month as a zero-padded decimal number
    /// %H: Hour (24-hour clock) as a zero-padded decimal number
    /// %M: Minute as a zero-padded decimal number
    /// %S: Second as a zero-padded decimal number
    /// %f: Microsecond as a decimal number, with possible trailing zeros (1 to 6 digits)
    /// %z: UTC offset in the form +HHMM or -HHMM
    /// %A: Weekday as locale’s full name
    /// %b: Month as locale’s abbreviated name
    /// %B: Month as locale’s full name
    /// %w: Weekday as a decimal number, where 0 is Sunday and 6 is Saturday
    ///
    fn strftime(&self, format: &str) -> Result<String> {
        self.0
            .strftime(format)
            .map_err(|e| anyhow::anyhow!("Could not format time string: {}", e))
    }

    /// Create satkit.time object from an RFC 3339 string
    ///
    /// Notes:
    ///   - Format ``YYYY-MM-DDTHH:MM:SS[.fff...][zone]``
    ///     (https://tools.ietf.org/html/rfc3339, which overlaps with
    ///     ISO 8601). ``T`` may be ``t``. The fraction has one or more digits
    ///     and is rounded to the nearest microsecond beyond six. Surrounding
    ///     whitespace is ignored; anything else left over is an error.
    ///   - The zone is ``Z`` / ``z``, or a UTC offset ``±HH:MM`` (RFC 3339),
    ///     ``±HHMM`` or ``±HH`` (ISO 8601 forms, also accepted), with hours
    ///     00-23 and minutes 00-59. The offset is applied:
    ///     ``2024-01-01T12:00:00+01:00`` is ``11:00:00Z``. It shifts the
    ///     calendar label, so it is exact across a leap second.
    ///   - Without a zone the time is taken as UTC (RFC 3339 itself requires
    ///     one).
    ///   - The year is four digits, or a sign and at least four digits
    ///     (ISO 8601 expanded years such as ``-0001`` or ``+10000``, as
    ///     ``to_rfc3339`` writes them outside 0000-9999).
    ///
    /// Args:
    ///   rfc3339 (str): String representing time
    ///
    /// Returns:
    ///   satkit.time: Time object representing input time
    ///
    /// Raises:
    ///   ValueError: If input string cannot be parsed; the message gives the
    ///       reason
    ///
    #[staticmethod]
    fn from_rfc3339(rfc3339: &str) -> PyResult<Self> {
        Instant::from_rfc3339(rfc3339).map(Self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Could not parse time string: {e}"))
        })
    }

    /// Convert satkit.time object to RFC3339 string
    ///
    /// Returns:
    /// str: String representing time in RFC3339 format : "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// Notes:
    ///  RFC3339 is a standard for representing time in a string format
    ///  Return string also matches ISO8601
    fn to_rfc3339(&self) -> String {
        self.0.as_rfc3339()
    }

    /// Convert satkit.time object to ISO8601 string
    ///
    /// Returns:
    /// str: String representing time in ISO8601 format : "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// Notes:
    /// ISO8601 is a standard for representing time in a string format
    /// Return string also matches RFC3339
    fn to_iso8601(&self) -> String {
        self.0.as_iso8601()
    }

    /// Return current time
    ///
    /// Returns:
    ///     satkit.time: Time object representing current time
    #[staticmethod]
    fn now() -> Self {
        Self(Instant::now())
    }

    /// Return time object representing input date
    ///
    /// Args:
    ///     year (int): Gregorian year (e.g., 2024)
    ///     month (int): Gregorian month (1 = January, 2 = February, ...)
    ///     day (int): Day of month, beginning with 1
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input date
    #[staticmethod]
    fn from_date(year: i32, month: i32, day: i32) -> Result<Self> {
        Ok(Self(Instant::from_date(year, month, day)?))
    }

    /// Return time object representing input modified Julian date and time scale
    ///
    /// Args:
    ///   mjd (float): The Modified Julian Date, days
    ///   scale (satkit.timescale, optional): The time scale. Default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of modified julian date with given scale
    #[staticmethod]
    #[pyo3(signature=(mjd, scale=&PyTimeScale::UTC))]
    fn from_mjd(mjd: f64, scale: &PyTimeScale) -> Self {
        Self(Instant::from_mjd_with_scale(mjd, scale.into()))
    }

    /// Return time object representing input unix time, which is UTC seconds
    /// since Jan 1, 1970 00:00:00 (not counting leap seconds)
    ///
    /// Args:
    ///    unixtime (float): the unixtime, UTC seconds since 1970-01-01 00:00:00 (excluding leap seconds)
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input unixtime
    #[staticmethod]
    fn from_unixtime(unixtime: f64) -> Self {
        Self(Instant::from_unixtime(unixtime))
    }

    /// Return time object representing input Julian date and time scale
    ///
    /// Args:
    ///    jd (float): The Julian Date, days
    ///   scale (satkit.timescale, optional): The time scale. Default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of julian date with given scale
    #[staticmethod]
    #[pyo3(signature=(jd, scale=&PyTimeScale::UTC))]
    fn from_jd(jd: f64, scale: &PyTimeScale) -> Self {
        Self(Instant::from_jd_with_scale(jd, scale.into()))
    }

    /// Convert time object to UTC Gregorian date
    ///
    /// Returns:
    ///    (int, int, int): Tuple with 3 elements representing Gregorian year, month, and day
    fn to_date(&self) -> (i32, i32, i32) {
        let dt = self.0.as_datetime();
        (dt.0, dt.1, dt.2)
    }

    /// Convert time object to UTC Gregorian date and time, with fractional seconds
    ///
    /// Returns:
    ///     (int, int, int, int, int, float): Tuple with 6 elements representing Gregorian year, month, day, hour, minute, and second
    ///
    fn to_gregorian(&self) -> (i32, i32, i32, i32, i32, f64) {
        self.0.as_datetime()
    }

    /// Return the 1-based Gregorian day of the year (1 = January 1, 365 = December 31)
    /// Leap-year aware
    ///
    /// Returns:
    ///     int : The 1-based day of the year
    ///
    #[getter]
    fn day_of_year(&self) -> u32 {
        self.0.day_of_year()
    }

    /// Create satkit.time representing input UTC Gregorian date and time
    ///
    /// Args:
    ///     year (int): Gregorian year (e.g., 2024)
    ///     month (int): Gregorian month (1 = January, 2 = February, ...)
    ///     day (int): Day of month, beginning with 1
    ///     hour (int): Hour of day, in range [0,23]
    ///     min (int): Minute of hour, in range [0,59]
    ///     sec (float): floating point second of minute, in range [0,60)
    ///     scale (satkit.timescale, optional): Time scale, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///    satkit.time: satkit.time object representing input Gregorian date and time
    #[staticmethod]
    #[pyo3(signature=(year, month, day, hour, min, sec))]
    fn from_gregorian(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        min: i32,
        sec: f64,
    ) -> Result<Self> {
        Ok(Instant::from_datetime(year, month, day, hour, min, sec).map(Self)?)
    }

    /// Convert from Python datetime object
    ///
    /// Follows Python's own convention (``datetime.timestamp()``): a naive
    /// datetime (no ``tzinfo``) is interpreted in the machine's local time
    /// zone, not UTC; an aware datetime uses its own UTC offset. For UTC,
    /// pass ``tzinfo=datetime.timezone.utc`` or build a ``satkit.time``
    /// directly. The conversion is exact to the microsecond (it does not go
    /// through the float ``timestamp()``).
    ///
    /// Args:
    ///     datetime (datetime.datetime): datetime object to convert
    ///
    /// Returns:
    ///     satkit.time: satkit.time object representing the same instant as the input datetime
    #[staticmethod]
    fn from_datetime(dt: &Bound<'_, PyDateTime>) -> PyResult<Self> {
        Ok(Self(datetime_to_instant(dt)?))
    }

    /// Convert to Python datetime object
    ///
    /// Args:
    ///     utc (bool, optional): If true (default), return an aware datetime in UTC;
    ///         if false, return a naive datetime in the machine's local time zone,
    ///         which round-trips through ``from_datetime``
    ///
    /// Returns:
    ///     datetime.datetime:  datetime object matching the input satkit.time
    ///
    #[pyo3(signature = (utc=true))]
    fn to_datetime(&self, py: Python<'_>, utc: bool) -> PyResult<Py<PyAny>> {
        instant_to_datetime(py, &self.0, utc)
    }

    /// Convert to Python datetime object
    ///
    /// Args:
    ///     utc (bool, optional): If true (default), return an aware datetime in UTC;
    ///         if false, return a naive datetime in the machine's local time zone,
    ///         which round-trips through ``from_datetime``
    ///
    /// Returns:
    ///     datetime.datetime:  datetime object matching the input satkit.time
    ///
    #[pyo3(signature = (utc=true))]
    fn datetime(&self, py: Python<'_>, utc: bool) -> PyResult<Py<PyAny>> {
        warn_deprecated(
            py,
            c"satkit.time.datetime() is deprecated; use satkit.time.to_datetime() instead.",
        )?;
        self.to_datetime(py, utc)
    }

    /// Convert to Modified Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional): Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Modified Julian Date, days
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_mjd(&self, scale: &PyTimeScale) -> f64 {
        self.0.as_mjd_with_scale(scale.into())
    }

    /// Convert to Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional: Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Julian Date, days
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_jd(&self, scale: &PyTimeScale) -> f64 {
        self.0.as_jd_with_scale(scale.into())
    }

    /// Convert to Unix time (seconds since 1970-01-01 00:00:00 UTC)
    /// Excludes leap seconds
    ///
    /// Returns:
    ///     float: Unix time (seconds since 1970-01-01 00:00:00 UTC)
    fn to_unixtime(&self) -> f64 {
        self.0.as_unixtime()
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_rfc3339()``.
    fn as_rfc3339(&self, py: Python<'_>) -> PyResult<String> {
        warn_deprecated(
            py,
            c"time.as_rfc3339() is deprecated since 0.23 and will be removed in 0.25; use time.to_rfc3339()",
        )?;
        Ok(self.to_rfc3339())
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_iso8601()``.
    fn as_iso8601(&self, py: Python<'_>) -> PyResult<String> {
        warn_deprecated(
            py,
            c"time.as_iso8601() is deprecated since 0.23 and will be removed in 0.25; use time.to_iso8601()",
        )?;
        Ok(self.to_iso8601())
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_date()``.
    fn as_date(&self, py: Python<'_>) -> PyResult<(i32, i32, i32)> {
        warn_deprecated(
            py,
            c"time.as_date() is deprecated since 0.23 and will be removed in 0.25; use time.to_date()",
        )?;
        Ok(self.to_date())
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_gregorian()``.
    fn as_gregorian(&self, py: Python<'_>) -> PyResult<(i32, i32, i32, i32, i32, f64)> {
        warn_deprecated(
            py,
            c"time.as_gregorian() is deprecated since 0.23 and will be removed in 0.25; use time.to_gregorian()",
        )?;
        Ok(self.to_gregorian())
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_datetime()``.
    #[pyo3(signature = (utc=true))]
    fn as_datetime(&self, py: Python<'_>, utc: bool) -> PyResult<Py<PyAny>> {
        warn_deprecated(
            py,
            c"time.as_datetime() is deprecated since 0.23 and will be removed in 0.25; use time.to_datetime()",
        )?;
        self.to_datetime(py, utc)
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_mjd()``.
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn as_mjd(&self, py: Python<'_>, scale: &PyTimeScale) -> PyResult<f64> {
        warn_deprecated(
            py,
            c"time.as_mjd() is deprecated since 0.23 and will be removed in 0.25; use time.to_mjd()",
        )?;
        Ok(self.to_mjd(scale))
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_jd()``.
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn as_jd(&self, py: Python<'_>, scale: &PyTimeScale) -> PyResult<f64> {
        warn_deprecated(
            py,
            c"time.as_jd() is deprecated since 0.23 and will be removed in 0.25; use time.to_jd()",
        )?;
        Ok(self.to_jd(scale))
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_unixtime()``.
    fn as_unixtime(&self, py: Python<'_>) -> PyResult<f64> {
        warn_deprecated(
            py,
            c"time.as_unixtime() is deprecated since 0.23 and will be removed in 0.25; use time.to_unixtime()",
        )?;
        Ok(self.to_unixtime())
    }

    /// Return time object representing input GPS week and seconds of week
    ///
    /// Args:
    ///     week (int): GPS week number
    ///     seconds (float): GPS seconds of week, seconds
    ///
    /// Returns:
    ///     satkit.time: Time object representing input GPS week and second
    #[staticmethod]
    fn from_gps_week_and_second(week: i32, seconds: f64) -> Self {
        Self(Instant::from_gps_week_and_second(week, seconds))
    }

    /// Day of the week (UTC)
    ///
    /// Returns:
    ///     satkit.weekday: Day of the week
    #[getter]
    fn weekday(&self) -> PyWeekday {
        PyWeekday::from(self.0.day_of_week())
    }

    /// Add to satkit time a duration or list or numpy array of durations
    ///
    /// Args:
    ///     other (duration|list|numpy.ndarray|float): Duration or list of durations to add.
    ///         If type is float, units are days
    ///
    /// Returns:
    ///     satkit.time|numpy.ndarray: New time object or numpy array of time objects representing input time plus input duration(s)
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.shift(other, |t, d| t + d)
    }

    /// Subtract duration or take difference in times
    ///
    /// Args:
    ///     other (duration|list|numpy.ndarray|float|satkit.time): Duration or list of durations to subtract, or time object to take difference.
    ///         If type is float, units are days
    ///
    /// Returns:
    ///     satkit.time|numpy.ndarray|satkit.duration: New time object or numpy array of time objects representing input time minus input duration(s), or duration object representing difference between two time objects
    ///     (a numpy array of duration objects for a list of time objects)
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(tm2) = other.cast::<Self>() {
            return PyDuration(self.0 - tm2.borrow().0).into_py_any(py);
        }
        // A non-empty list of times: element-wise differences, as an object
        // array of durations (an empty list is handled by `shift`, giving an
        // empty array like every other list operand)
        if let Ok(list) = other.cast::<pyo3::types::PyList>() {
            if !list.is_empty() && list.iter().all(|x| x.is_instance_of::<Self>()) {
                let objs = list
                    .iter()
                    .map(|x| {
                        let t: PyRef<Self> = x.extract()?;
                        PyDuration(self.0 - t.0).into_py_any(py)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return np::PyArray1::<Py<PyAny>>::from_vec(py, objs).into_py_any(py);
            }
        }
        self.shift(other, |t, d| t - d)
    }

    // Comparison operators are below

    fn __le__(&self, other: &Self) -> bool {
        self.0 <= other.0
    }

    fn __ge__(&self, other: &Self) -> bool {
        self.0 >= other.0
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.0 < other.0
    }

    fn __gt__(&self, other: &Self) -> bool {
        self.0 > other.0
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    // Backed by an exact microsecond count, so hashing the raw integer is
    // consistent with __eq__ (defining __eq__ alone would make time unhashable).
    fn __hash__(&self) -> isize {
        self.0.raw as isize
    }

    ///
    /// Add given number of UTC days to a time object, and return the result
    ///
    /// Args:
    ///     days (float): Number of days to add
    ///
    /// Returns:
    ///     satkit.time: Time object representing input time plus given number of days
    ///
    /// Note:
    ///
    /// A UTC days is defined as being exactly 86400 seconds long.  This
    /// avoids the ambiguity of adding a "day" to a time that has a leap second
    fn add_utc_days(&self, days: f64) -> Self {
        Self(self.0.add_utc_days(days))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::empty(py);
        (tp, d)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let s = state.as_bytes(py);
        if s.len() != 8 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let raw = i64::from_le_bytes(s.try_into()?);
        self.0 = Instant::new(raw);
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &i64::to_le_bytes(self.0.raw)).into())
    }
}

impl PyInstant {
    /// `self (op) other` for a number of days, a duration, or a list / 1-D
    /// float array of either (element-wise, returning an object array)
    fn shift(
        &self,
        other: &Bound<'_, PyAny>,
        op: fn(Instant, satkit::Duration) -> Instant,
    ) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let days = finite_days;
        let durs: Vec<satkit::Duration> = if other.is_instance_of::<np::PyArray1<f64>>() {
            let arr = other.extract::<np::PyReadonlyArray1<f64>>()?;
            arr.as_array()
                .iter()
                .map(|x| days(*x))
                .collect::<PyResult<_>>()?
        } else if other.is_instance_of::<pyo3::types::PyList>() {
            if let Ok(v) = other.extract::<Vec<f64>>() {
                v.into_iter().map(days).collect::<PyResult<_>>()?
            } else if let Ok(v) = other.extract::<Vec<PyDuration>>() {
                v.into_iter().map(|d| d.0).collect()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid types in list",
                ));
            }
        } else if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
        {
            // A Python int too large for f64 raises OverflowError in extract
            return Self(op(self.0, days(other.extract::<f64>()?)?)).into_py_any(py);
        } else if let Ok(d) = other.cast::<PyDuration>() {
            return Self(op(self.0, d.borrow().0)).into_py_any(py);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid type for rhs",
            ));
        };
        let objs = durs
            .into_iter()
            .map(|d| Self(op(self.0, d)).into_py_any(py))
            .collect::<PyResult<Vec<_>>>()?;
        np::PyArray1::<Py<PyAny>>::from_vec(py, objs).into_py_any(py)
    }
}

/// A duration of `days` days, refusing NaN and infinities with `ValueError`
/// and out-of-range values with `OverflowError` (`Duration::from_days` maps
/// NaN to zero and saturates the rest, so `t + nan` silently returned `t`)
fn finite_days(days: f64) -> PyResult<satkit::Duration> {
    crate::pyduration::check_duration_value(days, 86_400.0e6, "number of days")?;
    Ok(satkit::Duration::from_days(days))
}

/// 1970-01-01T00:00:00+00:00 as an aware Python datetime
fn unix_epoch_utc(py: Python<'_>) -> PyResult<Bound<'_, PyDateTime>> {
    let utc = PyTzInfo::utc(py)?;
    PyDateTime::new(py, 1970, 1, 1, 0, 0, 0, 0, Some(&*utc))
}

/// Convert a Python `datetime` with Python's own convention
/// (`datetime.timestamp()`): a naive datetime is local time, an aware one
/// uses its own offset. Shared by every binding that accepts a datetime.
///
/// Exact: the datetime's own fields and UTC offset are combined in integer
/// microseconds (`aware - epoch` is a `timedelta`). Going through the f64
/// `timestamp()` instead lost a microsecond for ~2% of datetimes.
pub(crate) fn datetime_to_instant(tm: &Bound<PyDateTime>) -> PyResult<Instant> {
    let py = tm.py();
    // A naive datetime (or one whose tzinfo reports no offset, which Python
    // also treats as naive) is local time; `astimezone()` resolves it,
    // honouring `fold`, to an aware datetime with the local offset. It can
    // raise (e.g. years outside the platform's time_t range); propagate.
    let aware = if tm.call_method0("utcoffset")?.is_none() {
        tm.call_method0("astimezone")?
    } else {
        tm.clone().into_any()
    };
    let delta = aware.sub(unix_epoch_utc(py)?)?;
    let delta = delta.cast::<PyDelta>()?;
    let us = (delta.get_days() as i64 * 86_400 + delta.get_seconds() as i64) * 1_000_000
        + delta.get_microseconds() as i64;
    Ok(Instant::from_unixtime_microseconds(us))
}

/// Convert an instant to a Python `datetime`, exactly: an aware UTC
/// datetime (`utc = true`) or a naive local one. Python datetimes cannot
/// express a leap second, so `23:59:60.x` comes back as `23:59:59.x`, as
/// with Unix time.
fn instant_to_datetime(py: Python<'_>, t: &Instant, utc: bool) -> PyResult<Py<PyAny>> {
    let us = t.as_unixtime_microseconds();
    let secs = us.div_euclid(1_000_000);
    let micros = us.rem_euclid(1_000_000) as i32;
    let overflow =
        |_| pyo3::exceptions::PyOverflowError::new_err("satkit.time out of range for datetime");
    if utc {
        // epoch + timedelta: integer arithmetic, and not limited to the
        // platform's time_t range the way fromtimestamp() is
        let delta = PyDelta::new(
            py,
            i32::try_from(secs.div_euclid(86_400)).map_err(overflow)?,
            secs.rem_euclid(86_400) as i32,
            micros,
            false,
        )?;
        Ok(unix_epoch_utc(py)?.add(delta)?.unbind())
    } else {
        // Local time: fromtimestamp() of the whole second (exact for an
        // integer, and it sets `fold` for the repeated hour at the end of
        // DST, so the result round-trips), then the microseconds.
        let dt = PyDateTime::from_timestamp(py, secs as f64, None)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("microsecond", micros)?;
        Ok(dt.call_method("replace", (), Some(&kwargs))?.unbind())
    }
}

/// A single time argument: `satkit.time` or `datetime.datetime` (the stubs'
/// `TimeScalar`). Use as a `#[pyfunction]` parameter type in place of
/// `PyInstant` wherever the stub accepts either.
#[derive(Clone, Copy, Debug)]
pub struct TimeArg(pub Instant);

impl<'a, 'py> FromPyObject<'a, 'py> for TimeArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(t) = obj.cast::<PyInstant>() {
            return Ok(Self(t.borrow().0));
        }
        if let Ok(dt) = obj.cast::<PyDateTime>() {
            return Ok(Self(datetime_to_instant(&dt)?));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected satkit.time or datetime.datetime, got {}",
            obj.get_type()
        )))
    }
}

/// Times extracted from a Python time argument, remembering whether the
/// argument was a single time (`scalar`) or a list / array of times, so a
/// vectorised function returns a scalar only for scalar input: a one-element
/// list gives a one-element list (or a (1, ...) array), as the stubs promise.
pub struct TimeInput {
    pub times: Vec<Instant>,
    pub scalar: bool,
}

pub trait ToTimeVec {
    /// The times, whether the input was a single time or a list / array
    fn to_time_vec(&self) -> PyResult<Vec<Instant>>;
    /// The times, plus whether the input was a single time
    fn to_time_input(&self) -> PyResult<TimeInput>;
}

impl ToTimeVec for &Bound<'_, PyAny> {
    fn to_time_vec(&self) -> PyResult<Vec<Instant>> {
        Ok(self.to_time_input()?.times)
    }

    fn to_time_input(&self) -> PyResult<TimeInput> {
        // "Scalar" time input case
        if self.is_instance_of::<PyInstant>() || self.is_instance_of::<PyDateTime>() {
            let t: TimeArg = self.extract()?;
            return Ok(TimeInput {
                times: vec![t.0],
                scalar: true,
            });
        }
        Ok(TimeInput {
            times: time_array_to_vec(self)?,
            scalar: false,
        })
    }
}

/// Times from a list or 1-D numpy object array of `satkit.time` /
/// `datetime.datetime`
fn time_array_to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Instant>> {
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| item.extract::<TimeArg>().map(|t| t.0))
            .collect::<PyResult<Vec<_>>>()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Not a list of satkit.time or datetime.datetime: {e}"
                ))
            })
    } else if obj.is_instance_of::<numpy::PyArray1<Py<PyAny>>>() {
        let v = obj
            .extract::<numpy::PyReadonlyArray1<Py<PyAny>>>()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Invalid satkit.time or datetime.datetime input: {e}"
                ))
            })?;
        let py = obj.py();
        v.as_array()
            .iter()
            .map(|p| p.bind(py).extract::<TimeArg>().map(|t| t.0))
            .collect::<PyResult<Vec<_>>>()
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Invalid satkit.time input: numpy array must contain satkit.time \
                     or datetime.datetime elements",
                )
            })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Invalid satkit.time or datetime.datetime input",
        ))
    }
}
