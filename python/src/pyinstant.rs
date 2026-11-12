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
use numpy::PyArrayMethods;

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
/// * Invalid = placeholder that is not a time scale; passing it as a
///   ``scale`` argument raises ValueError
///
#[derive(Clone, PartialEq, Eq, Hash)]
#[pyclass(
    name = "timescale",
    module = "satkit",
    eq,
    eq_int,
    hash,
    frozen,
    from_py_object
)]
pub enum PyTimeScale {
    /// Placeholder that is not a time scale: every satkit.time method that
    /// takes a ``scale`` raises ValueError when given it
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

#[derive(Clone, PartialEq, Eq, Hash)]
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
/// - `Invalid`: placeholder that is not a day; ``time.weekday`` never returns it
#[pyclass(
    name = "weekday",
    module = "satkit",
    eq,
    eq_int,
    hash,
    frozen,
    from_py_object
)]
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
/// Note: UTC before 1972 follows the "rubber second" model of USNO
/// ``tai-utc.dat`` / ERFA ``dat`` from 1961-01-01 (TAI - UTC drifts and steps
/// by fractions of a second); before 1961, UTC is taken to equal TAI.
///
/// Accepted forms (the arguments are positional):
///
/// - ``time()``: the current time
/// - ``time(string)``: parse a string, RFC 3339 first, then other common formats
/// - ``time(t)``: the instant of a satkit.time, datetime.datetime (a naive
///   one is local time, as in ``from_datetime``) or numpy.datetime64 (UTC)
/// - ``time(year, month, day)``: midnight at the start of the day
/// - ``time(year, month, day, hour, min, sec)``: all six components; the
///   hour, minute and second are given together or not at all
///
/// Args:
///     string (str): String representation of time
///     year (int): Gregorian year (e.g., 2024)
///     month (int): Gregorian month (1 = January, 2 = February, ...)
///     day (int): Day of month, beginning with 1
///     hour (int): Hour of day, in range [0,23]
///     min (int): Minute of hour, in range [0,59]
///     sec (float): floating point second of minute, in range [0,60); up to 61
///         within a UTC leap second. Rounded to the nearest microsecond
///     scale (satkit.timescale, optional): Time scale in which the Gregorian
///         components are interpreted, default is satkit.timescale.UTC. Only
///         for the component forms
///
/// Returns:
///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
///
/// Raises:
///     TypeError: for a wrong number or type of arguments, or ``scale`` given
///         with the string or no-argument form
///     ValueError: if the string cannot be parsed, a component is out of
///         range, or ``scale`` is satkit.timescale.Invalid
#[pyclass(name = "time", module = "satkit", from_py_object)]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone, Debug)]
pub struct PyInstant(pub Instant);

// The Python API names conversions `to_*` (paired with `from_*`); PyInstant is
// Copy, so clippy would rather those took `self` by value. PyO3 methods take
// `&self`, and the name is fixed by the Python API, so the lint is silenced here.
#[allow(clippy::wrong_self_convention)]
#[pymethods]
impl PyInstant {
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
    ///     *args: Either no arguments (current time), a single string, a
    ///         single satkit.time, datetime.datetime or numpy.datetime64, or
    ///         Gregorian components (year, month, day) or
    ///         (year, month, day, hour, minute, second)
    ///     scale (satkit.timescale, optional): Time scale in which the Gregorian
    ///         components are interpreted. Default is satkit.timescale.UTC.
    ///         Only for the component forms: giving it with a string or with
    ///         no arguments raises TypeError.
    ///
    /// Returns:
    ///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
    ///
    /// Raises:
    ///     TypeError: If the number of positional arguments is not 0, 1, 3 or
    ///         6, a single argument is not a string or a time, a component is not a
    ///         number (the first five must be integers), or ``scale`` is
    ///         given with the string or no-argument form
    ///     ValueError: If the string form cannot be parsed, a component is
    ///         out of range, the date is beyond about ±292,000 years, or
    ///         ``scale`` is satkit.timescale.Invalid
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
                valid_scale(scale)?,
            )?))
        } else if py_args.len() == 1 {
            let item = py_args.get_item(0)?;
            if let Ok(s) = item.extract::<&str>() {
                // Input is a string: `from_string` tries RFC 3339 first, then
                // the looser formats, and raises ValueError if neither parses
                return Ok(Self::from_string(s)?);
            }
            if is_time_scalar(&item) {
                // satkit.time, datetime.datetime or numpy.datetime64
                return Ok(Self(item.extract::<TimeArg>()?.0));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "a single argument must be a str, satkit.time, datetime.datetime \
                 or numpy.datetime64, got {}",
                item.get_type()
            ))
            .into())
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
    ///   ValueError: If input string cannot be parsed; the message gives the
    ///       reason
    ///
    #[staticmethod]
    fn from_string(string: &str) -> PyResult<Self> {
        Instant::from_string(string).map(Self).map_err(parse_error)
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
    ///   ValueError: If the string does not match the format; the message
    ///       gives the reason
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
    fn strptime(date_string: &str, format: &str) -> PyResult<Self> {
        Instant::strptime(date_string, format)
            .map(Self)
            .map_err(parse_error)
    }

    /// Format time object (in UTC) as string
    ///
    /// Args:
    ///     format (str): Format string
    ///
    /// Returns:
    ///     str: String representing time in given format
    ///
    /// Raises:
    ///     ValueError: If the format has a code not listed below, or ends
    ///         with a lone %
    ///
    /// Format Codes (a subset of Python's; there is no %z, %j, %y or %p):
    /// %Y: Year, four digits (0000-9999); outside that range a sign and at
    ///     least four digits (-0044, +10000)
    /// %m: Month as a zero-padded decimal number (01-12)
    /// %d: Day of the month as a zero-padded decimal number (01-31)
    /// %H: Hour (24-hour clock) as a zero-padded decimal number (00-23)
    /// %M: Minute as a zero-padded decimal number (00-59)
    /// %S: Second as a zero-padded decimal number (00-59, or 60 inside a
    ///     leap second)
    /// %f: Microsecond, always six digits (000000-999999)
    /// %a: Abbreviated English weekday name (Sun, Mon, ...)
    /// %A: Full English weekday name (Sunday, Monday, ...)
    /// %b: Abbreviated English month name (Jan, Feb, ...)
    /// %B: Full English month name (January, February, ...)
    /// %w: Weekday as a decimal number, where 0 is Sunday and 6 is Saturday
    /// %%: A literal %
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
    ///     str: String representing time in RFC3339 format : "YYYY-MM-DDTHH:MM:SS.ssssssZ"
    ///
    /// Notes:
    ///     The fraction always has six digits (microseconds). Years outside
    ///     0000-9999 are written with a sign and at least four digits, as in
    ///     ISO 8601 expanded years (-0044-03-15T..., +10000-01-01T...).
    ///     The same string as ``to_iso8601``.
    fn to_rfc3339(&self) -> String {
        self.0.as_rfc3339()
    }

    /// Convert satkit.time object to ISO8601 string
    ///
    /// Returns:
    ///     str: String representing time in ISO8601 format : "YYYY-MM-DDTHH:MM:SS.ssssssZ"
    ///
    /// Notes:
    ///     The fraction always has six digits (microseconds). Years outside
    ///     0000-9999 are written with a sign and at least four digits, as in
    ///     ISO 8601 expanded years (-0044-03-15T..., +10000-01-01T...).
    ///     The same string as ``to_rfc3339``.
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

    /// Return time object representing the start (midnight UTC) of the input date
    ///
    /// Args:
    ///     year (int): Gregorian year (e.g., 2024)
    ///     month (int): Gregorian month (1 = January, 2 = February, ...)
    ///     day (int): Day of month, beginning with 1
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input date
    ///
    /// Raises:
    ///     TypeError: If an argument is not an integer
    ///     ValueError: If the month or day is out of range, or the date is
    ///         beyond about ±292,000 years
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
    ///
    /// Raises:
    ///     ValueError: If ``mjd`` is NaN or infinite, or ``scale`` is satkit.timescale.Invalid
    ///     OverflowError: If the result is beyond about ±292,000 years
    #[staticmethod]
    #[pyo3(signature=(mjd, scale=&PyTimeScale::UTC))]
    fn from_mjd(mjd: f64, scale: &PyTimeScale) -> PyResult<Self> {
        let scale = valid_scale(scale)?;
        finite_time_value(mjd, mjd * US_PER_DAY, "mjd")?;
        Ok(Self(Instant::from_mjd_with_scale(mjd, scale)))
    }

    /// Return time object representing input unix time, which is UTC seconds
    /// since Jan 1, 1970 00:00:00 (not counting leap seconds)
    ///
    /// Args:
    ///    unixtime (float): the unixtime, UTC seconds since 1970-01-01 00:00:00 (excluding leap seconds);
    ///        rounded to the nearest microsecond
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input unixtime
    ///
    /// Raises:
    ///     ValueError: If ``unixtime`` is NaN or infinite
    ///     OverflowError: If the result is beyond about ±292,000 years
    #[staticmethod]
    fn from_unixtime(unixtime: f64) -> PyResult<Self> {
        finite_time_value(unixtime, unixtime * 1.0e6, "unixtime")?;
        Ok(Self(Instant::from_unixtime(unixtime)))
    }

    /// Return time object representing input Julian date and time scale
    ///
    /// Args:
    ///    jd (float): The Julian Date, days
    ///   scale (satkit.timescale, optional): The time scale. Default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of julian date with given scale
    ///
    /// Raises:
    ///     ValueError: If ``jd`` is NaN or infinite, or ``scale`` is satkit.timescale.Invalid
    ///     OverflowError: If the result is beyond about ±292,000 years
    #[staticmethod]
    #[pyo3(signature=(jd, scale=&PyTimeScale::UTC))]
    fn from_jd(jd: f64, scale: &PyTimeScale) -> PyResult<Self> {
        let scale = valid_scale(scale)?;
        finite_time_value(jd, (jd - 2_400_000.5) * US_PER_DAY, "jd")?;
        Ok(Self(Instant::from_jd_with_scale(jd, scale)))
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
    ///     (int, int, int, int, int, float): Tuple with 6 elements representing Gregorian year, month, day, hour, minute, and second.
    ///     The second is the float nearest the exact (whole-microsecond) value,
    ///     in [0, 60), or [60, 61) inside a leap second
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
    ///     sec (float): floating point second of minute, in range [0,60); up
    ///         to 61 within a UTC leap second (23:59:60.5 on 2016-12-31) or a
    ///         pre-1972 positive UTC step. Rounded to the nearest microsecond
    ///
    /// Returns:
    ///    satkit.time: satkit.time object representing input Gregorian date and time
    ///
    /// Raises:
    ///     TypeError: If an argument is not a number (the first five must be integers)
    ///     ValueError: If a component is out of range (including a second of
    ///         60 or more where there is no leap second, and NaN), or the
    ///         date is beyond about ±292,000 years
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
    /// Raises:
    ///     OverflowError: If the time is outside the years 1 to 9999 that
    ///         datetime.datetime can represent
    ///
    /// Notes:
    ///     datetime.datetime has no leap seconds: a time inside one
    ///     (23:59:60.5) comes back as 23:59:59.5, as with Unix time.
    ///
    #[pyo3(signature = (utc=true))]
    fn to_datetime(&self, py: Python<'_>, utc: bool) -> PyResult<Py<PyAny>> {
        instant_to_datetime(py, &self.0, utc)
    }

    /// Deprecated: use ``to_datetime()``.
    ///
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
    ///
    /// Raises:
    ///     ValueError: If ``scale`` is satkit.timescale.Invalid
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_mjd(&self, scale: &PyTimeScale) -> PyResult<f64> {
        Ok(self.0.as_mjd_with_scale(valid_scale(scale)?))
    }

    /// Convert to Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional): Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Julian Date, days
    ///
    /// Raises:
    ///     ValueError: If ``scale`` is satkit.timescale.Invalid
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_jd(&self, scale: &PyTimeScale) -> PyResult<f64> {
        Ok(self.0.as_jd_with_scale(valid_scale(scale)?))
    }

    /// Convert to Unix time (seconds since 1970-01-01 00:00:00 UTC)
    /// Excludes leap seconds
    ///
    /// The result is the float nearest the exact (whole-microsecond) value.
    /// Inside a leap second it repeats the last second of the day.
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
        self.to_mjd(scale)
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_jd()``.
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn as_jd(&self, py: Python<'_>, scale: &PyTimeScale) -> PyResult<f64> {
        warn_deprecated(
            py,
            c"time.as_jd() is deprecated since 0.23 and will be removed in 0.25; use time.to_jd()",
        )?;
        self.to_jd(scale)
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
    ///     week (int): GPS week number, counted from the GPS epoch
    ///         (1980-01-06) without the 1024-week rollover
    ///     seconds (float): GPS seconds of week, seconds; rounded to the
    ///         nearest microsecond
    ///
    /// Returns:
    ///     satkit.time: Time object representing input GPS week and second
    ///
    /// Raises:
    ///     ValueError: If ``seconds`` is NaN or infinite
    ///     OverflowError: If the result is beyond about ±292,000 years
    #[staticmethod]
    fn from_gps_week_and_second(week: i32, seconds: f64) -> PyResult<Self> {
        // The week count is exact in f64 (|week| < 2^31, times 6.048e11)
        finite_time_value(
            seconds,
            (week as f64).mul_add(604_800.0e6, seconds * 1.0e6),
            "seconds",
        )?;
        Ok(Self(Instant::from_gps_week_and_second(week, seconds)))
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
    ///     other (duration|datetime.timedelta|list|numpy.ndarray|float): Duration or list of durations to add.
    ///         If type is float (or a list or real numeric 1-D numpy array of them), units are days
    ///         of exactly 86400 elapsed seconds (``add_utc_days`` keeps the UTC time of day
    ///         across a leap second instead). Sequences must be a list or a numpy array
    ///
    /// Returns:
    ///     satkit.time|numpy.ndarray: New time object or numpy array of time objects representing input time plus input duration(s)
    ///
    /// Raises:
    ///     ValueError: if a number of days is NaN or infinite
    ///     OverflowError: if a result is beyond about ±292,000 years
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.shift(other, |t, d| t.checked_add(d))
    }

    /// ``datetime.timedelta + time``: the reflected form exists for
    /// timedelta only (``duration + time`` is the duration's own ``+``)
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyDelta>() {
            self.shift(other, |t, d| t.checked_add(d))
        } else {
            Ok(other.py().NotImplemented())
        }
    }

    /// Subtract duration or take difference in times
    ///
    /// Args:
    ///     other (duration|datetime.timedelta|list|numpy.ndarray|float|satkit.time): Duration or list of durations to subtract, or time object to take difference.
    ///         If type is float, units are days
    ///
    /// Returns:
    ///     satkit.time|numpy.ndarray|satkit.duration: New time object or numpy array of time objects representing input time minus input duration(s), or duration object representing difference between two time objects
    ///     (a numpy array of duration objects for a list of time objects)
    ///
    /// Raises:
    ///     ValueError: if a number of days is NaN or infinite
    ///     OverflowError: if a result is beyond about ±292,000 years
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(tm2) = other.cast::<Self>() {
            return PyDuration(time_difference(&self.0, &tm2.borrow().0)?).into_py_any(py);
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
                        PyDuration(time_difference(&self.0, &t.0)?).into_py_any(py)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return np::PyArray1::<Py<PyAny>>::from_vec(py, objs).into_py_any(py);
            }
        }
        self.shift(other, |t, d| t.checked_sub(d))
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

    /// Add given number of UTC calendar days to a time object, and return the result
    ///
    /// The offset is applied to the UTC clock label (as with a Modified Julian
    /// Date in UTC), so the time of day is kept: noon plus one UTC day is noon
    /// of the next day, which is 86401 elapsed seconds later when a leap
    /// second lies between the two. ``days`` may be fractional or negative.
    /// To add exactly 86400 elapsed seconds per day, use ``t + days``.
    ///
    /// Args:
    ///     days (float): Number of UTC days to add; rounded to the nearest microsecond
    ///
    /// Returns:
    ///     satkit.time: Time object representing input time plus given number of UTC days
    ///
    /// Raises:
    ///     ValueError: If ``days`` is NaN or infinite
    ///     OverflowError: If the result is beyond about ±292,000 years
    fn add_utc_days(&self, days: f64) -> PyResult<Self> {
        // The shift itself, then the shifted UTC-basis count (exact enough
        // in f64 for a range check with 700 years of margin)
        finite_time_value(days, days * US_PER_DAY, "days")?;
        finite_time_value(
            days,
            days.mul_add(US_PER_DAY, self.0.as_unixtime_microseconds() as f64),
            "days",
        )?;
        Ok(Self(self.0.add_utc_days(days)))
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
    /// float array of either (element-wise, returning an object array).
    /// `op` is a checked operation: a result out of range is `OverflowError`
    fn shift(
        &self,
        other: &Bound<'_, PyAny>,
        op: fn(Instant, satkit::Duration) -> Option<Instant>,
    ) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let op = |t: Instant, d: satkit::Duration| {
            op(t, d).ok_or_else(|| crate::pyduration::arithmetic_overflow("time"))
        };
        let days = finite_days;
        let durs: Vec<satkit::Duration> = if other.is_instance_of::<np::PyUntypedArray>() {
            // Any real numeric 1-D array, as days: float32 and integer
            // arrays are converted (they used to raise TypeError)
            let arr = crate::pyutils::to_f64_ndarray(other)?;
            let ro = arr.readonly();
            let a = ro.as_array();
            if a.ndim() != 1 {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "expected a 1-D array of days, got {} dimensions",
                    a.ndim()
                )));
            }
            a.iter().map(|x| days(*x)).collect::<PyResult<_>>()?
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
            return Self(op(self.0, days(other.extract::<f64>()?)?)?).into_py_any(py);
        } else if let Ok(d) = other.cast::<PyDuration>() {
            return Self(op(self.0, d.borrow().0)?).into_py_any(py);
        } else if let Ok(td) = other.cast::<PyDelta>() {
            // datetime.timedelta: exact, it is a microsecond count too
            let d = crate::pyduration::timedelta_to_duration(td)?;
            return Self(op(self.0, d)?).into_py_any(py);
        } else {
            // Not a supported operand: Python raises its standard
            // `TypeError: unsupported operand type(s)` (after giving the
            // other operand's reflected method a chance)
            return Ok(py.NotImplemented());
        };
        let objs = durs
            .into_iter()
            .map(|d| Self(op(self.0, d)?).into_py_any(py))
            .collect::<PyResult<Vec<_>>>()?;
        np::PyArray1::<Py<PyAny>>::from_vec(py, objs).into_py_any(py)
    }
}

/// `a - b` as a duration, `OverflowError` if it does not fit (only for
/// times near the ends of the range)
fn time_difference(a: &Instant, b: &Instant) -> PyResult<satkit::Duration> {
    a.checked_duration_since(*b)
        .ok_or_else(|| crate::pyduration::arithmetic_overflow("time difference"))
}

/// A time-string parse failure as a Python `ValueError` carrying the
/// parser's message (`from_string`, `strptime` and the `time(str)`
/// constructor; `from_rfc3339` adds its own prefix)
fn parse_error(e: satkit::InstantError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

const US_PER_DAY: f64 = 86_400.0e6;

/// The core time scale for a `scale` argument; `timescale.Invalid` is
/// `ValueError`. The core maps it to `Instant::INVALID` (which prints as the
/// year −290308) on the way in and to NaN on the way out.
fn valid_scale(scale: &PyTimeScale) -> PyResult<TimeScale> {
    match scale {
        PyTimeScale::Invalid => Err(pyo3::exceptions::PyValueError::new_err(
            "satkit.timescale.Invalid is not a time scale",
        )),
        s => Ok(s.into()),
    }
}

/// Check a float that is about to become a time: `value` is the argument as
/// given (named `what`) and `us` the microsecond count it leads to. NaN and
/// infinities are `ValueError`; a count beyond about ±292,000 years is
/// `OverflowError`. The core constructors map NaN to their epoch and
/// saturate the rest, so `time.from_mjd(nan)` was 1858-11-17 and
/// `time.from_unixtime(inf)` the year 294247.
fn finite_time_value(value: f64, us: f64, what: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be finite, got {value}"
        )));
    }
    // 2^63 is 9.223e18; the margin (about 700 years) covers the epoch and
    // TAI − UTC offsets the constructors add to the count
    if us.abs() > 9.2e18 {
        return Err(crate::pyduration::arithmetic_overflow(&format!(
            "time for {what} = {value}"
        )));
    }
    Ok(())
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

/// The `numpy.datetime64` scalar type, imported once
fn datetime64_type(py: Python<'_>) -> PyResult<&Bound<'_, pyo3::types::PyType>> {
    static DATETIME64: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    DATETIME64.import(py, "numpy", "datetime64")
}

/// Whether `obj` is a single time: `satkit.time`, `datetime.datetime` or a
/// `numpy.datetime64` scalar (the stubs' `TimeScalar`)
pub(crate) fn is_time_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyInstant>()
        || obj.is_instance_of::<PyDateTime>()
        || datetime64_type(obj.py()).is_ok_and(|t| obj.is_instance(t).unwrap_or(false))
}

/// A single time argument: `satkit.time`, `datetime.datetime` or
/// `numpy.datetime64` (the stubs' `TimeScalar`). Use as a `#[pyfunction]`
/// parameter type in place of `PyInstant` wherever the stub accepts either.
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
        if obj.is_instance(datetime64_type(obj.py())?)? {
            // A 1-element array of the scalar's own unit
            let arr = obj.call_method1("reshape", (1,))?;
            let arr = arr.cast::<numpy::PyUntypedArray>()?;
            return Ok(Self(datetime64_array_to_vec(arr, true)?[0]));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected satkit.time, datetime.datetime or numpy.datetime64, got {}",
            obj.get_type()
        )))
    }
}

/// How a `datetime64` count converts to UTC microseconds since 1970
#[derive(Clone, Copy)]
enum Datetime64Unit {
    /// Calendar months (units `Y` and `M`): this many per count
    Months(i128),
    /// Fixed-length units: microseconds = count × `num` / `den`, rounded
    Fixed { num: i128, den: i128 },
}

/// The unit of a `datetime64` dtype, from `numpy.datetime_data` (which also
/// gives the multiplier of a unit such as `datetime64[10ns]`)
fn datetime64_unit(dtype: &Bound<'_, numpy::PyArrayDescr>) -> PyResult<Datetime64Unit> {
    static DATETIME_DATA: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    let py = dtype.py();
    let datetime_data = DATETIME_DATA.get_or_try_init(py, || {
        py.import("numpy")?
            .getattr("datetime_data")
            .map(Bound::unbind)
    })?;
    let (unit, count): (String, i64) = datetime_data.bind(py).call1((dtype,))?.extract()?;
    // |count| < 2^63, so none of these products overflows i128
    let count = count as i128;
    let fixed = |us: i128| Datetime64Unit::Fixed {
        num: us * count,
        den: 1,
    };
    let sub_us = |den: i128| Datetime64Unit::Fixed { num: count, den };
    Ok(match unit.as_str() {
        "Y" => Datetime64Unit::Months(12 * count),
        "M" => Datetime64Unit::Months(count),
        "W" => fixed(7 * 86_400_000_000),
        "D" => fixed(86_400_000_000),
        "h" => fixed(3_600_000_000),
        "m" => fixed(60_000_000),
        "s" => fixed(1_000_000),
        "ms" => fixed(1_000),
        "us" => fixed(1),
        "ns" => sub_us(1_000),
        "ps" => sub_us(1_000_000),
        "fs" => sub_us(1_000_000_000),
        "as" => sub_us(1_000_000_000_000),
        // A generic-unit array can only hold NaT, which is refused below
        "generic" => fixed(0),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported numpy.datetime64 unit: {other}"
            )))
        }
    })
}

/// Largest |UTC microseconds| accepted from a `datetime64`: the `Instant`
/// range less a day, so adding TAI − UTC cannot saturate (about ±292,000
/// years)
const DATETIME64_MAX_US: i128 = i64::MAX as i128 - 86_400_000_000;

/// Why a `datetime64` element was refused
enum Datetime64Error {
    NaT(usize),
    OutOfRange(usize),
}

/// Convert raw `datetime64` counts, as UTC labels, to instants: exact for
/// whole microseconds, finer units rounded to the nearest microsecond
/// (halves to the later one, as the seconds of a calendar time or a time
/// string round), calendar units through the Gregorian calendar. Pure Rust,
/// so it runs without the GIL.
fn datetime64_counts_to_instants(
    counts: &[i64],
    unit: Datetime64Unit,
) -> std::result::Result<Vec<Instant>, Datetime64Error> {
    counts
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            // numpy's NaT is the most negative int64
            if v == i64::MIN {
                return Err(Datetime64Error::NaT(i));
            }
            let v = v as i128;
            match unit {
                Datetime64Unit::Months(per) => {
                    let months = v.checked_mul(per).ok_or(Datetime64Error::OutOfRange(i))?;
                    let year = i32::try_from(1970 + months.div_euclid(12))
                        .map_err(|_| Datetime64Error::OutOfRange(i))?;
                    let month = months.rem_euclid(12) as i32 + 1;
                    Instant::from_date(year, month, 1).map_err(|_| Datetime64Error::OutOfRange(i))
                }
                Datetime64Unit::Fixed { num, den } => {
                    // Checked: a unit multiplier (`datetime64[1000000W]`)
                    // can make even i128 overflow
                    let x = v.checked_mul(num).ok_or(Datetime64Error::OutOfRange(i))?;
                    // floor(x / den + 1/2): halves go to the later
                    // microsecond, before 1970 too
                    let us = x
                        .checked_mul(2)
                        .and_then(|x2| x2.checked_add(den))
                        .ok_or(Datetime64Error::OutOfRange(i))?
                        .div_euclid(2 * den);
                    if us.abs() > DATETIME64_MAX_US {
                        return Err(Datetime64Error::OutOfRange(i));
                    }
                    Ok(Instant::from_unixtime_microseconds(us as i64))
                }
            }
        })
        .collect()
}

/// Times from a 1-D numpy `datetime64` array of any unit, read as UTC labels
/// (numpy's datetime64 has no time zone and no leap seconds). The int64
/// counts are read through a view, not element by element; a non-native
/// byte order is first converted by numpy. `scalar` only changes the
/// wording of errors.
///
/// Raises `ValueError` for `NaT` and `OverflowError` for a time beyond
/// satkit's range.
fn datetime64_array_to_vec(
    arr: &Bound<'_, numpy::PyUntypedArray>,
    scalar: bool,
) -> PyResult<Vec<Instant>> {
    use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
    let py = arr.py();
    if arr.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "numpy.datetime64 time arrays must be 1-D, got {} dimensions",
            arr.ndim()
        )));
    }
    let dtype = arr.dtype();
    let unit = datetime64_unit(&dtype)?;
    // A byte-swapped array is converted to native order (a copy; rare)
    let native = if dtype.is_native_byteorder() == Some(false) {
        arr.call_method1("astype", (dtype.call_method1("newbyteorder", ("=",))?,))?
    } else {
        arr.clone().into_any()
    };
    let ints = native.call_method1("view", ("i8",))?;
    let ints = ints.extract::<numpy::PyReadonlyArray1<i64>>()?;
    // Copied out (strided or not) so the conversion can release the GIL
    // without another thread changing the array underneath it
    let counts = ints.as_array().to_vec();
    drop(ints);
    py.detach(|| datetime64_counts_to_instants(&counts, unit))
        .map_err(|e| {
            let what = |i: usize| {
                if scalar {
                    "numpy.datetime64 value".to_string()
                } else {
                    format!("numpy.datetime64 value at index {i}")
                }
            };
            match e {
                Datetime64Error::NaT(i) => pyo3::exceptions::PyValueError::new_err(format!(
                    "{} is NaT (not a time)",
                    what(i)
                )),
                Datetime64Error::OutOfRange(i) => crate::pyduration::arithmetic_overflow(&what(i)),
            }
        })
}

/// A numpy array of dtype `datetime64` (of any shape), if `obj` is one
fn as_datetime64_array<'a, 'py>(
    obj: &'a Bound<'py, PyAny>,
) -> Option<&'a Bound<'py, numpy::PyUntypedArray>> {
    use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
    obj.cast::<numpy::PyUntypedArray>()
        .ok()
        .filter(|a| a.dtype().kind() == b'M')
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
        if is_time_scalar(self) {
            let t: TimeArg = self.extract()?;
            return Ok(TimeInput {
                times: vec![t.0],
                scalar: true,
            });
        }
        // numpy datetime64 array; a 0-d one is a scalar
        if let Some(arr) = as_datetime64_array(self) {
            use numpy::PyUntypedArrayMethods;
            if arr.ndim() == 0 {
                let arr = self.call_method1("reshape", (1,))?;
                return Ok(TimeInput {
                    times: datetime64_array_to_vec(arr.cast()?, true)?,
                    scalar: true,
                });
            }
            return Ok(TimeInput {
                times: datetime64_array_to_vec(arr, false)?,
                scalar: false,
            });
        }
        Ok(TimeInput {
            times: time_array_to_vec(self)?,
            scalar: false,
        })
    }
}

/// Times from a list or 1-D numpy object array of `satkit.time` /
/// `datetime.datetime` / `numpy.datetime64` (a `datetime64` array is read by
/// `datetime64_array_to_vec`)
fn time_array_to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Instant>> {
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| item.extract::<TimeArg>().map(|t| t.0))
            .collect::<PyResult<Vec<_>>>()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Not a list of satkit.time, datetime.datetime or numpy.datetime64: {e}"
                ))
            })
    } else if obj.is_instance_of::<numpy::PyArray1<Py<PyAny>>>() {
        let v = obj
            .extract::<numpy::PyReadonlyArray1<Py<PyAny>>>()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Invalid satkit.time, datetime.datetime or numpy.datetime64 input: {e}"
                ))
            })?;
        let py = obj.py();
        v.as_array()
            .iter()
            .map(|p| p.bind(py).extract::<TimeArg>().map(|t| t.0))
            .collect::<PyResult<Vec<_>>>()
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Invalid satkit.time input: numpy array must have dtype datetime64, \
                     or contain satkit.time, datetime.datetime or numpy.datetime64 elements",
                )
            })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Invalid satkit.time, datetime.datetime or numpy.datetime64 input",
        ))
    }
}
