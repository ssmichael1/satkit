use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDateTime;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::types::PyTzInfo;
use pyo3::IntoPyObjectExt;

use crate::{Instant, TimeScale, Weekday};

use super::pyduration::PyDuration;

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
#[pyclass(name = "timescale", module = "satkit", eq, eq_int)]
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

#[derive(Clone, PartialEq, Eq)]
#[pyclass(name = "weekday", module = "satkit", eq, eq_int)]
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
impl<'py> IntoPyObject<'py> for crate::Weekday {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Self::Sunday => PyWeekday::Sunday,
            Self::Monday => PyWeekday::Monday,
            Self::Tuesday => PyWeekday::Tuesday,
            Self::Wednesday => PyWeekday::Wednesday,
            Self::Thursday => PyWeekday::Thursday,
            Self::Friday => PyWeekday::Friday,
            Self::Saturday => PyWeekday::Saturday,
            Self::Invalid => PyWeekday::Invalid,
        }
        .into_bound_py_any(py)
        .unwrap())
    }
}

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
#[pyclass(name = "time", module = "satkit")]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone, Debug)]
pub struct PyInstant(pub Instant);

impl<'py> IntoPyObject<'py> for crate::Instant {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyInstant(self).into_bound_py_any(py).unwrap())
    }
}

impl<'py> FromPyObject<'py> for Instant {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let obj = PyInstant::extract_bound(ob)?;
        Ok(obj.0)
    }
}

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
    /// Returns:
    ///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
    #[new]
    #[pyo3(signature=(*py_args))]
    fn py_new(py_args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        if py_args.is_empty() {
            Ok(Self(Instant::now()))
        } else if py_args.len() == 3 {
            let year = py_args.get_item(0)?.extract::<i32>()?;
            let month = py_args.get_item(1)?.extract::<i32>()?;
            let day = py_args.get_item(2)?.extract::<i32>()?;
            Self::from_date(year, month, day)
        } else if py_args.len() == 6 {
            let year = py_args.get_item(0)?.extract::<i32>()?;
            let month = py_args.get_item(1)?.extract::<i32>()?;
            let day = py_args.get_item(2)?.extract::<i32>()?;
            let hour = py_args.get_item(3)?.extract::<i32>()?;
            let min = py_args.get_item(4)?.extract::<i32>()?;
            let sec = py_args.get_item(5)?.extract::<f64>()?;

            Ok(Self(Instant::from_datetime(
                year, month, day, hour, min, sec,
            )))
        } else if py_args.len() == 1 {
            let item = py_args.get_item(0)?;
            let s = item.extract::<&str>()?;

            // Input is a string, first try rfc3339 format
            match Instant::from_rfc3339(s) {
                Ok(v) => return Ok(Self(v)),
                Err(_) => {
                    // Now try multiple formats
                    return Self::from_string(s);
                }
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Must pass in year, month, day or year, month, day, hour, min, sec",
            ))
        }
    }

    /// Create satkit.time object from string
    ///
    /// Args:
    ///    s (str): String representing time
    ///
    /// Returns:
    ///   satkit.time: Time object representing input time
    ///
    /// Raises:
    ///   ValueError: If input string cannot be parsed
    ///
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        Instant::from_string(s).map_or_else(
            |_| {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Could not parse time string",
                ))
            },
            |v| Ok(Self(v)),
        )
    }

    /// Create satkit.time object from string with given format
    ///
    /// Args:
    ///   s (str): String representing time
    ///  fmt (str): Format string
    ///
    /// Returns:
    ///  satkit.time: Time object representing input time
    ///
    /// Raises:
    ///  ValueError: If input string cannot be parsed
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
    /// %b: Month as locale’s abbreviated name
    /// %B: Month as locale’s full name
    #[staticmethod]
    fn strptime(s: &str, fmt: &str) -> PyResult<Self> {
        Instant::strptime(s, fmt).map_or_else(
            |_| {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Could not parse time string",
                ))
            },
            |v| Ok(Self(v)),
        )
    }

    /// Format time object as string
    ///
    /// Args:
    ///  fmt (str): Format string
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
    fn strftime(&self, fmt: &str) -> PyResult<String> {
        self.0.strftime(fmt).map_or_else(
            |_| {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Could not format time string",
                ))
            },
            Ok,
        )
    }

    /// Create satkit.time object from RFC3339 string
    ///
    /// Notes:
    ///   RFC3339 is a standard for representing time in a string format
    ///   See: https://tools.ietf.org/html/rfc3339
    ///   This overlaps with ISO 8601
    ///
    /// Args:
    ///   s (str): String representing time
    ///
    /// Returns:
    ///   satkit.time: Time object representing input time
    ///
    /// Raises:
    ///   ValueError: If input string cannot be parsed
    ///
    #[staticmethod]
    fn from_rfc3339(s: &str) -> PyResult<Self> {
        Instant::from_rfc3339(s).map_or_else(
            |_| {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Could not parse time string",
                ))
            },
            |v| Ok(Self(v)),
        )
    }

    /// Convert satkit.time object to RFC3339 string
    ///
    /// Returns:
    /// str: String representing time in RFC3339 format : "YYYY-MM-DDTHH:MM:SS.sssZ"
    ///
    /// Notes:
    ///  RFC3339 is a standard for representing time in a string format
    ///  Return string also matches ISO8601
    fn as_rfc3339(&self) -> String {
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
    fn as_iso8601(&self) -> String {
        self.0.as_iso8601()
    }

    /// Return current time
    ///
    /// Returns:
    ///     satkit.time: Time object representing current time
    #[staticmethod]
    fn now() -> PyResult<Self> {
        Ok(Self(Instant::now()))
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
    fn from_date(year: i32, month: i32, day: i32) -> PyResult<Self> {
        Ok(Self(Instant::from_date(year, month, day)))
    }

    /// Return time object representing input modified Julian date and time scale
    ///
    /// Args:
    ///   mjd (float): The Modified Julian Date
    ///     scale (satkit.timescale): The time scale
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of modified julian date with given scale    
    #[staticmethod]
    fn from_mjd(mjd: f64, scale: &PyTimeScale) -> Self {
        Self(Instant::from_mjd_with_scale(mjd, scale.into()))
    }

    /// Return time object representing input unix time, which is UTC seconds
    /// since Jan 1, 1970 00:00:00 (not counting leap seconds)
    ///
    /// Args:
    ///    unixtime (float): the unixtime
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input unixtime
    #[staticmethod]
    fn from_unixtime(t: f64) -> Self {
        Self(Instant::from_unixtime(t))
    }

    /// Return time object representing input Julian date and time scale
    ///
    /// Args:
    ///    jd (float): The Julian Date
    ///   scale (satkit.timescale, optional): The time scale. Default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of julian date with given scale
    #[staticmethod]
    fn from_jd(jd: f64, scale: &PyTimeScale) -> Self {
        Self(Instant::from_jd_with_scale(jd, scale.into()))
    }

    /// Convert time object to UTC Gegorian date
    ///
    /// Returns:
    ///    (int, int, int): Tuple with 3 elements representing Gregorian year, month, and day
    fn as_date(&self) -> (i32, i32, i32) {
        let dt = self.0.as_datetime();
        (dt.0, dt.1, dt.2)
    }

    /// Convert time object to UTC Gegorian date and time, with fractional seconds
    ///
    /// Returns:
    ///     (int, int, int, int, int, float): Tuple with 6 elements representing Gregorian year, month, day, hour, minute, and second
    ///
    fn as_gregorian(&self) -> (i32, i32, i32, i32, i32, f64) {
        self.0.as_datetime()
    }

    /// Create satkit.time representing input UTC Gegorian date and time
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
    ) -> PyResult<Self> {
        Ok(Self(Instant::from_datetime(
            year, month, day, hour, min, sec,
        )))
    }

    /// Convert from Python datetime object
    ///
    /// Args:
    ///     datetime (datetime.datetime): datetime object to convert
    ///
    /// Returns:
    ///     satkit.time: satkit.time object that matches input datetime
    /// SatKit Time object representing input datetime
    #[staticmethod]
    fn from_datetime(tm: &Bound<'_, PyDateTime>) -> PyResult<Self> {
        let ts: f64 = tm
            .call_method("timestamp", (), None)
            .unwrap()
            .extract::<f64>()
            .unwrap();
        Ok(Self(Instant::from_unixtime(ts)))
    }

    /// Convert to Python datetime object
    ///
    /// Args:
    ///     utc (bool, optional): Use UTC as timezone; if not passed in, defaults to true
    ///
    /// Returns:
    ///     datetime.datetime:  datetime object matching the input satkit.time
    ///
    #[pyo3(signature = (utc=true))]
    fn datetime(&self, utc: bool) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let timestamp: f64 = self.as_unixtime();
            let tz = match utc {
                false => None,
                true => Some(PyTzInfo::utc(py)),
            };
            let tz = tz.as_ref().map(|r| r.as_ref().unwrap());
            Ok(PyDateTime::from_timestamp(py, timestamp, tz.map(|v| &**v))?.into())
        })
    }

    /// Convert to Modified Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional): Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Modified Julian Date
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn as_mjd(&self, scale: &PyTimeScale) -> f64 {
        self.0.as_mjd_with_scale(scale.into())
    }

    /// Convert to Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional: Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Julian Date
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn as_jd(&self, scale: &PyTimeScale) -> f64 {
        self.0.as_jd_with_scale(scale.into())
    }

    /// Convert to Unix time (seconds since 1970-01-01 00:00:00 UTC)
    /// Excludes leap seconds
    ///
    /// Returns:
    ///     float: Unix time (seconds since 1970-01-01 00:00:00 UTC)
    fn as_unixtime(&self) -> f64 {
        self.0.as_unixtime()
    }

    #[staticmethod]
    fn from_gps_week_and_second(week: i32, seconds: f64) -> Self {
        Self(Instant::from_gps_week_and_second(week, seconds))
    }

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
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Numpy array of floats
        if other.is_instance_of::<np::PyArray1<f64>>() {
            let parr = other.extract::<np::PyReadonlyArray1<f64>>()?;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let objarr = parr
                    .as_array()
                    .map(|x| {
                        Self(self.0 + crate::Duration::from_days(*x))
                            .into_py_any(py)
                            .unwrap()
                    })
                    .into_iter();
                let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
                parr.into_py_any(py)
            })
        }
        // list of floats or duration
        else if other.is_instance_of::<pyo3::types::PyList>() {
            other.extract::<Vec<f64>>().map_or_else(
                |_| {
                    other.extract::<Vec<PyDuration>>().map_or_else(
                        |_| {
                            Err(pyo3::exceptions::PyTypeError::new_err(
                                "Invalid types in list",
                            ))
                        },
                        |v| {
                            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                                let objarr = v.into_iter().map(|x| {
                                    let pyobj = Self(self.0 + x.0);
                                    pyobj.into_py_any(py).unwrap()
                                });

                                let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
                                parr.into_py_any(py)
                            })
                        },
                    )
                },
                |v| {
                    pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                        let objarr = v.iter().map(|x| {
                            let pyobj = Self(self.0 + crate::Duration::from_days(*x));
                            pyobj.into_py_any(py).unwrap()
                        });
                        let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
                        parr.into_py_any(py)
                    })
                },
            )
        }
        // Constant number
        else if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
        {
            let dt: f64 = other.extract::<f64>().unwrap();
            Self(self.0 + crate::Duration::from_days(dt)).into_py_any(other.py())
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Self(self.0 + dur.0).into_py_any(other.py())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid type for rhs",
            ))
        }
    }

    /// Subtract duration or take difference in times
    ///
    /// Args:
    ///     other (duration|list|numpy.ndarray|float|satkit.time): Duration or list of durations to subtract, or time object to take difference
    ///
    /// Returns:
    ///     satkit.time|numpy.ndarray|satkit.duration: New time object or numpy array of time objects representing input time minus input duration(s), or duration object representing difference between two time objects
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Numpy array of floats
        if other.is_instance_of::<np::PyArray1<f64>>() {
            let parr: np::PyReadonlyArray1<f64> = other.extract().unwrap();
            let objarr = parr.as_array().into_iter().map(|x| -> PyObject {
                let obj = Self(self.0 - crate::Duration::from_days(*x));
                obj.into_py_any(other.py()).unwrap()
            });
            let parr = np::PyArray1::<PyObject>::from_iter(other.py(), objarr);
            parr.into_py_any(other.py())
        }
        // list of floats
        else if other.is_instance_of::<pyo3::types::PyList>() {
            other.extract::<Vec<f64>>().map_or_else(
                |_| {
                    other.extract::<Vec<PyDuration>>().map_or_else(
                        |_| {
                            Err(pyo3::exceptions::PyTypeError::new_err(
                                "Invalid types in list",
                            ))
                        },
                        |v| {
                            let objarr = v.into_iter().map(|x| {
                                let pyobj = Self(self.0 - x.0);
                                pyobj.into_py_any(other.py()).unwrap()
                            });

                            let parr = np::PyArray1::<PyObject>::from_iter(other.py(), objarr);
                            parr.into_py_any(other.py())
                        },
                    )
                },
                |v| {
                    let objarr = v.into_iter().map(|x| {
                        let pyobj = Self(self.0 - crate::Duration::from_days(x));
                        pyobj.into_py_any(other.py()).unwrap()
                    });
                    let parr = np::PyArray1::<PyObject>::from_iter(other.py(), objarr);
                    parr.into_py_any(other.py())
                },
            )
        }
        // Constant number
        else if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
        {
            let dt: f64 = other.extract::<f64>().unwrap();
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Self(self.0 - crate::Duration::from_days(dt)).into_py_any(py)
            })
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Self(self.0 - dur.0).into_py_any(other.py())
        } else if other.is_instance_of::<Self>() {
            let tm2 = other.extract::<Self>().unwrap();
            PyDuration(self.0 - tm2.0).into_py_any(other.py())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid type for rhs",
            ))
        }
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

    fn __ne__(&self, other: &Self) -> bool {
        self.0 != other.0
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
        d.set_item("empty", true).unwrap();
        (PyTuple::empty(py), d)
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

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &i64::to_le_bytes(self.0.raw)).into())
    }
}

impl<'b> From<&'b PyInstant> for &'b Instant {
    fn from(s: &PyInstant) -> &Instant {
        &s.0
    }
}

fn datetime_to_instant(tm: &Bound<PyDateTime>) -> PyResult<Instant> {
    let ts: f64 = tm
        .call_method("timestamp", (), None)
        .unwrap()
        .extract::<f64>()
        .unwrap();
    Ok(Instant::from_unixtime(ts))
}

pub trait ToTimeVec {
    fn to_time_vec(&self) -> PyResult<Vec<Instant>>;
}

impl ToTimeVec for &Bound<'_, PyAny> {
    fn to_time_vec(&self) -> PyResult<Vec<Instant>> {
        // "Scalar" time input case
        if self.is_instance_of::<PyInstant>() {
            let tm: PyInstant = self.extract().unwrap();
            Ok(vec![tm.0])
        } else if self.is_instance_of::<PyDateTime>() {
            let dt: Py<PyDateTime> = self.extract().unwrap();
            pyo3::Python::with_gil(|py| Ok(vec![datetime_to_instant(dt.bind(py)).unwrap()]))
        }
        // List case
        else if self.is_instance_of::<pyo3::types::PyList>() {
            match self.extract::<Vec<PyInstant>>() {
                Ok(v) => Ok(v.iter().map(|x| x.0).collect::<Vec<_>>()),
                Err(_e) => match self.extract::<Vec<Py<PyDateTime>>>() {
                    Ok(v) => pyo3::Python::with_gil(|py| {
                        Ok(v.iter()
                            .map(|x| datetime_to_instant(x.bind(py)).unwrap())
                            .collect::<Vec<_>>())
                    }),
                    Err(e) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Not a list of satkit.time or datetime.datetime: {e}"
                    ))),
                },
            }
        }
        // numpy array case
        else if self.is_instance_of::<numpy::PyArray1<PyObject>>() {
            match self.extract::<numpy::PyReadonlyArray1<PyObject>>() {
                Ok(v) => pyo3::Python::with_gil(|py| -> PyResult<Vec<Instant>> {
                    // Extract times from numpya array of objects
                    let tmarray: Result<Vec<Instant>, _> = v
                        .as_array()
                        .into_iter()
                        .map(|p| -> Result<Instant, _> {
                            p.extract::<PyInstant>(py).map_or_else(|_| p.extract::<Py<PyDateTime>>(py).map_or_else(|_| Err(pyo3::exceptions::PyTypeError::new_err(
                                        "Input numpy array must contain satkit.time elements or datetime.datetime elements".to_string()
                                    )), |v3| pyo3::Python::with_gil(|py| {
                                        Ok(datetime_to_instant(v3.bind(py)).unwrap())
                                    })), |v2| Ok(v2.0))
                        })
                        .collect();

                    tmarray.map_or_else(
                        |_| {
                            Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "Invalid satkit.time input",
                            ))
                        },
                        Ok,
                    )
                }),

                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Invalid satkit.time or datetime.datetime input: {e}"
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Invalid satkit.time or datetime.datetime input",
            ))
        }
    }
}
