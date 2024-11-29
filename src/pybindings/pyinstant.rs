use pyo3::prelude::*;
use pyo3::types::timezone_utc_bound;
use pyo3::types::PyBytes;
use pyo3::types::PyDateTime;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;

use crate::{Instant, TimeScale};

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

impl From<&PyTimeScale> for TimeScale {
    fn from(s: &PyTimeScale) -> TimeScale {
        match s {
            PyTimeScale::Invalid => TimeScale::Invalid,
            PyTimeScale::UTC => TimeScale::UTC,
            PyTimeScale::TT => TimeScale::TT,
            PyTimeScale::UT1 => TimeScale::UT1,
            PyTimeScale::TAI => TimeScale::TAI,
            PyTimeScale::GPS => TimeScale::GPS,
            PyTimeScale::TDB => TimeScale::TDB,
        }
    }
}

impl From<PyTimeScale> for TimeScale {
    fn from(s: PyTimeScale) -> TimeScale {
        match s {
            PyTimeScale::Invalid => TimeScale::Invalid,
            PyTimeScale::UTC => TimeScale::UTC,
            PyTimeScale::TT => TimeScale::TT,
            PyTimeScale::UT1 => TimeScale::UT1,
            PyTimeScale::TAI => TimeScale::TAI,
            PyTimeScale::GPS => TimeScale::GPS,
            PyTimeScale::TDB => TimeScale::TDB,
        }
    }
}

impl IntoPy<PyObject> for TimeScale {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ts: PyTimeScale = match self {
            TimeScale::Invalid => PyTimeScale::Invalid,
            TimeScale::UTC => PyTimeScale::UTC,
            TimeScale::TT => PyTimeScale::TT,
            TimeScale::UT1 => PyTimeScale::UT1,
            TimeScale::TAI => PyTimeScale::TAI,
            TimeScale::GPS => PyTimeScale::GPS,
            TimeScale::TDB => PyTimeScale::TDB,
        };
        ts.into_py(py)
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
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug)]
pub struct PyInstant {
    pub inner: Instant,
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
    fn py_new(
        py_args: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
                

        if py_args.is_empty() {
            Ok(PyInstant { inner: Instant::now() })       
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
         
            Ok(PyInstant { inner: Instant::from_datetime(year, month, day, hour, min, sec) })
        } else if py_args.len() == 1 {
            let item = py_args.get_item(0)?;
            let s = item.extract::<&str>()?;
            
            // Input is a string, first try rfc3339 format
            match Instant::from_rfc3339(s) {
                Ok(v) => return Ok(PyInstant { inner: v }),
                Err(_) => {
                    // Now try multiple formats
                    return Self::from_string(s);
                }
            }            
        }
        else {
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
        match Instant::from_string(s) {
            Ok(v) => Ok(PyInstant { inner: v }),
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "Could not parse time string",
            )),
        }
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
    /// Notes:
    ///    See: https://docs.rs/chrono/latest/chrono/format/strftime/index.html
    ///    for format string options
    #[staticmethod]
    fn strptime(s: &str, fmt: &str) -> PyResult<Self> {
        match Instant::strptime(fmt, s) {
            Ok(v) => Ok(PyInstant { inner: v }),
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "Could not parse time string",
            )),
        }
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
    fn from_rfctime(s: &str) -> PyResult<Self> {
        match Instant::from_rfc3339(s) {
            Ok(v) => Ok(PyInstant { inner: v }),
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "Could not parse time string",
            )),
        }
    }

    /// Return current time
    ///
    /// Returns:
    ///     satkit.time: Time object representing current time
    #[staticmethod]
    fn now() -> PyResult<Self> {
        Ok(PyInstant { inner: Instant::now() })
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
        Ok(PyInstant {
            inner: Instant::from_date(year, month, day),
        })
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
        PyInstant {
            inner: Instant::from_mjd_with_scale(mjd, scale.into()),
        }
    }

    /// Return time object representing input unix time, which is UTC seconds
    /// since Jan 1, 1970 00:00:00
    ///
    /// Args:
    ///    unixtime (float): the unixtime
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of input unixtime
    #[staticmethod]
    fn from_unixtime(t: f64) -> Self {
        PyInstant {
            inner: Instant::from_unixtime(t),
        }
    }

    /// Return time object representing input Julian date and time scale
    ///
    /// Args:
    ///    jd (float): The Julian Date
    ///   scale (satkit.timescale): The time scale
    ///
    /// Returns:
    ///     satkit.time: Time object representing instant of julian date with given scale
    #[staticmethod]
    fn from_jd(jd: f64, scale: &PyTimeScale) -> Self {
        PyInstant {
            inner: Instant::from_jd_with_scale(jd, scale.into()),
        }
    }

    /// Convert time object to UTC Gegorian date
    ///
    /// Returns:
    ///    (int, int, int): Tuple with 3 elements representing Gregorian year, month, and day
    fn as_date(&self) -> (i32, i32, i32) {
        let dt = self.inner.as_datetime();
        (dt.0, dt.1, dt.2)
    }

    /// Convert time object to UTC Gegorian date and time, with fractional seconds
    ///
    /// Returns:
    ///     (int, int, int, int, int, float): Tuple with 6 elements representing Gregorian year, month, day, hour, minute, and second
    ///
    fn as_gregorian(&self) -> (i32, i32, i32, i32, i32, f64) {
        self.inner.as_datetime()
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
        Ok(PyInstant {
            inner: Instant::from_datetime(
                year,
                month,
                day,
                hour,
                min,
                sec,
            ),
        })
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
        Ok(PyInstant {
            inner: Instant::from_unixtime(ts),
        })
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
                true => Some(timezone_utc_bound(py)),
            };
            Ok(PyDateTime::from_timestamp_bound(py, timestamp, tz.as_ref())?.into_py(py))
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
        self.inner.as_mjd_with_scale(scale.into())
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
        self.inner.as_jd_with_scale(scale.into())
    }

    /// Convert to Unix time (seconds since 1970-01-01 00:00:00 UTC)
    ///
    /// Returns:
    ///     float: Unix time (seconds since 1970-01-01 00:00:00 UTC)
    fn as_unixtime(&self) -> f64 {
        self.inner.as_unixtime()
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
                        let obj = PyInstant {
                            inner: self.inner + crate::Duration::from_days(*x),
                        };
                        obj.into_py(py)
                    })
                    .into_iter();
                let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                Ok(parr.into_py(py))
            })
        }
        // list of floats or duration
        else if other.is_instance_of::<pyo3::types::PyList>() {
            if let Ok(v) = other.extract::<Vec<f64>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .iter()
                        .map(|x| {
                            let pyobj = PyInstant {
                                inner: self.inner + crate::Duration::from_days(*x),
                            };
                            pyobj.into_py(py)
                        });
                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else if let Ok(v) = other.extract::<Vec<PyDuration>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyInstant {
                                inner: self.inner + x.inner,
                            };
                            pyobj.into_py(py)
                        });

                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid types in list",
                ))
            }
        }
        // Constant number
        else if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
            || other.is_instance_of::<pyo3::types::PyLong>()
        {
            let dt: f64 = other.extract::<f64>().unwrap();
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(PyInstant {
                    inner: self.inner + crate::Duration::from_days(dt),
                }
                .into_py(py))
            })
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Ok(PyInstant {
                inner: self.inner + dur.inner,
            }
            .into_py(other.py()))
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
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let objarr = parr
                    .as_array()
                    .into_iter()
                    .map(|x| {
                        let obj = PyInstant {
                            inner: self.inner - crate::Duration::from_days(*x),
                        };
                        obj.into_py(py)
                    });                let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                Ok(parr.into_py(py))
            })
        }
        // list of floats
        else if other.is_instance_of::<pyo3::types::PyList>() {
            if let Ok(v) = other.extract::<Vec<f64>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyInstant {
                                inner: self.inner - crate::Duration::from_days(x),
                            };
                            pyobj.into_py(py)
                        });
                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else if let Ok(v) = other.extract::<Vec<PyDuration>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyInstant {
                                inner: self.inner - x.inner,
                            };
                            pyobj.into_py(py)
                        });
                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid types in list",
                ))
            }
        }
        // Constant number
        else if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
            || other.is_instance_of::<pyo3::types::PyLong>()
        {
            let dt: f64 = other.extract::<f64>().unwrap();
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(PyInstant {
                    inner: self.inner - crate::Duration::from_days(dt),
                }
                .into_py(py))
            })
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Ok(PyInstant {
                inner: self.inner - dur.inner,
            }
            .into_py(other.py()))
        } else if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            let pdiff: crate::Duration = self.inner - tm2.inner;
            Ok(PyDuration { inner: pdiff }.into_py(other.py()))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid type for rhs",
            ))
        }
    }

    /// Check for equality
    ///
    /// Args:
    ///     other (satkit.time): Time object to compare
    ///
    /// Returns:
    ///     bool: True if equal, False otherwise
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            Ok(self.inner == tm2.inner)
        } else {
            Ok(false)
        }
    }

    /// Less than comparison
    ///
    /// Args:
    ///     other (satkit.time): Time object to compare
    ///
    /// Returns:
    ///     bool: True if less than, False otherwise
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            Ok(self.inner < tm2.inner)
        } else {
            Ok(false)
        }
    }

    /// Less than or equal comparison
    ///
    /// Args:
    ///     other (satkit.time): Time object to compare
    ///
    /// Returns:
    ///     bool: True if less than or equal, False otherwise
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            Ok(self.inner <= tm2.inner)
        } else {
            Ok(false)
        }
    }

    /// Greater than comparison
    ///
    /// Args:
    ///     other (satkit.time): Time object to compare
    ///
    /// Returns:
    ///     bool: True if greater than, False otherwise
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            Ok(self.inner > tm2.inner)
        } else {
            Ok(false)
        }
    }

    /// Greater than or equal comparison
    ///
    /// Args:
    ///     other (satkit.time): Time object to compare
    ///
    /// Returns:
    ///     
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyInstant>() {
            let tm2 = other.extract::<PyInstant>().unwrap();
            Ok(self.inner >= tm2.inner)
        } else {
            Ok(false)
        }
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
    fn add_utc_days(&self, days: f64) -> PyInstant {
        PyInstant {
            inner: self.inner.add_utc_days(days),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new_bound(py);
        d.set_item("empty", true).unwrap();
        (PyTuple::empty_bound(py), d)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let s = state.as_bytes(py);
        if s.len() != 8 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let raw = i64::from_le_bytes(s.try_into()?);
        self.inner = Instant::new(raw);
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new_bound(
            py,
            &i64::to_le_bytes(self.inner.raw)
        )
        .to_object(py))
    }
}

impl IntoPy<PyObject> for Instant {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ts: PyInstant = PyInstant { inner: self };
        ts.into_py(py)
    }
}

impl<'b> From<&'b PyInstant> for &'b Instant {
    fn from(s: &PyInstant) -> &Instant {
        &s.inner
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
            Ok(vec![tm.inner])
        } else if self.is_instance_of::<PyDateTime>() {
            let dt: Py<PyDateTime> = self.extract().unwrap();
            pyo3::Python::with_gil(|py| Ok(vec![datetime_to_instant(dt.bind(py)).unwrap()]))
        }
        // List case
        else if self.is_instance_of::<pyo3::types::PyList>() {
            match self.extract::<Vec<PyInstant>>() {
                Ok(v) => Ok(v.iter().map(|x| x.inner).collect::<Vec<_>>()),
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
                            match p.extract::<PyInstant>(py) {
                                Ok(v2) => Ok(v2.inner),
                                Err(_) => match p.extract::<Py<PyDateTime>>(py) {
                                    Ok(v3) => 
                                    pyo3::Python::with_gil(|py| {
                                        Ok(datetime_to_instant(v3.bind(py)).unwrap())
                                    }),
                                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                                        "Input numpy array must contain satkit.time elements or datetime.datetime elements".to_string()
                                    )),
                            }
                        }
                        })
                        .collect();
              
                    if let Ok(tm) = tmarray {
                        Ok(tm)
                    } else {
                        Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "Invalid satkit.time input",
                        ))
                    }
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
