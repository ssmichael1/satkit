use pyo3::prelude::*;
use pyo3::types::timezone_utc_bound;
use pyo3::types::PyBytes;
use pyo3::types::PyDateTime;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;

use crate::astrotime::{self, AstroTime, Scale};

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
    Invalid = Scale::INVALID as isize,
    /// Universal Time Coordinate
    UTC = Scale::UTC as isize,
    /// Terrestrial Time
    TT = Scale::TT as isize,
    /// UT1
    UT1 = Scale::UT1 as isize,
    /// International Atomic Time
    TAI = Scale::TAI as isize,
    /// Global Positioning System (GPS) Time
    GPS = Scale::GPS as isize,
    /// Barycentric Dynamical Time
    TDB = Scale::TDB as isize,
}

impl From<&PyTimeScale> for astrotime::Scale {
    fn from(s: &PyTimeScale) -> astrotime::Scale {
        match s {
            PyTimeScale::Invalid => Scale::INVALID,
            PyTimeScale::UTC => Scale::UTC,
            PyTimeScale::TT => Scale::TT,
            PyTimeScale::UT1 => Scale::UT1,
            PyTimeScale::TAI => Scale::TAI,
            PyTimeScale::GPS => Scale::GPS,
            PyTimeScale::TDB => Scale::TDB,
        }
    }
}

impl From<PyTimeScale> for astrotime::Scale {
    fn from(s: PyTimeScale) -> astrotime::Scale {
        match s {
            PyTimeScale::Invalid => Scale::INVALID,
            PyTimeScale::UTC => Scale::UTC,
            PyTimeScale::TT => Scale::TT,
            PyTimeScale::UT1 => Scale::UT1,
            PyTimeScale::TAI => Scale::TAI,
            PyTimeScale::GPS => Scale::GPS,
            PyTimeScale::TDB => Scale::TDB,
        }
    }
}

impl IntoPy<PyObject> for astrotime::Scale {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ts: PyTimeScale = match self {
            Scale::INVALID => PyTimeScale::Invalid,
            Scale::UTC => PyTimeScale::UTC,
            Scale::TT => PyTimeScale::TT,
            Scale::UT1 => PyTimeScale::UT1,
            Scale::TAI => PyTimeScale::TAI,
            Scale::GPS => PyTimeScale::GPS,
            Scale::TDB => PyTimeScale::TDB,
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
pub struct PyAstroTime {
    pub inner: AstroTime,
}

#[pymethods]
impl PyAstroTime {
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
    ///     scale (satkit.timescale, optional): Time scale (optional), default is satkit.timescale.UTC    
    ///
    /// Note: If no arguments are passed in, the created object represents the current time
    ///
    /// Returns:
    ///     satkit.time: Time object representing input date and time, or if no arguments, the current date and time
    #[new]
    #[pyo3(signature=(*py_args, **py_kwargs))]
    fn py_new(
        py_args: &Bound<'_, PyTuple>,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut pyscale = PyTimeScale::UTC;
        if let Some(kw) = py_kwargs {
            if let Some(scale) = kw.get_item("scale")? {
                pyscale = scale.extract::<PyTimeScale>()?;
            }
            if let Some(empty) = kw.get_item("empty")? {
                let bempty = empty.extract::<bool>()?;
                if bempty == true {
                    return Ok(PyAstroTime {
                        inner: AstroTime { mjd_tai: 0.0 },
                    });
                }
            }
        }

        if py_args.is_empty() {
            match AstroTime::now() {
                Ok(v) => Ok(PyAstroTime { inner: v }),
                Err(_) => Err(pyo3::exceptions::PyOSError::new_err(
                    "Could not get current time",
                )),
            }
        } else if py_args.len() == 3 {
            let year = py_args.get_item(0)?.extract::<i32>()?;
            let month = py_args.get_item(1)?.extract::<u32>()?;
            let day = py_args.get_item(2)?.extract::<u32>()?;
            Self::from_date(year, month, day)
        } else if py_args.len() >= 6 {
            let year = py_args.get_item(0)?.extract::<i32>()?;
            let month = py_args.get_item(1)?.extract::<u32>()?;
            let day = py_args.get_item(2)?.extract::<u32>()?;
            let hour = py_args.get_item(3)?.extract::<u32>()?;
            let min = py_args.get_item(4)?.extract::<u32>()?;
            let sec = py_args.get_item(5)?.extract::<f64>()?;
            let pyscale = match py_args.len() > 6 {
                false => pyscale,
                true => py_args.get_item(6)?.extract::<PyTimeScale>()?,
            };
            Self::from_gregorian(year, month, day, hour, min, sec, pyscale)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Must pass in year, month, day or year, month, day, hour, min, sec",
            ))
        }
    }

    /// Return current time
    ///
    /// Returns:
    ///     satkit.time: Time object representing current time
    #[staticmethod]
    fn now() -> PyResult<Self> {
        match AstroTime::now() {
            Ok(v) => Ok(PyAstroTime { inner: v }),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err(
                "Could not get current time",
            )),
        }
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
    fn from_date(year: i32, month: u32, day: u32) -> PyResult<Self> {
        Ok(PyAstroTime {
            inner: AstroTime::from_date(year, month, day),
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
        PyAstroTime {
            inner: AstroTime::from_mjd(mjd, scale.into()),
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
        PyAstroTime {
            inner: AstroTime::from_unixtime(t),
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
        PyAstroTime {
            inner: AstroTime::from_jd(jd, scale.into()),
        }
    }

    /// Convert time object to UTC Gegorian date
    ///
    /// Returns:
    ///    (int, int, int): Tuple with 3 elements representing Gregorian year, month, and day
    fn to_date(&self) -> (u32, u32, u32) {
        self.inner.to_date()
    }

    /// Convert time object to UTC Gegorian date and time, with fractional seconds
    ///
    /// Returns:
    ///     (int, int, int, int, int, float): Tuple with 6 elements representing Gregorian year, month, day, hour, minute, and second
    ///
    fn to_gregorian(&self) -> (u32, u32, u32, u32, u32, f64) {
        self.inner.to_datetime()
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
    #[pyo3(signature=(year, month, day, hour, min, sec, scale=PyTimeScale::UTC))]
    fn from_gregorian(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
        sec: f64,
        scale: PyTimeScale,
    ) -> PyResult<Self> {
        Ok(PyAstroTime {
            inner: AstroTime::from_datetime_with_scale(
                year,
                month,
                day,
                hour,
                min,
                sec,
                scale.into(),
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
        Ok(PyAstroTime {
            inner: AstroTime::from_unixtime(ts),
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
            let timestamp: f64 = self.to_unixtime();
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
    fn to_mjd(&self, scale: &PyTimeScale) -> f64 {
        self.inner.to_mjd(scale.into())
    }

    /// Convert to Julian date
    ///
    /// Args:
    ///     scale (satkit.timescale, optional: Time scale to use for conversion, default is satkit.timescale.UTC
    ///
    /// Returns:
    ///     float: Julian Date
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_jd(&self, scale: &PyTimeScale) -> f64 {
        self.inner.to_jd(scale.into())
    }

    /// Convert to Unix time (seconds since 1970-01-01 00:00:00 UTC)
    ///
    /// Returns:
    ///     float: Unix time (seconds since 1970-01-01 00:00:00 UTC)
    fn to_unixtime(&self) -> f64 {
        self.inner.to_unixtime()
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
                    .into_iter()
                    .map(|x| {
                        let obj = PyAstroTime {
                            inner: self.inner + *x,
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
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyAstroTime {
                                inner: self.inner + x,
                            };
                            pyobj.into_py(py)
                        })
                        .into_iter();

                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else if let Ok(v) = other.extract::<Vec<PyDuration>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyAstroTime {
                                inner: self.inner + x.inner,
                            };
                            pyobj.into_py(py)
                        })
                        .into_iter();

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
                Ok(PyAstroTime {
                    inner: self.inner + dt,
                }
                .into_py(py))
            })
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Ok(PyAstroTime {
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
                        let obj = PyAstroTime {
                            inner: self.inner - *x,
                        };
                        obj.into_py(py)
                    })
                    .into_iter();
                let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
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
                            let pyobj = PyAstroTime {
                                inner: self.inner - x,
                            };
                            pyobj.into_py(py)
                        })
                        .into_iter();

                    let parr = np::PyArray1::<PyObject>::from_iter_bound(py, objarr);
                    Ok(parr.into_py(py))
                })
            } else if let Ok(v) = other.extract::<Vec<PyDuration>>() {
                pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                    let objarr = v
                        .into_iter()
                        .map(|x| {
                            let pyobj = PyAstroTime {
                                inner: self.inner - x.inner,
                            };
                            pyobj.into_py(py)
                        })
                        .into_iter();

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
                Ok(PyAstroTime {
                    inner: self.inner - dt,
                }
                .into_py(py))
            })
        } else if other.is_instance_of::<PyDuration>() {
            let dur: PyDuration = other.extract::<PyDuration>().unwrap();
            Ok(PyAstroTime {
                inner: self.inner - dur.inner,
            }
            .into_py(other.py()))
        } else if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
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
    fn add_utc_days(&self, days: f64) -> PyAstroTime {
        PyAstroTime {
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
        let t = f64::from_le_bytes(s.try_into()?);
        self.inner = AstroTime::from_mjd(t, astrotime::Scale::TAI);
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new_bound(
            py,
            f64::to_le_bytes(self.inner.to_mjd(astrotime::Scale::TAI)).as_slice(),
        )
        .to_object(py))
    }
}

impl IntoPy<PyObject> for astrotime::AstroTime {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ts: PyAstroTime = PyAstroTime { inner: self };
        ts.into_py(py)
    }
}

impl<'b> From<&'b PyAstroTime> for &'b astrotime::AstroTime {
    fn from<'a>(s: &'a PyAstroTime) -> &'a astrotime::AstroTime {
        &s.inner
    }
}

fn datetime2astrotime(tm: &Bound<PyDateTime>) -> PyResult<AstroTime> {
    let ts: f64 = tm
        .call_method("timestamp", (), None)
        .unwrap()
        .extract::<f64>()
        .unwrap();
    Ok(AstroTime::from_unixtime(ts))
}

pub trait ToTimeVec {
    fn to_time_vec(&self) -> PyResult<Vec<AstroTime>>;
}

impl ToTimeVec for &Bound<'_, PyAny> {
    fn to_time_vec(&self) -> PyResult<Vec<AstroTime>> {
        // "Scalar" time input case
        if self.is_instance_of::<PyAstroTime>() {
            let tm: PyAstroTime = self.extract().unwrap();
            Ok(vec![tm.inner.clone()])
        } else if self.is_instance_of::<PyDateTime>() {
            let dt: Py<PyDateTime> = self.extract().unwrap();
            pyo3::Python::with_gil(|py| Ok(vec![datetime2astrotime(dt.bind(py)).unwrap()]))
        }
        // List case
        else if self.is_instance_of::<pyo3::types::PyList>() {
            match self.extract::<Vec<PyAstroTime>>() {
                Ok(v) => Ok(v.iter().map(|x| x.inner).collect::<Vec<_>>()),
                Err(_e) => match self.extract::<Vec<Py<PyDateTime>>>() {
                    Ok(v) => pyo3::Python::with_gil(|py| {
                        Ok(v.iter()
                            .map(|x| datetime2astrotime(x.bind(py)).unwrap())
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
                Ok(v) => pyo3::Python::with_gil(|py| -> PyResult<Vec<AstroTime>> {
                    // Extract times from numpya array of objects
                    let tmarray: Result<Vec<AstroTime>, _> = v
                        .as_array()
                        .into_iter()
                        .map(|p| -> Result<AstroTime, _> {
                            match p.extract::<PyAstroTime>(py) {
                                Ok(v2) => Ok(v2.inner),
                                Err(_) => match p.extract::<Py<PyDateTime>>(py) {
                                    Ok(v3) => 
                                    pyo3::Python::with_gil(|py| {
                                        Ok(datetime2astrotime(v3.bind(py)).unwrap())
                                    }),
                                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                        "Input numpy array must contain satkit.time elements or datetime.datetime elements"
                                    ))),
                            }
                        }
                        })
                        .collect();
                    if !tmarray.is_ok() {
                        Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "Invalid satkit.time input",
                        ))
                    } else {
                        Ok(tmarray.unwrap())
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
