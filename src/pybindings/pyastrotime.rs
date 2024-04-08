use pyo3::prelude::*;
use pyo3::types::timezone_utc;
use pyo3::types::PyDateTime;
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
/// UTC = Universal Time Coordinate
/// TT = Terrestrial Time
/// UT1 = Universal time, corrected for polar wandering
/// TAI = International Atomic Time
/// GPS = Global Positioning System Time (epoch = 1/6/1980 00:00:00)
/// TDB = Barycentric Dynamical Time
///
#[derive(Clone)]
#[pyclass(name = "timescale")]
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

///
/// Object representing an instant in time
///
/// Used for orbit propagation, frame transformations, etc..
///
/// * Includes function for conversion to various time representations
/// (e.g., julian date, modified julian date, gps time, ...)
///
/// * Also includes conversions between various scales
/// (e.g., UTC, Terrestrial Time, GPS, ...)
///
/// * Methods also included for conversion to & from the more-standard
/// "datetime" object used in Python
///
///  # Constructor argument options:
///
///    1:  None: Output current date / time
///
///    2:  Year, Month, Day:
///              Output object representing associated date
///              (same as "fromdate" method)
///    
///    3:  Year, Month, Day, Hour, Minute, Second, Scale
///              Output object representing associated date & time
///              (same as "fromgregorian" method)
///   
///
#[pyclass(name = "time")]
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug)]
pub struct PyAstroTime {
    pub inner: AstroTime,
}

#[pymethods]
impl PyAstroTime {
    ///
    /// Create a new time object
    ///
    ///
    /// Inputs Options:
    ///
    ///    1:  None: Output current date / time
    ///
    ///    2:  Year, Month, Day:
    ///              Output object representing associated date
    ///              (same as "fromdate" method)
    ///    
    ///    3:  Year, Month, Day, Hour, Minute, Second, Scale
    ///              Output object representing associated date & time
    ///              (same as "fromgregorian" method)
    ///              with optional time scale, default is UTC
    ///
    #[new]
    #[pyo3(signature=(*py_args))]
    fn py_new(py_args: &PyTuple) -> PyResult<Self> {
        if py_args.is_empty() {
            match AstroTime::now() {
                Ok(v) => Ok(PyAstroTime { inner: v }),
                Err(_) => Err(pyo3::exceptions::PyOSError::new_err(
                    "Could not get current time",
                )),
            }
        } else if py_args.len() == 3 {
            let year = py_args.get_item(0)?.extract::<u32>()?;
            let month = py_args.get_item(1)?.extract::<u32>()?;
            let day = py_args.get_item(2)?.extract::<u32>()?;
            Self::from_date(year, month, day)
        } else if py_args.len() >= 6 {
            let year = py_args.get_item(0)?.extract::<u32>()?;
            let month = py_args.get_item(1)?.extract::<u32>()?;
            let day = py_args.get_item(2)?.extract::<u32>()?;
            let hour = py_args.get_item(3)?.extract::<u32>()?;
            let min = py_args.get_item(4)?.extract::<u32>()?;
            let sec = py_args.get_item(5)?.extract::<f64>()?;
            let pyscale = match py_args.len() > 6 {
                false => PyTimeScale::UTC,
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
    #[staticmethod]
    fn now() -> PyResult<Self> {
        match AstroTime::now() {
            Ok(v) => Ok(PyAstroTime { inner: v }),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err(
                "Could not get current time",
            )),
        }
    }

    /// Return time object representing input
    /// Gregorian year, month (1=January, 2=February, ...), and
    /// day of month, beginning with 1.  Inputs assumed to be UTC
    ///
    /// # Arguments:
    ///
    /// * `year` - The year
    /// * `month` - The month (1=January, 2 = February, ...)
    /// * `day` - The day of the month, starting with "1"
    #[staticmethod]
    fn from_date(year: u32, month: u32, day: u32) -> PyResult<Self> {
        Ok(PyAstroTime {
            inner: AstroTime::from_date(year, month, day),
        })
    }

    /// Return time object representing input
    /// modified Julian date and time scale
    ///
    /// # Arguments:
    ///
    /// * `mjd` - The modified Julian Date
    /// * `scale` - The time scale
    ///
    /// # Returns:
    ///
    /// Time object representing instant of modified julian date with given scale
    #[staticmethod]
    fn from_mjd(mjd: f64, scale: &PyTimeScale) -> Self {
        PyAstroTime {
            inner: AstroTime::from_mjd(mjd, scale.into()),
        }
    }

    /// Return time object representing input
    /// Julian date and time scale
    ///
    /// # Arguments:
    ///
    /// * `jd` - The Julian Date
    /// * `scale` - The time scale
    ///
    /// # Returns:
    ///
    /// Time object representing instant of julian date with given scale
    #[staticmethod]
    fn from_jd(jd: f64, scale: &PyTimeScale) -> Self {
        PyAstroTime {
            inner: AstroTime::from_jd(jd, scale.into()),
        }
    }

    /// Convert time object to UTC Gegorian date, with
    /// returns tuple with 3 elements:
    /// 1 : Gregorian Year
    /// 2 : Gregorian month (1 = January, 2 = February, ...)
    /// 3 : Day of month, beginning with 1
    ///
    fn to_date(&self) -> (u32, u32, u32) {
        self.inner.to_date()
    }

    /// Convert time object to UTC Gegorian date and time, with
    /// returns tuple with 6 elements:
    /// 1 : Gregorian Year
    /// 2 : Gregorian month (1 = January, 2 = February, ...)
    /// 3 : Day of month, beginning with 1
    /// 4 : Hour of day, in range [0,23]
    /// 5 : Minute of hour, in range [0,59]
    /// 6 : floating point second of minute, in range [0,60)
    ///
    fn to_gregorian(&self) -> (u32, u32, u32, u32, u32, f64) {
        self.inner.to_datetime()
    }

    /// Convert UTC Gegorian date and time to time object with
    /// 6-element input:
    /// 1 : Gregorian Year
    /// 2 : Gregorian month (1 = January, 2 = February, ...)
    /// 3 : Day of month, beginning with 1
    /// 4 : Hour of day, in range [0,23]
    /// 5 : Minute of hour, in range [0,59]
    /// 6 : floating point second of minute, in range [0,60)
    /// 7 : Time scale (optional), default is satkit.timescale.UTC
    ///
    #[staticmethod]
    #[pyo3(signature=(year, month, day, hour, min, sec, scale=PyTimeScale::UTC))]
    fn from_gregorian(
        year: u32,
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
    /// # Arguments:
    /// 
    /// * `tm` - Python datetime object
    /// 
    /// # Returns:
    /// 
    /// SatKit Time object representing input datetime
    #[staticmethod]
    fn from_datetime(tm: &PyDateTime) -> PyResult<Self> {
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
    /// # Arguments:
    ///   
    /// # `utc_timezone` - Optional bool indicating use UTC as timezone
    ///                    if not passed in, defaults to true
    ///
    #[pyo3(signature = (utc=true))]
    fn datetime(&self, utc: bool) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let timestamp: f64 = self.to_unixtime();
            let tz = match utc {
                false => None,
                true => Some(timezone_utc(py)),
            };
            Ok(PyDateTime::from_timestamp(py, timestamp, tz)?.into_py(py))
        })
    }

    /// Convert to Modified Julian date
    /// 
    /// # Arguments:
    /// 
    /// * `scale` - Time scale to use for conversion
    ///             default is UTC
    /// # Returns:
    /// 
    /// Modified Julian Date
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_mjd(&self, scale: &PyTimeScale) -> f64 {
        self.inner.to_mjd(scale.into())
    }

    /// Convert to Julian date
    /// 
    /// # Arguments:
    /// 
    /// * `scale` - Time scale to use for conversion
    ///             default is UTC
    ///
    /// # Returns:
    ///    
    /// Julian Date
    #[pyo3(signature=(scale=&PyTimeScale::UTC))]
    fn to_jd(&self, scale: &PyTimeScale) -> f64 {
        self.inner.to_jd(scale.into())
    }

    fn to_unixtime(&self) -> f64 {
        self.inner.to_unixtime()
    }

    fn __add__(&self, other: &PyAny) -> PyResult<PyObject> {
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
                let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

                    let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

                    let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

    fn __sub__(&self, other: &PyAny) -> PyResult<PyObject> {
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
                let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

                    let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

                    let parr = np::PyArray1::<PyObject>::from_iter(py, objarr);
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

    fn __eq__(&self, other: &PyAny) -> PyResult<bool> {
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
            Ok(self.inner == tm2.inner)
        } else {
            Ok(false)
        }
    }

    fn __lt__(&self, other: &PyAny) -> PyResult<bool> {
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
            Ok(self.inner < tm2.inner)
        } else {
            Ok(false)
        }
    }

    fn __le__(&self, other: &PyAny) -> PyResult<bool> {
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
            Ok(self.inner <= tm2.inner)
        } else {
            Ok(false)
        }
    }

    fn __gt__(&self, other: &PyAny) -> PyResult<bool> {
        if other.is_instance_of::<PyAstroTime>() {
            let tm2 = other.extract::<PyAstroTime>().unwrap();
            Ok(self.inner > tm2.inner)
        } else {
            Ok(false)
        }
    }

    fn __ge__(&self, other: &PyAny) -> PyResult<bool> {
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
    /// # Arguments:
    /// 
    /// * `days` - Number of days to add
    /// 
    /// # Returns:
    /// 
    /// Time object representing input time plus given number of days
    /// 
    /// # Note:
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

pub trait ToTimeVec {
    fn to_time_vec(&self) -> PyResult<Vec<AstroTime>>;
}

impl ToTimeVec for &PyAny {
    fn to_time_vec(&self) -> PyResult<Vec<AstroTime>> {
        // "Scalar" time input case
        if self.is_instance_of::<PyAstroTime>() {
            let tm: PyAstroTime = self.extract().unwrap();
            Ok(vec![tm.inner.clone()])
        }
        else if self.is_instance_of::<PyDateTime>() {
            let dt: &PyDateTime = self.extract().unwrap();
            Ok(vec![PyAstroTime::from_datetime(dt).unwrap().inner])
        }
        // List case
        else if self.is_instance_of::<pyo3::types::PyList>() {
            match self.extract::<Vec<PyAstroTime>>() {
                Ok(v) => Ok(v.iter().map(|x| x.inner).collect::<Vec<_>>()),
                Err(e) => {
                    match self.extract::<Vec<&PyDateTime>>() {
                        Ok(v) => Ok(v.iter().map(|x| 
                            PyAstroTime::from_datetime(x.extract().unwrap()).unwrap().inner)
                            .collect::<Vec<_>>()),
                        Err(e) => {
                            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                "Not a list of satkit.time or datetime.datetime: {e}"
                            )))
                        }
                    }
                }
               
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
                                Err(_) => match p.extract::<&PyDateTime>(py) {
                                    Ok(v3) => Ok(PyAstroTime::from_datetime(v3.extract().unwrap()).unwrap().inner),
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
