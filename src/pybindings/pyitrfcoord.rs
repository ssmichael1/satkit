use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::{PyList, PyTuple};

use numpy::{PyArray1, PyReadonlyArray1};

use crate::itrfcoord::ITRFCoord;
use crate::types::Vec3;

use super::Quaternion;

use super::pyutils::*;

///
/// Representation of a coordinate in the
/// International Terrestrial Reference Frame (ITRF)
///
/// This coordinate object can be created from and also
/// output to Geodetic coordinates (latitude, longitude,
/// height above ellipsoid).
///
/// Functions are also available to provide rotation
/// quaternions to the East-North-Up frame
/// and North-East-Down frame at this coordinate
///
#[pyclass(name = "itrfcoord")]
#[derive(Clone)]
pub struct PyITRFCoord {
    pub inner: ITRFCoord,
}

#[pymethods]
impl PyITRFCoord {
    /// Create a coordinate in the ITRF (International Terrestrial Reference Frame)
    ///
    ///
    /// Coordinate from "Cartesian" inputs can be set via folloing:
    ///
    ///   1: single 3-element list of floats representing ITRF Cartesian location in meters
    ///   2: single 3-element numpy array of floats representing ITRF Cartesian location in meters
    ///   3. 3 separate input arguments representing x,y,z ITRF Cartesian location in meters
    ///
    /// Input can also be set from geodetic coordinate using kwargs
    ///     Optional kwargs:
    ///
    ///     latitude_deg: latitude, degrees
    ///     longitude_deg: longitude, degrees
    ///     latitude_rad: latitude, radians
    ///     longitude_rad: longitude, radians
    ///     altitude: height above ellipsoid, meters
    ///     height: height above ellipsoid, meters
    ///
    ///   Note that there are 2 ways to specify latitude, longitude, and altitude
    ///   All 3 must be specified or an error will be triffered
    ///
    /// Output:
    ///
    ///   New "ITRF" Coordinate
    ///
    /// Example:
    ///
    ///   1. Create ITRF coord from cartesian
    ///
    ///   coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
    ///   print(coord)
    ///
    ///   ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
    ///
    ///   2. Create same ITRF coord from geodetic
    ///
    ///   coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
    ///  
    ///   print(coord)
    ///
    ///   ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
    ///

    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &PyTuple, mut kwargs: Option<&PyDict>) -> PyResult<Self> {
        // If kwargs are set, we get input from them
        if kwargs.is_some() {
            use std::f64::consts::PI;

            let mut latitude_deg: Option<f64> = kwargs_or_none(&mut kwargs, "latitude_deg")?;
            latitude_deg = match kwargs_or_none::<f64>(&mut kwargs, "latitude_rad")? {
                None => latitude_deg,
                Some(v) => Some(v * 180.0 / PI),
            };
            let mut longitude_deg: Option<f64> = kwargs_or_none(&mut kwargs, "longitude_deg")?;
            longitude_deg = match kwargs_or_none::<f64>(&mut kwargs, "longitude_rad")? {
                None => longitude_deg,
                Some(v) => Some(v * 180.0 / PI),
            };
            let mut altitude: f64 = kwargs_or_default(&mut kwargs, "altitude", 0.0)?;
            altitude = match kwargs_or_none(&mut kwargs, "height")? {
                None => altitude,
                Some(v) => v,
            };

            if latitude_deg.is_none() || longitude_deg.is_none() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Must set latitude, longitude",
                ));
            }
            Ok(PyITRFCoord {
                inner: ITRFCoord::from_geodetic_deg(
                    latitude_deg.unwrap(),
                    longitude_deg.unwrap(),
                    altitude,
                ),
            })
        } else {
            if args.len() == 3 {
                let x = args[0].extract::<f64>()?;
                let y = args[1].extract::<f64>()?;
                let z = args[2].extract::<f64>()?;
                Ok(PyITRFCoord {
                    inner: ITRFCoord::from_slice(&[x, y, z]).unwrap(),
                })
            } else if args.len() == 1 {
                if args[0].is_instance_of::<PyList>() {
                    match args[0].extract::<Vec<f64>>() {
                        Ok(xl) => {
                            if xl.len() != 3 {
                                return Err(pyo3::exceptions::PyTypeError::new_err(
                                    "Invalid number of elements",
                                ));
                            }
                            Ok(PyITRFCoord {
                                inner: ITRFCoord::from_slice(&xl).unwrap(),
                            })
                        }
                        Err(e) => Err(e),
                    }
                } else if args[0].is_instance_of::<PyArray1<f64>>() {
                    let xv = args[0].extract::<PyReadonlyArray1<f64>>().unwrap();
                    if xv.len() != 3 {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "Invalid number of elements",
                        ));
                    }
                    Ok(PyITRFCoord {
                        inner: ITRFCoord::from_slice(xv.as_slice().unwrap()).unwrap(),
                    })
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                "First input must be float, 3-element list of floats, or 3-element numpy array of float"
            ));
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                "First input must be float, 3-element list of floats, or 3-element numpy array of float"
            ));
            }
        }
    }

    #[getter]
    /// Latitude in degrees
    fn get_latitude_deg(&self) -> f64 {
        self.inner.latitude_deg()
    }

    /// Longitude in degrees
    #[getter]
    fn get_longitude_deg(&self) -> f64 {
        self.inner.longitude_deg()
    }

    /// Latitude in radians
    #[getter]
    fn get_latitude_rad(&self) -> f64 {
        self.inner.latitude_rad()
    }

    /// Longitude in radians
    #[getter]
    fn get_longitude_rad(&self) -> f64 {
        self.inner.longitude_rad()
    }

    /// Height above ellipsoid in meters
    #[getter]
    fn get_height(&self) -> f64 {
        self.inner.hae()
    }

    /// Height above ellipsoid, meters
    #[getter]
    fn get_altitude(&self) -> f64 {
        self.inner.hae()
    }

    /// Return Tuple with latitude in rad, longitude in rad,
    /// height above ellipsoid in meters
    #[getter]
    fn get_geodetic_rad(&self) -> (f64, f64, f64) {
        self.inner.to_geodetic_rad()
    }

    /// Return tuple with latitude in deg, longitude in deg,
    /// height above ellipsoid in meters
    #[getter]
    fn get_geodetic_deg(&self) -> (f64, f64, f64) {
        self.inner.to_geodetic_deg()
    }

    /// Return vector representing ITRF Cartesian coordinate
    /// in meters
    #[getter]
    fn get_vector(&self) -> PyObject {
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, self.inner.itrf.data.as_slice()).to_object(py)
        })
    }

    fn __str__(&self) -> String {
        let (lat, lon, hae) = self.inner.to_geodetic_deg();
        format!(
            "ITRFCoord(lat: {:8.4} deg, lon: {:8.4} deg, hae: {:5.2} km)",
            lat,
            lon,
            hae / 1.0e3
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    /// Quaternion representing rotation from
    /// North-East-Down (NED) coordinate frame
    /// to International Terrestrial Reference Frame
    /// (ITRF) at this point
    #[getter]
    fn get_qned2itrf(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.q_ned2itrf(),
        }
    }

    /// Quaternion representing rotation from
    /// East-North-Up (ENU) coordinate frame
    /// to International Terrestrial Reference Frame
    /// (ITRF) at this point
    #[getter]
    fn get_qenu2itrf(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.q_enu2itrf(),
        }
    }

    /// Return East-North-Up location of input
    /// coordinate relative to self
    fn to_enu(&self, other: &Self) -> PyObject {
        let v: Vec3 = self.inner.q_enu2itrf().conjugate() * (self.inner.itrf - other.inner.itrf);
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, v.data.as_slice()).to_object(py)
        })
    }

    /// Return North-East-Down location of input
    /// coordinate relative to self
    fn to_ned(&self, other: &Self) -> PyObject {
        let v: Vec3 = self.inner.q_ned2itrf().conjugate() * (self.inner.itrf - other.inner.itrf);
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, v.data.as_slice()).to_object(py)
        })
    }

    /// Use Vincenty formula to compute geodesic distance:
    /// https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    ///
    /// Return a tuple with:
    ///
    ///  1: geodesic distance (shortest distance between two points)
    ///  between this coordinate and given coordinate, in meters
    ///
    ///  2: initial heading, in radians
    ///
    ///  3. heading at destination, in radians
    ///
    fn geodesic_distance(&self, other: &Self) -> (f64, f64, f64) {
        self.inner.geodesic_distance(&other.inner)
    }

    /// Takes two input arguments:
    ///
    /// 1: distance (meters)
    /// 2: heading (rad)
    ///
    /// Return itrfcoord representing move along Earth surface by given distance
    /// in direction given by heading
    ///
    /// Altitude is assumed to be zero
    ///
    /// Use Vincenty formula to compute position:
    /// https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    ///
    fn move_with_heading(&self, distance: f64, heading_rad: f64) -> PyITRFCoord {
        PyITRFCoord {
            inner: self.inner.move_with_heading(distance, heading_rad),
        }
    }

    /// 3-vector representing cartesian distance between this
    /// and other point, in meters
    fn __sub__(&self, other: &PyITRFCoord) -> PyObject {
        let vout = self.inner - other.inner;
        pyo3::Python::with_gil(|py| -> PyObject {
            let vnd = PyArray1::<f64>::from_vec(py, vec![vout[0], vout[1], vout[2]]);
            vnd.into_py(py)
        })
    }
}

impl IntoPy<PyObject> for ITRFCoord {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyITRFCoord { inner: self }.into_py(py)
    }
}

impl<'b> From<&'b PyITRFCoord> for &'b ITRFCoord {
    fn from<'a>(s: &'a PyITRFCoord) -> &'a ITRFCoord {
        &s.inner
    }
}
