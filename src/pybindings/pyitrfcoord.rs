use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::{PyBytes, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use numpy::{PyArray1, PyReadonlyArray1};

use crate::itrfcoord::ITRFCoord;
use crate::types::Vec3;

use super::Quaternion;

use super::pyutils::*;

///
/// Representation of a coordinate in the International Terrestrial Reference Frame (ITRF)
///
/// Note:
/// This coordinate object can be created from and also
/// output to Geodetic coordinates (latitude, longitude,
/// height above ellipsoid).
///
/// Note:
/// Functions are also available to provide rotation
/// quaternions to the East-North-Up frame
/// and North-East-Down frame at this coordinate
///
/// Args:
///     vec (numpy.ndarray, list, or 3-element tuple of floats, optional): ITRF Cartesian location in meters
///  
/// Keyword Args:
///     latitude_deg (float, optional): Latitude in degrees
///     longitude_deg (float, optional): Longitude in degrees
///     latitude_rad (float, optional): Latitude in radians
///     longitude_rad (float, optional): Longitude in radians
///     altitude (float, optional): Height above ellipsoid, meters
///     height (float, optional): Height above ellipsoid, meters
///     
///
/// Returns:
///     itrfcoord: New ITRF coordinate
///
/// Example:
///     * Create ITRF coord from Cartesian        
///        >>> coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
///        >>> print(coord)
///        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
///
/// Example:
///     * Create ITRF coord from Geodetic
///        >>> coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
///        >>> print(coord)
///        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
///       
///
#[pyclass(name = "itrfcoord", module = "satkit")]
#[derive(Clone)]
pub struct PyITRFCoord(pub ITRFCoord);

#[pymethods]
impl PyITRFCoord {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, mut kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // If kwargs are set, we get input from them
        if kwargs.is_some() {
            let mut latitude_deg: Option<f64> = kwargs_or_none(&mut kwargs, "latitude_deg")?;
            latitude_deg = (kwargs_or_none::<f64>(&mut kwargs, "latitude_rad")?)
                .map_or(latitude_deg, |v| Some(v.to_degrees()));
            let mut longitude_deg: Option<f64> = kwargs_or_none(&mut kwargs, "longitude_deg")?;
            longitude_deg = (kwargs_or_none::<f64>(&mut kwargs, "longitude_rad")?)
                .map_or(longitude_deg, |v| Some(v.to_degrees()));
            let mut altitude: f64 = kwargs_or_default(&mut kwargs, "altitude", 0.0)?;
            altitude = (kwargs_or_none(&mut kwargs, "height")?).map_or(altitude, |v| v);

            if latitude_deg.is_none() || longitude_deg.is_none() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Must set latitude, longitude",
                ));
            }
            Ok(Self(ITRFCoord::from_geodetic_deg(
                latitude_deg.unwrap(),
                longitude_deg.unwrap(),
                altitude,
            )))
        } else if args.len() == 3 {
            let x = args.get_item(0)?.extract::<f64>()?;
            let y = args.get_item(1)?.extract::<f64>()?;
            let z = args.get_item(2)?.extract::<f64>()?;
            Ok(Self(ITRFCoord::from_slice(&[x, y, z]).unwrap()))
        } else if args.len() == 1 {
            if args.get_item(0)?.is_instance_of::<PyList>() {
                match args.get_item(0)?.extract::<Vec<f64>>() {
                    Ok(xl) => {
                        if xl.len() != 3 {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "Invalid number of elements",
                            ));
                        }
                        Ok(Self(ITRFCoord::from_slice(&xl).unwrap()))
                    }
                    Err(e) => Err(e),
                }
            } else if args.get_item(0)?.is_instance_of::<PyArray1<f64>>() {
                let xv = args
                    .get_item(0)?
                    .extract::<PyReadonlyArray1<f64>>()
                    .unwrap();
                if xv.len().unwrap() != 3 {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Invalid number of elements",
                    ));
                }
                Ok(Self(ITRFCoord::from_slice(xv.as_slice().unwrap()).unwrap()))
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

    #[getter]
    /// Latitude in degrees
    ///
    /// Returns:
    ///     float: Latitude in degrees
    fn get_latitude_deg(&self) -> f64 {
        self.0.latitude_deg()
    }

    /// Longitude in degrees
    ///
    /// Returns:
    ///     float: Longitude in degrees
    #[getter]
    fn get_longitude_deg(&self) -> f64 {
        self.0.longitude_deg()
    }

    /// Latitude in radians
    ///
    /// Returns:
    ///     float: Latitude in radians
    #[getter]
    fn get_latitude_rad(&self) -> f64 {
        self.0.latitude_rad()
    }

    /// Longitude in radians
    ///
    /// Returns:
    ///     float: Longitude in radians
    #[getter]
    fn get_longitude_rad(&self) -> f64 {
        self.0.longitude_rad()
    }

    /// Height above ellipsoid in meters
    ///
    /// Returns:
    ///     float: Height above ellipsoid in meters
    #[getter]
    fn get_height(&self) -> f64 {
        self.0.hae()
    }

    /// Height above ellipsoid, meters
    #[getter]
    fn get_altitude(&self) -> f64 {
        self.0.hae()
    }

    /// Return Tuple with latitude in rad, longitude in rad, height above ellipsoid in meters
    ///
    /// Returns:
    ///     tuple: (latitude_rad, longitude_rad, height)
    #[getter]
    fn get_geodetic_rad(&self) -> (f64, f64, f64) {
        self.0.to_geodetic_rad()
    }

    /// Return tuple with latitude in deg, longitude in deg, height above ellipsoid in meters
    ///
    /// Returns:
    ///     tuple: (latitude_deg, longitude_deg, height)
    #[getter]
    fn get_geodetic_deg(&self) -> (f64, f64, f64) {
        self.0.to_geodetic_deg()
    }

    /// Return vector representing ITRF Cartesian coordinate in meters
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array of floats representing ITRF Cartesian location in meters
    #[getter]
    fn get_vector(&self) -> PyObject {
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, self.0.itrf.data.as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }

    fn __str__(&self) -> String {
        let (lat, lon, hae) = self.0.to_geodetic_deg();
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

    /// Quaternion representing rotation from North-East-Down (NED) coordinate frame to International Terrestrial Reference Frame
    /// (ITRF) at this coordinate
    ///
    /// Returns:
    ///     satkit.quaternion: Quaternion representing rotation from NED to ITRF
    #[getter]
    fn get_qned2itrf(&self) -> Quaternion {
        self.0.q_ned2itrf().into()
    }

    /// Quaternion representing rotation from East-North-Up (ENU) coordinate frame to International Terrestrial Reference Frame
    /// (ITRF) at this coordinate
    ///
    /// Returns:
    ///     satkit.quaternion: Quaternion representing rotation from ENU to ITRF
    #[getter]
    fn get_qenu2itrf(&self) -> Quaternion {
        self.0.q_enu2itrf().into()
    }

    /// Return East-North-Up location of input coordinate relative to self
    ///
    /// Args:
    ///     other (itrfcoord): ITRF coordinate for which to compute ENU location
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array of floats representing ENU location in meters of other relative to self
    fn to_enu(&self, other: &Self) -> PyObject {
        let v: Vec3 = self.0.q_enu2itrf().conjugate() * (self.0.itrf - other.0.itrf);
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, v.data.as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }

    /// Return North-East-Down location of input coordinate relative to self
    ///
    /// Args:
    ///     other (itrfcoord): ITRF coordinate for which to compute NED location
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array of floats representing NED location in meters of other relative to self
    fn to_ned(&self, other: &Self) -> PyObject {
        let v: Vec3 = self.0.q_ned2itrf().conjugate() * (self.0.itrf - other.0.itrf);
        pyo3::Python::with_gil(|py| -> PyObject {
            numpy::PyArray::from_slice(py, v.data.as_slice())
                .into_py_any(py)
                .unwrap()
        })
    }

    /// Compute geodesic distance:
    ///
    /// Notes:
    ///     Uses Vincenty formula to compute distance: https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    ///
    ///
    /// Args:
    ///     other (itrfcoord): ITRF coordinate for which to compute distance
    ///
    /// Returns:
    ///     tuple: (distance in meters, initial heading in radians, heading at destination in radians)
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
        self.0.geodesic_distance(&other.0)
    }

    /// Move this coordinate along a given heading by a given distance
    ///
    /// Notes:
    ///     * Heading is in radians, where 0 is North, pi/2 is East, pi is South, 3pi/2 is West
    ///     * Distance is in meters
    ///     * Uses inverse Vincenty formula to compute new location
    ///
    /// Args:
    ///     distance (float): Distance to move in meters
    ///     heading_rad (float): Heading in radians
    ///
    /// Returns:
    ///     itrfcoord: New ITRF coordinate after moving
    fn move_with_heading(&self, distance: f64, heading_rad: f64) -> Self {
        Self(self.0.move_with_heading(distance, heading_rad))
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::new(py, vec![0.0, 0.0, 0.0]).unwrap();
        (tp, d)
    }

    fn __setstate__(&mut self, py: Python, s: Py<PyBytes>) -> PyResult<()> {
        let s = s.as_bytes(py);
        if s.len() != 24 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let x = f64::from_le_bytes(s[0..8].try_into()?);
        let y = f64::from_le_bytes(s[8..16].try_into()?);
        let z = f64::from_le_bytes(s[16..24].try_into()?);
        self.0.itrf = nalgebra::Vector3::<f64>::new(x, y, z);
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let mut raw = [0; 24];
        raw[0..8].clone_from_slice(f64::to_le_bytes(self.0.itrf[0]).as_slice());
        raw[8..16].clone_from_slice(f64::to_le_bytes(self.0.itrf[1]).as_slice());
        raw[16..24].clone_from_slice(f64::to_le_bytes(self.0.itrf[2]).as_slice());
        pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
    }

    /// 3-vector representing cartesian distance between this
    /// and other point, in meters
    fn __sub__(&self, other: &Self) -> PyObject {
        let vout = self.0 - other.0;
        pyo3::Python::with_gil(|py| -> PyObject {
            let vnd = PyArray1::<f64>::from_vec(py, vec![vout[0], vout[1], vout[2]]);
            vnd.into_py_any(py).unwrap()
        })
    }
}
