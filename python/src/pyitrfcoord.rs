use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::{PyBytes, PyTuple};

use satkit::itrfcoord::{Geodetic, ITRFCoord};

use crate::pyquaternion::PyQuaternion;

use crate::pyutils::*;

/// Geodetic coordinates with named fields
///
/// Returned by ``itrfcoord.geodetic``; it has no constructor and its fields
/// are read-only.
///
/// Attributes:
///     latitude_rad (float): Latitude in radians
///     longitude_rad (float): Longitude in radians
///     height_m (float): Height above WGS84 ellipsoid in meters
///     latitude_deg (float): Latitude in degrees
///     longitude_deg (float): Longitude in degrees
#[pyclass(name = "geodetic", module = "satkit", get_all, frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyGeodet {
    /// Latitude in radians
    pub latitude_rad: f64,
    /// Longitude in radians
    pub longitude_rad: f64,
    /// Height above WGS84 ellipsoid in meters
    pub height_m: f64,
}

impl From<Geodetic> for PyGeodet {
    fn from(g: Geodetic) -> Self {
        Self {
            latitude_rad: g.latitude_rad,
            longitude_rad: g.longitude_rad,
            height_m: g.height_m,
        }
    }
}

#[pymethods]
impl PyGeodet {
    /// Latitude in degrees
    #[getter]
    fn latitude_deg(&self) -> f64 {
        self.latitude_rad.to_degrees()
    }

    /// Longitude in degrees
    #[getter]
    fn longitude_deg(&self) -> f64 {
        self.longitude_rad.to_degrees()
    }

    fn __str__(&self) -> String {
        format!(
            "geodetic(lat: {:8.4} deg, lon: {:8.4} deg, height: {:5.2} m)",
            self.latitude_deg(),
            self.longitude_deg(),
            self.height_m,
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    /// True when ``other`` is a geodetic with exactly the same latitude,
    /// longitude and height (no tolerance)
    fn __eq__(&self, other: &Self) -> bool {
        self.latitude_rad == other.latitude_rad
            && self.longitude_rad == other.longitude_rad
            && self.height_m == other.height_m
    }

    /// Hash of the three fields, consistent with ``==``
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        (self.latitude_rad, self.longitude_rad, self.height_m)
            .into_pyobject(py)?
            .hash()
    }

    /// Rebuild a geodetic from its three fields. Used internally by
    /// `__reduce__`; not part of the public API.
    #[staticmethod]
    const fn _from_pickle(latitude_rad: f64, longitude_rad: f64, height_m: f64) -> Self {
        Self {
            latitude_rad,
            longitude_rad,
            height_m,
        }
    }

    /// Pickle (and `copy.deepcopy`) support. `geodetic` has no `__new__`,
    /// so it is rebuilt through the private `_from_pickle` staticmethod from
    /// all three fields (Python floats pickle exactly).
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (f64, f64, f64))> {
        let ctor = py.get_type::<Self>().getattr("_from_pickle")?;
        Ok((ctor, (self.latitude_rad, self.longitude_rad, self.height_m)))
    }
}

/// Define `fn $f(&Bound<PyAny>) -> PyResult<Option<f64>>` for
/// `#[pyo3(from_py_with = $f)]`: a geodetic keyword that is not a real
/// number raises `TypeError` naming the keyword and the type given.
macro_rules! real_keyword {
    ($f:ident, $name:literal) => {
        fn $f(v: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
            v.extract::<Option<f64>>().map_err(|_| {
                let tname = v
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_string(), |n| n.to_string());
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "{} must be a real number, got {tname}",
                    $name
                ))
            })
        }
    };
}
real_keyword!(latitude_deg_arg, "latitude_deg");
real_keyword!(longitude_deg_arg, "longitude_deg");
real_keyword!(latitude_rad_arg, "latitude_rad");
real_keyword!(longitude_rad_arg, "longitude_rad");
real_keyword!(altitude_arg, "altitude");
real_keyword!(height_arg, "height");

/// `ValueError` for a NaN or infinite value
fn finite(name: &str, v: f64) -> PyResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite, got {v}"
        )))
    }
}

/// One value for a quantity that has two spellings; giving both is a
/// `TypeError` rather than a silent choice of one of them. The flag is true
/// when the value came from the first spelling.
fn one_of(
    a: Option<f64>,
    aname: &str,
    b: Option<f64>,
    bname: &str,
) -> PyResult<Option<(f64, bool)>> {
    match (a, b) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "give {aname} or {bname}, not both"
        ))),
        (Some(v), None) => Ok(Some((v, true))),
        (None, Some(v)) => Ok(Some((v, false))),
        (None, None) => Ok(None),
    }
}

/// The ITRF coordinate of a geodetic position given by keywords, with the
/// checks shared by the keyword constructor and `from_geodetic`: exactly one
/// latitude and one longitude (`TypeError` otherwise), latitude within
/// [-90, 90] degrees and finite longitude and height (`ValueError`).
/// `hae` is the height and the keyword it came from (0 when not given).
fn geodetic_keywords(
    latitude_deg: Option<f64>,
    longitude_deg: Option<f64>,
    latitude_rad: Option<f64>,
    longitude_rad: Option<f64>,
    hae: Option<(f64, &str)>,
) -> PyResult<ITRFCoord> {
    use pyo3::exceptions::{PyTypeError, PyValueError};
    let lat = one_of(latitude_deg, "latitude_deg", latitude_rad, "latitude_rad")?;
    let lon = one_of(
        longitude_deg,
        "longitude_deg",
        longitude_rad,
        "longitude_rad",
    )?;
    let (Some((lat, lat_is_deg)), Some((lon, lon_is_deg))) = (lat, lon) else {
        return Err(PyTypeError::new_err(
            "a geodetic position needs a latitude (latitude_deg or latitude_rad) \
             and a longitude (longitude_deg or longitude_rad)",
        ));
    };
    // The range is checked in the unit given, so that exactly 90
    // degrees is accepted whatever its radian rounding
    let (lat_name, lat_limit) = if lat_is_deg {
        ("latitude_deg", 90.0)
    } else {
        ("latitude_rad", std::f64::consts::FRAC_PI_2)
    };
    if lat.is_nan() || lat.abs() > lat_limit {
        return Err(PyValueError::new_err(format!(
            "{lat_name} must be in [-{lat_limit}, {lat_limit}], got {lat}"
        )));
    }
    let lon = finite(
        if lon_is_deg {
            "longitude_deg"
        } else {
            "longitude_rad"
        },
        lon,
    )?;
    let hae = match hae {
        Some((h, name)) => finite(name, h)?,
        None => 0.0,
    };
    let lat = if lat_is_deg { lat.to_radians() } else { lat };
    let lon = if lon_is_deg { lon.to_radians() } else { lon };
    Ok(ITRFCoord::from_geodetic_rad(lat, lon, hae))
}

/// The ITRF coordinate of a Cartesian position: `TypeError` for a
/// non-numeric input, `ValueError` for a wrong shape or a NaN or infinity
fn cartesian_vector(obj: &Bound<'_, PyAny>) -> PyResult<ITRFCoord> {
    let v = to_vector3(obj, "the ITRF position")?;
    for x in v.iter() {
        finite("ITRF position component", *x)?;
    }
    Ok(ITRFCoord::from_vector(&v))
}

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
/// Give either the Cartesian position, positionally (one 3-element
/// array-like, or three numbers ``x, y, z``), or the geodetic position, by
/// keyword (a latitude and a longitude, each in degrees or in radians, and
/// optionally a height). The two forms cannot be mixed. The named
/// constructors ``itrfcoord.from_vector`` and ``itrfcoord.from_geodetic`` do
/// the same.
///
/// Args:
///     vec (array-like, optional): ITRF Cartesian location in meters: any real
///         numeric 3-element array-like (numpy array, list or tuple)
///
/// Keyword Args:
///     latitude_deg (float, optional): Geodetic latitude in degrees, in [-90, 90]
///     longitude_deg (float, optional): Longitude in degrees
///     latitude_rad (float, optional): Geodetic latitude in radians, in [-pi/2, pi/2]
///     longitude_rad (float, optional): Longitude in radians
///     altitude (float, optional): Height above ellipsoid, meters. Default is 0
///     height (float, optional): Height above ellipsoid, meters (alias of ``altitude``)
///
///
/// Returns:
///     itrfcoord: New ITRF coordinate
///
/// Raises:
///     TypeError: no position given; positional and keyword forms mixed; the
///         same quantity given twice (``latitude_deg`` with ``latitude_rad``,
///         ``longitude_deg`` with ``longitude_rad``, ``altitude`` with
///         ``height``); latitude or longitude missing; the positional
///         argument is not a real numeric array-like or three numbers; a
///         keyword that is not a real number
///     ValueError: latitude outside [-90, 90] degrees; a NaN or infinite
///         latitude, longitude, height or Cartesian component; a positional
///         array that does not have 3 elements
///
/// Example:
///     * Create ITRF coord from Cartesian
///        >>> coord = satkit.itrfcoord([1523128.63570828, -4461395.28873207, 4281865.94218203])
///        >>> print(coord)
///        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
///
/// Example:
///     * Create ITRF coord from Geodetic
///        >>> coord = satkit.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
///        >>> print(coord)
///        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
///
///
#[pyclass(name = "itrfcoord", module = "satkit", from_py_object)]
#[derive(Clone)]
pub struct PyITRFCoord(pub ITRFCoord);

#[pymethods]
impl PyITRFCoord {
    #[new]
    #[pyo3(signature=(
        *args,
        latitude_deg=None,
        longitude_deg=None,
        latitude_rad=None,
        longitude_rad=None,
        altitude=None,
        height=None,
    ))]
    fn new(
        args: &Bound<'_, PyTuple>,
        #[pyo3(from_py_with = latitude_deg_arg)] latitude_deg: Option<f64>,
        #[pyo3(from_py_with = longitude_deg_arg)] longitude_deg: Option<f64>,
        #[pyo3(from_py_with = latitude_rad_arg)] latitude_rad: Option<f64>,
        #[pyo3(from_py_with = longitude_rad_arg)] longitude_rad: Option<f64>,
        #[pyo3(from_py_with = altitude_arg)] altitude: Option<f64>,
        #[pyo3(from_py_with = height_arg)] height: Option<f64>,
    ) -> PyResult<Self> {
        use pyo3::exceptions::PyTypeError;

        let geodetic = [
            latitude_deg,
            longitude_deg,
            latitude_rad,
            longitude_rad,
            altitude,
            height,
        ];
        if geodetic.iter().any(Option::is_some) {
            if !args.is_empty() {
                return Err(PyTypeError::new_err(
                    "give either a positional Cartesian position or geodetic keywords, not both",
                ));
            }
            let hae = one_of(altitude, "altitude", height, "height")?
                .map(|(h, is_altitude)| (h, if is_altitude { "altitude" } else { "height" }));
            return geodetic_keywords(
                latitude_deg,
                longitude_deg,
                latitude_rad,
                longitude_rad,
                hae,
            )
            .map(Self);
        }

        match args.len() {
            // One array-like: any real numeric 3-element sequence
            1 => cartesian_vector(&args.get_item(0)?).map(Self),
            3 => {
                let mut v = [0.0; 3];
                for (i, x) in v.iter_mut().enumerate() {
                    *x = args.get_item(i)?.extract::<f64>()?;
                }
                for x in &v {
                    finite("ITRF position component", *x)?;
                }
                Ok(Self(ITRFCoord::from_vector(&numeris::vector![
                    v[0], v[1], v[2]
                ])))
            }
            n => Err(PyTypeError::new_err(format!(
                "itrfcoord takes one 3-element array-like, three numbers (x, y, z) \
                 or geodetic keywords; got {n} positional arguments"
            ))),
        }
    }

    /// Create an ITRF coordinate from a geodetic position
    ///
    /// The named-constructor form of ``itrfcoord(latitude_deg=...,
    /// longitude_deg=..., altitude=...)``, with the same checks. Give a
    /// latitude and a longitude, each in degrees or in radians, and
    /// optionally the height; all arguments are keyword-only.
    ///
    /// Keyword Args:
    ///     latitude_deg (float): Geodetic latitude in degrees, in [-90, 90]
    ///     longitude_deg (float): Longitude in degrees
    ///     latitude_rad (float): Geodetic latitude in radians, in [-pi/2, pi/2]
    ///     longitude_rad (float): Longitude in radians
    ///     height (float, optional): Height above the WGS84 ellipsoid, meters.
    ///         Default is 0
    ///
    /// Returns:
    ///     itrfcoord: New ITRF coordinate
    ///
    /// Raises:
    ///     TypeError: latitude or longitude missing; the same quantity given
    ///         in both units (``latitude_deg`` with ``latitude_rad``,
    ///         ``longitude_deg`` with ``longitude_rad``); an argument that is
    ///         not a real number
    ///     ValueError: latitude outside [-90, 90] degrees; a NaN or infinite
    ///         latitude, longitude or height
    ///
    /// Example:
    ///     >>> coord = satkit.itrfcoord.from_geodetic(latitude_deg=42.44, longitude_deg=-71.15, height=100)
    ///     >>> print(coord)
    ///     ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
    #[staticmethod]
    #[pyo3(signature = (*, latitude_deg=None, longitude_deg=None, latitude_rad=None, longitude_rad=None, height=None))]
    fn from_geodetic(
        #[pyo3(from_py_with = latitude_deg_arg)] latitude_deg: Option<f64>,
        #[pyo3(from_py_with = longitude_deg_arg)] longitude_deg: Option<f64>,
        #[pyo3(from_py_with = latitude_rad_arg)] latitude_rad: Option<f64>,
        #[pyo3(from_py_with = longitude_rad_arg)] longitude_rad: Option<f64>,
        #[pyo3(from_py_with = height_arg)] height: Option<f64>,
    ) -> PyResult<Self> {
        geodetic_keywords(
            latitude_deg,
            longitude_deg,
            latitude_rad,
            longitude_rad,
            height.map(|h| (h, "height")),
        )
        .map(Self)
    }

    /// Create an ITRF coordinate from a Cartesian position
    ///
    /// The named-constructor form of ``itrfcoord(vec)``; the inverse of
    /// the ``vector`` property.
    ///
    /// Args:
    ///     vec (array-like): ITRF Cartesian position in meters: any real
    ///         numeric 3-element array-like (numpy array, list or tuple)
    ///
    /// Returns:
    ///     itrfcoord: New ITRF coordinate
    ///
    /// Raises:
    ///     TypeError: ``vec`` is not a real numeric array-like
    ///     ValueError: ``vec`` does not have 3 elements, or holds a NaN or
    ///         infinity
    ///
    /// Example:
    ///     >>> coord = satkit.itrfcoord.from_vector([1523128.63570828, -4461395.28873207, 4281865.94218203])
    ///     >>> print(coord)
    ///     ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
    #[staticmethod]
    fn from_vector(vec: &Bound<'_, PyAny>) -> PyResult<Self> {
        cartesian_vector(vec).map(Self)
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

    /// Geodetic coordinates as a named struct
    ///
    /// Returns:
    ///     satkit.geodetic: Geodetic struct with latitude_rad, longitude_rad, height_m fields
    ///         and latitude_deg, longitude_deg computed properties
    #[getter]
    fn get_geodetic(&self) -> PyGeodet {
        self.0.to_geodetic().into()
    }

    /// Return vector representing ITRF Cartesian coordinate in meters
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array of floats representing ITRF Cartesian location in meters
    #[getter]
    fn get_vector(&self, py: Python) -> PyResult<Py<PyAny>> {
        vec2py(py, &self.0.itrf)
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

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Quaternion representing rotation from North-East-Down (NED) coordinate frame to International Terrestrial Reference Frame
    /// (ITRF) at this coordinate
    ///
    /// Returns:
    ///     satkit.quaternion: Quaternion representing rotation from NED to ITRF
    #[getter]
    fn get_qned2itrf(&self) -> PyQuaternion {
        self.0.q_ned2itrf().into()
    }

    /// Quaternion representing rotation from East-North-Up (ENU) coordinate frame to International Terrestrial Reference Frame
    /// (ITRF) at this coordinate
    ///
    /// Returns:
    ///     satkit.quaternion: Quaternion representing rotation from ENU to ITRF
    #[getter]
    fn get_qenu2itrf(&self) -> PyQuaternion {
        self.0.q_enu2itrf().into()
    }

    /// East-North-Up (ENU) vector from ``origin`` to ``self``, in ``origin``'s local-tangent frame.
    ///
    /// The ENU triad has its origin at ``origin``; ``self`` is the point being
    /// located. The ``Up`` component is positive when ``self`` is above
    /// ``origin`` along ``origin``'s local normal (further from Earth's center) —
    /// i.e. "what direction is ``self`` from where I'm standing at ``origin``?"
    ///
    /// Args:
    ///     origin (itrfcoord): ITRF coordinate at which the ENU frame is anchored
    ///         (the observer / station / base of the local tangent plane).
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element ``[E, N, U]`` vector from ``origin`` to
    ///     ``self``, in meters.
    ///
    /// Example:
    ///     >>> station = satkit.itrfcoord(latitude_deg=42.466, longitude_deg=-71.1516, altitude=0)
    ///     >>> satellite = satkit.itrfcoord(latitude_deg=42.466, longitude_deg=-71.1516, altitude=400_000)
    ///     >>> satellite.to_enu(station)  # satellite is overhead → Up ≈ +400_000
    ///     array([0., 0., 400000.])
    fn to_enu(&self, py: Python, origin: &Self) -> PyResult<Py<PyAny>> {
        vec2py(
            py,
            &(origin.0.q_enu2itrf().conjugate() * (self.0.itrf - origin.0.itrf)),
        )
    }

    /// North-East-Down (NED) vector from ``origin`` to ``self``, in ``origin``'s local-tangent frame.
    ///
    /// The NED triad has its origin at ``origin``; ``self`` is the point being
    /// located. The ``Down`` component is positive when ``self`` is below
    /// ``origin`` along ``origin``'s local normal (closer to Earth's center).
    ///
    /// Args:
    ///     origin (itrfcoord): ITRF coordinate at which the NED frame is anchored
    ///         (the observer / station / base of the local tangent plane).
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element ``[N, E, D]`` vector from ``origin`` to
    ///     ``self``, in meters.
    fn to_ned(&self, py: Python, origin: &Self) -> PyResult<Py<PyAny>> {
        vec2py(
            py,
            &(origin.0.q_ned2itrf().conjugate() * (self.0.itrf - origin.0.itrf)),
        )
    }

    /// Geodesic (shortest path on the WGS-84 ellipsoid) from this coordinate
    /// to another: distance and headings
    ///
    /// Notes:
    ///     * Both points are projected onto the ellipsoid: heights are ignored
    ///     * Vincenty's inverse formula, iterated to convergence:
    ///       https://en.wikipedia.org/wiki/Vincenty%27s_formulae. For nearly
    ///       antipodal points, where that iteration does not converge, the
    ///       same series are solved for the initial heading by bisection.
    ///       Accurate to better than a millimeter for any pair of points
    ///     * Headings are clockwise from north, in (-pi, pi]. The heading at
    ///       the destination is the direction of travel on arrival (add pi
    ///       for the back azimuth)
    ///     * For coincident points all three values are 0. Between exactly
    ///       antipodal points, or from a pole, the shortest path is not
    ///       unique: the distance is well defined, the headings are those of
    ///       one of the shortest paths
    ///
    /// Args:
    ///     other (itrfcoord): ITRF coordinate for which to compute distance
    ///
    /// Returns:
    ///     tuple: (distance in meters, initial heading in radians, heading at destination in radians)
    ///
    fn geodesic_distance(&self, other: &Self) -> (f64, f64, f64) {
        self.0.geodesic_distance(&other.0)
    }

    /// Geodesic distance in meters between this coordinate and another
    ///
    /// Convenience wrapper around `geodesic_distance` that returns only the
    /// distance (shortest distance along the WGS-84 ellipsoid; heights are
    /// ignored).
    ///
    /// Args:
    ///     other (itrfcoord): ITRF coordinate to measure distance to
    ///
    /// Returns:
    ///     float: Distance in meters
    fn distance_to(&self, other: &Self) -> f64 {
        self.0.distance_to(&other.0)
    }

    /// Move this coordinate along a given heading by a given distance
    ///
    /// Notes:
    ///     * Heading is in radians, where 0 is North, pi/2 is East, pi is South, 3pi/2 is West
    ///     * Distance is in meters
    ///     * Uses Vincenty's direct formula to compute new location
    ///     * The path follows the ellipsoid: the starting height is ignored
    ///       and the result has zero height
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
        let [x, y, z] = crate::pyutils::unpack_f64s(py, &s)?;
        self.0.itrf = numeris::vector![x, y, z];
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::pyutils::pack_f64s(py, &[self.0.itrf[0], self.0.itrf[1], self.0.itrf[2]])
    }

    /// 3-vector representing cartesian distance between this
    /// and other point, in meters
    fn __sub__(&self, py: Python, other: &Self) -> PyResult<Py<PyAny>> {
        vec2py(py, &(self.0 - other.0))
    }
}
