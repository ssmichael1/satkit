use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use satkit::kepler::{Anomaly, Kepler};

use crate::pyduration::PyDuration;
use crate::pyutils::py_to_smatrix;

///
/// Representation of Keplerian orbital elements
///
/// The anomaly can be given positionally as the true anomaly (6th argument)
/// or by exactly one of the keyword arguments ``true_anomaly``,
/// ``eccentric_anomaly`` or ``mean_anomaly``.
///
/// Args:
///     a: semi-major axis, meters
///     eccen: Eccentricity
///     incl: Inclination, radians
///     raan: Right Ascension of the Ascending Node, radians
///     w: Argument of Perigee, radians
///     nu: True Anomaly, radians
///
/// Keyword Args:
///     true_anomaly: True Anomaly, radians
///     eccentric_anomaly: Eccentric Anomaly, radians
///     mean_anomaly: Mean Anomaly, radians
///
/// Returns:
///     Kepler: Keplerian orbital elements
///
#[pyclass(name = "kepler", module = "satkit", from_py_object)]
#[derive(Clone)]
pub struct PyKepler(pub Kepler);

impl PyKepler {
    /// The same elements with the in-plane position replaced by `an`,
    /// converted to true anomaly by the core solver.
    fn with_anomaly(&self, an: Anomaly) -> Kepler {
        let k = &self.0;
        Kepler::new(k.a, k.eccen, k.incl, k.raan, k.w, an)
    }
}

#[pymethods]
impl PyKepler {
    #[new]
    #[pyo3(signature = (a, eccen, incl, raan, w, nu=None, *, true_anomaly=None, eccentric_anomaly=None, mean_anomaly=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: f64,
        eccen: f64,
        incl: f64,
        raan: f64,
        w: f64,
        nu: Option<f64>,
        true_anomaly: Option<f64>,
        eccentric_anomaly: Option<f64>,
        mean_anomaly: Option<f64>,
    ) -> PyResult<Self> {
        let an =
            match (nu, true_anomaly, eccentric_anomaly, mean_anomaly) {
                (Some(v), None, None, None) | (None, Some(v), None, None) => Anomaly::True(v),
                (None, None, Some(v), None) => Anomaly::Eccentric(v),
                (None, None, None, Some(v)) => Anomaly::Mean(v),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify exactly one of nu, true_anomaly, eccentric_anomaly, or mean_anomaly",
                )),
            };
        Ok(Self(Kepler::new(a, eccen, incl, raan, w, an)))
    }

    #[getter]
    /// Semi-major axis, meters
    fn get_a(&self) -> f64 {
        self.0.a
    }

    #[setter(a)]
    fn set_a(&mut self, val: f64) {
        self.0.a = val;
    }

    #[getter]
    /// Eccentricity
    fn get_eccen(&self) -> f64 {
        self.0.eccen
    }

    #[setter(eccen)]
    fn set_eccen(&mut self, val: f64) {
        self.0.eccen = val;
    }

    #[getter]
    /// Inclination, radians
    fn get_inclination(&self) -> f64 {
        self.0.incl
    }

    #[setter(inclination)]
    fn set_inclination(&mut self, val: f64) {
        self.0.incl = val;
    }

    #[getter]
    /// Right Ascension of the Ascending Node, radians
    fn get_raan(&self) -> f64 {
        self.0.raan
    }

    #[setter(raan)]
    fn set_raan(&mut self, val: f64) {
        self.0.raan = val;
    }

    #[getter]
    /// Argument of Perigee, radians
    fn get_w(&self) -> f64 {
        self.0.w
    }

    #[setter(w)]
    fn set_w(&mut self, val: f64) {
        self.0.w = val;
    }

    #[getter]
    /// True Anomaly, radians
    fn get_nu(&self) -> f64 {
        self.0.nu
    }

    #[setter(nu)]
    fn set_nu(&mut self, val: f64) {
        self.0.nu = val;
    }

    /// Convert Keplerian elements to Cartesian
    /// position (meters) and velocity (meters/second)
    fn to_pv(&self) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (r, v) = self.0.to_pv();
        pyo3::Python::attach(|py| -> PyResult<(Py<PyAny>, Py<PyAny>)> {
            Ok((
                numpy::PyArray::from_slice(py, r.as_slice()).into_py_any(py)?,
                numpy::PyArray::from_slice(py, v.as_slice()).into_py_any(py)?,
            ))
        })
    }

    /// Convert Cartesian elements to kepler
    #[staticmethod]
    fn from_pv(pos: &Bound<PyAny>, vel: &Bound<PyAny>) -> PyResult<Self> {
        let pos = py_to_smatrix(pos)?;
        let vel = py_to_smatrix(vel)?;
        match Kepler::from_pv(pos, vel) {
            Ok(k) => Ok(Self(k)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Propagate the elements forward (or backward) in time
    ///
    /// Args:
    ///     dt (duration | float | int): time to propagate; a number is
    ///         interpreted as seconds
    ///
    /// Returns:
    ///     kepler: new element set
    fn propagate(&self, dt: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(dt) = dt.extract::<PyDuration>() {
            Ok(Self(self.0.propagate(&dt.0)))
        } else {
            let secs = dt.extract::<f64>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "dt must be a satkit.duration or a number of seconds",
                )
            })?;
            Ok(Self(
                self.0.propagate(&satkit::Duration::from_seconds(secs)),
            ))
        }
    }

    /// Return the eccentric anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Eccentric Anomaly, radians
    #[getter]
    fn eccentric_anomaly(&self) -> f64 {
        self.0.eccentric_anomaly()
    }

    #[setter(eccentric_anomaly)]
    fn set_eccentric_anomaly(&mut self, val: f64) {
        self.0 = self.with_anomaly(Anomaly::Eccentric(val));
    }

    /// Return the mean motion of the satellite in radians/second
    ///
    /// Returns:
    ///    float: Mean motion, radians/second
    #[getter]
    fn mean_motion(&self) -> f64 {
        self.0.mean_motion()
    }

    /// Return the period of the satellite in seconds
    //
    /// Returns:
    ///   float: Period, seconds
    #[getter]
    fn period(&self) -> f64 {
        self.0.period()
    }

    /// Return the semiparameter (semi-latus rectum) of the orbit in meters
    ///
    /// Returns:
    ///     float: Semiparameter p = a (1 - e^2), meters
    #[getter]
    fn semiparameter(&self) -> f64 {
        self.0.semiparameter()
    }

    /// Return the mean anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Mean Anomaly, radians
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly()
    }

    #[setter(mean_anomaly)]
    fn set_mean_anomaly(&mut self, val: f64) {
        // Kepler's equation is solved by the core crate (range-reduced,
        // Danby start, iteration-capped), so NaN or e >= 1 cannot hang here.
        self.0 = self.with_anomaly(Anomaly::Mean(val));
    }

    /// Return the true anomaly of the satellite in radians
    ///
    /// Returns:
    ///   float: True Anomaly, radians
    #[getter]
    fn true_anomaly(&self) -> f64 {
        self.0.nu
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __eq__(&self, other: &Self) -> bool {
        let a = &self.0;
        let b = &other.0;
        a.a == b.a
            && a.eccen == b.eccen
            && a.incl == b.incl
            && a.raan == b.raan
            && a.w == b.w
            && a.nu == b.nu
    }

    fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        crate::pyutils::pack_f64s(
            py,
            &[
                self.0.a,
                self.0.eccen,
                self.0.incl,
                self.0.raan,
                self.0.w,
                self.0.nu,
            ],
        )
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let [a, eccen, incl, raan, w, nu] = crate::pyutils::unpack_f64s(py, &state)?;
        self.0.a = a;
        self.0.eccen = eccen;
        self.0.incl = incl;
        self.0.raan = raan;
        self.0.w = w;
        self.0.nu = nu;
        Ok(())
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::new(py, vec![6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        (tp, d)
    }
}
