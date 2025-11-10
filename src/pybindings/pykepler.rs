use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::kepler::{Anomaly, Kepler};

use super::pyduration::PyDuration;
use super::pyutils::kwargs_or_none;
use super::pyutils::py_to_smatrix;

///
/// Representation of Keplerian orbital elements
///
/// Note: True anomaly can be specified as a positional argument or
/// anomalies of different types can be specified as keyword arguments
///
/// If keyword argument is used, the positional argument should be left out
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
///      eccentric_anomaly: Eccentric Anomaly, radians
///      mean_anomaly: Mean Anomaly, radians
///
/// Returns:
///     Kepler: Keplerian orbital elements
///
#[pyclass(name = "kepler", module = "satkit")]
#[derive(Clone)]
pub struct PyKepler(pub Kepler);

#[pymethods]
impl PyKepler {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &Bound<PyTuple>, mut kwargs: Option<&Bound<PyDict>>) -> PyResult<Self> {
        let a = args.get_item(0)?.extract::<f64>().unwrap();
        let e = args.get_item(1)?.extract::<f64>().unwrap();
        let i = args.get_item(2)?.extract::<f64>().unwrap();
        let raan = args.get_item(3)?.extract::<f64>().unwrap();
        let w = args.get_item(4)?.extract::<f64>().unwrap();

        let nu: Option<f64>;
        let mut ea: Option<f64> = None;
        let mut ma: Option<f64> = None;
        if args.len() > 5 {
            nu = Some(args.get_item(5)?.extract::<f64>().unwrap());
        } else {
            nu = kwargs_or_none(&mut kwargs, "true_anomaly")?;
            ea = kwargs_or_none(&mut kwargs, "eccentric_anomaly")?;
            ma = kwargs_or_none(&mut kwargs, "mean_anomaly")?;
        }
        let an = match (nu, ea, ma) {
            (Some(v), None, None) => Anomaly::True(v),
            (None, Some(v), None) => Anomaly::Eccentric(v),
            (None, None, Some(v)) => Anomaly::Mean(v),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify only one of true_anomaly, eccentric_anomaly, or mean_anomaly",
                ))
            }
        };
        Ok(Self(Kepler::new(a, e, i, raan, w, an)))
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
    fn from_pv(r: &Bound<PyAny>, v: &Bound<PyAny>) -> PyResult<Self> {
        let r = py_to_smatrix(r)?;
        let v = py_to_smatrix(v)?;
        match Kepler::from_pv(r, v) {
            Ok(k) => Ok(Self(k)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    fn propagate(&self, dt: &Bound<'_, PyAny>) -> PyResult<Self> {
        if dt.is_instance_of::<pyo3::types::PyFloat>() {
            let dt = dt.extract::<f64>()?;
            let dt = crate::Duration::from_seconds(dt);
            Ok(Self(self.0.propagate(&dt)))
        } else {
            let dt: PyDuration = dt.extract()?;
            Ok(Self(self.0.propagate(&dt.0)))
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
        // Convert eccentric anomaly to true anomaly
        self.0.nu = f64::atan2(
            val.sin() * self.0.eccen.mul_add(-self.0.eccen, 1.0).sqrt(),
            val.cos() - self.0.eccen,
        );
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
        // Convert mean anomaly to true anomaly via eccentric anomaly
        // First convert mean to eccentric
        use std::f64::consts::PI;
        let mut ea = match (val > PI) || ((val < 0.0) && (val > -PI)) {
            true => val - self.0.eccen,
            false => val + self.0.eccen,
        };
        loop {
            let de =
                self.0.eccen.mul_add(ea.sin(), val - ea) / self.0.eccen.mul_add(-ea.cos(), 1.0);
            ea += de;
            if de.abs() < 1.0e-6 {
                break;
            }
        }
        // Then convert eccentric to true
        self.0.nu = f64::atan2(
            ea.sin() * self.0.eccen.mul_add(-self.0.eccen, 1.0).sqrt(),
            ea.cos() - self.0.eccen,
        );
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

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let mut state = [0; 48];
        state[0..8].clone_from_slice(&self.0.a.to_le_bytes());
        state[8..16].clone_from_slice(&self.0.eccen.to_le_bytes());
        state[16..24].clone_from_slice(&self.0.incl.to_le_bytes());
        state[24..32].clone_from_slice(&self.0.raan.to_le_bytes());
        state[32..40].clone_from_slice(&self.0.w.to_le_bytes());
        state[40..48].clone_from_slice(&self.0.nu.to_le_bytes());
        PyBytes::new(py, &state).into_py_any(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let state = state.extract::<&[u8]>(py)?;
        self.0.a = f64::from_le_bytes(state[0..8].try_into().unwrap());
        self.0.eccen = f64::from_le_bytes(state[8..16].try_into().unwrap());
        self.0.incl = f64::from_le_bytes(state[16..24].try_into().unwrap());
        self.0.raan = f64::from_le_bytes(state[24..32].try_into().unwrap());
        self.0.w = f64::from_le_bytes(state[32..40].try_into().unwrap());
        self.0.nu = f64::from_le_bytes(state[40..48].try_into().unwrap());
        Ok(())
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::new(py, vec![6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        (tp, d)
    }
}
