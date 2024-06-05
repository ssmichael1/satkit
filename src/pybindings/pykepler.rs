use numpy as np;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;

use nalgebra::Vector3;
type Vec3 = Vector3<f64>;

use crate::kepler::{Anomaly, Kepler};

use super::pyduration::PyDuration;
use super::pyutils::kwargs_or_none;

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
pub struct PyKepler {
    pub inner: Kepler,
}

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
        Ok(PyKepler {
            inner: Kepler::new(a, e, i, raan, w, an),
        })
    }

    #[getter]
    /// Semi-major axis, meters
    fn get_a(&self) -> f64 {
        self.inner.a
    }

    #[getter]
    /// Eccentricity
    fn get_eccen(&self) -> f64 {
        self.inner.eccen
    }

    #[getter]
    /// Inclination, radians
    fn get_inclination(&self) -> f64 {
        self.inner.incl
    }

    #[getter]
    /// Right Ascension of the Ascending Node, radians
    fn get_raan(&self) -> f64 {
        self.inner.raan
    }

    #[getter]
    /// Argument of Perigee, radians
    fn get_w(&self) -> f64 {
        self.inner.w
    }

    #[getter]
    /// True Anomaly, radians
    fn get_nu(&self) -> f64 {
        self.inner.nu
    }

    /// Convert Keplerian elements to Cartesian
    /// position (meters) and velocity (meters/second)
    fn to_pv(&self) -> (PyObject, PyObject) {
        let (r, v) = self.inner.to_pv();
        pyo3::Python::with_gil(|py| -> (PyObject, PyObject) {
            (
                numpy::PyArray::from_slice_bound(py, r.as_slice()).to_object(py),
                numpy::PyArray::from_slice_bound(py, v.as_slice()).to_object(py),
            )
        })
    }

    /// Convert Cartesian elements to kepler
    #[staticmethod]
    fn from_pv(r: np::PyReadonlyArray1<f64>, v: np::PyReadonlyArray1<f64>) -> PyResult<Self> {
        let r = Vec3::from_row_slice(r.as_slice().unwrap());
        let v = Vec3::from_row_slice(v.as_slice().unwrap());
        Ok(PyKepler {
            inner: Kepler::from_pv(r, v).unwrap(),
        })
    }

    #[staticmethod]
    fn propagate(k: &PyKepler, dt: &Bound<'_, PyAny>) -> PyResult<PyKepler> {
        if dt.is_instance_of::<pyo3::types::PyFloat>() {
            let dt = dt.extract::<f64>()?;
            let dt = crate::Duration::Seconds(dt);
            Ok(PyKepler {
                inner: k.inner.propagate(&dt),
            })
        } else {
            let dt: PyDuration = dt.extract()?;
            Ok(PyKepler {
                inner: k.inner.propagate(&dt.inner),
            })
        }
    }

    /// Return the eccentric anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Eccentric Anomaly, radians
    #[getter]
    fn eccentric_anomaly(&self) -> f64 {
        self.inner.eccentric_anomaly()
    }

    /// Return the mean motion of the satellite in radians/second
    ///
    /// Returns:
    ///    float: Mean motion, radians/second
    #[getter]
    fn mean_motion(&self) -> f64 {
        self.inner.mean_motion()
    }

    /// Return the period of the satellite in seconds
    ///
    /// Returns:
    ///   float: Period, seconds
    #[getter]
    fn period(&self) -> f64 {
        self.inner.period()
    }

    /// Return the mean anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Mean Anomaly, radians
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.inner.mean_anomaly()
    }

    /// Return the true anomaly of the satellite in radians
    ///
    /// Returns:
    ///   float: True Anomaly, radians
    #[getter]
    fn true_anomaly(&self) -> f64 {
        self.inner.nu
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        let mut state = [0 as u8; 48];
        state[0..8].clone_from_slice(&self.inner.a.to_le_bytes());
        state[8..16].clone_from_slice(&self.inner.eccen.to_le_bytes());
        state[16..24].clone_from_slice(&self.inner.incl.to_le_bytes());
        state[24..32].clone_from_slice(&self.inner.raan.to_le_bytes());
        state[32..40].clone_from_slice(&self.inner.w.to_le_bytes());
        state[40..48].clone_from_slice(&self.inner.nu.to_le_bytes());
        Ok(PyBytes::new_bound(py, &state).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let state = state.extract::<&[u8]>(py)?;
        self.inner.a = f64::from_le_bytes(state[0..8].try_into().unwrap());
        self.inner.eccen = f64::from_le_bytes(state[8..16].try_into().unwrap());
        self.inner.incl = f64::from_le_bytes(state[16..24].try_into().unwrap());
        self.inner.raan = f64::from_le_bytes(state[24..32].try_into().unwrap());
        self.inner.w = f64::from_le_bytes(state[32..40].try_into().unwrap());
        self.inner.nu = f64::from_le_bytes(state[40..48].try_into().unwrap());
        Ok(())
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new_bound(py);
        let tp = PyTuple::new_bound(py, vec![6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        (tp, d)
    }
}
