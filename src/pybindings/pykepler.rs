use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy as np;

use nalgebra::Vector3;
type Vec3 = Vector3<f64>;

use crate::kepler::Kepler;

///
/// Representation of Keplerian orbital elements
/// 
#[pyclass(name="kepler")]
#[derive(Clone)]
pub struct PyKepler {
    pub inner: Kepler,
}

#[pymethods]
impl PyKepler {
    #[new]
    #[pyo3(signature=(*args))]
    fn new(args: &Bound<PyTuple>) -> PyResult<Self> {
        let a = args.get_item(0)?.extract::<f64>().unwrap();
        let e = args.get_item(1)?.extract::<f64>().unwrap();
        let i = args.get_item(2)?.extract::<f64>().unwrap();
        let raan = args.get_item(3)?.extract::<f64>().unwrap();
        let w = args.get_item(4)?.extract::<f64>().unwrap();
        let nu = args.get_item(5)?.extract::<f64>().unwrap();
        Ok(PyKepler {
            inner: Kepler::new(a, e, i, raan, w, nu),
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
            inner: Kepler::from_pv(r, v),
        })
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }


}