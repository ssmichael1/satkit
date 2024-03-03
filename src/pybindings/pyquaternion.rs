use nalgebra as na;
use numpy as np;
use numpy::ToPyArray;
use pyo3::prelude::*;

type Quat = na::UnitQuaternion<f64>;
type Vec3 = na::Vector3<f64>;

#[pyclass(name = "quaternion")]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Quaternion {
    pub inner: Quat,
}

///
/// Quaternion representing rotation of 3D Cartesian axes
///
/// Quaternion is right-handed rotation of a vector,
/// e.g. rotation of +xhat 90 degrees by +zhat give +yhat
///
/// This is different than the convention used in Vallado, but
/// it is the way it is commonly used in mathematics and it is
/// the way it should be done.
///
/// For the uninitiated: quaternions are a more-compact and
/// computationally efficient way of representing 3D rotations.  
/// They can also be multipled together and easily renormalized to
/// avoid problems with floating-point precision eventually causing
/// changes in the rotated vecdtor norm.
///
/// For details, see:
///
/// https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
///
///
#[pymethods]
impl Quaternion {
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::x_axis(), 0.0),
        })
    }

    #[staticmethod]
    fn rotx(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::x_axis(), theta_rad),
        })
    }

    #[staticmethod]
    fn roty(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::y_axis(), theta_rad),
        })
    }

    #[staticmethod]
    fn rotz(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::z_axis(), theta_rad),
        })
    }

    #[staticmethod]
    fn from_axis_angle(axis: np::PyReadonlyArray1<f64>, angle: f64) -> PyResult<Self> {
        let v = Vec3::from_row_slice(axis.as_slice().unwrap());
        let u = na::UnitVector3::try_new(v, 1.0e-9);
        match u {
            Some(ax) => Ok(Quaternion {
                inner: Quat::from_axis_angle(&ax, angle),
            }),
            None => Err(pyo3::exceptions::PyArithmeticError::new_err(
                "Axis norm is 0",
            )),
        }
    }

    ///    
    /// Return 3x3 rotation matrix representing rotation
    /// identical to this quaternion
    ///
    fn to_rotation_matrix(&self) -> PyObject {
        let rot = self.inner.to_rotation_matrix();

        pyo3::Python::with_gil(|py| -> PyObject {
            let phi = unsafe { np::PyArray2::<f64>::new(py, [3, 3], false) };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rot.matrix().as_ptr(),
                    phi.as_raw_array_mut().as_mut_ptr(),
                    9,
                );
            }
            phi.to_object(py)
        })
    }

    ///
    /// Return rotation represented as
    /// "roll", "pitch", "yaw" euler angles
    /// in radians.  Return is a tuple
    fn to_euler(&self) -> (f64, f64, f64) {
        self.inner.euler_angles()
    }

    fn __str__(&self) -> PyResult<String> {
        let ax: na::Unit<Vec3> = match self.inner.axis() {
            Some(v) => v,
            None => na::Unit::new_normalize(Vec3::new(1.0, 0.0, 0.0)),
        };
        let angle = self.inner.angle();
        Ok(format!(
            "Quaternion(Axis = [{:6.4}, {:6.4}, {:6.4}], Angle = {:6.4} rad)",
            ax[0], ax[1], ax[2], angle
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }

    #[getter]
    fn angle(&self) -> PyResult<f64> {
        Ok(self.inner.angle())
    }

    #[getter]
    fn axis(&self) -> PyResult<PyObject> {
        let a = match self.inner.axis() {
            Some(ax) => ax,
            None => Vec3::x_axis(),
        };
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            Ok(numpy::ndarray::arr1(a.as_slice())
                .to_pyarray(py)
                .to_object(py))
        })
    }

    #[getter]
    fn conj(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    #[getter]
    fn conjugate(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    fn __mul__(&self, other: &PyAny) -> PyResult<PyObject> {
        // Multiply quaternion by quaternion
        if other.is_instance_of::<Quaternion>() {
            let q: PyRef<Quaternion> = other.extract()?;
            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                return Ok(Quaternion {
                    inner: self.inner * q.inner,
                }
                .into_py(py));
            })
        }
        // This incorrectly matches for all PyArray types
        else if let Ok(v) = other.downcast::<np::PyArray2<f64>>() {
            if v.dims()[1] != 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid rhs. 2nd dimension must be 3 in size",
                ));
            }
            let rot = self.inner.to_rotation_matrix();
            let qmat = rot.matrix().conjugate();

            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let nd = unsafe { np::ndarray::ArrayView2::from_shape_ptr((3, 3), qmat.as_ptr()) };
                let res = v.readonly().as_array().dot(&nd).to_pyarray(py);

                Ok(res.into_py(py))
            })
        } else if let Ok(v1d) = other.downcast::<np::PyArray1<f64>>() {
            if v1d.len() != 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid rhs.  1D array must be of length 3",
                ));
            }

            let storage = unsafe {
                na::ViewStorage::<f64, na::U3, na::U1, na::U1, na::U1>::from_raw_parts(
                    v1d.readonly().as_array().as_ptr(),
                    (na::U3, na::U1),
                    (na::U1, na::U1),
                )
            };
            let vout = self.inner * na::Matrix::from_data(storage);

            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let vnd = np::PyArray1::<f64>::from_vec(py, vec![vout[0], vout[1], vout[2]]);
                Ok(vnd.into_py(py))
            })
        } else {
            let s = format!("invalid type: {}", other.get_type());
            Err(pyo3::exceptions::PyTypeError::new_err(s))
        }
    }
}