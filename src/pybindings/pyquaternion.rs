use nalgebra as na;
use numpy as np;
use numpy::ToPyArray;
use pyo3::prelude::*;

type Quat = na::UnitQuaternion<f64>;
type Vec3 = na::Vector3<f64>;

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
#[pyclass(name = "quaternion")]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Quaternion {
    pub inner: Quat,
}

#[pyclass(name = "quaternion_array")]
#[derive(PartialEq, Clone, Debug)]
pub struct QuaternionVec {
    pub inner: Vec<Quat>,
}

#[pymethods]
impl Quaternion {
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::x_axis(), 0.0),
        })
    }

    /// Quaternion representing rotation about
    /// xhat axis by `theta-rad` degrees
    #[staticmethod]
    fn rotx(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::x_axis(), theta_rad),
        })
    }

    /// Quaternion representing rotation about
    /// yhat axis by `theta-rad` degrees
    #[staticmethod]
    fn roty(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::y_axis(), theta_rad),
        })
    }

    /// Quaternion representing rotation about
    /// zhat axis by `theta-rad` degrees
    #[staticmethod]
    fn rotz(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::z_axis(), theta_rad),
        })
    }

    
    /// Quaternion representing rotation about given axis by
    /// given angle in radians
    ///
    /// # Arguments:
    ///
    /// * `axis` - 3-element numpy array representing axis about which to rotate
    ///            (does not need to be normalized)
    /// 
    /// * 'angle`  - Angle in radians to rotate about axis (right-handed rotation of vector)
    #[staticmethod]
    fn from_axis_angle(axis: np::PyReadonlyArray1<f64>, angle: f64) -> PyResult<Self> {
        let v = Vec3::from_row_slice(axis.as_slice()?);
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
    /// Return quaternion represention rotation from V1 to V2
    ///
    /// # Arguments:
    ///
    /// * `v1` - vector rotating from
    /// * `v2` - vector rotating to
    ///
    /// # Returns:
    ///
    /// * Quaternion that rotates from v1 to v2
    ///
    #[staticmethod]
    fn rotation_between(
        v1: np::PyReadonlyArray1<f64>,
        v2: np::PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let v1 = Vec3::from_row_slice(v1.as_slice()?);
        let v2 = Vec3::from_row_slice(v2.as_slice()?);
        match Quat::rotation_between(&v1, &v2) {
            Some(q) => Ok(Quaternion { inner: q }),
            None => Err(pyo3::exceptions::PyArithmeticError::new_err(
                "Norms are 0 or vectors are 180° apart",
            )),
        }
    }


    /// Return quaternion representing same rotation as input
    /// direction cosine matrix (3x3 rotation matrix)
    /// 
    /// # Arguments:
    /// 
    /// * `dcm` - 3x3 numpy array representing rotation matrix
    /// 
    /// # Returns:
    /// 
    /// * Quaternion representing same rotation as input matrix
    /// 
    #[staticmethod]
    fn from_rotation_matrix(dcm: np::PyReadonlyArray2<f64>) -> PyResult<Self> {
        if dcm.dims() != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid DCM. Must be 3x3",
            ));
        }
        let dcm = dcm.as_array();
        let mat = na::Matrix3::from_iterator(dcm.iter().cloned());
        let rot = na::Rotation3::from_matrix(&mat.transpose());
        Ok(Quaternion{inner: Quat::from_rotation_matrix(&rot) })
    }

    ///    
    /// Return 3x3 rotation matrix (also called direction cosine matrix)
    /// representing rotation identical to this quaternion
    ///
    fn to_rotation_matrix(&self) -> PyObject {
        let rot = self.inner.to_rotation_matrix();

        pyo3::Python::with_gil(|py| -> PyObject {
            let phi = unsafe { np::PyArray2::<f64>::new(py, [3, 3], true) };
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

    ///
    /// Spherical linear interpolation between self and other quaternion
    ///
    /// # Arguments:
    ///
    /// * `other` - Quaternion to perform interpolation to
    /// * `frac` - Number in range [0,1] representing fractional distance
    ///            from self to other of result quaternion
    /// * `epsilon` - Value below which the sin of the angle separating both quaternion must be to return an error.  Default is 1.0e-6
    ///
    /// # Returns:
    ///
    /// * Quaterion represention fracional spherical interpolation between self and other
    ///
    #[pyo3(signature=(other, frac,  epsilon=1.0e-6))]
    fn slerp(&self, other: &Quaternion, frac: f64, epsilon: f64) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: match self.inner.try_slerp(&other.inner, frac, epsilon) {
                Some(v) => v,
                None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Quaternions cannot be 180 deg apart",
                    ))
                }
            },
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
