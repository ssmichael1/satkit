use nalgebra as na;
use numpy as np;
use numpy::PyArrayMethods;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
#[pyclass(name = "quaternion", module = "satkit")]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Quaternion {
    pub inner: Quat,
}

#[pyclass(name = "quaternion_array", module = "satkit")]
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

    /// Quaternion representing rotation about xhat axis by `theta-rad` degrees
    ///
    /// Args:
    ///     theta_rad: Angle in radians to rotate about xhat axis
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about xhat axis
    ///
    /// Notes:
    ///     This is a right-handed rotation of the vector
    ///     e.g. rotation of +xhat 90 degrees by +zhat gives +yhat
    #[staticmethod]
    fn rotx(theta_rad: f64) -> PyResult<Self> {
        Ok(Quaternion {
            inner: Quat::from_axis_angle(&Vec3::x_axis(), theta_rad),
        })
    }

    /// Quaternion representing rotation about yhat axis by `theta-rad` degrees
    ///
    /// Args:
    ///     theta_rad: Angle in radians to rotate about yhat axis
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about yhat axis
    ///
    /// Notes:
    ///     This is a right-handed rotation of the vector
    ///     e.g. rotation of +xhat by +yhat 90 degrees gives -zhat
    ///     
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

    /// Quaternion representing rotation about given axis by given angle in radians
    ///
    /// Args:
    ///     axis (numpy.ndarray): 3-element numpy array representing axis about which to rotate (does not need to be normalized)
    ///     angle (float): Angle in radians to rotate about axis (right-handed rotation of vector)
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about given axis by given angle
    ///     
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

    /// Quaternion representing rotation from V1 to V2
    ///
    /// Args:
    ///     v1 (numpy.ndarray): 3-element numpy array representing vector rotating from
    ///     v2 (numpy.ndarray): 3-element numpy array representing vector rotating to
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation from v1 to v2
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

    /// Return quaternion representing same rotation as input direction cosine matrix (3x3 rotation matrix)
    ///
    /// Args:
    ///     dcm (numpy.ndarray): 3x3 numpy array representing rotation matrix
    ///
    /// Returns:
    ///     quaternion: Quaternion representing same rotation as input matrix
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
        Ok(Quaternion {
            inner: Quat::from_rotation_matrix(&rot),
        })
    }

    /// Return rotation matrix representing identical rotation to quaternion
    ///
    /// Returns:
    ///     numpy.ndarray: 3x3 numpy array representing rotation matrix
    fn to_rotation_matrix(&self) -> PyObject {
        let rot = self.inner.to_rotation_matrix();

        pyo3::Python::with_gil(|py| -> PyObject {
            let phi = unsafe { np::PyArray2::<f64>::new_bound(py, [3, 3], true) };
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

    ///Return rotation represented as "roll", "pitch", "yaw" euler angles in radians.
    ///
    /// Returns:
    ///     (f64, f64, f64): Tuple of roll, pitch, yaw angles in radians
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

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let state = state.as_bytes(py);
        if state.len() != 32 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid serialization length",
            ));
        }
        let w = f64::from_le_bytes(state[0..8].try_into()?);
        let x = f64::from_le_bytes(state[8..16].try_into()?);
        let y = f64::from_le_bytes(state[16..24].try_into()?);
        let z = f64::from_le_bytes(state[24..32].try_into()?);
        self.inner = Quat::from_quaternion(na::Quaternion::<f64>::new(w, x, y, z));
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let mut raw = [0 as u8; 32];
        raw[0..8].clone_from_slice(f64::to_le_bytes(self.inner.w).as_slice());
        raw[8..16].clone_from_slice(f64::to_le_bytes(self.inner.i).as_slice());
        raw[16..24].clone_from_slice(f64::to_le_bytes(self.inner.j).as_slice());
        raw[24..32].clone_from_slice(f64::to_le_bytes(self.inner.k).as_slice());
        Ok(PyBytes::new_bound(py, &raw).to_object(py))
    }

    /// Angle of rotation in radians
    ///
    /// Returns:
    ///     float: Angle of rotation in radians
    #[getter]
    fn angle(&self) -> PyResult<f64> {
        Ok(self.inner.angle())
    }

    /// Axis of rotation
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array representing axis of rotation
    #[getter]
    fn axis(&self) -> PyResult<PyObject> {
        let a = match self.inner.axis() {
            Some(ax) => ax,
            None => Vec3::x_axis(),
        };
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            Ok(numpy::ndarray::arr1(a.as_slice())
                .to_pyarray_bound(py)
                .to_object(py))
        })
    }

    /// Quaternion representing inverse rotation
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
    #[getter]
    fn conj(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    /// Quaternion representing inverse rotation
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
    #[getter]
    fn conjugate(&self) -> PyResult<Quaternion> {
        Ok(Quaternion {
            inner: self.inner.conjugate(),
        })
    }

    /// Spherical linear interpolation between self and other quaternion
    ///
    /// Args:
    ///     other (quaternion): Quaternion to perform interpolation to
    ///     frac (float): Number in range [0,1] representing fractional distance from self to other of result quaternion
    ///     epsilon (float): Value below which the sin of the angle separating both quaternion must be to return an error.  Default is 1.0e-6
    ///
    /// Returns:
    ///     quaternion: Quaterion represention fracional spherical interpolation between self and other    
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

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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
                let res = v.readonly().as_array().dot(&nd).to_pyarray_bound(py);

                Ok(res.into_py(py))
            })
        } else if let Ok(v1d) = other.downcast::<np::PyArray1<f64>>() {
            if v1d.len()? != 3 {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Invalid rhs.  1D array must be of length 3",
                ));
            }

            let m = na::vector![
                v1d.get_owned(0).unwrap(),
                v1d.get_owned(1).unwrap(),
                v1d.get_owned(2).unwrap()
            ];

            let vout = self.inner * m;

            pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let vnd = np::PyArray1::<f64>::from_vec_bound(py, vec![vout[0], vout[1], vout[2]]);
                Ok(vnd.into_py(py))
            })
        } else {
            let s = format!("invalid type: {}", other.get_type());
            Err(pyo3::exceptions::PyTypeError::new_err(s))
        }
    }
}
