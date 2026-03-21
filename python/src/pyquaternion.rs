use anyhow::Context;
use numpy as np;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;

use anyhow::{bail, Result};

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
#[pyclass(name = "quaternion", module = "satkit", from_py_object)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct PyQuaternion(pub Quaternion);

#[pyclass(name = "quaternion_array", module = "satkit", from_py_object)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyQuaternionVec(Vec<Quaternion>);

impl From<Quaternion> for PyQuaternion {
    fn from(q: Quaternion) -> Self {
        Self(q)
    }
}

#[pymethods]
impl PyQuaternion {
    #[new]
    #[pyo3(signature=(*args))]
    fn py_new(args: &Bound<'_, PyTuple>) -> Result<Self> {
        if args.len() == 0 {
            Ok(Quaternion::identity().into())
        } else if args.len() == 4 {
            let w = args.get_item(0)?.extract::<f64>()?;
            let x = args.get_item(1)?.extract::<f64>()?;
            let y = args.get_item(2)?.extract::<f64>()?;
            let z = args.get_item(3)?.extract::<f64>()?;
            Ok(Quaternion::new(w, x, y, z).into())
        } else {
            bail!("Invalid input.  Must be empty or 4 floats");
        }
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
    fn rotx(theta_rad: f64) -> Result<Self> {
        Ok(Quaternion::rotx(theta_rad).into())
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
    fn roty(theta_rad: f64) -> Result<Self> {
        Ok(Quaternion::roty(theta_rad).into())
    }

    /// Quaternion representing rotation about
    /// zhat axis by `theta-rad` degrees
    #[staticmethod]
    fn rotz(theta_rad: f64) -> Result<Self> {
        Ok(Quaternion::rotz(theta_rad).into())
    }

    /// Quaternion representing rotation about given axis by given angle in radians
    ///
    /// Args:
    ///     axis (numpy.ndarray): 3-element numpy array representing axis about which to rotate (does not need to be normalized)
    ///     angle (float): Angle in radians to rotate about axis (right-handed rotation of vector)
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about given axis by given angle.  If axis norm is < 1e-9,
    ///     unit quaternion is returned
    ///
    #[staticmethod]
    fn from_axis_angle(axis: np::PyReadonlyArray1<f64>, angle: f64) -> Result<Self> {
        let s = axis.as_slice()?;
        let v = Vector3::from_array([s[0], s[1], s[2]]);
        let n = v.norm();
        if n < 1.0e-9 {
            // If the axis is zero, return identity quaternion
            Ok(Quaternion::identity().into())
        } else {
            Ok(Quaternion::from_axis_angle(v, angle).into())
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
    ) -> Result<Self> {
        if v1.len() != 3 || v2.len() != 3 {
            bail!("Invalid input.  Must be two 3-element vectors");
        }
        let v1 = match v1.is_contiguous() {
            true => {
                let s = v1.as_slice().context("Cannot convert v1 to 3D vector")?;
                Vector3::from_array([s[0], s[1], s[2]])
            }
            false => Vector3::from_array([
                *v1.get(0).unwrap(),
                *v1.get(1).unwrap(),
                *v1.get(2).unwrap(),
            ]),
        };
        let v2 = match v2.is_contiguous() {
            true => {
                let s = v2.as_slice().context("Cannot convert v2 to 3D vector")?;
                Vector3::from_array([s[0], s[1], s[2]])
            }
            false => Vector3::from_array([
                *v2.get(0).unwrap(),
                *v2.get(1).unwrap(),
                *v2.get(2).unwrap(),
            ]),
        };

        // Compute rotation between two vectors
        let n1 = v1.norm();
        let n2 = v2.norm();
        if n1 < 1.0e-9 || n2 < 1.0e-9 {
            bail!("Norms are 0 or vectors are 180° apart");
        }
        let u1 = v1 * (1.0 / n1);
        let u2 = v2 * (1.0 / n2);
        let cross = u1.cross(&u2);
        let dot = u1.dot(&u2);
        if cross.norm() < 1.0e-9 && dot < 0.0 {
            bail!("Norms are 0 or vectors are 180° apart");
        }
        let q = Quaternion::from_axis_angle(cross, dot.clamp(-1.0, 1.0).acos());
        Ok(q.into())
    }

    /// Return quaternion representing same rotation as input direction cosine matrix (3x3 rotation matrix)
    ///
    /// Args:
    ///     dcm (numpy.ndarray): 3x3 numpy array representing rotation matrix
    ///
    /// Returns:
    ///     quaternion: Quaternion representing same rotation as input matrix
    #[staticmethod]
    fn from_rotation_matrix(dcm: np::PyReadonlyArray2<f64>) -> Result<Self> {
        if dcm.dims() != [3, 3] {
            bail!("Invalid DCM.  Must be 3x3 matrix");
        }
        let dcm = dcm.as_array();
        // numpy arrays are row-major, build Matrix3 row-by-row
        let mat = Matrix3::new([
            [dcm[(0, 0)], dcm[(0, 1)], dcm[(0, 2)]],
            [dcm[(1, 0)], dcm[(1, 1)], dcm[(1, 2)]],
            [dcm[(2, 0)], dcm[(2, 1)], dcm[(2, 2)]],
        ]);
        Ok(Quaternion::from_rotation_matrix(&mat).into())
    }

    /// Return rotation matrix representing identical rotation to quaternion
    ///
    /// Returns:
    ///     numpy.ndarray: 3x3 numpy array representing rotation matrix
    fn as_rotation_matrix(&self) -> Py<PyAny> {
        let rot = self.0.to_rotation_matrix();

        pyo3::Python::attach(|py| -> Py<PyAny> {
            let phi = unsafe { np::PyArray2::<f64>::new(py, [3, 3], true) };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rot.as_slice().as_ptr(),
                    phi.as_raw_array_mut().as_mut_ptr(),
                    9,
                );
            }
            phi.into_py_any(py).unwrap()
        })
    }

    ///Return rotation represented as "roll", "pitch", "yaw" euler angles in radians.
    ///
    /// Returns:
    ///     (f64, f64, f64): Tuple of roll, pitch, yaw angles in radians
    fn as_euler(&self) -> (f64, f64, f64) {
        self.0.to_euler()
    }

    fn __str__(&self) -> Result<String> {
        let (ax, angle) = self.0.to_axis_angle();
        let n = ax.norm();
        let ax = if n < 1.0e-9 {
            Vector3::from_array([1.0, 0.0, 0.0])
        } else {
            ax * (1.0 / n)
        };
        Ok(format!(
            "Quaternion(Axis = [{:6.4}, {:6.4}, {:6.4}], Angle = {:6.4} rad)",
            ax[0], ax[1], ax[2], angle
        ))
    }

    fn __repr__(&self) -> Result<String> {
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
        self.0 = Quaternion::new(w, x, y, z);
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut raw = [0; 32];
        raw[0..8].clone_from_slice(f64::to_le_bytes(self.0.w).as_slice());
        raw[8..16].clone_from_slice(f64::to_le_bytes(self.0.x).as_slice());
        raw[16..24].clone_from_slice(f64::to_le_bytes(self.0.y).as_slice());
        raw[24..32].clone_from_slice(f64::to_le_bytes(self.0.z).as_slice());
        PyBytes::new(py, &raw).into_py_any(py)
    }

    /// Angle of rotation in radians
    ///
    /// Returns:
    ///     float: Angle of rotation in radians
    #[getter]
    fn angle(&self) -> f64 {
        self.0.to_axis_angle().1
    }

    /// Axis of rotation
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array representing axis of rotation
    #[getter]
    fn axis(&self) -> PyResult<Py<PyAny>> {
        let (ax, _) = self.0.to_axis_angle();
        let n = ax.norm();
        let a = if n < 1.0e-9 {
            Vector3::from_array([1.0, 0.0, 0.0])
        } else {
            ax * (1.0 / n)
        };
        pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
            numpy::ndarray::arr1(a.as_slice())
                .to_pyarray(py)
                .into_py_any(py)
        })
    }

    /// Quaternion representing inverse rotation
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
    #[getter]
    fn conj(&self) -> Self {
        self.0.conjugate().into()
    }

    /// Quaternion representing inverse rotation
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
    #[getter]
    fn conjugate(&self) -> Self {
        self.0.conjugate().into()
    }

    #[getter]
    fn x(&self) -> f64 {
        self.0.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.0.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.0.z
    }

    #[getter]
    fn w(&self) -> f64 {
        self.0.w
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
    #[pyo3(signature=(other, frac, epsilon=1.0e-6))]
    #[allow(unused_variables)]
    fn slerp(&self, other: &Self, frac: f64, epsilon: f64) -> Result<Self> {
        Ok(self.0.slerp(&other.0, frac).into())
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
        // Multiply quaternion by quaternion
        if other.is_instance_of::<Self>() {
            let q: PyRef<Self> = other
                .extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract quaternion: {}", e))?;
            Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                Self(self.0 * q.0).into_py_any(py)
            })?)
        }
        // This incorrectly matches for all PyArray types
        else if let Ok(v) = other.cast::<np::PyArray2<f64>>() {
            if v.dims()[1] != 3 {
                bail!("Invalid rhs.  2nd dimension must be 3 in size");
            }
            let rot = self.0.to_rotation_matrix();
            let qmat = rot.transpose();

            Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                let nd = unsafe { np::ndarray::ArrayView2::from_shape_ptr((3, 3), qmat.as_slice().as_ptr()) };
                let res = v.readonly().as_array().dot(&nd).to_pyarray(py);

                res.into_py_any(py)
            })?)
        } else if let Ok(v1d) = other.cast::<np::PyArray1<f64>>() {
            if v1d.len() != 3 {
                bail!("Invalid rhs.  1D array must be of length 3");
            }

            let m = Vector3::from_array([
                v1d.get_owned(0).unwrap(),
                v1d.get_owned(1).unwrap(),
                v1d.get_owned(2).unwrap(),
            ]);

            let vout = self.0 * m;

            Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                let vnd = np::PyArray1::<f64>::from_vec(py, vec![vout[0], vout[1], vout[2]]);
                vnd.into_py_any(py)
            })?)
        } else {
            bail!("Invalid type: {}", other.get_type());
        }
    }
}
