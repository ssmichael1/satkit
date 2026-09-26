use anyhow::Context;
use numpy as np;
use numpy::PyArrayMethods;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;

use crate::pyutils::{slice2py2d, to_f64_ndarray, to_vector3, vec2py, warn_deprecated};

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

impl From<Quaternion> for PyQuaternion {
    fn from(q: Quaternion) -> Self {
        Self(q)
    }
}

// The Python API names conversions `to_*` (paired with `from_*`); PyQuaternion
// is Copy, so clippy would rather those took `self` by value. PyO3 methods take
// `&self`, and the name is fixed by the Python API, so the lint is silenced here.
#[allow(clippy::wrong_self_convention)]
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

    /// Quaternion representing rotation about xhat axis by `theta_rad` radians
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

    /// Quaternion representing rotation about yhat axis by `theta_rad` radians
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

    /// Quaternion representing rotation about zhat axis by `theta_rad` radians
    ///
    /// Args:
    ///     theta_rad: Angle in radians to rotate about zhat axis
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about zhat axis
    ///
    /// Notes:
    ///     This is a right-handed rotation of the vector
    ///     e.g. rotation of +xhat 90 degrees by +zhat gives +yhat
    #[staticmethod]
    fn rotz(theta_rad: f64) -> Result<Self> {
        Ok(Quaternion::rotz(theta_rad).into())
    }

    /// Quaternion representing rotation about given axis by given angle in radians
    ///
    /// Args:
    ///     axis (array-like): 3-element vector (any real numeric array-like) representing axis about which to rotate (does not need to be normalized)
    ///     angle (float): Angle in radians to rotate about axis (right-handed rotation of vector)
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation about given axis by given angle.  If axis norm is < 1e-9,
    ///     unit quaternion is returned
    ///
    #[staticmethod]
    fn from_axis_angle(axis: &Bound<'_, PyAny>, angle: f64) -> Result<Self> {
        let v = to_vector3(axis, "axis")?;
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
    ///     v1 (array-like): 3-element vector (any real numeric array-like) representing vector rotating from
    ///     v2 (array-like): 3-element vector (any real numeric array-like) representing vector rotating to
    ///
    /// Returns:
    ///     quaternion: Quaternion representing rotation from v1 to v2
    #[staticmethod]
    fn rotation_between(v1: &Bound<'_, PyAny>, v2: &Bound<'_, PyAny>) -> Result<Self> {
        let v1 = to_vector3(v1, "v1")?;
        let v2 = to_vector3(v2, "v2")?;

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
    ///     dcm (array-like): 3x3 array representing rotation matrix
    ///
    /// Returns:
    ///     quaternion: Quaternion representing same rotation as input matrix
    #[staticmethod]
    fn from_rotation_matrix(dcm: &Bound<'_, PyAny>) -> Result<Self> {
        let dcm = to_f64_ndarray(dcm)?;
        let dcm = dcm.readonly();
        let dcm = dcm.as_array();
        if dcm.shape() != [3, 3] {
            bail!("Invalid DCM.  Must be 3x3 matrix");
        }
        // numpy arrays are row-major, build Matrix3 row-by-row
        let mat = Matrix3::new([
            [dcm[[0, 0]], dcm[[0, 1]], dcm[[0, 2]]],
            [dcm[[1, 0]], dcm[[1, 1]], dcm[[1, 2]]],
            [dcm[[2, 0]], dcm[[2, 1]], dcm[[2, 2]]],
        ]);
        Ok(Quaternion::from_rotation_matrix(&mat).into())
    }

    /// Return rotation matrix representing identical rotation to quaternion
    ///
    /// Returns:
    ///     numpy.ndarray: 3x3 numpy array representing rotation matrix
    fn to_rotation_matrix(&self, py: Python) -> PyResult<Py<PyAny>> {
        // numeris storage is column-major: the transpose's is row-major
        let rot = self.0.to_rotation_matrix();
        slice2py2d(py, rot.transpose().as_slice(), 3, 3)
    }

    ///Return rotation represented as "roll", "pitch", "yaw" euler angles in radians.
    ///
    /// Returns:
    ///     (f64, f64, f64): Tuple of roll, pitch, yaw angles in radians
    fn to_euler(&self) -> (f64, f64, f64) {
        self.0.to_euler()
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_rotation_matrix()``.
    fn as_rotation_matrix(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        warn_deprecated(
            py,
            c"quaternion.as_rotation_matrix() is deprecated since 0.23 and will be removed in 0.25; use quaternion.to_rotation_matrix()",
        )?;
        self.to_rotation_matrix(py)
    }

    /// Deprecated since 0.23, removed in 0.25. Use ``to_euler()``.
    fn as_euler(&self, py: Python<'_>) -> PyResult<(f64, f64, f64)> {
        warn_deprecated(
            py,
            c"quaternion.as_euler() is deprecated since 0.23 and will be removed in 0.25; use quaternion.to_euler()",
        )?;
        Ok(self.to_euler())
    }

    /// Create quaternion from "roll", "pitch", "yaw" euler angles in radians
    /// (inverse of `to_euler`)
    ///
    /// Args:
    ///     roll (float): Roll angle in radians
    ///     pitch (float): Pitch angle in radians
    ///     yaw (float): Yaw angle in radians
    ///
    /// Returns:
    ///     quaternion: Quaternion representing the input euler-angle rotation
    #[staticmethod]
    fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        Quaternion::from_euler(roll, pitch, yaw).into()
    }

    /// The identity (no-rotation) quaternion
    ///
    /// Returns:
    ///     quaternion: Identity quaternion (w=1, x=y=z=0)
    #[staticmethod]
    fn identity() -> Self {
        Quaternion::identity().into()
    }

    /// Quaternion norm (Euclidean length of the 4 components; 1 for a unit
    /// rotation quaternion)
    ///
    /// Returns:
    ///     float: Norm of the quaternion
    #[getter]
    fn norm(&self) -> f64 {
        self.0.norm()
    }

    /// Return this quaternion normalized to unit length
    ///
    /// Returns:
    ///     quaternion: Normalized (unit) quaternion
    fn normalize(&self) -> Self {
        self.0.normalize().into()
    }

    /// Quaternion inverse. For a unit (rotation) quaternion this equals the
    /// conjugate; for a non-unit quaternion the conjugate is scaled by the
    /// squared norm.
    ///
    /// Returns:
    ///     quaternion: Inverse quaternion
    fn inverse(&self) -> Self {
        self.0.inverse().into()
    }

    /// Dot product of the 4 quaternion components with another quaternion
    ///
    /// Args:
    ///     other (quaternion): Quaternion to dot with
    ///
    /// Returns:
    ///     float: Dot product
    fn dot(&self, other: &Self) -> f64 {
        self.0.dot(&other.0)
    }

    fn __str__(&self) -> Result<String> {
        let (ax, angle) = self.0.to_axis_angle();
        let n = ax.norm();
        let ax = if n < 1.0e-9 {
            numeris::vector![1.0, 0.0, 0.0]
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
        let [w, x, y, z] = crate::pyutils::unpack_f64s(py, &state)?;
        self.0 = Quaternion::new(w, x, y, z);
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::pyutils::pack_f64s(py, &[self.0.w, self.0.x, self.0.y, self.0.z])
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
    fn axis(&self, py: Python) -> PyResult<Py<PyAny>> {
        let (ax, _) = self.0.to_axis_angle();
        let n = ax.norm();
        let a = if n < 1.0e-9 {
            numeris::vector![1.0, 0.0, 0.0]
        } else {
            ax * (1.0 / n)
        };
        vec2py(py, &a)
    }

    /// Quaternion conjugate, which for a unit (rotation) quaternion
    /// is the inverse rotation. Same as ``conjugate()`` and ``inverse()``.
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
    fn conj(&self) -> Self {
        self.0.conjugate().into()
    }

    /// Quaternion conjugate, which for a unit (rotation) quaternion
    /// is the inverse rotation. Same as ``conj()`` and ``inverse()``.
    ///
    /// Returns:
    ///     quaternion: Quaternion representing inverse rotation
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
    ///
    /// Returns:
    ///     quaternion: Quaterion represention fracional spherical interpolation between self and other
    fn slerp(&self, other: &Self, frac: f64) -> Result<Self> {
        Ok(self.0.slerp(&other.0, frac).into())
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
        // Multiply quaternion by quaternion
        if other.is_instance_of::<Self>() {
            let q: PyRef<Self> = other
                .extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract quaternion: {}", e))?;
            return Ok(Self(self.0 * q.0).into_py_any(other.py())?);
        }
        // Rotate a 3-vector or an Nx3 array of vectors: any real numeric
        // array-like (integer arrays, lists) is converted to float64 first
        let arr = to_f64_ndarray(other)?;
        let ro = arr.readonly();
        let a = ro.as_array();
        match a.shape() {
            [3] => {
                let vout = self.0 * numeris::vector![a[[0]], a[[1]], a[[2]]];
                Ok(vec2py(other.py(), &vout)?)
            }
            [_, 3] => {
                // Row i of the result is (R v_i)^T, i.e. V · Rᵀ. Built by
                // index: numeris matrices are column-major, so viewing their
                // storage as a row-major ndarray would silently transpose R
                // (which is how this path used to return the inverse rotation).
                let r = self.0.to_rotation_matrix();
                let rt = np::ndarray::Array2::from_shape_fn((3, 3), |(i, j)| r[(j, i)]);
                let a2 = a
                    .into_dimensionality::<np::ndarray::Ix2>()
                    .context("Invalid rhs")?;
                Ok(a2.dot(&rt).to_pyarray(other.py()).into_py_any(other.py())?)
            }
            shape => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid rhs.  Expected a quaternion, a 3-element vector or an Nx3 array, got shape {:?}",
                shape
            ))
            .into()),
        }
    }
}
