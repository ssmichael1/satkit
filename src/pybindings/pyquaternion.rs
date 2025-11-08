use anyhow::Context;
use numpy as np;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::mathtypes::*;

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
#[pyclass(name = "quaternion", module = "satkit")]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct PyQuaternion(pub Quaternion);

#[pyclass(name = "quaternion_array", module = "satkit")]
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
            // Create a nalgebra quaternion from 4 input scalars
            let q = nalgebra::Quaternion::<f64>::new(w, x, y, z);
            Ok(Quaternion::from_quaternion(q).into())
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
        Ok(Quaternion::from_axis_angle(&Vector3::x_axis(), theta_rad).into())
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
        Ok(Quaternion::from_axis_angle(&Vector3::y_axis(), theta_rad).into())
    }

    /// Quaternion representing rotation about
    /// zhat axis by `theta-rad` degrees
    #[staticmethod]
    fn rotz(theta_rad: f64) -> Result<Self> {
        Ok(Quaternion::from_axis_angle(&Vector3::z_axis(), theta_rad).into())
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
        let v = Vector3::from_row_slice(axis.as_slice()?);
        let u = nalgebra::UnitVector3::try_new(v, 1.0e-9);
        if let Some(unit_axis) = u {
            Ok(Quaternion::from_axis_angle(&unit_axis, angle).into())
        } else {
            // If the axis is zero, return identity quaternion
            Ok(Quaternion::identity().into())
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
                Vector3::from_row_slice(v1.as_slice().context("Cannot convert v1 to 3D vector")?)
            }
            false => Vector3::from_row_slice(&[
                *v1.get(0).unwrap(),
                *v1.get(1).unwrap(),
                *v1.get(2).unwrap(),
            ]),
        };
        let v2 = match v2.is_contiguous() {
            true => {
                Vector3::from_row_slice(v2.as_slice().context("Cannot convert vd2 to 3D vector")?)
            }
            false => Vector3::from_row_slice(&[
                *v2.get(0).unwrap(),
                *v2.get(1).unwrap(),
                *v2.get(2).unwrap(),
            ]),
        };
        let q = Quaternion::rotation_between(&v1, &v2)
            .context("Norms are 0 or vectors are 180° apart")?;

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
        let mat = nalgebra::Matrix3::from_iterator(dcm.iter().cloned());
        let rot = nalgebra::Rotation3::from_matrix(&mat.transpose());

        Ok(Quaternion::from_rotation_matrix(&rot).into())
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
                    rot.matrix().as_ptr(),
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
        self.0.euler_angles()
    }

    fn __str__(&self) -> Result<String> {
        let ax: nalgebra::Unit<Vector3> = self.0.axis().map_or_else(
            || nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            |v| v,
        );
        let angle = self.0.angle();
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
        self.0 = Quaternion::from_quaternion(nalgebra::Quaternion::<f64>::new(w, x, y, z));
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut raw = [0; 32];
        raw[0..8].clone_from_slice(f64::to_le_bytes(self.0.w).as_slice());
        raw[8..16].clone_from_slice(f64::to_le_bytes(self.0.i).as_slice());
        raw[16..24].clone_from_slice(f64::to_le_bytes(self.0.j).as_slice());
        raw[24..32].clone_from_slice(f64::to_le_bytes(self.0.k).as_slice());
        PyBytes::new(py, &raw).into_py_any(py)
    }

    /// Angle of rotation in radians
    ///
    /// Returns:
    ///     float: Angle of rotation in radians
    #[getter]
    fn angle(&self) -> f64 {
        self.0.angle()
    }

    /// Axis of rotation
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element numpy array representing axis of rotation
    #[getter]
    fn axis(&self) -> PyResult<Py<PyAny>> {
        let a = self.0.axis().map_or_else(Vector3::x_axis, |ax| ax);
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
    fn slerp(&self, other: &Self, frac: f64, epsilon: f64) -> Result<Self> {
        self.0.try_slerp(&other.0, frac, epsilon).map_or_else(
            || bail!("Quaternions cannot be 180 deg apart"),
            |v| Ok(v.into()),
        )
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
            let qmat = rot.matrix().conjugate();

            Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                let nd = unsafe { np::ndarray::ArrayView2::from_shape_ptr((3, 3), qmat.as_ptr()) };
                let res = v.readonly().as_array().dot(&nd).to_pyarray(py);

                res.into_py_any(py)
            })?)
        } else if let Ok(v1d) = other.cast::<np::PyArray1<f64>>() {
            if v1d.len() != 3 {
                bail!("Invalid rhs.  1D array must be of length 3");
            }

            let m = nalgebra::vector![
                v1d.get_owned(0).unwrap(),
                v1d.get_owned(1).unwrap(),
                v1d.get_owned(2).unwrap()
            ];

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
