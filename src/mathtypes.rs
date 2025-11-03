//!
//! Mathematical types used throughout the library
//!
//! This module defines commonly used mathematical types such as vectors, matrices, and quaternions using the `nalgebra` crate.
//! These types are used for representing positions, velocities, orientations, and other mathematical constructs in the library.
//!
//! There may be an attempt to abstract over the underlying math library in the future, but for now we directly use `nalgebra` types.
//!

pub type Vec3 = nalgebra::Vector3<f64>;
pub type Quaternion = nalgebra::UnitQuaternion<f64>;
pub type Vector<const T: usize> = nalgebra::SVector<f64, T>;
pub type Matrix<const M: usize, const N: usize> = nalgebra::SMatrix<f64, M, N>;
pub type Vector6 = Vector<6>;
pub type Vector3 = Vector<3>;
pub type Matrix3 = Matrix<3, 3>;
pub type Matrix6 = Matrix<6, 6>;
pub type Matrix67 = Matrix<6, 7>;
pub type DMatrix<T> = nalgebra::DMatrix<T>;

/// Create a statically-sized vector with f64 elements
///
/// # Examples
///
/// ```
/// let v = vector![1.0, 2.0, 3.0];  // Creates a Vector3
/// let v6 = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // Creates a Vector6
/// ```
#[macro_export]
macro_rules! vector {
    ($($x:expr),* $(,)?) => {
        nalgebra::vector![$($x),*]
    };
}

// Matrix macro
#[macro_export]
macro_rules! matrix {
    ($($x:expr),* $(,)?) => {
        nalgebra::matrix![$($x),*]
    };
}
