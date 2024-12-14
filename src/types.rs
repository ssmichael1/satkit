use thiserror::Error;
pub type SKError = Box<dyn std::error::Error + Send + Sync>;
pub type SKResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Error)]
pub enum SKErr {
    #[error("SatKit Error: {0}")]
    Error(String),
}

macro_rules! skerror {
    ($($arg:tt)*) => {
        Err(Box::new(crate::SKErr::Error(format!($($arg)*))))
    };
}

pub(crate) use skerror;

pub type Vec3 = nalgebra::Vector3<f64>;
pub type Quaternion = nalgebra::UnitQuaternion<f64>;
pub type Vector<const T: usize> = nalgebra::SVector<f64, T>;
pub type Matrix<const M: usize, const N: usize> = nalgebra::SMatrix<f64, M, N>;
pub type Vector6 = Vector<6>;
pub type Vector3 = Vector<3>;
pub type Matrix3 = Matrix<3, 3>;
pub type Matrix6 = Matrix<6, 6>;
pub type Matrix67 = Matrix<6, 7>;
