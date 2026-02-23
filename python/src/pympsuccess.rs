use rmpfit::MPSuccess;

use pyo3::prelude::*;

///
/// MPFit success codes
/// returned when generating a TLE by fitting it
/// to satellite state vectors
///
/// see: <https://docs.rs/rmpfit/latest/rmpfit/> for details
///
#[derive(PartialEq, Eq)]
#[pyclass(name = "mpsuccess", eq, eq_int)]
pub enum PyMPSuccess {
    NotDone = MPSuccess::NotDone as isize,
    Chi = MPSuccess::Chi as isize,
    Par = MPSuccess::Par as isize,
    Both = MPSuccess::Both as isize,
    Dir = MPSuccess::Dir as isize,
    MaxIter = MPSuccess::MaxIter as isize,
    Ftol = MPSuccess::Ftol as isize,
    Xtol = MPSuccess::Xtol as isize,
    Gtol = MPSuccess::Gtol as isize,
}

impl From<MPSuccess> for PyMPSuccess {
    fn from(success: MPSuccess) -> Self {
        match success {
            MPSuccess::NotDone => Self::NotDone,
            MPSuccess::Chi => Self::Chi,
            MPSuccess::Par => Self::Par,
            MPSuccess::Both => Self::Both,
            MPSuccess::Dir => Self::Dir,
            MPSuccess::MaxIter => Self::MaxIter,
            MPSuccess::Ftol => Self::Ftol,
            MPSuccess::Xtol => Self::Xtol,
            MPSuccess::Gtol => Self::Gtol,
        }
    }
}

#[pymethods]
impl PyMPSuccess {
    pub fn __str__(&self) -> &str {
        match self {
            Self::NotDone => "Not Finished Iterations",
            Self::Chi => "Convergence in chi-square Value",
            Self::Par => "Convergence in parameter value",
            Self::Both => "Convergence in both chi-square and parameter",
            Self::Dir => "Convergence in orthogonality",
            Self::MaxIter => "Maximum number of iterations reached",
            Self::Ftol => "ftol is too small; no further improvement",
            Self::Xtol => "xtol is too small; no further improvement",
            Self::Gtol => "gtol is too small; no further improvement",
        }
    }
}
