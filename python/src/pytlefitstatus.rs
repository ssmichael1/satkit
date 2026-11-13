use satkit::tle::TleFitStatus;

use pyo3::prelude::*;

/// Termination status of the TLE non-linear least-squares fit performed by
/// ``TLE.fit_from_states``
///
/// Members: ``GradientConverged`` (0: converged on gradient norm tolerance),
/// ``StepConverged`` (1: converged on relative step size tolerance),
/// ``CostConverged`` (2: converged on relative cost change tolerance),
/// ``MaxIterations`` (3: maximum number of iterations reached) and
/// ``DampingSaturated`` (4: Levenberg-Marquardt damping parameter saturated).
///
/// ``str(status)`` is a one-line description, and each member compares
/// equal to its integer value. Members are hashable and immutable.
/// ``satkit.tlefitstatus(value)`` is the member with that integer value
/// (``ValueError`` for any other).
#[derive(PartialEq, Eq, Clone, Copy)]
#[pyclass(name = "tlefitstatus", module = "satkit", eq, eq_int, frozen)]
pub enum PyTleFitStatus {
    GradientConverged = 0,
    StepConverged = 1,
    CostConverged = 2,
    MaxIterations = 3,
    DampingSaturated = 4,
}

impl From<TleFitStatus> for PyTleFitStatus {
    fn from(status: TleFitStatus) -> Self {
        match status {
            TleFitStatus::GradientConverged => Self::GradientConverged,
            TleFitStatus::StepConverged => Self::StepConverged,
            TleFitStatus::CostConverged => Self::CostConverged,
            TleFitStatus::MaxIterations => Self::MaxIterations,
            TleFitStatus::DampingSaturated => Self::DampingSaturated,
        }
    }
}

#[pymethods]
impl PyTleFitStatus {
    // The member with integer value `value` (documented on the class)
    #[new]
    fn new(value: i32) -> PyResult<Self> {
        Ok(match value {
            0 => Self::GradientConverged,
            1 => Self::StepConverged,
            2 => Self::CostConverged,
            3 => Self::MaxIterations,
            4 => Self::DampingSaturated,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{value} is not a tlefitstatus value (0 to 4)"
                )))
            }
        })
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (&'static str, String))> {
        crate::pyutils::enum_reduce(slf.as_any(), "tlefitstatus")
    }

    // A member equals its integer value (`eq_int`), so it hashes like it
    // (PyO3's `hash` option would hash the Rust value instead)
    const fn __hash__(&self) -> isize {
        *self as isize
    }

    pub fn __str__(&self) -> &str {
        match self {
            Self::GradientConverged => "Converged (gradient tolerance)",
            Self::StepConverged => "Converged (step size tolerance)",
            Self::CostConverged => "Converged (cost change tolerance)",
            Self::MaxIterations => "Maximum iterations reached",
            Self::DampingSaturated => "Damping parameter saturated",
        }
    }

    /// True for the three ``...Converged`` members, False for ``MaxIterations`` and ``DampingSaturated``
    #[getter]
    pub fn converged(&self) -> bool {
        matches!(
            self,
            Self::GradientConverged | Self::StepConverged | Self::CostConverged
        )
    }
}
