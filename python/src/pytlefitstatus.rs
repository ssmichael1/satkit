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
/// equal to its integer value.
#[derive(PartialEq, Eq, Clone, Copy)]
#[pyclass(name = "tlefitstatus", eq, eq_int)]
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
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (&'static str, String))> {
        crate::pyutils::enum_reduce(slf.as_any(), "tlefitstatus")
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
