use satkit::tle::TleFitStatus;

use pyo3::prelude::*;

///
/// Termination status of the TLE non-linear least-squares fit performed by
/// [`satkit.TLE.fit_from_states`].
///
#[derive(PartialEq, Eq, Clone, Copy)]
#[pyclass(name = "tlefitstatus", eq, eq_int, skip_from_py_object)]
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
    pub fn __str__(&self) -> &str {
        match self {
            Self::GradientConverged => "Converged (gradient tolerance)",
            Self::StepConverged => "Converged (step size tolerance)",
            Self::CostConverged => "Converged (cost change tolerance)",
            Self::MaxIterations => "Maximum iterations reached",
            Self::DampingSaturated => "Damping parameter saturated",
        }
    }

    /// True if the fit converged successfully.
    pub fn converged(&self) -> bool {
        matches!(
            self,
            Self::GradientConverged | Self::StepConverged | Self::CostConverged
        )
    }
}
