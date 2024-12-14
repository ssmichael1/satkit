#[derive(Clone, Debug)]

/// Settings for adaptive Runge-Kutta methods.
pub struct RKAdaptiveSettings {
    /// Absolute error tolerance
    pub abserror: f64,
    /// Relative error tolerance
    pub relerror: f64,
    /// Minimum factor for step size
    pub minfac: f64,
    /// Maximum factor for step size
    pub maxfac: f64,
    /// Safety factor
    pub gamma: f64,
    /// Minimum step size
    pub dtmin: f64,
    /// Enable dense output (more storage, but allows interpolation)
    pub dense_output: bool,
}

impl Default for RKAdaptiveSettings {
    fn default() -> Self {
        Self {
            abserror: 1.0e-8,
            relerror: 1.0e-8,
            minfac: 0.2,
            maxfac: 10.0,
            gamma: 0.9,
            dtmin: 1.0e-6,
            dense_output: false,
        }
    }
}
