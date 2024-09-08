#[derive(Clone, Debug)]
pub struct RKAdaptiveSettings {
    pub abserror: f64,
    pub relerror: f64,
    pub minfac: f64,
    pub maxfac: f64,
    pub gamma: f64,
    pub dtmin: f64,
    pub dense_output: bool,
}

impl RKAdaptiveSettings {
    pub fn default() -> RKAdaptiveSettings {
        RKAdaptiveSettings {
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
