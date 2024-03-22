//! Orbit Propagation Settings

/// Propagation settings
///
/// These include
///
/// * `gravity_order` - integer gravity order to use when computing Earth gravity.  Default is 4
/// * `gravity_interp_dt_seconds` - Interpolation interval for rotation to ITRF frame for gravity calc.  Default is 60 seconds
/// * `abs_error` - the maximum absolute error for the infinity norm of the state in Runga-Kutta integrator.  Default is 1e-8
/// * `rel_error` - the maximum relative error for the infinity norm of the state in Runga-Kutta integrator.  Default is 1e-8
/// * `use_spaceweather` -  Do we use space weather when computing the atmospheric density.  Default is true
/// * `use_jplephem` -  Use very high precision JPL ephemerides when computing force of sun & moon.  Default is true
///
#[derive(Debug, Clone)]
pub struct PropSettings {
    pub gravity_order: u16,
    pub gravity_interp_dt_secs: f64,
    pub abs_error: f64,
    pub rel_error: f64,
    pub use_spaceweather: bool,
    /// Use JPL ephemeris (vs low-precision ephemeris) for sun & moon
    pub use_jplephem: bool,
}

impl PropSettings {
    pub fn default() -> PropSettings {
        PropSettings {
            gravity_order: 4,
            gravity_interp_dt_secs: 60.0,
            abs_error: 1e-8,
            rel_error: 1e-8,
            use_spaceweather: true,
            use_jplephem: true,
        }
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"Orbit Propagation Settings
            Gravity Order: {},
            Max Abs Error: {:e},
            Max Rel Error: {:e},
            Space Weather: {},
            JPL Ephemeris: {}"#,
            self.gravity_order,
            self.abs_error,
            self.rel_error,
            self.use_spaceweather,
            self.use_jplephem
        )
    }
}

impl std::fmt::Display for PropSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn testdisplay() {
        let props = PropSettings::default();
        println!("props = {}", props);
    }
}
