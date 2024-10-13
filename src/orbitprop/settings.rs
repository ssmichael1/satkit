//! Orbit Propagation Settings

use crate::orbitprop::Precomputed;
use crate::utils::SKResult;
use crate::AstroTime;

/// Propagation settings
///
/// These include
///
/// * `gravity_order` - integer gravity order to use when computing Earth gravity.  Default is 4
/// * `gravity_interp_dt_seconds` - Interpolation interval for rotation to ITRF frame for gravity calc.  Default is 60 seconds
/// * `abs_error` - the maximum absolute error for the infinity norm of the state in Runga-Kutta integrator.  Default is 1e-8
/// * `rel_error` - the maximum relative error for the infinity norm of the state in Runga-Kutta integrator.  Default is 1e-8
/// * `use_spaceweather` -  Do we use space weather when computing the atmospheric density.  Default is true
/// * `enable_interp` - Do we enable interpolation of the state between start and stop times.  Default is true
///                     slight comptuation savings if set to false
///
#[derive(Debug, Clone)]
pub struct PropSettings {
    pub gravity_order: u16,
    pub abs_error: f64,
    pub rel_error: f64,
    pub use_spaceweather: bool,
    pub enable_interp: bool,
    pub precomputed: Option<Precomputed>,
}

impl PropSettings {
    pub fn default() -> PropSettings {
        PropSettings {
            gravity_order: 4,
            abs_error: 1e-8,
            rel_error: 1e-8,
            use_spaceweather: true,
            enable_interp: true,
            precomputed: None,
        }
    }

    pub fn precompute_terms(&mut self, start: &AstroTime, stop: &AstroTime) -> SKResult<()> {
        self.precomputed = Some(Precomputed::new(start, stop)?);
        Ok(())
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"Orbit Propagation Settings
            Gravity Order: {},
            Max Abs Error: {:e},
            Max Rel Error: {:e},
            Space Weather: {},
            Interpolation: {}
            {}"#,
            self.gravity_order,
            self.abs_error,
            self.rel_error,
            self.use_spaceweather,
            self.enable_interp,
            match &self.precomputed {
                Some(p) => format!("Precomputed: {} to {}", p.start, p.stop),
                None => "No Precomputed".to_string(),
            }
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
