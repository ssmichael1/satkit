//! Orbit Propagation Settings

use crate::orbitprop::Precomputed;
use crate::Instant;

use anyhow::Result;

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

impl Default for PropSettings {
    fn default() -> Self {
        Self {
            gravity_order: 4,
            abs_error: 1e-8,
            rel_error: 1e-8,
            use_spaceweather: true,
            enable_interp: true,
            precomputed: None,
        }
    }
}

impl PropSettings {
    pub fn precompute_terms(&mut self, start: &Instant, stop: &Instant) -> Result<()> {
        self.precomputed = Some(Precomputed::new(start, stop)?);
        Ok(())
    }
}

impl std::fmt::Display for PropSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
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
            self.precomputed.as_ref().map_or_else(
                || "No Precomputed".to_string(),
                |p| format!("Precomputed: {} to {}", p.start, p.stop)
            )
        )
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
