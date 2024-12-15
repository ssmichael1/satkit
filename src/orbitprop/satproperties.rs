use super::propagator::SimpleState;
use crate::Instant;

/// Generic trate for satellite properties
/// This allows for setting C_d A / M (coeffeicient of drag times area over mass)
/// in units of m^2/kg as a function of time and state
///
/// And also Cr A over M (coeffeienct of radiation pressure times area over mass)
/// in units of m^2/kg as function of time and state
///
pub trait SatProperties {
    // Coefficient of drag times normal area over mass
    fn cd_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64;

    // Coefficient of radiation pressure times normal area over mass
    fn cr_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64;
}

/// Convenience structure for setting fixed values for drag and
/// radiation pressure susceptibility for propagator
///
/// cdaoverm = C_d A / M = coefficient of drag times area over mass, in meters^2 / kg
/// craoverm = C_r A / M = coefficient of radiation pressure time area over mass, in meters^2 / kg
#[derive(Debug, Clone)]
pub struct SatPropertiesStatic {
    pub cdaoverm: f64,
    pub craoverm: f64,
}

impl SatPropertiesStatic {
    pub const fn new(cdaoverm: f64, craoverm: f64) -> Self {
        Self { cdaoverm, craoverm }
    }
}

impl Default for SatPropertiesStatic {
    fn default() -> Self {
        Self {
            cdaoverm: 0.0,
            craoverm: 0.0,
        }
    }
}

impl std::fmt::Display for SatPropertiesStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"Static Sat Properties:
              Cd A / M : {} m^2/kg
              Cr A / M : {} m^2/kg"#,
            self.cdaoverm, self.craoverm,
        )
    }
}

impl SatProperties for SatPropertiesStatic {
    fn cd_a_over_m(&self, _tm: &Instant, _state: &SimpleState) -> f64 {
        self.cdaoverm
    }

    fn cr_a_over_m(&self, _tm: &Instant, _state: &SimpleState) -> f64 {
        self.craoverm
    }
}
