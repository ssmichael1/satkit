use super::propagator::SimpleState;
use crate::AstroTime;

pub trait SatProperties {
    // Coefficient of drag times normal area over mass
    fn cd_a_over_m(&self, tm: &AstroTime, state: &SimpleState) -> f64;

    // Coefficient of radiation pressure times normal area over mass
    fn cr_a_over_m(&self, tm: &AstroTime, state: &SimpleState) -> f64;
}

#[derive(Debug, Clone)]
pub struct SatPropertiesStatic {
    pub cdaoverm: f64,
    pub craoverm: f64,
}

impl SatPropertiesStatic {
    pub fn new(cdaoverm: f64, craoverm: f64) -> SatPropertiesStatic {
        SatPropertiesStatic {
            cdaoverm: cdaoverm,
            craoverm: craoverm,
        }
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"Static Sat Properties:
              Cd A / M : {} m^2/kg
              Cr A / M : {} m^2/kg"#,
            self.cdaoverm, self.craoverm,
        )
    }
}

impl std::fmt::Display for SatPropertiesStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl SatProperties for SatPropertiesStatic {
    fn cd_a_over_m(&self, _tm: &AstroTime, _state: &SimpleState) -> f64 {
        self.cdaoverm
    }

    fn cr_a_over_m(&self, _tm: &AstroTime, _state: &SimpleState) -> f64 {
        self.craoverm
    }
}
