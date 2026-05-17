use super::propagator::SimpleState;
use super::thrust::ThrustProfile;
use crate::mathtypes::*;
use crate::Instant;

/// Generic trait for satellite properties
/// This allows for setting C_d A / M (coefficient of drag times area over mass)
/// in units of m^2/kg as a function of time and state
///
/// And also Cr A over M (coefficient of radiation pressure times area over mass)
/// in units of m^2/kg as function of time and state
///
/// And optionally a thrust profile for continuous thrust maneuvers
///
pub trait SatProperties {
    // Coefficient of drag times normal area over mass
    fn cd_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64;

    // Coefficient of radiation pressure times normal area over mass
    fn cr_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64;

    /// Thrust acceleration in GCRF [m/s^2], or None if no thrust is active
    ///
    /// Default implementation returns None (no thrust)
    fn thrust_accel(
        &self,
        _tm: &Instant,
        _pos_gcrf: &Vector3,
        _vel_gcrf: &Vector3,
    ) -> Option<Vector3> {
        None
    }
}

/// Convenience structure for setting fixed values for drag and
/// radiation pressure susceptibility for propagator
///
/// cdaoverm = C_d A / M = coefficient of drag times area over mass, in meters^2 / kg
/// craoverm = C_r A / M = coefficient of radiation pressure time area over mass, in meters^2 / kg
#[derive(Debug, Clone)]
pub struct SatPropertiesSimple {
    pub cdaoverm: f64,
    pub craoverm: f64,
    pub thrust: ThrustProfile,
}

impl SatPropertiesSimple {
    pub fn new(cdaoverm: f64, craoverm: f64) -> Self {
        Self {
            cdaoverm,
            craoverm,
            thrust: ThrustProfile {
                thrusts: Vec::new(),
            },
        }
    }

    pub fn with_thrust(mut self, thrust: ThrustProfile) -> Self {
        self.thrust = thrust;
        self
    }
}

impl Default for SatPropertiesSimple {
    fn default() -> Self {
        Self {
            cdaoverm: 0.0,
            craoverm: 0.0,
            thrust: ThrustProfile::default(),
        }
    }
}

impl std::fmt::Display for SatPropertiesSimple {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"Static Sat Properties:
              Cd A / M : {} m^2/kg
              Cr A / M : {} m^2/kg
          Thrust arcs : {}"#,
            self.cdaoverm,
            self.craoverm,
            self.thrust.thrusts.len(),
        )
    }
}

impl SatProperties for SatPropertiesSimple {
    fn cd_a_over_m(&self, _tm: &Instant, _state: &SimpleState) -> f64 {
        self.cdaoverm
    }

    fn cr_a_over_m(&self, _tm: &Instant, _state: &SimpleState) -> f64 {
        self.craoverm
    }

    fn thrust_accel(
        &self,
        tm: &Instant,
        pos_gcrf: &Vector3,
        vel_gcrf: &Vector3,
    ) -> Option<Vector3> {
        if self.thrust.is_empty() {
            None
        } else {
            self.thrust.accel_gcrf(tm, pos_gcrf, vel_gcrf)
        }
    }
}
