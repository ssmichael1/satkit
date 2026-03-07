//! Orbit Propagation Settings

use crate::earthgravity::GravityModel;
use crate::orbitprop::Precomputed;
use crate::TimeLike;

use anyhow::Result;

/// Choice of ODE integrator for orbit propagation
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Integrator {
    /// Verner 9(8) with dense output, 21 stages (default)
    RKV98,
    /// Verner 9(8) without interpolation, 16 stages
    RKV98NoInterp,
    /// Verner 8(7), 17 stages
    RKV87,
    /// Verner 6(5), 10 stages
    RKV65,
    /// Tsitouras 5(4) with FSAL, 7 stages
    RKTS54,
}

impl Default for Integrator {
    fn default() -> Self {
        Self::RKV98
    }
}

impl std::fmt::Display for Integrator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Integrator::RKV98 => write!(f, "RKV98 (9th order, 21 stages)"),
            Integrator::RKV98NoInterp => write!(f, "RKV98NoInterp (9th order, 16 stages)"),
            Integrator::RKV87 => write!(f, "RKV87 (8th order, 17 stages)"),
            Integrator::RKV65 => write!(f, "RKV65 (6th order, 10 stages)"),
            Integrator::RKTS54 => write!(f, "RKTS54 (5th order, 7 stages, FSAL)"),
        }
    }
}

/// Propagation settings
///
/// These include
///
/// * `gravity_degree` - maximum degree of spherical harmonic gravity model.  Default is 4
/// * `gravity_order` - maximum order of spherical harmonic gravity model.  Default is same as `gravity_degree`.
///    Must be ≤ `gravity_degree`.
/// * `gravity_model` - gravity model to use.  Default is JGM3.  Options: JGM3, JGM2, EGM96, ITUGrace16
/// * `abs_error` - the maximum absolute error for the infinity norm of the state in Runge-Kutta integrator.  Default is 1e-8
/// * `rel_error` - the maximum relative error for the infinity norm of the state in Runge-Kutta integrator.  Default is 1e-8
/// * `use_spaceweather` -  Do we use space weather when computing the atmospheric density.  Default is true
/// * `use_sun_gravity` - Do we include sun third-body gravitational perturbation.  Default is true
/// * `use_moon_gravity` - Do we include moon third-body gravitational perturbation.  Default is true
/// * `enable_interp` - Do we enable interpolation of the state between begin and end times.  Default is true
///   slight computation savings if set to false
/// * `integrator` - which Runge-Kutta integrator to use.  Default is RKV98
///
#[derive(Debug, Clone)]
pub struct PropSettings {
    pub gravity_degree: u16,
    pub gravity_order: u16,
    pub gravity_model: GravityModel,
    pub abs_error: f64,
    pub rel_error: f64,
    pub use_spaceweather: bool,
    pub use_sun_gravity: bool,
    pub use_moon_gravity: bool,
    pub enable_interp: bool,
    pub integrator: Integrator,
    pub precomputed: Option<Precomputed>,
}

impl Default for PropSettings {
    fn default() -> Self {
        Self {
            gravity_degree: 4,
            gravity_order: 4,
            gravity_model: GravityModel::JGM3,
            abs_error: 1e-8,
            rel_error: 1e-8,
            use_spaceweather: true,
            use_sun_gravity: true,
            use_moon_gravity: true,
            enable_interp: true,
            integrator: Integrator::default(),
            precomputed: None,
        }
    }
}

impl PropSettings {
    /// Set gravity degree and order, with validation
    ///
    /// # Arguments
    /// * `degree` - Maximum degree of spherical harmonic gravity model
    /// * `order` - Maximum order (must be ≤ degree)
    ///
    /// # Errors
    /// Returns error if order > degree
    pub fn set_gravity(&mut self, degree: u16, order: u16) -> Result<()> {
        if order > degree {
            anyhow::bail!(
                "Gravity order ({}) must be ≤ degree ({})",
                order,
                degree
            );
        }
        self.gravity_degree = degree;
        self.gravity_order = order;
        Ok(())
    }

    /// Precompute terms between begin and end instants
    ///
    /// # Arguments
    /// * `begin` - Begin instant
    /// * `end` - End instant
    ///
    /// Pre-computes inertial to earth-fixed rotation vector (used for Earth gravity calculation),
    /// sun, and moon positions between the begin and end instants.  These are used in the
    /// force model when propagating orbits
    ///
    /// Pre-computing these terms means the settings can be used for multiple propagations
    /// between the same begin and end instants without needing to recompute these terms each time.
    /// (significant speedup when propagating many orbits over the same time span)
    ///
    /// # Errors
    /// Returns error if precomputation fails
    ///
    /// # Example
    /// ```
    /// use satkit::orbitprop::PropSettings;
    /// use satkit::Instant;
    ///
    /// let begin = Instant::now();
    /// let end = begin + satkit::Duration::from_hours(1.0);
    /// let mut props = PropSettings::default();
    /// props.precompute_terms(&begin, &end).unwrap();
    ///
    /// ```
    pub fn precompute_terms<T: TimeLike>(&mut self, begin: &T, end: &T) -> Result<()> {
        self.precomputed = Some(Precomputed::new(begin, end)?);
        Ok(())
    }

    pub fn precompute_terms_with_step<T: TimeLike>(
        &mut self,
        begin: &T,
        end: &T,
        step_secs: f64,
    ) -> Result<()> {
        self.precomputed = Some(Precomputed::new_with_step(begin, end, step_secs)?);
        Ok(())
    }
}

impl std::fmt::Display for PropSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"Orbit Propagation Settings
            Gravity Degree: {},
            Gravity Order: {},
            Gravity Model: {},
            Max Abs Error: {:e},
            Max Rel Error: {:e},
            Space Weather: {},
            Sun Gravity: {},
            Moon Gravity: {},
            Interpolation: {},
            Integrator: {},
            {}"#,
            self.gravity_degree,
            self.gravity_order,
            self.gravity_model,
            self.abs_error,
            self.rel_error,
            self.use_spaceweather,
            self.use_sun_gravity,
            self.use_moon_gravity,
            self.enable_interp,
            self.integrator,
            self.precomputed.as_ref().map_or_else(
                || "No Precomputed".to_string(),
                |p| format!("Precomputed: {} to {}", p.begin, p.end)
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
