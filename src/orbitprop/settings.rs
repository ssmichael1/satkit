//! Orbit Propagation Settings

use crate::earthgravity::GravityModel;
use crate::orbitprop::Precomputed;
use crate::TimeLike;

use anyhow::Result;

/// Choice of ODE integrator for orbit propagation
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Integrator {
    /// Verner 9(8) with 9th-order dense output, 26 stages (default)
    RKV98,
    /// Verner 9(8) without interpolation, 16 stages
    RKV98NoInterp,
    /// Verner 8(7) with 8th-order dense output, 21 stages
    RKV87,
    /// Verner 6(5), 10 stages
    RKV65,
    /// Tsitouras 5(4) with FSAL, 7 stages
    RKTS54,
    /// RODAS4 — L-stable Rosenbrock 4(3), 6 stages. For stiff problems (re-entry, low perigee).
    /// Does not support state transition matrix propagation or dense output interpolation.
    RODAS4,
    /// Gauss-Jackson 8 — 8th-order fixed-step multistep predictor-corrector
    /// specialised for 2nd-order ODEs (Berry & Healy 2004). The dominant
    /// integrator in high-precision astrodynamics codes (GMAT, STK, ODTK).
    /// Typically uses 3-10× fewer force evaluations than a Runge-Kutta method
    /// of comparable accuracy on smooth orbit propagation problems.
    ///
    /// Uses a fixed step size set via [`PropSettings::gj_step_seconds`].
    /// Does not support state transition matrix propagation (C=7) or dense
    /// output interpolation. Not recommended for highly eccentric orbits or
    /// integration across discontinuities (eclipse boundaries, maneuvers).
    GaussJackson8,
}

impl Default for Integrator {
    fn default() -> Self {
        Self::RKV98
    }
}

impl std::fmt::Display for Integrator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Integrator::RKV98 => write!(f, "RKV98 (9th order, 26 stages)"),
            Integrator::RKV98NoInterp => write!(f, "RKV98NoInterp (9th order, 16 stages)"),
            Integrator::RKV87 => write!(f, "RKV87 (8th order, 21 stages)"),
            Integrator::RKV65 => write!(f, "RKV65 (6th order, 10 stages)"),
            Integrator::RKTS54 => write!(f, "RKTS54 (5th order, 7 stages, FSAL)"),
            Integrator::RODAS4 => write!(f, "RODAS4 (4th order, 6 stages, L-stable)"),
            Integrator::GaussJackson8 => write!(f, "Gauss-Jackson 8 (8th order, fixed-step multistep)"),
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
/// * `gravity_model` - gravity model to use.  Default is EGM96.  Options: EGM96, JGM3, JGM2, ITUGrace16
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
    /// Fixed step size (seconds) used by [`Integrator::GaussJackson8`].
    /// Ignored by adaptive integrators. Typical values: 30-120 s for LEO,
    /// 60-300 s for MEO, 300-600 s for GEO. Default: 60 s.
    pub gj_step_seconds: f64,
    pub precomputed: Option<Precomputed>,
}

impl Default for PropSettings {
    fn default() -> Self {
        Self {
            gravity_degree: 4,
            gravity_order: 4,
            gravity_model: GravityModel::EGM96,
            abs_error: 1e-8,
            rel_error: 1e-8,
            use_spaceweather: true,
            use_sun_gravity: true,
            use_moon_gravity: true,
            enable_interp: true,
            integrator: Integrator::default(),
            gj_step_seconds: 60.0,
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

    /// Compute the required padding (in seconds) beyond the nominal
    /// propagation interval when building a [`Precomputed`] interp table
    /// for these settings.
    ///
    /// Most integrators only evaluate the force within the nominal
    /// interval and need the default padding. The
    /// [`Integrator::GaussJackson8`] integrator, however, runs a
    /// symmetric ±4·h_gj startup around the starting epoch — requiring
    /// the interp table to cover times up to `4·gj_step_seconds` outside
    /// the interval on the startup side. A small safety margin is added
    /// to guard against floating-point round-off.
    pub fn required_precompute_padding(&self) -> f64 {
        match self.integrator {
            Integrator::GaussJackson8 => {
                // 4 startup steps backward from epoch + safety margin
                (4.0 * self.gj_step_seconds.abs() + 60.0)
                    .max(crate::orbitprop::precomputed::DEFAULT_PADDING_SECS)
            }
            Integrator::RKV98
            | Integrator::RKV98NoInterp
            | Integrator::RKV87
            | Integrator::RKV65
            | Integrator::RKTS54
            | Integrator::RODAS4 => crate::orbitprop::precomputed::DEFAULT_PADDING_SECS,
        }
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
    /// The precomputed range is automatically padded to accommodate the
    /// selected integrator. For [`Integrator::GaussJackson8`] the padding
    /// is extended to `4·gj_step_seconds + 60 s` on each end to cover the
    /// backward RK4 startup stencil; for other integrators a default
    /// 240 s padding is used.
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
        let padding = self.required_precompute_padding();
        self.precomputed = Some(Precomputed::new_padded(begin, end, 60.0, padding)?);
        Ok(())
    }

    pub fn precompute_terms_with_step<T: TimeLike>(
        &mut self,
        begin: &T,
        end: &T,
        step_secs: f64,
    ) -> Result<()> {
        let padding = self.required_precompute_padding();
        self.precomputed = Some(Precomputed::new_padded(begin, end, step_secs, padding)?);
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
