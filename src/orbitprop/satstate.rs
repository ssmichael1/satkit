use nalgebra as na;

use crate::orbitprop;
use crate::orbitprop::PropSettings;
use crate::Instant;

use anyhow::Result;

type PVCovType = na::SMatrix<f64, 6, 6>;

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum StateCov {
    None,
    PVCov(PVCovType),
}

///
/// A Satellite State object
///
/// This is a convenience structure for representing
/// a satellite state and state uncertainty, where a state is
/// represented as a position (meters) and velocity (meters) in the
/// Geocentric Celestial Reference Frame (GCRS)
///
/// The structure allows for propagation of the state to a new time
/// using the high-precision orbit propagator associated with this package
///
/// The satellite state can also include a 6x6 covariance matrix representing
/// the uncertainty in position and velocity.
///
/// If the state is propagated, the state uncertainty will be propagated as well
/// via the state transition matrix
///
#[derive(Clone, Debug)]
pub struct SatState {
    pub time: Instant,
    pub pv: na::Vector6<f64>,
    pub cov: StateCov,
}

impl SatState {
    pub fn from_pv(time: &Instant, pos: &na::Vector3<f64>, vel: &na::Vector3<f64>) -> Self {
        Self {
            time: *time,
            pv: na::vector![pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]],
            cov: StateCov::None,
        }
    }

    pub fn pos_gcrf(&self) -> na::Vector3<f64> {
        self.pv.fixed_view::<3, 1>(0, 0).into()
    }

    pub fn vel_gcrf(&self) -> na::Vector3<f64> {
        self.pv.fixed_view::<3, 1>(3, 0).into()
    }

    /// Set covariance
    ///
    /// # Arguments
    ///
    /// * `cov` -  Covariance matrix.  6x6 or larger if including terms like drag.
    ///            Upper-left 6x6 is covariance for position & velocity, in units of
    ///            meters and meters / second
    ///
    pub fn set_cov(&mut self, cov: StateCov) {
        self.cov = cov;
    }

    /// Return Quaternion to go from gcrf (Geocentric Celestial Reference Frame)
    /// to lvlh (Local-Vertical, Local-Horizontal) frame
    ///
    /// Note: lvlh:
    ///       z axis = -r (nadir)
    ///       y axis = -h (h = p cross v)
    ///       x axis such that x cross y = z
    pub fn qgcrf2lvlh(&self) -> na::UnitQuaternion<f64> {
        type Quat = na::UnitQuaternion<f64>;

        let p = self.pos_gcrf();
        let v = self.vel_gcrf();
        let h = p.cross(&v);
        let q1 = Quat::rotation_between(&(-1.0 * p), &na::Vector3::z_axis()).unwrap();
        let q2 = Quat::rotation_between(&(-1.0 * (q1 * h)), &na::Vector3::y_axis()).unwrap();
        q2 * q1
    }

    pub fn cov(&self) -> StateCov {
        self.cov.clone()
    }

    /// Set position uncertainty (1-sigma, meters) in the
    /// lvlh (local-vertical, local-horizontal) frame
    ///
    /// # Arguments
    ///
    /// * `sigma_lvlh` - 3-vector with 1-sigma position uncertainty in LVLH frame
    ///
    pub fn set_lvlh_pos_uncertainty(&mut self, sigma_lvlh: &na::Vector3<f64>) {
        let dcm = self.qgcrf2lvlh().to_rotation_matrix();

        let mut pcov = na::Matrix3::<f64>::zeros();
        pcov.set_diagonal(&sigma_lvlh.map(|x| x * x));

        let mut m = na::Matrix6::<f64>::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(dcm.transpose() * pcov * dcm));
        self.cov = StateCov::PVCov(m);
    }

    /// Set velocity uncertainty (1-sigma, meters/second) in the
    /// lvlh (local-vertical, local-horizontal) frame
    ///
    /// # Arguments
    ///
    /// * `sigma_lvlh` - 3-vector with 1-sigma velocity uncertainty in LVLH frame
    ///
    pub fn set_lvlh_vel_uncertainty(&mut self, sigma_lvlh: &na::Vector3<f64>) {
        let dcm = self.qgcrf2lvlh().to_rotation_matrix();

        let mut pcov = na::Matrix3::<f64>::zeros();
        pcov.set_diagonal(&sigma_lvlh.map(|x| x * x));

        let mut m = na::Matrix6::<f64>::zeros();
        m.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(dcm.transpose() * pcov * dcm));
        self.cov = StateCov::PVCov(m);
    }

    /// Set position uncertainty (1-sigma, meters) in the
    /// gcrf (Geocentric Celestial Reference Frame)
    ///
    /// # Arguments
    ///
    /// * `sigma_gcrf` - 3-vector with 1-sigma position uncertainty in GCRF frame    
    ///
    pub fn set_gcrf_pos_uncertainty(&mut self, sigma_cart: &na::Vector3<f64>) {
        self.cov = StateCov::PVCov({
            let mut m = PVCovType::zeros();
            let mut diag = na::Vector3::<f64>::zeros();
            diag[0] = sigma_cart[0] * sigma_cart[0];
            diag[1] = sigma_cart[1] * sigma_cart[1];
            diag[2] = sigma_cart[2] * sigma_cart[2];
            let mut pcov = na::Matrix3::<f64>::zeros();
            pcov.set_diagonal(&diag);
            m.fixed_view_mut::<3, 3>(0, 0).copy_from(&pcov);
            m
        })
    }

    /// Set velocity uncertainty (1-sigma, meters / second) in the
    /// gcrf (Geocentric Celestial Reference Frame)
    ///
    /// # Arguments
    ///
    /// * `sigma_gcrf` - 3-vector with 1-sigma velocity uncertainty in GCRF frame    
    ///
    pub fn set_gcrf_vel_uncertainty(&mut self, sigma_cart: &na::Vector3<f64>) {
        self.cov = StateCov::PVCov({
            let mut m = PVCovType::zeros();
            let mut diag = na::Vector3::<f64>::zeros();
            diag[0] = sigma_cart[0] * sigma_cart[0];
            diag[1] = sigma_cart[1] * sigma_cart[1];
            diag[2] = sigma_cart[2] * sigma_cart[2];
            let mut pcov = na::Matrix3::<f64>::zeros();
            pcov.set_diagonal(&diag);
            m.fixed_view_mut::<3, 3>(3, 3).copy_from(&pcov);
            m
        })
    }

    ///
    /// Propagate state to a new time
    ///
    /// # Arguments:
    ///
    /// * `time` - Time for which to compute new state
    /// * `settings` - Settings for the propagator
    ///
    /// # Returns:
    ///
    /// New satellite state representing ballistic propgation to new time
    ///
    pub fn propagate(
        &self,
        time: &Instant,
        option_settings: Option<&PropSettings>,
    ) -> Result<Self> {
        let default = orbitprop::PropSettings::default();
        let settings = option_settings.unwrap_or(&default);
        match self.cov {
            // Simple case: do not compute state transition matrix, since covariance is not set
            StateCov::None => {
                let res = orbitprop::propagate(&self.pv, &self.time, time, settings, None)?;
                Ok(Self {
                    time: *time,
                    pv: res.state_end,
                    cov: StateCov::None,
                })
            }
            // Compute state transition matrix & propagate covariance as well
            StateCov::PVCov(cov) => {
                let mut state = na::SMatrix::<f64, 6, 7>::zeros();

                // First row of state is 6-element position & velocity
                state.fixed_view_mut::<6, 1>(0, 0).copy_from(&self.pv);

                // See equation 7.42 of Montenbruck & Gill
                // State transition matrix initializes to identity matrix
                // State transition matrix is columns 1-7 of state (0-based)
                state
                    .fixed_view_mut::<6, 6>(0, 1)
                    .copy_from(&na::Matrix6::<f64>::identity());

                // Propagate
                let res = orbitprop::propagate(&state, &self.time, time, settings, None)?;

                Ok(Self {
                    time: *time,
                    pv: res.state_end.fixed_view::<6, 1>(0, 0).into(),
                    cov: {
                        // Extract state transition matrix from the propagated state
                        let phi = res.state_end.fixed_view::<6, 6>(0, 1);
                        // Evolve the covariance
                        StateCov::PVCov(phi * cov * phi.transpose())
                    },
                })
            }
        }
    }
}

impl std::fmt::Display for SatState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s1 = format!(
            r#"Satellite State
                       Time: {}
              GCRF Position: [{:+8.0}, {:+8.0}, {:+8.0}] m,
              GCRF Velocity: [{:+8.3}, {:+8.3}, {:+8.3}] m/s"#,
            self.time, self.pv[0], self.pv[1], self.pv[2], self.pv[3], self.pv[4], self.pv[5],
        );
        match self.cov {
            StateCov::None => {}
            StateCov::PVCov(cov) => {
                s1.push_str(
                    format!(
                        r#"
            Covariance: {cov:+8.2e}"#
                    )
                    .as_str(),
                );
            }
        }
        write!(f, "{}", s1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_qgcrf2lvlh() -> Result<()> {
        let satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );

        let state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_hours(3.56)), None)?;

        let rz = -1.0 / state2.pos_gcrf().norm() * (state2.qgcrf2lvlh() * state2.pos_gcrf());
        let h = state2.pos_gcrf().cross(&state2.vel_gcrf());
        let ry = -1.0 / h.norm() * (state2.qgcrf2lvlh() * h);
        let rx = 1.0 / state2.vel_gcrf().norm() * (state2.qgcrf2lvlh() * state2.vel_gcrf());

        assert_relative_eq!(rz, na::Vector3::z_axis(), epsilon = 1.0e-6);
        assert_relative_eq!(ry, na::Vector3::y_axis(), epsilon = 1.0e-6);
        assert_relative_eq!(rx, na::Vector3::x_axis(), epsilon = 1.0e-4);

        Ok(())
    }

    #[test]
    fn test_satstate() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_lvlh_pos_uncertainty(&na::vector![1.0, 1.0, 1.0]);
        satstate.set_lvlh_vel_uncertainty(&na::vector![0.01, 0.02, 0.03]);

        let state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_days(0.5)), None)?;

        // Propagate back to original time
        let state0 = state2.propagate(&satstate.time, None)?;

        // Check that propagating backwards in time results in the original state
        assert_abs_diff_eq!(satstate.pos_gcrf(), state0.pos_gcrf(), epsilon = 0.1);
        assert_abs_diff_eq!(satstate.vel_gcrf(), state0.vel_gcrf(), epsilon = 0.001);
        let cov1 = match satstate.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        let cov2 = match state0.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        assert_abs_diff_eq!(cov1, cov2, epsilon = 0.001);

        Ok(())
    }

    #[test]
    fn test_satcov() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_lvlh_pos_uncertainty(&na::vector![1.0, 1.0, 1.0]);

        let _state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_days(1.0)), None)?;

        Ok(())
    }
}
