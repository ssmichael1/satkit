use crate::orbitprop;
use crate::orbitprop::PropSettings;
use crate::Instant;
use crate::TimeLike;

use crate::mathtypes::*;

use anyhow::Result;

type PVCovType = Matrix<6, 6>;

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
    pub pv: Vector6,
    pub cov: StateCov,
}

/// Compute a quaternion that rotates vector `from` to align with vector `to`.
/// Returns `None` if either vector is zero-length or they are exactly anti-parallel.
fn rotation_between(from: &Vector3, to: &Vector3) -> Option<Quaternion> {
    let from_n = from.norm();
    let to_n = to.norm();
    if from_n < 1e-15 || to_n < 1e-15 {
        return None;
    }
    let a = Vector3::from_array([from[0] / from_n, from[1] / from_n, from[2] / from_n]);
    let b = Vector3::from_array([to[0] / to_n, to[1] / to_n, to[2] / to_n]);
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    if dot > 1.0 - 1e-12 {
        // Vectors are parallel
        return Some(Quaternion::identity());
    }
    if dot < -1.0 + 1e-12 {
        // Vectors are anti-parallel — no unique rotation
        return None;
    }

    let axis = a.cross(&b).normalize();
    let angle = dot.acos();
    Some(Quaternion::from_axis_angle(axis, angle))
}

impl SatState {
    pub fn from_pv<T: TimeLike>(time: &T, pos: &Vector3, vel: &Vector3) -> Self {
        Self {
            time: time.as_instant(),
            pv: Vector::from_array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]),
            cov: StateCov::None,
        }
    }

    pub fn pos_gcrf(&self) -> Vector3 {
        self.pv.block::<3, 1>(0, 0)
    }

    pub fn vel_gcrf(&self) -> Vector3 {
        self.pv.block::<3, 1>(3, 0)
    }

    /// Set covariance
    ///
    /// # Arguments
    ///
    /// * `cov` -  Covariance matrix.  6x6 or larger if including terms like drag.
    ///   Upper-left 6x6 is covariance for position & velocity, in units of
    ///   meters and meters / second
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
    pub fn qgcrf2lvlh(&self) -> Quaternion {
        let p = self.pos_gcrf();
        let v = self.vel_gcrf();
        let h = p.cross(&v);
        let neg_p = p * -1.0;
        let neg_h_dir = Vector3::from_array([0.0, 0.0, 1.0]);
        let q1 = rotation_between(&neg_p, &neg_h_dir).unwrap();
        let rotated_h = q1 * (h * -1.0);
        let y_axis = Vector3::from_array([0.0, 1.0, 0.0]);
        let q2 = rotation_between(&rotated_h, &y_axis).unwrap();
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
    pub fn set_lvlh_pos_uncertainty(&mut self, sigma_lvlh: &Vector3) {
        let dcm = self.qgcrf2lvlh().to_rotation_matrix();

        let mut pcov = Matrix3::zeros();
        pcov[(0, 0)] = sigma_lvlh[0] * sigma_lvlh[0];
        pcov[(1, 1)] = sigma_lvlh[1] * sigma_lvlh[1];
        pcov[(2, 2)] = sigma_lvlh[2] * sigma_lvlh[2];

        let mut m = Matrix6::zeros();
        m.set_block(0, 0, &(dcm.transpose() * pcov * dcm));
        self.cov = StateCov::PVCov(m);
    }

    /// Set velocity uncertainty (1-sigma, meters/second) in the
    /// lvlh (local-vertical, local-horizontal) frame
    ///
    /// # Arguments
    ///
    /// * `sigma_lvlh` - 3-vector with 1-sigma velocity uncertainty in LVLH frame
    ///
    pub fn set_lvlh_vel_uncertainty(&mut self, sigma_lvlh: &Vector3) {
        let dcm = self.qgcrf2lvlh().to_rotation_matrix();

        let mut pcov = Matrix3::zeros();
        pcov[(0, 0)] = sigma_lvlh[0] * sigma_lvlh[0];
        pcov[(1, 1)] = sigma_lvlh[1] * sigma_lvlh[1];
        pcov[(2, 2)] = sigma_lvlh[2] * sigma_lvlh[2];

        let mut m = Matrix6::zeros();
        m.set_block(3, 3, &(dcm.transpose() * pcov * dcm));
        self.cov = StateCov::PVCov(m);
    }

    /// Set position uncertainty (1-sigma, meters) in the
    /// gcrf (Geocentric Celestial Reference Frame)
    ///
    /// # Arguments
    ///
    /// * `sigma_gcrf` - 3-vector with 1-sigma position uncertainty in GCRF frame
    ///
    pub fn set_gcrf_pos_uncertainty(&mut self, sigma_cart: &Vector3) {
        self.cov = StateCov::PVCov({
            let mut m = PVCovType::zeros();
            let mut pcov = Matrix3::zeros();
            pcov[(0, 0)] = sigma_cart[0] * sigma_cart[0];
            pcov[(1, 1)] = sigma_cart[1] * sigma_cart[1];
            pcov[(2, 2)] = sigma_cart[2] * sigma_cart[2];
            m.set_block(0, 0, &pcov);
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
    pub fn set_gcrf_vel_uncertainty(&mut self, sigma_cart: &Vector3) {
        self.cov = StateCov::PVCov({
            let mut m = PVCovType::zeros();
            let mut pcov = Matrix3::zeros();
            pcov[(0, 0)] = sigma_cart[0] * sigma_cart[0];
            pcov[(1, 1)] = sigma_cart[1] * sigma_cart[1];
            pcov[(2, 2)] = sigma_cart[2] * sigma_cart[2];
            m.set_block(3, 3, &pcov);
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
        time: &impl TimeLike,
        option_settings: Option<&PropSettings>,
    ) -> Result<Self> {
        let time = time.as_instant();

        // Handle zero-duration propagation: return current state unchanged
        // This avoids numerical issues in the ODE integrator when dt=0
        if time == self.time {
            return Ok(self.clone());
        }

        let default = orbitprop::PropSettings::default();
        let settings = option_settings.unwrap_or(&default);
        match self.cov {
            // Simple case: do not compute state transition matrix, since covariance is not set
            StateCov::None => {
                let res = orbitprop::propagate(&self.pv, &self.time, &time, settings, None)?;
                Ok(Self {
                    time,
                    pv: res.state_end,
                    cov: StateCov::None,
                })
            }
            // Compute state transition matrix & propagate covariance as well
            StateCov::PVCov(cov) => {
                let mut state = Matrix::<6, 7>::zeros();

                // First row of state is 6-element position & velocity
                state.set_block(0, 0, &self.pv);

                // See equation 7.42 of Montenbruck & Gill
                // State transition matrix initializes to identity matrix
                // State transition matrix is columns 1-7 of state (0-based)
                state.set_block(0, 1, &Matrix6::eye());

                // Propagate
                let res = orbitprop::propagate(&state, &self.time, &time, settings, None)?;

                Ok(Self {
                    time,
                    pv: res.state_end.block::<6, 1>(0, 0),
                    cov: {
                        // Extract state transition matrix from the propagated state
                        let phi = res.state_end.block::<6, 6>(0, 1);
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
            StateCov::PVCov(ref cov) => {
                s1.push_str(
                    format!(
                        r#"
            Covariance: {cov:?}"#
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

    #[test]
    fn test_qgcrf2lvlh() -> Result<()> {
        let satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &Vector::from_array([consts::GEO_R, 0.0, 0.0]),
            &Vector::from_array([0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0]),
        );

        let state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_hours(3.56)), None)?;

        let rz = (state2.qgcrf2lvlh() * state2.pos_gcrf()) * (-1.0 / state2.pos_gcrf().norm());
        let h = state2.pos_gcrf().cross(&state2.vel_gcrf());
        let ry = (state2.qgcrf2lvlh() * h) * (-1.0 / h.norm());
        let rx = (state2.qgcrf2lvlh() * state2.vel_gcrf()) * (1.0 / state2.vel_gcrf().norm());

        let z_axis = Vector3::from_array([0.0, 0.0, 1.0]);
        let y_axis = Vector3::from_array([0.0, 1.0, 0.0]);
        let x_axis = Vector3::from_array([1.0, 0.0, 0.0]);
        assert!((rz - z_axis).norm() < 1.0e-6);
        assert!((ry - y_axis).norm() < 1.0e-6);
        assert!((rx - x_axis).norm() < 1.0e-4);

        Ok(())
    }

    #[test]
    fn test_satstate() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &Vector::from_array([consts::GEO_R, 0.0, 0.0]),
            &Vector::from_array([0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0]),
        );
        satstate.set_lvlh_pos_uncertainty(&Vector::from_array([1.0, 1.0, 1.0]));
        satstate.set_lvlh_vel_uncertainty(&Vector::from_array([0.01, 0.02, 0.03]));

        let state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_days(0.5)), None)?;

        // Propagate back to original time
        let state0 = state2.propagate(&satstate.time, None)?;

        // Check that propagating backwards in time results in the original state
        assert!((satstate.pos_gcrf() - state0.pos_gcrf()).norm() < 0.1);
        assert!((satstate.vel_gcrf() - state0.vel_gcrf()).norm() < 0.001);
        let cov1 = match satstate.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        let cov2 = match state0.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        assert!((cov1 - cov2).norm_inf() < 0.001);

        Ok(())
    }

    #[test]
    fn test_satcov() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &Vector::from_array([consts::GEO_R, 0.0, 0.0]),
            &Vector::from_array([0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0]),
        );
        satstate.set_lvlh_pos_uncertainty(&Vector::from_array([1.0, 1.0, 1.0]));

        let _state2 =
            satstate.propagate(&(satstate.time + crate::Duration::from_days(1.0)), None)?;

        Ok(())
    }

    #[test]
    fn test_zero_duration_propagation() -> Result<()> {
        // Test that propagating with dt=0 returns the same state
        let satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &Vector::from_array([consts::GEO_R, 0.0, 0.0]),
            &Vector::from_array([0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0]),
        );

        // Propagate to the same time (zero duration)
        let state2 = satstate.propagate(&satstate.time, None)?;

        // Verify the state is unchanged
        assert!((satstate.pos_gcrf() - state2.pos_gcrf()).norm() < 1e-15);
        assert!((satstate.vel_gcrf() - state2.vel_gcrf()).norm() < 1e-15);
        assert_eq!(satstate.time, state2.time);

        Ok(())
    }

    #[test]
    fn test_zero_duration_propagation_with_cov() -> Result<()> {
        // Test that propagating with dt=0 returns the same state including covariance
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &Vector::from_array([consts::GEO_R, 0.0, 0.0]),
            &Vector::from_array([0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0]),
        );
        satstate.set_lvlh_pos_uncertainty(&Vector::from_array([1.0, 1.0, 1.0]));

        // Propagate to the same time (zero duration)
        let state2 = satstate.propagate(&satstate.time, None)?;

        // Verify the state is unchanged
        assert!((satstate.pos_gcrf() - state2.pos_gcrf()).norm() < 1e-15);
        assert!((satstate.vel_gcrf() - state2.vel_gcrf()).norm() < 1e-15);

        // Verify covariance is unchanged
        let cov1 = match satstate.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        let cov2 = match state2.cov() {
            StateCov::PVCov(v) => v,
            StateCov::None => anyhow::bail!("cov is not none"),
        };
        assert!((cov1 - cov2).norm_inf() < 1e-15);

        Ok(())
    }
}
