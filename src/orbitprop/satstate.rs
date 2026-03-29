use crate::frametransform;
use crate::orbitprop;
use crate::orbitprop::PropSettings;
use crate::orbitprop::SatProperties;
use crate::{Frame, Instant};
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

/// An instantaneous velocity change (delta-v) at a specific time
#[derive(Clone, Debug)]
pub struct ImpulsiveManeuver {
    /// Time at which the maneuver is applied
    pub time: Instant,
    /// Delta-v vector in the specified frame [m/s]
    pub delta_v: Vector3,
    /// Coordinate frame for the delta-v vector (GCRF or RIC)
    pub frame: Frame,
}

impl ImpulsiveManeuver {
    pub fn new(time: Instant, delta_v: Vector3, frame: Frame) -> Self {
        Self {
            time,
            delta_v,
            frame,
        }
    }

    /// Compute the delta-v in GCRF given the state at maneuver time
    fn delta_v_gcrf(&self, pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Vector3 {
        match self.frame {
            Frame::GCRF => self.delta_v,
            Frame::RIC => {
                let dcm = frametransform::ric_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.delta_v
            }
            _ => panic!("Unsupported frame for maneuver: {}. Must be GCRF or RIC", self.frame)
        }
    }
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
/// Impulsive maneuvers can be added to the state. When propagating, the
/// propagator automatically segments at each maneuver time and applies
/// the delta-v.
///
#[derive(Clone, Debug)]
pub struct SatState {
    pub time: Instant,
    pub pv: Vector6,
    pub cov: StateCov,
    pub maneuvers: Vec<ImpulsiveManeuver>,
}

impl SatState {
    pub fn from_pv<T: TimeLike>(time: &T, pos: &Vector3, vel: &Vector3) -> Self {
        Self {
            time: time.as_instant(),
            pv: numeris::vector![pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]],
            cov: StateCov::None,
            maneuvers: Vec::new(),
        }
    }

    pub fn pos_gcrf(&self) -> Vector3 {
        self.pv.block::<3, 1>(0, 0)
    }

    pub fn vel_gcrf(&self) -> Vector3 {
        self.pv.block::<3, 1>(3, 0)
    }

    /// Add an impulsive maneuver to the state
    pub fn add_maneuver(&mut self, maneuver: ImpulsiveManeuver) {
        self.maneuvers.push(maneuver);
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
        let z_target = numeris::vector![0.0, 0.0, 1.0];
        let q1 = Quaternion::rotation_between(neg_p, z_target);
        let rotated_h = q1 * (h * -1.0);
        let y_axis = numeris::vector![0.0, 1.0, 0.0];
        let q2 = Quaternion::rotation_between(rotated_h, y_axis);
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

    /// Propagate a single segment (no maneuvers) from current pv/cov to target time
    fn propagate_segment(
        pv: &Vector6,
        cov: &StateCov,
        from: &Instant,
        to: &Instant,
        settings: &PropSettings,
        satprops: Option<&dyn SatProperties>,
    ) -> Result<(Vector6, StateCov)> {
        if from == to {
            return Ok((*pv, cov.clone()));
        }

        match cov {
            StateCov::None => {
                let res = orbitprop::propagate(pv, from, to, settings, satprops)?;
                Ok((res.state_end, StateCov::None))
            }
            StateCov::PVCov(cov_mat) => {
                let mut state = Matrix::<6, 7>::zeros();
                state.set_block(0, 0, pv);
                state.set_block(0, 1, &Matrix6::eye());

                let res = orbitprop::propagate(&state, from, to, settings, satprops)?;

                let new_pv = res.state_end.block::<6, 1>(0, 0);
                let phi = res.state_end.block::<6, 6>(0, 1);
                let new_cov = StateCov::PVCov(phi * *cov_mat * phi.transpose());
                Ok((new_pv, new_cov))
            }
        }
    }

    /// Apply an impulsive maneuver to a state vector
    fn apply_maneuver(pv: &mut Vector6, maneuver: &ImpulsiveManeuver, sign: f64) {
        let pos = pv.block::<3, 1>(0, 0);
        let vel = pv.block::<3, 1>(3, 0);
        let dv = maneuver.delta_v_gcrf(&pos, &vel) * sign;
        pv[3] += dv[0];
        pv[4] += dv[1];
        pv[5] += dv[2];
    }

    ///
    /// Propagate state to a new time
    ///
    /// Automatically segments propagation at impulsive maneuver times,
    /// applying delta-v at each maneuver epoch.
    ///
    /// # Arguments:
    ///
    /// * `time` - Time for which to compute new state
    /// * `settings` - Settings for the propagator
    /// * `satprops` - Optional satellite properties (drag, SRP, thrust)
    ///
    /// # Returns:
    ///
    /// New satellite state representing propagation to new time,
    /// with maneuvers applied at their scheduled times.
    ///
    pub fn propagate(
        &self,
        time: &impl TimeLike,
        option_settings: Option<&PropSettings>,
        satprops: Option<&dyn SatProperties>,
    ) -> Result<Self> {
        let target = time.as_instant();

        if target == self.time {
            return Ok(self.clone());
        }

        let default = orbitprop::PropSettings::default();
        let settings = option_settings.unwrap_or(&default);

        let forward = target > self.time;
        let sign = if forward { 1.0 } else { -1.0 };

        // Collect maneuvers between current time and target, sorted in propagation order
        let (t_min, t_max) = if forward {
            (self.time, target)
        } else {
            (target, self.time)
        };

        let mut active_maneuvers: Vec<&ImpulsiveManeuver> = self
            .maneuvers
            .iter()
            .filter(|m| m.time >= t_min && m.time < t_max)
            .collect();

        if forward {
            active_maneuvers.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        } else {
            // Backward: apply in reverse chronological order
            active_maneuvers.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap());
        }

        // Propagate through segments
        let mut current_pv = self.pv;
        let mut current_cov = self.cov.clone();
        let mut current_time = self.time;

        for maneuver in &active_maneuvers {
            // Propagate to maneuver time
            let (new_pv, new_cov) = Self::propagate_segment(
                &current_pv,
                &current_cov,
                &current_time,
                &maneuver.time,
                settings,
                satprops,
            )?;
            current_pv = new_pv;
            current_cov = new_cov;
            current_time = maneuver.time;

            // Apply delta-v
            Self::apply_maneuver(&mut current_pv, maneuver, sign);
        }

        // Propagate from last maneuver (or start) to target
        let (final_pv, final_cov) = Self::propagate_segment(
            &current_pv,
            &current_cov,
            &current_time,
            &target,
            settings,
            satprops,
        )?;

        Ok(Self {
            time: target,
            pv: final_pv,
            cov: final_cov,
            maneuvers: self.maneuvers.clone(),
        })
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
        if !self.maneuvers.is_empty() {
            s1.push_str(&format!(
                "\n       Maneuvers: {}",
                self.maneuvers.len()
            ));
        }
        write!(f, "{}", s1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts;
    use crate::Duration;

    #[test]
    fn test_qgcrf2lvlh() -> Result<()> {
        let satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &numeris::vector![consts::GEO_R, 0.0, 0.0],
            &numeris::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );

        let state2 =
            satstate.propagate(&(satstate.time + Duration::from_hours(3.56)), None, None)?;

        let rz = (state2.qgcrf2lvlh() * state2.pos_gcrf()) * (-1.0 / state2.pos_gcrf().norm());
        let h = state2.pos_gcrf().cross(&state2.vel_gcrf());
        let ry = (state2.qgcrf2lvlh() * h) * (-1.0 / h.norm());
        let rx = (state2.qgcrf2lvlh() * state2.vel_gcrf()) * (1.0 / state2.vel_gcrf().norm());

        let z_axis = numeris::vector![0.0, 0.0, 1.0];
        let y_axis = numeris::vector![0.0, 1.0, 0.0];
        let x_axis = numeris::vector![1.0, 0.0, 0.0];
        assert!((rz - z_axis).norm() < 1.0e-6);
        assert!((ry - y_axis).norm() < 1.0e-6);
        assert!((rx - x_axis).norm() < 1.0e-4);

        Ok(())
    }

    #[test]
    fn test_satstate() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &numeris::vector![consts::GEO_R, 0.0, 0.0],
            &numeris::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_lvlh_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0]);
        satstate.set_lvlh_vel_uncertainty(&numeris::vector![0.01, 0.02, 0.03]);

        let state2 =
            satstate.propagate(&(satstate.time + Duration::from_days(0.5)), None, None)?;

        // Propagate back to original time
        let state0 = state2.propagate(&satstate.time, None, None)?;

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
            &numeris::vector![consts::GEO_R, 0.0, 0.0],
            &numeris::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_lvlh_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0]);

        let _state2 =
            satstate.propagate(&(satstate.time + Duration::from_days(1.0)), None, None)?;

        Ok(())
    }

    #[test]
    fn test_zero_duration_propagation() -> Result<()> {
        let satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &numeris::vector![consts::GEO_R, 0.0, 0.0],
            &numeris::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );

        let state2 = satstate.propagate(&satstate.time, None, None)?;

        assert!((satstate.pos_gcrf() - state2.pos_gcrf()).norm() < 1e-15);
        assert!((satstate.vel_gcrf() - state2.vel_gcrf()).norm() < 1e-15);
        assert_eq!(satstate.time, state2.time);

        Ok(())
    }

    #[test]
    fn test_zero_duration_propagation_with_cov() -> Result<()> {
        let mut satstate = SatState::from_pv(
            &Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap(),
            &numeris::vector![consts::GEO_R, 0.0, 0.0],
            &numeris::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_lvlh_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0]);

        let state2 = satstate.propagate(&satstate.time, None, None)?;

        assert!((satstate.pos_gcrf() - state2.pos_gcrf()).norm() < 1e-15);
        assert!((satstate.vel_gcrf() - state2.vel_gcrf()).norm() < 1e-15);

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

    #[test]
    fn test_impulsive_maneuver_gcrf() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let t_burn = t0 + Duration::from_hours(1.0);
        let t_end = t0 + Duration::from_hours(2.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();

        // State without maneuver
        let sat_no_burn = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );
        let state_no_burn = sat_no_burn.propagate(&t_end, None, None)?;

        // State with prograde maneuver (approximate: +Y at t=0 is roughly prograde)
        let mut sat_burn = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );
        sat_burn.add_maneuver(ImpulsiveManeuver::new(
            t_burn,
            numeris::vector![0.0, 0.0, 10.0], // 10 m/s in +Z GCRF
            Frame::GCRF,
        ));

        let state_burn = sat_burn.propagate(&t_end, None, None)?;

        // Maneuver should produce a different state
        let pos_diff = (state_burn.pos_gcrf() - state_no_burn.pos_gcrf()).norm();
        assert!(
            pos_diff > 100.0,
            "Impulsive maneuver should change position: diff = {} m",
            pos_diff
        );

        // Maneuvers should persist on the result
        assert_eq!(state_burn.maneuvers.len(), 1);

        Ok(())
    }

    #[test]
    fn test_impulsive_maneuver_ric() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let t_burn = t0 + Duration::from_hours(1.0);
        let t_end = t0 + Duration::from_hours(3.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();

        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );
        let sat_no_burn = sat.clone();

        // 10 m/s in-track (prograde) in RIC frame
        sat.add_maneuver(ImpulsiveManeuver::new(
            t_burn,
            numeris::vector![0.0, 10.0, 0.0], // [radial, in-track, cross-track]
            Frame::RIC,
        ));

        let state_burn = sat.propagate(&t_end, None, None)?;
        let state_no_burn = sat_no_burn.propagate(&t_end, None, None)?;

        // Prograde burn should raise the orbit
        let r_burn = state_burn.pos_gcrf().norm();
        let r_no_burn = state_no_burn.pos_gcrf().norm();

        let pos_diff = (state_burn.pos_gcrf() - state_no_burn.pos_gcrf()).norm();
        assert!(
            pos_diff > 1000.0,
            "10 m/s prograde should produce large position diff: {} m",
            pos_diff
        );

        // After a prograde burn, semi-major axis increases, so on average radius is larger
        // (though instantaneous radius depends on where in the orbit we are)
        // Just check the states are meaningfully different
        assert!(
            (r_burn - r_no_burn).abs() > 100.0,
            "Radius should differ: burn={}, no_burn={}",
            r_burn,
            r_no_burn
        );

        Ok(())
    }

    #[test]
    fn test_impulsive_maneuver_backward() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let t_burn = t0 + Duration::from_hours(1.0);
        let t_end = t0 + Duration::from_hours(2.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();

        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );
        sat.add_maneuver(ImpulsiveManeuver::new(
            t_burn,
            numeris::vector![0.0, 0.0, 5.0], // 5 m/s in +Z GCRF
            Frame::GCRF,
        ));

        // Propagate forward past maneuver
        let state_fwd = sat.propagate(&t_end, None, None)?;

        // Propagate backward to original time
        let state_back = state_fwd.propagate(&t0, None, None)?;

        // Should recover original state
        assert!(
            (sat.pos_gcrf() - state_back.pos_gcrf()).norm() < 1.0,
            "Backward propagation should recover original position: diff = {} m",
            (sat.pos_gcrf() - state_back.pos_gcrf()).norm()
        );
        assert!(
            (sat.vel_gcrf() - state_back.vel_gcrf()).norm() < 0.01,
            "Backward propagation should recover original velocity: diff = {} m/s",
            (sat.vel_gcrf() - state_back.vel_gcrf()).norm()
        );

        Ok(())
    }

    #[test]
    fn test_multiple_maneuvers() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let t_burn1 = t0 + Duration::from_hours(1.0);
        let t_burn2 = t0 + Duration::from_hours(2.0);
        let t_end = t0 + Duration::from_hours(3.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();

        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );

        // Two burns in opposite Z directions should roughly cancel
        sat.add_maneuver(ImpulsiveManeuver::new(
            t_burn1,
            numeris::vector![0.0, 0.0, 10.0],
            Frame::GCRF,
        ));
        sat.add_maneuver(ImpulsiveManeuver::new(
            t_burn2,
            numeris::vector![0.0, 0.0, -10.0],
            Frame::GCRF,
        ));

        let sat_no_burn = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );

        let state_burn = sat.propagate(&t_end, None, None)?;
        let state_no_burn = sat_no_burn.propagate(&t_end, None, None)?;

        // Opposite burns don't perfectly cancel (nonlinear dynamics),
        // but the difference should be much smaller than a single 10 m/s burn
        let pos_diff = (state_burn.pos_gcrf() - state_no_burn.pos_gcrf()).norm();
        // Opposing burns at different orbital positions don't perfectly cancel
        // due to nonlinear dynamics, but should be much less than a single 10 m/s burn
        // over 3 hours (which would produce ~100 km difference)
        assert!(
            pos_diff < 50000.0,
            "Opposing burns should mostly cancel: diff = {} m",
            pos_diff
        );

        assert_eq!(state_burn.maneuvers.len(), 2);

        Ok(())
    }
}
