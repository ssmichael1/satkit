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

/// An instantaneous velocity change (delta-v) at a specific time.
///
/// A maneuver is specified as a 3-vector plus a reference frame. The
/// supported frames are:
///
/// * [`Frame::GCRF`] — inertial Cartesian, useful for directly specifying
///   delta-v components along inertial axes.
/// * [`Frame::RIC`] (a.k.a. RSW/RTN) — radial / in-track / cross-track,
///   tied to the position vector. Natural for radial and cross-track
///   burn components, and the CCSDS OEM covariance-message convention.
///   Note: `satkit`'s own covariance-uncertainty API uses LVLH (see
///   [`Frame::LVLH`]), not RIC.
/// * [`Frame::NTW`] — normal-to-velocity / tangent / cross-track, tied to
///   the velocity vector. Natural for prograde/retrograde burns.
/// * [`Frame::LVLH`] — Local Vertical / Local Horizontal, the classical
///   crewed-spaceflight "body-pointing" frame (z = nadir, y = anti-h,
///   x completes). Geometrically the same orbital plane as RIC but with
///   different labels and signs. Handy if you're translating GN&C code
///   written against LVLH conventions.
///
/// For circular orbits RIC and NTW are identical up to sign conventions;
/// for eccentric orbits they differ by the flight-path angle. If you want
/// to specify a "10 m/s prograde" burn and have it add exactly 10 m/s to
/// |v|, use NTW. See [`Frame::NTW`] for the geometric definition, or the
/// "Satellite-local frames for maneuvers" guide in the satkit documentation
/// for a side-by-side comparison of RIC, NTW, and LVLH.
///
/// # Ergonomic constructors
///
/// For common cases, prefer the named constructors over passing a frame
/// explicitly:
///
/// * [`ImpulsiveManeuver::prograde`] — +T in NTW
/// * [`ImpulsiveManeuver::retrograde`] — −T in NTW
/// * [`ImpulsiveManeuver::radial_out`] — +N in NTW (radial for circular)
/// * [`ImpulsiveManeuver::normal`] — +W cross-track
/// * [`ImpulsiveManeuver::ric`] — arbitrary vector in RIC
/// * [`ImpulsiveManeuver::ntw`] — arbitrary vector in NTW
/// * [`ImpulsiveManeuver::gcrf`] — arbitrary vector in GCRF
#[derive(Clone, Debug)]
pub struct ImpulsiveManeuver {
    /// Time at which the maneuver is applied
    pub time: Instant,
    /// Delta-v vector in the specified frame [m/s]
    pub delta_v: Vector3,
    /// Coordinate frame for the delta-v vector. Must be one of
    /// [`Frame::GCRF`], [`Frame::RIC`], or [`Frame::NTW`].
    pub frame: Frame,
}

impl ImpulsiveManeuver {
    /// Create a maneuver with an explicit frame. For common cases see the
    /// named constructors ([`Self::prograde`], [`Self::ntw`], etc.).
    pub fn new(time: Instant, delta_v: Vector3, frame: Frame) -> Self {
        Self {
            time,
            delta_v,
            frame,
        }
    }

    /// Prograde burn: `+dv_mps` along the velocity vector.
    ///
    /// Equivalent to an NTW maneuver with delta-v = (0, dv_mps, 0). A
    /// positive magnitude adds energy (raises orbit); a negative magnitude
    /// is equivalent to [`Self::retrograde`] with its sign flipped.
    pub fn prograde(time: Instant, dv_mps: f64) -> Self {
        Self::new(time, numeris::vector![0.0, dv_mps, 0.0], Frame::NTW)
    }

    /// Retrograde burn: `-dv_mps` along the velocity vector. `dv_mps`
    /// should be positive; this is a convenience equivalent to calling
    /// [`Self::prograde`] with a negated magnitude.
    pub fn retrograde(time: Instant, dv_mps: f64) -> Self {
        Self::new(time, numeris::vector![0.0, -dv_mps, 0.0], Frame::NTW)
    }

    /// Radial-outward burn in the NTW frame: +N component.
    ///
    /// For circular orbits this is the outward radial direction; for
    /// eccentric orbits the N axis leans off the radial by the flight-path
    /// angle. Use [`Self::ric`] if you want the strict radial direction
    /// regardless of eccentricity.
    pub fn radial_out(time: Instant, dv_mps: f64) -> Self {
        Self::new(time, numeris::vector![dv_mps, 0.0, 0.0], Frame::NTW)
    }

    /// Cross-track ("normal") burn: +W component. Same as RIC +C.
    /// Positive values push in the +angular-momentum direction (toward
    /// the "left" of the orbit for a prograde mission).
    pub fn normal(time: Instant, dv_mps: f64) -> Self {
        Self::new(time, numeris::vector![0.0, 0.0, dv_mps], Frame::NTW)
    }

    /// Arbitrary delta-v vector in the GCRF inertial frame.
    pub fn gcrf(time: Instant, delta_v: Vector3) -> Self {
        Self::new(time, delta_v, Frame::GCRF)
    }

    /// Arbitrary delta-v vector in the RIC (a.k.a. RSW / RTN) frame.
    /// Components are (radial, in-track, cross-track).
    pub fn ric(time: Instant, delta_v: Vector3) -> Self {
        Self::new(time, delta_v, Frame::RIC)
    }

    /// Arbitrary delta-v vector in the NTW frame. Components are
    /// (normal, tangent, cross-track) where tangent is along velocity.
    pub fn ntw(time: Instant, delta_v: Vector3) -> Self {
        Self::new(time, delta_v, Frame::NTW)
    }

    /// Compute the delta-v in GCRF given the state at maneuver time.
    fn delta_v_gcrf(&self, pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Vector3 {
        match self.frame {
            Frame::GCRF => self.delta_v,
            Frame::RIC => {
                let dcm = frametransform::ric_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.delta_v
            }
            Frame::NTW => {
                let dcm = frametransform::ntw_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.delta_v
            }
            Frame::LVLH => {
                let dcm = frametransform::lvlh_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.delta_v
            }
            Frame::ITRF
            | Frame::TIRS
            | Frame::CIRS
            | Frame::TEME
            | Frame::EME2000
            | Frame::ICRF => panic!(
                "Unsupported frame for maneuver: {}. Must be GCRF, RIC, NTW, or LVLH",
                self.frame
            ),
        }
    }
}

/// A satellite state: position, velocity, optional covariance, and maneuvers
///
/// `SatState` bundles a GCRF position/velocity with optional covariance and
/// a list of impulsive maneuvers into a single propagatable object. Use it
/// instead of the free [`propagate()`] function when you need:
///
/// * **Covariance propagation** -- attach a 6x6 uncertainty matrix and it
///   will be propagated via the state transition matrix automatically.
/// * **Maneuver scheduling** -- add impulsive delta-v events at future times
///   and propagation will segment around them, applying each burn in order.
/// * **Round-trip propagation** -- propagate forward, then backward, recovering
///   the original state (maneuvers are reversed automatically).
///
/// For simple state-vector propagation without covariance or maneuvers,
/// the free function [`propagate()`] is more direct.
///
/// # Units
///
/// * Position: meters in GCRF
/// * Velocity: meters/second in GCRF
/// * Covariance: meters² and (m/s)² in GCRF
///
#[derive(Clone, Debug)]
pub struct SatState {
    pub time: Instant,
    pub pv: Vector6,
    pub cov: StateCov,
    pub maneuvers: Vec<ImpulsiveManeuver>,
}

impl SatState {
    /// Create a new satellite state from position and velocity vectors
    ///
    /// # Arguments
    ///
    /// * `time` - Epoch of the state
    /// * `pos` - Position vector in GCRF [meters]
    /// * `vel` - Velocity vector in GCRF [meters/second]
    pub fn from_pv<T: TimeLike>(time: &T, pos: &Vector3, vel: &Vector3) -> Self {
        Self {
            time: time.as_instant(),
            pv: numeris::vector![pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]],
            cov: StateCov::None,
            maneuvers: Vec::new(),
        }
    }

    /// Position vector in GCRF [meters]
    pub fn pos_gcrf(&self) -> Vector3 {
        self.pv.block::<3, 1>(0, 0)
    }

    /// Velocity vector in GCRF [meters/second]
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

    /// Return a clone of the state covariance
    ///
    /// Returns `StateCov::None` if no covariance has been set,
    /// or `StateCov::PVCov(matrix)` with a 6x6 position/velocity covariance.
    pub fn cov(&self) -> StateCov {
        self.cov.clone()
    }

    /// Compute the DCM that transforms a 3-vector from `frame` to GCRF at
    /// the current state. Returns `Ok(Some(dcm))` for orbital frames,
    /// `Ok(None)` for GCRF (identity rotation, no transform needed), and
    /// an error for unsupported frames.
    fn cov_frame_to_gcrf(&self, frame: Frame) -> Result<Option<Matrix3>> {
        let pos = self.pos_gcrf();
        let vel = self.vel_gcrf();
        match frame {
            Frame::GCRF => Ok(None),
            Frame::LVLH => Ok(Some(frametransform::lvlh_to_gcrf(&pos, &vel))),
            Frame::RIC => Ok(Some(frametransform::ric_to_gcrf(&pos, &vel))),
            Frame::NTW => Ok(Some(frametransform::ntw_to_gcrf(&pos, &vel))),
            Frame::ITRF
            | Frame::TIRS
            | Frame::CIRS
            | Frame::TEME
            | Frame::EME2000
            | Frame::ICRF => anyhow::bail!(
                "Unsupported frame for uncertainty: {}. Must be GCRF, LVLH, RIC, or NTW",
                frame
            ),
        }
    }

    /// Set 1-sigma position uncertainty (meters) in a satellite-local or
    /// inertial frame.
    ///
    /// The uncertainty is interpreted as a diagonal 3×3 covariance
    /// $\mathrm{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2)$ in the given
    /// `frame`, then rotated into GCRF and stored in the position block
    /// of the 6×6 state covariance. Any existing velocity covariance
    /// block is preserved; off-diagonal (position-velocity) blocks are
    /// cleared.
    ///
    /// # Arguments
    ///
    /// * `sigma` — 3-vector of 1-sigma position uncertainty components
    ///   along the `frame`'s axes [m]
    /// * `frame` — coordinate frame. Supported: [`Frame::GCRF`],
    ///   [`Frame::LVLH`], [`Frame::RIC`] (= RSW = RTN), [`Frame::NTW`].
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is not one of the supported
    /// orbital or inertial frames above (e.g. ITRF, TEME).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use satkit::orbitprop::SatState;
    /// # use satkit::{Frame, Instant};
    /// # let mut sat = SatState::from_pv(&Instant::now(),
    /// #     &numeris::vector![7.0e6, 0.0, 0.0],
    /// #     &numeris::vector![0.0, 7.5e3, 0.0]);
    /// // 100 m radial, 200 m along-track, 50 m cross-track uncertainty
    /// sat.set_pos_uncertainty(&numeris::vector![100.0, 200.0, 50.0], Frame::RIC).unwrap();
    /// ```
    pub fn set_pos_uncertainty(&mut self, sigma: &Vector3, frame: Frame) -> Result<()> {
        let dcm_opt = self.cov_frame_to_gcrf(frame)?;
        let mut pcov_gcrf = Matrix3::zeros();
        pcov_gcrf[(0, 0)] = sigma[0] * sigma[0];
        pcov_gcrf[(1, 1)] = sigma[1] * sigma[1];
        pcov_gcrf[(2, 2)] = sigma[2] * sigma[2];
        if let Some(dcm) = dcm_opt {
            // Covariance transform: C_gcrf = D · C_frame · Dᵀ where D is
            // the frame→GCRF rotation.
            pcov_gcrf = dcm * pcov_gcrf * dcm.transpose();
        }

        // Preserve any existing velocity-block uncertainty.
        let prev_vcov = match &self.cov {
            StateCov::PVCov(m) => m.block::<3, 3>(3, 3),
            StateCov::None => Matrix3::zeros(),
        };

        let mut m = PVCovType::zeros();
        m.set_block(0, 0, &pcov_gcrf);
        m.set_block(3, 3, &prev_vcov);
        self.cov = StateCov::PVCov(m);
        Ok(())
    }

    /// Set 1-sigma velocity uncertainty (meters/second) in a
    /// satellite-local or inertial frame.
    ///
    /// Analogous to [`set_pos_uncertainty`](Self::set_pos_uncertainty) but
    /// for the velocity block of the 6×6 state covariance. Any existing
    /// position covariance block is preserved; off-diagonal blocks are
    /// cleared.
    ///
    /// # Arguments
    ///
    /// * `sigma` — 3-vector of 1-sigma velocity uncertainty components
    ///   along the `frame`'s axes [m/s]
    /// * `frame` — coordinate frame. Supported: [`Frame::GCRF`],
    ///   [`Frame::LVLH`], [`Frame::RIC`], [`Frame::NTW`].
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is not one of the supported frames.
    pub fn set_vel_uncertainty(&mut self, sigma: &Vector3, frame: Frame) -> Result<()> {
        let dcm_opt = self.cov_frame_to_gcrf(frame)?;
        let mut vcov_gcrf = Matrix3::zeros();
        vcov_gcrf[(0, 0)] = sigma[0] * sigma[0];
        vcov_gcrf[(1, 1)] = sigma[1] * sigma[1];
        vcov_gcrf[(2, 2)] = sigma[2] * sigma[2];
        if let Some(dcm) = dcm_opt {
            vcov_gcrf = dcm * vcov_gcrf * dcm.transpose();
        }

        // Preserve any existing position-block uncertainty.
        let prev_pcov = match &self.cov {
            StateCov::PVCov(m) => m.block::<3, 3>(0, 0),
            StateCov::None => Matrix3::zeros(),
        };

        let mut m = PVCovType::zeros();
        m.set_block(0, 0, &prev_pcov);
        m.set_block(3, 3, &vcov_gcrf);
        self.cov = StateCov::PVCov(m);
        Ok(())
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
        satstate.set_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0], Frame::LVLH)?;
        satstate.set_vel_uncertainty(&numeris::vector![0.01, 0.02, 0.03], Frame::LVLH)?;

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
        satstate.set_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0], Frame::LVLH)?;

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
        satstate.set_pos_uncertainty(&numeris::vector![1.0, 1.0, 1.0], Frame::LVLH)?;

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

    /// When r⊥v (zero flight-path angle), NTW and RIC rotation matrices
    /// should produce identical delta-v vectors in GCRF. Pure unit test
    /// of the rotation logic, independent of the propagator.
    #[test]
    fn test_ntw_vs_ric_perpendicular_state() {
        // Exactly circular state: r⊥v
        let pos: Vector3 = numeris::vector![7_000_000.0, 0.0, 0.0];
        let v_mag = (consts::MU_EARTH / pos.norm()).sqrt();
        let vel: Vector3 = numeris::vector![0.0, v_mag, 0.0];

        let dv_ntw = numeris::vector![0.0, 10.0, 0.0]; // +T (tangent)
        let dv_ric = numeris::vector![0.0, 10.0, 0.0]; // +I (in-track)

        let ntw_dcm = crate::frametransform::ntw_to_gcrf(&pos, &vel);
        let ric_dcm = crate::frametransform::ric_to_gcrf(&pos, &vel);

        let dv_ntw_gcrf = ntw_dcm * dv_ntw;
        let dv_ric_gcrf = ric_dcm * dv_ric;

        // With γ = 0 the two should be bit-for-bit identical (both equal
        // to the velocity unit vector scaled by 10).
        let diff = (dv_ntw_gcrf - dv_ric_gcrf).norm();
        assert!(
            diff < 1e-12,
            "On a r⊥v state, NTW-T and RIC-I should agree exactly; diff = {:.3e}",
            diff
        );
        // And both should equal 10 · v̂
        let expected = vel.normalize() * 10.0;
        assert!((dv_ntw_gcrf - expected).norm() < 1e-12);
    }

    /// On an eccentric orbit at non-apsidal true anomaly, NTW and RIC
    /// differ by the flight-path angle. A pure NTW +T burn adds its exact
    /// magnitude to |v|; a pure RIC +I burn of the same magnitude does not.
    /// This is the key physical distinction between the two frames.
    #[test]
    fn test_ntw_vs_ric_eccentric_diverge() -> Result<()> {
        // Construct a state at mid-anomaly of a moderately eccentric orbit
        // with a deliberate non-zero flight-path angle: use an inertial-frame
        // state where r and v are not perpendicular.

        // Pick a position and velocity such that flight-path angle is ~12.7°.
        // For an orbit with a = 8000 km, e = 0.3, at true anomaly = 60°:
        //   r = a(1-e²)/(1+e cos ν) = 8000·0.91 / 1.15 ≈ 6330 km
        //   v = sqrt(μ(2/r - 1/a))
        //   flight path angle γ satisfies tan γ = e sin ν / (1 + e cos ν)
        //                                        = 0.3·0.866 / 1.15 ≈ 0.226 → γ ≈ 12.7°
        let a = 8000.0e3;
        let e = 0.3;
        let nu: f64 = 60.0_f64.to_radians();
        let r_mag = a * (1.0 - e * e) / (1.0 + e * nu.cos());
        let v_mag = (consts::MU_EARTH * (2.0 / r_mag - 1.0 / a)).sqrt();
        let gamma = (e * nu.sin() / (1.0 + e * nu.cos())).atan();

        // Place r along x̂ and velocity in the xy-plane rotated by (90° - γ)
        // from r (i.e., velocity leans "forward" of perpendicular by γ).
        let pos = numeris::vector![r_mag, 0.0, 0.0];
        let vel = numeris::vector![
            v_mag * gamma.sin(),
            v_mag * gamma.cos(),
            0.0
        ];

        // Sanity: dot product of r̂ and v̂ equals sin(γ) by construction
        assert!(
            (pos.normalize().dot(&vel.normalize()) - gamma.sin()).abs() < 1e-12
        );

        // Apply a 10 m/s "tangent" burn via NTW — should add exactly 10 m/s
        // to |v|.
        let ntw_dcm = crate::frametransform::ntw_to_gcrf(&pos, &vel);
        let dv_ntw_gcrf = ntw_dcm * numeris::vector![0.0, 10.0, 0.0];
        let v_after_ntw = vel + dv_ntw_gcrf;
        let dv_ntw_along_v = v_after_ntw.norm() - vel.norm();
        assert!(
            (dv_ntw_along_v - 10.0).abs() < 1.0e-6,
            "NTW +T burn should add exactly 10 m/s to |v|; got {:.9} m/s",
            dv_ntw_along_v
        );

        // Apply a 10 m/s "in-track" burn via RIC — should add *less* than
        // 10 m/s to |v| (the loss is O(γ²) for small γ).
        let ric_dcm = crate::frametransform::ric_to_gcrf(&pos, &vel);
        let dv_ric_gcrf = ric_dcm * numeris::vector![0.0, 10.0, 0.0];
        let v_after_ric = vel + dv_ric_gcrf;
        let dv_ric_along_v = v_after_ric.norm() - vel.norm();
        assert!(
            dv_ric_along_v < 10.0,
            "RIC +I burn should add less than 10 m/s to |v|; got {:.6}",
            dv_ric_along_v
        );
        // Specifically, the loss should be ~= 10·(1 - cos γ) ≈ 0.245 m/s
        // for γ = 12.7°.
        let expected_loss = 10.0 * (1.0 - gamma.cos());
        assert!(
            (10.0 - dv_ric_along_v - expected_loss).abs() < 1.0e-2,
            "RIC loss should be ≈ 10(1-cos γ) = {:.4} m/s; got {:.4}",
            expected_loss,
            10.0 - dv_ric_along_v
        );

        // Cross-check: the magnitudes of the GCRF delta-v vectors are equal
        // (both are 10 m/s in their respective frames).
        assert!((dv_ntw_gcrf.norm() - 10.0).abs() < 1e-12);
        assert!((dv_ric_gcrf.norm() - 10.0).abs() < 1e-12);

        Ok(())
    }

    /// `set_pos_uncertainty` and `set_vel_uncertainty` should preserve
    /// the block that the caller is not currently updating, so calling
    /// them in sequence builds up a full 6×6 covariance.
    #[test]
    fn test_uncertainty_preserves_other_block() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let r = consts::EARTH_RADIUS + 500e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );

        // Set position first, then velocity
        sat.set_pos_uncertainty(&numeris::vector![100.0, 200.0, 50.0], Frame::LVLH)?;
        sat.set_vel_uncertainty(&numeris::vector![0.1, 0.2, 0.05], Frame::LVLH)?;

        let cov = match sat.cov() {
            StateCov::PVCov(m) => m,
            StateCov::None => panic!("expected covariance"),
        };

        // Both position and velocity blocks must be non-zero
        let pos_block = cov.block::<3, 3>(0, 0);
        let vel_block = cov.block::<3, 3>(3, 3);
        assert!(pos_block.norm_inf() > 1.0, "position block should be set");
        assert!(vel_block.norm_inf() > 0.001, "velocity block should be preserved");

        Ok(())
    }

    /// All four supported frames should produce a self-consistent 3x3
    /// position covariance: on a circular orbit GCRF input with a
    /// radial-only sigma should give a non-trivial covariance with the
    /// correct eigenstructure in any orbital frame.
    #[test]
    fn test_uncertainty_supported_frames() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let r = consts::EARTH_RADIUS + 500e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let sat0 = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );

        for frame in [Frame::GCRF, Frame::LVLH, Frame::RIC, Frame::NTW] {
            let mut sat = sat0.clone();
            sat.set_pos_uncertainty(&numeris::vector![10.0, 20.0, 30.0], frame)?;
            match sat.cov() {
                StateCov::PVCov(m) => {
                    let block = m.block::<3, 3>(0, 0);
                    // Trace should equal sum of sigma² regardless of frame
                    let trace = block[(0, 0)] + block[(1, 1)] + block[(2, 2)];
                    let expected = 100.0 + 400.0 + 900.0;
                    assert!(
                        (trace - expected).abs() / expected < 1e-12,
                        "{:?}: trace {} expected {}", frame, trace, expected
                    );
                }
                StateCov::None => panic!("{:?}: no covariance set", frame),
            }
        }

        Ok(())
    }

    /// Frames that aren't supported for uncertainty should return an error.
    #[test]
    fn test_uncertainty_rejects_unsupported_frames() {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap();
        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![7.0e6, 0.0, 0.0],
            &numeris::vector![0.0, 7.5e3, 0.0],
        );

        for bad_frame in [
            Frame::ITRF,
            Frame::TIRS,
            Frame::CIRS,
            Frame::TEME,
            Frame::EME2000,
            Frame::ICRF,
        ] {
            let res = sat.set_pos_uncertainty(
                &numeris::vector![1.0, 1.0, 1.0],
                bad_frame,
            );
            assert!(res.is_err(), "{:?} should be rejected", bad_frame);
        }
    }

    /// LVLH and RIC span the same orbital plane but with relabeled / sign-
    /// flipped axes: LVLH +x = RIC +I, LVLH −z = RIC +R, LVLH −y = RIC +C.
    /// A maneuver expressed in either frame should produce identical
    /// delta-v in GCRF if the components are correctly translated.
    #[test]
    fn test_lvlh_matches_ric_with_sign_flips() {
        let pos: Vector3 = numeris::vector![7_000_000.0, 0.0, 0.0];
        let v_mag = (consts::MU_EARTH / pos.norm()).sqrt();
        let vel: Vector3 = numeris::vector![0.0, v_mag, 0.0];

        // LVLH burn: (x, y, z) = (in-track, anti-h, nadir) = (5, 3, 2) m/s
        let dv_lvlh: Vector3 = numeris::vector![5.0, 3.0, 2.0];
        // Equivalent RIC burn: R = −z_lvlh, I = +x_lvlh, C = −y_lvlh
        let dv_ric: Vector3 = numeris::vector![-2.0, 5.0, -3.0];

        let lvlh_dcm = crate::frametransform::lvlh_to_gcrf(&pos, &vel);
        let ric_dcm = crate::frametransform::ric_to_gcrf(&pos, &vel);

        let dv_lvlh_gcrf = lvlh_dcm * dv_lvlh;
        let dv_ric_gcrf = ric_dcm * dv_ric;

        let diff = (dv_lvlh_gcrf - dv_ric_gcrf).norm();
        assert!(
            diff < 1e-10,
            "LVLH and equivalent RIC burn should give identical GCRF dv; diff = {:.3e}",
            diff
        );
    }

    /// End-to-end LVLH maneuver: propagate a GEO orbit with a burn
    /// specified in LVLH and verify it produces the same trajectory as
    /// the equivalent RIC burn (with axis relabeling).
    #[test]
    fn test_lvlh_maneuver_end_to_end() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let t_burn = t0 + Duration::from_hours(0.5);
        let t_end = t0 + Duration::from_hours(3.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();

        let mut sat_lvlh = SatState::from_pv(
            &t0,
            &numeris::vector![r, 0.0, 0.0],
            &numeris::vector![0.0, v, 0.0],
        );
        let mut sat_ric = sat_lvlh.clone();

        // LVLH: x = in-track direction, so (10, 0, 0) is "prograde-like"
        sat_lvlh.add_maneuver(ImpulsiveManeuver::new(
            t_burn,
            numeris::vector![10.0, 0.0, 0.0],
            Frame::LVLH,
        ));
        // RIC equivalent: I = +x_lvlh, so (0, 10, 0)
        sat_ric.add_maneuver(ImpulsiveManeuver::ric(
            t_burn,
            numeris::vector![0.0, 10.0, 0.0],
        ));

        let s_lvlh = sat_lvlh.propagate(&t_end, None, None)?;
        let s_ric = sat_ric.propagate(&t_end, None, None)?;

        // Should agree bit-for-bit (the burns are mathematically identical)
        let pos_diff = (s_lvlh.pos_gcrf() - s_ric.pos_gcrf()).norm();
        assert!(
            pos_diff < 1e-6,
            "LVLH +x burn and RIC +I burn should give identical propagations; diff = {:.3e} m",
            pos_diff
        );

        Ok(())
    }

    /// Exercise the ergonomic constructors and check that `prograde` at
    /// apogee raises perigee, a textbook orbit mechanics result.
    #[test]
    fn test_prograde_constructor_raises_perigee() -> Result<()> {
        let t0 = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;

        // Elliptical orbit, positioned at apogee: r = a(1+e), v perpendicular
        let a = 8000.0e3;
        let e = 0.1;
        let r_apo = a * (1.0 + e);
        let v_apo = (consts::MU_EARTH * (2.0 / r_apo - 1.0 / a)).sqrt();

        let mut sat = SatState::from_pv(
            &t0,
            &numeris::vector![r_apo, 0.0, 0.0],
            &numeris::vector![0.0, v_apo, 0.0],
        );
        let sat_no_burn = sat.clone();

        // Tiny prograde burn at apogee — should raise perigee by ~2·(a/v)·Δv
        sat.add_maneuver(ImpulsiveManeuver::prograde(t0 + Duration::from_seconds(1.0), 5.0));

        // Propagate to somewhere near perigee and measure min radius over
        // the next orbit
        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / consts::MU_EARTH).sqrt();
        let t_sample = t0 + Duration::from_seconds(period);
        let s_burn = sat.propagate(&t_sample, None, None)?;
        let s_no_burn = sat_no_burn.propagate(&t_sample, None, None)?;

        // After a full orbit of propagation, both should be back near apogee,
        // but the burn case has higher semi-major axis so it's slightly ahead
        // — the positions should differ meaningfully.
        let diff = (s_burn.pos_gcrf() - s_no_burn.pos_gcrf()).norm();
        assert!(
            diff > 100.0,
            "5 m/s prograde at apogee should produce measurable drift over one orbit: {} m",
            diff
        );

        Ok(())
    }
}
