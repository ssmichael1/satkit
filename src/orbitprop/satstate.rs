use nalgebra as na;

use crate::orbitprop;
use crate::orbitprop::PropSettings;
use crate::AstroTime;
use crate::SKResult;

type PVCovType = na::SMatrix<f64, 6, 6>;

#[derive(Clone, Debug)]
pub enum StateCov {
    None,
    PVCov(PVCovType),
}

#[derive(Clone, Debug)]
pub struct SatState {
    pub time: AstroTime,
    pub pv: na::Vector6<f64>,
    pub cov: StateCov,
}

impl SatState {
    pub fn from_pv(time: &AstroTime, pos: &na::Vector3<f64>, vel: &na::Vector3<f64>) -> SatState {
        SatState {
            time: time.clone(),
            pv: na::vector![pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]],
            cov: StateCov::None,
        }
    }

    pub fn pos(&self) -> na::Vector3<f64> {
        self.pv.fixed_view::<3, 1>(0, 0).into()
    }

    pub fn vel(&self) -> na::Vector3<f64> {
        self.pv.fixed_view::<3, 1>(3, 0).into()
    }

    /// set covariance
    pub fn set_cov(&mut self, cov: StateCov) {
        self.cov = cov;
    }

    /// Quaternion to go from gcrf to position, velocity, angular momentum frame
    pub fn qgcrf2pvh(&self) -> na::UnitQuaternion<f64> {
        type Quat = na::UnitQuaternion<f64>;

        // position and velocity might not be orthogonal ; orbits are not perfect
        // 2-body problems, so remove any position component of velocity
        let p = self.pos();
        let mut v = self.vel();
        v = v - p * (p.dot(&v)) / p.norm_squared();
        let q1 = Quat::rotation_between(&na::Vector3::x_axis(), &p).unwrap();
        let q2 = Quat::rotation_between(&(q1 * na::Vector3::y_axis()), &v).unwrap();
        q2 * q1
    }

    /// Set position uncertainty (1-sigma) in the
    /// position, velocity, angular momentum frame
    pub fn set_pvh_pos_uncertainty(&mut self, sigma_pvh: &na::Vector3<f64>) {
        self.cov = StateCov::PVCov({
            // Compute rotation from pvh to cartesian frame
            let q1 = na::UnitQuaternion::<f64>::rotation_between(
                &self.pv.fixed_view::<3, 1>(0, 0),
                &na::Vector3::x_axis(),
            )
            .unwrap();
            let q2 = na::UnitQuaternion::<f64>::rotation_between(
                &self.pv.fixed_view::<3, 1>(3, 0),
                &na::Vector3::y_axis(),
            )
            .unwrap();
            let q = q2 * q1;
            let rot = q.to_rotation_matrix();

            // 3x3 covariance in pvh frame
            let mut pcov = na::Matrix3::<f64>::zeros();
            pcov.set_diagonal(&sigma_pvh.map(|x| x * x));

            let mut m = na::Matrix6::<f64>::zeros();
            m.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(rot * pcov * rot.transpose()));

            m
        })
    }

    // Set 1-sigma position undertainty in the Cartesian frame
    pub fn set_cartesian_pos_uncertainty(&mut self, sigma_cart: &na::Vector3<f64>) {
        self.cov = StateCov::PVCov({
            let mut m = PVCovType::zeros();
            let mut diag = na::Vector6::<f64>::zeros();
            diag[0] = sigma_cart[0] * sigma_cart[0];
            diag[1] = sigma_cart[1] * sigma_cart[1];
            diag[2] = sigma_cart[2] * sigma_cart[2];
            m.set_diagonal(&diag);
            m
        })
    }

    ///
    /// Propagate state to a new time
    pub fn propagate(
        &self,
        time: &AstroTime,
        option_settings: Option<&PropSettings>,
    ) -> SKResult<SatState> {
        let default = orbitprop::PropSettings::default();
        let settings = option_settings.unwrap_or(&default);
        match self.cov {
            // Simple case: do not compute state transition matrix, since covariance is not set
            StateCov::None => {
                let res = orbitprop::propagate(&self.pv, &self.time, time, None, settings, None)?;
                Ok(SatState {
                    time: time.clone(),
                    pv: res.state[0],
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
                let res = orbitprop::propagate(&state, &self.time, time, None, settings, None)?;

                Ok(SatState {
                    time: time.clone(),
                    pv: res.state[0].fixed_view::<6, 1>(0, 0).into(),
                    cov: {
                        // Extract state transition matrix from the propagated state
                        let phi = res.state[0].fixed_view::<6, 6>(0, 1);
                        // Evolve the covariance
                        StateCov::PVCov(phi * cov * phi.transpose())
                    },
                })
            }
        }
    }

    pub fn to_string(&self) -> String {
        let mut s1 = format!(
            r#"Satellite State
                Time: {}
            Position: [{:+8.0}, {:+8.0}, {:+8.0}] m,
            Velocity: [{:+8.3}, {:+8.3}, {:+8.3}] m/s"#,
            self.time, self.pv[0], self.pv[1], self.pv[2], self.pv[3], self.pv[4], self.pv[5],
        );
        match self.cov {
            StateCov::None => s1,
            StateCov::PVCov(cov) => {
                s1.push_str(
                    format!(
                        r#"
            Covariance: {cov:+8.2e}"#
                    )
                    .as_str(),
                );
                s1
            }
        }
    }
}

impl std::fmt::Display for SatState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts;

    #[test]
    fn test_qgcrf2pvh() -> SKResult<()> {
        let satstate = SatState::from_pv(
            &AstroTime::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );

        let state2 = satstate.propagate(&(satstate.time + crate::Duration::Hours(3.56)), None)?;

        let rx = state2.qgcrf2pvh().conjugate() * state2.pos();
        let ry = state2.qgcrf2pvh().conjugate() * state2.vel();
        let rz = state2.qgcrf2pvh().conjugate() * (state2.pos().cross(&state2.vel()));

        assert!((rx.dot(&na::Vector3::<f64>::x_axis()) / rx.norm() - 1.0).abs() < 1.0e-6);
        assert!((ry.dot(&na::Vector3::<f64>::y_axis()) / ry.norm() - 1.0).abs() < 1.0e-6);
        assert!((rz.dot(&na::Vector3::<f64>::z_axis()) / rz.norm() - 1.0).abs() < 1.0e-6);

        Ok(())
    }

    #[test]
    fn test_satstate() -> SKResult<()> {
        let satstate = SatState::from_pv(
            &AstroTime::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        println!("state orig = {:?}", satstate);

        let state2 = satstate.propagate(&(satstate.time + 1.0), None)?;

        println!("state 2 = {:?}", state2);

        let state0 = state2.propagate(&satstate.time, None);
        println!("state 0 = {:?}", state0);
        Ok(())
    }

    #[test]
    fn test_satcov() -> SKResult<()> {
        let mut satstate = SatState::from_pv(
            &AstroTime::from_datetime(2015, 3, 20, 0, 0, 0.0),
            &na::vector![consts::GEO_R, 0.0, 0.0],
            &na::vector![0.0, (consts::MU_EARTH / consts::GEO_R).sqrt(), 0.0],
        );
        satstate.set_pvh_pos_uncertainty(&na::vector![1.0, 1.0, 1.0]);
        println!("state orig = {:?}", satstate.cov);

        let state2 = satstate.propagate(&(satstate.time + 1.0), None)?;

        println!("state 2 = {:?}", state2.cov);

        Ok(())
    }
}
