use super::drag::{drag_and_partials, drag_force};
use super::point_gravity::{point_gravity, point_gravity_and_partials};
use super::settings::PropSettings;

use crate::earthgravity;
use crate::lpephem;
use crate::ode;
use crate::ode::ODEError;
use crate::ode::ODEResult;
use crate::ode::RKAdaptive;
use crate::orbitprop::Precomputed;
use crate::{Duration, Instant};
use lpephem::sun::shadowfunc;

use anyhow::{Context, Result};

use crate::types::*;

use crate::consts;
use crate::orbitprop::SatProperties;
use num_traits::identities::Zero;

use thiserror::Error;

use nalgebra as na;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PropagationResult<const T: usize> {
    pub time_start: Instant,
    pub state_start: Matrix<6, T>,
    pub time_end: Instant,
    pub state_end: Matrix<6, T>,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
    pub num_eval: u32,
    pub odesol: Option<ode::ODESolution<Matrix<6, T>>>,
}

impl<const T: usize> PropagationResult<T> {
    pub fn interp(&self, time: &Instant) -> Result<Matrix<6, T>> {
        interp_propresult(self, time)
    }
}

pub type StateType<const C: usize> = na::SMatrix<f64, 6, C>;

// Simple state with position & velocity
pub type SimpleState = StateType<1>;

// Covariance State in includes
pub type CovState = StateType<7>;

#[derive(Debug, Error)]
pub enum PropagationError {
    #[error("Invalid number of columns: {c}")]
    InvalidStateColumns { c: usize },
    #[error("No Dense Output in Solution")]
    NoDenseOutputInSolution,
    #[error("ODE Error: {0}")]
    ODEError(ode::ODEError),
}

//
// This actually implements the force model that is used to
// integrate the ODE to get position and velocity
//
// State is position and velocity
// Force is computed and integrated to get velocity
// Velocity is integrated to get position
//
// If C=7, a 6x6 state transition matrix is appended as additional
// colums, making the integrated "state" a 6x7 matrix
// The state transition matrix can be used to propagate covariances
//
// See Montenbruk & Gill for details (Chapter 7)
//

///
/// High-precision Propagation a satellite state from a given start time
/// to a given stop time, with input settings and
/// satellite properties
///
/// Uses Runga-kutta methods for integrating the force equations
///
/// The default propagator uses a Runga-Kutta 9(8) integrator
/// with coefficients computed by Verner:
/// <https://www.sfu.a/~jverner//>
///
/// This works much better than lower-order Runga-Kutta solvers such as
/// Dorumund-Prince, and I don't know why it isn't more popular in
/// numerical packages
///
/// # Forces included in the propagator:
///
/// * Earth gravity with higher-order zonal terms
/// * Gravitational pull of sun, moon
/// * Solar radiation pressure
/// * Atmospheric drag: NRL-MSISE 2000 model, with option to include space weather
///   (effects can be large)
///
/// # Arguments:
///
/// * `state` - The satellite state, represented as:
///    * `SimpleState` - a 6x1 matrix where the 1st three elements represent GCRF position in meters,
///       and the 2nd three elements represent GCRF velocity in meters / second
///    * `CovState` - a 6x7 matrix where the first column is the same as SimpleState above, and columns
///       2-7 represent the 6x6 state transition matrix, dS/dS0
///       The state transition matrix should be initialized to identity when running
///       The output of the state transition matrix can be used to compute the evolution of the
///       state covariance  (see Montenbruck and Gill for details)
///  * `start` - The time at the initial state
///  * `stop` - The time at which to propagate for computing new states
///  * `step_seconds` - An optional value representing intervals between `start` and `stop` at which
///     the new state will be computed
///  * `settings` - Settings for the Runga-Kutta propagator
///  * `satprops` - Properties of the satellite, such as ballistic coefficient & susceptibility to
///     radiation pressure
///
/// # Returns
/// * `PropagationResult` object with details of the propagation compute, the final state, and intermediate states if step size
///    is set
///
/// # Example:
///
/// ```
/// // Setup a simple Geosynchronous orbit with initial position along the x axis
/// // and initial velocity along the y axis
/// let mut state = satkit::orbitprop::SimpleState::zeros();
/// state[0] = satkit::consts::GEO_R;
/// state[4] = (satkit::consts::MU_EARTH / satkit::consts::GEO_R).sqrt();
///
/// // Setup the details of the propagation
/// let mut settings = satkit::orbitprop::PropSettings::default();
/// settings.abs_error = 1.0e-9;
/// settings.rel_error = 1.0e-14;
/// settings.gravity_order = 4;
///
/// // Pick an arbitrary start time
/// let starttime = satkit::Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
/// // Propagate to 1/2 day ahead
/// let stoptime = starttime + satkit::Duration::from_days(0.5);
///
/// // Look at the results
/// let res = satkit::orbitprop::propagate(&state, &starttime, &stoptime, &settings, None).unwrap();
///
/// println!("results = {:?}", res);
/// // Expect:
/// // res = PropagationResult { time: [Instant { mjd_tai: 57101.50040509259 }],
/// //                           state: [[[-42153870.84175911, -379423.6616440884, -26.239180898423687,
/// //                                     27.66411233952899, -3075.146656613106, 0.0020580348953689828]]],
/// //                           accepted_steps: 45, rejected_steps: 0, num_eval: 722
/// //                          }
/// ```
///
/// # Example 2:
///
/// ```
/// // Now, propagate the state transition matrix
///
/// // Setup a simple Geosynchronous orbit with initial position along the x axis
/// // and initial velocity along the y axis
/// use nalgebra as na;
/// let mut state = satkit::orbitprop::CovState::zeros();
/// state.fixed_view_mut::<3, 1>(0, 0).copy_from(&na::vector![satkit::consts::GEO_R, 0.0, 0.0]);
/// state.fixed_view_mut::<3, 1>(3, 0).copy_from(&na::vector![0.0, (satkit::consts::MU_EARTH/satkit::consts::GEO_R).sqrt(), 0.0]);
/// // initialize state transition matrix to zero
/// state.fixed_view_mut::<6, 6>(0, 1).copy_from(&na::Matrix6::<f64>::identity());
///
///
/// // Setup the details of the propagation
/// let mut settings = satkit::orbitprop::PropSettings::default();
/// settings.abs_error = 1.0e-9;
/// settings.rel_error = 1.0e-14;
/// settings.gravity_order = 4;
///
/// // Pick an arbitrary start time
/// let starttime = satkit::Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
/// // Propagate to 1/2 day ahead
/// let stoptime = starttime + satkit::Duration::from_days(0.5);
///
/// // Look at the results
/// let res = satkit::orbitprop::propagate(&state, &starttime, &stoptime, &settings, None).unwrap();
///
/// println!("results = {:?}", res);
/// ```
///
///
pub fn propagate<const C: usize>(
    state: &StateType<C>,
    start: &Instant,
    stop: &Instant,
    settings: &PropSettings,
    satprops: Option<&dyn SatProperties>,
) -> Result<PropagationResult<C>> {
    // Duration to end of integration, in seconds
    let x_end: f64 = (*stop - *start).as_seconds();

    let odesettings = crate::ode::RKAdaptiveSettings {
        abserror: settings.abs_error,
        relerror: settings.rel_error,
        dense_output: settings.enable_interp,
        ..Default::default()
    };

    // Get or create data for interpolation
    let interp: &Precomputed = {
        if let Some(sinterp) = &settings.precomputed {
            if stop > start {
                if (*start >= sinterp.start) && (*stop <= sinterp.stop) {
                    sinterp
                } else {
                    &Precomputed::new(start, stop)
                        .context("Cannot compute precomputed interpolation data for propagation")?
                }
            } else if (*stop >= sinterp.start) && (*start <= sinterp.stop) {
                sinterp
            } else {
                &Precomputed::new(start, stop)
                    .context("Cannot compute precomputed interpolation data for propagation")?
            }
        } else {
            &Precomputed::new(start, stop)
                .context("Cannot compute precomputed interpolation dat for propagation")?
        }
    };

    let ydot = |x: f64, y: &Matrix<6, C>| -> ODEResult<Matrix<6, C>> {
        // The time variable in the ODE is in seconds
        let time: Instant = *start + Duration::from_seconds(x);

        // get GCRS position & velocity;
        let pos_gcrf: na::Vector3<f64> = y.fixed_view::<3, 1>(0, 0).into();
        let vel_gcrf: na::Vector3<f64> = y.fixed_view::<3, 1>(3, 0).into();

        // Get interpolated values
        let (qgcrf2itrf, sun_gcrf, moon_gcrf) = match interp.interp(&time) {
            Ok(v) => v,
            Err(e) => return Err(ODEError::YDotError(e.to_string())),
        };
        let qitrf2gcrf = qgcrf2itrf.conjugate();

        // Position in ITRF coordinates
        let pos_itrf = qgcrf2itrf * pos_gcrf;

        const fn is_one<const C2: usize>() -> bool {
            C2 == 1
        }
        const fn is_seven<const C2: usize>() -> bool {
            C2 == 7
        }

        // Propagating a "simple" 6-dof (position, velocity) state
        if is_one::<C>() {
            let mut accel = Vector3::zeros();

            // Gravity in the ITRF frame
            let gravity_itrf =
                earthgravity::jgm3().accel(&pos_itrf, settings.gravity_order as usize);

            // Gravity in the GCRS frame
            accel += qitrf2gcrf * gravity_itrf;

            // Acceleration due to moon
            accel += point_gravity(&pos_gcrf, &moon_gcrf, crate::consts::MU_MOON);

            // Acceleration due to sun
            accel += point_gravity(&pos_gcrf, &sun_gcrf, crate::consts::MU_SUN);

            // Add solar pressure & drag if that is defined in satellite properties
            if let Some(props) = satprops {
                let ss = y.fixed_view::<6, 1>(0, 0);

                // Compute solar pressure
                let solarpressure = -shadowfunc(&sun_gcrf, &pos_gcrf)
                    * props.cr_a_over_m(&time, &ss.into())
                    * 4.56e-6
                    * sun_gcrf
                    / sun_gcrf.norm();
                accel += solarpressure;

                // Compute drag
                if pos_gcrf.norm() < 700.0e3 + crate::consts::EARTH_RADIUS {
                    let cd_a_over_m = props.cd_a_over_m(&time, &ss.into());

                    if cd_a_over_m > 1e-6 {
                        accel += drag_force(
                            &pos_gcrf,
                            &pos_itrf,
                            &vel_gcrf,
                            &time,
                            cd_a_over_m,
                            settings.use_spaceweather,
                        );
                    }
                }
            } // end of handling drag & solarpressure

            let mut dy = Matrix::<6, C>::zeros();
            // change in position is velocity
            dy.fixed_view_mut::<3, 1>(0, 0).copy_from(&vel_gcrf);

            // Change in velocity is acceleration
            dy.fixed_view_mut::<3, 1>(3, 0).copy_from(&accel);

            Ok(dy)
        }
        // If C==7, we are also integrating the state transition matrix
        else if is_seven::<C>() {
            // For state transition matrix, we need to compute force partials with respect to position
            // (for all forces but drag, partial with respect to velocity are zero)
            let (gravity_accel, gravity_partials) =
                earthgravity::jgm3().accel_and_partials(&pos_itrf, settings.gravity_order as usize);
            let (sun_accel, sun_partials) =
                point_gravity_and_partials(&pos_gcrf, &sun_gcrf, consts::MU_SUN);
            let (moon_accel, moon_partials) =
                point_gravity_and_partials(&pos_gcrf, &moon_gcrf, consts::MU_MOON);

            let mut accel = qitrf2gcrf * gravity_accel + sun_accel + moon_accel;

            // Equation 7.42 in Montenbruck & Gill
            let mut dfdy: StateType<6> = StateType::<6>::zeros();
            dfdy.fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&na::Matrix3::<f64>::identity());

            let ritrf2gcrf = qitrf2gcrf.to_rotation_matrix();
            // Sum partials with respect to position for gravity, sun, and moon
            // Note: gravity partials need to be rotated into the gcrf frame from itrf
            let mut dadr = ritrf2gcrf * gravity_partials * ritrf2gcrf.transpose()
                + sun_partials
                + moon_partials;

            // Handle satellite properties for drag and radiation pressure
            if let Some(props) = satprops {
                // Satellite state as 6-element position, velcoity matrix
                // used to query cd_a_over_m
                let ss = y.fixed_view::<6, 1>(0, 0);

                // Compute solar pressure
                // Partials for this are very small since the sun is very very far away, changes in
                // satellite position don't change radiaion pressure much, so we will ignore...
                let solarpressure = -shadowfunc(&sun_gcrf, &pos_gcrf)
                    * props.cr_a_over_m(&time, &ss.into())
                    * 4.56e-6
                    * sun_gcrf
                    / sun_gcrf.norm();
                accel += solarpressure;

                // We know drag is negligible above 700 km, so ignore if this is the case
                if pos_gcrf.norm() < 700.0e3 + crate::consts::EARTH_RADIUS {
                    let cd_a_over_m = props.cd_a_over_m(&time, &ss.into());
                    if cd_a_over_m > 1e-6 {
                        let (drag_accel, ddragaccel_dr, ddragaccel_dv) = drag_and_partials(
                            &pos_gcrf,
                            &qgcrf2itrf,
                            &vel_gcrf,
                            &time,
                            cd_a_over_m,
                            settings.use_spaceweather,
                        );

                        // Add acceleration from drag to accel vector
                        accel += drag_accel;

                        // Add drag partials with respect to position to
                        // daccel dr
                        dadr += ddragaccel_dr;

                        // Drag is the only force term that produces a finite partial with respect
                        // to velocity, so copy it directly into dfdy here.
                        dfdy.fixed_view_mut::<3, 3>(3, 3).copy_from(&ddragaccel_dv);
                    }
                }
            }
            dfdy.fixed_view_mut::<3, 3>(3, 0).copy_from(&dadr);

            // Derivative of state transition matrix is dfdy * state transition matrix
            let dphi: na::Matrix<f64, na::Const<6>, na::Const<6>, na::ArrayStorage<f64, 6, 6>> =
                dfdy * y.fixed_view::<6, 6>(0, 1);

            let mut dy = Matrix::<6, C>::zero();
            dy.fixed_view_mut::<3, 1>(0, 0).copy_from(&vel_gcrf);
            dy.fixed_view_mut::<3, 1>(3, 0).copy_from(&accel);
            dy.fixed_view_mut::<6, 6>(0, 1).copy_from(&dphi);
            Ok(dy)
        } else {
            ODEError::YDotError(PropagationError::InvalidStateColumns { c: C }.to_string()).into()
        }
    };

    match settings.enable_interp {
        false => {
            let res = match crate::ode::solvers::RKV98NoInterp::integrate(
                0.0,
                x_end,
                state,
                ydot,
                &odesettings,
            ) {
                Ok(res) => res,
                Err(e) => return Err(PropagationError::ODEError(e).into()),
            };

            Ok(PropagationResult {
                time_start: *start,
                state_start: *state,
                time_end: *stop,
                state_end: res.y,
                accepted_steps: res.naccept as u32,
                rejected_steps: res.nreject as u32,
                num_eval: res.nevals as u32,
                odesol: Some(res),
            })
        }
        true => {
            let res = crate::ode::solvers::RKV98::integrate(0.0, x_end, state, ydot, &odesettings)?;
            Ok(PropagationResult {
                time_start: *start,
                state_start: *state,
                time_end: *stop,
                state_end: res.y,
                accepted_steps: res.naccept as u32,
                rejected_steps: res.nreject as u32,
                num_eval: res.nevals as u32,
                odesol: Some(res),
            })
        }
    }
}

pub fn interp_propresult<const C: usize>(
    res: &PropagationResult<C>,
    time: &Instant,
) -> Result<StateType<C>> {
    if let Some(sol) = &res.odesol {
        if sol.dense.is_some() {
            let x = (time - res.time_start).as_seconds();
            let y = crate::ode::solvers::RKV98::interpolate(x, sol)?;
            Ok(y)
        } else {
            Err(PropagationError::NoDenseOutputInSolution.into())
        }
    } else {
        Err(PropagationError::NoDenseOutputInSolution.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consts, orbitprop::SatPropertiesStatic};
    use std::f64::consts::PI;

    use std::fs::File;

    use crate::Duration;
    use std::io::{self, BufRead};

    #[test]
    fn test_short_propagate() -> Result<()> {
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
        let stoptime = starttime + Duration::from_seconds(0.1);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_order: 4,
            ..Default::default()
        };

        let _res1 = propagate(&state, &starttime, &stoptime, &settings, None)?;

        Ok(())
    }

    #[test]
    fn test_propagate() -> Result<()> {
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
        let stoptime = starttime + Duration::from_days(0.25);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let mut settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_order: 4,
            ..Default::default()
        };
        settings.precompute_terms(&starttime, &stoptime)?;

        let res1 = propagate(&state, &starttime, &stoptime, &settings, None)?;
        // Try to propagate back to original time
        let res2 = propagate(&res1.state_end, &stoptime, &starttime, &settings, None)?;
        // See if propagating back to original time matches
        for ix in 0..6_usize {
            assert!((res2.state_end[ix] - state[ix]).abs() < 1.0)
        }

        Ok(())
    }

    #[test]
    fn test_interp() -> Result<()> {
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
        let stoptime = starttime + Duration::from_days(1.0);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_order: 4,
            ..Default::default()
        };

        // Propagate forward
        let res = propagate(&state, &starttime, &stoptime, &settings, None)?;
        // propagate backward to original time
        let res2 = propagate(&res.state_end, &stoptime, &starttime, &settings, None)?;
        // Check that we recover the original state
        for ix in 0..6_usize {
            assert!((state[ix] - res2.state_end[ix]).abs() < 1.0e-3);
        }

        // Check interpolation forward and backward return the same result
        let newtime = starttime + Duration::from_days(0.45);
        let interp = res.interp(&newtime)?;
        let interp2 = res2.interp(&newtime)?;
        for ix in 0..6_usize {
            assert!((interp[ix] - interp2[ix]).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn test_state_transition() -> Result<()> {
        // Check the state transition matrix:
        // Explicitly propagate two slightly different states,
        // separated by "dstate",
        // and compare with difference in final states as predicted
        // by state transition matrix
        // Note also: drag partials are very small relative to other terms,
        // making it difficult to confirm that calculations are correct.

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
        let stoptime = starttime + Duration::from_days(0.5);

        let mut state: CovState = CovState::zeros();

        let theta = PI / 6.0;
        state[0] = consts::GEO_R * theta.cos();
        state[2] = consts::GEO_R * theta.sin();
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.cos();
        state[5] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.sin();
        state
            .fixed_view_mut::<6, 6>(0, 1)
            .copy_from(&na::Matrix6::<f64>::identity());

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_order: 4,
            ..Default::default()
        };

        // Made-up small variations in the state
        let dstate = na::vector![6.0, -10.0, 120.5, 0.1, 0.2, -0.3];

        // Propagate state (and state-transition matrix)
        let res = propagate(&state, &starttime, &stoptime, &settings, None)?;

        // Explicitly propagate state + dstate
        let res2 = propagate(
            &(state.fixed_view::<6, 1>(0, 0) + dstate),
            &starttime,
            &stoptime,
            &settings,
            None,
        )?;

        // Difference in states from explicitly propagating with
        // "dstate" change in initial conditions
        let dstate_prop = res2.state_end - res.state_end.fixed_view::<6, 1>(0, 0);

        // Difference in states estimated from state transition matrix
        let dstate_phi = res.state_end.fixed_view::<6, 6>(0, 1) * dstate;
        for ix in 0..6_usize {
            assert!((dstate_prop[ix] - dstate_phi[ix]).abs() / dstate_prop[ix] < 1e-3);
        }

        Ok(())
    }

    #[test]
    fn test_state_transition_drag() -> Result<()> {
        // Check the state transition matrix:
        // Explicitly propagate two slightly different states,
        // separated by "dstate",
        // and compare with difference in final states as predicted
        // by state transition matrix
        // This version has a low-altitude satellite and we will
        // set a fininte cdaoverm value so that there is drag
        // and we can check drag partials

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0);
        let stoptime = starttime + crate::Duration::from_days(0.2);

        let mut state: CovState = CovState::zeros();

        let pgcrf = na::vector![3059573.85713792, 5855177.98848048, -7191.45042671];
        let vgcrf = na::vector![916.08123489, -468.22498656, 7700.48460839];

        // 30-deg inclination
        state.fixed_view_mut::<3, 1>(0, 0).copy_from(&pgcrf);
        state.fixed_view_mut::<3, 1>(3, 0).copy_from(&vgcrf);
        state
            .fixed_view_mut::<6, 6>(0, 1)
            .copy_from(&na::Matrix6::<f64>::identity());

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_order: 4,
            ..Default::default()
        };

        let satprops: SatPropertiesStatic = SatPropertiesStatic::new(2.0 * 0.3 * 0.1 / 5.0, 0.0);

        // Made-up small variations in the state
        let dstate = na::vector![2.0, -4.0, 20.5, 0.05, 0.02, -0.01];

        // Propagate state (and state-transition matrix)

        let res = propagate(&state, &starttime, &stoptime, &settings, Some(&satprops))?;

        // Explicitly propagate state + dstate
        let res2 = propagate(
            &(state.fixed_view::<6, 1>(0, 0) + dstate),
            &starttime,
            &stoptime,
            &settings,
            Some(&satprops),
        )?;

        // Difference in states from explicitly propagating with
        // "dstate" change in initial conditions
        let dstate_prop = res2.state_end - res.state_end.fixed_view::<6, 1>(0, 0);

        let dstate_phi = res.state_end.fixed_view::<6, 6>(0, 1) * dstate;

        // Are differences within 1%?
        for ix in 0..6_usize {
            assert!((dstate_prop[ix] - dstate_phi[ix]).abs() / dstate_prop[ix] < 0.1);
        }

        Ok(())
    }

    #[test]
    fn test_gps() -> Result<()> {
        let testvecfile = crate::utils::test::get_testvec_dir()
            .unwrap()
            .join("orbitprop")
            .join("ESA0OPSFIN_20233640000_01D_05M_ORB.SP3");

        if !testvecfile.is_file() {
            anyhow::bail!(
                "Required GPS SP3 File: \"{}\" does not exist
                clone test vectors from:
                <https://storage.googleapis.com/satkit-testvecs/>
                using python script in satkit repo: `python/test/download-testvecs.py`
                or set \"SATKIT_TESTVEC_ROOT\" to point to directory",
                testvecfile.to_string_lossy()
            );
        }
        let file: File = File::open(testvecfile.clone())?;

        let times: Vec<crate::Instant> = io::BufReader::new(file)
            .lines()
            .filter(|x: &Result<String, io::Error>| x.as_ref().unwrap().starts_with('*'))
            .map(|rline| -> Result<crate::Instant> {
                let line = rline.unwrap();
                let lvals: Vec<&str> = line.split_whitespace().collect();
                let year: i32 = lvals[1].parse()?;
                let mon: i32 = lvals[2].parse()?;
                let day: i32 = lvals[3].parse()?;
                let hour: i32 = lvals[4].parse()?;
                let min: i32 = lvals[5].parse()?;
                let sec: f64 = lvals[6].parse()?;
                Ok(Instant::from_datetime(year, mon, day, hour, min, sec))
            }).collect::<Result<Vec<crate::Instant>, _>>()?;
        

        let file: File = File::open(testvecfile)?;

        let satnum: usize = 20;
        let satstr = format!("PG{}", satnum);
        let pitrf: Vec<na::Vector3<f64>> = io::BufReader::new(file)
            .lines()
            .filter(|x| {
                let rline = &x.as_ref().unwrap()[0..4];
                rline == satstr
            })
            .map(|rline| -> Result<na::Vector3<f64>> {
                let line = rline.unwrap();
                let lvals: Vec<&str> = line.split_whitespace().collect();
                let px: f64 = lvals[1].parse()?;
                let py: f64 = lvals[2].parse()?;
                let pz: f64 = lvals[3].parse()?;
                Ok(na::vector![px, py, pz] * 1.0e3)
            })
            .collect::<Result<Vec<na::Vector3<f64>>>>()?;

        assert!(times.len() == pitrf.len());
        let pgcrf: Vec<na::Vector3<f64>> = pitrf
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let q = crate::frametransform::qitrf2gcrf(&times[idx]);
                q * p
            })
            .collect();

        let v0 = na::vector![
            2.47130562e+03,
            2.94682753e+03,
            -5.34172176e+02,
            2.32565692e-02
        ];

        let state0 = na::vector![pgcrf[0][0], pgcrf[0][1], pgcrf[0][2], v0[0], v0[1], v0[2]];
        let satprops: SatPropertiesStatic = SatPropertiesStatic::new(0.0, v0[3]);

        let settings = PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let res = propagate(
            &state0,
            &times[0],
            &times[times.len() - 1],
            &settings,
            Some(&satprops),
        )?;

        // We've propagated over a day; assert that the difference in position on all three coordinate axes
        // is less than 10 meters for all 5-minute intervals
        for iv in 0..(pgcrf.len()) {
            let interp_state = res.interp(&times[iv])?;
            for ix in 0..3 {
                assert!((pgcrf[iv][ix] - interp_state[ix]).abs() < 8.0);
            }
        }

        Ok(())
    }
}
