use super::drag::{drag_and_partials, drag_force};
use super::point_gravity::{point_gravity, point_gravity_and_partials};
use super::settings::PropSettings;
use super::tides::{self, TideModel};

use crate::lpephem;
use crate::orbitprop::Precomputed;
use crate::{Duration, Instant, TimeLike};
use lpephem::sun::shadowfunc;

use numeris::ode::{self, RKAdaptive, Rosenbrock};

use super::error::{Error, Result};

use crate::mathtypes::*;

use crate::consts;
use crate::orbitprop::SatProperties;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct PropagationResult<const T: usize> {
    pub time_begin: Instant,
    pub state_begin: Matrix<6, T>,
    pub time_end: Instant,
    pub state_end: Matrix<6, T>,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
    pub num_eval: u32,
    pub odesol: Option<ode::Solution<f64, 6, T>>,
    /// Dense output from Gauss-Jackson 8 (populated only when the propagation
    /// used `Integrator::GaussJackson8` with `settings.enable_interp = true`).
    /// Stores per-step (t, r, v, a) samples for quintic Hermite interpolation
    /// via [`interp_propresult`]. The RK-based integrators use `odesol`
    /// instead; exactly one of the two is populated for a given propagation.
    pub gj_dense: Option<crate::orbitprop::ode::GJDenseOutput<f64, 3>>,
    pub integrator: super::settings::Integrator,
}

impl<const T: usize> std::fmt::Debug for PropagationResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropagationResult")
            .field("time_begin", &self.time_begin)
            .field("state_begin", &self.state_begin)
            .field("time_end", &self.time_end)
            .field("state_end", &self.state_end)
            .field("accepted_steps", &self.accepted_steps)
            .field("rejected_steps", &self.rejected_steps)
            .field("num_eval", &self.num_eval)
            .field("odesol", &self.odesol.as_ref().map(|_| "..."))
            .field("integrator", &self.integrator)
            .finish()
    }
}

impl<const T: usize> PropagationResult<T> {
    pub fn interp<U: TimeLike>(&self, time: &U) -> Result<Matrix<6, T>> {
        interp_propresult(self, time)
    }

    pub fn interp_batch(&self, times: &[Instant]) -> Result<Vec<StateType<T>>> {
        interp_propresult_batch(self, times)
    }
}

pub type StateType<const C: usize> = Matrix<6, C>;

// Simple state with position & velocity
pub type SimpleState = StateType<1>;

// Covariance State in includes
pub type CovState = StateType<7>;

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

/// Solar radiation pressure acceleration in GCRF
fn solar_pressure_accel(
    sun_gcrf: &Vector3,
    pos_gcrf: &Vector3,
    time: &Instant,
    props: &dyn SatProperties,
    state: &SimpleState,
) -> Vector3 {
    sun_gcrf
        * (-shadowfunc(sun_gcrf, pos_gcrf) * props.cr_a_over_m(time, state) * 4.56e-6
            / sun_gcrf.norm())
}

///
/// High-precision propagation of a satellite state from a given begin time
/// to a given end time, with input settings and
/// satellite properties
///
/// Uses Runge-Kutta methods for integrating the force equations
///
/// The default propagator uses a Runge-Kutta 9(8) integrator
/// with coefficients computed by Verner:
/// <https://www.sfu.a/~jverner//>
///
/// This works much better than lower-order Runge-Kutta solvers such as
/// Dormand-Prince, and I don't know why it isn't more popular in
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
///      and the 2nd three elements represent GCRF velocity in meters / second
///    * `CovState` - a 6x7 matrix where the first column is the same as SimpleState above, and columns
///      2-7 represent the 6x6 state transition matrix, dS/dS0
///      The state transition matrix should be initialized to identity when running
///      The output of the state transition matrix can be used to compute the evolution of the
///      state covariance  (see Montenbruck and Gill for details)
///  * `begin` - The time at the initial state
///  * `end` - Propagate to this time from the begin
///  * `step_seconds` - An optional value representing intervals between `begin` and `end` at which
///    the new state will be computed
///  * `settings` - Settings for the Runge-Kutta propagator
///  * `satprops` - Properties of the satellite, such as ballistic coefficient & susceptibility to
///    radiation pressure
///
/// # Returns
/// * `PropagationResult` object with details of the propagation compute, the final state, and intermediate states if step size
///   is set
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
/// settings.gravity_degree = 4;
///
/// // Pick an arbitrary begin time
/// let begintime = satkit::Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap();
/// // Propagate to 1/2 day ahead
/// let endtime = begintime + satkit::Duration::from_days(0.5);
///
/// // Look at the results
/// let res = satkit::orbitprop::propagate(&state, &begintime, &endtime, &settings, None).unwrap();
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
/// use satkit::mathtypes::*;
/// let mut state = satkit::orbitprop::CovState::zeros();
/// state.set_block(0, 0, &numeris::vector![satkit::consts::GEO_R, 0.0, 0.0]);
/// state.set_block(3, 0, &numeris::vector![0.0, (satkit::consts::MU_EARTH/satkit::consts::GEO_R).sqrt(), 0.0]);
/// // initialize state transition matrix to identity
/// state.set_block(0, 1, &Matrix6::eye());
///
///
/// // Setup the details of the propagation
/// let mut settings = satkit::orbitprop::PropSettings::default();
/// settings.abs_error = 1.0e-9;
/// settings.rel_error = 1.0e-14;
/// settings.gravity_degree = 4;
///
/// // Pick an arbitrary begin time
/// let begintime = satkit::Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap();
/// // Propagate to 1/2 day ahead
/// let endtime = begintime + satkit::Duration::from_days(0.5);
///
/// // Look at the results
/// let res = satkit::orbitprop::propagate(&state, &begintime, &endtime, &settings, None).unwrap();
///
/// println!("results = {:?}", res);
/// ```
///
///
pub fn propagate<const C: usize, T: TimeLike>(
    state: &StateType<C>,
    begin: &T,
    end: &T,
    settings: &PropSettings,
    satprops: Option<&dyn SatProperties>,
) -> Result<PropagationResult<C>> {
    let begin = begin.as_instant();
    let end = end.as_instant();

    // Check for zero-duration case and return immediately
    if end == begin {
        return Ok(PropagationResult {
            time_begin: begin,
            state_begin: *state,
            time_end: end,
            state_end: *state,
            accepted_steps: 0,
            rejected_steps: 0,
            num_eval: 0,
            odesol: None,
            gj_dense: None,
            integrator: settings.integrator,
        });
    }

    // RODAS4 does not support state transition matrix (C==7)
    if C == 7 && settings.integrator == crate::orbitprop::Integrator::RODAS4 {
        return Err(Error::RODAS4NoSTM);
    }
    // Gauss-Jackson 8 is 2nd-order only and does not propagate STM
    if C == 7 && settings.integrator == crate::orbitprop::Integrator::GaussJackson8 {
        return Err(Error::GaussJackson8NoSTM);
    }

    // Duration to end of integration, in seconds
    let x_end: f64 = (end - begin).as_seconds();

    let odesettings = ode::AdaptiveSettings {
        abs_tol: settings.abs_error,
        rel_tol: settings.rel_error,
        dense_output: settings.enable_interp,
        max_steps: settings.max_steps,
        ..Default::default()
    };

    // Get or create precomputed ephemeris data.
    //
    // The integrator determines how far outside the nominal [begin, end]
    // interval the force closure can be evaluated. For Runge-Kutta methods
    // this is effectively zero; for Gauss-Jackson 8 the symmetric ±4·h_gj
    // startup extends the required range by ~4·gj_step_seconds on both
    // ends. We use `required_precompute_padding()` to size this correctly,
    // and validate that any user-supplied precomputed table covers the
    // padded range — not just the nominal interval.
    let (tmin, tmax) = if end > begin {
        (begin, end)
    } else {
        (end, begin)
    };
    let padding_secs = settings.required_precompute_padding();
    let padding = Duration::from_seconds(padding_secs);
    let required_min = tmin - padding;
    let required_max = tmax + padding;
    let interp: &Precomputed = match &settings.precomputed {
        Some(p) if required_min >= p.begin && required_max <= p.end => p,
        _ => &Precomputed::new_padded(&begin, &end, 60.0, padding_secs)?,
    };

    let gravity = settings.gravity_model.get();

    let ydot = |x: f64, y: &Matrix<6, C>| -> Matrix<6, C> {
        let time: Instant = begin + Duration::from_seconds(x);
        let pos_gcrf: Vector3 = y.block::<3, 1>(0, 0);
        let vel_gcrf: Vector3 = y.block::<3, 1>(3, 0);

        let (qgcrf2itrf, sun_gcrf, moon_gcrf) = interp.interp(&time).unwrap();
        let qitrf2gcrf = qgcrf2itrf.conjugate();
        let pos_itrf = qgcrf2itrf * pos_gcrf;

        // Decide whether to compute partials:
        // - C == 7: always (state transition matrix)
        // - C == 1: never (explicit integrators don't need partials in ydot)
        let need_partials = C == 7;

        let (mut accel, mut dadr, mut dadv) = if need_partials {
            let (ga, gp) = gravity.accel_and_partials(
                &pos_itrf,
                settings.gravity_degree as usize,
                settings.gravity_order as usize,
            );
            let ritrf2gcrf = qitrf2gcrf.to_rotation_matrix();
            (
                qitrf2gcrf * ga,
                ritrf2gcrf * gp * ritrf2gcrf.transpose(),
                Matrix3::zeros(),
            )
        } else {
            (
                qitrf2gcrf
                    * gravity.accel(
                        &pos_itrf,
                        settings.gravity_degree as usize,
                        settings.gravity_order as usize,
                    ),
                Matrix3::zeros(),
                Matrix3::zeros(),
            )
        };

        if settings.use_sun_gravity {
            if need_partials {
                let (a, p) = point_gravity_and_partials(&pos_gcrf, &sun_gcrf, consts::MU_SUN);
                accel += a;
                dadr += p;
            } else {
                accel += point_gravity(&pos_gcrf, &sun_gcrf, consts::MU_SUN);
            }
        }
        if settings.use_moon_gravity {
            if need_partials {
                let (a, p) = point_gravity_and_partials(&pos_gcrf, &moon_gcrf, consts::MU_MOON);
                accel += a;
                dadr += p;
            } else {
                accel += point_gravity(&pos_gcrf, &moon_gcrf, consts::MU_MOON);
            }
        }

        // Solid Earth tides. Partials w.r.t. position are negligible
        // (≲1e-12 of J2 partials) and are omitted from the STM update.
        if settings.tide_model != TideModel::None {
            let sun_itrf = qgcrf2itrf * sun_gcrf;
            let moon_itrf = qgcrf2itrf * moon_gcrf;
            let deltas =
                tides::solid_tide_deltas(&sun_itrf, &moon_itrf, &time, settings.tide_model);
            accel += qitrf2gcrf
                * tides::tide_accel(&pos_itrf, &deltas, gravity.gravity_constant, gravity.radius);
        }

        if let Some(props) = satprops {
            let ss: SimpleState = y.block::<6, 1>(0, 0);
            accel += solar_pressure_accel(&sun_gcrf, &pos_gcrf, &time, props, &ss);
            if pos_gcrf.norm() < 700.0e3 + consts::EARTH_RADIUS {
                let cd_a_over_m = props.cd_a_over_m(&time, &ss);
                if cd_a_over_m > 1e-6 {
                    if need_partials {
                        let (drag_a, drag_dr, drag_dv) = drag_and_partials(
                            &pos_gcrf,
                            &qgcrf2itrf,
                            &vel_gcrf,
                            &time,
                            cd_a_over_m,
                            settings.use_spaceweather,
                        );
                        accel += drag_a;
                        dadr += drag_dr;
                        dadv = drag_dv;
                    } else {
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
            }
            // Continuous thrust acceleration
            if let Some(a_thrust) = props.thrust_accel(&time, &pos_gcrf, &vel_gcrf) {
                accel += a_thrust;
            }
        }

        if C == 1 {
            let mut dy = Matrix::<6, C>::zeros();
            dy.set_block(0, 0, &vel_gcrf);
            dy.set_block(3, 0, &accel);
            dy
        } else if C == 7 {
            // Equation 7.42: dfdy * state_transition_matrix
            let mut dfdy = Matrix::<6, 6>::zeros();
            dfdy.set_block(0, 3, &Matrix3::eye());
            dfdy.set_block(3, 0, &dadr);
            dfdy.set_block(3, 3, &dadv);
            let dphi = dfdy * y.block::<6, 6>(0, 1);

            let mut dy = Matrix::<6, C>::zeros();
            dy.set_block(0, 0, &vel_gcrf);
            dy.set_block(3, 0, &accel);
            dy.set_block(0, 1, &dphi);
            dy
        } else {
            panic!("Invalid number of columns: {}", C);
        }
    };

    // Jacobian closure for RODAS4: computes the 6x6 df/dy matrix
    let jac_fn = |x: f64, y: &numeris::Vector<f64, 6>| -> Matrix<6, 6> {
        let time: Instant = begin + Duration::from_seconds(x);
        let pos_gcrf: Vector3 = y.block::<3, 1>(0, 0);
        let vel_gcrf: Vector3 = y.block::<3, 1>(3, 0);

        let (qgcrf2itrf, sun_gcrf, moon_gcrf) = interp.interp(&time).unwrap();
        let qitrf2gcrf = qgcrf2itrf.conjugate();
        let pos_itrf = qgcrf2itrf * pos_gcrf;

        let (_, gp) = gravity.accel_and_partials(
            &pos_itrf,
            settings.gravity_degree as usize,
            settings.gravity_order as usize,
        );
        let ritrf2gcrf = qitrf2gcrf.to_rotation_matrix();
        let mut dadr = ritrf2gcrf * gp * ritrf2gcrf.transpose();
        let mut dadv = Matrix3::zeros();

        if settings.use_sun_gravity {
            let (_, p) = point_gravity_and_partials(&pos_gcrf, &sun_gcrf, consts::MU_SUN);
            dadr += p;
        }
        if settings.use_moon_gravity {
            let (_, p) = point_gravity_and_partials(&pos_gcrf, &moon_gcrf, consts::MU_MOON);
            dadr += p;
        }

        if let Some(props) = satprops {
            let ss: SimpleState = y.block::<6, 1>(0, 0);
            if pos_gcrf.norm() < 700.0e3 + consts::EARTH_RADIUS {
                let cd_a_over_m = props.cd_a_over_m(&time, &ss);
                if cd_a_over_m > 1e-6 {
                    let (_, drag_dr, drag_dv) = drag_and_partials(
                        &pos_gcrf,
                        &qgcrf2itrf,
                        &vel_gcrf,
                        &time,
                        cd_a_over_m,
                        settings.use_spaceweather,
                    );
                    dadr += drag_dr;
                    dadv = drag_dv;
                }
            }
        }

        // Build the 6x6 Jacobian: df/dy where f = [vel; accel]
        let mut dfdy = Matrix::<6, 6>::zeros();
        dfdy.set_block(0, 3, &Matrix3::eye());
        dfdy.set_block(3, 0, &dadr);
        dfdy.set_block(3, 3, &dadv);
        dfdy
    };

    use crate::orbitprop::Integrator;

    let res = match settings.integrator {
        Integrator::RKV98 => ode::RKV98::integrate(0.0, x_end, state, &ydot, &odesettings),
        Integrator::RKV98NoInterp => {
            ode::RKV98NoInterp::integrate(0.0, x_end, state, &ydot, &odesettings)
        }
        Integrator::RKV87 => ode::RKV87::integrate(0.0, x_end, state, &ydot, &odesettings),
        Integrator::RKV65 => ode::RKV65::integrate(0.0, x_end, state, &ydot, &odesettings),
        Integrator::RKTS54 => ode::RKTS54::integrate(0.0, x_end, state, &ydot, &odesettings),
        Integrator::RODAS4 => {
            // RODAS4 only supports SimpleState (6x1 = Vector<f64, 6>)
            // At this point C==1 is guaranteed (C==7 was rejected above)
            let y0_vec: numeris::Vector<f64, 6> = state.block::<6, 1>(0, 0);
            let ydot_vec = |x: f64, y: &numeris::Vector<f64, 6>| -> numeris::Vector<f64, 6> {
                let time: Instant = begin + Duration::from_seconds(x);
                let pos_gcrf: Vector3 = y.block::<3, 1>(0, 0);
                let vel_gcrf: Vector3 = y.block::<3, 1>(3, 0);

                let (qgcrf2itrf, sun_gcrf, moon_gcrf) = interp.interp(&time).unwrap();
                let qitrf2gcrf = qgcrf2itrf.conjugate();
                let pos_itrf = qgcrf2itrf * pos_gcrf;

                let mut accel = qitrf2gcrf
                    * gravity.accel(
                        &pos_itrf,
                        settings.gravity_degree as usize,
                        settings.gravity_order as usize,
                    );

                if settings.use_sun_gravity {
                    accel += point_gravity(&pos_gcrf, &sun_gcrf, consts::MU_SUN);
                }
                if settings.use_moon_gravity {
                    accel += point_gravity(&pos_gcrf, &moon_gcrf, consts::MU_MOON);
                }

                if settings.tide_model != TideModel::None {
                    let sun_itrf = qgcrf2itrf * sun_gcrf;
                    let moon_itrf = qgcrf2itrf * moon_gcrf;
                    let deltas =
                        tides::solid_tide_deltas(&sun_itrf, &moon_itrf, &time, settings.tide_model);
                    accel += qitrf2gcrf
                        * tides::tide_accel(
                            &pos_itrf,
                            &deltas,
                            gravity.gravity_constant,
                            gravity.radius,
                        );
                }

                if let Some(props) = satprops {
                    let ss: SimpleState = y.block::<6, 1>(0, 0);
                    accel += solar_pressure_accel(&sun_gcrf, &pos_gcrf, &time, props, &ss);
                    if pos_gcrf.norm() < 700.0e3 + consts::EARTH_RADIUS {
                        let cd_a_over_m = props.cd_a_over_m(&time, &ss);
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
                    // Continuous thrust acceleration
                    if let Some(a_thrust) = props.thrust_accel(&time, &pos_gcrf, &vel_gcrf) {
                        accel += a_thrust;
                    }
                }

                let mut dy = numeris::Vector::<f64, 6>::zeros();
                dy.set_block(0, 0, &vel_gcrf);
                dy.set_block(3, 0, &accel);
                dy
            };

            let rosenbrock_res =
                ode::RODAS4::integrate(0.0, x_end, &y0_vec, ydot_vec, jac_fn, &odesettings)?;

            // Convert RosenbrockSolution<f64, 6> to Solution<f64, 6, C>
            // Since C==1, this is essentially the same data
            return Ok(PropagationResult {
                time_begin: begin,
                state_begin: *state,
                time_end: end,
                state_end: {
                    let mut s = Matrix::<6, C>::zeros();
                    s.set_block(0, 0, &rosenbrock_res.y);
                    s
                },
                accepted_steps: rosenbrock_res.accepted as u32,
                rejected_steps: rosenbrock_res.rejected as u32,
                num_eval: rosenbrock_res.evals as u32,
                odesol: None,
                gj_dense: None,
                integrator: settings.integrator,
            });
        }
        Integrator::GaussJackson8 => {
            // Gauss-Jackson 8 is specialised for 2nd-order ODEs: r'' = f(t, r, v).
            // It takes position and velocity separately (not a flat 6-vector)
            // and uses a fixed step size. Only supports C==1 (no STM).
            use crate::orbitprop::ode::{GJSettings, GaussJackson8};

            let r0: Vector3 = state.block::<3, 1>(0, 0);
            let v0: Vector3 = state.block::<3, 1>(3, 0);

            let accel_fn = |x: f64, r: &Vector3, v: &Vector3| -> Vector3 {
                let time: Instant = begin + Duration::from_seconds(x);
                let (qgcrf2itrf, sun_gcrf, moon_gcrf) = interp.interp(&time).unwrap();
                let qitrf2gcrf = qgcrf2itrf.conjugate();
                let pos_itrf = qgcrf2itrf * *r;

                let mut accel = qitrf2gcrf
                    * gravity.accel(
                        &pos_itrf,
                        settings.gravity_degree as usize,
                        settings.gravity_order as usize,
                    );

                if settings.use_sun_gravity {
                    accel += point_gravity(r, &sun_gcrf, consts::MU_SUN);
                }
                if settings.use_moon_gravity {
                    accel += point_gravity(r, &moon_gcrf, consts::MU_MOON);
                }

                if settings.tide_model != TideModel::None {
                    let sun_itrf = qgcrf2itrf * sun_gcrf;
                    let moon_itrf = qgcrf2itrf * moon_gcrf;
                    let deltas =
                        tides::solid_tide_deltas(&sun_itrf, &moon_itrf, &time, settings.tide_model);
                    accel += qitrf2gcrf
                        * tides::tide_accel(
                            &pos_itrf,
                            &deltas,
                            gravity.gravity_constant,
                            gravity.radius,
                        );
                }

                if let Some(props) = satprops {
                    // Reconstruct SimpleState from (r, v) for the force model's
                    // state-aware methods (area/mass, drag coefficient, etc.)
                    let mut ss: SimpleState = SimpleState::zeros();
                    ss.set_block(0, 0, r);
                    ss.set_block(3, 0, v);

                    accel += solar_pressure_accel(&sun_gcrf, r, &time, props, &ss);

                    if r.norm() < 700.0e3 + consts::EARTH_RADIUS {
                        let cd_a_over_m = props.cd_a_over_m(&time, &ss);
                        if cd_a_over_m > 1e-6 {
                            accel += drag_force(
                                r,
                                &pos_itrf,
                                v,
                                &time,
                                cd_a_over_m,
                                settings.use_spaceweather,
                            );
                        }
                    }

                    if let Some(a_thrust) = props.thrust_accel(&time, r, v) {
                        accel += a_thrust;
                    }
                }

                accel
            };

            let gj_settings = GJSettings {
                h: settings.gj_step_seconds,
                dense_output: settings.enable_interp,
                max_steps: settings.max_steps,
                ..GJSettings::default()
            };

            let mut gj_sol =
                GaussJackson8::integrate(0.0, x_end, &r0, &v0, accel_fn, &gj_settings)?;

            // Assemble final 6x1 state
            let mut final_state = Matrix::<6, C>::zeros();
            let mut r_final: numeris::Vector<f64, 6> = numeris::Vector::<f64, 6>::zeros();
            r_final.set_block(0, 0, &gj_sol.r);
            r_final.set_block(3, 0, &gj_sol.v);
            final_state.set_block(0, 0, &r_final);

            let dense = gj_sol.dense.take();
            return Ok(PropagationResult {
                time_begin: begin,
                state_begin: *state,
                time_end: end,
                state_end: final_state,
                accepted_steps: gj_sol.steps as u32,
                rejected_steps: 0,
                num_eval: gj_sol.evals as u32,
                odesol: None,
                gj_dense: dense,
                integrator: settings.integrator,
            });
        }
    }?;

    Ok(PropagationResult {
        time_begin: begin,
        state_begin: *state,
        time_end: end,
        state_end: res.y,
        accepted_steps: res.accepted as u32,
        rejected_steps: res.rejected as u32,
        num_eval: res.evals as u32,
        odesol: Some(res),
        gj_dense: None,
        integrator: settings.integrator,
    })
}

pub fn interp_propresult<const C: usize, T: TimeLike>(
    res: &PropagationResult<C>,
    time: &T,
) -> Result<StateType<C>> {
    use crate::orbitprop::Integrator;

    let time = time.as_instant();
    if time == res.time_begin {
        return Ok(res.state_begin);
    }
    let x = (time - res.time_begin).as_seconds();

    // Gauss-Jackson 8 uses its own dense-output format (per-step (t, r, v, a)
    // samples with quintic Hermite interpolation) rather than the RK solvers'
    // stage-based interpolant.
    if res.integrator == Integrator::GaussJackson8 {
        let dense = res
            .gj_dense
            .as_ref()
            .ok_or(Error::NoDenseOutputInSolution)?;
        // Rehydrate a minimal GJSolution just enough for the interpolator
        let gj_sol = crate::orbitprop::ode::GJSolution::<f64, 3> {
            t: 0.0, // unused by interpolate
            r: Vector3::zeros(),
            v: Vector3::zeros(),
            evals: 0,
            steps: 0,
            startup_iters: 0,
            dense: Some(dense.clone()),
        };
        let (r, v) = crate::orbitprop::ode::GaussJackson8::interpolate(x, &gj_sol)?;
        let mut out: StateType<C> = Matrix::<6, C>::zeros();
        let mut rv: numeris::Vector<f64, 6> = numeris::Vector::<f64, 6>::zeros();
        rv.set_block(0, 0, &r);
        rv.set_block(3, 0, &v);
        out.set_block(0, 0, &rv);
        return Ok(out);
    }

    let sol = res
        .odesol
        .as_ref()
        .filter(|s| s.dense.is_some())
        .ok_or(Error::NoDenseOutputInSolution)?;
    let result = match res.integrator {
        Integrator::RKV98 => ode::RKV98::interpolate(x, sol),
        Integrator::RKV98NoInterp => ode::RKV98NoInterp::interpolate(x, sol),
        Integrator::RKV87 => ode::RKV87::interpolate(x, sol),
        Integrator::RKV65 => ode::RKV65::interpolate(x, sol),
        Integrator::RKTS54 => ode::RKTS54::interpolate(x, sol),
        Integrator::RODAS4 => {
            return Err(Error::NoDenseOutputInSolution);
        }
        Integrator::GaussJackson8 => unreachable!("handled above"),
    };
    Ok(result?)
}

pub fn interp_propresult_batch<const C: usize>(
    res: &PropagationResult<C>,
    times: &[Instant],
) -> Result<Vec<StateType<C>>> {
    use crate::orbitprop::Integrator;

    let xs: Vec<f64> = times
        .iter()
        .map(|t| (*t - res.time_begin).as_seconds())
        .collect();

    if res.integrator == Integrator::GaussJackson8 {
        let dense = res
            .gj_dense
            .as_ref()
            .ok_or(Error::NoDenseOutputInSolution)?;
        let gj_sol = crate::orbitprop::ode::GJSolution::<f64, 3> {
            t: 0.0,
            r: Vector3::zeros(),
            v: Vector3::zeros(),
            evals: 0,
            steps: 0,
            startup_iters: 0,
            dense: Some(dense.clone()),
        };
        let pairs = crate::orbitprop::ode::GaussJackson8::interpolate_batch(&xs, &gj_sol)?;
        return Ok(pairs
            .into_iter()
            .map(|(r, v)| {
                let mut out: StateType<C> = Matrix::<6, C>::zeros();
                let mut rv: numeris::Vector<f64, 6> = numeris::Vector::<f64, 6>::zeros();
                rv.set_block(0, 0, &r);
                rv.set_block(3, 0, &v);
                out.set_block(0, 0, &rv);
                out
            })
            .collect());
    }

    let sol = res
        .odesol
        .as_ref()
        .filter(|s| s.dense.is_some())
        .ok_or(Error::NoDenseOutputInSolution)?;

    let results = match res.integrator {
        Integrator::RKV98 => ode::RKV98::interpolate_batch(&xs, sol),
        Integrator::RKV98NoInterp => ode::RKV98NoInterp::interpolate_batch(&xs, sol),
        Integrator::RKV87 => ode::RKV87::interpolate_batch(&xs, sol),
        Integrator::RKV65 => ode::RKV65::interpolate_batch(&xs, sol),
        Integrator::RKTS54 => ode::RKTS54::interpolate_batch(&xs, sol),
        Integrator::RODAS4 => {
            return Err(Error::NoDenseOutputInSolution);
        }
        Integrator::GaussJackson8 => unreachable!("handled above"),
    };
    Ok(results?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consts, orbitprop::SatPropertiesSimple};
    use std::f64::consts::PI;

    use std::fs::File;

    use crate::Duration;
    use std::io::{self, BufRead};

    // Tests use anyhow::Result so we can `?`-convert errors from various
    // crates (parse, Instant::from_datetime, etc.) that aren't part of
    // orbitprop::Error.
    use anyhow::Result;

    #[test]
    fn test_short_propagate() -> Result<()> {
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_seconds(0.1);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
            ..Default::default()
        };

        let _res1 = propagate(&state, &starttime, &stoptime, &settings, None)?;

        Ok(())
    }

    #[test]
    fn test_propagate() -> Result<()> {
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.25);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let mut settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
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
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(1.0);

        let mut state: SimpleState = SimpleState::zeros();

        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
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

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.5);

        let mut state: CovState = CovState::zeros();

        let theta = PI / 6.0;
        state[(0, 0)] = consts::GEO_R * theta.cos();
        state[(2, 0)] = consts::GEO_R * theta.sin();
        state[(4, 0)] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.cos();
        state[(5, 0)] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.sin();
        state.set_block(0, 1, &Matrix6::eye());

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
            ..Default::default()
        };

        // Made-up small variations in the state
        let dstate = numeris::vector![6.0, -10.0, 120.5, 0.1, 0.2, -0.3];

        // Propagate state (and state-transition matrix)
        let res = propagate(&state, &starttime, &stoptime, &settings, None)?;

        // Explicitly propagate state + dstate
        let res2 = propagate(
            &(state.block::<6, 1>(0, 0) + dstate),
            &starttime,
            &stoptime,
            &settings,
            None,
        )?;

        // Difference in states from explicitly propagating with
        // "dstate" change in initial conditions
        let dstate_prop = res2.state_end - res.state_end.block::<6, 1>(0, 0);

        // Difference in states estimated from state transition matrix
        let dstate_phi = res.state_end.block::<6, 6>(0, 1) * dstate;
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

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + crate::Duration::from_days(0.2);

        let mut state: CovState = CovState::zeros();

        let pgcrf = numeris::vector![3059573.85713792, 5855177.98848048, -7191.45042671];
        let vgcrf = numeris::vector![916.08123489, -468.22498656, 7700.48460839];

        // 30-deg inclination
        state.set_block(0, 0, &pgcrf);
        state.set_block(3, 0, &vgcrf);
        state.set_block(0, 1, &Matrix6::eye());

        let settings = PropSettings {
            abs_error: 1.0e-8,
            rel_error: 1.0e-8,
            gravity_degree: 4,
            ..Default::default()
        };

        let satprops: SatPropertiesSimple = SatPropertiesSimple::new(2.0 * 0.3 * 0.1 / 5.0, 0.0);

        // Made-up small variations in the state
        let dstate = numeris::vector![2.0, -4.0, 20.5, 0.05, 0.02, -0.01];

        // Propagate state (and state-transition matrix)

        let res = propagate(&state, &starttime, &stoptime, &settings, Some(&satprops))?;

        // Explicitly propagate state + dstate
        let res2 = propagate(
            &(state.block::<6, 1>(0, 0) + dstate),
            &starttime,
            &stoptime,
            &settings,
            Some(&satprops),
        )?;

        // Difference in states from explicitly propagating with
        // "dstate" change in initial conditions
        let dstate_prop = res2.state_end - res.state_end.block::<6, 1>(0, 0);

        let dstate_phi = res.state_end.block::<6, 6>(0, 1) * dstate;

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
                "Required GPS SP3 File: \"{}\" does not exist.
                Clone test vectors from:
                <https://storage.googleapis.com/satkit-testvecs/>
                or using python script in satkit repo: `python/test/download_testvecs.py`
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
                Ok(Instant::from_datetime(year, mon, day, hour, min, sec)?)
            })
            .collect::<Result<Vec<crate::Instant>, _>>()?;

        let file: File = File::open(testvecfile)?;

        let satnum: usize = 20;
        let satstr = format!("PG{}", satnum);
        let pitrf: Vec<Vector3> = io::BufReader::new(file)
            .lines()
            .filter(|x| {
                let rline = &x.as_ref().unwrap()[0..4];
                rline == satstr
            })
            .map(|rline| -> Result<Vector3> {
                let line = rline.unwrap();
                let lvals: Vec<&str> = line.split_whitespace().collect();
                let px: f64 = lvals[1].parse()?;
                let py: f64 = lvals[2].parse()?;
                let pz: f64 = lvals[3].parse()?;
                Ok(numeris::vector![px, py, pz] * 1.0e3)
            })
            .collect::<Result<Vec<Vector3>>>()?;

        assert!(times.len() == pitrf.len());
        let pgcrf: Vec<Vector3> = pitrf
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let q = crate::frametransform::qitrf2gcrf(&times[idx]);
                q * p
            })
            .collect();

        let v0 = numeris::vector![
            2.47130562e+03,
            2.94682753e+03,
            -5.34172176e+02,
            2.32565692e-02,
        ];

        let state0 = numeris::vector![pgcrf[0][0], pgcrf[0][1], pgcrf[0][2], v0[0], v0[1], v0[2]];
        let satprops: SatPropertiesSimple = SatPropertiesSimple::new(0.0, v0[3]);

        let settings = PropSettings {
            enable_interp: true,
            ..Default::default()
        };
        // Tides-off twin propagation: lets the test confirm that enabling
        // solid Earth tides is a strict improvement on this real-data
        // trajectory, not just that the wiring runs.
        let settings_no_tides = PropSettings {
            tide_model: crate::orbitprop::TideModel::None,
            ..settings.clone()
        };

        let res = propagate(
            &state0,
            &times[0],
            &times[times.len() - 1],
            &settings,
            Some(&satprops),
        )?;
        let res_nt = propagate(
            &state0,
            &times[0],
            &times[times.len() - 1],
            &settings_no_tides,
            Some(&satprops),
        )?;

        let max_per_axis = |r: &super::PropagationResult<1>| -> Result<f64> {
            let mut m = 0.0_f64;
            for iv in 0..pgcrf.len() {
                let interp_state = r.interp(&times[iv])?;
                for ix in 0..3 {
                    m = m.max((pgcrf[iv][ix] - interp_state[ix]).abs());
                }
            }
            Ok(m)
        };
        let max_axis_err = max_per_axis(&res)?;
        let max_axis_err_nt = max_per_axis(&res_nt)?;
        println!(
            "GPS SP3 max per-axis residual: with tides = {:.4} m, no tides = {:.4} m",
            max_axis_err, max_axis_err_nt
        );

        // Tightened threshold (was 8.0 m before solid Earth tides landed).
        // With Step 1 tides enabled and degree-4 gravity, residual sits
        // around 5.7 m for this 1-day GPS arc.
        assert!(
            max_axis_err < 6.5,
            "Max per-axis residual = {} m exceeds 6.5 m threshold",
            max_axis_err
        );
        assert!(
            max_axis_err < max_axis_err_nt,
            "Enabling tides should improve residual: with = {} m, no = {} m",
            max_axis_err,
            max_axis_err_nt
        );

        Ok(())
    }

    #[test]
    fn test_sun_moon_toggles() -> Result<()> {
        // Verify that disabling sun/moon gravity produces different results
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.5);

        let mut state: SimpleState = SimpleState::zeros();
        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings_all = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
            ..Default::default()
        };

        let settings_no_sun = PropSettings {
            use_sun_gravity: false,
            ..settings_all.clone()
        };

        let settings_no_moon = PropSettings {
            use_moon_gravity: false,
            ..settings_all.clone()
        };

        let settings_no_both = PropSettings {
            use_sun_gravity: false,
            use_moon_gravity: false,
            ..settings_all.clone()
        };

        let res_all = propagate(&state, &starttime, &stoptime, &settings_all, None)?;
        let res_no_sun = propagate(&state, &starttime, &stoptime, &settings_no_sun, None)?;
        let res_no_moon = propagate(&state, &starttime, &stoptime, &settings_no_moon, None)?;
        let res_no_both = propagate(&state, &starttime, &stoptime, &settings_no_both, None)?;

        // All results should be different from each other
        let pos_all = res_all.state_end.block::<3, 1>(0, 0);
        let pos_no_sun = res_no_sun.state_end.block::<3, 1>(0, 0);
        let pos_no_moon = res_no_moon.state_end.block::<3, 1>(0, 0);
        let pos_no_both = res_no_both.state_end.block::<3, 1>(0, 0);

        let diff_sun = (pos_all - pos_no_sun).norm();
        let diff_moon = (pos_all - pos_no_moon).norm();
        let diff_both = (pos_all - pos_no_both).norm();

        // Sun and moon perturbations should be measurable over half a day at GEO
        assert!(
            diff_sun > 1.0,
            "Disabling sun gravity should matter, diff = {} m",
            diff_sun
        );
        assert!(
            diff_moon > 1.0,
            "Disabling moon gravity should matter, diff = {} m",
            diff_moon
        );
        assert!(
            diff_both > diff_sun,
            "Disabling both should differ more than just sun"
        );

        Ok(())
    }

    #[test]
    fn test_solid_tides_perturb_orbit() -> Result<()> {
        // Verify that enabling solid Earth tides actually changes the
        // propagated state. Expected magnitude: ~tens of cm to a few m
        // over half a day at GEO (M&G Table 3.1).
        use crate::orbitprop::TideModel;

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.5);

        let mut state: SimpleState = SimpleState::zeros();
        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings_with = PropSettings {
            abs_error: 1.0e-10,
            rel_error: 1.0e-13,
            gravity_degree: 8,
            gravity_order: 8,
            tide_model: TideModel::SolidStep1,
            ..Default::default()
        };
        let settings_without = PropSettings {
            tide_model: TideModel::None,
            ..settings_with.clone()
        };

        let res_with = propagate(&state, &starttime, &stoptime, &settings_with, None)?;
        let res_without = propagate(&state, &starttime, &stoptime, &settings_without, None)?;

        let diff = (res_with.state_end.block::<3, 1>(0, 0)
            - res_without.state_end.block::<3, 1>(0, 0))
        .norm();
        println!("Tide-induced GEO position diff over 0.5d = {:.4} m", diff);
        // GEO half-day: tide-driven position drift should be roughly
        // 0.1 m to 10 m. Bounds are wide to absorb model & geometry effects.
        assert!(
            (0.05..50.0).contains(&diff),
            "Tide-induced GEO position diff over 0.5d = {} m (expected ~0.1-10)",
            diff
        );

        Ok(())
    }

    #[test]
    fn test_gravity_degree_order_in_propagator() -> Result<()> {
        // Verify that separate degree and order affect propagation results
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.25);

        let mut state: SimpleState = SimpleState::zeros();
        // Use an inclined orbit so tesseral harmonics matter
        let r = 7000.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let inc: f64 = std::f64::consts::PI / 4.0;
        state[0] = r;
        state[4] = v * inc.cos();
        state[5] = v * inc.sin();

        let settings_full = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 8,
            gravity_order: 8,
            ..Default::default()
        };

        let settings_zonal = PropSettings {
            gravity_order: 0,
            ..settings_full.clone()
        };

        let res_full = propagate(&state, &starttime, &stoptime, &settings_full, None)?;
        let res_zonal = propagate(&state, &starttime, &stoptime, &settings_zonal, None)?;

        let pos_full = res_full.state_end.block::<3, 1>(0, 0);
        let pos_zonal = res_zonal.state_end.block::<3, 1>(0, 0);
        let diff = (pos_full - pos_zonal).norm();

        assert!(
            diff > 0.1,
            "degree=8,order=8 vs degree=8,order=0 should differ, diff = {} m",
            diff
        );

        Ok(())
    }

    #[test]
    fn test_state_transition_with_toggles() -> Result<()> {
        // Verify state transition matrix still works with sun/moon disabled
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_days(0.5);

        let mut state: CovState = CovState::zeros();
        let theta = PI / 6.0;
        state[(0, 0)] = consts::GEO_R * theta.cos();
        state[(2, 0)] = consts::GEO_R * theta.sin();
        state[(4, 0)] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.cos();
        state[(5, 0)] = (consts::MU_EARTH / consts::GEO_R).sqrt() * theta.sin();
        state.set_block(0, 1, &Matrix6::eye());

        let settings = PropSettings {
            abs_error: 1.0e-9,
            rel_error: 1.0e-14,
            gravity_degree: 4,
            use_sun_gravity: false,
            use_moon_gravity: false,
            ..Default::default()
        };

        let dstate = numeris::vector![6.0, -10.0, 120.5, 0.1, 0.2, -0.3];

        let res = propagate(&state, &starttime, &stoptime, &settings, None)?;
        let res2 = propagate(
            &(state.block::<6, 1>(0, 0) + dstate),
            &starttime,
            &stoptime,
            &settings,
            None,
        )?;

        let dstate_prop = res2.state_end - res.state_end.block::<6, 1>(0, 0);
        let dstate_phi = res.state_end.block::<6, 6>(0, 1) * dstate;

        for ix in 0..6_usize {
            assert!(
                (dstate_prop[ix] - dstate_phi[ix]).abs() / dstate_prop[ix] < 1e-3,
                "State transition matrix mismatch at index {}",
                ix
            );
        }

        Ok(())
    }

    #[test]
    fn test_rodas4_low_orbit() -> Result<()> {
        // Propagate a low orbit (200 km) with drag using RODAS4.
        // Compare result with the default RKV98 integrator to verify consistency.
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_hours(2.0);

        let mut state: SimpleState = SimpleState::zeros();

        // ~200 km altitude circular orbit
        let r = consts::EARTH_RADIUS + 200.0e3;
        state[0] = r;
        state[4] = (consts::MU_EARTH / r).sqrt();

        // Typical small satellite: Cd=2.2, A=0.01 m^2, mass=1 kg
        let satprops = SatPropertiesSimple::new(2.2 * 0.01 / 1.0, 0.0);

        let settings_rodas4 = PropSettings {
            abs_error: 1.0e-8,
            rel_error: 1.0e-8,
            gravity_degree: 4,
            integrator: crate::orbitprop::Integrator::RODAS4,
            ..Default::default()
        };

        let settings_rkv98 = PropSettings {
            abs_error: 1.0e-8,
            rel_error: 1.0e-8,
            gravity_degree: 4,
            ..Default::default()
        };

        let res_rodas4 = propagate(
            &state,
            &starttime,
            &stoptime,
            &settings_rodas4,
            Some(&satprops),
        )?;
        let res_rkv98 = propagate(
            &state,
            &starttime,
            &stoptime,
            &settings_rkv98,
            Some(&satprops),
        )?;

        // Position agreement within 100 m over 2 hours at this altitude
        let pos_diff = (res_rodas4.state_end.block::<3, 1>(0, 0)
            - res_rkv98.state_end.block::<3, 1>(0, 0))
        .norm();
        assert!(
            pos_diff < 100.0,
            "RODAS4 vs RKV98 position diff = {} m (expected < 100 m)",
            pos_diff
        );

        Ok(())
    }

    #[test]
    fn test_rodas4_rejects_stm() {
        // RODAS4 should return an error when asked to propagate with STM
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap();
        let stoptime = starttime + Duration::from_hours(1.0);

        let mut state: CovState = CovState::zeros();
        state[(0, 0)] = consts::EARTH_RADIUS + 400.0e3;
        state[(4, 0)] = (consts::MU_EARTH / state[(0, 0)]).sqrt();
        state.set_block(0, 1, &Matrix6::eye());

        let settings = PropSettings {
            integrator: crate::orbitprop::Integrator::RODAS4,
            ..Default::default()
        };

        let result = propagate(&state, &starttime, &stoptime, &settings, None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("state transition matrix"),
            "Expected STM error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_gauss_jackson8_geo_matches_rkv98() -> Result<()> {
        // Propagate a GEO orbit for 2 hours with GJ8 and compare against
        // RKV98. For smooth high-altitude orbits the two should agree to
        // well within 1 m.
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_hours(2.0);

        let mut state: SimpleState = SimpleState::zeros();
        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings_gj8 = PropSettings {
            gravity_degree: 4,
            integrator: crate::orbitprop::Integrator::GaussJackson8,
            gj_step_seconds: 60.0,
            ..Default::default()
        };
        let settings_rkv98 = PropSettings {
            abs_error: 1.0e-11,
            rel_error: 1.0e-11,
            gravity_degree: 4,
            ..Default::default()
        };

        let res_gj8 = propagate(&state, &starttime, &stoptime, &settings_gj8, None)?;
        let res_rkv98 = propagate(&state, &starttime, &stoptime, &settings_rkv98, None)?;

        let pos_diff = (res_gj8.state_end.block::<3, 1>(0, 0)
            - res_rkv98.state_end.block::<3, 1>(0, 0))
        .norm();
        assert!(
            pos_diff < 1.0,
            "GJ8 vs RKV98 position diff = {:.3e} m (expected < 1 m)",
            pos_diff
        );

        Ok(())
    }

    #[test]
    fn test_gauss_jackson8_interp_matches_rkv98() -> Result<()> {
        // Propagate the same GEO orbit with GJ8 and RKV98, then interpolate
        // both at a grid of intermediate times. The interpolated positions
        // should agree to ~meter level (quintic Hermite is 5th-order, so we
        // won't match RKV98's 8th-order dense output perfectly, but for a
        // smooth GEO orbit the agreement should be excellent).
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_hours(2.0);

        let mut state: SimpleState = SimpleState::zeros();
        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        let settings_gj8 = PropSettings {
            gravity_degree: 4,
            integrator: crate::orbitprop::Integrator::GaussJackson8,
            gj_step_seconds: 60.0,
            enable_interp: true,
            ..Default::default()
        };
        let settings_rkv98 = PropSettings {
            abs_error: 1.0e-11,
            rel_error: 1.0e-11,
            gravity_degree: 4,
            enable_interp: true,
            ..Default::default()
        };

        let res_gj8 = propagate(&state, &starttime, &stoptime, &settings_gj8, None)?;
        let res_rkv98 = propagate(&state, &starttime, &stoptime, &settings_rkv98, None)?;

        // Interpolate at 12 non-boundary times
        let dt_hours: Vec<f64> = (1..=12).map(|i| (i as f64) * 2.0 / 13.0).collect();
        for dt_h in &dt_hours {
            let t = starttime + Duration::from_hours(*dt_h);
            let s_gj8 = res_gj8.interp(&t)?;
            let s_rkv98 = res_rkv98.interp(&t)?;
            let pos_diff = (s_gj8.block::<3, 1>(0, 0) - s_rkv98.block::<3, 1>(0, 0)).norm();
            assert!(
                pos_diff < 10.0,
                "GJ8 vs RKV98 interp diff at dt={}h = {:.3e} m (expected < 10 m)",
                dt_h,
                pos_diff
            );
        }

        // Batch interp should match point-wise interp
        let times: Vec<Instant> = dt_hours
            .iter()
            .map(|dt_h| starttime + Duration::from_hours(*dt_h))
            .collect();
        let batch = res_gj8.interp_batch(&times)?;
        for (i, t) in times.iter().enumerate() {
            let single = res_gj8.interp(t)?;
            let diff = (batch[i].block::<3, 1>(0, 0) - single.block::<3, 1>(0, 0)).norm();
            assert!(diff < 1e-9, "batch vs single interp diff = {:.3e}", diff);
        }

        Ok(())
    }

    #[test]
    fn test_gauss_jackson8_precompute_bounds() -> Result<()> {
        // GJ8 with a step size larger than 60 s needs the precomputed interp
        // table extended beyond the default 240-s padding. Before the fix,
        // `settings.precompute_terms(begin, end)` produced a table that was
        // too narrow for the backward startup stencil, and propagation
        // failed with "time outside of precomputed range".
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_hours(2.0);

        let mut state: SimpleState = SimpleState::zeros();
        state[0] = consts::GEO_R;
        state[4] = (consts::MU_EARTH / consts::GEO_R).sqrt();

        // 120-second step — GJ8 startup goes to t0 - 480 s, beyond the
        // old 240-s default padding.
        let mut settings = PropSettings {
            integrator: crate::orbitprop::Integrator::GaussJackson8,
            gj_step_seconds: 120.0,
            ..Default::default()
        };

        // 1. Auto-constructed precomputed (no user call) should work.
        let res_auto = propagate(&state, &starttime, &stoptime, &settings, None)?;

        // 2. User-precomputed should also work — `precompute_terms` must
        //    pick up the integrator-specific padding.
        settings.precompute_terms(&starttime, &stoptime)?;
        let res_user = propagate(&state, &starttime, &stoptime, &settings, None)?;

        // Both should agree
        let diff = (res_auto.state_end.block::<3, 1>(0, 0)
            - res_user.state_end.block::<3, 1>(0, 0))
        .norm();
        assert!(
            diff < 1e-6,
            "Auto- and user-precomputed GJ8 propagation should agree: diff = {:.3e}",
            diff
        );

        // 3. Sanity: the precomputed table's begin must actually be at
        //    least 4·gj_step before starttime.
        let pc = settings.precomputed.as_ref().expect("should be set");
        let pad_needed = Duration::from_seconds(4.0 * 120.0);
        assert!(
            pc.begin <= starttime - pad_needed,
            "Precomputed begin {} should cover startup back to {}",
            pc.begin,
            starttime - pad_needed
        );

        Ok(())
    }

    #[test]
    fn test_gauss_jackson8_rejects_stm() {
        // GJ8 should return an error when asked to propagate with STM (C=7)
        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0).unwrap();
        let stoptime = starttime + Duration::from_hours(1.0);

        let mut state: CovState = CovState::zeros();
        state[(0, 0)] = consts::GEO_R;
        state[(4, 0)] = (consts::MU_EARTH / consts::GEO_R).sqrt();
        state.set_block(0, 1, &Matrix6::eye());

        let settings = PropSettings {
            integrator: crate::orbitprop::Integrator::GaussJackson8,
            gj_step_seconds: 60.0,
            ..Default::default()
        };

        let result = propagate(&state, &starttime, &stoptime, &settings, None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("state transition matrix"),
            "Expected STM error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_continuous_thrust() -> Result<()> {
        use crate::orbitprop::{ContinuousThrust, ThrustProfile};
        use crate::Frame;

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        let stoptime = starttime + Duration::from_hours(2.0);

        // Circular orbit at 500 km
        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let mut state: SimpleState = SimpleState::zeros();
        state[0] = r;
        state[4] = v;

        let settings = PropSettings {
            gravity_degree: 4,
            ..Default::default()
        };

        // Propagate without thrust
        let res_no_thrust = propagate(&state, &starttime, &stoptime, &settings, None)?;

        // Propagate with in-track thrust in RIC
        let thrust = ThrustProfile::new(vec![ContinuousThrust::new(
            numeris::vector![0.0, 1.0e-4, 0.0], // 0.1 mm/s^2 in-track
            Frame::RTN,
            starttime,
            stoptime,
        )]);
        let satprops = SatPropertiesSimple::default().with_thrust(thrust);

        let res_thrust = propagate(&state, &starttime, &stoptime, &settings, Some(&satprops))?;

        // Thrust should increase the orbit — final radius should be larger
        let r_no_thrust = res_no_thrust.state_end.block::<3, 1>(0, 0).norm();
        let r_thrust = res_thrust.state_end.block::<3, 1>(0, 0).norm();
        assert!(
            r_thrust > r_no_thrust,
            "Along-track thrust should raise orbit: r_thrust={}, r_no_thrust={}",
            r_thrust,
            r_no_thrust
        );

        // The states should differ meaningfully (thrust had an effect)
        let pos_diff = (res_thrust.state_end.block::<3, 1>(0, 0)
            - res_no_thrust.state_end.block::<3, 1>(0, 0))
        .norm();
        assert!(
            pos_diff > 100.0,
            "Thrust should produce significant position difference: {} m",
            pos_diff
        );

        Ok(())
    }

    #[test]
    fn test_continuous_thrust_gcrf() -> Result<()> {
        use crate::orbitprop::{ContinuousThrust, ThrustProfile};
        use crate::Frame;

        let starttime = Instant::from_datetime(2015, 3, 20, 0, 0, 0.0)?;
        // Short propagation to verify GCRF thrust direction is respected
        let stoptime = starttime + Duration::from_minutes(10.0);

        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let mut state: SimpleState = SimpleState::zeros();
        state[0] = r;
        state[4] = v;

        let settings = PropSettings {
            gravity_degree: 4,
            ..Default::default()
        };

        // Thrust in +Z GCRF direction
        let thrust = ThrustProfile::new(vec![ContinuousThrust::new(
            numeris::vector![0.0, 0.0, 1.0e-3], // 1 mm/s^2 in +Z
            Frame::GCRF,
            starttime,
            stoptime,
        )]);
        let satprops = SatPropertiesSimple::default().with_thrust(thrust);

        let res_no_thrust = propagate(&state, &starttime, &stoptime, &settings, None)?;
        let res_thrust = propagate(&state, &starttime, &stoptime, &settings, Some(&satprops))?;

        // Z component of position should be larger with +Z thrust
        let z_no_thrust = res_no_thrust.state_end[(2, 0)];
        let z_thrust = res_thrust.state_end[(2, 0)];
        assert!(
            z_thrust > z_no_thrust,
            "+Z thrust should increase Z position: z_thrust={}, z_no_thrust={}",
            z_thrust,
            z_no_thrust
        );

        Ok(())
    }
}
