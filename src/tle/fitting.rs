use super::{Error, Result, TLE};

use crate::Instant;

use numeris::{Matrix, Vector};

/// Number of parameters being fit in the non-linear least-squares fit.
const NPARAM: usize = 7;

/// Termination status of TLE fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TleFitStatus {
    /// Converged: gradient norm below tolerance.
    GradientConverged,
    /// Converged: relative step size below tolerance.
    StepConverged,
    /// Converged: relative cost change below tolerance.
    CostConverged,
    /// Maximum number of iterations reached without convergence.
    MaxIterations,
    /// Damping parameter saturated; cannot make further progress.
    DampingSaturated,
}

impl core::fmt::Display for TleFitStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GradientConverged => write!(f, "Converged (gradient tolerance)"),
            Self::StepConverged => write!(f, "Converged (step size tolerance)"),
            Self::CostConverged => write!(f, "Converged (cost change tolerance)"),
            Self::MaxIterations => write!(f, "Maximum iterations reached"),
            Self::DampingSaturated => write!(f, "Damping parameter saturated"),
        }
    }
}

/// Result of a TLE fit, summarizing the Levenberg-Marquardt solver state
/// at termination.
#[derive(Debug, Clone)]
pub struct TleFitResult {
    /// Termination status.
    pub status: TleFitStatus,
    /// Initial residual norm `||r(x0)||`.
    pub orig_norm: f64,
    /// Final residual norm `||r(x*)||`.
    pub best_norm: f64,
    /// Final gradient norm `||J^T r||`.
    pub grad_norm: f64,
    /// Number of LM iterations performed.
    pub n_iter: usize,
    /// Total number of residual evaluations.
    pub n_res_evals: usize,
}

fn wrap_deg(x: f64) -> f64 {
    let mut v = x % 360.0;
    if v < 0.0 {
        v += 360.0;
    }
    v
}

fn tle_from_params(p: &[f64; NPARAM], epoch: Instant) -> TLE {
    TLE {
        epoch,
        inclination: wrap_deg(p[0]),
        eccen: p[1] % 360.0,
        raan: wrap_deg(p[2]),
        arg_of_perigee: wrap_deg(p[3]),
        mean_motion: p[4],
        mean_anomaly: wrap_deg(p[5]),
        bstar: p[6],
        ..Default::default()
    }
}

/// Evaluate position residuals (in TEME, meters) for the given parameters.
///
/// Returns a `3 * n_states` element vector laid out as
/// `[x0, y0, z0, x1, y1, z1, ...]` where element `i` is
/// `sgp4_pos[i] - target_pos[i]`.
fn residuals(
    params: &[f64; NPARAM],
    times: &[Instant],
    states_teme: &[[f64; 6]],
    epoch: Instant,
) -> Result<Vec<f64>> {
    let mut tle = tle_from_params(params, epoch);
    let out = crate::sgp4::sgp4(&mut tle, times)
        .map_err(|e| Error::Sgp4(format!("{e:?}")))?;
    let mut r = vec![0.0; states_teme.len() * 3];
    for (i, state) in states_teme.iter().enumerate() {
        for j in 0..3 {
            r[i * 3 + j] = out.pos[(j, i)] - state[j];
        }
    }
    Ok(r)
}

fn norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

impl TLE {
    ///
    /// Fit a TLE from a set of states and times.
    ///
    /// This function uses a Levenberg-Marquardt non-linear least squares fit
    /// to find TLE parameters that best match the provided states and times.
    ///
    /// # Arguments
    /// * `states_gcrf` - A slice of state vectors in GCRF coordinates.
    ///   State vectors are [f64; 6] with 1st three elements representing position
    ///   in meters and last three elements representing velocity in meters / second.
    /// * `times` - A slice of times corresponding to the state vectors.
    /// * `epoch` - The epoch time for the TLE.
    ///
    /// # Returns
    /// A tuple containing the fitted TLE and a [`TleFitResult`] describing the
    /// termination state of the fit.
    ///
    /// # Notes:
    ///
    /// * The optimizer is a Levenberg-Marquardt loop built on top of
    ///   [`numeris`] fixed-size linear algebra. Only 7 parameters are fit, so
    ///   the normal equations are a 7×7 system. The Jacobian is formed by
    ///   forward finite differences of the SGP4 propagator output.
    ///
    /// * The fitting process is performed in the TEME frame, with SGP4 used to
    ///   generate the states from the TLE. The input GCRF states are rotated
    ///   into the TEME frame by this function.
    ///
    /// * First and second derivatives of mean motion are ignored, as they are
    ///   not used by SGP4.
    ///
    /// * Parameters in the TLE that are fit:
    ///   - 0: Inclination (degrees)
    ///   - 1: Eccentricity
    ///   - 2: Right Ascension of Ascending Node (RAAN) (degrees)
    ///   - 3: Argument of Perigee (degrees)
    ///   - 4: Mean Motion (revolutions per day)
    ///   - 5: Mean Anomaly (degrees)
    ///   - 6: BSTAR drag term
    ///
    /// # Example:
    ///
    /// ```rust
    /// // Construct a GCRF state vector
    /// let altitude = 400.0e3;
    /// let r0 = satkit::consts::EARTH_RADIUS + altitude;
    /// let v0 = (satkit::consts::MU_EARTH / r0).sqrt();
    /// let inc: f64 = 97.0_f64.to_radians();
    /// let state0 = numeris::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
    /// let time0 = satkit::Instant::from_datetime(2016, 5, 16, 12, 0, 0.0).unwrap();
    ///
    /// // High-fidelity orbit propagation settings
    /// let settings = satkit::orbitprop::PropSettings {
    ///     enable_interp: true,
    ///     ..Default::default()
    /// };
    ///
    /// // Satellite has finite drag
    /// let satprops = satkit::orbitprop::SatPropertiesSimple::new(
    ///     2.0 * 10.0 / 3500.0,
    ///     10.0 / 3500.0,
    /// );
    ///
    /// // Propagate over a day
    /// let res = satkit::orbitprop::propagate(
    ///     &state0,
    ///     &time0,
    ///     &(time0 + satkit::Duration::from_seconds(86400.0)),
    ///     &settings,
    ///     Some(&satprops),
    /// ).unwrap();
    ///
    /// // Get high-fidelity states every 10 seconds via interpolation
    /// let times = (0..860)
    ///     .map(|i| time0 + satkit::Duration::from_seconds(i as f64 * 10.0))
    ///     .collect::<Vec<_>>();
    /// let states = times
    ///     .iter()
    ///     .map(|t| {
    ///         let s = res.interp(t).unwrap();
    ///         [s[0], s[1], s[2], s[3], s[4], s[5]]
    ///     })
    ///     .collect::<Vec<_>>();
    ///
    /// // Fit a TLE from the states and times
    /// let (tle, result) = satkit::TLE::fit_from_states(&states, &times, time0).unwrap();
    ///
    /// // Print results
    /// println!("status = {}", result.status);
    /// println!("Fitted TLE: {}", tle);
    ///
    /// ```
    pub fn fit_from_states(
        states_gcrf: &[[f64; 6]],
        times: &[Instant],
        epoch: Instant,
    ) -> Result<(Self, TleFitResult)> {
        // Make sure lengths are identical
        if states_gcrf.len() != times.len() {
            return Err(Error::StatesTimesLengthMismatch);
        } else if states_gcrf.is_empty() {
            return Err(Error::EmptyStates);
        }

        // Get the minimum time
        let min_time = times.iter().min().unwrap();
        let max_time = times.iter().max().unwrap();
        if epoch < *min_time || epoch > *max_time {
            return Err(Error::EpochOutOfRange {
                min: min_time.to_string(),
                max: max_time.to_string(),
            });
        }

        // Find the point that is closest to the epoch
        let mut closest_index = 0;
        let mut closest_time = *min_time;
        for (i, time) in times.iter().enumerate() {
            if *time == epoch {
                closest_index = i;
                closest_time = *time;
                break;
            } else if *time < epoch && *time > closest_time {
                closest_index = i;
                closest_time = *time;
            }
        }

        // Rotate states to the TEME frame from GCRF
        // (TLEs represent state in TEME)
        let states_teme = times
            .iter()
            .enumerate()
            .map(|(i, time)| {
                let q = crate::frametransform::qteme2gcrf(time).conjugate();
                let p = q * numeris::vector![
                    states_gcrf[i][0],
                    states_gcrf[i][1],
                    states_gcrf[i][2],
                ];
                let v = q * numeris::vector![
                    states_gcrf[i][3],
                    states_gcrf[i][4],
                    states_gcrf[i][5],
                ];
                [p[0], p[1], p[2], v[0], v[1], v[2]]
            })
            .collect::<Vec<_>>();

        // Get the state
        let closest_state = states_teme[closest_index];
        // Kepler representation
        let mut kepler = crate::kepler::Kepler::from_pv(
            numeris::vector![closest_state[0], closest_state[1], closest_state[2]],
            numeris::vector![closest_state[3], closest_state[4], closest_state[5]],
        )?;

        // Move Kepler state to epoch
        if (epoch - closest_time).as_microseconds().abs() > 10 {
            kepler = kepler.propagate(&(epoch - closest_time));
        }

        // Create initial guess of parameters from 2-body Kepler
        let mut params: [f64; NPARAM] = [
            kepler.incl.to_degrees(),
            kepler.eccen,
            kepler.raan.to_degrees(),
            kepler.w.to_degrees(),
            kepler.mean_motion() * 60.0 * 60.0 * 24.0 / (2.0 * std::f64::consts::PI),
            kepler.mean_anomaly().to_degrees(),
            0.0,
        ];

        // --- Levenberg-Marquardt loop -------------------------------------
        let grad_tol = 1e-8_f64;
        let x_tol = 1e-12_f64;
        let f_tol = 1e-12_f64;
        let max_iter = 100_usize;
        let mu_min = 1e-10_f64;
        let mu_max = 1e10_f64;
        let sqrt_eps = f64::EPSILON.sqrt();

        let mut r = residuals(&params, times, &states_teme, epoch)?;
        let mut n_res_evals: usize = 1;
        let orig_norm = norm_sq(&r).sqrt();
        let mut cost = 0.5 * norm_sq(&r);
        let mut mu = 1e-3_f64;

        let mut status = TleFitStatus::MaxIterations;
        let mut final_grad_norm = 0.0_f64;
        let mut iter_performed = 0_usize;

        'outer: for iter in 0..max_iter {
            iter_performed = iter + 1;

            // Finite-difference Jacobian columns: each column is a
            // (3 * n_states)-long vector. If perturbing a parameter puts
            // SGP4 into an unsupported region, fall back to a centered
            // difference with the opposite sign.
            let mut cols: Vec<Vec<f64>> = Vec::with_capacity(NPARAM);
            for k in 0..NPARAM {
                let h = sqrt_eps * params[k].abs().max(1.0);
                let col = {
                    let mut pert = params;
                    pert[k] += h;
                    match residuals(&pert, times, &states_teme, epoch) {
                        Ok(rp) => {
                            n_res_evals += 1;
                            rp.iter()
                                .zip(r.iter())
                                .map(|(a, b)| (a - b) / h)
                                .collect::<Vec<f64>>()
                        }
                        Err(_) => {
                            // Try a backward difference instead.
                            let mut pert = params;
                            pert[k] -= h;
                            let rp =
                                residuals(&pert, times, &states_teme, epoch)?;
                            n_res_evals += 1;
                            rp.iter()
                                .zip(r.iter())
                                .map(|(a, b)| (b - a) / h)
                                .collect::<Vec<f64>>()
                        }
                    }
                };
                cols.push(col);
            }

            // Accumulate J^T J (symmetric, NxN) and J^T r (N-vector).
            let mut jtj = Matrix::<f64, NPARAM, NPARAM>::zeros();
            let mut jtr = Vector::<f64, NPARAM>::zeros();
            for k in 0..NPARAM {
                jtr[k] = cols[k].iter().zip(r.iter()).map(|(a, b)| a * b).sum();
                for l in k..NPARAM {
                    let s: f64 = cols[k]
                        .iter()
                        .zip(cols[l].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    jtj[(k, l)] = s;
                    jtj[(l, k)] = s;
                }
            }

            let g_norm = jtr.norm();
            final_grad_norm = g_norm;
            if g_norm < grad_tol {
                status = TleFitStatus::GradientConverged;
                break 'outer;
            }

            // Inner loop: try increasingly damped steps until one is accepted
            // or the damping saturates.
            loop {
                let mut damped = jtj;
                for i in 0..NPARAM {
                    damped[(i, i)] += mu;
                }
                let lu = damped
                    .lu()
                    .map_err(|e| Error::SingularNormalEquations(format!("{e:?}")))?;
                let neg_g: Vector<f64, NPARAM> = -jtr;
                let delta = lu.solve(&neg_g);

                let mut trial = params;
                for k in 0..NPARAM {
                    trial[k] += delta[k];
                }
                // Treat SGP4 failures on trial parameters as a rejected step.
                let r_new = match residuals(&trial, times, &states_teme, epoch) {
                    Ok(rn) => {
                        n_res_evals += 1;
                        rn
                    }
                    Err(_) => {
                        let new_mu = mu * 10.0;
                        if new_mu >= mu_max {
                            status = TleFitStatus::DampingSaturated;
                            break 'outer;
                        }
                        mu = new_mu;
                        continue;
                    }
                };
                let cost_new = 0.5 * norm_sq(&r_new);

                // Predicted reduction: delta^T (mu * delta - g)
                //                   = mu * |delta|^2 - delta . g
                let delta_norm_sq: f64 =
                    (0..NPARAM).map(|i| delta[i] * delta[i]).sum();
                let delta_dot_g: f64 =
                    (0..NPARAM).map(|i| delta[i] * jtr[i]).sum();
                let predicted = mu * delta_norm_sq - delta_dot_g;
                let actual = cost - cost_new;

                if predicted > 0.0 && actual > 0.0 {
                    // Accept step
                    params = trial;
                    r = r_new;
                    cost = cost_new;
                    mu = (mu * 0.1).max(mu_min);

                    let delta_norm = delta_norm_sq.sqrt();
                    let params_norm =
                        params.iter().map(|x| x * x).sum::<f64>().sqrt();

                    if delta_norm < x_tol * (1.0 + params_norm) {
                        status = TleFitStatus::StepConverged;
                        break 'outer;
                    }
                    if actual.abs() < f_tol * (1.0 + cost.abs()) {
                        status = TleFitStatus::CostConverged;
                        break 'outer;
                    }
                    break; // proceed to next outer iteration
                } else {
                    // Reject step, increase damping.
                    let new_mu = mu * 10.0;
                    if new_mu >= mu_max {
                        status = TleFitStatus::DampingSaturated;
                        break 'outer;
                    }
                    mu = new_mu;
                }
            }
        }

        let final_tle = tle_from_params(&params, epoch);
        Ok((
            final_tle,
            TleFitResult {
                status,
                orig_norm,
                best_norm: norm_sq(&r).sqrt(),
                grad_norm: final_grad_norm,
                n_iter: iter_performed,
                n_res_evals,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_fit_from_states() -> Result<()> {
        let r0 = crate::consts::GEO_R;
        let v0 = (crate::consts::MU_EARTH / r0).sqrt();
        let inc: f64 = 15.0_f64.to_radians();
        let state0 = numeris::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2022, 5, 16, 12, 0, 0.0)?;

        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(86400.0)),
            &settings,
            None,
        )?;

        let times = (0..8640)
            .map(|i| time0 + crate::Duration::from_seconds(i as f64 * 10.0))
            .collect::<Vec<_>>();
        let states = times
            .iter()
            .map(|t| {
                let s = res.interp(t).unwrap();
                [s[0], s[1], s[2], s[3], s[4], s[5]]
            })
            .collect::<Vec<_>>();

        let (_tle, _result) = TLE::fit_from_states(&states, &times, time0)?;
        Ok(())
    }

    #[test]
    fn test_fit_from_states_with_drag() -> Result<()> {
        let altitude = 400.0e3;
        let r0 = crate::consts::EARTH_RADIUS + altitude;
        let v0 = (crate::consts::MU_EARTH / r0).sqrt();
        let inc: f64 = 97.0_f64.to_radians();
        let state0 = numeris::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2016, 5, 16, 12, 0, 0.0)?;

        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let satprops = crate::orbitprop::SatPropertiesSimple::new(
            2.0 * 10.0 / 3500.0,
            10.0 / 3500.0,
        );

        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(86400.0)),
            &settings,
            Some(&satprops),
        )?;

        let times = (0..8640)
            .map(|i| time0 + crate::Duration::from_seconds(i as f64 * 10.0))
            .collect::<Vec<_>>();
        let states = times
            .iter()
            .map(|t| {
                let s = res.interp(t).unwrap();
                [s[0], s[1], s[2], s[3], s[4], s[5]]
            })
            .collect::<Vec<_>>();

        let (tle, result) = TLE::fit_from_states(&states, &times, time0)?;
        println!("status = {}", result.status);
        println!("Fitted TLE: {}", tle);
        Ok(())
    }
}
