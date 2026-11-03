use super::{Error, Result, TLE};

use crate::sgp4::{GravConst, OpsMode};
use crate::Instant;

use numeris::{Matrix, Vector};

/// Number of parameters being fit in the non-linear least-squares fit.
const NPARAM: usize = 7;

/// Floor on the magnitude each parameter is taken to have when choosing its
/// finite-difference step `sqrt(ε)·max(|x|, FD_SCALE)`. Mean motion
/// (rev/day), mean longitude (degrees) and B* are of order one or smaller
/// and take the usual floor of 1. The dimensionless (h, k) and (p, q) pairs
/// need a smaller one: near a singular element (e ≈ 0 or i ≈ 0) they are
/// themselves tiny, and a step of 1.5e-8 swung the node or perigee direction
/// by tens of degrees, so the Jacobian was a secant across the singularity
/// and the fit crawled (an exactly equatorial arc ran out of iterations at
/// ~1 m). With 1e-3 the step is 1.5e-11, which still moves the position by
/// ~1 mm at GEO, well above round-off.
const FD_SCALE: [f64; NPARAM] = [1e-3, 1e-3, 1e-3, 1e-3, 1.0, 1.0, 1.0];

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

/// Equinoctial fit parameters for `tle` with retrograde factor `fr`.
///
/// The fit is done in equinoctial elements, which are non-singular at
/// e = 0 and i = 0. Parameters:
///   - 0: k = e·cos(ϖ)
///   - 1: h = e·sin(ϖ)
///   - 2: p = tan(i/2)·sin(Ω)  (tan((180° − i)/2) for a retrograde fit)
///   - 3: q = tan(i/2)·cos(Ω)  (likewise)
///   - 4: mean motion (revolutions per day)
///   - 5: mean longitude λ = M + ϖ (degrees)
///   - 6: BSTAR drag term
///
/// where ϖ = ω + f·Ω and the retrograde factor f is +1, or −1 for an
/// orbit with i > 90°, which moves the singularity of `p`, `q` from
/// i = 180° to i = 0°, well away from the orbit being fit.
///
/// With the classical (e, i, Ω, ω, M) parameters a near-equatorial orbit
/// cannot be fit: Ω is undetermined at i = 0, and the residuals depend on a
/// node rotation only in proportion to i, so a fit seeded with the wrong
/// node collapses the inclination to zero instead (a GEO arc came back
/// kilometres off with i ≈ 1e-10°). Near e = 0 the ω and M columns are
/// likewise collinear.
fn params_from_tle(tle: &TLE, fr: f64) -> [f64; NPARAM] {
    let half = if fr > 0.0 {
        tle.inclination
    } else {
        180.0 - tle.inclination
    }
    .to_radians()
        / 2.0;
    let t = half.tan();
    let lp = tle.arg_of_perigee + fr * tle.raan;
    [
        tle.eccen * lp.to_radians().cos(),
        tle.eccen * lp.to_radians().sin(),
        t * tle.raan.to_radians().sin(),
        t * tle.raan.to_radians().cos(),
        tle.mean_motion,
        wrap_deg(tle.mean_anomaly + lp),
        tle.bstar,
    ]
}

/// Convert equinoctial fit parameters (see [`params_from_tle`]) with
/// retrograde factor `fr` to a TLE at `epoch`.
fn tle_from_params(p: &[f64; NPARAM], fr: f64, epoch: Instant) -> TLE {
    let eccen = p[0].hypot(p[1]);
    let lp = p[1].atan2(p[0]).to_degrees();
    let half = p[2].hypot(p[3]).atan().to_degrees();
    let inclination = if fr > 0.0 {
        2.0 * half
    } else {
        180.0 - 2.0 * half
    };
    let raan = p[2].atan2(p[3]).to_degrees();
    TLE {
        epoch,
        inclination,
        eccen,
        raan: wrap_deg(raan),
        arg_of_perigee: wrap_deg(lp - fr * raan),
        mean_motion: p[4],
        mean_anomaly: wrap_deg(p[5] - lp),
        bstar: p[6],
        ..Default::default()
    }
}

/// Check that a fitted element set lies in the domain a TLE can represent.
fn validate_fit(tle: &TLE) -> Result<()> {
    let check = |field: &'static str, value: f64, ok: bool, range: &'static str| {
        if ok {
            Ok(())
        } else {
            Err(Error::FitElementOutOfRange {
                field,
                value,
                range,
            })
        }
    };
    check(
        "eccentricity",
        tle.eccen,
        (0.0..1.0).contains(&tle.eccen),
        "[0, 1)",
    )?;
    check(
        "mean motion",
        tle.mean_motion,
        tle.mean_motion > 0.0 && tle.mean_motion.is_finite(),
        "> 0",
    )?;
    check(
        "inclination",
        tle.inclination,
        (0.0..=180.0).contains(&tle.inclination),
        "[0, 180] degrees",
    )?;
    Ok(())
}

/// Evaluate position residuals (in TEME, meters) for the given parameters.
///
/// Returns a `3 * n_states` element vector laid out as
/// `[x0, y0, z0, x1, y1, z1, ...]` where element `i` is
/// `sgp4_pos[i] - target_pos[i]`.
fn residuals(
    params: &[f64; NPARAM],
    fr: f64,
    times: &[Instant],
    states_teme: &[[f64; 6]],
    epoch: Instant,
    model: (GravConst, OpsMode),
) -> Result<Vec<f64>> {
    let mut tle = tle_from_params(params, fr, epoch);
    let out = crate::sgp4::sgp4_full(&mut tle, times, model.0, model.1)
        .map_err(|e| Error::Sgp4(format!("{e:?}")))?;
    // Reject any trial step whose propagation failed at one or more times (e.g.
    // a decayed orbit): those columns are NaN, and folding them into the
    // residuals would silently corrupt the least-squares step.
    if let Some(err) = out
        .errcode
        .iter()
        .find(|e| **e != crate::sgp4::SGP4Error::SGP4Success)
    {
        return Err(Error::Sgp4(format!("{err:?}")));
    }
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
    ///   into the TEME frame by this function, with the full IERS 2010
    ///   rotation (`frametransform::rotation(Frame::GCRF, Frame::TEME, t)`).
    ///
    /// * SGP4 runs with the WGS72 gravity model and the AFSPC ops mode, the
    ///   defaults of [`sgp4`](crate::sgp4::sgp4) and of catalog element sets,
    ///   so propagate the fitted TLE with those. Use
    ///   [`fit_from_states_full`](Self::fit_from_states_full) to fit for
    ///   another gravity model or ops mode (satkit before 0.24 fitted with
    ///   WGS84).
    ///
    /// * First and second derivatives of mean motion are ignored, as they are
    ///   not used by SGP4.
    ///
    /// * The damping is Marquardt-scaled (`μ·diag(JᵀJ)`), so it is
    ///   effective regardless of the residual count or the units of the
    ///   parameters. The fit fails with [`Error::FitElementOutOfRange`]
    ///   if it terminates outside the domain a TLE can represent.
    ///
    /// * The fit is parameterized in equinoctial elements (Broucke and
    ///   Cefola: `h = e·sin(ω+Ω)`, `k = e·cos(ω+Ω)`, `p = tan(i/2)·sin Ω`,
    ///   `q = tan(i/2)·cos Ω`, mean longitude `λ = M+ω+Ω`, with the
    ///   retrograde form for `i > 90°`), plus mean motion and BSTAR, and
    ///   converted to the classical TLE elements at the end. Unlike the
    ///   classical elements these are not singular for circular or
    ///   equatorial orbits, so near-circular and near-equatorial (e.g.
    ///   geostationary) arcs fit as well as any other.
    ///   The returned eccentricity is always non-negative.
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
        Self::fit_from_states_full(states_gcrf, times, epoch, GravConst::WGS72, OpsMode::AFSPC)
    }

    /// [`fit_from_states`](Self::fit_from_states) with an explicit SGP4
    /// gravity model and ops mode; the fitted TLE reproduces the states when
    /// propagated with [`sgp4_full`](crate::sgp4::sgp4_full) under the same
    /// two.
    pub fn fit_from_states_full(
        states_gcrf: &[[f64; 6]],
        times: &[Instant],
        epoch: Instant,
        gravconst: GravConst,
        opsmode: OpsMode,
    ) -> Result<(Self, TleFitResult)> {
        let model = (gravconst, opsmode);
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
        // (TLEs represent state in TEME). The full rotation, not the
        // approximate `qteme2gcrf` (0.3", ~10-20 m at LEO).
        let states_teme = times
            .iter()
            .enumerate()
            .map(|(i, time)| -> Result<[f64; 6]> {
                let q =
                    crate::frametransform::rotation(crate::Frame::GCRF, crate::Frame::TEME, time)?;
                let p =
                    q * numeris::vector![states_gcrf[i][0], states_gcrf[i][1], states_gcrf[i][2],];
                let v =
                    q * numeris::vector![states_gcrf[i][3], states_gcrf[i][4], states_gcrf[i][5],];
                Ok([p[0], p[1], p[2], v[0], v[1], v[2]])
            })
            .collect::<Result<Vec<_>>>()?;

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

        // Initial guess of parameters from 2-body Kepler, in equinoctial
        // elements (prograde unless the orbit is retrograde).
        let incl_deg = kepler.incl.to_degrees();
        let fr = if incl_deg > 90.0 { -1.0 } else { 1.0 };
        let mut params = params_from_tle(
            &Self {
                inclination: incl_deg,
                eccen: kepler.eccen,
                raan: kepler.raan.to_degrees(),
                arg_of_perigee: kepler.argp.to_degrees(),
                mean_motion: kepler.mean_motion() * 60.0 * 60.0 * 24.0
                    / (2.0 * std::f64::consts::PI),
                mean_anomaly: kepler.mean_anomaly().to_degrees(),
                ..Default::default()
            },
            fr,
        );

        // --- Levenberg-Marquardt loop -------------------------------------
        let grad_tol = 1e-8_f64;
        let x_tol = 1e-12_f64;
        let f_tol = 1e-12_f64;
        let max_iter = 100_usize;
        // The damping parameter is dimensionless: it multiplies the diagonal
        // of JᵀJ (Marquardt scaling), so its useful range does not depend on
        // the number of residuals or the units of the parameters. (An
        // absolute `μ·I` never bit here — diag(JᵀJ) is ~1e14–1e18 for a day
        // of 10 s samples, so the loop degenerated to undamped Gauss-Newton
        // and stalled near e = 0, where the ω and M columns are collinear.)
        let mu_min = 1e-12_f64;
        let mu_max = 1e12_f64;
        let sqrt_eps = f64::EPSILON.sqrt();

        let mut r = residuals(&params, fr, times, &states_teme, epoch, model)?;
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
                let h = sqrt_eps * params[k].abs().max(FD_SCALE[k]);
                let col = {
                    let mut pert = params;
                    pert[k] += h;
                    match residuals(&pert, fr, times, &states_teme, epoch, model) {
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
                            let rp = residuals(&pert, fr, times, &states_teme, epoch, model)?;
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
                    let s: f64 = cols[k].iter().zip(cols[l].iter()).map(|(a, b)| a * b).sum();
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

            // Marquardt scaling: damp each parameter in proportion to its own
            // curvature, and solve the normal equations in the column-scaled
            // space (unit diagonal) so the 1e20-fold spread between, say, the
            // mean-motion and B* columns does not swamp the LU solve. A
            // column that does not affect the residuals at all (e.g. B* on a
            // drag-free arc) has a zero diagonal; floor it so that direction
            // is still damped rather than left singular.
            let max_diag = (0..NPARAM).fold(0.0_f64, |m, i| m.max(jtj[(i, i)]));
            let diag_floor = (max_diag * 1e-12).max(f64::MIN_POSITIVE);
            let mut dscale = [0.0_f64; NPARAM];
            for (i, d) in dscale.iter_mut().enumerate() {
                *d = jtj[(i, i)].max(diag_floor);
            }
            let sqrt_d: [f64; NPARAM] = core::array::from_fn(|i| dscale[i].sqrt());
            let mut jtj_s = Matrix::<f64, NPARAM, NPARAM>::zeros();
            let mut neg_g_s = Vector::<f64, NPARAM>::zeros();
            for i in 0..NPARAM {
                for j in 0..NPARAM {
                    jtj_s[(i, j)] = jtj[(i, j)] / (sqrt_d[i] * sqrt_d[j]);
                }
                neg_g_s[i] = -jtr[i] / sqrt_d[i];
            }

            // Inner loop: try increasingly damped steps until one is accepted
            // or the damping saturates.
            loop {
                let mut damped = jtj_s;
                for i in 0..NPARAM {
                    damped[(i, i)] += mu;
                }
                let lu = damped
                    .lu()
                    .map_err(|e| Error::SingularNormalEquations(format!("{e:?}")))?;
                let delta_s = lu.solve(&neg_g_s);
                let mut delta = Vector::<f64, NPARAM>::zeros();
                for i in 0..NPARAM {
                    delta[i] = delta_s[i] / sqrt_d[i];
                }

                let mut trial = params;
                for k in 0..NPARAM {
                    trial[k] += delta[k];
                }
                // Treat SGP4 failures on trial parameters as a rejected step.
                let r_new = match residuals(&trial, fr, times, &states_teme, epoch, model) {
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

                // Predicted reduction: delta^T (mu * D * delta - g)
                let delta_norm_sq: f64 = (0..NPARAM).map(|i| delta[i] * delta[i]).sum();
                let delta_dot_dd: f64 = (0..NPARAM).map(|i| dscale[i] * delta[i] * delta[i]).sum();
                let delta_dot_g: f64 = (0..NPARAM).map(|i| delta[i] * jtr[i]).sum();
                let predicted = mu * delta_dot_dd - delta_dot_g;
                let actual = cost - cost_new;

                if predicted > 0.0 && actual > 0.0 {
                    // Accept step, keeping the mean longitude wrapped.
                    params = trial;
                    params[5] = wrap_deg(params[5]);
                    r = r_new;
                    cost = cost_new;
                    mu = (mu * 0.1).max(mu_min);

                    let delta_norm = delta_norm_sq.sqrt();
                    let params_norm = params.iter().map(|x| x * x).sum::<f64>().sqrt();

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

        let final_tle = tle_from_params(&params, fr, epoch);
        validate_fit(&final_tle)?;
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

        let satprops =
            crate::orbitprop::SatPropertiesSimple::new(2.0 * 10.0 / 3500.0, 10.0 / 3500.0);

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

    /// Propagate a state for `hours` with the high-fidelity propagator and
    /// fit a TLE to 10 s samples of the arc. Returns the TLE, the fit
    /// result and the position RMS (metres) over the samples.
    fn fit_hifi_arc(
        r0: f64,
        inc_deg: f64,
        eccen: f64,
        hours: f64,
        satprops: Option<&dyn crate::orbitprop::SatProperties>,
    ) -> Result<(TLE, TleFitResult, f64)> {
        let a = r0 / (1.0 - eccen);
        let v0 = (crate::consts::MU_EARTH * (2.0 / r0 - 1.0 / a)).sqrt();
        let inc = inc_deg.to_radians();
        let state0 = numeris::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2023, 5, 16, 12, 0, 0.0)?;
        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };
        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(hours * 3600.0)),
            &settings,
            satprops,
        )?;
        let n = (hours * 360.0) as usize;
        let times = (0..n)
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
        let rms = result.best_norm / ((3 * n) as f64).sqrt();
        println!(
            "fit r0={:.0} i={} e0={}: e={:+.3e} rms={:.1} m orig_rms={:.1} m iters={} evals={} status={}",
            r0,
            inc_deg,
            eccen,
            tle.eccen,
            rms,
            result.orig_norm / ((3 * n) as f64).sqrt(),
            result.n_iter,
            result.n_res_evals,
            result.status
        );
        Ok((tle, result, rms))
    }

    /// Regression for issue #189: a circular LEO arc (the doc example above)
    /// used to stall at ~7.5 km RMS with a *negative* eccentricity because
    /// the damping never bit and the fit wandered across e = 0. The fitted
    /// TLE must be physical and must survive a trip through its two lines.
    #[test]
    fn test_fit_circular_leo_eccentricity_nonnegative() -> Result<()> {
        let satprops =
            crate::orbitprop::SatPropertiesSimple::new(2.0 * 10.0 / 3500.0, 10.0 / 3500.0);
        let r0 = crate::consts::EARTH_RADIUS + 400.0e3;
        for props in [
            None,
            Some(&satprops as &dyn crate::orbitprop::SatProperties),
        ] {
            let (tle, _result, rms) = fit_hifi_arc(r0, 97.0, 0.0, 24.0, props)?;
            assert!(tle.eccen >= 0.0, "negative eccentricity: {}", tle.eccen);
            assert!(rms < 1500.0, "fit RMS {rms} m too large");
            let lines = tle.to_2line()?;
            let back = TLE::from_lines(&lines)?.remove(0);
            assert!((back.eccen - tle.eccen).abs() < 1e-7);
            assert!(back.eccen >= 0.0);
        }
        // A circular geosynchronous arc. With classical elements seeded at
        // e = 0 this used to stall at ~7 km RMS against the e = 0
        // singularity (it looked like a deep-space model floor); it fits to
        // ~650 m.
        let (tle, _result, rms) = fit_hifi_arc(crate::consts::GEO_R, 15.0, 0.0, 24.0, None)?;
        assert!(tle.eccen >= 0.0, "negative eccentricity: {}", tle.eccen);
        assert!(rms < 1500.0, "GEO fit RMS {rms} m too large");
        // Slightly eccentric seed fits well already; must not regress.
        let (tle, _result, rms) = fit_hifi_arc(r0, 97.0, 1e-3, 24.0, None)?;
        assert!(tle.eccen >= 0.0);
        assert!(rms < 1000.0, "fit RMS {rms} m too large");
        Ok(())
    }

    /// A negative eccentricity has no TLE representation; writing `|e|`
    /// would silently describe a different orbit.
    #[test]
    fn test_to_2line_rejects_negative_eccentricity() {
        let tle = TLE {
            epoch: Instant::from_datetime(2023, 5, 16, 12, 0, 0.0).unwrap(),
            inclination: 97.0,
            eccen: -1.2e-4,
            mean_motion: 15.5,
            ..Default::default()
        };
        assert!(matches!(
            tle.to_2line(),
            Err(Error::EccentricityOutOfRange(e)) if e < 0.0
        ));
        let tle = TLE {
            eccen: 1.2e-4,
            ..tle
        };
        assert!(tle.to_2line().is_ok());
    }

    /// GCRF states of `tle` from SGP4 under `gc` at `n` times 10 min apart,
    /// rotated from TEME with the full rotation.
    fn sgp4_states_gcrf(tle: &TLE, n: usize, gc: GravConst) -> (Vec<[f64; 6]>, Vec<Instant>) {
        let times: Vec<Instant> = (0..n)
            .map(|i| tle.epoch + crate::Duration::from_minutes(10.0 * i as f64))
            .collect();
        let mut src = tle.clone();
        let out = crate::sgp4::sgp4_full(&mut src, &times, gc, OpsMode::AFSPC).unwrap();
        let states = times
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let q = crate::frametransform::rotation(crate::Frame::TEME, crate::Frame::GCRF, t)
                    .unwrap();
                let p = q * numeris::vector![out.pos[(0, i)], out.pos[(1, i)], out.pos[(2, i)]];
                let v = q * numeris::vector![out.vel[(0, i)], out.vel[(1, i)], out.vel[(2, i)]];
                [p[0], p[1], p[2], v[0], v[1], v[2]]
            })
            .collect();
        (states, times)
    }

    /// States generated by SGP4 are reproduced by the fitted TLE only when
    /// the fit uses the same gravity model and the exact TEME rotation.
    /// Before 0.24 the fit used WGS84 and the approximate `qteme2gcrf`, so a
    /// WGS72 element set (the catalog default) came back tens of metres off.
    #[test]
    fn test_fit_recovers_sgp4_states() -> Result<()> {
        let tle = TLE::load_2line(
            "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9994",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815350434159",
        )?;
        let n = 145; // one day
        for gc in [GravConst::WGS72, GravConst::WGS84] {
            let (states, times) = sgp4_states_gcrf(&tle, n, gc);
            let (fit, result) = if gc == GravConst::WGS72 {
                TLE::fit_from_states(&states, &times, tle.epoch)?
            } else {
                TLE::fit_from_states_full(&states, &times, tle.epoch, gc, OpsMode::AFSPC)?
            };
            let rms = result.best_norm / ((3 * n) as f64).sqrt();
            assert!(rms < 0.01, "{gc:?}: fit RMS {rms} m");
            assert!((fit.inclination - tle.inclination).abs() < 1e-6, "{gc:?}");
            assert!((fit.mean_motion - tle.mean_motion).abs() < 1e-9, "{gc:?}");
        }
        Ok(())
    }

    /// Near-equatorial geostationary element sets (GOES-19, and the same
    /// orbit at other near-equatorial and retrograde inclinations). With
    /// classical elements the fit collapsed the inclination to ~1e-10° with
    /// the node stuck at its two-body seed, and came back ~3 km off.
    #[test]
    fn test_fit_recovers_equatorial_geo() -> Result<()> {
        let line1 = "1 60133U 24119A   26306.21003205 -.00000256  00000-0  00000+0 0  9993";
        let base = TLE::load_2line(
            line1,
            "2 60133   0.0418  80.4472 0000999 203.8933 117.6064  1.00271484  8345",
        )?;
        // At exactly i = 0 SGP4 still depends on the (meaningless) node: its
        // lunisolar inclination periodic tilts the plane about it, ~80 m per
        // degree of Ω. In equinoctial elements that element set is only
        // reached as the limit i → 0 along Ω, so the fit converges at ~8 cm
        // (and ran out of iterations at ~1 m before the finite-difference
        // step was scaled down with p and q; see `FD_SCALE`).
        for (inc, tol) in [(0.0418, 0.01), (1.0e-3, 0.01), (179.95, 0.01), (0.0, 0.5)] {
            let tle = TLE {
                inclination: inc,
                ..base.clone()
            };
            let n = 145; // one day
            let (states, times) = sgp4_states_gcrf(&tle, n, GravConst::WGS72);
            let (fit, result) = TLE::fit_from_states(&states, &times, tle.epoch)?;
            let rms = result.best_norm / ((3 * n) as f64).sqrt();
            assert!(rms < tol, "i = {inc}: fit RMS {rms} m");
            assert!(
                matches!(
                    result.status,
                    TleFitStatus::GradientConverged
                        | TleFitStatus::StepConverged
                        | TleFitStatus::CostConverged
                ),
                "i = {inc}: {} after {} iterations",
                result.status,
                result.n_iter
            );
            assert!((fit.inclination - inc).abs() < 1e-4, "i = {inc}: {fit}");
            assert!(
                (fit.mean_motion - tle.mean_motion).abs() < 1e-9,
                "i = {inc}"
            );
        }
        // A geostationary arc from the high-fidelity propagator.
        let (tle, _result, rms) = fit_hifi_arc(crate::consts::GEO_R, 0.05, 0.0, 24.0, None)?;
        assert!(rms < 1500.0, "GEO fit RMS {rms} m too large: {tle}");
        Ok(())
    }
}
