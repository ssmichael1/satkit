use super::types::*;
use super::RKAdaptiveSettings;

pub trait RKAdaptive<const N: usize, const NI: usize> {
    // Butcher Tableau Coefficients
    const A: [[f64; N]; N];
    const C: [f64; N];
    const B: [f64; N];
    const BERR: [f64; N];

    // Interpolation coefficients
    const BI: [[f64; NI]; N];

    // order
    const ORDER: usize;

    /// First Same as Last
    /// (first compute of next iteration is same as last compute of last iteration)
    const FSAL: bool;

    fn interpolate<S: ODEState>(xinterp: f64, sol: &ODESolution<S>) -> ODEResult<S> {
        let dense = match sol.dense.as_ref() {
            Some(d) if !d.x.is_empty() => d,
            _ => return ODEError::NoDenseOutputInSolution.into(),
        };

        let forward = sol.x > dense.x[0];

        // Bounds check
        let (lo, hi) = if forward {
            (dense.x[0], sol.x)
        } else {
            (sol.x, dense.x[0])
        };
        if xinterp < lo || xinterp > hi {
            return ODEError::InterpExceedsSolutionBounds {
                interp: xinterp,
                begin: dense.x[0],
                end: sol.x,
            }
            .into();
        }

        // Find the step containing xinterp
        let idx = if forward {
            dense.x.iter().position(|&x| x >= xinterp)
        } else {
            dense.x.iter().position(|&x| x <= xinterp)
        }
        .unwrap_or(dense.x.len())
        .saturating_sub(1);

        // t is fractional distance within the step, in range [0,1]
        let t = (xinterp - dense.x[idx]) / dense.h[idx];

        // Compute interpolant coefficients bi[i] = sum_j(BI[i][j] * t^(j+1))
        // Equation (6) of Verner 2010
        let bi: Vec<f64> = Self::BI
            .iter()
            .map(|biarr| {
                let mut tj = 1.0;
                biarr.iter().fold(0.0, |acc, bij| {
                    tj *= t;
                    acc + bij * tj
                })
            })
            .collect();

        // Compute interpolated value
        // Equation (5) of Verner 2010: y_interp = (y/h + sum(k[i] * bi[i])) * h
        let mut y = dense.yprime[idx]
            .iter()
            .enumerate()
            .fold(dense.y[idx].clone() / dense.h[idx], |acc, (ix, k)| {
                acc + k.clone() * bi[ix]
            });
        y = y * dense.h[idx];
        Ok(y)
    }

    fn integrate<S: ODEState>(
        begin: f64,
        end: f64,
        y0: &S,
        ydot: impl Fn(f64, &S) -> ODEResult<S>,
        settings: &RKAdaptiveSettings,
    ) -> ODEResult<ODESolution<S>> {
        let mut nevals: usize = 0;
        let mut naccept: usize = 0;
        let mut nreject: usize = 0;
        let mut x = begin;
        let mut y = y0.clone();

        // PID controller state: two previous error norms (Söderlind & Wang 2006)
        let mut enorm_prev: f64 = 1.0e-4;
        let mut enorm_prev2: f64 = 1.0e-4;
        let tdir = match end > begin {
            true => 1.0,
            false => -1.0,
        };

        // Take guess at initial stepsize
        let mut h = {
            // Adapted from OrdinaryDiffEq.jl
            let sci = (y0.ode_abs() * settings.relerror).ode_scalar_add(settings.abserror);

            let d0 = y0.ode_elem_div(&sci).ode_scaled_norm();
            let ydot0 = ydot(begin, y0)?;
            let d1 = ydot0.ode_elem_div(&sci).ode_scaled_norm();
            let h0 = 0.01 * d0 / d1 * tdir;
            let y1 = y0.clone() + ydot0.clone() * h0;
            let ydot1 = ydot(begin + h0, &y1)?;
            let d2 = (ydot1 - ydot0).ode_elem_div(&sci).ode_scaled_norm() / h0;
            let dmax = f64::max(d1, d2);
            let h1: f64 = match dmax < 1e-15 {
                false => 10.0_f64.powf(-(2.0 + dmax.log10()) / (Self::ORDER as f64)),
                true => f64::max(1e-6, h0.abs() * 1e-3),
            };
            nevals += 2;
            f64::min(100.0 * h0.abs(), h1.abs()) * tdir
        };
        let mut accepted_steps: Option<DenseOutput<S>> = match settings.dense_output {
            false => None,
            true => Some(DenseOutput {
                x: Vec::new(),
                h: Vec::new(),
                yprime: Vec::new(),
                y: Vec::new(),
            }),
        };

        // For FSAL methods, cache the last k evaluation
        let mut k_last: Option<S> = None;

        // OK ... lets integrate!
        loop {
            if (tdir > 0.0 && (x + h) >= end) || (tdir < 0.0 && (x + h) <= end) {
                h = end - x;
            }
            let mut karr = Vec::with_capacity(N);

            // Use FSAL optimization: reuse last stage from previous step as first stage
            if Self::FSAL && k_last.is_some() {
                karr.push(k_last.take().unwrap());
            } else {
                karr.push(ydot(x, &y)?);
                nevals += 1;
            }

            // Create the remaining "k"s
            for k in 1..N {
                karr.push(ydot(
                    h.mul_add(Self::C[k], x),
                    &(karr.iter().enumerate().fold(y.clone(), |acc, (idx, ki)| {
                        acc + ki.clone() * Self::A[k][idx] * h
                    })),
                )?);
                nevals += 1;
            }

            // Sum the "k"s
            let ynp1 = karr
                .iter()
                .enumerate()
                .fold(y.clone() * 1.0 / h, |acc, (idx, k)| {
                    acc + k.clone() * Self::B[idx]
                })
                * h;

            // Compute the "error" state by differencing the p and p* orders
            let yerr = karr
                .iter()
                .enumerate()
                .fold(S::ode_zero(), |acc, (idx, k)| {
                    if Self::BERR[idx].abs() > 1.0e-9 {
                        acc + k.clone() * Self::BERR[idx]
                    } else {
                        acc
                    }
                })
                * h;

            // Compute normalized error
            let enorm = {
                let mut ymax = y.ode_abs().ode_elem_max(&ynp1.ode_abs()) * settings.relerror;
                ymax = ymax.ode_scalar_add(settings.abserror);
                let ydiv = yerr.ode_elem_div(&ymax);
                ydiv.ode_scaled_norm()
            };

            if !enorm.is_finite() {
                return ODEError::StepErrorToSmall.into();
            }

            // PID step-size controller (Söderlind & Wang 2006, §4)
            //
            // The step-size ratio is: h_{n+1}/h_n = 1/q, where
            //   q = (e_n)^β₁ · (e_{n-1})^β₂ · (e_{n-2})^β₃ / safety
            //
            // β₁ = 0.7/p, β₂ = -0.4/p, β₃ = 0.1/p
            // (note β₂ is negative, implemented by dividing by enorm_prev^0.4/p)
            let order_f = Self::ORDER as f64;
            let beta1 = 0.7 / order_f;
            let beta2 = 0.4 / order_f;
            let beta3 = 0.1 / order_f;

            if (enorm < 1.0) || (h.abs() <= settings.dtmin) {
                // PID controller for accepted steps
                let q = {
                    let raw = enorm.powf(beta1)
                        / enorm_prev.powf(beta2)
                        * enorm_prev2.powf(beta3)
                        / settings.gamma;
                    raw.clamp(1.0 / settings.maxfac, 1.0 / settings.minfac)
                };

                // If dense output requested, record dense output
                if settings.dense_output {
                    let astep = accepted_steps.as_mut().unwrap();
                    astep.x.push(x);
                    astep.h.push(h);
                    astep.yprime.push(karr.clone());
                    astep.y.push(y.clone());
                }

                // For FSAL methods, save the last k for next iteration
                if Self::FSAL {
                    k_last = Some(karr[N - 1].clone());
                }

                // Update PID history (floor at 1e-4 to avoid division artifacts)
                enorm_prev2 = enorm_prev;
                enorm_prev = f64::max(enorm, 1.0e-4);
                x += h;
                y = ynp1;
                h /= q;

                naccept += 1;
                if (tdir > 0.0 && x >= end) || (tdir < 0.0 && x <= end) {
                    break;
                }
            } else {
                // Step rejected — use P-only controller (more conservative)
                if Self::FSAL {
                    k_last = None;
                }
                nreject += 1;
                let reject_q = enorm.powf(beta1) / settings.gamma;
                h /= reject_q.min(1.0 / settings.minfac);
            }
        }

        Ok(ODESolution {
            nevals,
            naccept,
            nreject,
            x,
            y,
            dense: accepted_steps,
        })
    }
}
