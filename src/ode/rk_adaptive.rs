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
        if sol.dense.is_none() {
            return ODEError::NoDenseOutputInSolution.into();
        }
        let dense = sol.dense.as_ref().unwrap();

        // These could probably be combined into a single function, but...
        // keeping forward and backward separate makes it simpler in my mind
        if sol.x > dense.x[0] {
            Self::interpolate_forward(xinterp, sol)
        } else {
            Self::interpolate_backward(xinterp, sol)
        }
    }

    /// Interpolate densely calculated solution onto
    /// values that are evenly spaced in "x"
    /// for forward direction
    fn interpolate_forward<S: ODEState>(xinterp: f64, sol: &ODESolution<S>) -> ODEResult<S> {
        if sol.dense.is_none() {
            return ODEError::NoDenseOutputInSolution.into();
        }
        let dense = sol.dense.as_ref().unwrap();

        // Check if interpolation point is within bounds
        if sol.x < xinterp {
            return ODEError::InterpExceedsSolutionBounds {
                interp: xinterp,
                start: dense.x[0],
                stop: sol.x,
            }
            .into();
        }
        if xinterp < dense.x[0] {
            return ODEError::InterpExceedsSolutionBounds {
                interp: xinterp,
                start: dense.x[0],
                stop: sol.x,
            }
            .into();
        }

        // We know indices are monotonically increasing, so only search from
        // last found position in the array forward
        let mut idx = dense
            .x
            .iter()
            .position(|x| *x >= xinterp)
            .map_or(dense.x.len(), |v| v);
        idx = idx.saturating_sub(1);

        // t is fractional distance beween x at idx and idx+1
        // and is in range [0,1]
        let t = (xinterp - dense.x[idx]) / dense.h[idx];

        // Compute interpolant coefficient as funciton of t
        // note that t is in range [0,1]
        //
        // This is equation (6) of
        // https://link.springer.com/article/10.1023/A:1021190918665
        //
        // Note: equation (6) of paper incorrectly has sum index "j"
        //       starting from 0.  It should start from 1.
        //
        let bi: Vec<f64> = Self::BI
            .iter()
            .map(|biarr| {
                // Coefficients multiply increasing powers of t
                let mut tj = 1.0;
                biarr.iter().fold(0.0, |acc, bij| {
                    tj *= t;
                    acc + bij * tj
                })
            })
            .collect();

        //
        // Compute interpolated value
        //
        // This is equation(5) of:
        // https://link.springer.com/article/10.1023/A:1021190918665
        //
        let mut y = dense.yprime[idx]
            .iter()
            .enumerate()
            .fold(dense.y[idx].clone() / dense.h[idx], |acc, (ix, k)| {
                acc + k.clone() * bi[ix]
            });
        y = y * dense.h[idx];
        Ok(y)
    }

    /// Interpolate densely calculated solution onto
    /// values that are evenly spaced in "x"
    /// for backward direction
    fn interpolate_backward<S: ODEState>(xinterp: f64, sol: &ODESolution<S>) -> ODEResult<S> {
        if sol.dense.is_none() {
            return ODEError::NoDenseOutputInSolution.into();
        }
        let dense = sol.dense.as_ref().unwrap();

        // Check if interpolation point is within bounds
        if sol.x > xinterp {
            return ODEError::InterpExceedsSolutionBounds {
                interp: xinterp,
                start: dense.x[0],
                stop: sol.x,
            }
            .into();
        }
        if xinterp > dense.x[0] {
            return ODEError::InterpExceedsSolutionBounds {
                interp: xinterp,
                start: dense.x[0],
                stop: sol.x,
            }
            .into();
        }

        // We know indices are monotonically increasing, so only search from
        // last found position in the array forward
        let mut idx = dense
            .x
            .iter()
            .position(|x| *x <= xinterp)
            .map_or(dense.x.len(), |v| v);
        idx = idx.saturating_sub(1);

        // t is fractional distance beween x at idx and idx+1
        // and is in range [0,1]
        let t = (xinterp - dense.x[idx]) / dense.h[idx];

        // Compute interpolant coefficient as funciton of t
        // note that t is in range [0,1]
        //
        // This is equation (6) of
        // https://link.springer.com/article/10.1023/A:1021190918665
        //
        // Note: equation (6) of paper incorrectly has sum index "j"
        //       starting from 0.  It should start from 1.
        //
        let bi: Vec<f64> = Self::BI
            .iter()
            .map(|biarr| {
                // Coefficients multiply increasing powers of t
                let mut tj = 1.0;
                biarr.iter().fold(0.0, |acc, bij| {
                    tj *= t;
                    acc + bij * tj
                })
            })
            .collect();

        //
        // Compute interpolated value
        //
        // This is equation(5) of:
        // https://link.springer.com/article/10.1023/A:1021190918665
        //
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
        start: f64,
        stop: f64,
        y0: &S,
        ydot: impl Fn(f64, &S) -> ODEResult<S>,
        settings: &RKAdaptiveSettings,
    ) -> ODEResult<ODESolution<S>> {
        let mut nevals: usize = 0;
        let mut naccept: usize = 0;
        let mut nreject: usize = 0;
        let mut x = start;
        let mut y = y0.clone();

        let mut qold: f64 = 1.0e-4;
        let tdir = match stop > start {
            true => 1.0,
            false => -1.0,
        };

        // Take guess at initial stepsize
        let mut h = {
            // Adapted from OrdinaryDiffEq.jl
            let sci = (y0.ode_abs() * settings.relerror).ode_scalar_add(settings.abserror);

            let d0 = y0.ode_elem_div(&sci).ode_scaled_norm();
            let ydot0 = ydot(start, y0)?;
            let d1 = ydot0.ode_elem_div(&sci).ode_scaled_norm();
            let h0 = 0.01 * d0 / d1 * tdir;
            let y1 = y0.clone() + ydot0.clone() * h0;
            let ydot1 = ydot(start + h0, &y1)?;
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

        // OK ... lets integrate!
        loop {
            if (tdir > 0.0 && (x + h) >= stop) || (tdir < 0.0 && (x + h) <= stop) {
                h = stop - x;
            }
            let mut karr = Vec::with_capacity(N);
            karr.push(ydot(x, &y)?);

            // Create the "k"s
            for k in 1..N {
                karr.push(ydot(
                    h.mul_add(Self::C[k], x),
                    &(karr.iter().enumerate().fold(y.clone(), |acc, (idx, ki)| {
                        acc + ki.clone() * Self::A[k][idx] * h
                    })),
                )?);
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
            nevals += N;

            if !enorm.is_finite() {
                return ODEError::StepErrorToSmall.into();
            }

            // Run proportional-integral controller on error
            // references Julia's OrdinaryDiffEq.jl
            let beta1 = 7.0 / (5.0 * Self::ORDER as f64);
            let beta2 = 2.0 / (5.0 * Self::ORDER as f64);
            let q11 = enorm.powf(beta1);
            let q = {
                let q = q11 / qold.powf(beta2);
                f64::max(
                    1.0 / settings.maxfac,
                    f64::min(1.0 / settings.minfac, q / settings.gamma),
                )
            };

            if (enorm < 1.0) || (h.abs() <= settings.dtmin) {
                // If dense output requested, record dense output
                if settings.dense_output {
                    let astep = accepted_steps.as_mut().unwrap();
                    astep.x.push(x);
                    astep.h.push(h);
                    astep.yprime.push(karr);
                    astep.y.push(y.clone());
                }

                // Adjust step size
                qold = f64::max(enorm, 1.0e-4);
                x += h;
                y = ynp1;
                h /= q;

                naccept += 1;
                if (tdir > 0.0 && x >= stop) || (tdir < 0.0 && x <= stop) {
                    break;
                }
            } else {
                nreject += 1;
                h /= f64::min(1.0 / settings.minfac, q11 / settings.gamma);
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
