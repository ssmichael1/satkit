use super::rk_adaptive_settings::RKAdaptiveSettings;
use super::types::*;

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

    /// Interpolate densely calculated solution onto
    /// values that are evenly spaced in "x"
    ///
    fn interpolate<S: ODEState>(
        sol: &ODESolution<S>,
        xstart: f64,
        xend: f64,
        dx: f64,
    ) -> ODEResult<ODEInterp<S>> {
        if sol.dense.is_none() {
            return Err(Box::new(ODEError::NoDenseOutputInSolution));
        }
        if sol.x > xend {
            return Err(Box::new(ODEError::InterpExceedsSolutionBounds));
        }
        let dense = sol.dense.as_ref().unwrap();
        if sol.x < dense.x[0] {
            return Err(Box::new(ODEError::InterpExceedsSolutionBounds));
        }
        let n = ((xend - xstart) / dx) as usize + 1;
        let mut xarr: Vec<f64> = (0..n).map(|v| v as f64 * dx + xstart).collect();
        if *xarr.last().unwrap() > xend {
            xarr.pop();
            xarr.push(xend);
        }

        let mut lastidx: usize = 0;

        let yarr: Vec<S> = xarr
            .iter()
            .map(|v| {
                // We know indices are monotonically increasing, so only search from
                // last found position in the array forward
                let mut idx = match dense.x[lastidx..].iter().position(|x| *x >= *v) {
                    Some(v) => v + lastidx,
                    None => dense.x.len(),
                };
                lastidx = idx;
                if idx > 0 {
                    idx -= 1;
                }

                // t is fractional distance beween x at idx and idx+1
                // and is in range [0,1]
                let t = (*v - dense.x[idx]) / dense.h[idx];

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
                            tj = tj * t;
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
                let yarr = dense.yprime[idx]
                    .iter()
                    .enumerate()
                    .fold(dense.y[idx].clone() / dense.h[idx], |acc, (ix, k)| {
                        acc + k.clone() * bi[ix]
                    });
                yarr * dense.h[idx]
            })
            .collect();

        Ok(ODEInterp::<S> { x: xarr, y: yarr })
    }

    /// Convenience function to perform ODE integration
    /// and interpolate from start to finish of integration
    /// at fixed intervals
    fn integrate_dense<S: ODESystem>(
        x0: f64,
        x_end: f64,
        dx: f64,
        y0: &S::Output,
        system: &mut S,
        settings: &RKAdaptiveSettings,
    ) -> ODEResult<(ODESolution<S::Output>, ODEInterp<S::Output>)> {
        // Make sure dense output is enabled
        let res = match settings.dense_output {
            true => Self::integrate(x0, x_end, y0, system, settings)?,
            false => {
                let mut sc = (*settings).clone();
                sc.dense_output = true;
                Self::integrate(x0, x_end, y0, system, &sc)?
            }
        };
        // Interpolate the result
        let interp = Self::interpolate(&res, x0, x_end, dx)?;
        Ok((res, interp))
    }

    ///
    /// Runga-Kutta integration
    /// with Proportional-Integral controller
    fn integrate<S: ODESystem>(
        xstart: f64,
        xend: f64,
        y0: &S::Output,
        system: &mut S,
        settings: &RKAdaptiveSettings,
    ) -> ODEResult<ODESolution<S::Output>> {
        let mut nevals: usize = 0;
        let mut naccept: usize = 0;
        let mut nreject: usize = 0;
        let mut x = xstart.clone();
        let mut y = y0.clone();

        let mut qold: f64 = 1.0e-4;
        let tdir = match xend > xstart {
            true => 1.0,
            false => -1.0,
        };

        // Take guess at initial stepsize
        let mut h = {
            // Adapted from OrdinaryDiffEq.jl
            let sci = (y0.ode_abs() * settings.relerror).ode_scalar_add(settings.abserror);

            let d0 = y0.ode_elem_div(&sci).ode_norm();
            let ydot0 = system.ydot(xstart.clone(), &y0)?;
            let d1 = ydot0.ode_elem_div(&sci).ode_norm();
            let h0 = 0.01 * d0 / d1 * tdir;
            let y1 = y0.clone() + ydot0.clone() * h0;
            let ydot1 = system.ydot(xstart + h0, &y1)?;
            let d2 = (ydot1 - ydot0).ode_elem_div(&sci).ode_norm() / h0;
            let dmax = f64::max(d1, d2);
            let h1: f64 = match dmax < 1e-15 {
                false => (10.0 as f64).powf(-(2.0 + dmax.log10()) / (Self::ORDER as f64)),
                true => f64::max(1e-6, h0.abs() * 1e-3),
            };
            nevals += 2;
            f64::min(100.0 * h0.abs(), h1.abs()) * tdir
        };
        let mut accepted_steps: Option<DenseOutput<S::Output>> = match settings.dense_output {
            false => None,
            true => Some(DenseOutput {
                x: Vec::new(),
                h: Vec::new(),
                yprime: Vec::new(),
                y: Vec::new(),
            }),
        };

        // OK ... lets integrate!
        let mut runloop: bool = true;
        while runloop {
            if (tdir > 0.0 && x + h >= xend) || (tdir < 0.0 && x + h <= xend) {
                h = xend - x;
                runloop = false;
            }
            let mut karr = Vec::new();
            karr.push(system.ydot(x, &y)?);

            // Create the "k"s
            for k in 1..N {
                karr.push(system.ydot(
                    x + h * Self::C[k],
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
                .fold(S::Output::ode_zero(), |acc, (idx, k)| {
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
                ydiv.ode_norm()
            };
            nevals += N;

            if !enorm.is_finite() {
                return Err(Box::new(ODEError::StepErrorToSmall));
            }

            // Run proportional-integral controller on error
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
                match settings.dense_output {
                    true => {
                        let astep = accepted_steps.as_mut().unwrap();
                        astep.x.push(x);
                        astep.h.push(h);
                        astep.yprime.push(karr);
                        astep.y.push(y.clone());
                    }
                    false => {}
                }

                // Adjust step size
                qold = f64::max(enorm, 1.0e-4);
                x += h;
                y = ynp1;
                h = h / q;

                naccept += 1;
                // If dense output, limit step size
            } else {
                nreject += 1;
                h = h / f64::min(1.0 / settings.minfac, q11 / settings.gamma);
            }
            /*
            h = h * f64::min(
                settings.maxfac,
                f64::max(
                    settings.minfac,
                    0.9 * (1.0 / enorm).powf(1.0 / (Self::ORDER + 3) as f64),
                ),
            );
            */
        }

        Ok(ODESolution {
            nevals: nevals,
            naccept: naccept,
            nreject: nreject,
            x: x,
            y: y,
            dense: accepted_steps,
        })
    }
}
