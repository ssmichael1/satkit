//! Rosenbrock (linearly-implicit Runge-Kutta) solver for stiff ODEs.
//!
//! Unlike explicit RK methods, Rosenbrock methods solve linear systems
//! involving the Jacobian at each stage, making them suitable for stiff
//! ODEs without requiring nonlinear Newton iterations.
//!
//! Reference:
//!   E. Hairer & G. Wanner, "Solving Ordinary Differential Equations II" (1996), §IV.7

use super::types::*;
use super::RKAdaptiveSettings;

use nalgebra as na;

/// Trait for Rosenbrock solvers parameterized by number of stages.
///
/// Each step solves:
///   (I/(hγ) − J) k_i = f(t + α_i h, y + Σ a_ij k_j) + Σ (c_ij/h) k_j
///
/// where J = ∂f/∂y is LU-factored once per step.
pub trait Rosenbrock<const STAGES: usize> {
    /// Stage coupling matrix (lower-triangular, zero diagonal).
    const A: [[f64; STAGES]; STAGES];
    /// Off-diagonal Γ coupling (lower-triangular, zero diagonal).
    const C: [[f64; STAGES]; STAGES];
    /// Shared diagonal element of the Γ matrix.
    const GAMMA_DIAG: f64;
    /// Time offsets for each stage.
    const ALPHA: [f64; STAGES];
    /// Solution weights (higher-order).
    const M: [f64; STAGES];
    /// Embedded solution weights (for error estimation).
    const MHAT: [f64; STAGES];
    /// Order of the higher-order method.
    const ORDER: usize;

    /// Integrate from `begin` to `end`.
    ///
    /// `ydot` computes the RHS f(t, y).
    /// `jac` computes the Jacobian ∂f/∂y as a 6x6 matrix (for the orbital state).
    ///
    /// The state type S is typically `SMatrix<f64, 6, 1>` for simple propagation
    /// or `SMatrix<f64, 6, 7>` when computing the state transition matrix.
    /// The Jacobian is always 6x6 since the STM derivative uses the same Jacobian.
    fn integrate<S: ODEState>(
        begin: f64,
        end: f64,
        y0: &S,
        ydot: impl Fn(f64, &S) -> ODEResult<S>,
        jac: impl Fn(f64, &S) -> ODEResult<na::SMatrix<f64, 6, 6>>,
        settings: &RKAdaptiveSettings,
    ) -> ODEResult<ODESolution<S>> {
        let mut nevals: usize = 0;
        let mut naccept: usize = 0;
        let mut nreject: usize = 0;
        let mut x = begin;
        let mut y = y0.clone();

        // PID controller state
        let mut enorm_prev: f64 = 1.0e-4;
        let mut enorm_prev2: f64 = 1.0e-4;

        let tdir: f64 = if end > begin { 1.0 } else { -1.0 };

        // Initial step-size guess (same heuristic as RKAdaptive)
        let mut h = {
            let sci = (y0.ode_abs() * settings.relerror).ode_scalar_add(settings.abserror);
            let d0 = y0.ode_elem_div(&sci).ode_scaled_norm();
            let ydot0 = ydot(begin, y0)?;
            let d1 = ydot0.ode_elem_div(&sci).ode_scaled_norm();
            let h0 = 0.01 * d0 / d1 * tdir;
            let y1 = y0.clone() + ydot0.clone() * h0;
            let ydot1 = ydot(begin + h0, &y1)?;
            let d2 = (ydot1 - ydot0).ode_elem_div(&sci).ode_scaled_norm() / h0;
            let dmax = f64::max(d1, d2);
            nevals += 2;
            let h1: f64 = if dmax < 1e-15 {
                f64::max(1e-6, h0.abs() * 1e-3)
            } else {
                10.0_f64.powf(-(2.0 + dmax.log10()) / (Self::ORDER as f64))
            };
            f64::min(100.0 * h0.abs(), h1.abs()) * tdir
        };

        // PID controller constants
        let order_f = Self::ORDER as f64;
        let beta1 = 0.7 / order_f;
        let beta2 = 0.4 / order_f;
        let beta3 = 0.1 / order_f;

        loop {
            // Clamp step to not overshoot end
            if (tdir > 0.0 && (x + h) >= end) || (tdir < 0.0 && (x + h) <= end) {
                h = end - x;
            }

            let gamma = Self::GAMMA_DIAG;
            let inv_hgamma = 1.0 / (h * gamma);

            // Evaluate f and J at current point
            let fy = ydot(x, &y)?;
            let jac_mat = jac(x, &y)?;
            nevals += 1;

            // Form W = I/(hγ) − J and LU-factorize
            let mut w_mat = -jac_mat;
            for i in 0..6 {
                w_mat[(i, i)] += inv_hgamma;
            }
            let lu = w_mat.lu();

            // Compute stages k_1, ..., k_STAGES
            //
            // Each k_i is a full state (S), but the linear system solve
            // operates on the 6x1 "simple state" columns. For C==7, we
            // solve 7 columns independently through the same LU factorization.
            let nelem = y.ode_nelem();
            let ncols = nelem / 6;
            let mut karr: Vec<S> = Vec::with_capacity(STAGES);

            for i in 0..STAGES {
                // Stage argument: y + Σ a_ij k_j
                let mut y_stage = y.clone();
                for jj in 0..i {
                    let a_ij = Self::A[i][jj];
                    if a_ij.abs() > 1.0e-30 {
                        y_stage.ode_add_scaled(&karr[jj], a_ij);
                    }
                }

                // f at the stage point (reuse fy for stage 0)
                let fi = if i == 0 {
                    fy.clone()
                } else {
                    let ti = x + Self::ALPHA[i] * h;
                    nevals += 1;
                    ydot(ti, &y_stage)?
                };

                // RHS = fi + Σ (c_ij/h) k_j
                let mut rhs = fi;
                if i > 0 {
                    let inv_h = 1.0 / h;
                    for jj in 0..i {
                        let c_ij = Self::C[i][jj];
                        if c_ij.abs() > 1.0e-30 {
                            rhs.ode_add_scaled(&karr[jj], c_ij * inv_h);
                        }
                    }
                }

                // Solve W * k_i = rhs for each column of the state
                // Extract as flat array, solve column-by-column, reassemble
                let mut ki = S::ode_zero();
                // We need to solve the 6x6 system for each of the ncols columns
                // Access via raw slice manipulation since ODEState is generic
                {
                    let rhs_slice = unsafe {
                        std::slice::from_raw_parts(
                            &rhs as *const S as *const f64,
                            nelem,
                        )
                    };
                    let ki_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            &mut ki as *mut S as *mut f64,
                            nelem,
                        )
                    };
                    for col in 0..ncols {
                        let offset = col * 6;
                        let rhs_col = na::Vector6::from_column_slice(
                            &rhs_slice[offset..offset + 6],
                        );
                        let sol = lu.solve(&rhs_col)
                            .ok_or_else(|| ODEError::YDotError(
                                "Singular Jacobian in Rosenbrock solver".to_string(),
                            ))?;
                        ki_slice[offset..offset + 6].copy_from_slice(sol.as_slice());
                    }
                }
                karr.push(ki);
            }

            // Solution: y_{n+1} = y_n + Σ m_i k_i
            let mut ynp1 = y.clone();
            for (idx, ki) in karr.iter().enumerate() {
                let m_idx = Self::M[idx];
                if m_idx.abs() > 1.0e-30 {
                    ynp1.ode_add_scaled(ki, m_idx);
                }
            }

            // Error estimate: err = Σ (m_i − m̂_i) k_i
            let mut yerr = S::ode_zero();
            for (idx, ki) in karr.iter().enumerate() {
                let diff = Self::M[idx] - Self::MHAT[idx];
                if diff.abs() > 1.0e-20 {
                    yerr.ode_add_scaled(ki, diff);
                }
            }

            // Normalized error
            let enorm = {
                let ymax = y.ode_abs().ode_elem_max(&ynp1.ode_abs()) * settings.relerror;
                let ymax = ymax.ode_scalar_add(settings.abserror);
                yerr.ode_elem_div(&ymax).ode_scaled_norm()
            };

            if !enorm.is_finite() {
                return ODEError::StepErrorToSmall.into();
            }

            // PID step-size controller (Söderlind & Wang 2006)
            if (enorm < 1.0) || (h.abs() <= settings.dtmin) {
                let q = {
                    let raw = enorm.powf(beta1)
                        / enorm_prev.powf(beta2)
                        * enorm_prev2.powf(beta3)
                        / settings.gamma;
                    raw.clamp(1.0 / settings.maxfac, 1.0 / settings.minfac)
                };

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
                // Reject step
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
            dense: None, // Rosenbrock dense output not implemented
        })
    }

    /// Interpolation — not supported for Rosenbrock methods
    fn interpolate<S: ODEState>(_xinterp: f64, _sol: &ODESolution<S>) -> ODEResult<S> {
        ODEError::InterpNotImplemented.into()
    }
}
