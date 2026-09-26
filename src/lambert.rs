//! Lambert's problem solver
//!
//! Solves Lambert's problem: given two position vectors and a time of flight,
//! find the orbit(s) connecting them. This is fundamental to orbital targeting,
//! rendezvous planning, and interplanetary trajectory design.
//!
//! Implements Izzo's algorithm (2015) with Householder 4th-order iterations
//! for fast, robust convergence across all geometries including multi-revolution
//! transfers.
//!
//! # References
//!
//! * D. Izzo, "Revisiting Lambert's problem," Celestial Mechanics and
//!   Dynamical Astronomy, vol. 121, pp. 1-15, 2015.
//!
//! # Example
//!
//! ```
//! use satkit::lambert::lambert;
//! use satkit::consts::MU_EARTH;
//!
//! let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
//! let r2 = numeris::vector![0.0, 7000.0e3, 0.0];
//! let tof = 3600.0; // 1 hour
//!
//! let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
//! let (v1, v2) = &solutions[0];
//! ```

use crate::mathtypes::Vector3;

use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Time of flight must be positive, got {0}")]
    InvalidTof(f64),
    #[error("Position vectors must be non-zero")]
    ZeroPosition,
    #[error("Gravitational parameter must be positive, got {0}")]
    InvalidMu(f64),
    #[error("Convergence failure for revolution {0}")]
    ConvergenceFailed(u32),
    /// An input (`r1`, `r2`, `tof` or `mu`) contains a NaN or infinity.
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    /// `r1` and `r2` are the same point (zero chord), so the transfer plane
    /// and angle are undefined.
    #[error("r1 and r2 coincide: the transfer is undefined")]
    CoincidentPositions,
}

/// Result type for Lambert's problem.
pub type Result<T> = std::result::Result<T, Error>;

#[deprecated(note = "use lambert::Error instead")]
pub type LambertError = Error;

/// Result of Lambert's problem: departure and arrival velocity vectors.
pub type LambertSolution = (Vector3, Vector3);

/// Solve Lambert's problem using Izzo's algorithm.
///
/// Given two position vectors and a time of flight, find the initial and final
/// velocity vectors for transfer orbits connecting them.
///
/// # Arguments
///
/// * `r1` - Initial position vector (meters)
/// * `r2` - Final position vector (meters)
/// * `tof` - Time of flight (seconds), must be positive
/// * `mu` - Gravitational parameter (m³/s²)
/// * `prograde` - If true, assume prograde (counterclockwise) transfer;
///   if false, assume retrograde transfer. This resolves the
///   short-way / long-way ambiguity: the transfer's angular momentum has
///   `h_z >= 0` for prograde and `h_z <= 0` for retrograde. For collinear
///   positions (a 180° transfer) the plane is not defined by `r1` and `r2`;
///   the solver picks one containing `r1` and the x (or y) axis, oriented
///   by the same rule.
///
/// # Returns
///
/// Vector of `(v1, v2)` solutions. The first element is the zero-revolution
/// solution. Additional elements are multi-revolution solutions (if any exist
/// for the given time of flight), returned in pairs (short-period, long-period)
/// for each revolution count.
///
/// # Errors
///
/// [`Error::NonFinite`] for a NaN or infinite input, [`Error::InvalidTof`],
/// [`Error::InvalidMu`] and [`Error::ZeroPosition`] for out-of-domain
/// inputs, [`Error::CoincidentPositions`] when `r1 == r2`, and
/// [`Error::ConvergenceFailed`] if the zero-revolution iteration fails.
pub fn lambert(
    r1: &Vector3,
    r2: &Vector3,
    tof: f64,
    mu: f64,
    prograde: bool,
) -> Result<Vec<LambertSolution>> {
    for (name, ok) in [
        ("r1", r1.iter().all(|x| x.is_finite())),
        ("r2", r2.iter().all(|x| x.is_finite())),
        ("tof", tof.is_finite()),
        ("mu", mu.is_finite()),
    ] {
        if !ok {
            return Err(Error::NonFinite(name));
        }
    }
    if tof <= 0.0 {
        return Err(Error::InvalidTof(tof));
    }
    if mu <= 0.0 {
        return Err(Error::InvalidMu(mu));
    }

    let r1_norm = r1.norm();
    let r2_norm = r2.norm();
    if r1_norm < 1.0e-10 || r2_norm < 1.0e-10 {
        return Err(Error::ZeroPosition);
    }

    // Chord and semiperimeter
    let c = (r2 - r1).norm();
    let s = (r1_norm + r2_norm + c) / 2.0;
    // Zero chord: rho = (r1 - r2) / c below is 0/0.
    if c <= f64::EPSILON * r1_norm.max(r2_norm) {
        return Err(Error::CoincidentPositions);
    }

    // Unit vectors
    let ir1 = r1 / r1_norm;
    let ir2 = r2 / r2_norm;
    let ih_raw = ir1.cross(&ir2);
    let ih_norm = ih_raw.norm();

    // Transfer angle, short way. atan2 keeps full precision near 0 and π,
    // where acos of the dot product loses half the digits.
    let mut dtheta = f64::atan2(ih_norm, ir1.dot(&ir2));

    let ih = if ih_norm < 1.0e-12 {
        // Collinear positions (180-degree transfer): r1 and r2 do not fix
        // the plane. Pick one containing r1, then orient its normal by the
        // `prograde` flag directly (h_z >= 0 prograde, <= 0 retrograde);
        // the transfer angle stays at the short-way value.
        let ih = if ir1.x().abs() < 0.9 {
            ir1.cross(&numeris::vector![1.0, 0.0, 0.0]).normalize()
        } else {
            ir1.cross(&numeris::vector![0.0, 1.0, 0.0]).normalize()
        };
        let ih = if ih.z() < 0.0 { -ih } else { ih };
        if prograde {
            ih
        } else {
            -ih
        }
    } else {
        // The short way goes counterclockwise about ih; take the long way
        // when that contradicts the requested direction.
        let ih = ih_raw / ih_norm;
        if prograde {
            if ih.z() < 0.0 {
                dtheta = 2.0 * PI - dtheta;
            }
        } else if ih.z() >= 0.0 {
            dtheta = 2.0 * PI - dtheta;
        }
        ih
    };

    // Tangent unit vectors (perpendicular to position in orbital plane, in
    // the direction of motion). The long way (dtheta > π) travels clockwise
    // about ih, so the tangents flip with lambda (Izzo 2015, Algorithm 1).
    let (it1, it2) = if dtheta > PI {
        (ir1.cross(&ih), ir2.cross(&ih))
    } else {
        (ih.cross(&ir1), ih.cross(&ir2))
    };

    // Lambda parameter: lambda^2 = 1 - c/s, but the half-angle form keeps
    // precision (and the sign, negative for the long way) near 180°, where
    // 1 - c/s cancels to rounding noise that can even go negative.
    let lambda = (r1_norm * r2_norm).sqrt() / s * (dtheta / 2.0).cos();

    // Non-dimensional time of flight
    let t_norm = tof * (2.0 * mu / s.powi(3)).sqrt();

    // Velocity reconstruction constants
    let gamma = (mu * s / 2.0).sqrt();
    let rho = (r1_norm - r2_norm) / c;
    let sigma = (1.0 - rho * rho).sqrt();

    let mut solutions = Vec::new();

    // --- Zero-revolution solution ---
    let x0 = initial_guess_0rev(lambda, t_norm);
    let x = householder(lambda, t_norm, x0, 0).ok_or(Error::ConvergenceFailed(0))?;
    solutions.push(build_velocity(
        &ir1, &ir2, &it1, &it2, r1_norm, r2_norm, lambda, gamma, rho, sigma, x,
    ));

    // --- Multi-revolution solutions ---
    let max_revs = (t_norm / PI).floor().max(0.0) as u32;

    for m in 1..=max_revs {
        let (_x_min, t_min_m) = compute_t_min(lambda, m);

        if t_norm < t_min_m {
            break;
        }

        // Left (short-period) solution: x > x_min, toward +1
        let x_l = initial_guess_mrev(t_norm, m, true);
        if let Some(x) = householder(lambda, t_norm, x_l, m) {
            solutions.push(build_velocity(
                &ir1, &ir2, &it1, &it2, r1_norm, r2_norm, lambda, gamma, rho, sigma, x,
            ));
        }

        // Right (long-period) solution: x < x_min, toward -1
        if t_norm > t_min_m + 1.0e-6 {
            let x_r = initial_guess_mrev(t_norm, m, false);
            if let Some(x) = householder(lambda, t_norm, x_r, m) {
                solutions.push(build_velocity(
                    &ir1, &ir2, &it1, &it2, r1_norm, r2_norm, lambda, gamma, rho, sigma, x,
                ));
            }
        }
    }

    Ok(solutions)
}

// ---------------------------------------------------------------------------
// Velocity reconstruction (Izzo, eq. 12-14)
// ---------------------------------------------------------------------------

/// Reconstruct departure and arrival velocities from the solution parameter x.
#[allow(clippy::too_many_arguments)] // matches the Izzo (2015) reference formulation
fn build_velocity(
    ir1: &Vector3,
    ir2: &Vector3,
    it1: &Vector3,
    it2: &Vector3,
    r1_norm: f64,
    r2_norm: f64,
    lambda: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
    x: f64,
) -> LambertSolution {
    let y = compute_y(x, lambda);

    let vr1 = gamma * ((lambda * y - x) - rho * (lambda * y + x)) / r1_norm;
    let vr2 = -gamma * ((lambda * y - x) + rho * (lambda * y + x)) / r2_norm;
    let vt = gamma * sigma * (y + lambda * x);
    let vt1 = vt / r1_norm;
    let vt2 = vt / r2_norm;

    let v1 = vr1 * ir1 + vt1 * it1;
    let v2 = vr2 * ir2 + vt2 * it2;

    (v1, v2)
}

// ---------------------------------------------------------------------------
// TOF equation (Izzo 2015, eq. 17)
// ---------------------------------------------------------------------------

/// y(x, lambda) = sqrt(1 - lambda^2 * (1 - x^2))
#[inline]
fn compute_y(x: f64, lambda: f64) -> f64 {
    (1.0 - lambda * lambda * (1.0 - x * x)).max(0.0).sqrt()
}

/// Battin's hypergeometric series 2F1(3, 1; 5/2; x)
fn hyp2f1b(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        return 1.0;
    }
    let mut res = 1.0;
    let mut term = 1.0;
    for i in 0..100 {
        let n = i as f64;
        term *= (3.0 + n) * (1.0 + n) / ((2.5 + n) * (n + 1.0)) * x;
        res += term;
        if term.abs() < 1.0e-15 {
            break;
        }
    }
    res
}

/// Compute non-dimensional TOF as a function of x, lambda, and revolution count M.
///
/// T(x) = [(psi + M*pi)/sqrt(|1-x^2|) - x + lambda*y] / (1 - x^2)
fn tof_equation(x: f64, lambda: f64, m: u32) -> f64 {
    let omx2 = 1.0 - x * x;
    let y = compute_y(x, lambda);

    // Near-parabolic: use Battin's series (avoids 0/0 at x=1)
    if m == 0 && (0.6_f64).sqrt() < x && x < (1.4_f64).sqrt() {
        let eta = y - lambda * x;
        let s1 = (1.0 - lambda - x * eta) * 0.5;
        let q = 4.0 / 3.0 * hyp2f1b(s1);
        return (eta.powi(3) * q + 4.0 * lambda * eta) / 2.0;
    }

    if omx2.abs() < 1.0e-14 {
        // Parabolic limit
        return 2.0 / 3.0 * (1.0 - lambda.powi(3));
    }

    if x < 1.0 {
        // Elliptic
        let cos_psi = x * y + lambda * omx2;
        let psi = f64::acos(cos_psi.clamp(-1.0, 1.0));
        ((psi + (m as f64) * PI) / omx2.sqrt() - x + lambda * y) / omx2
    } else {
        // Hyperbolic
        let cosh_psi = x * y - lambda * (x * x - 1.0);
        let psi_h = cosh_psi.max(1.0).acosh();
        (-x + lambda * y + psi_h / (x * x - 1.0).sqrt()) / omx2
    }
}

// ---------------------------------------------------------------------------
// Derivatives (Izzo 2015, recurrence relations)
// ---------------------------------------------------------------------------

/// First three derivatives of T(x) for Householder iteration.
fn tof_derivatives(x: f64, lambda: f64, t: f64) -> (f64, f64, f64) {
    let lambda2 = lambda * lambda;
    let lambda3 = lambda2 * lambda;
    let lambda5 = lambda2 * lambda3;
    let omx2 = 1.0 - x * x;
    let y = compute_y(x, lambda);

    if omx2.abs() < 1.0e-12 || y < 1.0e-14 {
        return (0.0, 0.0, 0.0);
    }

    let dt = (3.0 * t * x - 2.0 + 2.0 * lambda3 * x / y) / omx2;
    let d2t = (3.0 * t + 5.0 * x * dt + 2.0 * (1.0 - lambda2) * lambda3 / y.powi(3)) / omx2;
    let d3t = (7.0 * x * d2t + 8.0 * dt - 6.0 * (1.0 - lambda2) * lambda5 * x / y.powi(5)) / omx2;

    (dt, d2t, d3t)
}

// ---------------------------------------------------------------------------
// Root finding
// ---------------------------------------------------------------------------

/// Householder 4th-order iteration to solve T(x) = T_target.
fn householder(lambda: f64, t_target: f64, x0: f64, m: u32) -> Option<f64> {
    let mut x = x0;

    for _ in 0..35 {
        let t = tof_equation(x, lambda, m);
        let delta = t - t_target;

        if delta.abs() < 1.0e-12 {
            return Some(x);
        }

        let (dt, d2t, d3t) = tof_derivatives(x, lambda, t);
        if dt.abs() < 1.0e-15 {
            return None;
        }

        // Householder step (Izzo eq. 20)
        let dt2 = dt * dt;
        let step = delta * (dt2 - delta * d2t / 2.0)
            / (dt * (dt2 - delta * d2t) + d3t * delta * delta / 6.0);

        x -= step;
        // x < -1 is unphysical (elliptic energy limit) for all cases. The
        // upper bound depends on revolutions: multi-rev (m > 0) solutions are
        // strictly elliptic (x < 1), but the zero-rev case admits hyperbolic
        // transfers (x > 1) for short times of flight, so only the lower bound
        // is clamped there. Clamping the upper bound unconditionally made every
        // hyperbolic solution fail to converge.
        if m == 0 {
            x = x.max(-1.0 + 1.0e-9);
        } else {
            x = x.clamp(-1.0 + 1.0e-9, 1.0 - 1.0e-9);
        }
    }

    let t_final = tof_equation(x, lambda, m);
    if (t_final - t_target).abs() < 1.0e-8 {
        Some(x)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Initial guesses (Izzo 2015, Section 3)
// ---------------------------------------------------------------------------

/// Zero-revolution initial guess.
fn initial_guess_0rev(lambda: f64, t: f64) -> f64 {
    // T at x=0: parabolic boundary
    let t00 = f64::acos(lambda) + lambda * (1.0 - lambda * lambda).sqrt();
    // T at x=1: limit
    let t1 = 2.0 / 3.0 * (1.0 - lambda.powi(3));

    if t >= t00 {
        // Long-TOF: x in [-1, 0], elliptic with large semi-major axis
        -(t - t00) / (t - t00 + 4.0)
    } else if t <= t1 {
        // Short-TOF: x > 1, hyperbolic
        t1 * (t1 - t) / (0.4 * (1.0 - lambda.powi(5)) * t) + 1.0
    } else {
        // Intermediate
        (t / t00).powf(f64::ln(2.0) / f64::ln(t1 / t00)) - 1.0
    }
}

/// Multi-revolution initial guess.
fn initial_guess_mrev(t: f64, m: u32, left: bool) -> f64 {
    let m_pi = (m as f64) * PI;
    if left {
        let t_ratio = ((m_pi + PI) / (8.0 * t)).powf(2.0 / 3.0);
        (t_ratio - 1.0) / (t_ratio + 1.0)
    } else {
        let t_ratio = ((8.0 * t) / m_pi).powf(2.0 / 3.0);
        (t_ratio - 1.0) / (t_ratio + 1.0)
    }
}

/// Compute minimum T for m revolutions. Returns (x_min, T_min).
fn compute_t_min(lambda: f64, m: u32) -> (f64, f64) {
    // Initial guess for x at dT/dx = 0
    let mut x = 0.0;

    // Halley iteration on dT/dx = 0
    for _ in 0..50 {
        let t = tof_equation(x, lambda, m);
        let (dt, d2t, _) = tof_derivatives(x, lambda, t);

        if dt.abs() < 1.0e-14 {
            break;
        }

        if d2t.abs() < 1.0e-15 {
            break;
        }

        // Newton step on dT/dx = 0: x -= dT/d2T
        x -= dt / d2t;
        x = x.clamp(-0.999, 0.999);
    }

    (x, tof_equation(x, lambda, m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::MU_EARTH;

    /// Stumpff functions C(z), S(z).
    fn stumpff(z: f64) -> (f64, f64) {
        if z > 1.0e-8 {
            let sz = z.sqrt();
            ((1.0 - sz.cos()) / z, (sz - sz.sin()) / (sz * z))
        } else if z < -1.0e-8 {
            let sz = (-z).sqrt();
            ((sz.cosh() - 1.0) / (-z), (sz.sinh() - sz) / (sz * (-z)))
        } else {
            (0.5 - z / 24.0, 1.0 / 6.0 - z / 120.0)
        }
    }

    /// Two-body propagation with universal variables (Curtis, Algorithms
    /// 3.3 and 3.4). Independent of the solver and of `Kepler`, and valid
    /// for elliptic, parabolic and hyperbolic orbits in any plane.
    fn propagate_uv(r0: &Vector3, v0: &Vector3, dt: f64, mu: f64) -> (Vector3, Vector3) {
        let r0n = r0.norm();
        let vr0 = r0.dot(v0) / r0n;
        let alpha = 2.0 / r0n - v0.norm_squared() / mu;
        let smu = mu.sqrt();
        // Universal Kepler equation F(chi) = 0; F is increasing (F' = r > 0)
        // with F(0) < 0 for dt > 0, so Newton is safeguarded by bisection.
        let kepler_uv = |chi: f64| {
            let (c, s) = stumpff(alpha * chi * chi);
            let f =
                r0n * vr0 / smu * chi * chi * c + (1.0 - alpha * r0n) * chi.powi(3) * s + r0n * chi
                    - smu * dt;
            let fp = r0n * vr0 / smu * chi * (1.0 - alpha * chi * chi * s)
                + (1.0 - alpha * r0n) * chi * chi * c
                + r0n;
            (f, fp)
        };
        let (mut lo, mut hi) = (0.0, smu * dt / r0n);
        while kepler_uv(hi).0 < 0.0 {
            lo = hi;
            hi *= 2.0;
        }
        let mut chi = 0.5 * (lo + hi);
        for _ in 0..500 {
            let (f, fp) = kepler_uv(chi);
            if f < 0.0 {
                lo = chi;
            } else {
                hi = chi;
            }
            let newton = chi - f / fp;
            let next = if newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let done = (next - chi).abs() <= 1.0e-15 * chi.abs();
            chi = next;
            if done {
                break;
            }
        }
        let (c, s) = stumpff(alpha * chi * chi);
        let f = 1.0 - chi * chi / r0n * c;
        let g = dt - chi.powi(3) / smu * s;
        let r = f * r0 + g * v0;
        let rn = r.norm();
        let fd = smu / (rn * r0n) * (alpha * chi.powi(3) * s - chi);
        let gd = 1.0 - chi * chi / rn * c;
        (r, fd * r0 + gd * v0)
    }

    /// Verify a Lambert solution: propagating (r1, v1) for `tof` must reach
    /// (r2, v2), and the transfer must run in the requested direction.
    fn verify_solution_mu(
        r1: &Vector3,
        r2: &Vector3,
        v1: &Vector3,
        v2: &Vector3,
        tof: f64,
        mu: f64,
        prograde: bool,
    ) {
        let (r2_prop, v2_prop) = propagate_uv(r1, v1, tof, mu);
        let pos_err = (r2_prop - r2).norm();
        let vel_err = (v2_prop - v2).norm();
        assert!(
            pos_err < 1.0e-8 * r2.norm(),
            "Propagation misses r2 by {pos_err:.3e} m (r1={r1:?} r2={r2:?} prograde={prograde})"
        );
        assert!(
            vel_err < 1.0e-8 * v2.norm(),
            "Propagated velocity misses v2 by {vel_err:.3e} m/s"
        );

        // Energy and angular momentum conservation between the endpoints
        let energy1 = v1.norm_squared() / 2.0 - mu / r1.norm();
        let energy2 = v2.norm_squared() / 2.0 - mu / r2.norm();
        let scale = v1.norm_squared() / 2.0 + mu / r1.norm();
        assert!(
            (energy1 - energy2).abs() / scale < 1.0e-10,
            "Energy mismatch: E1={energy1:.6e}, E2={energy2:.6e}"
        );
        let h1 = r1.cross(v1);
        let h2 = r2.cross(v2);
        let h_err = (h1 - h2).norm() / h1.norm();
        assert!(h_err < 1.0e-10, "Angular momentum mismatch: {h_err:.2e}");

        // Direction of motion. A polar plane (h_z = 0) satisfies either.
        let hz = h1.z() / h1.norm();
        if prograde {
            assert!(hz > -1.0e-12, "prograde transfer has h_z = {hz:.3e}");
        } else {
            assert!(hz < 1.0e-12, "retrograde transfer has h_z = {hz:.3e}");
        }
    }

    fn verify_solution(r1: &Vector3, r2: &Vector3, v1: &Vector3, v2: &Vector3, tof: f64) {
        verify_solution_mu(r1, r2, v1, v2, tof, MU_EARTH, true);
    }

    /// Solve and verify every returned solution; returns the solution count.
    fn solve_and_verify(r1: &Vector3, r2: &Vector3, tof: f64, prograde: bool) -> usize {
        let solutions = lambert(r1, r2, tof, MU_EARTH, prograde).unwrap();
        assert!(!solutions.is_empty());
        for (v1, v2) in &solutions {
            verify_solution_mu(r1, r2, v1, v2, tof, MU_EARTH, prograde);
        }
        solutions.len()
    }

    #[test]
    fn test_lambert_90deg_transfer() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 7000.0e3, 0.0];
        let period = 2.0 * PI * (7000.0e3_f64.powi(3) / MU_EARTH).sqrt();
        let tof = period / 4.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_hohmann() {
        let r1_mag: f64 = 7000.0e3;
        let r2_mag: f64 = 10000.0e3;
        let r1 = numeris::vector![r1_mag, 0.0, 0.0];
        let r2 = numeris::vector![-r2_mag, 0.0, 0.0];

        let a_transfer = (r1_mag + r2_mag) / 2.0;
        let tof = PI * (a_transfer.powi(3) / MU_EARTH).sqrt();

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());

        let (v1, v2) = &solutions[0];
        assert!(v1.x().abs() < 10.0, "vr should be ~0: {}", v1.x());
        assert!(v1.y() > 0.0, "vt should be positive");

        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_retrograde() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 7000.0e3, 0.0];
        let period = 2.0 * PI * (7000.0e3_f64.powi(3) / MU_EARTH).sqrt();
        let tof = period * 0.75;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, false).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution_mu(&r1, &r2, v1, v2, tof, MU_EARTH, false);
    }

    #[test]
    fn test_lambert_inclined() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 5000.0e3, 5000.0e3];
        let tof = 3600.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_invalid_inputs() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 7000.0e3, 0.0];

        assert!(lambert(&r1, &r2, -1.0, MU_EARTH, true).is_err());
        assert!(lambert(&r1, &r2, 3600.0, -1.0, true).is_err());

        let zero = numeris::vector![0.0, 0.0, 0.0];
        assert!(lambert(&zero, &r2, 3600.0, MU_EARTH, true).is_err());
    }

    #[test]
    fn test_lambert_symmetry() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 7000.0e3, 0.0];
        let tof = 2000.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        let (v1, v2) = &solutions[0];
        let speed_diff = (v1.norm() - v2.norm()).abs();
        assert!(
            speed_diff < 1.0,
            "Speed difference for equal-radius transfer: {} m/s",
            speed_diff
        );
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_large_transfer_angle() {
        let r1 = numeris::vector![8000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![-7500.0e3, 2000.0e3, 1000.0e3];
        let tof = 5000.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_short_tof() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![6800.0e3, 1000.0e3, 0.0];
        let tof = 200.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    #[test]
    fn test_lambert_gto_to_geo() {
        let r1 = numeris::vector![6678.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 42164.0e3, 0.0];
        let tof = 5.0 * 3600.0;

        let solutions = lambert(&r1, &r2, tof, MU_EARTH, true).unwrap();
        assert!(!solutions.is_empty());
        let (v1, v2) = &solutions[0];
        verify_solution(&r1, &r2, v1, v2, tof);
    }

    /// Circular-orbit period at radius `r`.
    fn period(r: f64) -> f64 {
        2.0 * PI * (r.powi(3) / MU_EARTH).sqrt()
    }

    #[test]
    fn test_lambert_long_way_equatorial() {
        // Transfer angles past 180° in the equatorial plane: prograde to
        // 240° / 270°, and retrograde to 90° (the long way clockwise). These
        // used to leave r1 in the wrong direction and miss r2 by ~2r.
        let r: f64 = 7000.0e3;
        let r1 = numeris::vector![r, 0.0, 0.0];
        for deg in [200.0_f64, 240.0, 270.0, 330.0] {
            let th = deg.to_radians();
            let r2 = numeris::vector![1.3 * r * th.cos(), 1.3 * r * th.sin(), 0.0];
            for frac in [0.5, 0.75, 1.2] {
                solve_and_verify(&r1, &r2, frac * period(r), true);
            }
            // The mirror image, retrograde
            let r2m = numeris::vector![r2.x(), -r2.y(), 0.0];
            solve_and_verify(&r1, &r2m, 0.75 * period(r), false);
        }
        // Retrograde short way (r2 at -90°) and long way (r2 at +90°)
        let r2 = numeris::vector![0.0, r, 0.0];
        solve_and_verify(&r1, &r2, 0.75 * period(r), false);
        let r2 = numeris::vector![0.0, -r, 0.0];
        solve_and_verify(&r1, &r2, 0.25 * period(r), false);
    }

    #[test]
    fn test_lambert_long_way_inclined() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        // h_z of the short way is negative, so prograde goes the long way
        let r2 = numeris::vector![-5000.0e3, -3000.0e3, 4000.0e3];
        for tof in [3000.0, 4000.0, 8000.0] {
            solve_and_verify(&r1, &r2, tof, true);
            solve_and_verify(&r1, &r2, tof, false);
        }
        let r1 = numeris::vector![-2000.0e3, 6500.0e3, 1500.0e3];
        let r2 = numeris::vector![3000.0e3, -9000.0e3, -6000.0e3];
        for tof in [2500.0, 6000.0] {
            solve_and_verify(&r1, &r2, tof, true);
            solve_and_verify(&r1, &r2, tof, false);
        }
    }

    #[test]
    fn test_lambert_near_zero_angle_hyperbolic() {
        // Tiny transfer angle and short time of flight: strongly hyperbolic
        // (x well above sqrt(1.4), outside the Battin series). The
        // hyperbolic time-of-flight branch had the wrong sign on psi and
        // missed r2 by 7 km at 10 s and 500 km at 100 s.
        let r: f64 = 7000.0e3;
        let r1 = numeris::vector![r, 0.0, 0.0];
        for deg in [1.0_f64, 5.0] {
            let th = deg.to_radians();
            let r2 = numeris::vector![1.01 * r * th.cos(), 1.01 * r * th.sin(), 0.0];
            let r2i = numeris::vector![1.01 * r * th.cos(), 0.6 * r * th.sin(), 0.8 * r * th.sin()];
            for tof in [10.0, 30.0, 100.0] {
                solve_and_verify(&r1, &r2, tof, true);
                solve_and_verify(&r1, &r2i, tof, true);
            }
        }
    }

    #[test]
    fn test_lambert_battin_window() {
        // Times of flight near the parabolic one put x in the window
        // sqrt(0.6) < x < sqrt(1.4) where the Battin series is used. Its
        // recurrence had the wrong denominator, so the time of flight was
        // off by up to 100% there (missed r2 or failed to converge).
        let r: f64 = 7000.0e3;
        let r1 = numeris::vector![r, 0.0, 0.0];
        for deg in [30.0_f64, 90.0, 150.0, 210.0, 300.0] {
            let th = deg.to_radians();
            let r2 = numeris::vector![2.0 * r * th.cos(), 2.0 * r * th.sin(), 0.3 * r];
            let r2n = r2.norm();
            let c = (r2 - r1).norm();
            let s = (r + r2n + c) / 2.0;
            let lambda = {
                let l = (1.0 - c / s).sqrt();
                let prograde_short = r1.cross(&r2).z() >= 0.0;
                if prograde_short {
                    l
                } else {
                    -l
                }
            };
            // Parabolic time of flight, T(x = 1) = 2/3 (1 - lambda^3)
            let t_par = 2.0 / 3.0 * (1.0 - lambda.powi(3)) / (2.0 * MU_EARTH / s.powi(3)).sqrt();
            for f in [0.6, 0.8, 0.95, 1.05, 1.3, 1.6] {
                let tof = f * t_par;
                solve_and_verify(&r1, &r2, tof, true);
            }
        }
    }

    #[test]
    fn test_tof_equation_matches_lagrange() {
        // Non-dimensional time of flight against Lagrange's closed form
        // (as in pykep's x2tof2), across the elliptic branch, the Battin
        // window and the hyperbolic branch.
        fn lagrange(x: f64, lambda: f64) -> f64 {
            let a = 1.0 / (1.0 - x * x);
            if a > 0.0 {
                let alfa = 2.0 * x.acos();
                let beta = 2.0 * (lambda * lambda / a).sqrt().asin() * lambda.signum();
                a * a.sqrt() * ((alfa - alfa.sin()) - (beta - beta.sin())) / 2.0
            } else {
                let alfa = 2.0 * x.acosh();
                let beta = 2.0 * (-lambda * lambda / a).sqrt().asinh() * lambda.signum();
                -a * (-a).sqrt() * ((beta - beta.sinh()) - (alfa - alfa.sinh())) / 2.0
            }
        }
        for lambda in [-0.95, -0.5, -0.1, 0.1, 0.5, 0.95] {
            for x in [
                -0.5, 0.0, 0.5, 0.78, 0.9, 0.99, 1.01, 1.1, 1.18, 1.3, 2.0, 5.0,
            ] {
                let t = tof_equation(x, lambda, 0);
                let t_ref = lagrange(x, lambda);
                assert!(
                    (t - t_ref).abs() < 1.0e-10 * t_ref.abs(),
                    "T({x}, {lambda}) = {t}, Lagrange {t_ref}"
                );
            }
        }
    }

    #[test]
    fn test_lambert_multirev_retrograde() {
        let r: f64 = 7000.0e3;
        let r1 = numeris::vector![r, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 1.2 * r, 0.0];
        let r2i = numeris::vector![-3000.0e3, 5000.0e3, 4000.0e3];
        for prograde in [true, false] {
            // 3.3 periods admits up to 3 complete revolutions
            let n = solve_and_verify(&r1, &r2, 3.3 * period(r), prograde);
            assert!(n >= 5, "expected multi-rev solutions, got {n}");
            let n = solve_and_verify(&r1, &r2i, 3.3 * period(r), prograde);
            assert!(n >= 5, "expected multi-rev solutions, got {n}");
        }
    }

    #[test]
    fn test_lambert_curtis_example_5_2() {
        // Curtis, Orbital Mechanics for Engineering Students, Example 5.2
        let mu = 398_600.0e9;
        let r1 = numeris::vector![5000.0e3, 10000.0e3, 2100.0e3];
        let r2 = numeris::vector![-14600.0e3, 2500.0e3, 7000.0e3];
        let tof = 3600.0;
        let solutions = lambert(&r1, &r2, tof, mu, true).unwrap();
        let (v1, v2) = &solutions[0];
        let v1_ref = numeris::vector![-5.9925e3, 1.9254e3, 3.2456e3];
        let v2_ref = numeris::vector![-3.3125e3, -4.1966e3, -0.38529e3];
        assert!((v1 - v1_ref).norm() < 0.5, "v1 = {v1:?}");
        assert!((v2 - v2_ref).norm() < 0.5, "v2 = {v2:?}");
        verify_solution_mu(&r1, &r2, v1, v2, tof, mu, true);
    }

    #[test]
    fn test_lambert_180deg_follows_prograde_flag() {
        // Collinear positions leave the plane free; `prograde` must still
        // pick the direction of motion, whichever axis r1 lies along.
        let r: f64 = 7000.0e3;
        for r1 in [
            numeris::vector![r, 0.0, 0.0],
            numeris::vector![0.0, r, 0.0],
            numeris::vector![-r, 0.0, 0.0],
            numeris::vector![0.0, -r, 0.0],
            numeris::vector![0.6 * r, -0.3 * r, 0.5 * r],
            numeris::vector![-0.2 * r, 0.7 * r, -0.4 * r],
            numeris::vector![0.1 * r, 0.1 * r, 0.9 * r],
        ] {
            let r2 = -1.2 * r1;
            for prograde in [true, false] {
                let solutions = lambert(&r1, &r2, 0.6 * period(r), MU_EARTH, prograde).unwrap();
                let (v1, v2) = &solutions[0];
                verify_solution_mu(&r1, &r2, v1, v2, 0.6 * period(r), MU_EARTH, prograde);
                let hz = r1.cross(v1).z();
                assert!(hz != 0.0 || r1.z() != 0.0);
                if r1.z() == 0.0 {
                    // Equatorial r1: the transfer lies in the equator
                    assert!(
                        if prograde { hz > 0.0 } else { hz < 0.0 },
                        "r1={r1:?} prograde={prograde}: h_z={hz}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lambert_random_sweep() {
        // Deterministic pseudo-random geometries: any plane, both
        // directions, times of flight from strongly hyperbolic to several
        // revolutions. Every solution returned must reach r2.
        fn uniform(state: &mut u64) -> f64 {
            // xorshift64
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 11) as f64 / (1u64 << 53) as f64
        }
        fn unit(state: &mut u64) -> Vector3 {
            loop {
                let v = numeris::vector![
                    2.0 * uniform(state) - 1.0,
                    2.0 * uniform(state) - 1.0,
                    2.0 * uniform(state) - 1.0
                ];
                let n = v.norm();
                if n > 0.1 && n < 1.0 {
                    return v / n;
                }
            }
        }
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..1000 {
            let r1 = 7000.0e3 * unit(&mut state);
            let r2 = (6600.0e3 + 30000.0e3 * uniform(&mut state)) * unit(&mut state);
            // log-uniform from 0.02 to 4 periods of the r1 circular orbit
            let tof = period(7000.0e3) * 0.02 * 200.0_f64.powf(uniform(&mut state));
            solve_and_verify(&r1, &r2, tof, i % 2 == 0);
        }
    }

    #[test]
    fn test_lambert_rejects_nonfinite_and_coincident() {
        let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
        let r2 = numeris::vector![0.0, 7000.0e3, 0.0];
        let bad = numeris::vector![7000.0e3, f64::NAN, 0.0];
        let inf = numeris::vector![f64::INFINITY, 0.0, 0.0];
        assert!(matches!(
            lambert(&bad, &r2, 3600.0, MU_EARTH, true),
            Err(Error::NonFinite("r1"))
        ));
        assert!(matches!(
            lambert(&r1, &inf, 3600.0, MU_EARTH, true),
            Err(Error::NonFinite("r2"))
        ));
        for tof in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                lambert(&r1, &r2, tof, MU_EARTH, true),
                Err(Error::NonFinite("tof"))
            ));
        }
        assert!(matches!(
            lambert(&r1, &r2, 3600.0, f64::NAN, true),
            Err(Error::NonFinite("mu"))
        ));
        // r1 == r2 used to return NaN velocities
        assert!(matches!(
            lambert(&r1, &r1, 3600.0, MU_EARTH, true),
            Err(Error::CoincidentPositions)
        ));
    }
}
