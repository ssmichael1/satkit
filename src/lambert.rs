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
pub enum LambertError {
    #[error("Time of flight must be positive, got {0}")]
    InvalidTof(f64),
    #[error("Position vectors must be non-zero")]
    ZeroPosition,
    #[error("Gravitational parameter must be positive, got {0}")]
    InvalidMu(f64),
    #[error("Convergence failure for revolution {0}")]
    ConvergenceFailed(u32),
}

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
///   short-way / long-way ambiguity.
///
/// # Returns
///
/// Vector of `(v1, v2)` solutions. The first element is the zero-revolution
/// solution. Additional elements are multi-revolution solutions (if any exist
/// for the given time of flight), returned in pairs (short-period, long-period)
/// for each revolution count.
pub fn lambert(
    r1: &Vector3,
    r2: &Vector3,
    tof: f64,
    mu: f64,
    prograde: bool,
) -> Result<Vec<LambertSolution>, LambertError> {
    if tof <= 0.0 {
        return Err(LambertError::InvalidTof(tof));
    }
    if mu <= 0.0 {
        return Err(LambertError::InvalidMu(mu));
    }

    let r1_norm = r1.norm();
    let r2_norm = r2.norm();
    if r1_norm < 1.0e-10 || r2_norm < 1.0e-10 {
        return Err(LambertError::ZeroPosition);
    }

    // Chord and semiperimeter
    let c = (r2 - r1).norm();
    let s = (r1_norm + r2_norm + c) / 2.0;

    // Unit vectors
    let ir1 = r1 / r1_norm;
    let ir2 = r2 / r2_norm;
    let ih_raw = ir1.cross(&ir2);
    let ih_norm = ih_raw.norm();

    // Handle collinear positions (180-degree transfer)
    let ih = if ih_norm < 1.0e-12 {
        if ir1.x().abs() < 0.9 {
            ir1.cross(&numeris::vector![1.0, 0.0, 0.0]).normalize()
        } else {
            ir1.cross(&numeris::vector![0.0, 1.0, 0.0]).normalize()
        }
    } else {
        ih_raw / ih_norm
    };

    // Tangent unit vectors (perpendicular to position in orbital plane)
    let it1 = ih.cross(&ir1);
    let it2 = ih.cross(&ir2);

    // Transfer angle
    let mut dtheta = f64::acos(ir1.dot(&ir2).clamp(-1.0, 1.0));

    if prograde {
        if ih.z() < 0.0 {
            dtheta = 2.0 * PI - dtheta;
        }
    } else if ih.z() >= 0.0 {
        dtheta = 2.0 * PI - dtheta;
    }

    // Lambda parameter
    let lambda2 = 1.0 - c / s;
    let lambda = if dtheta > PI {
        -lambda2.sqrt()
    } else {
        lambda2.sqrt()
    };

    // Non-dimensional time of flight
    let t_norm = tof * (2.0 * mu / s.powi(3)).sqrt();

    // Velocity reconstruction constants
    let gamma = (mu * s / 2.0).sqrt();
    let rho = (r1_norm - r2_norm) / c;
    let sigma = (1.0 - rho * rho).sqrt();

    let mut solutions = Vec::new();

    // --- Zero-revolution solution ---
    let x0 = initial_guess_0rev(lambda, t_norm);
    let x = householder(lambda, t_norm, x0, 0)
        .ok_or(LambertError::ConvergenceFailed(0))?;
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
        term *= (3.0 + n) * (1.0 + n) / ((2.5 + n) * (n + 2.0)) * x;
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
        (-x + lambda * y - psi_h / (x * x - 1.0).sqrt()) / omx2
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
    let d3t = (7.0 * x * d2t + 8.0 * dt - 6.0 * (1.0 - lambda2) * lambda5 * x / y.powi(5))
        / omx2;

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
        x = x.clamp(-0.999, 0.999);
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

    /// Verify a Lambert solution by checking energy and angular momentum
    /// conservation, plus Keplerian propagation for non-equatorial orbits.
    fn verify_solution(r1: &Vector3, r2: &Vector3, v1: &Vector3, v2: &Vector3, tof: f64) {
        let r1n = r1.norm();
        let r2n = r2.norm();

        // Energy conservation
        let energy1 = v1.norm_squared() / 2.0 - MU_EARTH / r1n;
        let energy2 = v2.norm_squared() / 2.0 - MU_EARTH / r2n;
        let e_err = (energy1 - energy2).abs() / energy1.abs();
        assert!(
            e_err < 1.0e-8,
            "Energy mismatch: {:.2e} (E1={:.6e}, E2={:.6e})",
            e_err,
            energy1,
            energy2
        );

        // Angular momentum conservation
        let h1 = r1.cross(v1);
        let h2 = r2.cross(v2);
        let h_err = (h1 - h2).norm() / h1.norm();
        assert!(
            h_err < 1.0e-8,
            "Angular momentum mismatch: {:.2e}",
            h_err
        );

        // Propagation check for non-equatorial orbits
        let h = r1.cross(v1);
        let h_xy = (h.x() * h.x() + h.y() * h.y()).sqrt();
        if h_xy / h.norm() > 0.01 {
            if let Ok(k) = crate::Kepler::from_pv(*r1, *v1) {
                let dt = crate::Duration::from_seconds(tof);
                let k2 = k.propagate(&dt);
                let (r2_prop, _) = k2.to_pv();
                let pos_err = (r2_prop - r2).norm();
                assert!(
                    pos_err < 100.0,
                    "Propagation error: {:.1} m",
                    pos_err
                );
            }
        }
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
        verify_solution(&r1, &r2, v1, v2, tof);
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
}
