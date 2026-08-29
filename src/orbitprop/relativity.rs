//! General-relativistic corrections to satellite motion.
//!
//! Implements the three terms of IERS Conventions 2010 (IERS Technical Note
//! 36) §10.3 Eq. 10.12 with PPN parameters β = γ = 1:
//!
//! * **Schwarzschild** (static post-Newtonian field of the Earth) — the
//!   dominant term, ~1e-9 m/s² at LEO, ~3e-10 m/s² at GEO.
//! * **Geodesic (de Sitter) precession** — the Earth-centred frame precesses
//!   at |Ω| ≈ 1.92″/century (≈ 2.9e-15 rad/s) as the Earth falls around the
//!   Sun; the resulting Coriolis-like acceleration `2(Ω × v)` is ~4e-11 m/s²
//!   at LEO and ~8e-12 m/s² at 200,000 km, where it exceeds Schwarzschild.
//! * **Lense–Thirring** (frame dragging by the Earth's spin angular momentum)
//!   — ~1e-10 m/s² at LEO, mostly periodic, falling as 1/r³.
//!
//! Their effect on a propagated position depends on the orbit, arc length,
//! and which initial conditions are fitted, so none is a fixed drift per
//! day. The total is applied when
//! [`PropSettings::use_relativistic_correction`](super::PropSettings) is set
//! (the default) and matches GMAT's `RelativisticCorrection` (GMAT
//! Mathematical Specification §4.1.1, Table 4.1).
//!
//! All inputs and outputs are SI (m, m/s, m/s²) in the GCRF frame.

use crate::consts;
use crate::mathtypes::*;

/// Schwarzschild post-Newtonian acceleration on a satellite in the
/// non-rotating geocentric (GCRF) frame.
///
/// Implements IERS 2010 Eq. 10.12 with PPN parameters β = γ = 1:
///
/// ```text
/// a_GR = (GM / c² r³) · { (4 GM/r − v²) r  +  4 (r·v) v }
/// ```
///
/// For a state satisfying the Newtonian circular-orbit relation
/// `v² = GM / r` and `r·v = 0`, this correction points radially outward;
/// it is added to the much larger inward Newtonian acceleration.
///
/// Inputs and output are in SI units (m, m/s, m/s²) in the GCRF frame.
pub fn gr_schwarzschild_accel(pos_gcrf: &Vector3, vel_gcrf: &Vector3, mu_e: f64) -> Vector3 {
    let r2 = pos_gcrf.norm_squared();
    let r = r2.sqrt();
    let v2 = vel_gcrf.norm_squared();
    let rdotv = pos_gcrf.dot(vel_gcrf);

    let c2 = consts::C * consts::C;
    let factor = mu_e / (c2 * r2 * r);
    let radial_coeff = 4.0 * mu_e / r - v2;
    factor * (radial_coeff * pos_gcrf + 4.0 * rdotv * vel_gcrf)
}

/// Earth's spin angular momentum per unit mass, |J| = (2/5) R² ω, for a
/// homogeneous rigid sphere. The real Earth's moment-of-inertia factor is
/// ≈0.33 (not 0.4), so this overstates |J| by ~20%; the Lense–Thirring
/// acceleration it feeds is ≤1e-10 m/s² and mostly periodic, far below
/// any current validation floor, and the same approximation is used by
/// GMAT (MathSpec §4.2.6). Radius [`consts::EARTH_RADIUS`] rotating at
/// [`consts::OMEGA_EARTH`] (≈ 1.19e9 m²/s). This is the approximation used
/// by IERS 2010 Eq. 10.12 and by GMAT.
pub const EARTH_ANGULAR_MOMENTUM_PER_MASS: f64 =
    0.4 * consts::EARTH_RADIUS * consts::EARTH_RADIUS * consts::OMEGA_EARTH;

/// Geodesic (de Sitter) precession rate of the geocentric frame, in rad/s,
/// from the Earth's heliocentric motion:
///
/// ```text
/// Ω = (3/2) · v_E⊙ × ( −μ⊙ r_E⊙ / (c² |r_E⊙|³) )
///   = (3/2) · μ⊙ / (c² |r_E⊙|³) · (r_E⊙ × v_E⊙)
/// ```
///
/// `sun_pos_gcrf` / `sun_vel_gcrf` are the Sun's geocentric position and
/// velocity (so the Earth's heliocentric state is their negative; the cross
/// product is unchanged). |Ω| ≈ 2.9e-15 rad/s (1.92″/century).
pub fn geodesic_precession_rate(sun_pos_gcrf: &Vector3, sun_vel_gcrf: &Vector3) -> Vector3 {
    let r = sun_pos_gcrf.norm();
    let c2 = consts::C * consts::C;
    (1.5 * consts::MU_SUN / (c2 * r * r * r)) * sun_pos_gcrf.cross(sun_vel_gcrf)
}

/// Geodesic (de Sitter) precession acceleration, `2 (Ω × v)`, the
/// third bracket of IERS 2010 Eq. 10.12 with γ = 1 (the `(1 + 2γ) = 3`
/// factor is absorbed into [`geodesic_precession_rate`]).
pub fn gr_geodesic_accel(vel_gcrf: &Vector3, omega: &Vector3) -> Vector3 {
    2.0 * omega.cross(vel_gcrf)
}

/// Lense–Thirring (frame-dragging) acceleration, the second bracket of
/// IERS 2010 Eq. 10.12 with γ = 1:
///
/// ```text
/// a_LT = 2 μ / (c² r³) · { (3 / r²) (r × v) (r · J)  +  (v × J) }
/// ```
///
/// `j_gcrf` is the Earth's spin angular momentum per unit mass in GCRF
/// (magnitude [`EARTH_ANGULAR_MOMENTUM_PER_MASS`] along the ITRF pole).
pub fn gr_lense_thirring_accel(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    mu_e: f64,
    j_gcrf: &Vector3,
) -> Vector3 {
    let r2 = pos_gcrf.norm_squared();
    let r = r2.sqrt();
    let c2 = consts::C * consts::C;
    let factor = 2.0 * mu_e / (c2 * r2 * r);
    factor * ((3.0 / r2) * pos_gcrf.dot(j_gcrf) * pos_gcrf.cross(vel_gcrf) + vel_gcrf.cross(j_gcrf))
}

/// Total relativistic acceleration of IERS 2010 Eq. 10.12 (β = γ = 1):
/// Schwarzschild + geodesic precession + Lense–Thirring.
///
/// * `sun_pos_gcrf`, `sun_vel_gcrf` — geocentric Sun state (m, m/s)
/// * `qitrf2gcrf` — rotation taking the ITRF pole (ẑ) into GCRF, used to
///   orient the Earth's angular momentum
pub fn gr_accel(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    mu_e: f64,
    sun_pos_gcrf: &Vector3,
    sun_vel_gcrf: &Vector3,
    qitrf2gcrf: &Quaternion,
) -> Vector3 {
    let omega = geodesic_precession_rate(sun_pos_gcrf, sun_vel_gcrf);
    let j_gcrf = *qitrf2gcrf * numeris::vector![0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS];
    gr_schwarzschild_accel(pos_gcrf, vel_gcrf, mu_e)
        + gr_geodesic_accel(vel_gcrf, &omega)
        + gr_lense_thirring_accel(pos_gcrf, vel_gcrf, mu_e, &j_gcrf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative geocentric Sun state (m, m/s): 1 AU along +x, orbital
    /// velocity along +y (Earth's heliocentric velocity is then −y).
    fn sun_state() -> (Vector3, Vector3) {
        (
            numeris::vector![consts::AU, 0.0, 0.0],
            numeris::vector![0.0, 29_780.0, 0.0],
        )
    }

    #[test]
    fn geodesic_rate_is_1_9_arcsec_per_century() {
        let (sp, sv) = sun_state();
        let omega = geodesic_precession_rate(&sp, &sv);
        // 1.92″/century = 1.92 * 4.848e-6 rad / 3.156e9 s ≈ 2.95e-15 rad/s
        let expected = 1.92 * 4.848_137e-6 / (100.0 * 365.25 * 86400.0);
        assert!(
            (omega.norm() / expected - 1.0).abs() < 0.05,
            "|Ω| = {:e} rad/s, expected ≈ {:e}",
            omega.norm(),
            expected
        );
        // Ω is normal to the ecliptic (here the x-y plane)
        assert!(omega[0].abs() < 1e-30 && omega[1].abs() < 1e-30 && omega[2] > 0.0);
    }

    #[test]
    fn geodesic_accel_magnitude_and_direction() {
        let (sp, sv) = sun_state();
        let omega = geodesic_precession_rate(&sp, &sv);
        for r in [consts::EARTH_RADIUS + 400.0e3, 200_000.0e3] {
            let v = (consts::MU_EARTH / r).sqrt();
            let vel = numeris::vector![v, 0.0, 0.0]; // ⊥ Ω → |Ω×v| = |Ω||v|
            let a = gr_geodesic_accel(&vel, &omega);
            let expected = 2.0 * omega.norm() * v;
            assert!((a.norm() / expected - 1.0).abs() < 1e-12);
            assert!(a.dot(&vel).abs() < 1e-30, "geodesic term must be ⊥ v");
        }
        // Order of magnitude at LEO: ~4e-11 m/s²
        let v_leo = (consts::MU_EARTH / (consts::EARTH_RADIUS + 400.0e3)).sqrt();
        let a_leo = gr_geodesic_accel(&numeris::vector![v_leo, 0.0, 0.0], &omega).norm();
        assert!(
            (1e-11..1e-10).contains(&a_leo),
            "geodesic at LEO = {a_leo:e}"
        );
    }

    #[test]
    fn lense_thirring_magnitude_at_leo() {
        // Equatorial LEO with J along +z: r·J = 0 and v ⊥ J, so only the
        // v×J term survives with full magnitude: |a| = 2μ/(c²r³) · v · J.
        // (For a polar orbit at the equator crossing v ∥ J and the term
        // vanishes — Lense–Thirring is strongest for equatorial motion.)
        let r = consts::EARTH_RADIUS + 400.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v, 0.0];
        let j = numeris::vector![0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS];
        let a = gr_lense_thirring_accel(&pos, &vel, consts::MU_EARTH, &j);
        let c2 = consts::C * consts::C;
        let expected =
            2.0 * consts::MU_EARTH / (c2 * r * r * r) * v * EARTH_ANGULAR_MOMENTUM_PER_MASS;
        assert!((a.norm() / expected - 1.0).abs() < 1e-12);
        assert!(
            (1e-11..1e-9).contains(&a.norm()),
            "LT at LEO = {:e}",
            a.norm()
        );
        // |J| ≈ 1.19e9 m²/s
        assert!((EARTH_ANGULAR_MOMENTUM_PER_MASS / 1.186e9 - 1.0).abs() < 0.01);
    }

    #[test]
    fn total_reduces_to_schwarzschild_without_sun_motion_and_spin() {
        let r = consts::EARTH_RADIUS + 700.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v * 0.8, v * 0.6];
        // Ω = 0 when the Sun is at rest; J = 0 explicitly.
        let omega =
            geodesic_precession_rate(&numeris::vector![consts::AU, 0.0, 0.0], &Vector3::zeros());
        assert_eq!(omega.norm(), 0.0);
        let total = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH)
            + gr_geodesic_accel(&vel, &omega)
            + gr_lense_thirring_accel(&pos, &vel, consts::MU_EARTH, &Vector3::zeros());
        let s = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH);
        assert!((total - s).norm() < 1e-30);
    }

    #[test]
    fn total_at_high_altitude_is_geodesic_dominated() {
        // At 200,000 km the geodesic term (~8e-12) exceeds Schwarzschild (~5e-13).
        let (sp, sv) = sun_state();
        let r = 200_000.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v, 0.0];
        let q = Quaternion::identity();
        let total = gr_accel(&pos, &vel, consts::MU_EARTH, &sp, &sv, &q);
        let s = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH);
        assert!(
            total.norm() > 5.0 * s.norm(),
            "total {:e} vs schwarzschild {:e}",
            total.norm(),
            s.norm()
        );
    }

    #[test]
    fn schwarzschild_at_geo_has_expected_magnitude() {
        // GEO circular orbit: r ≈ 4.2e7, v ≈ 3075 m/s. The Schwarzschild
        // term magnitude should be a few times 10⁻¹⁰ m/s² there.
        let r = consts::GEO_R;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v, 0.0];
        let a = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH);
        let mag = a.norm();
        assert!(
            (1e-11..1e-8).contains(&mag),
            "GR accel at GEO = {:e} m/s², expected ~few×1e-10",
            mag
        );
    }

    #[test]
    fn schwarzschild_at_leo_has_expected_magnitude() {
        // ~500 km LEO: r ≈ 6.87e6, v ≈ 7.6 km/s. The Schwarzschild
        // term is roughly an order of magnitude larger than at GEO due
        // to the 1/r³ prefactor and higher v².
        let r = consts::EARTH_RADIUS + 500.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v, 0.0];
        let a = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH);
        let mag = a.norm();
        assert!(
            (1e-10..1e-7).contains(&mag),
            "GR accel at 500 km LEO = {:e} m/s², expected ~1e-9",
            mag
        );
    }

    #[test]
    fn schwarzschild_points_outward_on_circular_orbit() {
        // On a circular orbit r·v = 0, so only the radial term survives.
        // With v² = GM/r, the radial coefficient is 3GM/r > 0, so the
        // post-Newtonian correction points along +r̂ (outward). It is added
        // to the much larger Newtonian acceleration, which points along -r̂.
        let r = consts::EARTH_RADIUS + 1000.0e3;
        let v = (consts::MU_EARTH / r).sqrt();
        let pos = numeris::vector![r, 0.0, 0.0];
        let vel = numeris::vector![0.0, v, 0.0];
        let a = gr_schwarzschild_accel(&pos, &vel, consts::MU_EARTH);
        // r·v = 0 → no along-velocity contribution
        assert!(a[1].abs() < 1e-20);
        assert!(a[2].abs() < 1e-20);
        // 4GM/r − v² = 4GM/r − GM/r = 3GM/r > 0 → component along +x is positive
        assert!(a[0] > 0.0);
    }
}
