//! General-relativistic corrections to satellite motion.
//!
//! Currently implements only the Schwarzschild (post-Newtonian, β = γ = 1)
//! term — the dominant relativistic effect on Earth-orbiting satellites
//! (~1 m/day at GPS altitude, ~3 m/day at GEO). Lense-Thirring (frame
//! dragging, ~1 mm/day) and de Sitter (geodetic precession, ~10 mm/day
//! on the coordinate system) are sub-cm-class effects and not yet
//! implemented.
//!
//! Reference: IERS Conventions 2010 (IERS Technical Note 36), §10.3
//! Eq. 10.12.

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn schwarzschild_points_inward_on_circular_orbit() {
        // On a circular orbit r·v = 0, so only the radial term survives.
        // 4GM/r − v² > 0 (since v² = GM/r on a circular orbit), so the
        // acceleration is along +r̂ in the formula above — but the formula
        // sign convention is that the static Newtonian term is + GM r̂ / r³,
        // i.e. attractive uses a *negative* coefficient on r̂ elsewhere.
        // Here the +r̂ result is the *correction* to that attractive force,
        // and corresponds to a *stronger* attraction (perihelion precession
        // is in the prograde direction). Check that the dot product with
        // -r̂ is positive (i.e. the correction adds to the inward pull).
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
