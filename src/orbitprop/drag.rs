use crate::nrlmsise::nrlmsise;
use crate::ITRFCoord;
use crate::Instant;

use crate::mathtypes::*;

const OMEGA_EARTH: Vector3 = nalgebra::vector![0.0, 0.0, crate::consts::OMEGA_EARTH];
const OMEGA_EARTH_MATRIX: Matrix3 = Matrix3::new(
    0.0,
    -crate::consts::OMEGA_EARTH,
    0.0,
    crate::consts::OMEGA_EARTH,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
);

// Compute and return force from drag in the gcrf frame
pub fn drag_force(
    pos_gcrf: &Vector3,
    pos_itrf: &Vector3,
    vel_gcrf: &Vector3,
    time: &crate::Instant,
    cd_a_over_m: f64,
    use_spaceweather: bool,
) -> Vector3 {
    let itrf = ITRFCoord::from_vector(pos_itrf);
    let (lat, lon, hae) = itrf.to_geodetic_rad();
    let (density, _temperature) = nrlmsise(
        hae / 1.0e3,
        Some(lat),
        Some(lon),
        Some(time),
        use_spaceweather,
    );

    // The "wind" moves along with the rotation earth, so we subtract off the
    // rotation Earth part in while still staying in the gcrf frame
    // to get velocity relative to wind in gcrf frame
    // This is a little confusing, but if you think about it long enough
    // it will make sense
    let vrel = vel_gcrf - OMEGA_EARTH.cross(pos_gcrf);

    -0.5 * cd_a_over_m * density * vrel * vrel.norm()
}

// Compute density and its gradient with respect to GCRF position.
//
// Density varies almost entirely with altitude. We compute dρ/dh via
// a single forward difference in altitude, then project along the
// geodetic normal (the direction in which altitude increases) to get
// the full 3D gradient in GCRF.
//
// This replaces an earlier approach that did 3 finite differences
// in the NED frame (4 NRLMSISE calls total). The new version uses
// only 2 NRLMSISE calls and is more physically motivated.
fn compute_rho_drhodr(
    pgcrf: &Vector3,
    qgcrf2itrf: &Quaternion,
    time: &Instant,
    use_spaceweather: bool,
) -> (f64, Vector3) {
    let pitrf = qgcrf2itrf * pgcrf;
    let itrf = ITRFCoord::from(pitrf);
    let hae = itrf.hae();
    let lat = itrf.latitude_rad();
    let lon = itrf.longitude_rad();

    let (rho0, _) = nrlmsise(hae / 1.0e3, Some(lat), Some(lon), Some(time), use_spaceweather);

    // Forward difference in altitude only (same lat/lon)
    let dh = 100.0; // meters
    let (rho1, _) = nrlmsise(
        (hae + dh) / 1.0e3,
        Some(lat),
        Some(lon),
        Some(time),
        use_spaceweather,
    );
    let drho_dh = (rho1 - rho0) / dh;

    // Geodetic "up" direction in ITRF: NED down is [0,0,1], so up is [0,0,-1]
    // rotated to ITRF via q_ned2itrf
    let up_itrf = itrf.q_ned2itrf() * nalgebra::vector![0.0, 0.0, -1.0];

    // Rotate to GCRF and scale by dρ/dh
    let up_gcrf = qgcrf2itrf.conjugate() * up_itrf;
    (rho0, drho_dh * up_gcrf)
}

// Compute drag force and partials with respect to
// position and velocity
//
// All in the gcrf frame
pub fn drag_and_partials(
    pos_gcrf: &Vector3,
    qgcrf2itrf: &Quaternion,
    vel_gcrf: &Vector3,
    time: &crate::Instant,
    cd_a_over_m: f64,
    use_spaceweather: bool,
) -> (Vector3, Matrix3, Matrix3) {
    let (density, drhodr) = compute_rho_drhodr(pos_gcrf, qgcrf2itrf, time, use_spaceweather);

    // The "wind" moves along with the rotation earth, so we subtract off the
    // rotation Earth part in while still staying in the gcrf frame
    // to get velocity relative to wind in gcrf frame
    // This is a little confusing, but if you think about it long enough
    // it will make sense
    let vrel = vel_gcrf - OMEGA_EARTH.cross(pos_gcrf);
    let vrel_norm = vrel.norm();

    let drag_accel_gcrf = -0.5 * cd_a_over_m * density * vrel * vrel_norm;

    // Partials of drag acceleration (Montenbruck & Gill, §3.5 / eq 7.81, 7.84)
    //
    // ∂a/∂v = -0.5 * CdA/m * ρ * (v_rel * v_rel^T / |v_rel| + |v_rel| * I)
    let dacceldv = -0.5
        * cd_a_over_m
        * density
        * (vrel * vrel.transpose() / vrel_norm + vrel_norm * Matrix3::identity());

    // ∂a/∂r has two terms:
    //   1) from ρ(r):    -0.5 * CdA/m * v_rel * |v_rel| * (∂ρ/∂r)^T
    //   2) from v_rel(r): (∂a/∂v) * (∂v_rel/∂r) = (∂a/∂v) * (-[ω×])
    let dacceldr = -0.5 * cd_a_over_m * vrel * vrel_norm * drhodr.transpose()
        - dacceldv * OMEGA_EARTH_MATRIX;

    (drag_accel_gcrf, dacceldr, dacceldv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drag_force_direction() {
        // Circular prograde orbit at ~400 km altitude
        let r = 6778.0e3; // Earth radius + 400 km
        let v_circ = (crate::consts::MU_EARTH / r).sqrt();
        let pos_gcrf = Vector3::new(r, 0.0, 0.0);
        let pos_itrf = pos_gcrf; // Approximate: ignore frame rotation for this test
        let vel_gcrf = Vector3::new(0.0, v_circ, 0.0); // prograde
        let time = Instant::from_datetime(2020, 1, 1, 0, 0, 0.0).unwrap();
        let cd_a_over_m = 0.01; // typical value

        let drag = drag_force(&pos_gcrf, &pos_itrf, &vel_gcrf, &time, cd_a_over_m, false);

        // Drag should oppose relative velocity direction
        let vrel = vel_gcrf - OMEGA_EARTH.cross(&pos_gcrf);
        let drag_dot_vrel = drag.dot(&vrel);
        assert!(
            drag_dot_vrel < 0.0,
            "Drag should oppose velocity, dot product = {}",
            drag_dot_vrel
        );
        assert!(drag.norm() > 0.0, "Drag magnitude should be > 0");

        // Drag should scale with cd_a_over_m
        let drag2 = drag_force(
            &pos_gcrf, &pos_itrf, &vel_gcrf, &time,
            cd_a_over_m * 2.0, false,
        );
        approx::assert_relative_eq!(drag2.norm(), drag.norm() * 2.0, max_relative = 1.0e-10);
    }

    #[test]
    fn test_drag_force_zero_at_high_alt() {
        // At 2000 km altitude, atmospheric density should be negligible
        let r = 8378.0e3; // Earth radius + 2000 km
        let v_circ = (crate::consts::MU_EARTH / r).sqrt();
        let pos_gcrf = Vector3::new(r, 0.0, 0.0);
        let pos_itrf = pos_gcrf;
        let vel_gcrf = Vector3::new(0.0, v_circ, 0.0);
        let time = Instant::from_datetime(2020, 1, 1, 0, 0, 0.0).unwrap();

        let drag = drag_force(&pos_gcrf, &pos_itrf, &vel_gcrf, &time, 0.01, false);
        assert!(
            drag.norm() < 1.0e-10,
            "Drag at 2000 km = {:.3e}, expected < 1e-10",
            drag.norm()
        );
    }
}
