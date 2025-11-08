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
    let itrf = ITRFCoord::from(pos_itrf.as_slice());
    let (density, _temperature) = nrlmsise(
        itrf.hae() / 1.0e3,
        Some(itrf.latitude_rad()),
        Some(itrf.longitude_rad()),
        Some(*time),
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

// Partials are used for computing state transition matrix
// and must be explicitly computed ... ughhh.
//
// Return density (rho)
// and density partials (drho / dr)
// All in the gcrf frame

fn compute_rho_drhodr(
    pgcrf: &Vector3,
    qgcrf2itrf: &Quaternion,
    time: &Instant,
    use_spaceweather: bool,
) -> (f64, Vector3) {
    let dx = 100.0;

    let pitrf = qgcrf2itrf * pgcrf;
    let itrf = ITRFCoord::from(pitrf);
    let qned2itrf = itrf.q_ned2itrf();

    // Offset in the NED frame
    let offset_vecs = [
        nalgebra::vector![dx, 0.0, 0.0],
        nalgebra::vector![0.0, dx, 0.0],
        nalgebra::vector![0.0, 0.0, dx],
    ];
    let (density0, _temperature) = crate::nrlmsise::nrlmsise(
        itrf.hae() / 1.0e3,
        Some(itrf.latitude_rad()),
        Some(itrf.longitude_rad()),
        Some(*time),
        use_spaceweather,
    );

    // Compute drhodr in the ned frame
    let drhodr_ned: Vec<f64> = offset_vecs
        .iter()
        .map(|v| {
            let itrf_off = itrf + qned2itrf * v;

            let (density, _temperature) = crate::nrlmsise::nrlmsise(
                itrf_off.hae() / 1.0e3,
                Some(itrf_off.latitude_rad()),
                Some(itrf_off.longitude_rad()),
                Some(*time),
                use_spaceweather,
            );
            (density - density0) / dx
        })
        .collect();
    // note: we have checked ... this appears to be correct in ned frame

    let qned2gcrf = qgcrf2itrf.conjugate() * qned2itrf;
    (
        density0,
        qned2gcrf * nalgebra::vector![drhodr_ned[0], drhodr_ned[1], drhodr_ned[2]],
    )
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

    let drag_accel_gcrf = -0.5 * cd_a_over_m * density * vrel * vrel.norm();

    // Now partials
    // Equation 7.81 and 7.84
    let dacceldv = -0.5
        * cd_a_over_m
        * density
        * (vrel * vrel.transpose() / vrel.norm() + vrel.norm() * Matrix3::identity());

    let dacceldr = -0.5 * cd_a_over_m * density * vrel * vrel.norm() * drhodr.transpose()
        - dacceldv * OMEGA_EARTH_MATRIX;

    (drag_accel_gcrf, dacceldr, dacceldv)
}
