use crate::mathtypes::*;

// Third-body acceleration, Equation 3.37 in Montenbruck & Gill:
//
//   a = -mu (d/|d|^3 + s/|s|^3),   d = r - s
//
// For a distant attractor the direct and indirect terms nearly cancel, losing
// about log10(|s| / 2|r|) digits (4 for the Sun in LEO). Battin's form
// (Battin, "An Introduction to the Mathematics and Methods of Astrodynamics",
// Encke's method) computes the difference without subtracting them:
//
//   a = -mu/|d|^3 (r + f(q) s),   q = r.(r - 2s) / s.s,
//   f(q) = (1+q)^(3/2) - 1 = q (3 + 3q + q^2) / (1 + (1+q)^(3/2))
//
// Returns the acceleration and d = r - s, |d|^2, |d|^3 for the partials.
#[inline]
fn battin_accel(r: &Vector3, s: &Vector3, mu: f64) -> (Vector3, Vector3, f64, f64) {
    let rs = r - s;
    let rsnorm2 = rs.norm_squared();
    let rsnorm3 = rsnorm2 * rsnorm2.sqrt();
    let q = r.dot(&(r - 2.0 * s)) / s.norm_squared();
    let f = q * (3.0 + 3.0 * q + q * q) / (1.0 + (1.0 + q) * (1.0 + q).sqrt());
    (-mu / rsnorm3 * (r + f * s), rs, rsnorm2, rsnorm3)
}

pub fn point_gravity(
    r: &Vector3, // object
    s: &Vector3, // distant attractor
    mu: f64,
) -> Vector3 {
    battin_accel(r, s, mu).0
}

// Return tuple with point gravity force and
// point gravity partial (da/dr)
// Battin's form of Equation 3.37 in Montenbruck & Gill for point gravity
// Equation 7.75 in Montenbruck & Gill for partials

pub fn point_gravity_and_partials(
    r: &Vector3, // object
    s: &Vector3, // distant attractor
    mu: f64,
) -> (Vector3, Matrix3) {
    let (accel, rs, rsnorm2, rsnorm3) = battin_accel(r, s, mu);
    (
        accel,
        -mu * (Matrix3::eye() / rsnorm3 - 3.0 * rs * rs.transpose() / (rsnorm2 * rsnorm3)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_gravity_known() {
        // Moon at ~384,400 km from Earth center, satellite at GEO (~42,164 km)
        let mu_moon = crate::consts::MU_MOON;
        let s_moon = numeris::vector![384_400.0e3, 0.0, 0.0]; // Moon position
        let r_sat = numeris::vector![42_164.0e3, 0.0, 0.0]; // GEO satellite

        let accel = point_gravity(&r_sat, &s_moon, mu_moon);
        // Lunar perturbation at GEO should be ~1e-6 m/s² order of magnitude
        let mag = accel.norm();
        assert!(
            mag > 1.0e-7 && mag < 1.0e-4,
            "Lunar perturbation at GEO = {:.3e}, expected ~1e-6",
            mag
        );
    }

    #[test]
    fn test_point_gravity_precision() {
        // Reference values from 50-digit decimal arithmetic of Eq. 3.37. The
        // direct-minus-indirect form is good to only 1.5e-12 (Sun) and
        // 1.5e-14 (Moon) here; Battin's form is at machine precision.
        let r = numeris::vector![5.1e6, 4.2e6, 2.3e6];
        let cases = [
            (
                numeris::vector![1.2e11, -8.1e10, -3.5e10],
                1.32712440018e20,
                numeris::vector![
                    -8.017300151724125e-08,
                    -2.528135504540415e-07,
                    -1.287259378922576e-07
                ],
            ),
            (
                numeris::vector![3.1e8, -2.0e8, -9.0e7],
                4.9028e12,
                numeris::vector![
                    -1.6466721096055815e-07,
                    -5.715870008073985e-07,
                    -2.943163980383449e-07
                ],
            ),
        ];
        for (s, mu, expected) in cases {
            let accel = point_gravity(&r, &s, mu);
            let err = (accel - expected).norm() / expected.norm();
            assert!(err < 1.0e-15, "relative error {err:.3e}");
            let (accel_p, _) = point_gravity_and_partials(&r, &s, mu);
            assert_eq!(accel, accel_p);
        }
    }

    #[test]
    fn test_point_gravity_partials() {
        let mu = crate::consts::MU_MOON;
        let s = numeris::vector![384_400.0e3, 50_000.0e3, 20_000.0e3];
        let r = numeris::vector![42_164.0e3, 1000.0e3, 500.0e3];
        let dr = numeris::vector![10.0, 20.0, -15.0];

        let (accel0, partials) = point_gravity_and_partials(&r, &s, mu);
        let accel1 = point_gravity(&(r + dr), &s, mu);

        // accel(r+dr) ≈ accel(r) + (da/dr)*dr
        let predicted = accel0 + partials * dr;
        let err = (accel1 - predicted).norm() / accel1.norm();
        assert!(
            err < 1.0e-6,
            "Partial derivative error = {:.3e}, expected < 1e-6",
            err
        );
    }
}
