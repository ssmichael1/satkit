use nalgebra as na;

// Equation 3.37 in Montenbruck & Gill
pub fn point_gravity(
    r: &na::Vector3<f64>, // object
    s: &na::Vector3<f64>, // distant attractor
    mu: f64,
) -> na::Vector3<f64> {
    let sr = s - r;
    let srnorm2 = sr.norm_squared();
    let srnorm = srnorm2.sqrt();
    let snorm2 = s.norm_squared();
    let snorm = snorm2.sqrt();
    mu * (sr / (srnorm * srnorm2) - s / (snorm * snorm2))
}

// Return tuple with point gravity force and
// point gravity partial (da/dr)
// Equation 3.37 in Montenbruck & Gill for point gravity
// Equation 7.75 in Montenbruck & Gill for partials

pub fn point_gravity_and_partials(
    r: &na::Vector3<f64>, // object
    s: &na::Vector3<f64>, // distant attractur
    mu: f64,
) -> (na::Vector3<f64>, na::Matrix3<f64>) {
    let rs = r - s;
    let rsnorm2 = rs.norm_squared();
    let rsnorm = rsnorm2.sqrt();
    let snorm2 = s.norm_squared();
    let snorm = snorm2.sqrt();
    (
        -mu * (rs / (rsnorm * rsnorm2) + s / (snorm * snorm2)),
        -mu * (na::Matrix3::<f64>::identity() / (rsnorm2 * rsnorm)
            - 3.0 * rs * rs.transpose() / (rsnorm2 * rsnorm2 * rsnorm)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_gravity_known() {
        // Moon at ~384,400 km from Earth center, satellite at GEO (~42,164 km)
        let mu_moon = 4.9048695e12; // m³/s²
        let s_moon = na::Vector3::new(384_400.0e3, 0.0, 0.0); // Moon position
        let r_sat = na::Vector3::new(42_164.0e3, 0.0, 0.0); // GEO satellite

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
    fn test_point_gravity_partials() {
        let mu = 4.9048695e12;
        let s = na::Vector3::new(384_400.0e3, 50_000.0e3, 20_000.0e3);
        let r = na::Vector3::new(42_164.0e3, 1000.0e3, 500.0e3);
        let dr = na::Vector3::new(10.0, 20.0, -15.0);

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
