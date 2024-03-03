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
