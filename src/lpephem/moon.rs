use crate::consts;
use crate::Instant;
use crate::TimeScale;

use nalgebra as na;

///
/// Approximate Moon position in the GCRF Frame
///
/// From Vallado Algorithm 31
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon position
///
/// Output:
///
///  * Vector representing moon position in GCRF frame
///    at given time.  Units are meters
///
/// # Notes
///
/// * Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
///   and 1275 km in range
///
pub fn pos_gcrf(time: &Instant) -> na::Vector3<f64> {
    // Julian centuries since Jan 1, 2000 12pm

    let t: f64 = (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0;

    #[allow(non_upper_case_globals)]
    const deg2rad: f64 = std::f64::consts::PI / 180.;

    let lambda_ecliptic: f64 = deg2rad
        * 0.11f64.mul_add(-f64::sin(deg2rad * 966404.05f64.mul_add(t, 186.6)), 0.19f64.mul_add(-f64::sin(deg2rad * 35999.05f64.mul_add(t, 357.5)), 0.21f64.mul_add(f64::sin(deg2rad * 954397.70f64.mul_add(t, 269.9)), 0.66f64.mul_add(f64::sin(deg2rad * 890534.23f64.mul_add(t, 235.7)), 1.27f64.mul_add(-f64::sin(deg2rad * 413335.38f64.mul_add(-t, 259.2)), 6.29f64.mul_add(f64::sin(deg2rad * 477198.85f64.mul_add(t, 134.9)), 481267.8813f64.mul_add(t, 218.32)))))));

    let phi_ecliptic: f64 = deg2rad
        * 0.17f64.mul_add(-f64::sin(deg2rad * 407332.20f64.mul_add(-t, 217.6)), 0.28f64.mul_add(-f64::sin(deg2rad * 6003.18f64.mul_add(t, 318.3)), 5.13f64.mul_add(f64::sin(deg2rad * 483202.03f64.mul_add(t, 93.3)), 0.28 * f64::sin(deg2rad * 960400.87f64.mul_add(t, 228.2)))));

    let hparallax: f64 = deg2rad
        * 0.0028f64.mul_add(f64::cos(deg2rad * 954397.70f64.mul_add(t, 269.9)), 0.0078f64.mul_add(f64::cos(deg2rad * 890534.23f64.mul_add(t, 235.7)), 0.0095f64.mul_add(f64::cos(deg2rad * 413335.38f64.mul_add(-t, 259.2)), 0.0518f64.mul_add(f64::cos(deg2rad * 477198.85f64.mul_add(t, 134.9)), 0.9508))));

    let epsilon: f64 =
        deg2rad * (5.04E-7 * t * t).mul_add(t, (1.64e-7 * t).mul_add(-t, 0.0130042f64.mul_add(-t, 23.439291)));

    // Convert values above from degrees to radians
    // for remainder of computations

    let rmag: f64 = consts::EARTH_RADIUS / f64::sin(hparallax);

    rmag * na::Vector3::<f64>::new(
        f64::cos(phi_ecliptic) * f64::cos(lambda_ecliptic),
        (f64::cos(epsilon) * f64::cos(phi_ecliptic)).mul_add(f64::sin(lambda_ecliptic), -(f64::sin(epsilon) * f64::sin(phi_ecliptic))),
        (f64::sin(epsilon) * f64::cos(phi_ecliptic)).mul_add(f64::sin(lambda_ecliptic), f64::cos(epsilon) * f64::sin(phi_ecliptic)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonpos() {
        //! This is Vallado example 5-3
        let t0 = Instant::from_date(1994, 4, 28);
        // Approximate this UTC as TDB to match example...
        let t = Instant::from_mjd_with_scale(t0.as_mjd_with_scale(TimeScale::UTC), TimeScale::TDB);

        let pos = pos_gcrf(&t);

        // Below value is from Vallado example
        let ref_pos = [-134240.626E3, -311571.590E3, -126693.785E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            assert!(err < 1.0e-6);
        }
    }
}
