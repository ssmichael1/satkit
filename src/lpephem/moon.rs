use crate::consts;
use crate::AstroTime;
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
/// and 1275 km in range
///
pub fn pos_gcrf(time: &AstroTime) -> na::Vector3<f64> {
    // Julian centuries since Jan 1, 2000 12pm

    let t: f64 = (time.to_jd(TimeScale::TDB) - 2451545.0) / 36525.0;

    #[allow(non_upper_case_globals)]
    const deg2rad: f64 = std::f64::consts::PI / 180.;

    let lambda_ecliptic: f64 = deg2rad
        * (218.32 + 481267.8813 * t + 6.29 * f64::sin(deg2rad * (134.9 + 477198.85 * t))
            - 1.27 * f64::sin(deg2rad * (259.2 - 413335.38 * t))
            + 0.66 * f64::sin(deg2rad * (235.7 + 890534.23 * t))
            + 0.21 * f64::sin(deg2rad * (269.9 + 954397.70 * t))
            - 0.19 * f64::sin(deg2rad * (357.5 + 35999.05 * t))
            - 0.11 * f64::sin(deg2rad * (186.6 + 966404.05 * t)));

    let phi_ecliptic: f64 = deg2rad
        * (5.13 * f64::sin(deg2rad * (93.3 + 483202.03 * t))
            + 0.28 * f64::sin(deg2rad * (228.2 + 960400.87 * t))
            - 0.28 * f64::sin(deg2rad * (318.3 + 6003.18 * t))
            - 0.17 * f64::sin(deg2rad * (217.6 - 407332.20 * t)));

    let hparallax: f64 = deg2rad
        * (0.9508
            + 0.0518 * f64::cos(deg2rad * (134.9 + 477198.85 * t))
            + 0.0095 * f64::cos(deg2rad * (259.2 - 413335.38 * t))
            + 0.0078 * f64::cos(deg2rad * (235.7 + 890534.23 * t))
            + 0.0028 * f64::cos(deg2rad * (269.9 + 954397.70 * t)));

    let epsilon: f64 =
        deg2rad * (23.439291 - 0.0130042 * t - 1.64e-7 * t * t + 5.04E-7 * t * t * t);

    // Convert values above from degrees to radians
    // for remainder of computations

    let rmag: f64 = consts::EARTH_RADIUS / f64::sin(hparallax);

    rmag * na::Vector3::<f64>::new(
        f64::cos(phi_ecliptic) * f64::cos(lambda_ecliptic),
        f64::cos(epsilon) * f64::cos(phi_ecliptic) * f64::sin(lambda_ecliptic)
            - f64::sin(epsilon) * f64::sin(phi_ecliptic),
        f64::sin(epsilon) * f64::cos(phi_ecliptic) * f64::sin(lambda_ecliptic)
            + f64::cos(epsilon) * f64::sin(phi_ecliptic),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonpos() {
        //! This is Vallado example 5-3
        let t0 = AstroTime::from_date(1994, 4, 28);
        // Approximate this UTC as TDB to match example...
        let t = AstroTime::from_mjd(t0.to_mjd(TimeScale::UTC), TimeScale::TDB);

        let pos = pos_gcrf(&t);

        // Below value is from Vallado example
        let ref_pos = vec![-134240.626E3, -311571.590E3, -126693.785E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            assert!(err < 1.0e-6);
        }
    }
}
