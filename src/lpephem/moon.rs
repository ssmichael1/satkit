use crate::consts;
use crate::Instant;
use crate::TimeScale;

use crate::mathtypes::*;

/// Compute approximate ecliptic longitude of the moon
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon ecliptic longitude
///
/// Returns:
///
/// * Ecliptic longitude of moon, radians
///
/// # Notes
///
/// See Vallado Algorithm 31
///
pub fn ecliptic_longitude(time: &Instant) -> f64 {
    // Julian centuries since Jan 1, 2000 12pm
    let t: f64 = (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0;

    #[allow(non_upper_case_globals)]
    const deg2rad: f64 = std::f64::consts::PI / 180.;

    let lon = deg2rad
        * 0.11f64.mul_add(
            -f64::sin(deg2rad * 966404.05f64.mul_add(t, 186.6)),
            0.19f64.mul_add(
                -f64::sin(deg2rad * 35999.05f64.mul_add(t, 357.5)),
                0.21f64.mul_add(
                    f64::sin(deg2rad * 954397.70f64.mul_add(t, 269.9)),
                    0.66f64.mul_add(
                        f64::sin(deg2rad * 890534.23f64.mul_add(t, 235.7)),
                        1.27f64.mul_add(
                            -f64::sin(deg2rad * 413335.38f64.mul_add(-t, 259.2)),
                            6.29f64.mul_add(
                                f64::sin(deg2rad * 477198.85f64.mul_add(t, 134.9)),
                                481267.8813f64.mul_add(t, 218.32),
                            ),
                        ),
                    ),
                ),
            ),
        );

    lon % (2.0 * std::f64::consts::PI)
}

/// Compute approximate phase of the moon
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon phase
///
/// Returns:
/// * Phase of the moon, radians
///
/// # Notes
///
/// See Vallado Section 5.2.3
///
pub fn phase(time: &Instant) -> f64 {
    let lambda_moon = ecliptic_longitude(time);
    let lambda_sun = crate::lpephem::sun::ecliptic_longitude(time);

    let phase = (lambda_moon - lambda_sun) % (2.0 * std::f64::consts::PI);
    if phase < 0.0 {
        phase + 2.0 * std::f64::consts::PI
    } else if phase > 2.0 * std::f64::consts::PI {
        phase - 2.0 * std::f64::consts::PI
    } else {
        phase
    }
}

/// Compute fraction of moon illuminated at given time
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon illumination
///
/// Returns:
///
/// * Fraction of moon illuminated, range 0.0 to 1.0
///
/// # Notes
///
/// See Vallado Section 5.2.3
pub fn illumination(time: &Instant) -> f64 {
    let phase = phase(time);
    0.5 * (1.0 - f64::cos(phase))
}

/// Moon phase names
#[derive(Debug, Clone, Copy, std::cmp::PartialEq, std::cmp::Eq)]
pub enum MoonPhase {
    /// New Moon (0° - 22.5°)
    NewMoon,
    /// Waxing Crescent (22.5° - 67.5°)
    WaxingCrescent,
    /// First Quarter (67.5° - 112.5°)
    FirstQuarter,
    /// Waxing Gibbous (112.5° - 157.5°)
    WaxingGibbous,
    /// Full Moon (157.5° - 202.5°)
    FullMoon,
    /// Waning Gibbous (202.5° - 247.5°)
    WaningGibbous,
    /// Last Quarter (247.5° - 292.5°)
    LastQuarter,
    /// Waning Crescent (292.5° - 337.5°)
    WaningCrescent,
}

impl MoonPhase {
    /// Get the name of the moon phase as a string
    pub fn name(&self) -> &'static str {
        match self {
            Self::NewMoon => "New Moon",
            Self::WaxingCrescent => "Waxing Crescent",
            Self::FirstQuarter => "First Quarter",
            Self::WaxingGibbous => "Waxing Gibbous",
            Self::FullMoon => "Full Moon",
            Self::WaningGibbous => "Waning Gibbous",
            Self::LastQuarter => "Last Quarter",
            Self::WaningCrescent => "Waning Crescent",
        }
    }
}

/// Determine the phase name of the moon at given time
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon phase name
///
/// Returns:
///
/// * MoonPhase enum value representing the current phase
///
/// # Notes
///
/// Phase boundaries:
/// - New Moon: 0° - 22.5° (or 337.5° - 360°)
/// - Waxing Crescent: 22.5° - 67.5°
/// - First Quarter: 67.5° - 112.5°
/// - Waxing Gibbous: 112.5° - 157.5°
/// - Full Moon: 157.5° - 202.5°
/// - Waning Gibbous: 202.5° - 247.5°
/// - Last Quarter: 247.5° - 292.5°
/// - Waning Crescent: 292.5° - 337.5°
///
pub fn phase_name(time: &Instant) -> MoonPhase {
    let phase_rad = phase(time);
    let phase_deg = phase_rad.to_degrees();

    // Normalize to 0-360 range
    let phase_deg = if phase_deg < 0.0 {
        phase_deg + 360.0
    } else if phase_deg >= 360.0 {
        phase_deg - 360.0
    } else {
        phase_deg
    };

    match phase_deg {
        p if p < 22.5 => MoonPhase::NewMoon,
        p if p < 67.5 => MoonPhase::WaxingCrescent,
        p if p < 112.5 => MoonPhase::FirstQuarter,
        p if p < 157.5 => MoonPhase::WaxingGibbous,
        p if p < 202.5 => MoonPhase::FullMoon,
        p if p < 247.5 => MoonPhase::WaningGibbous,
        p if p < 292.5 => MoonPhase::LastQuarter,
        p if p < 337.5 => MoonPhase::WaningCrescent,
        _ => MoonPhase::NewMoon, // 337.5 - 360
    }
}

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
pub fn pos_gcrf(time: &Instant) -> Vector3 {
    // Julian centuries since Jan 1, 2000 12pm

    let t: f64 = (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0;

    #[allow(non_upper_case_globals)]
    const deg2rad: f64 = std::f64::consts::PI / 180.;

    let lambda_ecliptic: f64 = deg2rad
        * 0.11f64.mul_add(
            -f64::sin(deg2rad * 966404.05f64.mul_add(t, 186.6)),
            0.19f64.mul_add(
                -f64::sin(deg2rad * 35999.05f64.mul_add(t, 357.5)),
                0.21f64.mul_add(
                    f64::sin(deg2rad * 954397.70f64.mul_add(t, 269.9)),
                    0.66f64.mul_add(
                        f64::sin(deg2rad * 890534.23f64.mul_add(t, 235.7)),
                        1.27f64.mul_add(
                            -f64::sin(deg2rad * 413335.38f64.mul_add(-t, 259.2)),
                            6.29f64.mul_add(
                                f64::sin(deg2rad * 477198.85f64.mul_add(t, 134.9)),
                                481267.8813f64.mul_add(t, 218.32),
                            ),
                        ),
                    ),
                ),
            ),
        );

    let phi_ecliptic: f64 = deg2rad
        * 0.17f64.mul_add(
            -f64::sin(deg2rad * 407332.20f64.mul_add(-t, 217.6)),
            0.28f64.mul_add(
                -f64::sin(deg2rad * 6003.18f64.mul_add(t, 318.3)),
                5.13f64.mul_add(
                    f64::sin(deg2rad * 483202.03f64.mul_add(t, 93.3)),
                    0.28 * f64::sin(deg2rad * 960400.87f64.mul_add(t, 228.2)),
                ),
            ),
        );

    let hparallax: f64 = deg2rad
        * 0.0028f64.mul_add(
            f64::cos(deg2rad * 954397.70f64.mul_add(t, 269.9)),
            0.0078f64.mul_add(
                f64::cos(deg2rad * 890534.23f64.mul_add(t, 235.7)),
                0.0095f64.mul_add(
                    f64::cos(deg2rad * 413335.38f64.mul_add(-t, 259.2)),
                    0.0518f64.mul_add(f64::cos(deg2rad * 477198.85f64.mul_add(t, 134.9)), 0.9508),
                ),
            ),
        );

    let epsilon: f64 = deg2rad
        * (5.04E-7 * t * t).mul_add(
            t,
            (1.64e-7 * t).mul_add(-t, 0.0130042f64.mul_add(-t, 23.439291)),
        );

    // Convert values above from degrees to radians
    // for remainder of computations

    let rmag: f64 = consts::EARTH_RADIUS / f64::sin(hparallax);

    rmag * Vector3::new(
        f64::cos(phi_ecliptic) * f64::cos(lambda_ecliptic),
        (f64::cos(epsilon) * f64::cos(phi_ecliptic)).mul_add(
            f64::sin(lambda_ecliptic),
            -(f64::sin(epsilon) * f64::sin(phi_ecliptic)),
        ),
        (f64::sin(epsilon) * f64::cos(phi_ecliptic)).mul_add(
            f64::sin(lambda_ecliptic),
            f64::cos(epsilon) * f64::sin(phi_ecliptic),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonpos() {
        //! This is Vallado example 5-3
        let t0 = Instant::from_date(1994, 4, 28).unwrap();
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

    #[test]
    fn test_moon_ecliptic() {
        // Vallado Example 5-4 (subset)
        let time = Instant::from_datetime(1998, 8, 21, 0, 12, 0.0).unwrap();
        let jd = time.as_jd_with_scale(TimeScale::TDB);
        println!("JD: {}", jd);
        let lambda = ecliptic_longitude(&time);
        let lambda_deg = lambda.to_degrees();
        println!("Ecliptic Longitude: {} degrees", lambda_deg);
        approx::assert_abs_diff_eq!(lambda_deg, -225.05353, epsilon = 0.5);
    }

    #[test]
    fn test_phase() {
        // Check against https://www.timeanddate.com/moon/phases/
        let time = Instant::from_datetime(2025, 11, 12, 0, 46, 0.0).unwrap();
        let phasename = phase_name(&time);
        let illumination = illumination(&time);
        approx::assert_relative_eq!(illumination, 0.52, epsilon = 0.02);
        assert!(phasename == MoonPhase::LastQuarter);

        let time = Instant::from_datetime(2025, 11, 5, 13, 19, 0.0).unwrap();
        let phase_rad = phase(&time);
        let phase_deg = phase_rad.to_degrees();
        println!("Phase degrees: {}", phase_deg);
        approx::assert_relative_eq!(phase_deg, 180.0, epsilon = 0.2);
    }

    #[test]
    fn test_moon_phases() {
        // Test various moon phases throughout a lunar cycle
        // These dates are approximate known moon phases
        // phases compared against results from https://www.moongiant.com/

        // New Moon - January 11, 2024
        let new_moon = Instant::from_datetime(2024, 1, 11, 12, 0, 0.0).unwrap();
        assert_eq!(phase_name(&new_moon), MoonPhase::NewMoon);

        // First Quarter - January 18, 2024
        let first_quarter = Instant::from_datetime(2024, 1, 18, 12, 0, 0.0).unwrap();

        let phase = phase_name(&first_quarter);
        assert!(
            phase == MoonPhase::FirstQuarter
                || phase == MoonPhase::WaxingCrescent
                || phase == MoonPhase::WaxingGibbous,
            "First quarter should be near 90 degrees, got {:?}",
            phase
        );

        // Full Moon - January 25, 2024
        let full_moon = Instant::from_datetime(2024, 1, 25, 12, 0, 0.0).unwrap();
        let phase = phase_name(&full_moon);
        assert!(
            phase == MoonPhase::FullMoon
                || phase == MoonPhase::WaxingGibbous
                || phase == MoonPhase::WaningGibbous,
            "Full moon should be near 180 degrees, got {:?}",
            phase
        );

        // Last Quarter - February 2, 2024
        let last_quarter = Instant::from_datetime(2024, 2, 2, 12, 0, 0.0).unwrap();
        let phase = phase_name(&last_quarter);
        assert!(
            phase == MoonPhase::LastQuarter
                || phase == MoonPhase::WaningGibbous
                || phase == MoonPhase::WaningCrescent,
            "Last quarter should be near 270 degrees, got {:?}",
            phase
        );
    }

    #[test]
    fn test_phase_name_method() {
        // Test that the name() method returns the expected strings
        assert_eq!(MoonPhase::NewMoon.name(), "New Moon");
        assert_eq!(MoonPhase::WaxingCrescent.name(), "Waxing Crescent");
        assert_eq!(MoonPhase::FirstQuarter.name(), "First Quarter");
        assert_eq!(MoonPhase::WaxingGibbous.name(), "Waxing Gibbous");
        assert_eq!(MoonPhase::FullMoon.name(), "Full Moon");
        assert_eq!(MoonPhase::WaningGibbous.name(), "Waning Gibbous");
        assert_eq!(MoonPhase::LastQuarter.name(), "Last Quarter");
        assert_eq!(MoonPhase::WaningCrescent.name(), "Waning Crescent");
    }
}
