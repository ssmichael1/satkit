use crate::consts;
use crate::ITRFCoord;
use crate::Instant;
use crate::TimeScale;

use anyhow::Result;
use nalgebra as na;

type Vec3 = na::Vector3<f64>;

///
/// Sun position in the Geocentric Celestial Reference Frame (GCRF)
///
/// # Arguments
///
///    `time` - Instant at which to compute position
///
/// # Returns
///
/// * Vector representing sun position in GCRF frame
///   at given time.  Units are meters
///
/// # Notes
///
/// * Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
///   from MOD to GCRF via Equations 3-88 and 3-89 in Vallado
///
#[inline]
pub fn pos_gcrf(time: &Instant) -> na::Vector3<f64> {
    crate::frametransform::qmod2gcrf(time) * pos_mod(time)
}

///
/// Sun position in the Mean-of-Date (MOD) Frame
///
/// # Arguments
///
/// * `time` - Instant at which to compute position
///
/// # Returns
///
/// * Vector representing sun position in MOD frame
///    at given time.  Units are meters
///
/// # Notes:
///
/// * Algorithm 29 from Vallado for sun in Mean of Date (MOD)
/// * Valid with accuracy of .01 degrees from 1950 to 2050
///
pub fn pos_mod(time: &Instant) -> na::Vector3<f64> {
    let t: f64 = (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0;
    #[allow(non_upper_case_globals)]
    const deg2rad: f64 = std::f64::consts::PI / 180.;

    // Mean longitude
    let lambda: f64 = 36000.77f64.mul_add(t, 280.46);

    // mean anomaly
    #[allow(non_snake_case)]
    let M: f64 = deg2rad * 35999.05034f64.mul_add(t, 357.5277233);

    // obliquity
    let epsilon: f64 = deg2rad * 0.0130042f64.mul_add(-t, 23.439291);

    // Ecliptic
    let lambda_ecliptic: f64 = deg2rad
        * 0.019994643f64.mul_add(
            f64::sin(2.0 * M),
            1.914666471f64.mul_add(f64::sin(M), lambda),
        );

    // Magnitude of sun vector
    let r: f64 = consts::AU
        * 0.000139589f64.mul_add(
            -f64::cos(2. * M),
            0.016708617f64.mul_add(-f64::cos(M), 1.000140612),
        );

    na::Vector3::<f64>::new(
        r * f64::cos(lambda_ecliptic),
        r * f64::sin(lambda_ecliptic) * f64::cos(epsilon),
        r * f64::sin(lambda_ecliptic) * f64::sin(epsilon),
    )
}

///
/// Fraction of sunlight shadowed by Earth
/// in range \[0, 1\]
///
/// # Arguments:
///
/// * `psun` - Position of sun, meters
/// * `psat` - Position of satellite in same frame as sun position, meters
///
/// # Returns:
///
/// * Fractional amount of sunlight hitting satellite:
///   * 0 = full occlusion
///   * 1 = full sunlight
///
/// # Reference
///
/// * See algorithm in Section 3.4.2 of Montenbruck and Gill for calculation
///
///
pub fn shadowfunc(psun: &Vec3, psat: &Vec3) -> f64 {
    let a = (consts::SUN_RADIUS / (psun - psat).norm()).asin();
    let b = (consts::EARTH_RADIUS / psat.norm()).asin();
    let snorm = psat.norm();
    let c = (-psat.dot(&(psun - psat)) / snorm / (psun - psat).norm()).acos();
    if a + b <= c {
        1.0
    } else if c < (b - a) {
        0.0
    } else {
        let x = b.mul_add(-b, c.mul_add(c, a * a)) / 2.0 / c;
        let y = a.mul_add(a, -(x * x)).sqrt();
        let big_a = c.mul_add(
            -y,
            (a * a).mul_add((x / a).acos(), b * b * ((c - x) / b).acos()),
        );

        1.0 - big_a / std::f64::consts::PI / a / a
    }
}

///
/// # Compute sunrise and sunset
///
/// Sunrise and sunset times on the day given by input time
/// and at the given location.  
///
/// Since sunrise and sunset are local, the input time will have its
/// local hour angle subtracted off to compute the sunrise and sunset
/// at the date of the input time locally
///
/// For example, the time 2020-08-20 00:00:00.000Z is actually a date of
/// 2020-08-19 in local time of Boston, Ma.  The time will be shifted such
/// that the sunrise and sunset times are computed for 2020-08-20 in Boston.
///
///
/// Will return an error if the sun does not rise or set on the given date
/// at given location (e.g., Alaska in summer)
///
/// # Input Arguments
///
/// * `time`  - Date at which to compute sunrise & sunset
///
/// * `coord` - ITRFCoord representing location for which to compute
///             sunrise & sunset
///
/// * `sigma` - Angle in degrees between noon & rise/set
///    Common Values:
///    * "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
///    * "Civil Twilight": 96 deg
///    * "Nautical Twilight": 102 deg
///    * "Astronomical Twilight": 108 deg
///      
/// If None is passed in, "Standard" is used (90.0 + 50.0/60.0)
///
/// # Returns
///
/// * Result<(sunrise: Instant, sunset: Instant)>
///
/// # References
///
/// * Vallado Algorithm 30
///
pub fn riseset(
    time: &Instant,
    coord: &ITRFCoord,
    osigma: Option<f64>,
) -> Result<(Instant, Instant)> {
    use std::f64::consts::PI;
    let sigma = osigma.unwrap_or(90.0 + 50.0 / 60.0);
    let latitude: f64 = coord.latitude_deg();
    let longitude: f64 = coord.longitude_deg();

    let sind: fn(f64) -> f64 = |x: f64| x.to_radians().sin();
    let cosd: fn(f64) -> f64 = |x: f64| x.to_radians().cos();
    const RAD2DEG: f64 = 180.0 / PI;

    // Zero-hour GMST, equation 3-45 in Vallado
    let gmst0h = |t: f64| -> f64 {
        (2.6E-8 * t * t).mul_add(
            -t,
            (0.00038793 * t).mul_add(t, 36000.77005361f64.mul_add(t, 100.4606184)),
        ) % 360.0
    };

    let jd0h: f64 = (time.as_jd_with_scale(TimeScale::UTC) - longitude / 360.0).floor() + 0.5;
    let jdsunrise = jd0h + 0.25 - longitude / 360.0;
    let jdsunset = jd0h + 0.75 - longitude / 360.0;

    let criseset = |jd: f64, lhafunc: fn(f64) -> f64| -> Result<f64> {
        let t = (jd - 2451545.0) / 36525.0;

        let lambda_sun = 36000.77005361f64.mul_add(t, 280.4606184);
        let msun = 35999.05034f64.mul_add(t, 357.5291092);
        let lambda_ecliptic = 0.019994643f64.mul_add(
            sind(2.0 * msun),
            1.914666471f64.mul_add(sind(msun), lambda_sun),
        );
        // Longitude in ecliptl
        let epsilon = 0.0130042f64.mul_add(-t, 23.439291);

        let sindelta_sun = sind(epsilon) * sind(lambda_ecliptic);
        let deltasun = f64::asin(sindelta_sun) * RAD2DEG;
        //let alpha_sun = f64::atan(tanalpha_sun) * RAD2DEG;
        let alpha_sun =
            f64::atan2(cosd(epsilon) * sind(lambda_ecliptic), cosd(lambda_ecliptic)) * RAD2DEG;

        let coslha = sind(deltasun).mul_add(-sind(latitude), cosd(sigma))
            / (cosd(deltasun) * cosd(latitude));
        if coslha.abs() > 1.0 {
            anyhow::bail!(
                "Invalid position.  Sun doesn't rise/set on this day at \
                 this location (e.g., Alaska in summer)"
            );
        }
        let mut lha = f64::acos(coslha) * RAD2DEG;

        lha = lhafunc(lha);
        let gmst = gmst0h(t) % 360.0;
        let mut ret = (lha + alpha_sun - gmst) % 360.0;
        if ret < 0.0 {
            ret += 360.0;
        }
        Ok(ret / 360.0)
    };

    Ok((
        Instant::from_jd(jdsunrise + criseset(jdsunrise, |x| 360.0 - x)? - 0.25),
        Instant::from_jd(jdsunset + criseset(jdsunset, |x| x)? - 0.75),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunpos_mod() {
        // Example 5-1 in Vallado
        let t0: Instant = Instant::from_date(2006, 4, 2);
        // Approximate this UTC as TDB to match example...
        let t = Instant::from_mjd_with_scale(t0.as_mjd_with_scale(TimeScale::UTC), TimeScale::TDB);

        let pos = pos_mod(&t);
        // Below value is from Vallado example
        let ref_pos = [146186212.0E3, 28788976.0E3, 12481064.0E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            assert!(err < 1.0e-6);
        }
    }

    #[test]
    fn sunpos_gcrf() {
        // Example 5-1 in Vallado
        let t0: Instant = Instant::from_date(2006, 4, 2);
        // Approximate this UTC as TDB to match example...
        let t = Instant::from_mjd_with_scale(t0.as_mjd(), TimeScale::TDB);

        let pos = pos_gcrf(&t);
        // Below value is from Vallado example
        let ref_pos = [146259922.0E3, 28585947.0E3, 12397430.0E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            // Less exact here because we are comparing to JPL ephemeris.
            // as described by Vallado
            assert!(err < 5e-4);
        }
    }

    #[test]
    fn sunriseset() {
        // Example 5-2 from Vallado
        let itrf = ITRFCoord::from_geodetic_deg(40.0, 0.0, 0.0);
        let tm = Instant::from_datetime(1996, 3, 23, 0, 0, 0.0);
        let (sunrise, sunset) = riseset(&tm, &itrf, None).unwrap();
        let (ryear, rmon, rday, rhour, rmin, rsec) = sunrise.as_datetime();
        assert!(ryear == 1996);
        assert!(rmon == 3);
        assert!(rday == 23);
        assert!(rhour == 5);
        assert!(rmin == 58);
        assert!((rsec / 21.97 - 1.0).abs() < 1.0e-3);
        let (syear, smon, sday, shour, smin, ssec) = sunset.as_datetime();
        assert!(syear == 1996);
        assert!(smon == 3);
        assert!(sday == 23);
        assert!(shour == 18);
        assert!(smin == 15);
        assert!((ssec / 17.76 - 1.0).abs() < 1.0e-3);

        // Check for error returned on 24-hour sunlight condition
        let itrf2 = ITRFCoord::from_geodetic_deg(85.0, 30.0, 0.0);
        let tm2 = Instant::from_date(2020, 6, 20);
        let r = riseset(&tm2, &itrf2, None);
        assert!(r.is_err());
    }

    #[test]
    fn test_webexample() {
        let coord = ITRFCoord::from_geodetic_deg(42.4154, -71.1565, 0.0);
        let time = &Instant::from_date(2024, 10, 14);

        let (rise, set) = riseset(time, &coord, None).unwrap();

        // Check against web example
        // https://www.timeanddate.com/sun/@4929180
        let rise_web = Instant::from_datetime(2024, 10, 14, 10, 57, 0.0);
        let set_web = Instant::from_datetime(2024, 10, 14, 22, 4, 0.0);

        assert!((rise - rise_web).as_seconds().abs() < 60.0);
        assert!((set - set_web).as_seconds().abs() < 60.0);
    }
}
