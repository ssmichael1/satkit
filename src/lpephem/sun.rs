use crate::consts;
use crate::ITRFCoord;
use crate::Instant;
use crate::TimeLike;
use crate::TimeScale;

use super::{Error, Result};
use crate::mathtypes::*;

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
pub fn pos_gcrf<T: TimeLike>(time: &T) -> Vector3 {
    let time = time.as_instant();
    crate::frametransform::qmod2gcrf(&time) * pos_mod(&time)
}

/// Ecliptic longitude of the sun at given time
///
/// # Arguments
///
/// * `time` - Instant at which to compute sun ecliptic longitude
///
/// Returns:
///
/// * Ecliptic longitude of sun, radians
///
/// # Notes
///
/// See Vallado Algorithm 29
///
pub fn ecliptic_longitude<T: TimeLike>(time: &T) -> f64 {
    let time = time.as_instant();
    // Julian centuries since Jan 1, 2000 12pm
    let t: f64 = (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0;

    // mean anomaly
    #[allow(non_snake_case)]
    let M: f64 = (35999.05034f64.mul_add(t, 357.529102)).to_radians();

    let lambda_m: f64 = (36000.771f64.mul_add(t, 280.46)).to_radians();

    let lon: f64 = 0.019994643f64.mul_add(
        f64::sin(2.0 * M),
        1.914666471f64.mul_add(f64::sin(M), lambda_m.to_degrees()),
    );

    lon.to_radians() % std::f64::consts::TAU
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
///   at given time.  Units are meters
///
/// # Notes:
///
/// * Algorithm 29 from Vallado for sun in Mean of Date (MOD)
/// * Valid with accuracy of .01 degrees from 1950 to 2050
///
pub fn pos_mod<T: TimeLike>(time: &T) -> Vector3 {
    let time = time.as_instant();
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

    numeris::vector![
        r * f64::cos(lambda_ecliptic),
        r * f64::sin(lambda_ecliptic) * f64::cos(epsilon),
        r * f64::sin(lambda_ecliptic) * f64::sin(epsilon),
    ]
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
/// # Notes
///
/// * Beyond ~1.4 million km on the anti-Sun side (e.g. near Sun-Earth L2)
///   the Earth's disc is smaller than the Sun's, and a satellite on the
///   shadow axis sees an annular eclipse: the fraction is `1 - b²/a²`,
///   with `a` and `b` the apparent radii of the Sun and the Earth
/// * A position at or below the Earth's (spherical) surface is lit when the
///   Sun is above its local horizon plane and shadowed otherwise; the
///   Earth's center returns 0
///
/// # Reference
///
/// * See algorithm in Section 3.4.2 of Montenbruck and Gill for calculation
///
///
pub fn shadowfunc(psun: &Vector3, psat: &Vector3) -> f64 {
    let snorm = psat.norm();
    let dsun = psun - psat;
    let dnorm = dsun.norm();
    if snorm == 0.0 {
        return 0.0;
    }
    if dnorm == 0.0 {
        return 1.0;
    }
    // Apparent radii of the Sun (a) and the Earth (b), and their apparent
    // separation (c).  The clamps keep asin / acos in range at or below the
    // Earth's surface and against rounding.
    let a = (consts::SUN_RADIUS / dnorm).min(1.0).asin();
    let b = (consts::EARTH_RADIUS / snorm).min(1.0).asin();
    let c = (-psat.dot(&dsun) / snorm / dnorm).clamp(-1.0, 1.0).acos();
    if a + b <= c {
        // No occultation
        1.0
    } else if c <= b - a {
        // Total: the Earth's disc covers the Sun's
        0.0
    } else if c <= a - b {
        // Annular: the Earth's disc lies entirely inside the Sun's
        1.0 - (b * b) / (a * a)
    } else {
        // Partial; here c > |a - b| >= 0
        let x = b.mul_add(-b, c.mul_add(c, a * a)) / 2.0 / c;
        let y = a.mul_add(a, -(x * x)).max(0.0).sqrt();
        let big_a = c.mul_add(
            -y,
            (a * a).mul_add(
                (x / a).clamp(-1.0, 1.0).acos(),
                b * b * ((c - x) / b).clamp(-1.0, 1.0).acos(),
            ),
        );

        (1.0 - big_a / std::f64::consts::PI / a / a).clamp(0.0, 1.0)
    }
}

///
/// # Compute sunrise and sunset
///
/// Sunrise and sunset times on a calendar date at the given location.
///
/// The input time selects the date: its **UTC calendar date** is used and
/// its time of day is ignored.  The returned sunrise and sunset are those
/// of that date at the location's longitude, i.e. between the local
/// (mean solar) midnights that begin and end that date there.  Both are
/// returned as UTC instants, so at far-west longitudes the sunset may fall
/// on the next UTC date, and at far-east longitudes the sunrise may fall on
/// the previous one.
///
/// For example, any time on 2024-10-14 UTC gives the sunrise and sunset of
/// 2024-10-14 in Honolulu: sunrise at about 16:40 UTC on 2024-10-14 and
/// sunset at about 04:25 UTC on 2024-10-15 (06:40 and 18:25 local time).
///
/// To get the events of a *local* date, pass a time at local noon on that
/// date: for time zones within UTC-11 to UTC+11 local noon falls on the
/// same UTC date, while local midnight falls on the previous UTC date east
/// of Greenwich.
///
/// Returns [`Error::NoSunriseOrSunset`] if the Sun stays above the threshold
/// all day (polar day) or below it all day (polar night).  Near those
/// thresholds an event is returned only if the Sun reaches the threshold
/// at the time of the event.
///
/// # Accuracy
///
/// * Algorithm 30 evaluates the Sun once, at 6h (rise) or 18h (set) local
///   mean time.  Here that pass is repeated at the computed event until it
///   moves less than 0.1 s, and the main nutation term and the solar
///   parallax (8.8") are included.  Against Skyfield with the DE421
///   ephemeris, over 2024 at latitudes 60 S to 65 N, the times agree to
///   within 3 s, and typically about 1 s (the single pass was off by up
///   to 35 s at 65 N).  The remainder is the low-precision solar series,
///   which is up to ~20" off in longitude.  Errors grow near the polar-day
///   and polar-night thresholds, where the Sun meets the threshold at a
///   grazing angle (up to ~10 s at 67 to 72 N).
/// * UTC is used in place of UT1 (they differ by less than 0.9 s).
/// * The horizon is at sea level: the observer's altitude is ignored.  An
///   elevated observer sees a horizon lowered by the dip,
///   dip ≈ 1.76' × √h, with h the height in meters above the surrounding
///   terrain or sea (this includes typical terrestrial refraction), which
///   makes sunrise earlier and sunset later.  To account for it, pass
///   `sigma = 90° 50' + dip`, e.g. `90.0 + (50.0 + 1.76 * h.sqrt()) / 60.0`;
///   at 100 m this moves each event by 1.5 to 3 minutes at latitudes 30 to
///   55 deg.
///
/// # Input Arguments
///
/// * `time`  - Time whose UTC calendar date selects the day (time of day
///   is ignored)
///
/// * `coord` - ITRFCoord representing location for which to compute
///   sunrise & sunset
///
/// * `sigma` - Angle in degrees between noon & rise/set
///   Common Values:
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
/// * Meeus, "Astronomical Algorithms", ch. 25 (nutation terms)
///
pub fn riseset<T: TimeLike>(
    time: &T,
    coord: &ITRFCoord,
    osigma: Option<f64>,
) -> Result<(Instant, Instant)> {
    let time = time.as_instant();
    use std::f64::consts::PI;
    let sigma = osigma.unwrap_or(90.0 + 50.0 / 60.0);
    let latitude: f64 = coord.latitude_deg();
    let longitude: f64 = coord.longitude_deg();

    let sind: fn(f64) -> f64 = |x: f64| x.to_radians().sin();
    let cosd: fn(f64) -> f64 = |x: f64| x.to_radians().cos();

    // `sigma` is seen from the site; the Sun's direction below is
    // geocentric.  Solar parallax (8.794") puts the geocentric Sun that
    // much higher, which moves rise & set by several seconds at high latitude.
    let sigma = sigma - 8.794 / 3600.0 * sind(sigma);
    const RAD2DEG: f64 = 180.0 / PI;

    // Zero-hour GMST, equation 3-45 in Vallado
    let gmst0h = |t: f64| -> f64 {
        (2.6E-8 * t * t).mul_add(
            -t,
            (0.00038793 * t).mul_add(t, 36000.77005361f64.mul_add(t, 100.4606184)),
        ) % 360.0
    };

    // Local mean midnight that begins the input's UTC calendar date at the
    // site's longitude.  Each event is this plus its local mean time as a
    // fraction of a day in [0, 1), so it stays on that date.
    let (year, month, day, _, _, _) = time.as_datetime();
    let jd0h: f64 = Instant::from_date(year, month, day)?.as_jd_with_scale(TimeScale::UTC);
    let jdbase = jd0h - longitude / 360.0;

    // One pass of Algorithm 30, with the Sun and the "GMST" term evaluated
    // at UTC Julian date `jd`: the local mean time of the event as a
    // fraction of a day, or `Err(cos(LHA))` when |cos(LHA)| > 1, i.e. the
    // Sun doesn't reach the threshold that day
    let pass = |jd: f64, rising: bool| -> std::result::Result<f64, f64> {
        let t = (jd - 2451545.0) / 36525.0;

        let lambda_sun = 36000.77005361f64.mul_add(t, 280.4606184);
        let msun = 35999.05034f64.mul_add(t, 357.5291092);
        let lambda_ecliptic = 0.019994643f64.mul_add(
            sind(2.0 * msun),
            1.914666471f64.mul_add(sind(msun), lambda_sun),
        );
        // Longitude in ecliptic coordinates
        let epsilon = 0.0130042f64.mul_add(-t, 23.439291);

        // Nutation, main term (Meeus, "Astronomical Algorithms", ch. 25),
        // for the true equinox & obliquity; the longitude above already
        // includes annual aberration.  Nutation in obliquity (up to 9")
        // moves rise & set by several seconds at high latitude.
        let omega = 1934.136f64.mul_add(-t, 125.04);
        let dpsi = -0.00478 * sind(omega);
        let lambda_ecliptic = lambda_ecliptic + dpsi;
        let epsilon = 0.00256f64.mul_add(cosd(omega), epsilon);

        let sindelta_sun = sind(epsilon) * sind(lambda_ecliptic);
        let deltasun = f64::asin(sindelta_sun) * RAD2DEG;
        //let alpha_sun = f64::atan(tanalpha_sun) * RAD2DEG;
        // atan2 handles quadrant correctly ... very important!
        let alpha_sun =
            f64::atan2(cosd(epsilon) * sind(lambda_ecliptic), cosd(lambda_ecliptic)) * RAD2DEG;

        let coslha = sind(deltasun).mul_add(-sind(latitude), cosd(sigma))
            / (cosd(deltasun) * cosd(latitude));
        if coslha.abs() > 1.0 {
            return Err(coslha);
        }
        let mut lha = f64::acos(coslha) * RAD2DEG;
        if rising {
            lha = 360.0 - lha;
        }

        // Apparent sidereal angle: add the equation of the equinoxes
        let gmst = dpsi.mul_add(cosd(epsilon), gmst0h(t)) % 360.0;
        let mut ret = (lha + alpha_sun - gmst) % 360.0;
        if ret < 0.0 {
            ret += 360.0;
        }
        Ok(ret / 360.0)
    };

    // Algorithm 30 evaluates the Sun at a first guess of 6h (rise) or 18h
    // (set) local mean time, which costs up to ~35 s at 65 deg latitude.
    // Re-evaluate it at the computed event until the event moves < 0.1 s.
    let event = |guess: f64, rising: bool| -> Result<Instant> {
        let mut frac = match pass(jdbase + guess, rising) {
            Ok(frac) => frac,
            // No crossing at the guess.  Near polar night (cos(LHA) > 1) any
            // rise & set straddle noon; near polar day (cos(LHA) < -1) they
            // straddle midnight.  Decide there.
            Err(coslha) => {
                let retry = if coslha > 1.0 {
                    0.5
                } else if rising {
                    0.0
                } else {
                    1.0
                };
                pass(jdbase + retry, rising).map_err(|_| Error::NoSunriseOrSunset)?
            }
        };
        for _ in 0..10 {
            // The Sun doesn't reach the threshold at the time the event would
            // occur, so the event doesn't happen that day
            let next = pass(jdbase + frac, rising).map_err(|_| Error::NoSunriseOrSunset)?;
            let change = (next - frac).abs();
            frac = next;
            if change < 0.1 / 86400.0 {
                break;
            }
        }
        Ok(Instant::from_jd_utc(jdbase + frac))
    };

    Ok((event(0.25, true)?, event(0.75, false)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunpos_mod() {
        // Example 5-1 in Vallado
        let t0: Instant = Instant::from_date(2006, 4, 2).unwrap();
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
        let t0: Instant = Instant::from_date(2006, 4, 2).unwrap();
        // Approximate this UTC as TDB to match example...
        let t = Instant::from_mjd_with_scale(t0.as_mjd_utc(), TimeScale::TDB);

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
    fn test_ecliptic_longitude() {
        // Example 5-1 in Vallado
        let t0: Instant = Instant::from_date(2006, 4, 2).unwrap();
        // Approximate this UTC as TDB to match example...
        let t = Instant::from_mjd_with_scale(t0.as_mjd_utc(), TimeScale::TDB);

        let lambda = ecliptic_longitude(&t);
        let lambda_deg = lambda.to_degrees();
        let ref_lambda_deg = 12.114404; // Vallado example
        approx::assert_abs_diff_eq!(lambda_deg, ref_lambda_deg, epsilon = 0.5);
    }

    #[test]
    fn sunriseset() {
        // Example 5-2 from Vallado
        let itrf = ITRFCoord::from_geodetic_deg(40.0, 0.0, 0.0);
        let tm = Instant::from_datetime(1996, 3, 23, 0, 0, 0.0).unwrap();
        let (sunrise, sunset) = riseset(&tm, &itrf, None).unwrap();
        // The book's values are from a single pass of Algorithm 30, without
        // nutation or parallax; the refined times differ by about a second
        let rise_book = Instant::from_datetime(1996, 3, 23, 5, 58, 21.97).unwrap();
        let set_book = Instant::from_datetime(1996, 3, 23, 18, 15, 17.76).unwrap();
        let (drise, dset) = (
            (sunrise - rise_book).as_seconds(),
            (sunset - set_book).as_seconds(),
        );
        println!("Vallado example 5-2: rise {drise:+.2} s, set {dset:+.2} s");
        assert!(drise.abs() < 2.0, "{sunrise}");
        assert!(dset.abs() < 2.0, "{sunset}");

        // Check for error returned on 24-hour sunlight condition
        let itrf2 = ITRFCoord::from_geodetic_deg(85.0, 30.0, 0.0);
        let tm2 = Instant::from_date(2020, 6, 20).unwrap();
        let r = riseset(&tm2, &itrf2, None);
        assert!(r.is_err());
    }

    #[test]
    fn test_webexample() {
        let coord = ITRFCoord::from_geodetic_deg(42.4154, -71.1565, 0.0);
        let time = &Instant::from_date(2024, 10, 14).unwrap();

        let (rise, set) = riseset(time, &coord, None).unwrap();

        // Check against web example
        // https://www.timeanddate.com/sun/@4929180
        let rise_web = Instant::from_datetime(2024, 10, 14, 10, 57, 0.0).unwrap();
        let set_web = Instant::from_datetime(2024, 10, 14, 22, 4, 0.0).unwrap();

        assert!((rise - rise_web).as_seconds().abs() < 60.0);
        assert!((set - set_web).as_seconds().abs() < 60.0);
    }

    /// Geometric elevation of the Sun's center, degrees
    fn sun_elevation(t: &Instant, coord: &ITRFCoord) -> f64 {
        let s = crate::frametransform::qgcrf2itrf(t) * pos_gcrf(t) - coord.itrf;
        let enu = coord.q_enu2itrf().conjugate() * s;
        (enu[2] / enu.norm()).asin().to_degrees()
    }

    #[test]
    fn riseset_uses_utc_date() {
        // Any time of day on 2024-10-14 UTC selects 2024-10-14; the old day
        // selection returned the next day's events for about half the inputs
        for (lat, lon, rise_utc, set_utc) in [
            // Greenwich: rise ~06:20, set ~17:12 UTC on the 14th
            (51.48, 0.0, (14, 6), (14, 17)),
            // Honolulu (UTC-10): rise ~06:40 and set ~18:25 local time on
            // the 14th, which is 16:40 on the 14th and 04:25 on the 15th UTC
            (21.31, -157.86, (14, 16), (15, 4)),
        ] {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            let (rise0, set0) =
                riseset(&Instant::from_date(2024, 10, 14).unwrap(), &coord, None).unwrap();
            for hour in [0, 6, 12, 18, 23] {
                let t = Instant::from_datetime(2024, 10, 14, hour, 59, 59.0).unwrap();
                let (rise, set) = riseset(&t, &coord, None).unwrap();
                assert_eq!((rise, set), (rise0, set0), "lon {lon} input {t}");
            }
            let (_, _, rday, rhour, _, _) = rise0.as_datetime();
            let (_, _, sday, shour, _, _) = set0.as_datetime();
            assert_eq!((rday, rhour), rise_utc, "lon {lon} rise {rise0}");
            assert_eq!((sday, shour), set_utc, "lon {lon} set {set0}");
            // Standard rise/set: the Sun's center is 50' below the horizon
            for t in [rise0, set0] {
                let el = sun_elevation(&t, &coord);
                assert!((el + 50.0 / 60.0).abs() < 0.1, "lon {lon} {t}: {el}");
            }
        }
    }

    /// Sunrise & sunset in 2024 from Skyfield 1.55 (`almanac.find_risings` /
    /// `find_settings`) with the DE421 ephemeris: the topocentric apparent
    /// Sun (light time & aberration, no light deflection) crossing
    /// `horizon_degrees = -50/60` (34' refraction + 16' semidiameter,
    /// Skyfield's own value for the Sun, passed explicitly) at a sea-level
    /// `wgs84.latlon(lat, lon)` site.  The search window is this function's
    /// day: the local mean midnights that begin and end the UTC date.
    ///
    /// Columns: latitude, longitude (deg), month, day, then sunrise and
    /// sunset in seconds of UTC after 0h UTC on that date, to 0.1 s.  The
    /// generating script is in the description of PR #268.
    #[rustfmt::skip]
    const SKYFIELD_2024: [(f64, f64, i32, i32, f64, f64); 64] = [
        (0.0, -75.0, 3, 20, 39839.8, 83429.6),
        (0.0, -75.0, 6, 20, 39484.4, 83125.7),
        (0.0, -75.0, 9, 22, 38953.1, 82541.3),
        (0.0, -75.0, 12, 21, 39278.7, 82928.5),
        (30.0, -75.0, 3, 20, 39790.8, 83506.3),
        (30.0, -75.0, 6, 20, 35962.6, 86647.6),
        (30.0, -75.0, 9, 22, 38917.9, 82549.2),
        (30.0, -75.0, 12, 21, 42717.5, 79489.8),
        (45.0, -75.0, 3, 20, 39725.9, 83591.7),
        (45.0, -75.0, 6, 20, 33191.9, 89418.5),
        (45.0, -75.0, 9, 22, 38863.0, 82584.2),
        (45.0, -75.0, 12, 21, 45325.1, 76882.3),
        (55.0, -75.0, 3, 20, 39646.9, 83691.6),
        (55.0, -75.0, 6, 20, 30030.4, 92580.1),
        (55.0, -75.0, 9, 22, 38794.0, 82632.8),
        (55.0, -75.0, 12, 21, 48204.0, 74003.6),
        (60.0, -75.0, 3, 20, 39586.4, 83767.1),
        (60.0, -75.0, 6, 20, 27342.9, 95267.8),
        (60.0, -75.0, 9, 22, 38740.4, 82671.8),
        (60.0, -75.0, 12, 21, 50538.9, 71668.7),
        (65.0, -75.0, 3, 20, 39500.6, 83873.3),
        (65.0, -75.0, 6, 20, 21638.7, 100973.8),
        (65.0, -75.0, 9, 22, 38664.0, 82728.2),
        (65.0, -75.0, 12, 21, 54661.0, 67546.7),
        (-35.0, -75.0, 3, 20, 39817.8, 83418.1),
        (-35.0, -75.0, 6, 20, 43665.9, 78944.1),
        (-35.0, -75.0, 9, 22, 38914.3, 82613.1),
        (-35.0, -75.0, 12, 21, 34974.0, 87232.9),
        (-55.0, -75.0, 3, 20, 39736.4, 83464.6),
        (-55.0, -75.0, 6, 20, 48407.7, 74202.2),
        (-55.0, -75.0, 9, 22, 38815.7, 82746.5),
        (-55.0, -75.0, 12, 21, 29822.3, 92383.9),
        (0.0, 139.7, 3, 20, -11677.6, 31912.3),
        (0.0, 139.7, 6, 20, -12051.3, 31590.0),
        (0.0, 139.7, 9, 22, -12562.2, 31025.9),
        (0.0, 139.7, 12, 21, -12267.1, 31382.7),
        (30.0, 139.7, 3, 20, -11693.9, 31956.3),
        (30.0, 139.7, 6, 20, -15572.6, 35111.7),
        (30.0, 139.7, 9, 22, -12629.6, 31066.0),
        (30.0, 139.7, 12, 21, -8828.4, 27943.9),
        (45.0, 139.7, 3, 20, -11735.0, 32017.8),
        (45.0, 139.7, 6, 20, -18342.9, 37882.6),
        (45.0, 139.7, 9, 22, -12708.1, 31124.4),
        (45.0, 139.7, 12, 21, -6220.9, 25336.3),
        (55.0, 139.7, 3, 20, -11789.8, 32093.3),
        (55.0, 139.7, 6, 20, -21503.5, 41044.2),
        (55.0, 139.7, 9, 22, -12801.0, 31196.8),
        (55.0, 139.7, 12, 21, -3342.1, 22457.4),
        (60.0, 139.7, 3, 20, -11833.2, 32151.6),
        (60.0, 139.7, 6, 20, -24190.0, 43731.9),
        (60.0, 139.7, 9, 22, -12871.6, 31252.6),
        (60.0, 139.7, 12, 21, -1007.2, 20122.4),
        (65.0, 139.7, 3, 20, -11895.7, 32234.4),
        (65.0, 139.7, 6, 20, -29888.0, 49438.1),
        (65.0, 139.7, 9, 22, -12971.1, 31332.0),
        (65.0, 139.7, 12, 21, 3114.8, 16000.3),
        (-35.0, 139.7, 3, 20, -11739.2, 31940.3),
        (-35.0, 139.7, 6, 20, -7870.4, 27408.6),
        (-35.0, 139.7, 9, 22, -12562.1, 31058.7),
        (-35.0, 139.7, 12, 21, -16571.5, 35687.4),
        (-55.0, 139.7, 3, 20, -11861.9, 32027.8),
        (-55.0, 139.7, 6, 20, -3129.3, 22667.1),
        (-55.0, 139.7, 9, 22, -12620.3, 31151.4),
        (-55.0, 139.7, 12, 21, -21722.6, 40839.1),
    ];

    /// Against Skyfield over latitudes 0 to 65 N, 35 S and 55 S, at a
    /// western and an eastern longitude, at the 2024 equinoxes & solstices.
    /// Includes 65 N at the June solstice, where the Sun is below -50' for
    /// only about two hours around local midnight.
    #[test]
    fn riseset_vs_skyfield() {
        let mut worst = 0.0f64;
        for (lat, lon, month, day, rise, set) in SKYFIELD_2024 {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            let t0 = Instant::from_date(2024, month, day).unwrap();
            let (r, s) = riseset(&t0, &coord, None).unwrap();
            let dr = (r - t0).as_seconds() - rise;
            let ds = (s - t0).as_seconds() - set;
            println!("{lat:5} {lon:6} {month:2}/{day:2}: rise {dr:+5.2} s, set {ds:+5.2} s");
            worst = worst.max(dr.abs()).max(ds.abs());
        }
        println!("max |riseset - Skyfield| = {worst:.2} s");
        // Before the refinement, parallax & nutation: 13.9 s (65 N, December)
        assert!(worst < 1.5, "max error vs Skyfield {worst} s");
    }

    /// Against the US Naval Observatory's rise/set service (sea level),
    /// which gives times to the minute.  Fetched 2026-09-26 from
    /// `https://aa.usno.navy.mil/api/rstt/oneday?date=<date>&coords=<lat>,<lon>&tz=<tz>&dst=false`.
    #[test]
    fn riseset_vs_usno() {
        // (date, lat, lon, tz (h), USNO rise & set, local time)
        for ((year, month, day), lat, lon, tz, (rh, rm), (sh, sm)) in [
            ((2024, 6, 20), 38.8895, -77.0353, -5, (4, 43), (19, 37)), // Washington
            ((2024, 12, 21), 35.6762, 139.6503, 9, (6, 47), (16, 32)), // Tokyo
            ((2024, 6, 20), -33.8688, 151.2093, 10, (7, 0), (16, 54)), // Sydney
            ((2024, 12, 21), 64.1466, -21.9426, 0, (11, 23), (15, 30)), // Reykjavik
            ((2024, 9, 22), -0.1807, -78.4678, -5, (6, 3), (18, 10)),  // Quito
            ((2024, 3, 20), 64.8378, -147.7164, -9, (6, 49), (19, 9)), // Fairbanks
            ((2024, 10, 14), 21.3069, -157.8583, -10, (6, 27), (18, 7)), // Honolulu
        ] {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            // The local date is the UTC date that selects the day here
            let t0 = Instant::from_date(year, month, day).unwrap();
            let (r, s) = riseset(&t0, &coord, None).unwrap();
            for (t, h, m) in [(r, rh, rm), (s, sh, sm)] {
                let usno = t0 + crate::Duration::from_hours((h - tz) as f64 + m as f64 / 60.0);
                let err = (t - usno).as_seconds();
                assert!(
                    err.abs() < 60.0,
                    "{year}-{month}-{day} {lat} {lon}: {t} vs {usno}"
                );
            }
        }
    }

    #[test]
    fn riseset_polar_day_and_night() {
        for lon in [-75.0, 139.7] {
            // Polar day: at 66 N the Sun's center stays above -50' at the
            // June solstice (lowest ~ -0.56 deg) ...
            let coord = ITRFCoord::from_geodetic_deg(66.0, lon, 0.0);
            let t = Instant::from_date(2024, 6, 20).unwrap();
            assert!(matches!(
                riseset(&t, &coord, None),
                Err(Error::NoSunriseOrSunset)
            ));
            // ... and polar night at 70 N at the December solstice
            // (highest ~ -3.4 deg)
            let coord = ITRFCoord::from_geodetic_deg(70.0, lon, 0.0);
            let t = Instant::from_date(2024, 12, 21).unwrap();
            assert!(matches!(
                riseset(&t, &coord, None),
                Err(Error::NoSunriseOrSunset)
            ));
        }
        let msg = Error::NoSunriseOrSunset.to_string();
        assert!(
            msg.contains("polar day") && msg.contains("polar night"),
            "{msg}"
        );
    }

    /// Every day of 2024 at far-west and far-east sites, from mid-latitudes
    /// through the polar-day and polar-night thresholds: the events stay on
    /// the documented day, and are real -50' crossings
    #[test]
    fn riseset_stays_on_day() {
        for lon in [-179.5, 179.5] {
            for lat in [40.0, 64.0, 65.5, 66.5, 67.5, 69.0] {
                let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
                let t0 = Instant::from_date(2024, 1, 1).unwrap();
                let mut nevents = 0;
                for iday in 0..366 {
                    let t = t0 + crate::Duration::from_days(iday as f64);
                    let Ok((rise, set)) = riseset(&t, &coord, None) else {
                        assert!(lat > 64.0, "{lat} {lon} {t}");
                        continue;
                    };
                    nevents += 1;
                    // Local mean midnight to midnight
                    let base = t - crate::Duration::from_days(lon / 360.0);
                    for ev in [rise, set] {
                        let dt = (ev - base).as_days();
                        assert!((0.0..1.0).contains(&dt), "{lat} {lon} {t}: {ev}");
                        let el = sun_elevation(&ev, &coord);
                        assert!((el + 50.0 / 60.0).abs() < 0.02, "{lat} {lon} {ev}: {el}");
                    }
                }
                assert!(nevents > 250, "{lat} {lon}: {nevents}");
            }
        }
    }

    #[test]
    fn shadowfunc_edge_cases() {
        let psun = numeris::vector![consts::AU, 0.0, 0.0];
        let apparent = |d: f64, r: f64| (r / d).asin();

        // Anti-Sun axis beyond the umbra (~1.38e6 km): annular eclipse,
        // 1 - b^2/a^2 (Montenbruck & Gill 3.4.2).  Used to return NaN.
        for d_km in [1.5e6, 2.0e6] {
            let psat = numeris::vector![-d_km * 1.0e3, 0.0, 0.0];
            let a = apparent(consts::AU + d_km * 1.0e3, consts::SUN_RADIUS);
            let b = apparent(d_km * 1.0e3, consts::EARTH_RADIUS);
            let expected = 1.0 - (b / a).powi(2);
            approx::assert_relative_eq!(shadowfunc(&psun, &psat), expected, max_relative = 1e-12);
            assert!(expected > 0.1 && expected < 0.6);
            // Continuous with the partial branch just off the axis, and
            // never NaN off-axis
            let mut last = expected;
            for off_km in [1.0, 100.0, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0] {
                let psat = numeris::vector![-d_km * 1.0e3, off_km * 1.0e3, 0.0];
                let f = shadowfunc(&psun, &psat);
                assert!((0.0..=1.0).contains(&f), "{d_km} {off_km}: {f}");
                assert!(f >= last - 1.0e-9, "{d_km} {off_km}: {f} < {last}");
                last = f;
            }
            assert_eq!(last, 1.0);
        }
        // Inside the umbra, and in full sunlight
        assert_eq!(shadowfunc(&psun, &numeris::vector![-7.0e6, 0.0, 0.0]), 0.0);
        assert_eq!(shadowfunc(&psun, &numeris::vector![7.0e6, 0.0, 0.0]), 1.0);
        // At or inside the Earth: lit on the day side, dark on the night
        // side, and at the center
        assert_eq!(shadowfunc(&psun, &numeris::vector![6.0e6, 0.0, 0.0]), 1.0);
        assert_eq!(shadowfunc(&psun, &numeris::vector![-6.0e6, 0.0, 0.0]), 0.0);
        assert_eq!(shadowfunc(&psun, &Vector3::zeros()), 0.0);
        let f = shadowfunc(&psun, &numeris::vector![0.0, 6.0e6, 0.0]);
        assert!((0.0..=1.0).contains(&f));
    }
}
