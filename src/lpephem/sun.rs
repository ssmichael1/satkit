use crate::consts;
use crate::Duration;
use crate::ITRFCoord;
use crate::Instant;
use crate::SolarSystem;
use crate::TimeLike;
use crate::TimeScale;

use crate::mathtypes::*;

use thiserror::Error;

/// Errors from [`riseset_with`].
///
/// `#[non_exhaustive]`: variants may be added.  [`riseset`] keeps returning
/// [`lpephem::Error`](enum@super::Error).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The Sun does not rise or set on the given date at the given
    /// location: it stays above the threshold all day (polar day) or below
    /// it all day (polar night)
    #[error(
        "No sunrise or sunset on this day at this location: the Sun stays above \
         the threshold all day (polar day) or below it all day (polar night)"
    )]
    NoSunriseOrSunset,

    /// The JPL ephemeris could not be loaded or does not cover the
    /// requested date (only with `use_jpl = true`)
    #[error(transparent)]
    JplEphem(#[from] crate::jplephem::Error),

    /// The calendar date of the input time could not be turned into an
    /// [`Instant`]
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),
}

/// Arcseconds to radians
const AS2R: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// Largest periodic terms of the Earth's heliocentric longitude in VSOP87D
/// (Bretagnon & Francou 1988, series L0), each `A cos(B + C τ)` with `A` in
/// 1e-8 rad, `B` in rad, `C` in rad per Julian millennium and `τ` in Julian
/// millennia of TDB from J2000.  These are all the terms with
/// `A >= 240` (0.5") other than the Keplerian ones at multiples of the
/// Earth's mean motion (6283.08 rad/millennium), which Meeus's equation of
/// centre represents.  The geocentric Sun's longitude is the Earth's plus
/// 180 deg, so the terms carry over unchanged.
///
/// * 77713.77: the Earth's monthly motion about the Earth-Moon barycentre
///   (argument D, the Moon's mean elongation), 6.5" sin D
/// * Venus: 7860.42, 3930.21, 5884.93, 26.30, 1577.34, 775.52, ...
/// * Jupiter: 5753.38, 11506.77, 529.69, 5507.55, 5223.69, ...
/// * Mars: 398.15, ...; 3.52 and 0.07 are long-period (quasi-secular) terms
///
/// Coefficients as tabulated in Meeus, "Astronomical Algorithms", 2nd ed.,
/// Appendix III; the full series is VizieR catalogue VI/81.
const VSOP87_L0: [(f64, f64, f64); 19] = [
    (3497.0, 2.7441, 5753.3849),
    (3418.0, 2.8289, 3.5231),
    (3136.0, 3.6277, 77713.7715),
    (2676.0, 4.4181, 7860.4194),
    (2343.0, 6.1352, 3930.2097),
    (1324.0, 0.7425, 11506.7698),
    (1273.0, 2.0371, 529.6910),
    (1199.0, 1.1096, 1577.3435),
    (990.0, 5.233, 5884.927),
    (902.0, 2.045, 26.298),
    (857.0, 3.508, 398.149),
    (780.0, 1.179, 5223.694),
    (753.0, 2.533, 5507.553),
    (492.0, 4.205, 775.523),
    (357.0, 2.920, 0.067),
    (317.0, 5.849, 11790.629),
    (284.0, 1.899, 796.298),
    (271.0, 0.315, 10977.079),
    (243.0, 0.345, 5486.778),
];

/// Largest periodic terms of the Earth-Sun distance in VSOP87D (series R0),
/// as [`VSOP87_L0`] with `A` in 1e-8 AU: all non-Keplerian terms with
/// `A >= 300` (450 km), the first being the lunar term (4600 km cos D)
const VSOP87_R0: [(f64, f64, f64); 9] = [
    (3084.0, 5.1985, 77713.7715),
    (1628.0, 1.1739, 5753.3849),
    (1576.0, 2.8469, 7860.4194),
    (925.0, 5.453, 11506.770),
    (542.0, 4.564, 3930.210),
    (472.0, 3.661, 5884.927),
    (346.0, 0.964, 5507.553),
    (329.0, 5.900, 5223.694),
    (307.0, 0.299, 5573.143),
];

/// Low-precision solar coordinates, referred to the mean ecliptic and
/// equinox of date
struct SolarCoords {
    /// Geocentric ecliptic longitude including the annual aberration
    /// (-20.5"), radians
    lon: f64,
    /// Geocentric ecliptic latitude, radians (below 1")
    lat: f64,
    /// Geometric Earth-Sun distance, AU
    r: f64,
    /// Mean obliquity of the ecliptic, radians
    eps0: f64,
}

/// Low-precision solar coordinates at `t` Julian centuries of TDB from
/// J2000
///
/// Meeus, "Astronomical Algorithms", 2nd ed., ch. 25 (low accuracy, with
/// the mean obliquity of eq. 22.2), plus the largest planetary and lunar
/// terms of VSOP87D.  Against JPL DE440 over 1900-2100 the aberrated
/// longitude is within 3.6" (0.9" RMS) and the distance within 1600 km.
fn solar_coords(t: f64) -> SolarCoords {
    let sind = |x: f64| x.to_radians().sin();
    let tau = t / 10.0;

    // Geometric mean longitude and mean anomaly, degrees (eqs. 25.2, 25.3)
    let l0 = (0.0003032 * t).mul_add(t, 36000.76983f64.mul_add(t, 280.46646));
    let m = (-0.0001537 * t).mul_add(t, 35999.05029f64.mul_add(t, 357.52911));
    // Eccentricity of the Earth's orbit (eq. 25.4)
    let e = (-0.0000001267 * t).mul_add(t, 0.000042037f64.mul_add(-t, 0.016708634));
    // Equation of centre, degrees
    let c = (-0.000014 * t).mul_add(t, 0.004817f64.mul_add(-t, 1.914602)) * sind(m)
        + 0.000101f64.mul_add(-t, 0.019993) * sind(2.0 * m)
        + 0.000289 * sind(3.0 * m);
    // True anomaly and radius vector (eq. 25.5)
    let nu = m + c;
    let mut r = 1.000001018 * e.mul_add(-e, 1.0) / e.mul_add(nu.to_radians().cos(), 1.0);
    // True longitude, radians
    let mut lon = (l0 + c).to_radians();

    // Planetary and lunar perturbations
    for (a, b, freq) in VSOP87_L0 {
        lon = (a * 1.0e-8).mul_add(freq.mul_add(tau, b).cos(), lon);
    }
    for (a, b, freq) in VSOP87_R0 {
        r = (a * 1.0e-8).mul_add(freq.mul_add(tau, b).cos(), r);
    }
    // Latitude: the Earth-Moon barycentre term of VSOP87D B0, 0.58" sin F,
    // with F the Moon's argument of latitude (the Sun's latitude is minus
    // the Earth's heliocentric latitude)
    let lat = -280.0e-8 * 84334.662f64.mul_add(tau, 3.199).cos();

    // Annual aberration (eq. 25.10)
    let lon = lon - 20.4898 * AS2R / r;

    // Mean obliquity (eq. 22.2), arcseconds
    let eps0 = 0.001813f64
        .mul_add(t, -0.00059)
        .mul_add(t, -46.8150)
        .mul_add(t, 84381.448);

    SolarCoords {
        lon,
        lat,
        r,
        eps0: eps0 * AS2R,
    }
}

/// Nutation in longitude and obliquity, radians: the four largest terms,
/// good to 0.5" and 0.1" (Meeus, "Astronomical Algorithms", ch. 22)
fn nutation(t: f64) -> (f64, f64) {
    // Longitude of the Moon's ascending node, and mean longitudes of the
    // Sun and the Moon, radians
    let om = 1934.136261f64.mul_add(-t, 125.04452).to_radians();
    let ls = 36000.7698f64.mul_add(t, 280.4665).to_radians();
    let lm = 481267.8813f64.mul_add(t, 218.3165).to_radians();
    let dpsi = 0.21f64.mul_add(
        (2.0 * om).sin(),
        (-0.23f64).mul_add(
            (2.0 * lm).sin(),
            (-1.32f64).mul_add((2.0 * ls).sin(), -17.20 * om.sin()),
        ),
    );
    let deps = (-0.09f64).mul_add(
        (2.0 * om).cos(),
        0.10f64.mul_add(
            (2.0 * lm).cos(),
            0.57f64.mul_add((2.0 * ls).cos(), 9.20 * om.cos()),
        ),
    );
    (dpsi * AS2R, deps * AS2R)
}

/// Julian centuries of TDB from J2000
fn centuries_tdb(time: &Instant) -> f64 {
    (time.as_jd_with_scale(TimeScale::TDB) - 2451545.0) / 36525.0
}

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
/// * [`pos_mod`] rotated from mean of date to the GCRF (Vallado Equations
///   3-88 and 3-89); see there for the model and its accuracy.  Like
///   [`pos_mod`], the direction includes the annual aberration: it is the
///   apparent direction without nutation, 20.5" behind the geometric
///   direction (the JPL [`geocentric_pos`](crate::jplephem::geocentric_pos))
///   along the ecliptic.
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
/// * Geocentric ecliptic longitude of the Sun, radians in [0, 2π),
///   referred to the mean ecliptic and equinox of date and including the
///   annual aberration: the longitude of [`pos_mod`]
///
/// # Notes
///
/// * Within 3.6" of JPL DE440 over 1900 to 2100; see [`pos_mod`]
///
pub fn ecliptic_longitude<T: TimeLike>(time: &T) -> f64 {
    let time = time.as_instant();
    solar_coords(centuries_tdb(&time))
        .lon
        .rem_euclid(std::f64::consts::TAU)
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
/// * Meeus, "Astronomical Algorithms", 2nd ed., ch. 25 (the low-accuracy
///   solar coordinates), plus the largest planetary (Venus, Jupiter, Mars)
///   and lunar (Earth-Moon barycentre) terms of VSOP87D, Bretagnon &
///   Francou (1988): 19 terms in longitude, 9 in distance and 1 in
///   latitude
/// * As in Vallado's Algorithm 29, which this replaces, the direction
///   includes the annual aberration (-20.5" in longitude) and no nutation:
///   it is the apparent direction referred to the mean equator and equinox
///   of date.  The distance is geometric.
/// * Against JPL DE440 over 1900 to 2100 (apparent direction, light time
///   and aberration, in the same frame): ecliptic longitude within 3.6"
///   (0.9" RMS), latitude within 1", distance within 1600 km.  Algorithm
///   29 was off by up to 43" (13" RMS) and 16,000 km.
///
pub fn pos_mod<T: TimeLike>(time: &T) -> Vector3 {
    let time = time.as_instant();
    let s = solar_coords(centuries_tdb(&time));
    let (sl, cl) = s.lon.sin_cos();
    let (sb, cb) = s.lat.sin_cos();
    let (se, ce) = s.eps0.sin_cos();
    (consts::AU * s.r)
        * numeris::vector![
            cb * cl,
            (cb * sl).mul_add(ce, -(sb * se)),
            (cb * sl).mul_add(se, sb * ce),
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
/// Sunrise and sunset times on a calendar date at the given location,
/// from the built-in analytic Sun.  Same as
/// [`riseset_with(time, coord, sigma, false)`](riseset_with), which also
/// offers the JPL ephemeris; see there for the details.
///
/// Returns [`lpephem::Error::NoSunriseOrSunset`](super::Error::NoSunriseOrSunset)
/// if the Sun stays above the threshold all day (polar day) or below it
/// all day (polar night).
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
pub fn riseset<T: TimeLike>(
    time: &T,
    coord: &ITRFCoord,
    osigma: Option<f64>,
) -> super::Result<(Instant, Instant)> {
    let time = time.as_instant();
    let sigma = osigma.unwrap_or(STANDARD_SIGMA);
    riseset_impl(&time, coord, sigma, |jd, _| Ok(analytic_sun(jd)))
}

///
/// # Compute sunrise and sunset, optionally from the JPL ephemeris
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
/// # Definition
///
/// Rise and set are when the topocentric apparent Sun's **centre** is at
/// zenith distance `sigma`, by default 90° 50', i.e. 50' below a sea-level
/// horizon: 34' of standard refraction plus a fixed 16' semidiameter, the
/// almanac convention (USNO, Skyfield).  The Sun's true semidiameter varies
/// between 15.8' and 16.3' over the year, which would move the events by 1 to
/// 2 s at mid latitudes; it is not used, so that `sigma`
/// means the same for twilight as for sunrise.
///
/// # Method
///
/// Vallado's Algorithm 30 computes the event from the Sun's declination
/// and right ascension at a first guess of 6h (rise) or 18h (set) local
/// mean time.  Here that pass is repeated with the Sun at the computed
/// event until it moves less than 0.1 s.
///
/// * Built-in model (`use_jpl = false`): the analytic apparent Sun of
///   [`pos_mod`], plus nutation (the four largest terms) and the solar
///   parallax (8.8" / R) applied to `sigma`, with the Greenwich apparent
///   sidereal time from the IAU 1982 GMST and UTC in place of UT1.
/// * JPL (`use_jpl = true`): the apparent Sun from the JPL ephemeris, seen
///   from the site: light time from the Sun's barycentric position, and
///   aberration from the site's barycentric velocity (the Earth's plus its
///   rotation), rotated into the ITRF with the full IERS 2010 reduction
///   (UT1, precession-nutation, polar motion).  Parallax is exact.  The
///   JPL ephemeris is downloaded on first use.
///
/// Returns [`Error::NoSunriseOrSunset`] if the Sun stays above the threshold
/// all day (polar day) or below it all day (polar night).  Near those
/// thresholds an event is returned only if the Sun reaches the threshold
/// at the time of the event.
///
/// # Accuracy
///
/// Against Skyfield with the DE421 ephemeris (and the IERS polar motion),
/// every other day of 2024 at latitudes 60 S to 65 N:
///
/// * Built-in model: within 0.5 s, typically 0.2 s.  The remainder is the
///   analytic Sun (3.6" in longitude, 0.5" in nutation), UTC for UT1 and
///   the omitted polar motion (0.3", which moves a grazing event at 65 N by
///   0.3 s).  Between 65 N and the polar-day and polar-night thresholds,
///   where the Sun meets the threshold at a grazing angle, the error grows
///   to 1 s, and to a few seconds on the most grazing days.
/// * JPL: within 0.01 s, also up to the polar thresholds.
///
/// The horizon is at sea level: the observer's altitude is ignored.  An
/// elevated observer sees a horizon lowered by the dip,
/// dip ≈ 1.76' × √h, with h the height in meters above the surrounding
/// terrain or sea (this includes typical terrestrial refraction), which
/// makes sunrise earlier and sunset later.  To account for it, pass
/// `sigma = 90° 50' + dip`, e.g. `90.0 + (50.0 + 1.76 * h.sqrt()) / 60.0`;
/// at 100 m this moves each event by 1.5 to 3 minutes at latitudes 30 to
/// 55 deg.  Real refraction at the horizon varies by several arcminutes
/// with the weather, which moves the observed events by up to a minute or
/// more.
///
/// # Arguments
///
/// * `time`  - Time whose UTC calendar date selects the day (time of day
///   is ignored)
///
/// * `coord` - Location for which to compute sunrise & sunset
///
/// * `sigma` - Zenith distance of the Sun's centre at rise & set, degrees.
///   `None` selects "Standard", 90 deg 50 arcmin (90.0 + 50.0 / 60.0).
///   Other common values:
///    * "Civil Twilight": 96 deg
///    * "Nautical Twilight": 102 deg
///    * "Astronomical Twilight": 108 deg
///
/// * `use_jpl` - Use the JPL ephemeris (apparent, topocentric) and the
///   full Earth-orientation reduction instead of the built-in analytic Sun
///
/// # Returns
///
/// * `(sunrise, sunset)`
///
/// # Errors
///
/// * [`Error::NoSunriseOrSunset`] for polar day or night
/// * [`Error::JplEphem`] if `use_jpl` is set and the JPL ephemeris is
///   unavailable or does not cover the date (there is no fallback to the
///   analytic Sun)
///
/// # Example
///
/// ```no_run
/// use satkit::{Instant, ITRFCoord};
/// use satkit::lpephem::sun;
///
/// let greenwich = ITRFCoord::from_geodetic_deg(51.48, 0.0, 0.0);
/// let date = Instant::from_date(2024, 3, 20).unwrap();
/// let (rise, set) = sun::riseset_with(&date, &greenwich, None, true).unwrap();
/// println!("sunrise {rise}, sunset {set}");
/// ```
///
/// # References
///
/// * Vallado, "Fundamentals of Astrodynamics and Applications", Algorithm 30
/// * Meeus, "Astronomical Algorithms", 2nd ed., ch. 22 (nutation), 25
///
pub fn riseset_with<T: TimeLike>(
    time: &T,
    coord: &ITRFCoord,
    sigma: Option<f64>,
    use_jpl: bool,
) -> std::result::Result<(Instant, Instant), Error> {
    let time = time.as_instant();
    let sigma = sigma.unwrap_or(STANDARD_SIGMA);
    if !use_jpl {
        return riseset_impl(&time, coord, sigma, |jd, _| Ok(analytic_sun(jd)));
    }
    // The site's position and velocity (Earth rotation), ITRF
    let site = coord.itrf;
    let site_vel = numeris::vector![0.0, 0.0, consts::OMEGA_EARTH].cross(&site);
    riseset_impl(&time, coord, sigma, |jd, jd0h| {
        let t = Instant::from_jd_utc(jd);
        let q = crate::frametransform::qgcrf2itrf(&t);
        let qi = q.conjugate();
        let u = q * jpl_apparent_sun(&t, &(qi * site), &(qi * site_vel))?;
        // Earth-fixed longitude of the Sun, plus the UTC time of day: the
        // right ascension minus the sidereal angle at 0h
        Ok(SunDir {
            dec: u[2].asin().to_degrees(),
            x: u[1].atan2(u[0]).to_degrees() + 360.0 * (jd - jd0h),
            parallax: 0.0,
        })
    })
}

/// "Standard" rise / set: 50' below the horizon, degrees
const STANDARD_SIGMA: f64 = 90.0 + 50.0 / 60.0;

/// The Sun's direction for one pass of Algorithm 30
struct SunDir {
    /// Declination, degrees
    dec: f64,
    /// The local mean time of an event is `(LHA + x) / 360` days after the
    /// local mean midnight, with LHA the local hour angle: `x` is the
    /// right ascension minus the Greenwich sidereal angle at 0h UTC,
    /// degrees
    x: f64,
    /// Parallax to apply to `sigma`, degrees (0 for a topocentric
    /// direction)
    parallax: f64,
}

/// The analytic apparent Sun at UTC Julian date `jd`, with UTC in place of
/// UT1 for the sidereal angle
fn analytic_sun(jd: f64) -> SunDir {
    let t = (jd - 2451545.0) / 36525.0;
    // The Sun and nutation are in TDB (TT - UTC = 69 s moves the Sun 3")
    let t_tdb = centuries_tdb(&Instant::from_jd_utc(jd));
    let s = solar_coords(t_tdb);
    let (dpsi, deps) = nutation(t_tdb);
    let lon = s.lon + dpsi;
    let eps = s.eps0 + deps;
    let (sl, cl) = lon.sin_cos();
    let (sb, cb) = s.lat.sin_cos();
    let (se, ce) = eps.sin_cos();
    let dec = (cb * se).mul_add(sl, sb * ce).asin();
    // atan2 handles quadrant correctly ... very important!
    let ra = (cb * sl).mul_add(ce, -(sb * se)).atan2(cb * cl);

    // Zero-hour GMST at `jd`'s T (equation 3-45 in Vallado), plus the
    // equation of the equinoxes for the apparent sidereal angle
    let gmst0h = (2.6E-8 * t * t).mul_add(
        -t,
        (0.00038793 * t).mul_add(t, 36000.77005361f64.mul_add(t, 100.4606184)),
    );
    let gast0h = gmst0h + (dpsi * ce).to_degrees();
    SunDir {
        dec: dec.to_degrees(),
        x: (ra.to_degrees() - gast0h) % 360.0,
        // Solar parallax, 8.794" at 1 AU
        parallax: 8.794 / 3600.0 / s.r,
    }
}

/// Apparent direction of the Sun from the JPL ephemeris, GCRF, unit
/// vector, for an observer at `site` from the geocentre with velocity
/// `site_vel` (GCRF, meters, m/s)
///
/// Light time from the Sun's barycentric position, `sun(t - τ) - obs(t)`,
/// then aberration from the observer's barycentric velocity in its
/// relativistic form (the IERS Conventions / Kaplan et al. 1989); light
/// deflection by the Sun does not apply to the Sun itself.
fn jpl_apparent_sun(
    t: &Instant,
    site: &Vector3,
    site_vel: &Vector3,
) -> crate::jplephem::Result<Vector3> {
    let (sun_b, sun_bv) = crate::jplephem::barycentric_state(SolarSystem::Sun, t)?;
    let (sun_g, sun_gv) = crate::jplephem::geocentric_state(SolarSystem::Sun, t)?;
    // Barycentric position and velocity of the observer
    let obs = sun_b - sun_g + site;
    let obs_v = sun_bv - sun_gv + site_vel;

    // Light time; the Sun moves ~13 m/s about the barycentre, so two
    // iterations converge
    let mut p = sun_g - site;
    for _ in 0..2 {
        let tau = p.norm() / consts::C;
        p = crate::jplephem::barycentric_pos(
            SolarSystem::Sun,
            &(*t - Duration::from_seconds(tau)),
        )? - obs;
    }

    // Aberration
    let u = p / p.norm();
    let v = obs_v / consts::C;
    let binv = (1.0 - v.norm_squared()).sqrt();
    let uv = u.dot(&v);
    let dir = (u * binv + v * (1.0 + uv / (1.0 + binv))) / (1.0 + uv);
    Ok(dir / dir.norm())
}

/// Error types [`riseset_impl`] can return
trait RiseSetError: From<crate::time::InstantError> {
    fn no_event() -> Self;
}

impl RiseSetError for super::Error {
    fn no_event() -> Self {
        Self::NoSunriseOrSunset
    }
}

impl RiseSetError for Error {
    fn no_event() -> Self {
        Self::NoSunriseOrSunset
    }
}

/// Sunrise and sunset of `time`'s UTC date at `coord`, for the Sun's
/// direction given by `sun(jd, jd0h)` at UTC Julian date `jd`, with `jd0h`
/// the UTC Julian date of 0h on that date
fn riseset_impl<E: RiseSetError>(
    time: &Instant,
    coord: &ITRFCoord,
    sigma: f64,
    sun: impl Fn(f64, f64) -> std::result::Result<SunDir, E>,
) -> std::result::Result<(Instant, Instant), E> {
    let latitude: f64 = coord.latitude_deg();
    let longitude: f64 = coord.longitude_deg();

    let sind = |x: f64| x.to_radians().sin();
    let cosd = |x: f64| x.to_radians().cos();

    // Local mean midnight that begins the input's UTC calendar date at the
    // site's longitude.  Each event is this plus its local mean time as a
    // fraction of a day in [0, 1), so it stays on that date.
    let (year, month, day, _, _, _) = time.as_datetime();
    let jd0h: f64 = Instant::from_date(year, month, day)?.as_jd_with_scale(TimeScale::UTC);
    let jdbase = jd0h - longitude / 360.0;

    // One pass of Algorithm 30, with the Sun evaluated at UTC Julian date
    // `jd`: the local mean time of the event as a fraction of a day, or
    // `Err(cos(LHA))` when |cos(LHA)| > 1, i.e. the Sun doesn't reach the
    // threshold that day
    let pass = |jd: f64, rising: bool| -> std::result::Result<std::result::Result<f64, f64>, E> {
        let s = sun(jd, jd0h)?;
        // `sigma` is seen from the site; a geocentric direction needs the
        // parallax, which puts the geocentric Sun that much higher
        let sigma = s.parallax.mul_add(-sind(sigma), sigma);
        let coslha =
            sind(s.dec).mul_add(-sind(latitude), cosd(sigma)) / (cosd(s.dec) * cosd(latitude));
        if coslha.abs() > 1.0 {
            return Ok(Err(coslha));
        }
        let mut lha = coslha.acos().to_degrees();
        if rising {
            lha = 360.0 - lha;
        }
        Ok(Ok((lha + s.x).rem_euclid(360.0) / 360.0))
    };

    // Algorithm 30 evaluates the Sun at a first guess of 6h (rise) or 18h
    // (set) local mean time, which costs up to ~35 s at 65 deg latitude.
    // Re-evaluate it at the computed event until the event moves < 0.1 s.
    let event = |guess: f64, rising: bool| -> std::result::Result<Instant, E> {
        let mut frac = match pass(jdbase + guess, rising)? {
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
                pass(jdbase + retry, rising)?.map_err(|_| E::no_event())?
            }
        };
        for _ in 0..10 {
            // The Sun doesn't reach the threshold at the time the event would
            // occur, so the event doesn't happen that day
            let next = pass(jdbase + frac, rising)?.map_err(|_| E::no_event())?;
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

    /// Vallado Example 5-1, with UTC taken as TDB as in the book
    fn example_5_1() -> Instant {
        let t0: Instant = Instant::from_date(2006, 4, 2).unwrap();
        Instant::from_mjd_with_scale(t0.as_mjd_with_scale(TimeScale::UTC), TimeScale::TDB)
    }

    #[test]
    fn sunpos_mod() {
        let pos = pos_mod(&example_5_1());
        // Vallado's values are from Algorithm 29, which this model replaces;
        // here the two differ by 4.4e-5 (up to 43" in longitude elsewhere)
        let ref_pos = [146186212.0E3, 28788976.0E3, 12481064.0E3];
        // Regression pin for this model (Meeus + VSOP87 terms)
        let pin = [146189637037.2466, 28787706868.9137, 12480859205.0816];
        for idx in 0..3 {
            assert!((pos[idx] / ref_pos[idx] - 1.0).abs() < 1.0e-4, "{pos:?}");
            assert!((pos[idx] / pin[idx] - 1.0).abs() < 1.0e-10, "{pos:?}");
        }
    }

    #[test]
    fn sunpos_gcrf() {
        let pos = pos_gcrf(&example_5_1());
        // Below value is from Vallado example
        let ref_pos = [146259922.0E3, 28585947.0E3, 12397430.0E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            // Less exact here because we are comparing to JPL ephemeris.
            // as described by Vallado; also, the reference is the geometric
            // position, 20.5" ahead of this apparent one (4.3e-4 in z)
            assert!(err < 5e-4);
        }
    }

    #[test]
    fn test_ecliptic_longitude() {
        let lambda_deg = ecliptic_longitude(&example_5_1()).to_degrees();
        // Vallado's Algorithm 29 value; the two series differ by 2.7" here
        let ref_lambda_deg = 12.114404;
        approx::assert_abs_diff_eq!(lambda_deg, ref_lambda_deg, epsilon = 1.0e-3);
        // and the longitude of `pos_mod`
        let p = pos_mod(&example_5_1());
        let eps0 = solar_coords(centuries_tdb(&example_5_1())).eps0;
        let lon = eps0.sin().mul_add(p[2], eps0.cos() * p[1]).atan2(p[0]);
        approx::assert_abs_diff_eq!(lambda_deg, lon.to_degrees(), epsilon = 1.0e-12);
    }

    /// Longitude, latitude and distance against JPL DE440 (the Sun's
    /// apparent direction: light time and aberration) every 3 days over
    /// 1900-2100, in the mean ecliptic and equinox of date (so nutation is
    /// left out of both)
    #[test]
    fn pos_mod_vs_jpl() {
        let t0 = Instant::from_date(1900, 1, 1).unwrap();
        let (mut lmax, mut lsum, mut bmax, mut rmax) = (0.0f64, 0.0, 0.0f64, 0.0f64);
        let n = (200.0 * 365.25 / 3.0) as usize;
        for i in 0..n {
            let t = t0 + crate::Duration::from_days(3.0 * i as f64);
            let jpl = crate::frametransform::qmod2gcrf(&t).conjugate()
                * jpl_apparent_sun(&t, &Vector3::zeros(), &Vector3::zeros()).unwrap();
            let p = pos_mod(&t);
            let (se, ce) = solar_coords(centuries_tdb(&t)).eps0.sin_cos();
            // Ecliptic longitude & latitude, radians
            let ecl = |v: &Vector3| {
                let y = se.mul_add(v[2], ce * v[1]);
                let z = se.mul_add(-v[1], ce * v[2]);
                (y.atan2(v[0]), (z / v.norm()).asin())
            };
            let ((l0, b0), (l1, b1)) = (ecl(&p), ecl(&jpl));
            let dl = (l0 - l1 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            let dl = dl.to_degrees() * 3600.0;
            lmax = lmax.max(dl.abs());
            lsum += dl * dl;
            bmax = bmax.max((b0 - b1).to_degrees().abs() * 3600.0);
            let r = crate::jplephem::geocentric_pos(SolarSystem::Sun, &t)
                .unwrap()
                .norm();
            rmax = rmax.max((p.norm() - r).abs());
        }
        let lrms = (lsum / n as f64).sqrt();
        println!(
            "vs DE440, 1900-2100: longitude max {lmax:.2}\" RMS {lrms:.2}\", \
             latitude max {bmax:.2}\", distance max {:.0} km",
            rmax / 1.0e3
        );
        // Vallado's Algorithm 29: 43" max, 13" RMS, 16,000 km
        assert!(lmax < 4.0 && lrms < 1.0, "{lmax} {lrms}");
        assert!(bmax < 1.0, "{bmax}");
        assert!(rmax < 1.7e6, "{rmax}");
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
    /// `wgs84.latlon(lat, lon)` site, with the IERS polar motion installed
    /// (without it a grazing event at 65 N moves by 0.3 s).  The search
    /// window is this function's day: the local mean midnights that begin
    /// and end the UTC date.
    ///
    /// Columns: latitude, longitude (deg), month, day, then sunrise and
    /// sunset in seconds of UTC after 0h UTC on that date, to 0.01 s.  The
    /// generating script is in the description of PR #272.
    #[rustfmt::skip]
    const SKYFIELD_2024: [(f64, f64, i32, i32, f64, f64); 64] = [
        (0.0, -75.0, 3, 20, 39839.82, 83429.64),
        (0.0, -75.0, 6, 20, 39484.41, 83125.71),
        (0.0, -75.0, 9, 22, 38953.12, 82541.29),
        (0.0, -75.0, 12, 21, 39278.73, 82928.44),
        (30.0, -75.0, 3, 20, 39790.83, 83506.31),
        (30.0, -75.0, 6, 20, 35962.62, 86647.60),
        (30.0, -75.0, 9, 22, 38917.95, 82549.24),
        (30.0, -75.0, 12, 21, 42717.54, 79489.81),
        (45.0, -75.0, 3, 20, 39725.87, 83591.71),
        (45.0, -75.0, 6, 20, 33191.81, 89418.52),
        (45.0, -75.0, 9, 22, 38862.97, 82584.19),
        (45.0, -75.0, 12, 21, 45325.15, 76882.31),
        (55.0, -75.0, 3, 20, 39646.91, 83691.57),
        (55.0, -75.0, 6, 20, 30030.35, 92580.18),
        (55.0, -75.0, 9, 22, 38793.98, 82632.77),
        (55.0, -75.0, 12, 21, 48204.02, 74003.53),
        (60.0, -75.0, 3, 20, 39586.35, 83767.10),
        (60.0, -75.0, 6, 20, 27342.85, 95267.93),
        (60.0, -75.0, 9, 22, 38740.40, 82671.78),
        (60.0, -75.0, 12, 21, 50539.00, 71668.61),
        (65.0, -75.0, 3, 20, 39500.62, 83873.31),
        (65.0, -75.0, 6, 20, 21638.41, 100974.07),
        (65.0, -75.0, 9, 22, 38664.00, 82728.26),
        (65.0, -75.0, 12, 21, 54661.10, 67546.58),
        (-35.0, -75.0, 3, 20, 39817.83, 83418.15),
        (-35.0, -75.0, 6, 20, 43665.89, 78944.14),
        (-35.0, -75.0, 9, 22, 38914.34, 82613.13),
        (-35.0, -75.0, 12, 21, 34973.97, 87232.87),
        (-55.0, -75.0, 3, 20, 39736.38, 83464.58),
        (-55.0, -75.0, 6, 20, 48407.68, 74202.29),
        (-55.0, -75.0, 9, 22, 38815.72, 82746.49),
        (-55.0, -75.0, 12, 21, 29822.34, 92383.87),
        (0.0, 139.7, 3, 20, -11677.57, 31912.28),
        (0.0, 139.7, 6, 20, -12051.31, 31589.94),
        (0.0, 139.7, 9, 22, -12562.25, 31025.89),
        (0.0, 139.7, 12, 21, -12267.06, 31382.68),
        (30.0, 139.7, 3, 20, -11693.92, 31956.31),
        (30.0, 139.7, 6, 20, -15572.55, 35111.73),
        (30.0, 139.7, 9, 22, -12629.57, 31065.97),
        (30.0, 139.7, 12, 21, -8828.41, 27943.87),
        (45.0, 139.7, 3, 20, -11735.01, 32017.78),
        (45.0, 139.7, 6, 20, -18342.83, 37882.58),
        (45.0, 139.7, 9, 22, -12708.11, 31124.41),
        (45.0, 139.7, 12, 21, -6220.91, 25336.28),
        (55.0, 139.7, 3, 20, -11789.80, 32093.37),
        (55.0, 139.7, 6, 20, -21503.47, 41044.17),
        (55.0, 139.7, 9, 22, -12801.01, 31196.78),
        (55.0, 139.7, 12, 21, -3342.14, 22457.44),
        (60.0, 139.7, 3, 20, -11833.22, 32151.64),
        (60.0, 139.7, 6, 20, -24189.93, 43731.87),
        (60.0, 139.7, 9, 22, -12871.58, 31252.66),
        (60.0, 139.7, 12, 21, -1007.23, 20122.49),
        (65.0, 139.7, 3, 20, -11895.69, 32234.40),
        (65.0, 139.7, 6, 20, -29887.71, 49437.89),
        (65.0, 139.7, 9, 22, -12971.06, 31332.04),
        (65.0, 139.7, 12, 21, 3114.72, 16000.48),
        (-35.0, 139.7, 3, 20, -11739.20, 31940.33),
        (-35.0, 139.7, 6, 20, -7870.40, 27408.56),
        (-35.0, 139.7, 9, 22, -12562.08, 31058.71),
        (-35.0, 139.7, 12, 21, -16571.54, 35687.43),
        (-55.0, 139.7, 3, 20, -11861.94, 32027.80),
        (-55.0, 139.7, 6, 20, -3129.26, 22667.05),
        (-55.0, 139.7, 9, 22, -12620.28, 31151.43),
        (-55.0, 139.7, 12, 21, -21722.64, 40839.10),
    ];

    /// Largest |error| in seconds against [`SKYFIELD_2024`]
    fn worst_vs_skyfield(use_jpl: bool) -> f64 {
        let mut worst = 0.0f64;
        for (lat, lon, month, day, rise, set) in SKYFIELD_2024 {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            let t0 = Instant::from_date(2024, month, day).unwrap();
            let (r, s) = riseset_with(&t0, &coord, None, use_jpl).unwrap();
            let dr = (r - t0).as_seconds() - rise;
            let ds = (s - t0).as_seconds() - set;
            println!(
                "jpl {use_jpl:5} {lat:5} {lon:6} {month:2}/{day:2}: \
                 rise {dr:+5.2} s, set {ds:+5.2} s"
            );
            worst = worst.max(dr.abs()).max(ds.abs());
        }
        println!("jpl {use_jpl}: max |riseset - Skyfield| = {worst:.2} s");
        worst
    }

    /// Against Skyfield over latitudes 0 to 65 N, 35 S and 55 S, at a
    /// western and an eastern longitude, at the 2024 equinoxes & solstices.
    /// Includes 65 N at the June solstice, where the Sun is below -50' for
    /// only about two hours around local midnight.
    #[test]
    fn riseset_vs_skyfield() {
        // Algorithm 30 as published (one pass, no nutation or parallax):
        // 13.9 s (65 N, December); with those, and Algorithm 29's Sun
        // (0.24.1): 1.4 s
        let worst = worst_vs_skyfield(false);
        assert!(worst < 0.5, "max error vs Skyfield {worst} s");
    }

    #[test]
    fn riseset_jpl_vs_skyfield() {
        let worst = worst_vs_skyfield(true);
        assert!(worst < 0.05, "max error vs Skyfield {worst} s");
    }

    /// `riseset` is `riseset_with(.., false)`
    #[test]
    fn riseset_is_analytic_riseset_with() {
        let coord = ITRFCoord::from_geodetic_deg(42.4154, -71.1565, 0.0);
        for sigma in [None, Some(96.0), Some(108.0)] {
            let t = Instant::from_date(2024, 10, 14).unwrap();
            assert_eq!(
                riseset(&t, &coord, sigma).unwrap(),
                riseset_with(&t, &coord, sigma, false).unwrap()
            );
        }
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
            for use_jpl in [false, true] {
                let (r, s) = riseset_with(&t0, &coord, None, use_jpl).unwrap();
                for (t, h, m) in [(r, rh, rm), (s, sh, sm)] {
                    let usno = t0 + crate::Duration::from_hours((h - tz) as f64 + m as f64 / 60.0);
                    let err = (t - usno).as_seconds();
                    assert!(
                        err.abs() < 60.0,
                        "{year}-{month}-{day} {lat} {lon} jpl {use_jpl}: {t} vs {usno}"
                    );
                }
            }
        }
    }

    #[test]
    fn riseset_polar_day_and_night() {
        for lon in [-75.0, 139.7] {
            // Polar day: at 66 N the Sun's center stays above -50' at the
            // June solstice (lowest ~ -0.56 deg) ...
            // ... and polar night at 70 N at the December solstice
            // (highest ~ -3.4 deg)
            for (lat, month, day) in [(66.0, 6, 20), (70.0, 12, 21)] {
                let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
                let t = Instant::from_date(2024, month, day).unwrap();
                assert!(matches!(
                    riseset(&t, &coord, None),
                    Err(super::super::Error::NoSunriseOrSunset)
                ));
                for use_jpl in [false, true] {
                    assert!(matches!(
                        riseset_with(&t, &coord, None, use_jpl),
                        Err(Error::NoSunriseOrSunset)
                    ));
                }
            }
        }
        for msg in [
            super::super::Error::NoSunriseOrSunset.to_string(),
            Error::NoSunriseOrSunset.to_string(),
        ] {
            assert!(
                msg.contains("polar day") && msg.contains("polar night"),
                "{msg}"
            );
        }
    }

    /// With `use_jpl`, a date the JPL ephemeris doesn't cover (DE440:
    /// 1550-2650) is an error, not a silent fall back to the analytic Sun
    #[test]
    fn riseset_jpl_out_of_range() {
        let coord = ITRFCoord::from_geodetic_deg(40.0, -75.0, 0.0);
        let t = Instant::from_date(2700, 3, 20).unwrap();
        assert!(riseset_with(&t, &coord, None, false).is_ok());
        let r = riseset_with(&t, &coord, None, true);
        assert!(matches!(r, Err(Error::JplEphem(_))), "{r:?}");
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
