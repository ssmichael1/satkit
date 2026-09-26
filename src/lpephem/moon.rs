use crate::consts;
use crate::Duration;
use crate::ITRFCoord;
use crate::Instant;
use crate::SolarSystem;
use crate::TimeLike;
use crate::TimeScale;

use crate::mathtypes::*;

use thiserror::Error;

/// Errors from the Moon rise/set and phase-time searches
/// ([`riseset`], [`phase_times`], [`next_phase`]).
///
/// `#[non_exhaustive]`: variants may be added.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The JPL ephemeris could not be loaded or does not cover the
    /// requested time (only with `use_jpl = true`)
    #[error(transparent)]
    JplEphem(#[from] crate::jplephem::Error),

    /// [`next_phase`] was asked for a phase other than the four principal
    /// phases (New Moon, First Quarter, Full Moon, Last Quarter)
    #[error("{} is not a principal phase (New Moon, First Quarter, Full Moon or Last Quarter)", .0.name())]
    NotPrincipalPhase(MoonPhase),

    /// The search runs outside the representable time range (the input's
    /// calendar date, or the end of a [`next_phase`] search)
    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),
}

/// Result type of the Moon rise/set and phase-time searches.
pub type Result<T> = std::result::Result<T, Error>;

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
pub fn ecliptic_longitude<T: TimeLike>(time: &T) -> f64 {
    let time = time.as_instant();
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

    lon % std::f64::consts::TAU
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
pub fn phase<T: TimeLike>(time: &T) -> f64 {
    let time = time.as_instant();
    let lambda_moon = ecliptic_longitude(&time);
    let lambda_sun = crate::lpephem::sun::ecliptic_longitude(&time);

    (lambda_moon - lambda_sun).rem_euclid(std::f64::consts::TAU)
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
pub fn illumination<T: TimeLike>(time: &T) -> f64 {
    let time = time.as_instant();
    let phase = phase(&time);
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
pub fn phase_name<T: TimeLike>(time: &T) -> MoonPhase {
    let time = time.as_instant();
    let phase_rad = phase(&time);
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
/// * Algorithm 31 from Vallado for the moon in Mean of Date (MOD), then
///   rotated from MOD to GCRF via Equations 3-88 and 3-89 in Vallado
/// * Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst
///   case against JPL DE440 over 1950 - 2100), 0.2 degree in ecliptic
///   latitude, and 1275 km in range
///
#[inline]
pub fn pos_gcrf<T: TimeLike>(time: &T) -> Vector3 {
    let time = time.as_instant();
    crate::frametransform::qmod2gcrf(&time) * pos_mod(&time)
}

///
/// Approximate Moon position in the Mean-of-Date (MOD) Frame
///
/// From Vallado Algorithm 31
///
/// # Arguments
///
/// * `time` - Instant at which to compute moon position
///
/// Output:
///
///  * Vector representing moon position in MOD frame
///    at given time.  Units are meters
///
/// # Notes
///
/// * Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst
///   case), 0.2 degree in ecliptic latitude, and 1275 km in range
///
pub fn pos_mod<T: TimeLike>(time: &T) -> Vector3 {
    let time = time.as_instant();
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

    rmag * numeris::vector![
        f64::cos(phi_ecliptic) * f64::cos(lambda_ecliptic),
        (f64::cos(epsilon) * f64::cos(phi_ecliptic)).mul_add(
            f64::sin(lambda_ecliptic),
            -(f64::sin(epsilon) * f64::sin(phi_ecliptic)),
        ),
        (f64::sin(epsilon) * f64::cos(phi_ecliptic)).mul_add(
            f64::sin(lambda_ecliptic),
            f64::cos(epsilon) * f64::sin(phi_ecliptic),
        ),
    ]
}

/// Standard horizontal refraction, 34 arcmin, radians
const REFRACTION: f64 = 34.0 / 60.0 * std::f64::consts::PI / 180.0;

/// Apparent geocentric GCRF position of a JPL body, meters: its geometric
/// position one light time earlier, `r(t) - v(t) τ` with `τ = |r| / c`.
/// For an observer in uniform motion this light-time shift is the whole of
/// the annual aberration plus light-time correction (they cancel for the
/// Earth's own motion), to about 0.01 arcsec for the Sun and the Moon.
fn jpl_apparent(body: SolarSystem, t: &Instant) -> Result<Vector3> {
    let (p, v) = crate::jplephem::geocentric_state(body, t)?;
    Ok(p - v * (p.norm() / consts::C))
}

/// Height of the Moon's upper limb above the rise / set horizon, radians:
/// topocentric altitude of the Moon's centre + 34' refraction + semidiameter.
/// Positive when the Moon is up.
///
/// `jpl` selects the JPL Moon with the full IERS 2010 rotation, given as its
/// slowly varying factors `(q_cirs2gcrs, q_itrf2tirs)` (precession-nutation
/// and polar motion, which change by milliarcseconds in a day); `None`
/// selects the analytic Moon with the approximate rotation.
fn upper_limb_height(
    t: &Instant,
    coord: &ITRFCoord,
    up: &Vector3,
    jpl: Option<&(Quaternion, Quaternion)>,
) -> Result<f64> {
    let pitrf = match jpl {
        Some((q_c2g, w)) => {
            let q_itrf2gcrf = q_c2g * crate::frametransform::qtirs2cirs(t) * w;
            q_itrf2gcrf.conjugate() * jpl_apparent(SolarSystem::Moon, t)?
        }
        None => crate::frametransform::qgcrf2itrf_approx(t) * pos_gcrf(t),
    };
    let r = pitrf - coord.itrf;
    let d = r.norm();
    Ok((r.dot(up) / d).asin() + REFRACTION + (consts::MOON_RADIUS / d).asin())
}

///
/// # Compute moonrise and moonset
///
/// Moonrise and moonset times on a calendar date at the given location.
///
/// The input time selects the date: its **UTC calendar date** is used and
/// its time of day is ignored.  The returned moonrise and moonset are those
/// of that date at the location's longitude, i.e. between the local (mean
/// solar) midnights that begin and end that date there: the window is
/// `[0h UTC - longitude / 15 hours, + 24 hours)`.  Both are returned as UTC
/// instants, so at far-west longitudes the moonset (or moonrise) may fall
/// on the next UTC date, and at far-east longitudes on the previous one.
/// These are the same day semantics as [`sun::riseset`](super::sun::riseset).
///
/// To get the events of a *local* date, pass a time at local noon on that
/// date: for time zones within UTC-11 to UTC+11 local noon falls on the
/// same UTC date, while local midnight falls on the previous UTC date east
/// of Greenwich.
///
/// The Moon rises about 50 minutes later each day, so about once a month
/// there is no moonrise on a date, and on another date no moonset; that
/// event is `None`.  At high latitudes the Moon can stay up (or down) all
/// day, giving `(None, None)`, and a window can hold two moonrises (or two
/// moonsets): the first is returned.
///
/// # Definition
///
/// Rise and set are when the Moon's **upper limb** touches a sea-level
/// horizon with standard refraction, the almanac definition (USNO,
/// Astronomical Almanac): the topocentric altitude of the Moon's centre is
/// `-34' - SD`, with the semidiameter `SD = asin(R_moon / d)` for the
/// topocentric distance `d` and `R_moon = 1737.4 km`.  The Moon's position
/// relative to the site is computed in the ITRF, so parallax (up to 61') is
/// exact, and the altitude is measured from the plane normal to the
/// geodetic vertical; the site's height enters only through its position
/// (there is no dip correction for an elevated observer).
///
/// # Method
///
/// The upper-limb altitude is sampled every 10 minutes over the window (plus
/// one sample on each side), every sign change and every near-horizon
/// altitude extremum between samples (a grazing Moon) is bracketed, and
/// each event is refined by bisection to 0.05 s.
///
/// # Arguments
///
/// * `time` - Time whose UTC calendar date selects the day (time of day
///   is ignored)
///
/// * `coord` - Location for which to compute moonrise & moonset
///
/// * `use_jpl` - Use the JPL ephemeris (apparent position, including
///   light time and aberration) with the full IERS 2010 Earth-orientation
///   reduction, instead of the built-in analytic Moon ([`pos_gcrf`]) with
///   the approximate GCRF-to-ITRF rotation.  The JPL ephemeris is
///   downloaded on first use.
///
/// # Returns
///
/// * `(moonrise, moonset)`, each `None` if it does not happen on that date
///
/// # Accuracy
///
/// Against Skyfield (DE440s) for every day of 2024 at 15 sites from 35° S
/// to 62° N:
///
/// * With `use_jpl`: within 0.1 s, and 1.6 s where the Moon grazes the
///   horizon at 62° N; the same days without a moonrise or moonset
/// * Built-in model: the analytic Moon's 0.3° / 0.2° error in ecliptic
///   longitude / latitude, divided by the Moon's vertical rate at the
///   horizon, gives within 52 s at the equator, 89 s at 35°, 128 s at 52°
///   and a few minutes at 62°; a Moon that only just grazes the horizon, or
///   an event within a minute or two of midnight, may be found by one model
///   and not the other
///
/// Real refraction at the horizon varies by several arcminutes with the
/// weather, which moves the observed event by a minute or more.
///
/// # Errors
///
/// * [`Error::JplEphem`] if `use_jpl` is set and the JPL ephemeris is
///   unavailable or does not cover the date
///
/// # Example
///
/// ```
/// use satkit::{Instant, ITRFCoord};
/// use satkit::lpephem::moon;
///
/// let greenwich = ITRFCoord::from_geodetic_deg(51.48, 0.0, 0.0);
/// let date = Instant::from_date(2024, 3, 15).unwrap();
/// let (rise, set) = moon::riseset(&date, &greenwich, false).unwrap();
/// println!("moonrise {rise:?}, moonset {set:?}");
/// ```
///
pub fn riseset<T: TimeLike>(
    time: &T,
    coord: &ITRFCoord,
    use_jpl: bool,
) -> Result<(Option<Instant>, Option<Instant>)> {
    let time = time.as_instant();

    // The local mean day of the input's UTC calendar date
    let (year, month, day, _, _, _) = time.as_datetime();
    let jd0h: f64 = Instant::from_date(year, month, day)?.as_jd_with_scale(TimeScale::UTC);
    let lon = coord.longitude_deg();
    let start = Instant::from_jd_utc(jd0h - lon / 360.0);
    let span = (Instant::from_jd_utc(jd0h + 1.0 - lon / 360.0) - start).as_seconds();

    let up = coord.q_enu2itrf() * numeris::vector![0.0, 0.0, 1.0];
    // Precession-nutation and polar motion, once for the day
    let slow = use_jpl.then(|| {
        crate::frametransform::qitrf2gcrf_slow_parts(&(start + Duration::from_seconds(span / 2.0)))
    });
    let height = |s: f64| {
        upper_limb_height(
            &(start + Duration::from_seconds(s)),
            coord,
            &up,
            slow.as_ref(),
        )
    };

    // Sample every 10 minutes, with one extra sample on each side of the
    // window so a graze at a window edge is still seen
    const NSTEP: usize = 144;
    let step = span / NSTEP as f64;
    let ts: Vec<f64> = (0..NSTEP + 3).map(|i| (i as f64 - 1.0) * step).collect();
    let hs = ts
        .iter()
        .map(|&s| height(s))
        .collect::<Result<Vec<f64>>>()?;

    // Brackets (a, b) with the Moon up at a and down at b or vice versa
    let mut brackets: Vec<(f64, f64)> = Vec::new();
    for i in 0..ts.len() - 1 {
        if (hs[i] > 0.0) != (hs[i + 1] > 0.0) {
            brackets.push((ts[i], ts[i + 1]));
        }
    }
    // A Moon that grazes the horizon can rise and set (or set and rise)
    // between two samples: look for an altitude extremum within 1 deg of the
    // horizon between samples whose neighbours are all on the same side
    for i in 1..ts.len() - 1 {
        let (h0, h1, h2) = (hs[i - 1], hs[i], hs[i + 1]);
        let up1 = h1 > 0.0;
        let peak_below = !up1 && h1 >= h0 && h1 >= h2;
        let trough_above = up1 && h1 <= h0 && h1 <= h2;
        if (h0 > 0.0) != up1 || (h2 > 0.0) != up1 || h1.abs() > 1.0f64.to_radians() {
            continue;
        }
        if !(peak_below || trough_above) {
            continue;
        }
        // Golden-section search for the extremum of the altitude on
        // [ts[i-1], ts[i+1]]; `sgn * height` is maximised
        let sgn = if peak_below { 1.0 } else { -1.0 };
        let gr = (5.0f64.sqrt() - 1.0) / 2.0;
        let (mut a, mut b) = (ts[i - 1], ts[i + 1]);
        let mut c = gr.mul_add(-(b - a), b);
        let mut d = gr.mul_add(b - a, a);
        let mut fc = sgn * height(c)?;
        let mut fd = sgn * height(d)?;
        while b - a > 1.0 {
            if fc > fd {
                (b, d, fd) = (d, c, fc);
                c = gr.mul_add(-(b - a), b);
                fc = sgn * height(c)?;
            } else {
                (a, c, fc) = (c, d, fd);
                d = gr.mul_add(b - a, a);
                fd = sgn * height(d)?;
            }
        }
        let (text, hext) = if fc > fd { (c, fc) } else { (d, fd) };
        if (sgn * hext > 0.0) != up1 {
            brackets.push((ts[i - 1], text));
            brackets.push((text, ts[i + 1]));
        }
    }

    // Refine each bracket by bisection: a rise if the Moon is up at its end
    let mut events: Vec<(f64, bool)> = Vec::with_capacity(brackets.len());
    for (mut a, mut b) in brackets {
        let up_a = height(a)? > 0.0;
        while b - a > 0.05 {
            let m = 0.5 * (a + b);
            if (height(m)? > 0.0) == up_a {
                a = m;
            } else {
                b = m;
            }
        }
        events.push((0.5 * (a + b), !up_a));
    }
    events.sort_by(|x, y| x.0.total_cmp(&y.0));

    let first = |rising: bool| {
        events
            .iter()
            .find(|(s, r)| *r == rising && *s >= 0.0 && *s < span)
            .map(|(s, _)| start + Duration::from_seconds(*s))
    };
    Ok((first(true), first(false)))
}

/// The principal phases, in order of the Moon − Sun ecliptic longitude
/// difference: 0°, 90°, 180°, 270°
const PRINCIPAL_PHASES: [MoonPhase; 4] = [
    MoonPhase::NewMoon,
    MoonPhase::FirstQuarter,
    MoonPhase::FullMoon,
    MoonPhase::LastQuarter,
];

/// Moon − Sun geocentric ecliptic longitude difference in `[0, 2π)`,
/// radians: the built-in [`phase`], or apparent JPL positions in the
/// ecliptic of date
fn phase_angle(t: &Instant, use_jpl: bool) -> Result<f64> {
    if !use_jpl {
        return Ok(phase(t));
    }
    // GCRF → mean equator of date (IAU 2006 precession) → mean ecliptic of
    // date (IAU 2006 mean obliquity).  Nutation shifts both longitudes by
    // the same Δψ, so their difference is the same in the true ecliptic.
    let tt = (t.as_jd_with_scale(TimeScale::TT) - 2451545.0) / 36525.0;
    let eps_arcsec = tt.mul_add(
        tt.mul_add(
            tt.mul_add(
                tt.mul_add(tt.mul_add(-4.34e-8, -5.76e-7), 0.00200340),
                -0.0001831,
            ),
            -46.836769,
        ),
        84381.406,
    );
    let q = Quaternion::rotx(-(eps_arcsec / 3600.0).to_radians())
        * crate::frametransform::qmod2gcrf(t).conjugate();
    let lon = |v: Vector3| {
        let e = q * v;
        e[1].atan2(e[0])
    };
    let dlon = lon(jpl_apparent(SolarSystem::Moon, t)?) - lon(jpl_apparent(SolarSystem::Sun, t)?);
    Ok(dlon.rem_euclid(std::f64::consts::TAU))
}

///
/// Times of the principal Moon phases in a time interval
///
/// Returns every New Moon, First Quarter, Full Moon and Last Quarter in
/// `[start, end)`, in time order.
///
/// # Definition
///
/// The principal phases are the instants when the Moon's geocentric
/// apparent ecliptic longitude exceeds the Sun's by 0°, 90°, 180° and 270°,
/// the definition used by the Astronomical Almanac and the USNO.  Full
/// Moon is therefore not exactly the instant of greatest illuminated
/// fraction, which also depends on the Moon's ecliptic latitude.
///
/// # Method
///
/// The longitude difference, which grows by 11–14° a day, is sampled at
/// most a day apart and unwrapped; each crossing of a multiple of 90° is
/// refined by bisection to 0.05 s.
///
/// # Arguments
///
/// * `start` - Start of the interval (inclusive)
/// * `end` - End of the interval (exclusive)
/// * `use_jpl` - Use apparent JPL positions (light time and aberration,
///   which move the Sun by 20" and the phase times by about 40 s) in the
///   mean ecliptic of date, instead of the built-in analytic Sun and Moon
///   ([`phase`]).  The JPL ephemeris is downloaded on first use.
///
/// # Returns
///
/// * `(phase, time)` pairs in time order; empty if `end <= start`
///
/// # Accuracy
///
/// * With `use_jpl`: within 0.1 s of Skyfield (DE440s) and 40 s of USNO's
///   times (rounded to the minute) over 2024
/// * Built-in model: the analytic Moon's longitude error (0.3°, 0.36° worst
///   case) over the 12.2°/day relative motion allows up to about 42
///   minutes; 22 minutes at most over 2024
///
/// # Errors
///
/// * [`Error::JplEphem`] if `use_jpl` is set and the JPL ephemeris is
///   unavailable or does not cover the interval
///
/// # Example
///
/// ```
/// use satkit::Instant;
/// use satkit::lpephem::moon::{self, MoonPhase};
///
/// // The full moons of 2024
/// let start = Instant::from_date(2024, 1, 1).unwrap();
/// let end = Instant::from_date(2025, 1, 1).unwrap();
/// for (phase, t) in moon::phase_times(&start, &end, false).unwrap() {
///     if phase == MoonPhase::FullMoon {
///         println!("{t}");
///     }
/// }
/// ```
///
pub fn phase_times<T: TimeLike>(
    start: &T,
    end: &T,
    use_jpl: bool,
) -> Result<Vec<(MoonPhase, Instant)>> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let start = start.as_instant();
    let span = (end.as_instant() - start).as_seconds();
    let mut out = Vec::new();
    if span <= 0.0 {
        return Ok(out);
    }
    let raw = |s: f64| phase_angle(&(start + Duration::from_seconds(s)), use_jpl);
    // Unwrapped increment from angle a to angle b (less than 180° apart)
    let wrap = |x: f64| (x + PI).rem_euclid(TAU) - PI;

    // Steps of at most a day: under 15° of phase, so unwrapping is safe
    let nstep = (span / 86400.0).ceil() as usize;
    let step = span / nstep as f64;
    let (mut s0, mut r0) = (0.0, raw(0.0)?);
    let mut u0 = r0;
    for i in 1..=nstep {
        let s1 = if i == nstep { span } else { i as f64 * step };
        let r1 = raw(s1)?;
        let u1 = u0 + wrap(r1 - r0);
        // Every multiple of 90° in [u0, u1)
        let mut k = (u0 / FRAC_PI_2).ceil();
        while k * FRAC_PI_2 < u1 {
            let target = k * FRAC_PI_2;
            let (mut a, mut b) = (s0, s1);
            while b - a > 0.05 {
                let m = 0.5 * (a + b);
                if u0 + wrap(raw(m)? - r0) < target {
                    a = m;
                } else {
                    b = m;
                }
            }
            out.push((
                PRINCIPAL_PHASES[(k as i64).rem_euclid(4) as usize],
                start + Duration::from_seconds(0.5 * (a + b)),
            ));
            k += 1.0;
        }
        (s0, r0, u0) = (s1, r1, u1);
    }
    Ok(out)
}

///
/// Time of the next occurrence of a principal Moon phase
///
/// Returns the first time at or after `time` when the Moon reaches `phase`,
/// which must be one of the four principal phases
/// ([`MoonPhase::NewMoon`], [`MoonPhase::FirstQuarter`],
/// [`MoonPhase::FullMoon`], [`MoonPhase::LastQuarter`]).  See
/// [`phase_times`] for the definition, method and accuracy.
///
/// # Arguments
///
/// * `time` - Time from which to search
/// * `phase` - Principal phase to find
/// * `use_jpl` - Use apparent JPL positions instead of the built-in
///   analytic Sun and Moon (see [`phase_times`])
///
/// # Errors
///
/// * [`Error::NotPrincipalPhase`] if `phase` is not a principal phase
///   (e.g. [`MoonPhase::WaxingCrescent`], which covers a range of phases
///   rather than an instant)
/// * [`Error::JplEphem`] if `use_jpl` is set and the JPL ephemeris is
///   unavailable or does not cover the search
///
/// # Example
///
/// ```
/// use satkit::Instant;
/// use satkit::lpephem::moon::{self, MoonPhase};
///
/// let t = Instant::from_date(2024, 1, 1).unwrap();
/// let full = moon::next_phase(&t, MoonPhase::FullMoon, false).unwrap();
/// println!("next full moon: {full}");
/// ```
///
pub fn next_phase<T: TimeLike>(time: &T, phase: MoonPhase, use_jpl: bool) -> Result<Instant> {
    if !PRINCIPAL_PHASES.contains(&phase) {
        return Err(Error::NotPrincipalPhase(phase));
    }
    let start = time.as_instant();
    // Each phase recurs within a synodic month (29.3–29.9 days)
    let end = start + Duration::from_days(35.0);
    phase_times(&start, &end, use_jpl)?
        .into_iter()
        .find(|(p, _)| *p == phase)
        .map(|(_, t)| t)
        // Only when the search window is cut short at the end of the
        // representable time range
        .ok_or_else(|| crate::time::InstantError::InvalidYear(start.as_datetime().0).into())
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

        // Vallado's worked example is the mean-of-date output
        let pos = pos_mod(&t);

        // Below value is from Vallado example
        let ref_pos = [-134240.626E3, -311571.590E3, -126693.785E3];
        for idx in 0..3 {
            let err = f64::abs(pos[idx] / ref_pos[idx] - 1.0);
            assert!(err < 1.0e-6);
        }
    }

    /// Compare against the JPL ephemeris over 1950 - 2100.  `pos_gcrf` used
    /// to return mean-of-date coordinates labelled GCRF, which drifted from
    /// the truth by precession (~1.4 deg / century in longitude).
    #[test]
    fn moonpos_vs_jplephem() {
        // J2000 ecliptic longitude & latitude of a GCRF vector
        let eps0 = (84381.406f64 / 3600.0).to_radians();
        let ecliptic = |v: &Vector3| {
            let e = Quaternion::rotx(-eps0) * v;
            (e[1].atan2(e[0]), (e[2] / e.norm()).asin())
        };

        let t0 = Instant::from_date(1950, 1, 1).unwrap();
        let t1 = Instant::from_date(2100, 1, 1).unwrap();
        let (mut lmax, mut bmax, mut rmax) = (0.0f64, 0.0f64, 0.0f64);
        // Least-squares slope of the longitude error vs time, deg / century
        let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut t = t0;
        while t < t1 {
            let pjpl = crate::jplephem::geocentric_pos(crate::SolarSystem::Moon, &t).unwrap();
            let p = pos_gcrf(&t);
            let (l1, b1) = ecliptic(&p);
            let (l2, b2) = ecliptic(&pjpl);
            let dl = ((l1 - l2 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI)
                .to_degrees();
            lmax = lmax.max(dl.abs());
            bmax = bmax.max((b1 - b2).abs().to_degrees());
            rmax = rmax.max((p.norm() - pjpl.norm()).abs());
            let x = (t.as_jd_with_scale(TimeScale::TT) - 2451545.0) / 36525.0;
            n += 1.0;
            sx += x;
            sy += dl;
            sxx += x * x;
            sxy += x * dl;
            t += crate::Duration::from_days(1.37);
        }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        println!(
            "moon vs JPL 1950-2100: max lon {lmax:.3} deg, lat {bmax:.3} deg, \
             range {:.0} km, lon drift {slope:.4} deg/century",
            rmax / 1.0e3
        );
        // Vallado quotes 0.3 deg; the worst case here is 0.36 deg
        assert!(lmax < 0.37, "longitude error {lmax} deg");
        assert!(bmax < 0.2, "latitude error {bmax} deg");
        assert!(rmax < 1275.0e3, "range error {rmax} m");
        assert!(slope.abs() < 0.01, "longitude drift {slope} deg/century");
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

    /// Upper-limb height (radians) of the analytic and the JPL Moon
    fn heights(t: &Instant, coord: &ITRFCoord) -> (f64, f64) {
        let up = coord.q_enu2itrf() * numeris::vector![0.0, 0.0, 1.0];
        let slow = crate::frametransform::qitrf2gcrf_slow_parts(t);
        (
            upper_limb_height(t, coord, &up, None).unwrap(),
            upper_limb_height(t, coord, &up, Some(&slow)).unwrap(),
        )
    }

    type SkyfieldRow = (f64, f64, i32, i32, Option<f64>, Option<f64>);
    type UsnoRow = (
        f64,
        f64,
        i32,
        i32,
        i32,
        Option<(i32, i32)>,
        Option<(i32, i32)>,
    );

    /// Moon rise / set on a 2024 grid from Skyfield 1.55 with DE440s:
    /// Skyfield's default Moon horizon is the apparent topocentric centre at
    /// -34' - R/d with R = 1737.4 km (upper limb, standard refraction).
    /// The window is the local mean day, first event of each kind:
    ///
    /// ```python
    /// obs = eph["earth"] + wgs84.latlon(lat, lon, elevation_m=0.0)
    /// t0 = ts.utc(2024, month, day)
    /// start = ts.tt_jd(t0.tt - lon / 360.0); end = ts.tt_jd(start.tt + 1.0)
    /// t, ok = almanac.find_risings(obs, eph["moon"], start, end)   # and find_settings
    /// rise = round((t[ok][0].tt - t0.tt) * 86400.0, 1)             # None if no ok event
    /// ```
    ///
    /// Every time was checked against Skyfield's own altitude (residual under
    /// 0.05"); the full script, which also refines grazing events elsewhere
    /// in 2024, is in the PR that added this test.
    ///
    /// `(lat deg, lon deg, month, day, rise, set)`, events in seconds after
    /// 0h UTC on the date, `None` if none in the window
    #[rustfmt::skip]
    const SKYFIELD_RISESET_2024: [SkyfieldRow; 120] = [
        (  0.0,  -75.0,  1,  4, None, Some(62335.1)),
        (  0.0,    0.0,  1,  4, None, Some(43828.4)),
        (  0.0,  139.7,  1,  4, None, Some(9369.1)),
        ( 35.0,  -75.0,  1,  4, Some(18847.6), Some(60983.6)),
        ( 35.0,    0.0,  1,  4, Some(138.9), Some(42685.1)),
        ( 35.0,  139.7,  1,  4, None, Some(8612.5)),
        ( 52.0,  -75.0,  1,  4, Some(19593.2), Some(59873.6)),
        ( 52.0,    0.0,  1,  4, Some(703.8), Some(41747.4)),
        ( 52.0,  139.7,  1,  4, None, Some(7992.3)),
        ( 62.0,  -75.0,  1,  4, Some(20388.5), Some(58726.9)),
        ( 62.0,    0.0,  1,  4, Some(1305.1), Some(40782.3)),
        ( 62.0,  139.7,  1,  4, None, Some(7356.7)),
        (-35.0,  -75.0,  1,  4, None, Some(63706.9)),
        (-35.0,    0.0,  1,  4, None, Some(44987.4)),
        (-35.0,  139.7,  1,  4, Some(52575.9), Some(10133.0)),
        (  0.0,  -75.0,  3, 15, Some(56614.0), Some(101436.9)),
        (  0.0,    0.0,  3, 15, Some(37912.2), Some(82728.2)),
        (  0.0,  139.7,  3, 15, Some(3086.7), Some(47886.0)),
        ( 35.0,  -75.0,  3, 15, Some(51789.3), None),
        ( 35.0,    0.0,  3, 15, Some(33248.6), None),
        ( 35.0,  139.7,  3, 15, Some(-1244.0), Some(52718.5)),
        ( 52.0,  -75.0,  3, 15, Some(47413.7), Some(20338.2)),
        ( 52.0,    0.0,  3, 15, Some(29059.7), Some(1250.8)),
        ( 52.0,  139.7,  3, 15, Some(-5064.5), None),
        ( 62.0,  -75.0,  3, 15, Some(41480.5), Some(26102.6)),
        ( 62.0,    0.0,  3, 15, Some(23579.1), Some(6546.0)),
        ( 62.0,  139.7,  3, 15, Some(-9783.9), Some(-29836.6)),
        (-35.0,  -75.0,  3, 15, Some(61539.0), Some(96262.0)),
        (-35.0,    0.0,  3, 15, Some(42680.4), Some(77680.2)),
        (-35.0,  139.7,  3, 15, Some(7526.5), Some(43113.6)),
        (  0.0,  -75.0,  4, 23, Some(82476.4), Some(37955.7)),
        (  0.0,    0.0,  4, 23, Some(63941.4), Some(19437.6)),
        (  0.0,  139.7,  4, 23, Some(29435.4), Some(-15040.2)),
        ( 35.0,  -75.0,  4, 23, Some(85061.2), Some(35926.9)),
        ( 35.0,    0.0,  4, 23, Some(66314.8), Some(17617.3)),
        ( 35.0,  139.7,  4, 23, Some(31412.9), Some(-16471.5)),
        ( 52.0,  -75.0,  4, 23, Some(87320.4), Some(34246.5)),
        ( 52.0,    0.0,  4, 23, Some(68379.7), Some(16114.5)),
        ( 52.0,  139.7,  4, 23, Some(33120.6), Some(-17647.6)),
        ( 62.0,  -75.0,  4, 23, Some(89869.0), Some(32479.7)),
        ( 62.0,    0.0,  4, 23, Some(70683.5), Some(14544.6)),
        ( 62.0,  139.7,  4, 23, Some(34993.4), Some(-18864.2)),
        (-35.0,  -75.0,  4, 23, Some(79960.3), Some(40020.3)),
        (-35.0,    0.0,  4, 23, Some(61631.8), Some(21289.3)),
        (-35.0,  139.7,  4, 23, Some(27512.8), Some(-13586.0)),
        (  0.0,  -75.0,  6, 21, Some(82658.1), Some(37607.7)),
        (  0.0,    0.0,  6, 21, Some(63913.0), Some(18880.2)),
        (  0.0,  139.7,  6, 21, Some(29014.1), Some(-15978.9)),
        ( 35.0,  -75.0,  6, 21, Some(88219.1), Some(32162.7)),
        ( 35.0,    0.0,  6, 21, Some(69454.4), Some(13510.8)),
        ( 35.0,  139.7,  6, 21, Some(34466.9), Some(-21164.0)),
        ( 52.0,  -75.0,  6, 21, Some(93627.6), Some(26955.9)),
        ( 52.0,    0.0,  6, 21, Some(74850.3), Some(8411.5)),
        ( 52.0,  139.7,  6, 21, Some(39768.2), Some(-26016.8)),
        ( 62.0,  -75.0,  6, 21, None, None),
        ( 62.0,    0.0,  6, 21, None, None),
        ( 62.0,  139.7,  6, 21, None, None),
        (-35.0,  -75.0,  6, 21, Some(77128.8), Some(43065.7)),
        (-35.0,    0.0,  6, 21, Some(58414.5), Some(24272.8)),
        (-35.0,  139.7,  6, 21, Some(23623.0), Some(-10754.2)),
        (  0.0,  -75.0,  7,  5, Some(38368.7), Some(83221.3)),
        (  0.0,    0.0,  7,  5, Some(19641.4), Some(64508.4)),
        (  0.0,  139.7,  7,  5, Some(-15252.5), Some(29634.3)),
        ( 35.0,  -75.0,  7,  5, Some(32861.8), Some(88547.8)),
        ( 35.0,    0.0,  7,  5, Some(14109.6), Some(69912.7)),
        ( 35.0,  139.7,  7,  5, Some(-20779.2), Some(35137.6)),
        ( 52.0,  -75.0,  7,  5, Some(27542.7), Some(93561.9)),
        ( 52.0,    0.0,  7,  5, Some(8770.6), Some(75034.3)),
        ( 52.0,  139.7,  7,  5, Some(-26086.6), Some(40406.8)),
        ( 62.0,  -75.0,  7,  5, None, Some(101842.9)),
        ( 62.0,    0.0,  7,  5, None, Some(84017.9)),
        ( 62.0,  139.7,  7,  5, None, Some(51296.5)),
        (-35.0,  -75.0,  7,  5, Some(43874.1), Some(77828.1)),
        (-35.0,    0.0,  7,  5, Some(25182.9), Some(59046.1)),
        (-35.0,  139.7,  7,  5, Some(-9694.5), Some(24091.4)),
        (  0.0,  -75.0,  9, 24, None, Some(60499.5)),
        (  0.0,    0.0,  9, 24, None, Some(41764.9)),
        (  0.0,  139.7,  9, 24, Some(51905.6), Some(6858.5)),
        ( 35.0,  -75.0,  9, 24, Some(99934.4), Some(66118.5)),
        ( 35.0,    0.0,  9, 24, Some(81183.6), Some(47381.0)),
        ( 35.0,  139.7,  9, 24, Some(46291.0), Some(12416.3)),
        ( 52.0,  -75.0,  9, 24, Some(94519.3), Some(71560.6)),
        ( 52.0,    0.0,  9, 24, Some(75739.1), Some(52831.7)),
        ( 52.0,  139.7,  9, 24, Some(40861.5), Some(17811.9)),
        ( 62.0,  -75.0,  9, 24, None, None),
        ( 62.0,    0.0,  9, 24, None, None),
        ( 62.0,  139.7,  9, 24, None, None),
        (-35.0,  -75.0,  9, 24, Some(21207.6), Some(54857.6)),
        (-35.0,    0.0,  9, 24, Some(2415.8), Some(36137.2)),
        (-35.0,  139.7,  9, 24, Some(-32643.1), Some(1310.3)),
        (  0.0,  -75.0, 10, 17, Some(83341.4), Some(38426.2)),
        (  0.0,    0.0, 10, 17, Some(64659.1), Some(19761.1)),
        (  0.0,  139.7, 10, 17, Some(29879.5), Some(-14990.1)),
        ( 35.0,  -75.0, 10, 17, Some(80828.0), Some(40358.8)),
        ( 35.0,    0.0, 10, 17, Some(62408.5), Some(21418.1)),
        ( 35.0,  139.7, 10, 17, Some(28124.6), Some(-13849.6)),
        ( 52.0,  -75.0, 10, 17, Some(78755.9), Some(42019.1)),
        ( 52.0,    0.0, 10, 17, Some(60564.4), Some(22834.4)),
        ( 52.0,  139.7, 10, 17, Some(26701.2), Some(-12883.7)),
        ( 62.0,  -75.0, 10, 17, Some(76560.1), Some(43838.2)),
        ( 62.0,    0.0, 10, 17, Some(58631.2), Some(24371.5)),
        ( 62.0,  139.7, 10, 17, Some(25233.2), Some(-11850.0)),
        (-35.0,  -75.0, 10, 17, Some(85948.2), Some(36529.1)),
        (-35.0,    0.0, 10, 17, Some(66996.1), Some(18132.1)),
        (-35.0,  139.7, 10, 17, Some(31706.4), Some(-16116.7)),
        (  0.0,  -75.0, 12,  8, Some(61609.4), None),
        (  0.0,    0.0, 12,  8, Some(43017.5), None),
        (  0.0,  139.7, 12,  8, Some(8381.4), None),
        ( 35.0,  -75.0, 12,  8, Some(62647.7), None),
        ( 35.0,    0.0, 12,  8, Some(44299.8), None),
        ( 35.0,  139.7, 12,  8, Some(10114.1), Some(51750.9)),
        ( 52.0,  -75.0, 12,  8, Some(63498.6), None),
        ( 52.0,    0.0, 12,  8, Some(45351.5), Some(86231.8)),
        ( 52.0,  139.7, 12,  8, Some(11541.1), Some(50732.1)),
        ( 62.0,  -75.0, 12,  8, Some(64371.2), None),
        ( 62.0,    0.0, 12,  8, Some(46434.1), Some(85572.1)),
        ( 62.0,  139.7, 12,  8, Some(13024.7), Some(49634.8)),
        (-35.0,  -75.0, 12,  8, Some(60559.1), Some(18519.2)),
        (-35.0,    0.0, 12,  8, Some(41716.9), Some(161.8)),
        (-35.0,  139.7, 12,  8, Some(6619.2), None),
    ];

    #[test]
    fn riseset_vs_skyfield() {
        // Angular error of the analytic Moon (0.36 deg longitude, 0.2 deg
        // latitude, 1275 km range) bounds its upper-limb height error
        let model_err = 0.45f64.to_radians();
        let (mut max_jpl, mut max_lp) = (0.0f64, 0.0f64);
        for (lat, lon, month, day, rise, set) in SKYFIELD_RISESET_2024 {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            let t0 = Instant::from_date(2024, month, day).unwrap();
            let secs = |t: Option<Instant>| t.map(|t| (t - t0).as_seconds());
            let (jr, js) = riseset(&t0, &coord, true).unwrap();
            let (lr, ls) = riseset(&t0, &coord, false).unwrap();
            for (name, reference, jpl, lp) in [
                ("rise", rise, secs(jr), secs(lr)),
                ("set", set, secs(js), secs(ls)),
            ] {
                let case = format!("{name} {lat} {lon} 2024-{month}-{day}");
                // JPL: the same events, to about a second
                match (reference, jpl) {
                    (Some(r), Some(j)) => max_jpl = max_jpl.max((j - r).abs()),
                    (None, None) => {}
                    _ => panic!("{case}: JPL {jpl:?}, Skyfield {reference:?}"),
                }
                // Built-in: within 2 minutes; an event may be missing or
                // extra only where the other model's Moon is within the
                // analytic model's error of the horizon, i.e. a graze or an
                // event at the edge of the window
                match (reference, lp) {
                    (Some(r), Some(l)) => max_lp = max_lp.max((l - r).abs()),
                    (None, None) => {}
                    (Some(s), None) | (None, Some(s)) => {
                        let (h_lp, h_jpl) = heights(&(t0 + Duration::from_seconds(s)), &coord);
                        let h = if lp.is_some() { h_jpl } else { h_lp };
                        assert!(h.abs() < model_err, "{case}: {lp:?} vs {reference:?}");
                    }
                }
            }
        }
        println!("moon riseset vs Skyfield: JPL max {max_jpl:.2} s, built-in max {max_lp:.1} s");
        assert!(max_jpl < 2.0, "JPL max error {max_jpl} s");
        // The height error over the Moon's vertical rate at the horizon (at
        // most 0.25 cos(lat) deg/min) allows 2 minutes at the equator and
        // more further north; the largest on this grid is 88 s, at 62 deg N
        assert!(max_lp < 120.0, "built-in max error {max_lp} s");
    }

    /// USNO Astronomical Applications API, retrieved 2026-09-26:
    /// `https://aa.usno.navy.mil/api/rstt/oneday?date=2024-MM-DD&coords=LAT,LON&tz=TZ&dst=false`
    /// with `TZ = LON / 15`, so USNO's local day is the local mean day.
    /// `(lat, lon, tz, month, day, rise, set)`, local (hour, minute), rounded
    /// to the minute
    #[rustfmt::skip]
    const USNO_RISESET_2024: [UsnoRow; 7] = [
        ( 35.0,  -75.0,  -5,  3, 15, Some((9, 23)), None),
        ( 52.0,    0.0,   0,  1,  4, Some((0, 12)), Some((11, 36))),
        (  0.0,    0.0,   0, 10, 17, Some((17, 58)), Some((5, 29))),
        (-35.0,  150.0,  10,  6, 21, Some((15, 51)), Some((6, 18))),
        ( 62.0,   15.0,   1, 12,  8, Some((12, 54)), Some((23, 42))),
        ( 35.0,  135.0,   9,  9, 24, Some((22, 11)), Some((12, 47))),
        ( 52.0, -120.0,  -8,  7,  5, Some((2, 47)), Some((21, 4))),
    ];

    #[test]
    fn riseset_vs_usno() {
        let (mut max_jpl, mut max_lp) = (0.0f64, 0.0f64);
        for (lat, lon, tz, month, day, rise, set) in USNO_RISESET_2024 {
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            let t0 = Instant::from_date(2024, month, day).unwrap();
            let usno = |hm: Option<(i32, i32)>| hm.map(|(h, m)| ((h - tz) * 3600 + m * 60) as f64);
            let secs = |t: Option<Instant>| t.map(|t| (t - t0).as_seconds());
            for use_jpl in [true, false] {
                let (r, s) = riseset(&t0, &coord, use_jpl).unwrap();
                for (got, reference) in [(secs(r), usno(rise)), (secs(s), usno(set))] {
                    let case = format!("{lat} {lon} 2024-{month}-{day} jpl {use_jpl}");
                    let (Some(g), Some(u)) = (got, reference) else {
                        assert_eq!(got.is_none(), reference.is_none(), "{case}");
                        continue;
                    };
                    let max = if use_jpl { &mut max_jpl } else { &mut max_lp };
                    *max = max.max((g - u).abs());
                }
            }
        }
        println!("moon riseset vs USNO: JPL max {max_jpl:.1} s, built-in max {max_lp:.1} s");
        // USNO rounds to the minute
        assert!(max_jpl < 60.0, "JPL max error {max_jpl} s");
        assert!(max_lp < 120.0 + 30.0, "built-in max error {max_lp} s");
    }

    #[test]
    fn riseset_edge_cases() {
        let date = |m, d| Instant::from_date(2024, m, d).unwrap();
        for use_jpl in [true, false] {
            // Circumpolar and never-up days near the pole
            for (lat, m, d) in [(80.0, 1, 24), (80.0, 6, 21), (-80.0, 1, 24), (-80.0, 6, 21)] {
                let coord = ITRFCoord::from_geodetic_deg(lat, 30.0, 0.0);
                let (r, s) = riseset(&date(m, d), &coord, use_jpl).unwrap();
                assert_eq!((r, s), (None, None), "lat {lat} 2024-{m}-{d}");
            }
            // No moonrise on 2024-01-04 in Tokyo, no moonset on 2024-03-15
            // at Greenwich's longitude (see the Skyfield table)
            let tokyo = ITRFCoord::from_geodetic_deg(35.0, 139.7, 0.0);
            let (r, s) = riseset(&date(1, 4), &tokyo, use_jpl).unwrap();
            assert!(r.is_none() && s.is_some());
            let site = ITRFCoord::from_geodetic_deg(35.0, 0.0, 0.0);
            let (r, s) = riseset(&date(3, 15), &site, use_jpl).unwrap();
            assert!(r.is_some() && s.is_none());
            // Two moonrises in the day at 62 deg N on 2024-06-24 (Skyfield:
            // 00:07:32 and 23:59:51 UTC) and two moonsets on 2024-06-20
            // (00:06:11 and 23:52:31): the first is returned
            let site = ITRFCoord::from_geodetic_deg(62.0, 0.0, 0.0);
            let (r, _) = riseset(&date(6, 24), &site, use_jpl).unwrap();
            let first = Instant::from_datetime(2024, 6, 24, 0, 7, 32.0).unwrap();
            assert!((r.unwrap() - first).as_seconds().abs() < 120.0);
            let (_, s) = riseset(&date(6, 20), &site, use_jpl).unwrap();
            let first = Instant::from_datetime(2024, 6, 20, 0, 6, 11.0).unwrap();
            assert!((s.unwrap() - first).as_seconds().abs() < 120.0);

            for (lat, lon) in [(35.0, -75.0), (52.0, 0.0), (-35.0, 139.7)] {
                let coord = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
                // The time of day is ignored
                let ref_rs = riseset(&date(10, 17), &coord, use_jpl).unwrap();
                let late = Instant::from_datetime(2024, 10, 17, 23, 59, 59.999).unwrap();
                assert_eq!(riseset(&late, &coord, use_jpl).unwrap(), ref_rs);
                // Events lie in the local mean day, with the upper limb on
                // the horizon
                let start = date(10, 17) - Duration::from_days(lon / 360.0);
                for t in [ref_rs.0.unwrap(), ref_rs.1.unwrap()] {
                    let dt = (t - start).as_seconds();
                    assert!((0.0..86400.0).contains(&dt), "{lat} {lon} {t}");
                    let (h_lp, h_jpl) = heights(&t, &coord);
                    let h = if use_jpl { h_jpl } else { h_lp };
                    assert!(h.abs() < 1.0e-5, "{lat} {lon} {t}: {h}");
                }
            }
        }
    }

    /// Principal phases of 2024 from Skyfield 1.55 with DE440s:
    /// `almanac.find_discrete(ts.utc(2024, 1, 1), ts.utc(2025, 1, 1),
    /// almanac.moon_phases(eph))`, i.e. the apparent geocentric Moon - Sun
    /// longitude difference in the ecliptic of date reaching a multiple of
    /// 90°.  `(month, day, hour, minute, second UTC, phase)` with phase 0 =
    /// New, 1 = First Quarter, 2 = Full, 3 = Last Quarter
    #[rustfmt::skip]
    const SKYFIELD_PHASES_2024: [(i32, i32, i32, i32, f64, usize); 50] = [
        ( 1,  4,  3, 30, 27.1, 3),
        ( 1, 11, 11, 57, 24.6, 0),
        ( 1, 18,  3, 52, 36.5, 1),
        ( 1, 25, 17, 54,  0.4, 2),
        ( 2,  2, 23, 17, 58.7, 3),
        ( 2,  9, 22, 59, 10.9, 0),
        ( 2, 16, 15,  0, 56.4, 1),
        ( 2, 24, 12, 30, 25.9, 2),
        ( 3,  3, 15, 23, 29.5, 3),
        ( 3, 10,  9,  0, 26.3, 0),
        ( 3, 17,  4, 10, 43.3, 1),
        ( 3, 25,  7,  0, 19.6, 2),
        ( 4,  2,  3, 14, 43.8, 3),
        ( 4,  8, 18, 20, 51.5, 0),
        ( 4, 15, 19, 13,  7.1, 1),
        ( 4, 23, 23, 48, 58.8, 2),
        ( 5,  1, 11, 27, 16.5, 3),
        ( 5,  8,  3, 21, 56.1, 0),
        ( 5, 15, 11, 48,  0.1, 1),
        ( 5, 23, 13, 53,  8.5, 2),
        ( 5, 30, 17, 12, 40.4, 3),
        ( 6,  6, 12, 37, 44.3, 0),
        ( 6, 14,  5, 18, 27.1, 1),
        ( 6, 22,  1,  7, 52.9, 2),
        ( 6, 28, 21, 53, 24.9, 3),
        ( 7,  5, 22, 57, 24.2, 0),
        ( 7, 13, 22, 48, 49.0, 1),
        ( 7, 21, 10, 17,  9.3, 2),
        ( 7, 28,  2, 51, 33.7, 3),
        ( 8,  4, 11, 13,  3.8, 0),
        ( 8, 12, 15, 18, 47.8, 1),
        ( 8, 19, 18, 25, 48.8, 2),
        ( 8, 26,  9, 25, 51.5, 3),
        ( 9,  3,  1, 55, 35.4, 0),
        ( 9, 11,  6,  5, 39.7, 1),
        ( 9, 18,  2, 34, 28.2, 2),
        ( 9, 24, 18, 49, 52.8, 3),
        (10,  2, 18, 49, 16.8, 0),
        (10, 10, 18, 55,  8.6, 1),
        (10, 17, 11, 26, 24.4, 2),
        (10, 24,  8,  3,  5.5, 3),
        (11,  1, 12, 47,  8.6, 0),
        (11,  9,  5, 55, 28.5, 1),
        (11, 15, 21, 28, 31.0, 2),
        (11, 23,  1, 27, 55.8, 3),
        (12,  1,  6, 21, 25.3, 0),
        (12,  8, 15, 26, 37.3, 1),
        (12, 15,  9,  1, 41.2, 2),
        (12, 22, 22, 18, 11.3, 3),
        (12, 30, 22, 26, 47.9, 0),
    ];

    /// USNO "Phases of the Moon" for 2024, retrieved 2026-09-26 from
    /// `https://aa.usno.navy.mil/api/moon/phases/year?year=2024`:
    /// `(month, day, hour, minute UTC, phase)`, rounded to the minute
    #[rustfmt::skip]
    const USNO_PHASES_2024: [(i32, i32, i32, i32, usize); 50] = [
        ( 1,  4,  3, 30, 3),
        ( 1, 11, 11, 57, 0),
        ( 1, 18,  3, 52, 1),
        ( 1, 25, 17, 54, 2),
        ( 2,  2, 23, 18, 3),
        ( 2,  9, 22, 59, 0),
        ( 2, 16, 15,  1, 1),
        ( 2, 24, 12, 30, 2),
        ( 3,  3, 15, 23, 3),
        ( 3, 10,  9,  0, 0),
        ( 3, 17,  4, 11, 1),
        ( 3, 25,  7,  0, 2),
        ( 4,  2,  3, 15, 3),
        ( 4,  8, 18, 21, 0),
        ( 4, 15, 19, 13, 1),
        ( 4, 23, 23, 49, 2),
        ( 5,  1, 11, 27, 3),
        ( 5,  8,  3, 22, 0),
        ( 5, 15, 11, 48, 1),
        ( 5, 23, 13, 53, 2),
        ( 5, 30, 17, 13, 3),
        ( 6,  6, 12, 38, 0),
        ( 6, 14,  5, 18, 1),
        ( 6, 22,  1,  8, 2),
        ( 6, 28, 21, 53, 3),
        ( 7,  5, 22, 57, 0),
        ( 7, 13, 22, 49, 1),
        ( 7, 21, 10, 17, 2),
        ( 7, 28,  2, 51, 3),
        ( 8,  4, 11, 13, 0),
        ( 8, 12, 15, 19, 1),
        ( 8, 19, 18, 26, 2),
        ( 8, 26,  9, 26, 3),
        ( 9,  3,  1, 55, 0),
        ( 9, 11,  6,  5, 1),
        ( 9, 18,  2, 34, 2),
        ( 9, 24, 18, 50, 3),
        (10,  2, 18, 49, 0),
        (10, 10, 18, 55, 1),
        (10, 17, 11, 26, 2),
        (10, 24,  8,  3, 3),
        (11,  1, 12, 47, 0),
        (11,  9,  5, 55, 1),
        (11, 15, 21, 28, 2),
        (11, 23,  1, 28, 3),
        (12,  1,  6, 21, 0),
        (12,  8, 15, 26, 1),
        (12, 15,  9,  2, 2),
        (12, 22, 22, 18, 3),
        (12, 30, 22, 27, 0),
    ];

    #[test]
    fn phase_times_2024() {
        let start = Instant::from_date(2024, 1, 1).unwrap();
        let end = Instant::from_date(2025, 1, 1).unwrap();
        for use_jpl in [true, false] {
            let got = phase_times(&start, &end, use_jpl).unwrap();
            assert_eq!(got.len(), 50);
            let (mut max_sky, mut max_usno) = (0.0f64, 0.0f64);
            for (i, (phase, t)) in got.iter().enumerate() {
                let (mo, d, h, mi, s, p) = SKYFIELD_PHASES_2024[i];
                assert_eq!(*phase, PRINCIPAL_PHASES[p]);
                let sky = Instant::from_datetime(2024, mo, d, h, mi, s).unwrap();
                max_sky = max_sky.max((*t - sky).as_seconds().abs());
                let (mo, d, h, mi, p) = USNO_PHASES_2024[i];
                assert_eq!(*phase, PRINCIPAL_PHASES[p]);
                let usno = Instant::from_datetime(2024, mo, d, h, mi, 0.0).unwrap();
                max_usno = max_usno.max((*t - usno).as_seconds().abs());
            }
            println!(
                "moon phases 2024, use_jpl {use_jpl}: max {max_sky:.2} s vs Skyfield, \
                 {max_usno:.1} s vs USNO"
            );
            if use_jpl {
                assert!(max_sky < 0.5, "{max_sky} s vs Skyfield");
                // USNO rounds to the minute
                assert!(max_usno < 60.0, "{max_usno} s vs USNO");
            } else {
                // The analytic Moon's longitude error, 0.36 deg worst case,
                // over the 12.2 deg/day relative motion allows 42 minutes;
                // the 2024 maximum is 22.3 minutes
                assert!(max_sky < 1800.0, "{max_sky} s vs Skyfield");
                assert!(max_usno < 1800.0, "{max_usno} s vs USNO");
            }
        }
    }

    #[test]
    fn phase_times_consistency() {
        let start = Instant::from_date(2024, 1, 1).unwrap();
        let end = Instant::from_date(2024, 7, 1).unwrap();
        for use_jpl in [true, false] {
            let all = phase_times(&start, &end, use_jpl).unwrap();
            // In time order, cycling through the four phases
            for w in all.windows(2) {
                assert!(w[1].1 > w[0].1 + Duration::from_days(5.0));
                let i0 = PRINCIPAL_PHASES.iter().position(|p| *p == w[0].0).unwrap();
                assert_eq!(w[1].0, PRINCIPAL_PHASES[(i0 + 1) % 4]);
            }
            // Splitting the interval, including right at an event, neither
            // drops nor repeats an event
            for split in [all[7].1, all[7].1 + Duration::from_days(3.3)] {
                let mut parts = phase_times(&start, &split, use_jpl).unwrap();
                parts.extend(phase_times(&split, &end, use_jpl).unwrap());
                assert_eq!(parts.len(), all.len());
                for ((p0, t0), (p1, t1)) in all.iter().zip(parts.iter()) {
                    assert_eq!(p0, p1);
                    assert!((*t0 - *t1).as_seconds().abs() < 0.1);
                }
            }
            // next_phase finds the first event of that phase at or after
            // the time
            for (phase, t) in &all[1..] {
                let before = *t - Duration::from_days(2.0);
                let next = next_phase(&before, *phase, use_jpl).unwrap();
                assert!((next - *t).as_seconds().abs() < 0.1);
            }
            assert!(phase_times(&end, &start, use_jpl).unwrap().is_empty());
        }
        assert!(matches!(
            next_phase(&start, MoonPhase::WaxingGibbous, false),
            Err(Error::NotPrincipalPhase(MoonPhase::WaxingGibbous))
        ));
    }
}
