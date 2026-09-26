//!
//! Low-precision planetary ephemerides
//!
//! See: <https://ssd.jpl.nasa.gov/planets/approx_pos.html>
//! which references original paper at:
//! <https://www.researchgate.net/publication/232203657_Orbital_Ephemerides_of_the_Sun_Moon_and_Planets>
//!
//! Valid bodies are Mercury through Pluto plus the Earth-Moon barycenter
//! ([`SolarSystem::EMB`]); the Sun and Moon are rejected with
//! [`Error::InvalidBody`](super::Error::InvalidBody).
//!
//! Approximate errors for the given date ranges are reported below as
//! stated on the JPL website, in heliocentric J2000-ecliptic longitude λ,
//! latitude φ, and range.  Against DE440 the RMS error is within these
//! figures; single excursions reach up to ~4x them.
//!
//! The Uranus, Neptune and Pluto elements follow each planet's orbit about
//! the solar-system barycenter.  The Sun's own motion about the barycenter
//! (up to ~0.01 AU, driven by Jupiter and Saturn) is not modeled, so from
//! 1800 to 2050 their heliocentric errors exceed the table: up to ~2 arcmin
//! in λ and ~2.3 million km in range.
//!
//! For 1800 AD to 2050 AD:
//! |  Planet  | λ (arcsec) | φ (arcsec) | Range (Mm) |
//! | -------- | ---------- | ---------- | ---------- |
//! | Mercury  | 15          | 1            | 1          |
//! | Venus    | 20          | 1            | 4          |
//! | EM Bary  | 20          | 8            | 6          |
//! | Mars     | 40          | 2            | 25         |
//! | Jupiter  | 400         | 10           | 600        |
//! | Saturn   | 600         | 25           | 1500       |
//! | Uranus   | 50          | 2            | 1000       |
//! | Neptune  | 10          | 1            | 200        |
//! | Pluto    | 5           | 2            | 300        |
//!
//! From 3000 BC to 3000 AD:
//! |  Planet  | λ (arcsec) | φ (arcsec) | Range (Mm) |
//! | -------- | ---------- | ---------- | ---------- |
//! | Mercury  | 20          | 15           | 1          |
//! | Venus    | 40          | 30           | 8          |
//! | EM Bary  | 40          | 15           | 15         |
//! | Mars     | 100         | 40           | 30         |
//! | Jupiter  | 600         | 100          | 1000       |
//! | Saturn   | 1000        | 100          | 4000       |
//! | Uranus   | 2000        | 30           | 8000       |
//! | Neptune  | 400         | 15           | 4000       |
//! | Pluto    | 400         | 100          | 2500       |
//!

use crate::Instant;
use crate::SolarSystem;
use crate::TimeLike;
use crate::TimeScale;

use super::{Error, Result};
use crate::mathtypes::*;

/// Returns the approximate heliocentric position of a planet
///
/// Keplerian-element approximation of Standish & Williams
/// (<https://ssd.jpl.nasa.gov/planets/approx_pos.html>): the 1800 AD -
/// 2050 AD element set inside that span, the 3000 BC - 3000 AD set
/// (with the extra mean-anomaly terms for Jupiter through Pluto) outside it.
///
/// Approximate errors, in heliocentric J2000-ecliptic longitude / latitude /
/// range, range from 15" / 1" / 1000 km (Mercury, 1800 - 2050) to
/// 2000" / 30" / 8 million km (Uranus, 3000 BC - 3000 AD); see JPL's table.
/// The Uranus, Neptune and Pluto elements follow the orbit about the
/// solar-system barycenter, so from 1800 to 2050 their heliocentric errors
/// are dominated by the Sun's unmodeled barycentric motion (up to ~2 arcmin
/// and ~2.3 million km).
///
/// # Arguments
///
/// * `body` - Mercury through Pluto, or the Earth-Moon barycenter
///   ([`SolarSystem::EMB`])
/// * `time` - The time at which to compute the position
///
/// # Returns
///
/// * `Vector3` - The heliocentric position of the body in the ICRF (J2000
///   equatorial) frame, meters
///
/// # Errors
///
/// * [`Error::InvalidBody`] for the Sun or the Moon
/// * [`Error::TimeOutOfRange`] outside 3000 BC - 3000 AD
///
/// # Example
///
/// ```
/// use satkit::lpephem::heliocentric_pos;
/// use satkit::SolarSystem;
/// use satkit::Instant;
///
/// let time = Instant::from_date(2000, 1, 1).unwrap();
/// let pos = heliocentric_pos(SolarSystem::Mars, &time).unwrap();
/// println!("Position of Mars: {}", pos);
/// ```
///
pub fn heliocentric_pos<T: TimeLike>(body: SolarSystem, time: &T) -> Result<Vector3> {
    let time = time.as_instant();
    // Keplerian elements are provided separately and more accurately
    // for times in range of years 1800AD to 2050AD
    let tm0: Instant = Instant::from_date(-3000, 1, 1)?;
    let tm1: Instant = Instant::from_date(3000, 1, 1)?;
    let tmp0: Instant = Instant::from_date(1800, 1, 1)?;
    let tmp1: Instant = Instant::from_date(2050, 12, 31)?;
    let jcen = (time.as_jd_with_scale(TimeScale::TT) - 2451545.0) / 36525.0;

    #[allow(non_snake_case)]
    let (a, eccen, incl, l, wbar, Omega, terms) = {
        if time > tmp0 && time < tmp1 {
            let a: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
                ],
                SolarSystem::Venus => [
                    0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
                ],
                SolarSystem::EMB => [
                    1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
                ],
                SolarSystem::Mars => [
                    1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
                ],
                SolarSystem::Jupiter => [
                    5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
                ],
                SolarSystem::Saturn => [
                    9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448,
                ],
                SolarSystem::Uranus => [
                    19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503,
                ],
                SolarSystem::Neptune => [
                    30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574,
                ],
                SolarSystem::Pluto => [
                    39.48211675, 0.2488273, 17.14001206, 238.92903833, 224.06891629, 110.30393684,
                ],
                _ => return Err(Error::InvalidBody),
            };

            let adot: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.00000037,
                    0.00001906,
                    -0.00594749,
                    149472.67411175,
                    0.16047689,
                    -0.12534081,
                ],
                SolarSystem::Venus => [
                    0.00000390,
                    -0.00004107,
                    -0.00078890,
                    58517.81538729,
                    0.00268329,
                    -0.27769418,
                ],
                SolarSystem::EMB => [
                    0.00000562,
                    -0.00004392,
                    -0.01294668,
                    35999.37244981,
                    0.32327364,
                    0.0,
                ],
                SolarSystem::Mars => [
                    0.00001847,
                    0.00007882,
                    -0.00813131,
                    19140.30268499,
                    0.44441088,
                    -0.29257343,
                ],
                SolarSystem::Jupiter => [
                    -0.00011607,
                    -0.00013253,
                    -0.00183714,
                    3034.74612775,
                    0.21252668,
                    0.20469106,
                ],
                SolarSystem::Saturn => [
                    -0.00125060,
                    -0.00050991,
                    0.00193609,
                    1222.49362201,
                    -0.41897216,
                    -0.28867794,
                ],
                SolarSystem::Uranus => [
                    -0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589,
                ],
                SolarSystem::Neptune => [
                    0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664,
                ],
                SolarSystem::Pluto => [
                    -0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482,
                ],
                _ => return Err(Error::InvalidBody),
            };
            // Julian century
            (
                jcen.mul_add(adot[0], a[0]),
                jcen.mul_add(adot[1], a[1]),
                jcen.mul_add(adot[2], a[2]),
                jcen.mul_add(adot[3], a[3]),
                jcen.mul_add(adot[4], a[4]),
                jcen.mul_add(adot[5], a[5]),
                None,
            )
        } else if time > tm0 && time < tm1 {
            let a: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.38709843, 0.20563661, 7.00559432, 252.25166724, 77.45771895, 48.33961819,
                ],
                SolarSystem::Venus => [
                    0.72332102, 0.00676399, 3.39777545, 181.97970850, 131.76755713, 76.67261496,
                ],
                SolarSystem::EMB => [
                    1.00000018, 0.01673163, -0.00054346, 100.46691572, 102.93005885, -5.11260389,
                ],
                SolarSystem::Mars => [
                    1.52371243, 0.09336511, 1.85181869, -4.56813164, -23.91744784, 49.71320984,
                ],
                SolarSystem::Jupiter => [
                    5.20248019, 0.04853590, 1.29861416, 34.33479152, 14.27495244, 100.29282654,
                ],
                SolarSystem::Saturn => [
                    9.54149883, 0.05550825, 2.49424102, 50.07571329, 92.86136063, 113.63998702,
                ],
                SolarSystem::Uranus => [
                    19.18797948, 0.04685740, 0.77298127, 314.20276625, 172.43404441, 73.96250215,
                ],
                SolarSystem::Neptune => [
                    30.06952752, 0.00895439, 1.77005520, 304.22289287, 46.68158724, 131.78635853,
                ],
                SolarSystem::Pluto => [
                    39.48686035, 0.24885238, 17.1410426, 238.96535011, 224.09702598, 110.30167986,
                ],
                _ => return Err(Error::InvalidBody),
            };
            let adot: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.00000000,
                    0.00002123,
                    -0.00590158,
                    149472.67486623,
                    0.15940013,
                    -0.12214182,
                ],
                SolarSystem::Venus => [
                    -0.00000026,
                    -0.00005107,
                    0.00043494,
                    58517.81560260,
                    0.05679648,
                    -0.27274174,
                ],
                SolarSystem::EMB => [
                    -0.00000003,
                    -0.00003661,
                    -0.01337178,
                    35999.37306329,
                    0.31795260,
                    -0.24123856,
                ],
                SolarSystem::Mars => [
                    0.00000097,
                    0.00009149,
                    -0.00724757,
                    19140.29934243,
                    0.45223625,
                    -0.26852431,
                ],
                SolarSystem::Jupiter => [
                    -0.00002864,
                    0.00018026,
                    -0.00322699,
                    3034.90371757,
                    0.18199196,
                    0.13024619,
                ],
                SolarSystem::Saturn => [
                    -0.00003065,
                    -0.00032044,
                    0.00451969,
                    1222.11494724,
                    0.54179478,
                    -0.25015002,
                ],
                SolarSystem::Uranus => [
                    -0.00020455, -0.00001550, -0.00180155, 428.49512595, 0.09266985, 0.05739699,
                ],
                SolarSystem::Neptune => [
                    0.00006447, 0.00000818, 0.00022400, 218.46515314, 0.01009938, -0.00606302,
                ],
                SolarSystem::Pluto => [
                    0.00449751, 0.00006016, 0.00000501, 145.18042903, -0.00968827, -0.00809981,
                ],
                _ => return Err(Error::InvalidBody),
            };
            let error_terms: Option<[f64; 4]> = match body {
                SolarSystem::Jupiter => Some([-0.00012452, 0.06064060, -0.35635438, 38.35125000]),
                SolarSystem::Saturn => Some([0.00025899, -0.13434469, 0.87320147, 38.35125000]),
                SolarSystem::Uranus => Some([0.00058331, -0.97731848, 0.17689245, 7.67025000]),
                SolarSystem::Neptune => Some([-0.00041348, 0.68346318, -0.10162547, 7.67025000]),
                SolarSystem::Pluto => Some([-0.01262724, 0.0, 0.0, 0.0]),
                _ => None,
            };
            (
                jcen.mul_add(adot[0], a[0]),
                jcen.mul_add(adot[1], a[1]),
                jcen.mul_add(adot[2], a[2]),
                jcen.mul_add(adot[3], a[3]),
                jcen.mul_add(adot[4], a[4]),
                jcen.mul_add(adot[5], a[5]),
                error_terms,
            )
        } else {
            return Err(Error::TimeOutOfRange);
        }
    };

    // the 6 kepler elements computed above are:
    // a = semi-major axis, in AU
    // e = eccentricity
    // i = inclination in degrees
    // L = mean longitude at epoch, in degrees
    // wbar = longitude of perihelion, in degrees
    // Omega = longitude of the ascending node, in degrees

    // Argument of perihelion
    let w = wbar - Omega;
    // Mean anomaly
    let mut m = match terms {
        None => l - wbar,
        // Extra terms for Jupiter through Pluto (JPL Table 2b): b, c and s
        // are in degrees, f in degrees per Julian century
        Some([b, c, s, f]) => {
            let (fsin, fcos) = (f * jcen).to_radians().sin_cos();
            (b * jcen).mul_add(jcen, l - wbar) + c.mul_add(fcos, s * fsin)
        }
    };
    // Get m into range [-180, 180]
    m %= 360.0;
    if m > 180.0 {
        m -= 360.0;
    }
    if m <= -180.0 {
        m += 360.0;
    }
    // Convert to radians
    let mrad = m.to_radians();

    // Get the eccentric anomaly
    let enrad = eccentric_anomaly(mrad, eccen);
    // Get heliocentric coordinates in orbital plane
    let xprime = a * (enrad.cos() - eccen);
    let yprime = a * eccen.mul_add(-eccen, 1.0).sqrt() * enrad.sin();
    let rprime = numeris::vector![xprime, yprime, 0.0];
    let recl = Quaternion::rotz(Omega.to_radians())
        * Quaternion::rotx(incl.to_radians())
        * Quaternion::rotz(w.to_radians())
        * rprime;

    // Rotate from the J2000 ecliptic to the ICRF equator.  The elements are
    // referred to the mean ecliptic and equinox of J2000, so this uses the
    // fixed J2000 obliquity, not the obliquity of date
    const OBLIQUITY_J2000_DEG: f64 = 23.43928;

    Ok(Quaternion::rotx(OBLIQUITY_J2000_DEG.to_radians()) * recl * crate::consts::AU)
}

/// Solve Kepler's equation `M = E - e sin(E)` for the eccentric anomaly by
/// Newton iteration, with an absolute tolerance and an iteration cap.
/// Angles in radians.
fn eccentric_anomaly(mrad: f64, eccen: f64) -> f64 {
    let mut enrad = eccen.mul_add(mrad.sin(), mrad);
    for _ in 0..50 {
        let deltamrad = mrad - eccen.mul_add(-enrad.sin(), enrad);
        let deltaerad = deltamrad / eccen.mul_add(-enrad.cos(), 1.0);
        enrad += deltaerad;
        if deltaerad.abs() < 1.0e-12 {
            break;
        }
    }
    enrad
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jplephem;
    use crate::Duration;

    const PLANETS: [SolarSystem; 9] = [
        SolarSystem::Mercury,
        SolarSystem::Venus,
        SolarSystem::EMB,
        SolarSystem::Mars,
        SolarSystem::Jupiter,
        SolarSystem::Saturn,
        SolarSystem::Uranus,
        SolarSystem::Neptune,
        SolarSystem::Pluto,
    ];

    /// JPL's published approximate errors for 1800 AD - 2050 AD:
    /// heliocentric ecliptic longitude (arcsec), latitude (arcsec), range (Mm)
    const fn errors_1800_2050(planet: SolarSystem) -> (f64, f64, f64) {
        match planet {
            SolarSystem::Mercury => (15.0, 1.0, 1.0),
            SolarSystem::Venus => (20.0, 1.0, 4.0),
            SolarSystem::EMB => (20.0, 8.0, 6.0),
            SolarSystem::Mars => (40.0, 2.0, 25.0),
            SolarSystem::Jupiter => (400.0, 10.0, 600.0),
            SolarSystem::Saturn => (600.0, 25.0, 1500.0),
            SolarSystem::Uranus => (50.0, 2.0, 1000.0),
            SolarSystem::Neptune => (10.0, 1.0, 200.0),
            SolarSystem::Pluto => (5.0, 2.0, 300.0),
            _ => (0.0, 0.0, 0.0),
        }
    }

    /// JPL's published approximate errors for 3000 BC - 3000 AD, same units
    const fn errors_3000bc_3000ad(planet: SolarSystem) -> (f64, f64, f64) {
        match planet {
            SolarSystem::Mercury => (20.0, 15.0, 1.0),
            SolarSystem::Venus => (40.0, 30.0, 8.0),
            SolarSystem::EMB => (40.0, 15.0, 15.0),
            SolarSystem::Mars => (100.0, 40.0, 30.0),
            SolarSystem::Jupiter => (600.0, 100.0, 1000.0),
            SolarSystem::Saturn => (1000.0, 100.0, 4000.0),
            SolarSystem::Uranus => (2000.0, 30.0, 8000.0),
            SolarSystem::Neptune => (400.0, 15.0, 4000.0),
            SolarSystem::Pluto => (400.0, 100.0, 2500.0),
            _ => (0.0, 0.0, 0.0),
        }
    }

    /// Heliocentric J2000-ecliptic longitude, latitude (radians) and range
    fn ecliptic(v: &Vector3) -> (f64, f64, f64) {
        let e = Quaternion::rotx(-23.43928f64.to_radians()) * v;
        (e[1].atan2(e[0]), (e[2] / e.norm()).asin(), e.norm())
    }

    /// RMS and maximum errors in heliocentric ecliptic longitude (arcsec),
    /// latitude (arcsec) and range (Mm) against the JPL ephemeris, sampled
    /// every 7.3 days in [t0, t1).  With `barycentric`, the reference is the
    /// planet's position about the solar-system barycenter instead of the Sun.
    fn errors(
        planet: SolarSystem,
        t0: Instant,
        t1: Instant,
        barycentric: bool,
    ) -> ([f64; 3], [f64; 3]) {
        let (mut rms, mut max, mut n) = ([0.0f64; 3], [0.0f64; 3], 0.0);
        let mut t = t0;
        while t < t1 {
            let mut pjpl = jplephem::barycentric_pos(planet, &t).unwrap();
            if !barycentric {
                pjpl -= jplephem::barycentric_pos(SolarSystem::Sun, &t).unwrap();
            }
            let (l2, p2, r2) = ecliptic(&pjpl);
            let (l1, p1, r1) = ecliptic(&heliocentric_pos(planet, &t).unwrap());
            let dl = (l1 - l2 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            let e = [
                dl.to_degrees() * 3600.0,
                (p1 - p2).to_degrees() * 3600.0,
                (r1 - r2) * 1.0e-6,
            ];
            for i in 0..3 {
                rms[i] += e[i] * e[i];
                max[i] = max[i].max(e[i].abs());
            }
            n += 1.0;
            t += Duration::from_days(7.3);
        }
        (rms.map(|s| (s / n).sqrt()), max)
    }

    /// JPL's table gives "approximate" errors.  Against DE440 the RMS error
    /// is within the table value for every body and coordinate; the largest
    /// single excursion reaches ~3.6x it (Mercury latitude, 1800 - 2050).
    fn check(
        planet: SolarSystem,
        span: &str,
        (rms, max): ([f64; 3], [f64; 3]),
        tol: (f64, f64, f64),
    ) {
        let tol = [tol.0, tol.1, tol.2];
        println!("{planet:?} {span}: rms {rms:.1?} max {max:.1?} table {tol:?}");
        for i in 0..3 {
            assert!(
                rms[i] <= tol[i] && max[i] <= 4.0 * tol[i],
                "{planet:?} {span}: coordinate {i} rms {:.2} max {:.2} vs table {}",
                rms[i],
                max[i],
                tol[i]
            );
        }
    }

    #[test]
    fn compare_with_jplephem_1800_2050() {
        let t0 = Instant::from_date(1800, 1, 2).unwrap();
        let t1 = Instant::from_date(2050, 12, 30).unwrap();
        for planet in PLANETS {
            // The Uranus, Neptune and Pluto elements follow the orbit about
            // the barycenter: the Sun's own ~0.01 AU motion about it (from
            // Jupiter and Saturn) is not in the model and dominates their
            // heliocentric error over this span (see the module docs)
            let barycentric = matches!(
                planet,
                SolarSystem::Uranus | SolarSystem::Neptune | SolarSystem::Pluto
            );
            check(
                planet,
                "1800-2050",
                errors(planet, t0, t1, barycentric),
                errors_1800_2050(planet),
            );
        }
    }

    #[test]
    fn compare_with_jplephem_extended() {
        // DE440 spans 1550 - 2650; test the parts of that span outside
        // 1800 - 2050, where the 3000 BC - 3000 AD elements (and the extra
        // mean-anomaly terms for Jupiter through Pluto) apply
        for (span, t0, t1) in [
            (
                "1550-1800",
                Instant::from_date(1550, 1, 2).unwrap(),
                Instant::from_date(1799, 12, 30).unwrap(),
            ),
            (
                "2051-2650",
                Instant::from_date(2051, 1, 2).unwrap(),
                Instant::from_date(2649, 12, 30).unwrap(),
            ),
        ] {
            for planet in PLANETS {
                check(
                    planet,
                    span,
                    errors(planet, t0, t1, false),
                    errors_3000bc_3000ad(planet),
                );
            }
        }
    }

    #[test]
    fn zero_mean_anomaly_converges() {
        // M == 0 exactly made the old relative convergence test 0/0 = NaN,
        // so the Newton loop never exited
        assert_eq!(eccentric_anomaly(0.0, 0.2), 0.0);
        for e in [0.0, 0.2, 0.9] {
            for m in [-3.0, -1.0e-300, 1.0e-300, 0.5, 3.1] {
                let ea = eccentric_anomaly(m, e);
                assert!((ea - e * ea.sin() - m).abs() < 1.0e-12, "e={e} m={m}");
            }
        }

        // An instant at which Mercury's mean anomaly is exactly zero
        // (found by the review); run with a timeout so a regression fails
        // instead of hanging the test suite
        let t = Instant::from_rfc3339("1803-08-04T18:23:43.274833Z").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(heliocentric_pos(SolarSystem::Mercury, &t).map(|p| p.norm()));
        });
        let r = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("heliocentric_pos did not return")
            .unwrap();
        assert!(r.is_finite());
    }

    #[test]
    fn invalid_bodies() {
        let t = Instant::from_date(2020, 1, 1).unwrap();
        for body in [SolarSystem::Sun, SolarSystem::Moon] {
            assert!(matches!(
                heliocentric_pos(body, &t),
                Err(Error::InvalidBody)
            ));
        }
    }
}
