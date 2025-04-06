//!
//! Low-precision planetary ephemerides
//!
//! See: <https://ssd.jpl.nasa.gov/planets/approx_pos.html>
//! which references original paper at:
//! <https://www.researchgate.net/publication/232203657_Orbital_Ephemerides_of_the_Sun_Moon_and_Planets>
//!
//! Approximate uncertainties for the given date ranges are reported below
//! as stated in the JPL website.
//!
//!
//! For 1800 AD to 2050 AD:
//! |  Planet  | RA (arcsec) | Dec (arcsec) | Range (Mm) |
//! | -------- | ----------- | ------------ | ---------- |
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
//! |  Planet  | RA (arcsec) | Dec (arcsec) | Range (Mm) |
//! | -------- | ----------- | ------------ | ---------- |
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
use crate::TimeScale;

use anyhow::{bail, Result};

use nalgebra as na;
type Vec3 = na::Vector3<f64>;
type Quat = na::UnitQuaternion<f64>;

/// Returns the heliocentric position of a planet
///
/// # Arguments
///
/// * `body` - The planet to compute the position of
/// * `time` - The time at which to compute the position
///
/// # Returns
///
/// * `Vec3` - The heliocentric position of the planet
///
/// # Example
///
/// ```
/// use satkit::lpephem::heliocentric_pos;
/// use satkit::SolarSystem;
/// use satkit::Instant;
///
/// let time = Instant::from_date(2000, 1, 1);
/// let pos = heliocentric_pos(SolarSystem::Mars, &time).unwrap();
/// println!("Position of Mars: {}", pos);
/// ```
///
pub fn heliocentric_pos(body: SolarSystem, time: &Instant) -> Result<Vec3> {
    // Keplerian elements are provided separately and more accurately
    // for times in range of years 1800AD to 2050AD
    let tm0: Instant = Instant::from_date(-3000, 1, 1);
    let tm1: Instant = Instant::from_date(3000, 1, 1);
    let tmp0: Instant = Instant::from_date(1800, 1, 1);
    let tmp1: Instant = Instant::from_date(2050, 12, 31);
    let jcen = (time.as_jd_with_scale(TimeScale::TT) - 2451545.0) / 36525.0;

    #[allow(non_snake_case)]
    let (a, eccen, incl, l, wbar, Omega, terms) = {
        if time > &tmp0 && time < &tmp1 {
            let a: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.38709927,
                    0.20563593,
                    7.00497902,
                    252.25032350,
                    77.45779628,
                    48.33076593,
                ],
                SolarSystem::Venus => [
                    0.72333566,
                    0.00677672,
                    3.39467605,
                    181.97909950,
                    131.60246718,
                    76.67984255,
                ],
                SolarSystem::EMB => [
                    1.00000261,
                    0.01671123,
                    -0.00001531,
                    100.46457166,
                    102.93768193,
                    0.0,
                ],
                SolarSystem::Mars => [
                    1.52371034,
                    0.09339410,
                    1.84969142,
                    -4.55343205,
                    -23.94362959,
                    49.55953891,
                ],
                SolarSystem::Jupiter => [
                    5.20288700,
                    0.04838624,
                    1.30439695,
                    34.39644051,
                    14.72847983,
                    100.47390909,
                ],
                SolarSystem::Saturn => [
                    9.53667594,
                    0.05386179,
                    2.48599187,
                    49.95424423,
                    92.59887831,
                    113.66242448,
                ],
                SolarSystem::Uranus => [
                    19.18916464,
                    0.04725744,
                    0.77263783,
                    313.23810451,
                    170.95427630,
                    74.01692503,
                ],
                SolarSystem::Neptune => [
                    30.06992276,
                    0.00859048,
                    1.77004347,
                    -55.12002969,
                    44.96476227,
                    131.78422574,
                ],
                SolarSystem::Pluto => [
                    39.48211675,
                    0.2488273,
                    17.14001206,
                    238.92903833,
                    224.06891629,
                    110.30393684,
                ],
                _ => bail!("Invalid Body"),
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
                    -0.00196176,
                    -0.00004397,
                    -0.00242939,
                    428.48202785,
                    0.40805281,
                    0.04240589,
                ],
                SolarSystem::Neptune => [
                    0.00026291,
                    0.00005105,
                    0.00035372,
                    218.45945325,
                    -0.32241464,
                    -0.00508664,
                ],
                SolarSystem::Pluto => [
                    -0.00031596,
                    0.00005170,
                    0.00004818,
                    145.20780515,
                    -0.04062942,
                    -0.01183482,
                ],
                _ => bail!("Invalid Body"),
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
        } else if time > &tm0 && time < &tm1 {
            let a: [f64; 6] = match body {
                SolarSystem::Mercury => [
                    0.38709843,
                    0.20563661,
                    7.00559432,
                    252.25166724,
                    77.45771895,
                    48.33961819,
                ],
                SolarSystem::Venus => [
                    0.72332102,
                    0.00676399,
                    3.39777545,
                    181.97970850,
                    131.76755713,
                    76.67261496,
                ],
                SolarSystem::EMB => [
                    1.00000018,
                    0.01673163,
                    -0.00054346,
                    100.46691572,
                    102.93005885,
                    -5.11260389,
                ],
                SolarSystem::Mars => [
                    1.52371243,
                    0.09336511,
                    1.85181869,
                    -4.56813164,
                    -23.91744784,
                    49.71320984,
                ],
                SolarSystem::Jupiter => [
                    5.20248019,
                    0.04853590,
                    1.29861416,
                    34.33479152,
                    14.27495244,
                    100.29282654,
                ],
                SolarSystem::Saturn => [
                    9.54149883,
                    0.05550825,
                    2.49424102,
                    50.07571329,
                    92.86136063,
                    113.63998702,
                ],
                SolarSystem::Uranus => [
                    19.18797948,
                    0.04685740,
                    0.77298127,
                    314.20276625,
                    172.43404441,
                    73.96250215,
                ],
                SolarSystem::Neptune => [
                    30.06952752,
                    0.00895439,
                    1.77005520,
                    304.22289287,
                    46.68158724,
                    131.78635853,
                ],
                SolarSystem::Pluto => [
                    39.48686035,
                    0.24885238,
                    17.1410426,
                    238.96535011,
                    224.09702598,
                    110.30167986,
                ],
                _ => bail!("Invalid Body"),
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
                    -0.00020455,
                    -0.00001550,
                    -0.00180155,
                    428.49512595,
                    0.09266985,
                    0.05739699,
                ],
                SolarSystem::Neptune => [
                    0.00006447,
                    0.00000818,
                    0.00022400,
                    218.46515314,
                    0.01009938,
                    -0.00606302,
                ],
                SolarSystem::Pluto => [
                    0.00449751,
                    0.00006016,
                    0.00000501,
                    145.18042903,
                    -0.00968827,
                    -0.00809981,
                ],
                _ => bail!("Invalid Body"),
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
            bail!("Time out of range");
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
        Some([b, c, s, f]) => {
            (b * jcen).mul_add(jcen, l - wbar)
                + (c * (f * jcen).cos()).to_degrees()
                + (s * (f * jcen).sin()).to_degrees()
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
    let mut enrad = eccen.mul_add(mrad.sin(), mrad);
    loop {
        let deltamrad = mrad - eccen.mul_add(-enrad.sin(), enrad);
        let deltaerad = deltamrad / eccen.mul_add(-enrad.cos(), 1.0);
        enrad += deltaerad;
        if (deltaerad / enrad).abs() < 1.0e-8 {
            break;
        }
    }
    // Get heliocentric coordinates in orbital plane
    let xprime = a * (enrad.cos() - eccen);
    let yprime = a * eccen.mul_add(-eccen, 1.0).sqrt() * enrad.sin();
    let rprime = Vec3::new(xprime, yprime, 0.0);
    let recl = Quat::from_axis_angle(&Vec3::z_axis(), Omega.to_radians())
        * Quat::from_axis_angle(&Vec3::x_axis(), incl.to_radians())
        * Quat::from_axis_angle(&Vec3::z_axis(), w.to_radians())
        * rprime;

    // Rotate to the equatorial plane
    // Obliquity at J2000
    let obliquity = 1.21e-11f64
        .mul_add(
            jcen.powi(5),
            1.6e-10f64.mul_add(
                jcen.powi(4),
                5.565e-7f64.mul_add(
                    jcen.powi(3),
                    (5.086e-8 * jcen).mul_add(-jcen, 0.0130102f64.mul_add(-jcen, 23.439279)),
                ),
            ),
        )
        .to_radians();

    Ok(Quat::from_axis_angle(&Vec3::x_axis(), obliquity) * recl * crate::consts::AU)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jplephem;

    const fn errors_precise(planet: &SolarSystem) -> (usize, usize, usize) {
        match planet {
            SolarSystem::Mercury => (15, 1, 1),
            SolarSystem::Venus => (20, 1, 4),
            SolarSystem::EMB => (20, 1, 4),
            SolarSystem::Mars => (40, 2, 25),
            SolarSystem::Jupiter => (400, 10, 1600),
            SolarSystem::Saturn => (600, 25, 1500),
            SolarSystem::Uranus => (50, 2, 1000),
            SolarSystem::Neptune => (10, 1, 200),
            SolarSystem::Pluto => (5, 2, 300),
            _ => (0, 0, 0),
        }
    }

    #[test]
    fn compare_with_jplephem() {
        //               1800 AD - 2050             AD	3000 BC - 3000 AD
        //         λ (asec) : ϕ (asec)	: ρ (Mm) :: λ (asec): ϕ( asec) : ρ (Mm)
        // Mercury	15	1	1	20	15	1
        // Venus	20	1	4	40	30	8
        // EM Bary	20	8	6	40	15	15
        // Mars	40	2	25	100	40	30
        // Jupiter	400	10	600	600	100	1000
        // Saturn	600	25	1500	1000	100	4000
        // Uranus	50	2	1000	2000	30	8000
        // Neptune	10	1	200	400	15	4000
        // Pluto	5	2	300	400	100	2500

        let planets = [
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

        for planet in planets {
            //let time = Instant::from_date(2000, 1, 1);
            let time = Instant::from_datetime(2010, 1, 1, 12, 0, 0.0);
            let psun = jplephem::barycentric_pos(SolarSystem::Sun, &time).unwrap();
            let p2 = jplephem::barycentric_pos(planet, &time).unwrap() - psun;
            let p1 = heliocentric_pos(planet, &time).unwrap();
            let lambda1 = f64::atan2(p1[1], p1[0]);
            let lambda2 = f64::atan2(p2[1], p2[0]);
            let phi1 = f64::asin(p1[2] / p1.norm());
            let phi2 = f64::asin(p2[2] / p2.norm());
            let lerr = (lambda1 - lambda2).abs().to_degrees() * 3600.0;
            let perr = (phi1 - phi2).abs().to_degrees() * 3600.0;
            let rerr = (p1.norm() - p2.norm()).abs() * 1.0e-6;
            let (lerr_approx, perr_approx, rerr_approx) = errors_precise(&planet);

            assert!(lerr < f64::max(lerr_approx as f64, 15.0) * 8.0);
            assert!(perr < f64::max(perr_approx as f64, 15.0) * 8.0);
            assert!(rerr < rerr_approx as f64 * 12.0);
        }
    }
}
