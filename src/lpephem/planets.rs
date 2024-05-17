//! 
//! Low-precision planetary ephemerides
//! 
//! See: https://ssd.jpl.nasa.gov/planets/approx_pos.html
//! 
//! 
//! Approximate uncertainties for the given date ranges are reported below
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
//! 



use crate::SolarSystem;
use crate::AstroTime;
use crate::SKResult;
use crate::TimeScale;

use nalgebra as na;
type Vec3 = na::Vector3<f64>;
type Quat = na::UnitQuaternion<f64>;

pub fn barycentric_pos2(body: SolarSystem, time: &AstroTime) -> SKResult<Vec3> {
    // Appendix D.4 in Vallado
    
    let t: f64 = (time.to_jd(TimeScale::UT1) - 2451545.0)/36525.0;
    const RAD2DEG: f64 = 180.0/std::f64::consts::PI;

    #[allow(non_snake_case)]
    let (a, eccen, incl, Omega, wbar, l)= match body {
        SolarSystem::Mercury => (
            0.387098310,
            0.20563175 + 0.000020506*t - 2.84e-8*t*t - 1.7e-10*t*t*t,
            7.004986 - 0.0059516*t + 8.1e-7*t*t + 4.1e8*t*t*t,
            48.330893 - 0.1254229 - 8.833e-5*t*t-1.96e-7*t*t*t,
            77.456119 + 0.1588643*t - 1.343e-5*t*t+3.9e-8*t*t*t,
            252.250906 + 149472.6746358*t - 5.35e-6*t*t + 2.0e-9*t*t*t
        ),
        SolarSystem::Jupiter => (
            5.202603191 + 1.913e-7 * t,
            0.04849485 + 1.63244e-4*t - 4.719e-7*t*t - 1.97e-9*t*t*t,
            1.303270 - 1.9872e-3*t + 3.318e-5*t*t + 9.2e-8*t*t*t,
            100.464441 + 0.1766828*t + 9.0387e-4*t*t - 7.032e-6*t*t*t,
            14.331309 + 0.2155525*t + 7.2252e-4*t*t - 4.590e-6*t*t*t,
            34.351484 + 3034.9056746*t - 8.501e-5*t*t + 4.0e-9*t*t*t,
        ),
        _ => return Err("Invalid Body".into())
    };
    println!("t = {}", t);
    println!("{} {} {} {} {} {}", a, eccen, incl, Omega, wbar, l);
    let w = wbar - Omega;
    let m = l - wbar;
    println!("w = {} m = {}", w, m);
    

    // Convert to radians
    let mrad = (m%360.0) / RAD2DEG;
    #[allow(non_snake_case)]
    let mut E = match (mrad > std::f64::consts::PI) || ((mrad < 0.0) && (mrad > -1.0*std::f64::consts::PI)) {
        true => mrad - eccen,
        false => mrad + eccen
    };
    loop {
        let delta_e: f64 = (mrad - E + eccen*E.sin())/(1.0 - eccen*E.cos());
        E += delta_e;
        if delta_e.abs() < 1.0e-8 {
            break;
        }
    }
    println!("E = {}", E*RAD2DEG + 360.0);
    let nu = f64::asin((E.cos() - eccen)/(1.0-eccen*E.cos()));
    println!("nu = {}", nu*RAD2DEG + 360.0);
    let p = a * (1.0 - eccen*eccen);
    println!("p = {}", p);
    let den = 1.0 + eccen * nu.cos();
    let rpqw = Vec3::new(nu.cos(), nu.sin(), 0.0) * p/den;
    println!("rpqw = {}", rpqw.transpose());
    let rijk = 
        Quat::from_axis_angle(&Vec3::z_axis(), Omega/RAD2DEG) * 
        Quat::from_axis_angle(&Vec3::x_axis(), incl/RAD2DEG) * 
        Quat::from_axis_angle(&Vec3::z_axis(), w/RAD2DEG) * rpqw;
    println!("rijk = {}", rijk.transpose());
    let obliquity = (23.439279 - 0.0130102*t - 5.086e-8*t*t + 5.565e-7*t.powi(3)
        + 1.6e-10*t.powi(4) + 1.21e-11*t.powi(5)) / RAD2DEG;
    println!("obliquity = {}", obliquity* RAD2DEG);
    println!("rxyz = {}", (Quat::from_axis_angle(&Vec3::x_axis(), -obliquity) *  rijk).transpose() );
    Ok(Quat::from_axis_angle(&Vec3::x_axis(), obliquity) *  rijk * crate::consts::AU)
    

}

pub fn barycentric_pos(body: SolarSystem, time: &AstroTime) -> SKResult<Vec3> {
    // Keplerian elements are provided seaparately and more accurately
    // for times in range of years 1800AD to 2050AD
    let tm0: AstroTime = AstroTime::from_date(-3000, 1, 1);
    let tm1: AstroTime = AstroTime::from_date(3000, 1, 1);
    let tmp0: AstroTime = AstroTime::from_date(1800, 1, 1);
    let tmp1: AstroTime = AstroTime::from_date(2050, 12, 31);
    let jcen = (time.to_jd(TimeScale::TT) - 2451545.0)/36525.0;
    const RAD2DEG: f64 = 180.0/std::f64::consts::PI;

    #[allow(non_snake_case)]
    let (a, eccen, incl, l, wbar, Omega, terms) = {
        if time > &tmp0 && time < &tmp1 {
            let a: [f64;6] = match body {
                SolarSystem::Mercury => [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                SolarSystem::Venus => [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
                SolarSystem::EMB => [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
                SolarSystem::Mars => [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
                SolarSystem::Jupiter => [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                SolarSystem::Saturn => [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
                SolarSystem::Uranus => [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
                SolarSystem::Neptune => [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
                _ => return Err("Invalid Body".into()),
            };
            let adot: [f64;6] = match body {
                SolarSystem::Mercury => [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
                SolarSystem::Venus => [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
                SolarSystem::EMB => [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
                SolarSystem::Mars => [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
                SolarSystem::Jupiter => [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
                SolarSystem::Saturn => [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
                SolarSystem::Uranus => [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
                SolarSystem::Neptune => [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
                _ => return Err("Invalid Body".into()),
            };
            // Julian century
            (
                a[0] + jcen * adot[0], 
                a[1] + jcen * adot[1], 
                a[2] + jcen * adot[2], 
                a[3] + jcen * adot[3], 
                a[4] + jcen * adot[4], 
                a[5] + jcen * adot[5],
                None
            )
        }
        else if time > &tm0 && time < &tm1 {
            let a: [f64;6] = match body {
                SolarSystem::Mercury => [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                SolarSystem::Venus => [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
                SolarSystem::EMB => [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
                SolarSystem::Mars => [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
                SolarSystem::Jupiter => [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                SolarSystem::Saturn => [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
                SolarSystem::Uranus => [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
                SolarSystem::Neptune => [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
                _ => return Err("Invalid body".into()),
            };
            let adot: [f64;6] = match body {
                SolarSystem::Mercury => [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
                SolarSystem::Venus => [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
                SolarSystem::EMB => [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
                SolarSystem::Mars => [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
                SolarSystem::Jupiter => [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
                SolarSystem::Saturn => [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
                SolarSystem::Uranus => [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
                SolarSystem::Neptune => [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
                _ => return Err("Invalid body".into()),
            };
            let error_terms: Option<[f64;4]>  = match body {
                SolarSystem::Jupiter => Some([-0.00012452, 0.06064060, -0.35635438, 38.35125000]),
                SolarSystem::Saturn => Some([0.00025899, -0.13434469, 0.87320147, 38.35125000]),
                SolarSystem::Uranus => Some([0.00058331, -0.97731848, 0.17689245, 7.67025000]),
                SolarSystem::Neptune => Some([-0.00041348, 0.68346318, -0.10162547, 7.67025000]),
                _ => None,
            };
            (
                a[0] + jcen * adot[0], 
                a[1] + jcen * adot[1], 
                a[2] + jcen * adot[2], 
                a[3] + jcen * adot[3], 
                a[4] + jcen * adot[4], 
                a[5] + jcen * adot[5],
                error_terms
            )
        }
        else {
            return Err("Time out of range".into());
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
        Some([b,c,s,f]) => {
            l - wbar + b*jcen*jcen + c*(f*jcen).cos()*RAD2DEG + s*(f*jcen).sin()*RAD2DEG
        }
    };
    // Get m into range [-180, 180]
    m = m % 360.0;
    if m > 180.0 {
        m -= 360.0;
    }
    if m <= -180.0 {
        m += 360.0;
    }
    // Convert to radians
    let mrad = m / RAD2DEG;

    // Get the eccentric anomaly
    let mut enrad = mrad + eccen * mrad.sin();
    loop {
        let deltamrad = mrad - (enrad - eccen * enrad.sin());
        let deltaerad = deltamrad / (1.0 - eccen * enrad.cos());
        enrad += deltaerad;
        if (deltaerad/enrad).abs() < 1.0e-8 {
            break;
        }
    } 
    // Get heliocentric coordinates in orbital plane
    let xprime = a * (enrad.cos() - eccen);
    let yprime = a * (1.0 - eccen * eccen).sqrt() * enrad.sin();    
    let rprime = Vec3::new(xprime, yprime, 0.0);
    let recl = 
        Quat::from_axis_angle(&Vec3::z_axis(), Omega/RAD2DEG) * 
        Quat::from_axis_angle(&Vec3::x_axis(), incl/RAD2DEG) * 
        Quat::from_axis_angle(&Vec3::z_axis(), w/RAD2DEG) * rprime;

    // Rotate to the equatorial plane
    // Obliquity at J2000
    let obliquity = (23.439279 - 0.0130102*jcen - 5.086e-8*jcen*jcen + 5.565e-7*jcen.powi(3)
        + 1.6e-10*jcen.powi(4) + 1.21e-11*jcen.powi(5)) / RAD2DEG;
    
    Ok(Quat::from_axis_angle(&Vec3::x_axis(), obliquity) *  recl * crate::consts::AU)
    
}

#[cfg(test)]
mod test {
    use crate::jplephem;
    const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;
    use super::*;
    
    fn errors_precise(planet: &SolarSystem) -> (usize, usize, usize) {
        match planet {
            SolarSystem::Mercury => (15,1,1),
            SolarSystem::Venus => (20,1,4),
            SolarSystem::EMB => (20,1,4),
            SolarSystem::Mars => (40, 2, 25),
            SolarSystem::Jupiter => (400, 10, 1600),
            SolarSystem::Saturn => (600, 25, 1500),
            SolarSystem::Uranus => (50, 2, 1000),
            SolarSystem::Neptune => (10, 1, 200),
            _ => (0,0,0)
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


        let planets = [SolarSystem::Jupiter,
            SolarSystem::Mercury, SolarSystem::Venus, SolarSystem::EMB, 
            SolarSystem::Mars, SolarSystem::Jupiter, SolarSystem::Saturn, SolarSystem::Uranus, 
            SolarSystem::Neptune];

        for planet in planets {
            //let time = AstroTime::from_date(2000, 1, 1);
            let time = AstroTime::from_datetime(1994, 5, 20, 20, 0, 0.0);
            let p2 = jplephem::barycentric_pos(planet, &time).unwrap();
            let psun = jplephem::barycentric_pos(SolarSystem::Sun, &time).unwrap();
            let p1 = barycentric_pos(planet, &time).unwrap() - psun;
            let p3 = barycentric_pos2(planet,&time).unwrap() - psun;
            let lambda1 = f64::atan2(p1[1], p1[0]);
            let lambda2 = f64::atan2(p2[1], p2[0]);
            let phi1 = f64::asin(p1[2]/p1.norm());
            let phi2 = f64::asin(p2[2]/p2.norm());
            let lerr = (lambda1-lambda2).abs() * RAD2DEG * 3600.0;
            let perr = (phi1-phi2).abs() * RAD2DEG * 3600.0;
            let rerr = (p1.norm() - p2.norm()).abs() * 1.0e-6;
            let (lerr_approx, perr_approx, rerr_approx) = 
                errors_precise(&planet);
            println!("planet = {:?}", planet);
            println!("p1 = {}", p1.transpose());
            println!("p2 = {}", p2.transpose());
            println!("p3 = {}", p3.transpose());
            println!("l = {} {}", lerr, lerr_approx);
            println!("p = {} {}", perr, perr_approx);
            println!("r = {} {}", rerr, rerr_approx);

            assert!(lerr < lerr_approx as f64 * 3.0);
            assert!(perr < perr_approx as f64 * 3.0);
            assert!(rerr < rerr_approx as f64 * 3.0);
        }

    }


}