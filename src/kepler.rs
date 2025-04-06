//! Keplerian orbital elements module
//!

use anyhow::Result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KeplerError {
    #[error("Eccentricity Out of Bounds {0}")]
    EccenOutOfBound(f64),
}

impl<T> From<KeplerError> for Result<T> {
    fn from(e: KeplerError) -> Self {
        Err(e.into())
    }
}

/// Keplerian element can be defined by multiple
/// types of "anomalies", which describe the position
/// of the satellite orbiting the central body within the orbital plane
///
/// These are:
///
/// * `True Anomaly` - Denoted ν, is the Periapsis-Earth-Satellite
///    angle in the orbital plane
///
/// * `Mean Anomaly` - Denoted M, this does not have a great geographical
///    representation, but is an angle that increases monotonically in time
///    between 0 and 2π over the course of a single orbit.
///
/// * `Eccentric Anomaly` - Denoted E, is the Periaps-C-B
///    angle in the orbital plane, wehre "C" is the center of the orbital
///    ellipse, and "B" is a point on the auxilliary circle (the circle
///    bounding the orbital ellipse) along a line from the satellite
///    and perpendicular to the semimajor axis.  The eccentric anomaly is
///    a useful prerequisite to compute the mean anomaly
///
pub enum Anomaly {
    Mean(f64),
    True(f64),
    Eccentric(f64),
}

// External library imports
use nalgebra::{UnitQuaternion, Vector3};
type Vec3 = Vector3<f64>;
type Quat = UnitQuaternion<f64>;

/// Keplerian Orbital Elements
///
/// The 6 Keplerian orbital elements are:
/// a: semi-major axis, meters
/// eccen: Eccentricity
/// incl: Inclination, radians
/// RAAN: Right Ascension of the Ascending Node, radians
/// w: Argument of Perigee, radians
/// an: Anomaly of given type, radians
#[derive(Debug, Clone)]
pub struct Kepler {
    pub a: f64,
    pub eccen: f64,
    pub incl: f64,
    pub raan: f64,
    pub w: f64,
    pub nu: f64, // True anomaly
}

// Convert mean to eccentric anomaly
// iterative solution required
fn mean2eccentric(m: f64, eccen: f64) -> f64 {
    use std::f64::consts::PI;
    #[allow(non_snake_case)]
    let mut E = match (m > PI) || ((m < 0.0) && (m > -PI)) {
        true => m - eccen,
        false => m + eccen,
    };
    loop {
        let de = eccen.mul_add(E.sin(), m - E) / eccen.mul_add(-E.cos(), 1.0);
        E += de;
        if de.abs() < 1.0e-6 {
            break;
        }
    }
    E
}

fn eccentric2true(ea: f64, eccen: f64) -> f64 {
    f64::atan2(
        ea.sin() * eccen.mul_add(-eccen, 1.0).sqrt(),
        ea.cos() - eccen,
    )
}

fn mean2true(ma: f64, eccen: f64) -> f64 {
    eccentric2true(mean2eccentric(ma, eccen), eccen)
}

fn to_trueanomaly(an: Anomaly, eccen: f64) -> f64 {
    match an {
        Anomaly::True(v) => v,
        Anomaly::Mean(ma) => mean2true(ma, eccen),
        Anomaly::Eccentric(ea) => eccentric2true(ea, eccen),
    }
}

impl Kepler {
    /// Create a new Keplerian orbital element object
    ///
    /// # Arguments
    ///
    /// * `a` - Semi-major axis, meters
    /// * `e` - Eccentricity
    /// * `i` - Inclination, radians
    /// * `raan` - Right Ascension of the Ascending Node, radians
    /// * `argp` - Argument of Perigee, radians
    /// * `anomaly` - Anomaly type representing location of satellite along the
    ///               orbital plane
    ///
    /// # Returns
    ///
    /// * `Kepler` - A new Keplerian orbital element object
    pub fn new(a: f64, eccen: f64, i: f64, raan: f64, argp: f64, an: Anomaly) -> Self {
        Self {
            a,
            eccen,
            incl: i,
            raan,
            w: argp,
            nu: to_trueanomaly(an, eccen),
        }
    }

    /// Return the semiparameter of the satellite orbit
    ///
    /// The semiparameter is also known as the semi-latus rectum
    /// # Returns
    ///
    /// * `f64` - Semiparameter, meters
    pub fn semiparameter(&self) -> f64 {
        self.a * self.eccen.mul_add(-self.eccen, 1.0)
    }

    /// Propagate the orbit forward (or backward) in time
    /// by givend duration
    ///
    /// # Arguments
    ///
    /// * `dt` - `satkit.Duration` object representing the time to propagate
    ///
    /// # Returns
    ///
    /// * `Kepler` - A new Keplerian orbital element object
    pub fn propagate(&self, dt: &crate::Duration) -> Self {
        let n = self.mean_motion();
        let ma = n.mul_add(dt.as_seconds(), self.mean_anomaly());
        let nu = mean2true(ma, self.eccen);
        Self {
            a: self.a,
            eccen: self.eccen,
            incl: self.incl,
            raan: self.raan,
            w: self.w,
            nu,
        }
    }

    /// Return the eccentric anomaly of the satellite in radians
    pub fn eccentric_anomaly(&self) -> f64 {
        f64::atan2(
            self.nu.sin() * self.eccen.mul_add(-self.eccen, 1.0).sqrt(),
            self.eccen + self.nu.cos(),
        )
    }

    /// Return the mean anomaly of the satellite in radians
    pub fn mean_anomaly(&self) -> f64 {
        let ea = self.eccentric_anomaly();
        self.eccen.mul_add(-ea.sin(), ea)
    }

    /// Return the true anomaly of the satellite in radians
    pub const fn true_anomaly(&self) -> f64 {
        self.nu
    }

    /// Return the mean motion of the satellite in radians/second
    ///
    /// # Returns
    ///
    /// * `f64` - Mean motion, radians/second
    pub fn mean_motion(&self) -> f64 {
        (crate::consts::MU_EARTH / self.a.powi(3)).sqrt()
    }

    /// Return the period of the satellite in seconds
    ///
    /// # Returns
    ///
    /// * `f64` - Period, seconds
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.mean_motion()
    }

    /// Convert Cartesian coordinates to Keplerian orbital elements
    ///
    /// # Arguments
    ///
    /// * `r` - Position vector, meters
    /// * `v` - Velocity vector, meters/second
    ///
    /// # Returns
    ///
    /// * `Kepler` - A new Keplerian orbital element object
    ///
    pub fn from_pv(r: Vec3, v: Vec3) -> Result<Self> {
        let h = r.cross(&v);
        let n = Vec3::z_axis().cross(&h);
        let e = ((v.norm_squared() - crate::consts::MU_EARTH / r.norm()) * r - r.dot(&v) * v)
            / crate::consts::MU_EARTH;
        let eccen = e.norm();
        if eccen >= 1.0 {
            return KeplerError::EccenOutOfBound(eccen).into();
        }
        let xi = v.norm().powi(2) / 2.0 - crate::consts::MU_EARTH / r.norm();
        let a = -crate::consts::MU_EARTH / (2.0 * xi);
        let incl = (h.z / h.norm()).acos();
        let mut raan = (n.x / n.norm()).acos();
        if n.y < 0.0 {
            raan = 2.0f64.mul_add(std::f64::consts::PI, -raan);
        }
        let mut w = (n.dot(&e) / n.norm() / e.norm()).acos();
        if e.z < 0.0 {
            w = 2.0f64.mul_add(std::f64::consts::PI, -w);
        }
        let mut nu = (r.dot(&e) / r.norm() / e.norm()).acos();
        if r.dot(&v) < 0.0 {
            nu = 2.0f64.mul_add(std::f64::consts::PI, -nu);
        }
        Ok(Self::new(a, eccen, incl, raan, w, Anomaly::True(nu)))
    }

    /// Convert Keplerian orbital elements to Cartesian coordinates
    ///
    /// # Returns
    ///
    /// * `(Vec3, Vec3)` - Position and velocity vectors, meters and meters/second
    ///
    pub fn to_pv(&self) -> (Vec3, Vec3) {
        let p = self.a * self.eccen.mul_add(-self.eccen, 1.0);
        let r = p / self.eccen.mul_add(self.nu.cos(), 1.0);
        let r_pqw = Vec3::new(r * self.nu.cos(), r * self.nu.sin(), 0.0);
        let v_pqw = Vec3::new(-self.nu.sin(), self.eccen + self.nu.cos(), 0.0)
            * (crate::consts::MU_EARTH / p).sqrt();
        let q = Quat::from_axis_angle(&Vec3::z_axis(), self.raan)
            * Quat::from_axis_angle(&Vec3::x_axis(), self.incl)
            * Quat::from_axis_angle(&Vec3::z_axis(), self.w);
        (q * r_pqw, q * v_pqw)
    }
}

impl std::fmt::Display for Kepler {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Keplerian Elements:\n  a = {:.0} m\n  e = {:.3}\n  i = {:.3} rad\n",
            self.a, self.eccen, self.incl
        )?;
        write!(
            f,
            "  Ω = {:.3} rad\n  ω = {:.3} rad\n  ν = {:.3} rad\n",
            self.raan, self.w, self.nu
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topv() {
        // Example 2-6 from Vallado
        let p = 11067790.0;
        let eccen = 0.83285_f64;
        let incl = 87.87_f64.to_radians();
        let raan = 227.89_f64.to_radians();
        let w = 53.38_f64.to_radians();
        let nu = 92.335_f64.to_radians();

        let a = p / eccen.mul_add(-eccen, 1.0);

        let k = Kepler::new(a, eccen, incl, raan, w, Anomaly::True(nu));
        let (r, v) = k.to_pv();
        // Note: values below are not incorrect in the book, but are
        // corrected in the online errata
        // See: https://celestrak.org/software/vallado/ErrataVer4.pdf
        assert!((r * 1.0e-3 - Vec3::new(6525.368, 6861.532, 6449.119)).norm() < 1e-3);
        assert!((v * 1.0e-3 - Vec3::new(4.902279, 5.533140, -1.975710)).norm() < 1e-3);
    }

    #[test]
    fn test_frompv() {
        // Vallado example 2-5
        let r = Vec3::new(6524.834, 6862.875, 6448.296) * 1.0e3;
        let v = Vec3::new(4.901327, 5.533756, -1.976341) * 1.0e3;
        let k = Kepler::from_pv(r, v).unwrap();
        assert!((k.a - 36127343_f64).abs() < 1.0e3);
        assert!((k.eccen - 0.83285).abs() < 1e-3);
        assert!((k.incl - 87.87_f64.to_radians()).abs() < 1e-3);
        assert!((k.raan - 227.89_f64.to_radians()).abs() < 1e-3);
        assert!((k.w - 53.38_f64.to_radians()).abs() < 1e-3);
        assert!((k.nu - 92.335_f64.to_radians()).abs() < 1e-3);
    }
}
