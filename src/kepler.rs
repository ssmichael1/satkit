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
///   angle in the orbital plane
///
/// * `Mean Anomaly` - Denoted M, this does not have a great geographical
///   representation, but is an angle that increases monotonically in time
///   between 0 and 2π over the course of a single orbit.
///
/// * `Eccentric Anomaly` - Denoted E, is the Periaps-C-B
///   angle in the orbital plane, wehre "C" is the center of the orbital
///   ellipse, and "B" is a point on the auxilliary circle (the circle
///   bounding the orbital ellipse) along a line from the satellite
///   and perpendicular to the semimajor axis.  The eccentric anomaly is
///   a useful prerequisite to compute the mean anomaly
///
pub enum Anomaly {
    Mean(f64),
    True(f64),
    Eccentric(f64),
}

// External library imports
use crate::mathtypes::*;

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
    ///   orbital plane
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

    /// Create a new Keplerian orbital element object with true anomaly
    ///
    /// # Arguments
    /// * `a` - Semi-major axis, meters
    /// * `eccen` - Eccentricity
    /// * `incl` - Inclination, radians
    /// * `raan` - Right Ascension of the Ascending Node, radians
    /// * `argp` - Argument of Perigee, radians
    /// * `nu` - True anomaly, radians
    pub fn with_true_anomaly(a: f64, eccen: f64, incl: f64, raan: f64, argp: f64, nu: f64) -> Self {
        Self::new(a, eccen, incl, raan, argp, Anomaly::True(nu))
    }

    /// Create a new Keplerian orbital element object with mean anomaly
    ///
    /// # Arguments
    /// * `a` - Semi-major axis, meters
    /// * `eccen` - Eccentricity
    /// * `incl` - Inclination, radians
    /// * `raan` - Right Ascension of the Ascending Node, radians
    /// * `argp` - Argument of Perigee, radians
    /// * `ma` - Mean anomaly, radians
    pub fn with_mean_anomaly(
        a: f64,
        eccen: f64,
        incl: f64,
        raan: f64,
        argp: f64,
        ma: f64,
    ) -> Self {
        Self::new(a, eccen, incl, raan, argp, Anomaly::Mean(ma))
    }

    /// Create a new Keplerian orbital element object with eccentric anomaly
    ///
    /// # Arguments
    /// * `a` - Semi-major axis, meters
    /// * `eccen` - Eccentricity
    /// * `incl` - Inclination, radians
    /// * `raan` - Right Ascension of the Ascending Node, radians
    /// * `argp` - Argument of Perigee, radians
    /// * `ea` - Eccentric anomaly, radians
    pub fn with_eccentric_anomaly(
        a: f64,
        eccen: f64,
        incl: f64,
        raan: f64,
        argp: f64,
        ea: f64,
    ) -> Self {
        Self::new(a, eccen, incl, raan, argp, Anomaly::Eccentric(ea))
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
    /// by given duration
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
    pub fn from_pv(r: Vector3, v: Vector3) -> Result<Self> {
        let h = r.cross(&v);
        let n = Vector3::z_axis().cross(&h);
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
    /// * `(Vector3, Vector3)` - Position and velocity vectors, meters and meters/second
    ///
    pub fn to_pv(&self) -> (Vector3, Vector3) {
        let p = self.a * self.eccen.mul_add(-self.eccen, 1.0);
        let r = p / self.eccen.mul_add(self.nu.cos(), 1.0);
        let r_pqw = Vector3::new(r * self.nu.cos(), r * self.nu.sin(), 0.0);
        let v_pqw = Vector3::new(-self.nu.sin(), self.eccen + self.nu.cos(), 0.0)
            * (crate::consts::MU_EARTH / p).sqrt();
        let q = Quaternion::from_axis_angle(&Vector3::z_axis(), self.raan)
            * Quaternion::from_axis_angle(&Vector3::x_axis(), self.incl)
            * Quaternion::from_axis_angle(&Vector3::z_axis(), self.w);
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
    fn test_circular_orbit() {
        use std::f64::consts::PI;
        let a = 7000.0e3; // 7000 km
        let k = Kepler::new(a, 0.0, 0.5, 1.0, 0.0, Anomaly::True(0.5));
        let (r, v) = k.to_pv();
        let k2 = Kepler::from_pv(r, v).unwrap();
        let (r2, v2) = k2.to_pv();
        assert!((r - r2).norm() < 1.0e-6);
        assert!((v - v2).norm() < 1.0e-6);

        // Verify period = 2π√(a³/μ)
        let period = 2.0 * PI * (a.powi(3) / crate::consts::MU_EARTH).sqrt();
        assert!((k.period() - period).abs() < 1.0e-6);
    }

    #[test]
    fn test_equatorial_orbit() {
        // Near-equatorial orbit (i=0 is singular for from_pv)
        let a = 8000.0e3;
        let k = Kepler::new(a, 0.1, 1.0e-6, 0.0, 0.5, Anomaly::True(1.0));
        let (r, v) = k.to_pv();
        // z-component should be near zero for equatorial orbit
        assert!(r[2].abs() / r.norm() < 1.0e-4);
        assert!(v[2].abs() / v.norm() < 1.0e-4);

        let k2 = Kepler::from_pv(r, v).unwrap();
        let (r2, v2) = k2.to_pv();
        assert!((r - r2).norm() / r.norm() < 1.0e-6);
        assert!((v - v2).norm() / v.norm() < 1.0e-6);
    }

    #[test]
    fn test_polar_orbit() {
        use std::f64::consts::FRAC_PI_2;
        let a = 7500.0e3;
        let k = Kepler::new(a, 0.05, FRAC_PI_2, 0.0, 0.3, Anomaly::True(0.8));
        let (r, v) = k.to_pv();
        let k2 = Kepler::from_pv(r, v).unwrap();
        let (r2, v2) = k2.to_pv();
        assert!((r - r2).norm() < 1.0e-3);
        assert!((v - v2).norm() < 1.0e-3);
        assert!((k2.incl - FRAC_PI_2).abs() < 1.0e-6);
    }

    #[test]
    fn test_propagate_period() {
        let k = Kepler::new(7000.0e3, 0.01, 0.5, 1.0, 0.3, Anomaly::True(0.5));
        let (r0, v0) = k.to_pv();
        let period = k.period();
        let dt = crate::Duration::from_seconds(period);
        let k2 = k.propagate(&dt);
        let (r1, v1) = k2.to_pv();
        assert!(
            (r0 - r1).norm() < 0.01,
            "Position after one period differs by {} m",
            (r0 - r1).norm()
        );
        assert!(
            (v0 - v1).norm() < 1.0e-5,
            "Velocity after one period differs by {} m/s",
            (v0 - v1).norm()
        );
    }

    #[test]
    fn test_anomaly_conversions() {
        use std::f64::consts::PI;
        for &e in &[0.0, 0.1, 0.5, 0.9] {
            // For a range of mean anomalies, verify M→E→ν→E→M roundtrip
            for i in 0..10 {
                let m_orig = (i as f64) * 2.0 * PI / 10.0;
                let ea = mean2eccentric(m_orig, e);
                let nu = eccentric2true(ea, e);

                // Reconstruct eccentric anomaly from true anomaly
                let ea2 = f64::atan2(
                    nu.sin() * e.mul_add(-e, 1.0).sqrt(),
                    e + nu.cos(),
                );
                // Reconstruct mean anomaly from eccentric anomaly
                let m_back = e.mul_add(-ea2.sin(), ea2);

                // Normalize both to [0, 2π) for comparison
                let m_orig_norm = m_orig.rem_euclid(2.0 * PI);
                let m_back_norm = m_back.rem_euclid(2.0 * PI);
                let diff = (m_orig_norm - m_back_norm).abs();
                let diff = diff.min((2.0 * PI - diff).abs());
                assert!(
                    diff < 1.0e-10,
                    "Anomaly roundtrip failed for e={}, M={}: diff={}",
                    e,
                    m_orig,
                    diff
                );
            }
        }
    }

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
        assert!((r * 1.0e-3 - Vector3::new(6525.368, 6861.532, 6449.119)).norm() < 1e-3);
        assert!((v * 1.0e-3 - Vector3::new(4.902279, 5.533140, -1.975710)).norm() < 1e-3);
    }

    #[test]
    fn test_frompv() {
        // Vallado example 2-5
        let r = Vector3::new(6524.834, 6862.875, 6448.296) * 1.0e3;
        let v = Vector3::new(4.901327, 5.533756, -1.976341) * 1.0e3;
        let k = Kepler::from_pv(r, v).unwrap();
        assert!((k.a - 36127343_f64).abs() < 1.0e3);
        assert!((k.eccen - 0.83285).abs() < 1e-3);
        assert!((k.incl - 87.87_f64.to_radians()).abs() < 1e-3);
        assert!((k.raan - 227.89_f64.to_radians()).abs() < 1e-3);
        assert!((k.w - 53.38_f64.to_radians()).abs() < 1e-3);
        assert!((k.nu - 92.335_f64.to_radians()).abs() < 1e-3);
    }
}
