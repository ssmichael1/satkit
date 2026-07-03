//! Keplerian orbital elements module
//!

use thiserror::Error;

/// Errors that can occur while constructing or converting [`Kepler`] elements.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned by [`Kepler::from_pv`] when the computed eccentricity is
    /// outside the valid range for an elliptical orbit.
    #[error("Eccentricity Out of Bounds {0}")]
    EccenOutOfBound(f64),

    /// Returned by [`Kepler::from_pv`] when the state has (near-)zero angular
    /// momentum (a rectilinear trajectory), for which the orbital plane — and
    /// therefore inclination and RAAN — are undefined.
    #[error("Degenerate state: angular momentum is zero (rectilinear trajectory)")]
    Degenerate,
}

/// Convenient type alias used throughout the `kepler` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Backwards-compatible alias for [`Error`].
#[deprecated(note = "use kepler::Error instead")]
pub type KeplerError = Error;

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
/// * `Eccentric Anomaly` - Denoted E, is the Periapsis-C-B
///   angle in the orbital plane, where "C" is the center of the orbital
///   ellipse, and "B" is a point on the auxiliary circle (the circle
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
#[derive(Debug, Clone, Copy)]
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
    // Newton's method converges quadratically for bound orbits (< ~10 steps
    // for e < 0.99). Cap the iteration count so a pathological eccentricity
    // (e >= 1, where the step can go non-finite) cannot spin forever.
    for _ in 0..30 {
        let de = eccen.mul_add(E.sin(), m - E) / eccen.mul_add(-E.cos(), 1.0);
        E += de;
        if de.abs() < 1.0e-12 {
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
    pub fn with_mean_anomaly(a: f64, eccen: f64, incl: f64, raan: f64, argp: f64, ma: f64) -> Self {
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
        use std::f64::consts::PI;
        let mu = crate::consts::MU_EARTH;
        let rmag = r.norm();

        let h = r.cross(&v);
        let hmag = h.norm();
        // Zero angular momentum ⇒ no orbital plane; inclination/RAAN undefined.
        if hmag < 1.0e-9 {
            return Err(Error::Degenerate);
        }
        let n = numeris::vector![0.0, 0.0, 1.0].cross(&h);
        let nmag = n.norm();
        let e = ((v.norm_squared() - mu / rmag) * r - r.dot(&v) * v) / mu;
        let eccen = e.norm();
        if eccen >= 1.0 {
            return Err(Error::EccenOutOfBound(eccen));
        }
        let xi = v.norm_squared() / 2.0 - mu / rmag;
        let a = -mu / (2.0 * xi);
        let incl = (h.z() / hmag).clamp(-1.0, 1.0).acos();

        // `acos` with the argument clamped to [-1, 1] to absorb rounding error.
        let safe_acos = |x: f64| x.clamp(-1.0, 1.0).acos();
        // Below these tolerances the eccentricity / node vectors are
        // numerically zero, and the usual acos expressions would divide by
        // zero and yield NaN. Fall back to the standard Vallado special cases.
        const TOL: f64 = 1.0e-11;
        let circular = eccen < TOL;
        let equatorial = nmag < TOL;

        let (raan, w, nu) = if circular && equatorial {
            // Circular equatorial: RAAN and argument of perigee undefined;
            // report the true longitude in `nu`.
            let mut lambda = safe_acos(r.x() / rmag);
            if r.y() < 0.0 {
                lambda = 2.0 * PI - lambda;
            }
            (0.0, 0.0, lambda)
        } else if circular {
            // Circular inclined: argument of perigee undefined; report the
            // argument of latitude in `nu`.
            let mut raan = safe_acos(n.x() / nmag);
            if n.y() < 0.0 {
                raan = 2.0 * PI - raan;
            }
            let mut u = safe_acos(n.dot(&r) / (nmag * rmag));
            if r.z() < 0.0 {
                u = 2.0 * PI - u;
            }
            (raan, 0.0, u)
        } else if equatorial {
            // Elliptical equatorial: RAAN undefined; report the true longitude
            // of periapsis in `w`.
            let mut w_true = safe_acos(e.x() / eccen);
            if e.y() < 0.0 {
                w_true = 2.0 * PI - w_true;
            }
            let mut nu = safe_acos(r.dot(&e) / (rmag * eccen));
            if r.dot(&v) < 0.0 {
                nu = 2.0 * PI - nu;
            }
            (0.0, w_true, nu)
        } else {
            let mut raan = safe_acos(n.x() / nmag);
            if n.y() < 0.0 {
                raan = 2.0 * PI - raan;
            }
            let mut w = safe_acos(n.dot(&e) / (nmag * eccen));
            if e.z() < 0.0 {
                w = 2.0 * PI - w;
            }
            let mut nu = safe_acos(r.dot(&e) / (rmag * eccen));
            if r.dot(&v) < 0.0 {
                nu = 2.0 * PI - nu;
            }
            (raan, w, nu)
        };

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
        let r_pqw = numeris::vector![r * self.nu.cos(), r * self.nu.sin(), 0.0];
        let v_pqw = numeris::vector![-self.nu.sin(), self.eccen + self.nu.cos(), 0.0]
            * (crate::consts::MU_EARTH / p).sqrt();
        let q =
            Quaternion::rotz(self.raan) * Quaternion::rotx(self.incl) * Quaternion::rotz(self.w);
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
    fn test_circular_equatorial_orbit() {
        // e = 0 and i = 0 exactly: both RAAN and argp are singular. Must not
        // produce NaN, and must round-trip through the true-longitude fallback.
        let a = 7200.0e3;
        let k = Kepler::new(a, 0.0, 0.0, 0.0, 0.0, Anomaly::True(0.7));
        let (r, v) = k.to_pv();
        let k2 = Kepler::from_pv(r, v).unwrap();
        assert!(k2.raan.is_finite() && k2.w.is_finite() && k2.nu.is_finite());
        let (r2, v2) = k2.to_pv();
        assert!((r - r2).norm() / r.norm() < 1.0e-9);
        assert!((v - v2).norm() / v.norm() < 1.0e-9);
    }

    #[test]
    fn test_elliptical_equatorial_orbit() {
        // e != 0 but i = 0 exactly: RAAN is singular. Must not produce NaN and
        // must round-trip through the longitude-of-periapsis fallback.
        let a = 9000.0e3;
        let k = Kepler::new(a, 0.1, 0.0, 0.0, 0.5, Anomaly::True(1.0));
        let (r, v) = k.to_pv();
        let k2 = Kepler::from_pv(r, v).unwrap();
        assert!(k2.raan.is_finite() && k2.w.is_finite() && k2.nu.is_finite());
        let (r2, v2) = k2.to_pv();
        assert!((r - r2).norm() / r.norm() < 1.0e-9);
        assert!((v - v2).norm() / v.norm() < 1.0e-9);
    }

    #[test]
    fn test_from_pv_rejects_rectilinear() {
        // Radial (parallel r and v) has zero angular momentum: no orbital plane.
        let r = numeris::vector![7000.0e3, 0.0, 0.0];
        let v = numeris::vector![1000.0, 0.0, 0.0];
        assert!(matches!(Kepler::from_pv(r, v), Err(Error::Degenerate)));
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
                let ea2 = f64::atan2(nu.sin() * e.mul_add(-e, 1.0).sqrt(), e + nu.cos());
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
        assert!((r * 1.0e-3 - numeris::vector![6525.368, 6861.532, 6449.119]).norm() < 1e-3);
        assert!((v * 1.0e-3 - numeris::vector![4.902279, 5.533140, -1.975710]).norm() < 1e-3);
    }

    #[test]
    fn test_frompv() {
        // Vallado example 2-5
        let r = numeris::vector![6524.834, 6862.875, 6448.296] * 1.0e3;
        let v = numeris::vector![4.901327, 5.533756, -1.976341] * 1.0e3;
        let k = Kepler::from_pv(r, v).unwrap();
        assert!((k.a - 36127343_f64).abs() < 1.0e3);
        assert!((k.eccen - 0.83285).abs() < 1e-3);
        assert!((k.incl - 87.87_f64.to_radians()).abs() < 1e-3);
        assert!((k.raan - 227.89_f64.to_radians()).abs() < 1e-3);
        assert!((k.w - 53.38_f64.to_radians()).abs() < 1e-3);
        assert!((k.nu - 92.335_f64.to_radians()).abs() < 1e-3);
    }
}
