//! Keplerian orbital elements module
//!

use thiserror::Error;

/// Errors that can occur while constructing or converting [`Kepler`] elements.
#[derive(Debug, Error)]
#[non_exhaustive]
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

    /// Returned by [`Kepler::try_new`] and [`Kepler::validate`] for an
    /// element outside its domain: a non-finite value, `a <= 0`, `eccen`
    /// outside `[0, 1)`, `incl` outside `[0, π]`, or `mu <= 0`.
    #[error("invalid Keplerian element {name} = {value}: {reason}")]
    #[non_exhaustive]
    InvalidElement {
        name: &'static str,
        value: f64,
        reason: &'static str,
    },
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anomaly {
    Mean(f64),
    True(f64),
    Eccentric(f64),
}

impl Anomaly {
    /// The angle carried by the variant, radians.
    pub const fn value(self) -> f64 {
        match self {
            Self::Mean(v) | Self::True(v) | Self::Eccentric(v) => v,
        }
    }
}

// External library imports
use crate::mathtypes::*;

/// Keplerian Orbital Elements
///
/// The 6 Keplerian orbital elements, plus the gravitational parameter of the
/// central body they refer to:
///
/// * `a`: semi-major axis, meters
/// * `eccen`: eccentricity, `0 <= eccen < 1`
/// * `incl`: inclination, radians, `0 <= incl <= π`
/// * `raan`: right ascension of the ascending node, radians
/// * `argp`: argument of periapsis, radians
/// * `nu`: true anomaly, radians
/// * `mu`: gravitational parameter of the central body, m³/s²
///   ([`MU_EARTH`](crate::consts::MU_EARTH) unless set with [`Kepler::with_mu`])
///
/// The fields are public and may be assigned directly; nothing is validated
/// on assignment. [`Kepler::try_new`] and [`Kepler::validate`] are the
/// checked paths.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Kepler {
    pub a: f64,
    pub eccen: f64,
    pub incl: f64,
    pub raan: f64,
    pub argp: f64,
    /// True anomaly
    pub nu: f64,
    /// Gravitational parameter, m³/s². Serialized element sets that predate
    /// this field deserialize with Earth's value.
    #[serde(default = "default_mu")]
    pub mu: f64,
}

const fn default_mu() -> f64 {
    crate::consts::MU_EARTH
}

/// `Ok(())` when `value` is finite and `ok` holds; the error names the element.
fn check(name: &'static str, value: f64, ok: bool, reason: &'static str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::InvalidElement {
            name,
            value,
            reason: "must be finite",
        });
    }
    if !ok {
        return Err(Error::InvalidElement {
            name,
            value,
            reason,
        });
    }
    Ok(())
}

// Convert mean to eccentric anomaly
// iterative solution required
fn mean2eccentric(m: f64, eccen: f64) -> f64 {
    use std::f64::consts::TAU;
    // Range-reduce the mean anomaly to [0, 2π). Kepler's equation shifts by
    // 2πk in E and M together, so the solution for the reduced M is shifted
    // back at the end. Without this, an unwrapped M (e.g. after propagating
    // multiple revolutions) puts the naive initial guess in a near-flat region
    // of the equation at high eccentricity, where Newton's trajectory turns
    // chaotic and can exhaust the iteration cap with a wildly wrong root.
    let k = (m / TAU).floor();
    let mr = m - k * TAU;

    // Danby (1987) initial guess: E₀ = M + 0.85·e·sign(sin M). Together with
    // the range reduction this keeps plain Newton convergent in < ~10
    // iterations for all eccentricities below 1, including e > 0.9.
    #[allow(non_snake_case)]
    let mut E = mr + 0.85 * eccen * if mr.sin() >= 0.0 { 1.0 } else { -1.0 };

    // Cap the iteration count so a pathological eccentricity (e >= 1, where
    // the step can go non-finite) cannot spin forever.
    for _ in 0..50 {
        let de = eccen.mul_add(E.sin(), mr - E) / eccen.mul_add(-E.cos(), 1.0);
        E += de;
        if de.abs() < 1.0e-13 {
            break;
        }
    }
    E + k * TAU
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
    /// * `Kepler` - A new Keplerian orbital element object, with Earth's
    ///   gravitational parameter (see [`Kepler::with_mu`])
    ///
    /// Nothing is validated: `eccen >= 1`, `a <= 0` or non-finite inputs
    /// produce meaningless (NaN) anomaly conversions rather than an error.
    /// Use [`Kepler::try_new`] for a checked constructor.
    pub fn new(a: f64, eccen: f64, i: f64, raan: f64, argp: f64, an: Anomaly) -> Self {
        Self {
            a,
            eccen,
            incl: i,
            raan,
            argp,
            nu: to_trueanomaly(an, eccen),
            mu: default_mu(),
        }
    }

    /// Checked constructor: [`Kepler::new`] after validating every input.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidElement`] when any input is non-finite, `a <= 0`,
    /// `eccen` is outside `[0, 1)`, or `i` is outside `[0, π]`. The bounds
    /// are strict: `eccen = 1` and `i = π + 1e-16` are rejected.
    pub fn try_new(a: f64, eccen: f64, i: f64, raan: f64, argp: f64, an: Anomaly) -> Result<Self> {
        check("raan", raan, true, "")?;
        check("argp", argp, true, "")?;
        check("anomaly", an.value(), true, "")?;
        let k = Self::new(a, eccen, i, raan, argp, an);
        k.validate()?;
        Ok(k)
    }

    /// Check that the stored elements describe a closed orbit: every field
    /// finite, `a > 0`, `0 <= eccen < 1`, `0 <= incl <= π`, `mu > 0`.
    ///
    /// This is the check [`Kepler::try_new`] applies; call it after
    /// assigning fields directly.
    pub fn validate(&self) -> Result<()> {
        check(
            "a",
            self.a,
            self.a > 0.0,
            "semi-major axis must be positive",
        )?;
        check(
            "eccen",
            self.eccen,
            (0.0..1.0).contains(&self.eccen),
            "eccentricity must be in [0, 1) (closed orbits only)",
        )?;
        check(
            "incl",
            self.incl,
            (0.0..=std::f64::consts::PI).contains(&self.incl),
            "inclination must be in [0, π] radians",
        )?;
        check("raan", self.raan, true, "")?;
        check("argp", self.argp, true, "")?;
        check("nu", self.nu, true, "")?;
        check(
            "mu",
            self.mu,
            self.mu > 0.0,
            "gravitational parameter must be positive",
        )?;
        Ok(())
    }

    /// The same elements referred to a central body with gravitational
    /// parameter `mu` (m³/s²), e.g. [`MU_MOON`](crate::consts::MU_MOON).
    ///
    /// Only the dynamics change (mean motion, period, `propagate`, the
    /// element ↔ state conversions); the six geometric elements are kept
    /// as they are. Not validated; see [`Kepler::validate`].
    pub const fn with_mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
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
        Self { nu, ..*self }
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
        (self.mu / self.a.powi(3)).sqrt()
    }

    /// Return the period of the satellite in seconds
    ///
    /// # Returns
    ///
    /// * `f64` - Period, seconds
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.mean_motion()
    }

    /// Radius of periapsis `a (1 - e)`, meters
    pub fn periapsis(&self) -> f64 {
        self.a * (1.0 - self.eccen)
    }

    /// Radius of apoapsis `a (1 + e)`, meters
    pub fn apoapsis(&self) -> f64 {
        self.a * (1.0 + self.eccen)
    }

    /// Specific orbital energy `-μ / 2a`, J/kg (m²/s²)
    pub fn specific_energy(&self) -> f64 {
        -self.mu / (2.0 * self.a)
    }

    /// Magnitude of the specific angular momentum `√(μ p)`, m²/s
    pub fn angular_momentum(&self) -> f64 {
        (self.mu * self.semiparameter()).sqrt()
    }

    /// Flight-path angle `γ = atan2(e sin ν, 1 + e cos ν)`, radians: the
    /// angle of the velocity above the local horizontal, zero at periapsis
    /// and apoapsis, positive while climbing.
    pub fn flight_path_angle(&self) -> f64 {
        f64::atan2(
            self.eccen * self.nu.sin(),
            self.eccen.mul_add(self.nu.cos(), 1.0),
        )
    }

    /// Argument of latitude `u = ω + ν`, radians, reduced to `[0, 2π)`.
    /// Well defined for circular orbits, where ω and ν separately are not.
    pub fn argument_of_latitude(&self) -> f64 {
        (self.argp + self.nu).rem_euclid(std::f64::consts::TAU)
    }

    /// True longitude `λ = Ω + ω + ν`, radians, reduced to `[0, 2π)`.
    /// Well defined for circular equatorial orbits, where Ω, ω and ν
    /// separately are not.
    pub fn true_longitude(&self) -> f64 {
        (self.raan + self.argp + self.nu).rem_euclid(std::f64::consts::TAU)
    }

    /// Convert Cartesian coordinates to Keplerian orbital elements about
    /// the Earth ([`MU_EARTH`](crate::consts::MU_EARTH)); see
    /// [`Kepler::from_pv_with_mu`].
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
        Self::from_pv_with_mu(r, v, default_mu())
    }

    /// Convert Cartesian coordinates to Keplerian orbital elements about a
    /// central body with gravitational parameter `mu` (m³/s²).
    ///
    /// The returned elements carry `mu`, so their period, `propagate` and
    /// `to_pv` refer to the same body.
    ///
    /// # Errors
    ///
    /// [`Error::Degenerate`] for (near-)zero angular momentum,
    /// [`Error::EccenOutOfBound`] for an open (parabolic/hyperbolic) state.
    pub fn from_pv_with_mu(r: Vector3, v: Vector3, mu: f64) -> Result<Self> {
        use std::f64::consts::TAU;
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
        // atan2 form rather than acos(h_z / |h|): acos loses about half the
        // available precision for inclinations below ~1e-6 rad (and the
        // mirror case near π), where the argument sits at the edge of the
        // domain. `nmag` is |ẑ × h| = |h| sin i, so this is atan2(sin i, cos i).
        let incl = f64::atan2(nmag, h.z());

        // Every angle below is extracted with atan2(sin, cos) rather than the
        // textbook acos-plus-quadrant-test, which loses ~half the available
        // precision (≈1e-8 rad) whenever the angle is near 0 or π. The sine
        // terms come from triple products with the unit angular-momentum
        // vector: for any two vectors p, q in the orbital plane,
        // ĥ·(p × q) = |p||q| sin∠(p,q). Quadrant conventions are the same as
        // Vallado's Algorithm 9; results are reduced to [0, 2π).
        let hhat = h / hmag;
        let wrap = |x: f64| x.rem_euclid(TAU);
        // Below these tolerances the eccentricity / node vectors are
        // numerically zero and their directions are meaningless. Fall back
        // to the standard Vallado special cases. The node test uses
        // |n| / |h| = sin i, which is dimensionless; |n| itself is ~1e10 m²/s
        // for any bound Earth orbit, so an absolute tolerance never triggers.
        const TOL: f64 = 1.0e-11;
        let circular = eccen < TOL;
        let equatorial = nmag / hmag < TOL;

        let (raan, w, nu) = if circular && equatorial {
            // Circular equatorial: RAAN and argument of perigee undefined;
            // report the true longitude in `nu`.
            (0.0, 0.0, wrap(f64::atan2(r.y(), r.x())))
        } else if circular {
            // Circular inclined: argument of perigee undefined; report the
            // argument of latitude in `nu`.
            let raan = wrap(f64::atan2(n.y(), n.x()));
            let u = wrap(f64::atan2(hhat.dot(&n.cross(&r)), n.dot(&r)));
            (raan, 0.0, u)
        } else if equatorial {
            // Elliptical equatorial: RAAN undefined; report the true longitude
            // of periapsis in `w`.
            let w_true = wrap(f64::atan2(e.y(), e.x()));
            let nu = wrap(f64::atan2(hhat.dot(&e.cross(&r)), e.dot(&r)));
            (0.0, w_true, nu)
        } else {
            let raan = wrap(f64::atan2(n.y(), n.x()));
            let w = wrap(f64::atan2(hhat.dot(&n.cross(&e)), n.dot(&e)));
            let nu = wrap(f64::atan2(hhat.dot(&e.cross(&r)), e.dot(&r)));
            (raan, w, nu)
        };

        Ok(Self::new(a, eccen, incl, raan, w, Anomaly::True(nu)).with_mu(mu))
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
            * (self.mu / p).sqrt();
        let q =
            Quaternion::rotz(self.raan) * Quaternion::rotx(self.incl) * Quaternion::rotz(self.argp);
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
            "  Ω = {:.3} rad\n  ω = {:.3} rad\n  ν = {:.3} rad\n  μ = {:.6e} m³/s²\n",
            self.raan, self.argp, self.nu, self.mu
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
        assert!(k2.raan.is_finite() && k2.argp.is_finite() && k2.nu.is_finite());
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
        assert!(k2.raan.is_finite() && k2.argp.is_finite() && k2.nu.is_finite());
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
    fn test_mean2eccentric_nan_returns() {
        // A NaN mean anomaly must fall out of the capped Newton loop as NaN
        // rather than spinning forever (the Python setter used to hang here).
        assert!(mean2eccentric(f64::NAN, 0.5).is_nan());
        assert!(mean2eccentric(0.5, f64::NAN).is_nan());
        // e >= 1 is outside the domain of the elliptical solver; the result is
        // meaningless but the call must return.
        let _ = mean2eccentric(1.0, 1.0);
        let _ = mean2eccentric(1.0, 1.5);
        let _ = mean2eccentric(f64::INFINITY, 0.1);
    }

    #[test]
    fn test_mean_anomaly_setter_roundtrip_high_eccen() {
        use std::f64::consts::TAU;
        // Set M via the public constructor path, read it back.
        for &e in &[0.0, 0.1, 0.5, 0.9, 0.99, 0.999] {
            for i in 0..64 {
                let m = (i as f64) * TAU / 64.0;
                let k = Kepler::with_mean_anomaly(7000.0e3, e, 0.5, 1.0, 0.3, m);
                let mut dm = (k.mean_anomaly() - m).rem_euclid(TAU);
                if dm > TAU / 2.0 {
                    dm -= TAU;
                }
                assert!(
                    dm.abs() < 1.0e-12,
                    "M round-trip failed for e={e}, M={m}: dM={dm:e}"
                );
            }
        }
    }

    /// Round-trip r,v → elements → r,v over a grid of eccentricities up to
    /// 0.999 and inclinations down to 1e-9 rad (and the retrograde mirror
    /// near π), including the true-anomaly quadrants near periapsis and
    /// apoapsis where the acos branches are least conditioned.
    #[test]
    fn test_from_pv_roundtrip_grid() {
        use std::f64::consts::PI;
        let eccens = [0.0, 1.0e-6, 0.01, 0.3, 0.7, 0.9, 0.99, 0.999];
        let incls = [
            0.0,
            1.0e-9,
            1.0e-7,
            1.0e-5,
            1.0e-3,
            0.5,
            PI / 2.0,
            PI - 1.0e-3,
            PI - 1.0e-9,
        ];
        let nus = [
            0.0,
            1.0e-7,
            0.3,
            PI / 2.0,
            PI - 1.0e-6,
            PI,
            PI + 0.4,
            2.0 * PI - 1.0e-7,
        ];
        let a = 12_000.0e3;
        for &e in &eccens {
            for &i in &incls {
                for &nu in &nus {
                    let k = Kepler::new(a, e, i, 1.1, 0.7, Anomaly::True(nu));
                    let (r, v) = k.to_pv();
                    let k2 = Kepler::from_pv(r, v)
                        .unwrap_or_else(|err| panic!("from_pv failed e={e} i={i} nu={nu}: {err}"));
                    assert!(
                        k2.a.is_finite()
                            && k2.eccen.is_finite()
                            && k2.incl.is_finite()
                            && k2.raan.is_finite()
                            && k2.argp.is_finite()
                            && k2.nu.is_finite(),
                        "non-finite element e={e} i={i} nu={nu}: {k2:?}"
                    );
                    assert!(
                        (k2.a - a).abs() / a < 1.0e-9,
                        "a mismatch e={e} i={i} nu={nu}: {}",
                        (k2.a - a).abs() / a
                    );
                    assert!(
                        (k2.eccen - e).abs() < 1.0e-9,
                        "e mismatch e={e} i={i} nu={nu}: {}",
                        k2.eccen - e
                    );
                    assert!(
                        (k2.incl - i).abs() < 1.0e-9,
                        "i mismatch e={e} i={i} nu={nu}: {:e}",
                        k2.incl - i
                    );
                    let (r2, v2) = k2.to_pv();
                    let dr = (r - r2).norm() / r.norm();
                    let dv = (v - v2).norm() / v.norm();
                    assert!(
                        dr < 1.0e-6 && dv < 1.0e-6,
                        "round-trip e={e} i={i} nu={nu}: dr={dr:e} dv={dv:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_from_pv_tiny_inclination_precision() {
        // acos(h_z/|h|) would return exactly 0 (or ~1.5e-8) here; atan2 keeps
        // the inclination to full relative precision.
        for &i in &[1.0e-9, 1.0e-8, 1.0e-7] {
            let k = Kepler::new(7000.0e3, 0.2, i, 0.4, 1.2, Anomaly::True(2.0));
            let (r, v) = k.to_pv();
            let k2 = Kepler::from_pv(r, v).unwrap();
            assert!(
                (k2.incl - i).abs() / i < 1.0e-6,
                "i={i}: got {} (rel err {:e})",
                k2.incl,
                (k2.incl - i).abs() / i
            );
        }
    }

    #[test]
    fn test_try_new_rejects_out_of_domain_elements() {
        use std::f64::consts::PI;
        let ok = Kepler::try_new(7000.0e3, 0.1, 0.5, 1.0, 0.3, Anomaly::True(0.7));
        assert!(ok.is_ok());
        let bad = [
            (0.0, 0.1, 0.5, "a"),
            (-7000.0e3, 0.1, 0.5, "a"),
            (f64::NAN, 0.1, 0.5, "a"),
            (7000.0e3, 1.0, 0.5, "eccen"),
            (7000.0e3, -1.0e-3, 0.5, "eccen"),
            (7000.0e3, f64::INFINITY, 0.5, "eccen"),
            (7000.0e3, 0.1, -1.0e-9, "incl"),
            (7000.0e3, 0.1, PI + 1.0e-9, "incl"),
            (7000.0e3, 0.1, f64::NAN, "incl"),
        ];
        for (a, e, i, which) in bad {
            match Kepler::try_new(a, e, i, 1.0, 0.3, Anomaly::True(0.7)) {
                Err(Error::InvalidElement { name, .. }) => assert_eq!(name, which),
                other => panic!("expected InvalidElement({which}), got {other:?}"),
            }
        }
        // Non-finite angles, including the anomaly, are rejected too.
        assert!(Kepler::try_new(7000.0e3, 0.1, 0.5, f64::NAN, 0.3, Anomaly::True(0.7)).is_err());
        assert!(Kepler::try_new(7000.0e3, 0.1, 0.5, 1.0, 0.3, Anomaly::Mean(f64::NAN)).is_err());
        // Boundaries: e = 0 and i ∈ {0, π} are valid.
        assert!(Kepler::try_new(7000.0e3, 0.0, 0.0, 0.0, 0.0, Anomaly::True(0.0)).is_ok());
        assert!(Kepler::try_new(7000.0e3, 0.0, PI, 0.0, 0.0, Anomaly::True(0.0)).is_ok());
        // validate() catches a bad direct assignment, mu included.
        let mut k = ok.unwrap();
        k.mu = 0.0;
        assert!(matches!(
            k.validate(),
            Err(Error::InvalidElement { name: "mu", .. })
        ));
    }

    #[test]
    fn test_mu_changes_dynamics_not_geometry() {
        use crate::consts::{MU_EARTH, MU_MOON};
        let k_earth = Kepler::new(2000.0e3, 0.05, 1.0, 0.2, 0.3, Anomaly::True(0.4));
        assert_eq!(k_earth.mu, MU_EARTH);
        let k_moon = k_earth.with_mu(MU_MOON);
        // Same six elements …
        assert_eq!(k_moon.a, k_earth.a);
        assert_eq!(k_moon.nu, k_earth.nu);
        // … different period: T ∝ 1/√μ.
        let ratio = k_moon.period() / k_earth.period();
        assert!((ratio - (MU_EARTH / MU_MOON).sqrt()).abs() < 1.0e-12);
        // 2000 km about the Moon: ~ 2.1 h. Sanity check the magnitude.
        assert!(
            (k_moon.period() - 8022.0).abs() < 5.0,
            "{}",
            k_moon.period()
        );
        // The state round trip must use the same μ on both legs.
        let (r, v) = k_moon.to_pv();
        let back = Kepler::from_pv_with_mu(r, v, MU_MOON).unwrap();
        assert_eq!(back.mu, MU_MOON);
        assert!((back.a - k_moon.a).abs() / k_moon.a < 1.0e-9);
        assert!((back.eccen - k_moon.eccen).abs() < 1.0e-9);
        // Interpreting a lunar state with Earth's μ gives a different orbit.
        let wrong = Kepler::from_pv(r, v).unwrap();
        assert!((wrong.a - k_moon.a).abs() / k_moon.a > 0.1);
        // propagate keeps mu.
        assert_eq!(
            k_moon.propagate(&crate::Duration::from_seconds(10.0)).mu,
            MU_MOON
        );
    }

    #[test]
    fn test_derived_helpers() {
        use std::f64::consts::{PI, TAU};
        let a = 26_600.0e3;
        let e = 0.74;
        let k = Kepler::new(a, e, 1.1, 5.0, 4.0, Anomaly::True(0.0));
        assert!((k.periapsis() - a * (1.0 - e)).abs() < 1.0e-6);
        assert!((k.apoapsis() - a * (1.0 + e)).abs() < 1.0e-6);
        assert!((k.periapsis() + k.apoapsis() - 2.0 * a).abs() < 1.0e-6);
        // Vis-viva at periapsis agrees with the energy helper.
        let (r, v) = k.to_pv();
        let xi = v.norm_squared() / 2.0 - k.mu / r.norm();
        assert!((xi - k.specific_energy()).abs() / xi.abs() < 1.0e-12);
        // |r × v| agrees with the angular-momentum helper.
        assert!((r.cross(&v).norm() - k.angular_momentum()).abs() / k.angular_momentum() < 1.0e-12);
        // Flight-path angle: zero at periapsis and apoapsis, positive on the
        // outbound leg, and equal to atan(e sinν / (1 + e cosν)) elsewhere.
        assert_eq!(k.flight_path_angle(), 0.0);
        let k_apo = Kepler::new(a, e, 1.1, 5.0, 4.0, Anomaly::True(PI));
        assert!(k_apo.flight_path_angle().abs() < 1.0e-15);
        let k_out = Kepler::new(a, e, 1.1, 5.0, 4.0, Anomaly::True(1.0));
        assert!(k_out.flight_path_angle() > 0.0);
        let (r, v) = k_out.to_pv();
        let gamma = (r.dot(&v) / (r.norm() * v.norm())).asin();
        assert!((gamma - k_out.flight_path_angle()).abs() < 1.0e-12);
        // u and λ wrap into [0, 2π): ω + ν = 4 + 1 = 5; Ω + ω + ν = 10 → 10 - 2π.
        assert!((k_out.argument_of_latitude() - 5.0).abs() < 1.0e-12);
        assert!((k_out.true_longitude() - (10.0 - TAU)).abs() < 1.0e-12);
        let k_neg = Kepler::new(a, e, 1.1, -1.0, -1.0, Anomaly::True(-1.0));
        assert!((k_neg.argument_of_latitude() - (TAU - 2.0)).abs() < 1.0e-12);
        assert!((k_neg.true_longitude() - (TAU - 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn test_serde_roundtrip_and_missing_mu_defaults_to_earth() {
        let k = Kepler::new(7000.0e3, 0.1, 0.5, 1.0, 0.3, Anomaly::True(0.7))
            .with_mu(crate::consts::MU_MOON);
        let json = serde_json::to_string(&k).unwrap();
        let back: Kepler = serde_json::from_str(&json).unwrap();
        assert_eq!(back, k);
        // An element set serialized before `mu` existed still loads.
        let legacy = r#"{"a":7000000.0,"eccen":0.1,"incl":0.5,"raan":1.0,"argp":0.3,"nu":0.7}"#;
        let old: Kepler = serde_json::from_str(legacy).unwrap();
        assert_eq!(old.mu, crate::consts::MU_EARTH);
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
        assert!((k.argp - 53.38_f64.to_radians()).abs() < 1e-3);
        assert!((k.nu - 92.335_f64.to_radians()).abs() < 1e-3);
    }
}
