//! Keplerian orbital elements module
//! 

// External library imports
use nalgebra::{Vector3, UnitQuaternion};
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
/// nu: True Anomaly, radians
#[derive(Debug, Clone)]
pub struct Kepler {
    pub a: f64,
    pub eccen: f64,
    pub incl: f64,
    pub raan: f64,
    pub w: f64,
    pub nu: f64, // True anomaly
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
    /// * `nu` - True Anomaly, radians
    /// 
    /// # Returns
    /// 
    /// * `Kepler` - A new Keplerian orbital element object
    pub fn new(a: f64, e: f64, i: f64, raan: f64, argp: f64, nu: f64) -> Kepler {
        Kepler {
            a: a,
            eccen: e,
            incl: i,
            raan: raan,
            w: argp,
            nu: nu,
        }
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
    pub fn from_pv(r: Vec3, v: Vec3) -> Kepler {
        let h = r.cross(&v);
        let n = Vec3::z_axis().cross(&h);
        let e = 
        ((v.norm_squared() - 
            crate::consts::MU_EARTH / r.norm()) * r - r.dot(&v) * v) / crate::consts::MU_EARTH;
        let eccen = e.norm();
        let xi = v.norm().powi(2) / 2.0 - crate::consts::MU_EARTH / r.norm();
        let a = -crate::consts::MU_EARTH / (2.0 * xi);
        let incl = (h.z/h.norm()).acos();
        let mut raan = (n.x/n.norm()).acos();
        if n.y < 0.0 {
            raan = 2.0 * std::f64::consts::PI - raan;
        }
        let mut w = (n.dot(&e)/n.norm()/e.norm()).acos();
        if e.z < 0.0 {
            w = 2.0 * std::f64::consts::PI - w;
        }
        let mut nu =  (r.dot(&e)/r.norm()/e.norm()).acos();
        if r.dot(&v) < 0.0 {
            nu = 2.0 * std::f64::consts::PI - nu;
        }  
        Kepler::new(a, eccen, incl, raan, w, nu)
    }

    /// Convert Keplerian orbital elements to Cartesian coordinates
    /// 
    /// # Returns
    /// 
    /// * `(Vec3, Vec3)` - Position and velocity vectors, meters and meters/second
    /// 
    pub fn to_pv(&self) -> (Vec3, Vec3) {
        let p = self.a * (1.0 - self.eccen.powi(2));
        let r = p / (1.0 + self.eccen * self.nu.cos());
        let r_pqw = Vec3::new(r * self.nu.cos(), r * self.nu.sin(), 0.0);
        let v_pqw = Vec3::new(
            -self.nu.sin(),
            self.eccen + self.nu.cos(),
             0.0) * (crate::consts::MU_EARTH/p).sqrt();
        let q = Quat::from_axis_angle(&Vec3::z_axis(), self.raan) * 
                Quat::from_axis_angle(&Vec3::x_axis(), self.incl) *
                Quat::from_axis_angle(&Vec3::z_axis(), self.w);
        (q * r_pqw, q * v_pqw)
    }
}

impl std::fmt::Display for Kepler {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, 
            "Keplerian Elements:\n  a = {:.0} m\n  e = {:.3}\n  i = {:.3} rad\n",
            self.a, self.eccen, self.incl)?;
        write!(f, "  Ω = {:.3} rad\n  ω = {:.3} rad\n  ν = {:.3} rad\n", self.raan, self.w, self.nu)
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

        let a = p / (1.0 - eccen.powi(2));

        let k = Kepler::new(a, eccen, incl, raan, w, nu);
        let (r, v) = k.to_pv();
        // Note: values below are not incorrect in the book, but are 
        // corrected in the online errata
        // See: https://celestrak.org/software/vallado/ErrataVer4.pdf
        assert!((r*1.0e-3 - Vec3::new(6525.368, 6861.532, 6449.119)).norm() < 1e-3);
        assert!((v*1.0e-3 - Vec3::new(4.902279, 5.533140, -1.975710)).norm() < 1e-3);
    }

    #[test]
    fn test_frompv() {
        // Vallado example 2-5
        let r = Vec3::new(6524.834, 6862.875, 6448.296)*1.0e3;
        let v = Vec3::new(4.901327, 5.533756, -1.976341)*1.0e3;
        let k = Kepler::from_pv(r, v);
        assert!((k.a - 36127343_f64).abs() < 1.0e3);
        assert!((k.eccen - 0.83285).abs() < 1e-3);
        assert!((k.incl - 87.87_f64.to_radians()).abs() < 1e-3);
        assert!((k.raan - 227.89_f64.to_radians()).abs() < 1e-3);
        assert!((k.w - 53.38_f64.to_radians()).abs() < 1e-3);
        assert!((k.nu - 92.335_f64.to_radians()).abs() < 1e-3);
    }
}
  