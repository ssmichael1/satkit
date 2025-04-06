use std::f64::consts::PI;

use crate::consts::WGS84_A;
use crate::consts::WGS84_F;

use nalgebra as na;

use crate::types::Quaternion as Quat;
use crate::types::Vec3;

use anyhow::Result;

///
/// Representation of a coordinate in the
/// International Terrestrial Reference Frame (ITRF)
///
/// This coordinate object can be created from and also
/// output to Geodetic coordinates (latitude, longitude,
/// height above ellipsoid).
///
/// Functions are also available to provide rotation
/// quaternions to the East-North-Up frame
/// and North-East-Down frame at this coordinate
///
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug)]
pub struct ITRFCoord {
    pub itrf: Vec3,
}

impl std::fmt::Display for ITRFCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (lat, lon, hae) = self.to_geodetic_deg();
        write!(
            f,
            "ITRFCoord(lat: {:8.4} deg, lon: {:8.4} deg, altitude: {:5.2} km)",
            lat,
            lon,
            hae / 1.0e3
        )
    }
}

impl std::ops::Add<Vec3> for ITRFCoord {
    type Output = Self;
    fn add(self, other: Vec3) -> Self::Output {
        Self {
            itrf: self.itrf + other,
        }
    }
}

impl std::ops::Add<Vec3> for &ITRFCoord {
    type Output = ITRFCoord;
    fn add(self, other: Vec3) -> Self::Output {
        ITRFCoord {
            itrf: self.itrf + other,
        }
    }
}

impl std::ops::Add<&Vec3> for ITRFCoord {
    type Output = Self;
    fn add(self, other: &Vec3) -> Self::Output {
        Self {
            itrf: self.itrf + other,
        }
    }
}

impl std::ops::Add<&Vec3> for &ITRFCoord {
    type Output = ITRFCoord;
    fn add(self, other: &Vec3) -> Self::Output {
        ITRFCoord {
            itrf: self.itrf + other,
        }
    }
}

impl std::ops::Sub<Vec3> for ITRFCoord {
    type Output = Self;
    fn sub(self, other: Vec3) -> Self::Output {
        Self {
            itrf: self.itrf - other,
        }
    }
}

impl std::ops::Sub<Self> for ITRFCoord {
    type Output = Vec3;
    fn sub(self, other: Self) -> Vec3 {
        self.itrf - other.itrf
    }
}

impl std::ops::Sub<ITRFCoord> for &ITRFCoord {
    type Output = Vec3;
    fn sub(self, other: ITRFCoord) -> Vec3 {
        self.itrf - other.itrf
    }
}

impl std::ops::Sub<&ITRFCoord> for &ITRFCoord {
    type Output = Vec3;
    fn sub(self, other: &ITRFCoord) -> Vec3 {
        self.itrf - other.itrf
    }
}

impl std::ops::Sub<&Self> for ITRFCoord {
    type Output = Vec3;
    fn sub(self, other: &Self) -> Vec3 {
        self.itrf - other.itrf
    }
}

impl std::convert::From<[f64; 3]> for ITRFCoord {
    fn from(v: [f64; 3]) -> Self {
        Self {
            itrf: Vec3::from(v),
        }
    }
}

impl std::convert::From<&[f64]> for ITRFCoord {
    fn from(v: &[f64]) -> Self {
        assert!(v.len() == 3);
        Self {
            itrf: Vec3::from_row_slice(v),
        }
    }
}

impl std::convert::From<Vec3> for ITRFCoord {
    fn from(v: Vec3) -> Self {
        Self { itrf: v }
    }
}

impl std::convert::From<ITRFCoord> for Vec3 {
    fn from(itrf: ITRFCoord) -> Self {
        itrf.itrf
    }
}

impl ITRFCoord {
    /// Returns an ITRF Coordinate given the geodetic inputs
    ///   with degree units for latitude & longitude
    ///
    /// # Arguments:
    ///
    /// * `lat` - Geodetic latitude in degrees
    /// * `lon` - Geodetic longitude in degrees
    /// * `hae` - Height above ellipsoid, in meters
    ///
    /// # Examples:
    /// ```
    /// // Create coord for ~ Boston, MA
    /// use satkit::itrfcoord::ITRFCoord;
    /// let itrf = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 150.0);
    /// ```
    ///
    pub fn from_geodetic_deg(lat: f64, lon: f64, hae: f64) -> Self {
        Self::from_geodetic_rad(lat.to_radians(), lon.to_radians(), hae)
    }

    ///
    /// Returns an ITRF Coordinate given Cartesian ITRF coordinates
    ///
    /// # Arguments:
    ///
    /// * `v` - `nalgebra::Vector3<f64>` representing ITRF coordinates in meters
    ///
    /// # Examples:
    ///
    /// ```
    /// // Create coord for ~ Boston, MA
    /// use satkit::itrfcoord::ITRFCoord;
    /// use nalgebra as na;
    /// let itrf = ITRFCoord::from_vector(&na::Vector3::new(1522386.15660978, -4459627.78585002,  4284030.6890791));
    /// ```
    ///
    ///
    pub const fn from_vector(v: &na::Vector3<f64>) -> Self {
        Self { itrf: *v }
    }

    /// Returns an ITRF Coordinate given Cartesian ITRF coordinates represented as a slice
    ///
    /// # Arguments:
    ///
    /// * `v` - Slice representing ITRF coordinates in meters
    ///
    /// # Examples:
    ///
    /// ```
    /// // Create coord for ~ Boston, MA
    /// use satkit::itrfcoord::ITRFCoord;
    /// let itrf = ITRFCoord::from_slice(&[1522386.15660978, -4459627.78585002,  4284030.6890791]);
    /// ```
    ///
    pub fn from_slice(v: &[f64]) -> Result<Self> {
        if v.len() != 3 {
            anyhow::bail!("Input slice must have 3 elements");
        }
        Ok(Self {
            itrf: Vec3::from_row_slice(v),
        })
    }

    /// Returns an ITRF Coordinate given the geodetic inputs
    ///   with radian units for latitude & longitude
    ///
    /// # Arguments:
    ///
    /// * `lat` - Geodetic latitude in radians
    /// * `lon` - Geodetic longitude in radians
    /// * `hae` - Height above ellipsoid, in meters
    ///
    /// # Examples:
    /// ```
    /// // Create coord for ~ Boston, MA
    /// use satkit::itrfcoord::ITRFCoord;
    /// use std::f64::consts::PI;
    /// const DEG2RAD: f64 = PI / 180.0;
    /// let itrf = ITRFCoord::from_geodetic_rad(42.466*DEG2RAD, -71.1516*DEG2RAD, 150.0);
    /// ```
    ///
    pub fn from_geodetic_rad(lat: f64, lon: f64, hae: f64) -> Self {
        let sinp: f64 = lat.sin();
        let cosp: f64 = lat.cos();
        let sinl: f64 = lon.sin();
        let cosl: f64 = lon.cos();

        let f2 = (1.0 - WGS84_F).powi(2);
        let c = 1.0 / cosp.mul_add(cosp, f2 * sinp * sinp).sqrt();
        let s = f2 * c;

        Self {
            itrf: Vec3::from([
                WGS84_A.mul_add(c, hae) * cosp * cosl,
                WGS84_A.mul_add(c, hae) * cosp * sinl,
                WGS84_A.mul_add(s, hae) * sinp,
            ]),
        }
    }

    /// Returns 3-element tuple representing geodetic coordinates
    ///
    /// # Tuple contents:
    ///
    /// * `.0` - latitude in radians
    /// * `.1` - longitude in radians
    /// * `.2` - height above ellipsoid, in meters
    ///
    pub fn to_geodetic_rad(&self) -> (f64, f64, f64) {
        const B: f64 = WGS84_A * (1.0 - WGS84_F);
        const E2: f64 = 1.0 - (1.0 - WGS84_F) * (1.0 - WGS84_F);
        const EP2: f64 = E2 / (1.0 - E2);

        let rho = self.itrf[0].hypot(self.itrf[1]);
        let mut beta: f64 = f64::atan2(self.itrf[2], (1.0 - WGS84_F) * rho);
        let mut sinbeta: f64 = beta.sin();
        let mut cosbeta: f64 = beta.cos();
        let mut phi: f64 = f64::atan2(
            (B * EP2).mul_add(sinbeta.powi(3), self.itrf[2]),
            (WGS84_A * E2).mul_add(-cosbeta.powi(3), rho),
        );
        let mut betanew: f64 = f64::atan2((1.0 - WGS84_F) * phi.sin(), phi.cos());
        for _x in 0..5 {
            beta = betanew;
            sinbeta = beta.sin();
            cosbeta = beta.cos();
            phi = f64::atan2(
                (B * EP2).mul_add(sinbeta.powi(3), self.itrf[2]),
                (WGS84_A * E2).mul_add(-cosbeta.powi(3), rho),
            );
            betanew = f64::atan2((1.0 - WGS84_F) * phi.sin(), phi.cos());
        }
        let lat: f64 = phi;
        let lon: f64 = f64::atan2(self.itrf[1], self.itrf[0]);
        let sinphi: f64 = phi.sin();
        let n: f64 = WGS84_A / (E2 * sinphi).mul_add(-sinphi, 1.0).sqrt();
        let h = rho.mul_add(phi.cos(), (E2 * n).mul_add(sinphi, self.itrf[2]) * sinphi) - n;
        (lat, lon, h)
    }

    /// Returns 3-element tuple representing geodetic coordinates
    ///
    /// # Tuple contents:
    ///
    /// * `.0` - latitude in degrees
    /// * `.1` - longitude in degrees
    /// * `.2` - height above ellipsoid, in meters
    ///
    pub fn to_geodetic_deg(&self) -> (f64, f64, f64) {
        let (lat_rad, lon_rad, hae) = self.to_geodetic_rad();
        (lat_rad.to_degrees(), lon_rad.to_degrees(), hae)
    }

    /// Return geodetic longitude in radians, [-π, π]
    ///
    #[inline]
    pub fn longitude_rad(&self) -> f64 {
        f64::atan2(self.itrf[1], self.itrf[0])
    }

    /// Return geodetic longitude in degrees, [-180, 180]
    #[inline]
    pub fn longitude_deg(&self) -> f64 {
        self.longitude_rad().to_degrees()
    }

    /// return geodetic latitude in radians, [-π/2, π/2]
    #[inline]
    pub fn latitude_rad(&self) -> f64 {
        let (lat, _a, _b) = self.to_geodetic_rad();
        lat
    }

    /// Return height above ellipsoid in meters
    #[inline]
    pub fn hae(&self) -> f64 {
        let (_a, _b, hae) = self.to_geodetic_rad();
        hae
    }

    /// Return geodetic latitude in degrees, [-180, 180]
    #[inline]
    pub fn latitude_deg(&self) -> f64 {
        self.latitude_rad().to_degrees()
    }

    /// Compute location when moving a given Distance at a given heading along the Earth's surface
    /// Altitude assumed to be zero
    ///
    /// # Arguments:
    /// * `distance_m` - Distance in meters to travel along surface of Earth
    /// * `heading_rad` - Initial heading, in radians
    ///
    /// # Returns:
    /// * ITRFCoord representing final position
    ///
    /// # References:
    /// * Uses Vincenty's formula
    ///   See: <https://en.wikipedia.org/wiki/Vincenty%27s_formulae>
    ///
    /// # Arguments:
    ///
    /// * `distance_m` - Distance in meters to travel along surface of Earth
    /// * `heading_rad` - Initial heading, in radians
    ///
    /// # Returns:
    ///
    /// * ITRFCoord representing final position
    ///
    pub fn move_with_heading(&self, distance_m: f64, heading_rad: f64) -> Self {
        let phi1 = self.latitude_rad();
        #[allow(non_upper_case_globals)]
        const a: f64 = WGS84_A;
        #[allow(non_upper_case_globals)]
        const b: f64 = (1.0 - WGS84_F) * WGS84_A;

        let u1 = ((1.0 - WGS84_F) * phi1.tan()).atan();
        let sigma1 = f64::atan2(u1.tan(), heading_rad.cos());
        let sinalpha = u1.cos() * heading_rad.sin();
        let usq = sinalpha.mul_add(-sinalpha, 1.0) * (a / b).mul_add(a / b, -1.0);
        let big_a = (usq / 16384.0).mul_add(
            usq.mul_add(usq.mul_add(175.0f64.mul_add(-usq, 320.0), -768.0), 4096.0),
            1.0,
        );
        let big_b =
            usq / 1024.0 * usq.mul_add(usq.mul_add(47.0f64.mul_add(-usq, 74.0), -128.0), 256.0);
        let mut sigma = distance_m / b / big_a;
        let mut costwosigmam = 0.0;
        for _ in 0..5 {
            costwosigmam = 2.0f64.mul_add(sigma1, sigma).cos();
            let dsigma = big_b
                * sigma.sin()
                * (0.25 * big_b).mul_add(
                    sigma.cos().mul_add(
                        2.0f64.mul_add(costwosigmam.powi(2), -1.0),
                        -(big_b / 6.0
                            * costwosigmam
                            * 4.0f64.mul_add(sigma.sin().powi(2), -3.0)
                            * 4.0f64.mul_add(costwosigmam.powi(2), -3.0)),
                    ),
                    costwosigmam,
                );
            sigma = distance_m / b / big_a + dsigma;
        }
        let phi2 = f64::atan2(
            u1.sin()
                .mul_add(sigma.cos(), u1.cos() * sigma.sin() * heading_rad.cos()),
            (1.0 - WGS84_F)
                * sinalpha.hypot(
                    u1.sin()
                        .mul_add(sigma.sin(), -(u1.cos() * sigma.cos() * heading_rad.cos())),
                ),
        );
        let lam = f64::atan2(
            sigma.sin() * heading_rad.sin(),
            u1.cos()
                .mul_add(sigma.cos(), -(u1.sin() * sigma.sin() * heading_rad.cos())),
        );
        let cossqalpha = sinalpha.mul_add(-sinalpha, 1.0);
        let big_c =
            WGS84_F / 16.0 * cossqalpha * WGS84_F.mul_add(3.0f64.mul_add(-cossqalpha, 4.0), 4.0);
        let delta_lon = ((1.0 - big_c) * WGS84_F * sinalpha).mul_add(
            -(big_c * sigma.sin()).mul_add(
                (big_c * sigma.cos())
                    .mul_add(2.0f64.mul_add(costwosigmam.powi(2), -1.0), costwosigmam),
                sigma,
            ),
            lam,
        );
        let lambda2 = delta_lon + self.longitude_rad();
        Self::from_geodetic_rad(phi2, lambda2, 0.0)
    }

    /// Geodesic distance between two coordinates
    ///
    /// Return Geodesic distance (shortest distance along Earth's surface) in meters
    /// between self and another ITRF coordinate
    ///
    /// Also returns initial and final heading
    ///
    /// # Arguments:
    ///
    /// * `other` - ITRF coordinate for which distance will be computed
    ///
    /// # Outputs:
    ///   Tuple with following values
    ///
    /// * `0` - Distance in meters
    /// * `1` - Starting heading (at self) in radians
    /// * `2` - Final heading (at other) in radians
    //
    /// # References
    //  * Vincenty's formula inverse
    ///   See: <https://en.wikipedia.org/wiki/Vincenty%27s_formulae>
    ///   See: <https://geodesyapps.ga.gov.au/vincenty-inverse>
    ///
    pub fn geodesic_distance(&self, other: &Self) -> (f64, f64, f64) {
        #[allow(non_upper_case_globals)]
        const a: f64 = WGS84_A;
        #[allow(non_upper_case_globals)]
        const b: f64 = (1.0 - WGS84_F) * WGS84_A;

        let lata = self.latitude_rad();
        let latb = other.latitude_rad();
        let lona = self.longitude_rad();
        let lonb = other.longitude_rad();
        let u1 = ((1.0 - WGS84_F) * lata.tan()).atan();
        let u2 = ((1.0 - WGS84_F) * latb.tan()).atan();
        let lam = lonb - lona;
        let londiff = lam;

        let mut lam = lonb - lona;
        let mut cossqalpha = 0.0;
        let mut sinsigma = 0.0;
        let mut cossigma = 0.0;
        let mut cos2sm = 0.0;
        let mut sigma = 0.0;
        for _ in 0..5 {
            sinsigma = (u2.cos() * lam.sin()).hypot(
                u1.cos()
                    .mul_add(u2.sin(), -(u1.sin() * u2.cos() * lam.cos())),
            );
            cossigma = u1.sin().mul_add(u2.sin(), u1.cos() * u2.cos() * lam.cos());
            sigma = f64::atan2(sinsigma, cossigma);
            let sinalpha = (u1.cos() * u2.cos() * lam.sin()) / sigma.sin();
            cossqalpha = sinalpha.mul_add(-sinalpha, 1.0);
            cos2sm = sigma.cos() - (2.0 * u1.sin() * u2.sin()) / cossqalpha;
            let c = WGS84_F / 16.0
                * cossqalpha
                * WGS84_F.mul_add(3.0f64.mul_add(-cossqalpha, 4.0), 4.0);
            lam = ((1.0 - c) * WGS84_F * sinalpha).mul_add(
                (c * sinsigma).mul_add(
                    (c * cossigma).mul_add(2.0f64.mul_add(cos2sm.powi(2), -1.0), cos2sm),
                    sigma,
                ),
                londiff,
            );
        }

        let usq = cossqalpha * (a / b).mul_add(a / b, -1.0);
        let biga = (usq / 16384.0).mul_add(
            usq.mul_add(usq.mul_add(175.0f64.mul_add(-usq, 320.0), -768.0), 4096.0),
            1.0,
        );
        let bigb =
            usq / 1024.0 * usq.mul_add(usq.mul_add(47.0f64.mul_add(-usq, 74.0), -128.0), 256.0);
        let dsigma = bigb
            * sinsigma
            * (0.25 * bigb).mul_add(
                cossigma.mul_add(
                    2.0f64.mul_add(cos2sm.powi(2), -1.0),
                    -(bigb / 6.0
                        * cos2sm
                        * 4.0f64.mul_add(sinsigma.powi(2), -3.0)
                        * 4.0f64.mul_add(cos2sm.powi(2), -3.0)),
                ),
                cos2sm,
            );
        let s = b * biga * (sigma - dsigma);
        let alpha1 = f64::atan2(
            u2.cos() * lam.sin(),
            u1.cos()
                .mul_add(u2.sin(), -(u1.sin() * u2.cos() * lam.cos())),
        );
        let alpha2 = f64::atan2(
            u1.cos() * lam.sin(),
            (-u1.sin()).mul_add(u2.cos(), u1.cos() * u2.sin() * lam.cos()),
        );
        (s, alpha1, alpha2)
    }

    /// Return quaternion representing rotation from the
    /// North-East-Down (NED) coordinate frame to the
    /// ITRF coordinate frame
    #[inline]
    pub fn q_ned2itrf(&self) -> Quat {
        let (lat, lon, _) = self.to_geodetic_rad();
        Quat::from_axis_angle(&Vec3::z_axis(), lon)
            * Quat::from_axis_angle(&Vec3::y_axis(), -lat - PI / 2.0)
    }

    /// Convert coordinate to a North-East-Down (NED)
    /// coordinate relative to a reference coordinate
    ///
    /// # Arguments
    ///
    /// * `ref_coord`` - `&ITRFCoord`` representing reference
    ///
    /// # Return
    ///
    /// * `nalgebra::Vector3<f64>` representing NED position
    ///   relative to reference.  Units are meters
    ///
    /// # Examples:
    /// ```
    /// use satkit::itrfcoord::ITRFCoord;
    /// // Create coord
    /// let itrf1 = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 150.0);
    /// // Crate 2nd coord 100 meters above
    /// let itrf2 = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 250.0);
    ///
    /// // Get NED of itrf1 relative to itrf2
    /// let ned = itrf1.to_ned(&itrf2);
    /// // Should return [0.0, 0.0, 100.0]
    /// ```
    ///
    pub fn to_ned(&self, ref_coord: &Self) -> Vec3 {
        self.q_ned2itrf().conjugate() * (self.itrf - ref_coord.itrf)
    }

    /// Return quaternion representing rotation from the
    /// East-North-Up (ENU) coordinate frame to the
    /// ITRF coordinate frame
    pub fn q_enu2itrf(&self) -> Quat {
        let (lat, lon, _) = self.to_geodetic_rad();
        Quat::from_axis_angle(&Vec3::z_axis(), lon + PI / 2.0)
            * Quat::from_axis_angle(&Vec3::x_axis(), PI / 2.0 - lat)
    }

    /// Convert coordinate to a East-North-Up (ENU)
    /// coordinate relative to a reference coordinate
    ///
    /// # Arguments
    ///
    /// * ref_coord - &ITRFCoord representing reference
    ///
    /// # Return
    ///
    /// * `nalgebra::Vector3<f64>` representing ENU position
    ///   relative to reference.  Units are meters
    ///
    /// # Examples:
    /// ```
    /// use satkit::itrfcoord::ITRFCoord;
    /// // Create coord
    /// let itrf1 = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 150.0);
    /// // Crate 2nd coord 100 meters above
    /// let itrf2 = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 250.0);
    ///
    /// // Get ENU of itrf1 relative to itrf2
    /// let enu = itrf1.to_ned(&itrf2);
    /// // Should return [0.0, 0.0, -100.0]
    /// ```
    ///
    pub fn to_enu(&self, other: &Self) -> Vec3 {
        self.q_enu2itrf().conjugate() * (self.itrf - other.itrf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_geodesic() {
        // Lets pick a random point and try it out...
        let mumbai = ITRFCoord::from_geodetic_deg(19.16488608334183, 72.8314881731579, 0.0);
        let dubai = ITRFCoord::from_geodetic_deg(25.207843059422945, 55.27053859644447, 0.0);
        let (dist, h0, h1) = mumbai.geodesic_distance(&dubai);
        // from google maps, distance is 1,926.80 km
        // From <https://geodesyapps.ga.gov.au/vincenty-inverse>
        // Distance = 1928536.609m
        // Forward azimuth = 293.466588 deg
        // Reverse azimuth = 106.780805 deg
        assert_relative_eq!(dist, 1928536.609, max_relative = 1.0e-8);
        assert_relative_eq!(
            2.0f64.mul_add(PI, h0),
            293.466588f64.to_radians(),
            max_relative = 1.0e-6
        );
        assert_relative_eq!(h1 + PI, 106.780805f64.to_radians(), max_relative = 1.0e-6);

        // Moving from Mumbai at the given distance and heading should get us to Dubai
        let testpoint = mumbai.move_with_heading(dist, h0);

        // Check differences
        let delta = dubai - testpoint;
        assert_abs_diff_eq!(delta.norm(), 0.0, epsilon = 1.0e-6);
    }

    #[test]
    fn geodetic() {
        let lat_deg: f64 = 42.466;
        let lon_deg: f64 = -71.0;
        let hae: f64 = 150.0;
        let itrf = ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, hae);
        println!("{}", itrf);
        // Check conversions
        assert!(((lat_deg - 42.466) / 42.466).abs() < 1.0e-6);
        assert!(((lon_deg + 71.0) / 71.0).abs() < 1.0e-6);
        assert!(((hae - 150.0) / 150.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_ned_enu() {
        let lat_deg: f64 = 42.466;
        let lon_deg: f64 = -74.0;
        let hae: f64 = 150.0;
        let itrf1 = ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, hae);
        let itrf2 = ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, hae + 100.0);
        let ned = itrf2.to_ned(&itrf1);
        let enu = itrf2.to_enu(&itrf1);
        assert!(enu[0].abs() < 1.0e-6);
        assert!(enu[1].abs() < 1.0e-6);
        assert!(((enu[2] - 100.0) / 100.0).abs() < 1.0e-6);
        assert!(ned[0].abs() < 1.0e-6);
        assert!(ned[1].abs() < 1.0e-6);
        assert!(((ned[2] + 100.0) / 100.0).abs() < 1.0e-6);

        let dvec = Vec3::from([-100.0, -200.0, 300.0]);
        let itrf3 = itrf2 + itrf2.q_ned2itrf() * dvec;
        let nedvec = itrf3.to_ned(&itrf2);
        let itrf4 = itrf2 + itrf2.q_enu2itrf() * dvec;
        let enuvec = itrf4.to_enu(&itrf2);
        for x in 0..3 {
            assert!(((nedvec[x] - dvec[x]) / nedvec[x]).abs() < 1.0e-3);
            assert!(((enuvec[x] - dvec[x]) / nedvec[x]).abs() < 1.0e-3);
        }
        /*
        let q = Quat::from_axis_angle(&Vec3::z_axis(), -0.003);
        println!("{}", q);
        println!("{}", q.to_rotation_matrix());
        */

        let itrf1 = ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, hae);
        let itrf2 = itrf1 + itrf1.q_ned2itrf() * na::vector![0.0, 0.0, 10000.0];
        println!("height diff = {}", itrf2.hae() - itrf1.hae());
    }
}
