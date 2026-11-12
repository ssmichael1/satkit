use std::f64::consts::PI;

use crate::consts::WGS84_A;
use crate::consts::WGS84_F;

use crate::mathtypes::*;

use thiserror::Error;

/// Errors produced by the `itrfcoord` module.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned by [`ITRFCoord::from_slice`] and the [`TryFrom`] impls
    /// when the input slice does not contain exactly three elements.
    #[error("Input slice must have 3 elements, got {got}")]
    InvalidSliceLength { got: usize },
}

/// Convenient type alias used throughout the `itrfcoord` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Geodetic coordinates with named fields
///
/// Provides latitude, longitude, and height above ellipsoid
/// as named fields instead of a 3-element tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geodetic {
    /// Latitude in radians, [-π/2, π/2]
    pub latitude_rad: f64,
    /// Longitude in radians, [-π, π]
    pub longitude_rad: f64,
    /// Height above WGS84 ellipsoid in meters
    pub height_m: f64,
}

impl Geodetic {
    /// Return latitude in degrees
    pub fn latitude_deg(&self) -> f64 {
        self.latitude_rad.to_degrees()
    }

    /// Return longitude in degrees
    pub fn longitude_deg(&self) -> f64 {
        self.longitude_rad.to_degrees()
    }
}

impl std::fmt::Display for Geodetic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Geodetic(lat: {:8.4} deg, lon: {:8.4} deg, height: {:5.2} m)",
            self.latitude_deg(),
            self.longitude_deg(),
            self.height_m,
        )
    }
}

/// Representation of a coordinate in the
/// International Terrestrial Reference Frame (ITRF).
///
/// This coordinate object can be created from and also
/// output to Geodetic coordinates (latitude, longitude,
/// height above ellipsoid).
///
/// Functions are also available to provide rotation
/// quaternions to the East-North-Up frame
/// and North-East-Down frame at this coordinate.
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct ITRFCoord {
    /// The ITRF coordinate vector, in meters.
    pub itrf: Vector3,
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

impl std::ops::Add<Vector3> for ITRFCoord {
    type Output = Self;
    fn add(self, other: Vector3) -> Self::Output {
        Self {
            itrf: self.itrf + other,
        }
    }
}

impl std::ops::Sub<Vector3> for ITRFCoord {
    type Output = Self;
    fn sub(self, other: Vector3) -> Self::Output {
        Self {
            itrf: self.itrf - other,
        }
    }
}

impl std::ops::Sub<Self> for ITRFCoord {
    type Output = Vector3;
    fn sub(self, other: Self) -> Vector3 {
        self.itrf - other.itrf
    }
}

impl std::convert::From<[f64; 3]> for ITRFCoord {
    /// Converts an array of three `f64` values (in meters) to an [`ITRFCoord`].
    fn from(v: [f64; 3]) -> Self {
        Self {
            itrf: Vector3::from_array(v),
        }
    }
}

impl std::convert::TryFrom<&[f64]> for ITRFCoord {
    type Error = Error;
    /// Converts a slice of three `f64` values (in meters) to an [`ITRFCoord`].
    fn try_from(v: &[f64]) -> Result<Self> {
        if v.len() != 3 {
            return Err(Error::InvalidSliceLength { got: v.len() });
        }
        Ok(Self {
            itrf: numeris::vector![v[0], v[1], v[2]],
        })
    }
}

impl std::convert::From<Vector3> for ITRFCoord {
    /// Converts a [`Vector3`] (in meters) to an [`ITRFCoord`].
    fn from(v: Vector3) -> Self {
        Self { itrf: v }
    }
}

impl std::convert::From<ITRFCoord> for Vector3 {
    /// Converts an [`ITRFCoord`] to a [`Vector3`] (in meters).
    fn from(itrf: ITRFCoord) -> Self {
        itrf.itrf
    }
}

/// WGS-84 semi-minor axis, meters
const WGS84_B: f64 = (1.0 - WGS84_F) * WGS84_A;

/// Convergence tolerance on the auxiliary-sphere longitude, radians
/// (6e-7 m on the ground)
const VINCENTY_TOL: f64 = 1.0e-13;

/// Iteration cap for Vincenty's formulas. Away from the antipode the inverse
/// converges in a handful of iterations and the direct in fewer.
const VINCENTY_MAX_ITER: usize = 100;

/// Vincenty's longitude correction: the difference `λ - L` between the
/// longitude difference on the auxiliary sphere and on the ellipsoid, for a
/// geodesic of equatorial azimuth `α` (given as `sinalpha` and `cos²α`),
/// arc length `σ` and midpoint argument `cos 2σₘ`.
fn vincenty_dlon(
    sinalpha: f64,
    cossqalpha: f64,
    sigma: f64,
    sinsigma: f64,
    cossigma: f64,
    cos2sm: f64,
) -> f64 {
    let c = WGS84_F / 16.0 * cossqalpha * WGS84_F.mul_add(3.0f64.mul_add(-cossqalpha, 4.0), 4.0);
    (1.0 - c)
        * WGS84_F
        * sinalpha
        * (c * sinsigma).mul_add(
            (c * cossigma).mul_add(2.0f64.mul_add(cos2sm.powi(2), -1.0), cos2sm),
            sigma,
        )
}

/// Vincenty's distance series: the length in meters of a geodesic arc `σ`
/// on the auxiliary sphere.
fn vincenty_arc(cossqalpha: f64, sigma: f64, sinsigma: f64, cossigma: f64, cos2sm: f64) -> f64 {
    let usq = cossqalpha * (WGS84_A / WGS84_B).mul_add(WGS84_A / WGS84_B, -1.0);
    let biga = (usq / 16384.0).mul_add(
        usq.mul_add(usq.mul_add(175.0f64.mul_add(-usq, 320.0), -768.0), 4096.0),
        1.0,
    );
    let bigb = usq / 1024.0 * usq.mul_add(usq.mul_add(47.0f64.mul_add(-usq, 74.0), -128.0), 256.0);
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
    WGS84_B * biga * (sigma - dsigma)
}

/// Vincenty's inverse formula between two points given by the sine and
/// cosine of their reduced latitudes and their longitude difference `londiff`
/// in (-π, π]. Returns `(distance, initial heading, final heading)`, or
/// `None` when the fixed-point iteration does not settle on the shortest
/// path (nearly antipodal points): it fails to converge within
/// [`VINCENTY_MAX_ITER`] iterations, or leaves the half-turn `|λ| <= π`.
fn vincenty_inverse(
    su1: f64,
    cu1: f64,
    su2: f64,
    cu2: f64,
    londiff: f64,
) -> Option<(f64, f64, f64)> {
    let mut lam = londiff;
    for _ in 0..VINCENTY_MAX_ITER {
        let (sinlam, coslam) = lam.sin_cos();
        let north = cu1 * su2 - su1 * cu2 * coslam;
        let sinsigma = (cu2 * sinlam).hypot(north);
        let cossigma = su1 * su2 + cu1 * cu2 * coslam;
        if sinsigma < 1.0e-15 {
            // Coincident to rounding (zero distance), or antipodal (left to
            // the bisection, which does not divide by sin σ).
            return (cossigma > 0.0).then_some((0.0, 0.0, 0.0));
        }
        let sigma = f64::atan2(sinsigma, cossigma);
        let sinalpha = cu1 * cu2 * sinlam / sinsigma;
        let cossqalpha = sinalpha.mul_add(-sinalpha, 1.0);
        // On the equator cos²α = 0 and cos 2σₘ is undefined; it multiplies
        // zero below and is conventionally taken as zero.
        let cos2sm = if cossqalpha == 0.0 {
            0.0
        } else {
            cossigma - 2.0 * su1 * su2 / cossqalpha
        };
        let lam_new =
            londiff + vincenty_dlon(sinalpha, cossqalpha, sigma, sinsigma, cossigma, cos2sm);
        if lam_new.is_nan() || lam_new.abs() > PI {
            return None;
        }
        if (lam_new - lam).abs() < VINCENTY_TOL {
            let s = vincenty_arc(cossqalpha, sigma, sinsigma, cossigma, cos2sm);
            let alpha1 = f64::atan2(cu2 * sinlam, north);
            let alpha2 = f64::atan2(cu1 * sinlam, cu1 * su2 * coslam - su1 * cu2);
            return Some((s, alpha1, alpha2));
        }
        lam = lam_new;
    }
    None
}

/// The geodesic leaving reduced latitude β₁ (`sb1`, `cb1`; β₁ <= 0) at
/// azimuth α₁ (`sa1`, `ca1`) and ending at reduced latitude β₂ (`sb2`, `cb2`;
/// |β₂| <= -β₁) on its northbound leg. Returns the longitude difference on
/// the ellipsoid, the distance in meters, and sin α₂, cos α₂.
///
/// With these conventions the longitude difference increases monotonically
/// from 0 to π as α₁ goes from 0 to π (Karney 2013, section 5), which is what
/// makes the bisection in [`bisection_inverse`] safe.
fn geodesic_from_azimuth(
    sb1: f64,
    cb1: f64,
    sb2: f64,
    cb2: f64,
    sa1: f64,
    ca1: f64,
) -> (f64, f64, f64, f64) {
    // α₀: azimuth at the equator crossing (Clairaut's relation)
    let salp0 = sa1 * cb1;
    let cossqalpha = ca1.mul_add(ca1, (sa1 * sb1).powi(2));
    // Arc lengths σ and longitudes ω on the auxiliary sphere, measured from
    // the northbound equator crossing: tan σ = sin β / (cos α cos β),
    // tan ω = sin α₀ sin β / (cos α cos β)
    let csig1 = ca1 * cb1;
    let sa2 = if cb2 == cb1 { sa1 } else { salp0 / cb2 };
    let ca2 = if cb2 != cb1 || sb2.abs() != -sb1 {
        // cos²α₂ cos²β₂ = cos²α₁ cos²β₁ + (cos²β₂ - cos²β₁), with the
        // difference taken in whichever form does not cancel
        let dcos = if cb1 < -sb1 {
            (cb2 - cb1) * (cb1 + cb2)
        } else {
            (sb1 - sb2) * (sb1 + sb2)
        };
        (csig1.mul_add(csig1, dcos)).max(0.0).sqrt() / cb2
    } else {
        ca1.abs()
    };
    let csig2 = ca2 * cb2;
    // A difference that is zero to rounding must be +0, not -0, or atan2
    // returns -π instead of π
    let positive = |x: f64| if x > 0.0 { x } else { 0.0 };
    let sigma = f64::atan2(
        positive(csig1 * sb2 - sb1 * csig2),
        csig1.mul_add(csig2, sb1 * sb2),
    );
    let (somg1, somg2) = (salp0 * sb1, salp0 * sb2);
    let omega = f64::atan2(
        positive(csig1 * somg2 - somg1 * csig2),
        csig1.mul_add(csig2, somg1 * somg2),
    );
    let (sinsigma, cossigma) = sigma.sin_cos();
    let cos2sm = (f64::atan2(sb1, csig1) + f64::atan2(sb2, csig2)).cos();
    let lon = omega - vincenty_dlon(salp0, cossqalpha, sigma, sinsigma, cossigma, cos2sm);
    let s = vincenty_arc(cossqalpha, sigma, sinsigma, cossigma, cos2sm);
    (lon, s, sa2, ca2)
}

/// Inverse geodesic problem solved for the initial azimuth by bisection, for
/// the point pairs on which [`vincenty_inverse`] fails. Same arguments and
/// return value. It uses the same series, so the two agree where both work;
/// it is slower (about 60 evaluations) but cannot fail to converge.
fn bisection_inverse(su1: f64, cu1: f64, su2: f64, cu2: f64, londiff: f64) -> (f64, f64, f64) {
    // Reduce to the canonical configuration of Karney (2013): β₁ <= 0,
    // β₁ <= β₂ <= -β₁ and 0 <= λ₁₂ <= π, by swapping the end points and
    // mirroring in the equator and in the meridian, and remember how to map
    // the azimuths back.
    let mut lonsign = if londiff >= 0.0 { 1.0 } else { -1.0 };
    let lam12 = londiff.abs();
    let (mut sb1, mut cb1, mut sb2, mut cb2) = (su1, cu1, su2, cu2);
    let swapp = if sb1.abs() < sb2.abs() { -1.0 } else { 1.0 };
    if swapp < 0.0 {
        lonsign = -lonsign;
        std::mem::swap(&mut sb1, &mut sb2);
        std::mem::swap(&mut cb1, &mut cb2);
    }
    let latsign = if sb1 < 0.0 { 1.0 } else { -1.0 };
    sb1 *= latsign;
    sb2 *= latsign;
    // A pole has cos β = 0 exactly only in exact arithmetic; keep it positive
    cb1 = cb1.max(f64::MIN_POSITIVE.sqrt());
    cb2 = cb2.max(f64::MIN_POSITIVE.sqrt());

    let (s, mut sa1, mut ca1, mut sa2, mut ca2) = if sb1 == 0.0 && lam12 <= (1.0 - WGS84_F) * PI {
        // Both points on the equator and close enough that the equator
        // itself is the shortest path (beyond that it leaves the equator)
        (WGS84_A * lam12, 1.0, 0.0, 1.0, 0.0)
    } else {
        let (mut lo, mut hi) = (0.0_f64, PI);
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let (sa, ca) = mid.sin_cos();
            if geodesic_from_azimuth(sb1, cb1, sb2, cb2, sa, ca).0 < lam12 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (sa1, ca1) = (0.5 * (lo + hi)).sin_cos();
        let (_, s, sa2, ca2) = geodesic_from_azimuth(sb1, cb1, sb2, cb2, sa1, ca1);
        (s, sa1, ca1, sa2, ca2)
    };

    if swapp < 0.0 {
        std::mem::swap(&mut sa1, &mut sa2);
        std::mem::swap(&mut ca1, &mut ca2);
    }
    sa1 *= swapp * lonsign;
    ca1 *= swapp * latsign;
    sa2 *= swapp * lonsign;
    ca2 *= swapp * latsign;
    (s, f64::atan2(sa1, ca1), f64::atan2(sa2, ca2))
}

impl ITRFCoord {
    /// Returns an ITRF Coordinate given the geodetic inputs
    ///   with degree units for latitude & longitude.
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

    /// Returns an ITRF Coordinate given Cartesian ITRF coordinates.
    ///
    /// # Arguments:
    ///
    /// * `v` - `Vector3` representing ITRF coordinates in meters
    ///
    /// # Examples:
    ///
    /// ```
    /// // Create coord for ~ Boston, MA
    /// use satkit::itrfcoord::ITRFCoord;
    /// let v = numeris::vector![1522386.15660978, -4459627.78585002, 4284030.6890791];
    /// let itrf = ITRFCoord::from_vector(&v);
    /// ```
    ///
    ///
    pub const fn from_vector(v: &Vector3) -> Self {
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
            return Err(Error::InvalidSliceLength { got: v.len() });
        }
        Ok(Self {
            itrf: numeris::vector![v[0], v[1], v[2]],
        })
    }

    /// Returns an ITRF Coordinate given the geodetic inputs
    ///   with radian units for latitude & longitude.
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
            itrf: numeris::vector![
                WGS84_A.mul_add(c, hae) * cosp * cosl,
                WGS84_A.mul_add(c, hae) * cosp * sinl,
                WGS84_A.mul_add(s, hae) * sinp,
            ],
        }
    }

    /// Returns geodetic coordinates as a named struct.
    ///
    /// # Returns
    /// A [`Geodetic`] struct with latitude (radians), longitude (radians),
    /// and height above ellipsoid (meters)
    pub fn to_geodetic(&self) -> Geodetic {
        let (lat, lon, h) = self.to_geodetic_rad();
        Geodetic {
            latitude_rad: lat,
            longitude_rad: lon,
            height_m: h,
        }
    }

    /// Returns 3-element tuple representing geodetic coordinates.
    ///
    /// # Tuple contents:
    ///
    /// * `.0` - latitude in radians
    /// * `.1` - longitude in radians
    /// * `.2` - height above ellipsoid, in meters
    ///
    pub fn to_geodetic_rad(&self) -> (f64, f64, f64) {
        // Bowring's reduced-latitude iteration: B. R. Bowring, "Transformation
        // from Spatial to Geographical Coordinates", Survey Review 23(181),
        // 323–327, 1976, <https://doi.org/10.1179/sre.1976.23.181.323>.
        const B: f64 = WGS84_A * (1.0 - WGS84_F);
        const E2: f64 = 1.0 - (1.0 - WGS84_F) * (1.0 - WGS84_F);
        const EP2: f64 = E2 / (1.0 - E2);

        // One refinement of the reduced latitude reaches double precision
        // from 20 km below the surface to beyond GEO.
        let rho = self.itrf[0].hypot(self.itrf[1]);
        let beta: f64 = f64::atan2(self.itrf[2], (1.0 - WGS84_F) * rho);
        let phi: f64 = f64::atan2(
            (B * EP2).mul_add(beta.sin().powi(3), self.itrf[2]),
            (WGS84_A * E2).mul_add(-beta.cos().powi(3), rho),
        );
        let beta: f64 = f64::atan2((1.0 - WGS84_F) * phi.sin(), phi.cos());
        let phi: f64 = f64::atan2(
            (B * EP2).mul_add(beta.sin().powi(3), self.itrf[2]),
            (WGS84_A * E2).mul_add(-beta.cos().powi(3), rho),
        );
        let lat: f64 = phi;
        let lon: f64 = f64::atan2(self.itrf[1], self.itrf[0]);
        let sinphi: f64 = phi.sin();
        let n: f64 = WGS84_A / (E2 * sinphi).mul_add(-sinphi, 1.0).sqrt();
        let h = rho.mul_add(phi.cos(), (E2 * n).mul_add(sinphi, self.itrf[2]) * sinphi) - n;
        (lat, lon, h)
    }

    /// Returns 3-element tuple representing geodetic coordinates.
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

    /// Return geodetic latitude in degrees, [-90, 90]
    #[inline]
    pub fn latitude_deg(&self) -> f64 {
        self.latitude_rad().to_degrees()
    }

    /// Compute location when moving a given Distance at a given heading along the Earth's surface
    /// The path follows the WGS-84 ellipsoid: the starting height is ignored
    /// and the returned coordinate has zero height above the ellipsoid.
    ///
    /// # Arguments:
    /// * `distance_m` - Distance in meters to travel along surface of Earth
    /// * `heading_rad` - Initial heading, in radians
    ///
    /// # Returns:
    /// * ITRFCoord representing final position
    ///
    /// # References:
    /// * Uses Vincenty's direct formula, iterated to convergence: T. Vincenty, "Direct and Inverse
    ///   Solutions of Geodesics on the Ellipsoid with Application of Nested
    ///   Equations", Survey Review 23(176), 88–93, 1975,
    ///   <https://doi.org/10.1179/sre.1975.23.176.88>
    ///
    pub fn move_with_heading(&self, distance_m: f64, heading_rad: f64) -> Self {
        let (phi1, lon1, _) = self.to_geodetic_rad();
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
        for _ in 0..VINCENTY_MAX_ITER {
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
            let sigma_new = distance_m / b / big_a + dsigma;
            let converged = (sigma_new - sigma).abs() < VINCENTY_TOL;
            sigma = sigma_new;
            if converged {
                break;
            }
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
        let lambda2 = delta_lon + lon1;
        Self::from_geodetic_rad(phi2, lambda2, 0.0)
    }

    /// Geodesic distance in meters between two coordinates
    ///
    /// Returns the shortest distance along the Earth's surface.
    /// This is a convenience wrapper around [`geodesic_distance`](Self::geodesic_distance)
    /// that returns only the distance.
    ///
    /// # Arguments
    /// * `other` - ITRF coordinate to measure distance to
    ///
    /// # Returns
    /// Distance in meters
    pub fn distance_to(&self, other: &Self) -> f64 {
        self.geodesic_distance(other).0
    }

    /// Geodesic distance between two coordinates
    ///
    /// Return Geodesic distance (shortest distance along the WGS-84
    /// ellipsoid) in meters between self and another ITRF coordinate. Both
    /// points are projected onto the ellipsoid: heights are ignored.
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
    /// * `1` - Starting heading (at self) in radians, clockwise from north,
    ///   in (-π, π]
    /// * `2` - Final heading (at other) in radians: the direction of travel
    ///   on arrival (add π for the back azimuth)
    ///
    /// # Accuracy
    /// Vincenty's inverse formula is iterated to convergence, which gives
    /// the distance to better than a millimeter. For nearly antipodal
    /// points that iteration does not converge (or converges on a path that
    /// is not the shortest); the same series are then solved for the
    /// initial heading by bisection, which always converges, in the manner
    /// of Karney (2013). Coincident points return `(0, 0, 0)`. Between
    /// exactly antipodal points, or from a pole, the shortest path is not
    /// unique: the distance is well defined and the headings are those of
    /// one of the shortest paths.
    ///
    /// # References
    /// * T. Vincenty, "Direct and Inverse Solutions of Geodesics on the
    ///   Ellipsoid with Application of Nested Equations", Survey Review
    ///   23(176), 88–93, 1975, <https://doi.org/10.1179/sre.1975.23.176.88>
    /// * C. F. F. Karney, "Algorithms for geodesics", J. Geodesy 87, 43–55,
    ///   2013, <https://doi.org/10.1007/s00190-012-0578-z> (the canonical
    ///   configuration and the monotonic longitude function used near the
    ///   antipode)
    ///
    pub fn geodesic_distance(&self, other: &Self) -> (f64, f64, f64) {
        let (lata, lona, _) = self.to_geodetic_rad();
        let (latb, lonb, _) = other.to_geodetic_rad();
        // Longitude difference reduced to (-π, π]
        let mut londiff = (lonb - lona) % (2.0 * PI);
        if londiff > PI {
            londiff -= 2.0 * PI;
        } else if londiff <= -PI {
            londiff += 2.0 * PI;
        }
        // Coincident points: zero distance, headings defined as zero. (The
        // general formulas would divide rounding noise by rounding noise.)
        if lata == latb && londiff == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        // Reduced latitudes
        let (su1, cu1) = ((1.0 - WGS84_F) * lata.tan()).atan().sin_cos();
        let (su2, cu2) = ((1.0 - WGS84_F) * latb.tan()).atan().sin_cos();
        vincenty_inverse(su1, cu1, su2, cu2, londiff)
            .unwrap_or_else(|| bisection_inverse(su1, cu1, su2, cu2, londiff))
    }

    /// Return quaternion representing rotation from the
    /// North-East-Down (NED) coordinate frame to the
    /// ITRF coordinate frame
    #[inline]
    pub fn q_ned2itrf(&self) -> Quaternion {
        let (lat, lon, _) = self.to_geodetic_rad();
        Quaternion::rotz(lon) * Quaternion::roty(-lat - PI / 2.0)
    }

    /// North-East-Down (NED) vector from `origin` to `self`, expressed in
    /// `origin`'s local-tangent frame.
    ///
    /// The NED triad has its origin at `origin`; `self` is the point being
    /// located. The `Down` component is positive when `self` is below `origin`
    /// along `origin`'s local normal (closer to the Earth's center).
    ///
    /// # Arguments
    ///
    /// * `origin` - `&ITRFCoord` at which the NED frame is anchored
    ///   (the observer / station / base of the local tangent plane).
    ///
    /// # Return
    ///
    /// * `Vector3` `[N, E, D]` from `origin` to `self`, in meters.
    ///
    /// # Note:
    ///   Equivalent to:
    ///         `origin.q_ned2itrf().conjugate() * (self.itrf - origin.itrf)`
    ///
    /// # Examples:
    /// ```
    /// use satkit::itrfcoord::ITRFCoord;
    /// // Ground station at sea level
    /// let station   = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 0.0);
    /// // Aircraft 1 km directly above the station
    /// let aircraft  = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 1_000.0);
    ///
    /// // NED of the aircraft as seen from the station
    /// let ned = aircraft.to_ned(&station);
    /// // ned ≈ [0.0, 0.0, -1_000.0]  (aircraft is *above*, so Down is negative)
    /// ```
    ///
    pub fn to_ned(&self, origin: &Self) -> Vector3 {
        origin.q_ned2itrf().conjugate() * (self.itrf - origin.itrf)
    }

    /// Return quaternion representing rotation from the
    /// East-North-Up (ENU) coordinate frame to the
    /// ITRF coordinate frame
    pub fn q_enu2itrf(&self) -> Quaternion {
        let (lat, lon, _) = self.to_geodetic_rad();
        Quaternion::rotz(lon + PI / 2.0) * Quaternion::rotx(PI / 2.0 - lat)
    }

    /// East-North-Up (ENU) vector from `origin` to `self`, expressed in
    /// `origin`'s local-tangent frame.
    ///
    /// The ENU triad has its origin at `origin`; `self` is the point being
    /// located. The `Up` component is positive when `self` is above `origin`
    /// along `origin`'s local normal (further from the Earth's center) —
    /// the natural mental model is "what direction is `self` from where I'm
    /// standing at `origin`?"
    ///
    /// # Arguments
    ///
    /// * `origin` - `&ITRFCoord` at which the ENU frame is anchored
    ///   (the observer / station / base of the local tangent plane).
    ///
    /// # Return
    ///
    /// * `Vector3` `[E, N, U]` from `origin` to `self`, in meters.
    ///
    ///
    /// # Note:
    ///   Equivalent to:
    ///       `origin.q_enu2itrf().conjugate() * (self.itrf - origin.itrf)`
    ///
    /// # Examples:
    /// ```
    /// use satkit::itrfcoord::ITRFCoord;
    /// // Ground station at sea level
    /// let station   = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 0.0);
    /// // Satellite 400 km directly above the station
    /// let satellite = ITRFCoord::from_geodetic_deg(42.466, -71.1516, 400_000.0);
    ///
    /// // ENU of the satellite as seen from the station
    /// let enu = satellite.to_enu(&station);
    /// // enu ≈ [0.0, 0.0, 400_000.0]  (satellite is overhead — Up is positive)
    /// ```
    ///
    pub fn to_enu(&self, origin: &Self) -> Vector3 {
        origin.q_enu2itrf().conjugate() * (self.itrf - origin.itrf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_poles() {
        use crate::consts::WGS84_A;
        use crate::consts::WGS84_F;
        let polar_radius = WGS84_A * (1.0 - WGS84_F);

        // North pole
        let north = ITRFCoord::from_geodetic_deg(90.0, 0.0, 0.0);
        assert_abs_diff_eq!(north.itrf[0], 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(north.itrf[1], 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(north.itrf[2], polar_radius, epsilon = 1.0e-3);

        // South pole
        let south = ITRFCoord::from_geodetic_deg(-90.0, 0.0, 0.0);
        assert_abs_diff_eq!(south.itrf[0], 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(south.itrf[1], 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(south.itrf[2], -polar_radius, epsilon = 1.0e-3);
    }

    #[test]
    fn test_equator_prime_meridian() {
        use crate::consts::WGS84_A;
        let coord = ITRFCoord::from_geodetic_deg(0.0, 0.0, 0.0);
        assert_abs_diff_eq!(coord.itrf[0], WGS84_A, epsilon = 1.0e-3);
        assert_abs_diff_eq!(coord.itrf[1], 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(coord.itrf[2], 0.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_antimeridian() {
        let c1 = ITRFCoord::from_geodetic_deg(0.0, 180.0, 0.0);
        let c2 = ITRFCoord::from_geodetic_deg(0.0, -180.0, 0.0);
        assert_abs_diff_eq!(c1.itrf[0], c2.itrf[0], epsilon = 1.0e-6);
        assert_abs_diff_eq!(c1.itrf[1], c2.itrf[1], epsilon = 1.0e-6);
        assert_abs_diff_eq!(c1.itrf[2], c2.itrf[2], epsilon = 1.0e-6);
    }

    #[test]
    fn test_geodetic_roundtrip_random() {
        use rand::Rng;
        let mut rng = rand::rng();
        for _ in 0..100 {
            let lat = rng.random_range(-90.0..90.0_f64);
            let lon = rng.random_range(-180.0..180.0_f64);
            let hae = rng.random_range(-500.0..100_000.0_f64);
            let coord = ITRFCoord::from_geodetic_deg(lat, lon, hae);
            let (lat2, lon2, hae2) = coord.to_geodetic_deg();
            assert_abs_diff_eq!(lat, lat2, epsilon = 1.0e-10);
            assert_abs_diff_eq!(lon, lon2, epsilon = 1.0e-10);
            assert_abs_diff_eq!(hae, hae2, epsilon = 1.0e-6);
        }
    }

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
    fn test_geodesic_coincident() {
        // Identical points give exactly zero distance and zero headings, at
        // any latitude (rounding in sin σ used to leave a heading of π away
        // from the equator).
        for (lat, lon) in [(0.0, 0.0), (42.0, -71.0), (-89.0, 179.0), (90.0, 0.0)] {
            let p = ITRFCoord::from_geodetic_deg(lat, lon, 0.0);
            assert_eq!(p.geodesic_distance(&p), (0.0, 0.0, 0.0), "{lat} {lon}");
            assert_eq!(p.distance_to(&p), 0.0);
        }
        // The same ground point at another height is also coincident
        let lo = ITRFCoord::from_geodetic_deg(0.0, 10.0, 0.0);
        let hi = ITRFCoord::from_geodetic_deg(0.0, 10.0, 1000.0);
        assert_eq!(lo.distance_to(&hi), 0.0);
    }

    /// Reference geodesics from GeographicLib 2.x (Karney 2013), WGS-84:
    /// (lat1, lon1, lat2, lon2, distance m, azimuth 1 deg, azimuth 2 deg)
    const GEODESIC_REFERENCE: [(f64, f64, f64, f64, f64, f64, f64); 9] = [
        // Boston - New York
        (
            42.36,
            -71.06,
            40.71,
            -74.01,
            306867.9261,
            -125.678572512,
            -127.635097300,
        ),
        // Greenwich - Sydney
        (
            51.4778,
            -0.0015,
            -33.8568,
            151.2153,
            16982642.2418,
            60.560478098,
            139.170422320,
        ),
        // Wellington - Salamanca: nearly antipodal (Karney's example)
        (
            -41.32,
            174.81,
            40.96,
            -5.5,
            19959679.2674,
            161.067669986,
            18.825195123,
        ),
        (
            35.0,
            139.0,
            -35.0,
            -40.0,
            19935220.3916,
            -90.144894470,
            -90.144894470,
        ),
        // A quarter of the equator
        (0.0, 0.0, 0.0, 90.0, 10018754.1714, 90.0, 90.0),
        // The last equatorial pair whose shortest path is the equator
        (0.0, 0.0, 0.0, 179.0, 19926188.8520, 90.0, 90.0),
        // Nearly antipodal: Vincenty's iteration does not converge on these
        (
            0.0,
            0.0,
            0.5,
            179.7,
            19944127.4208,
            15.556882793,
            164.442513891,
        ),
        (
            30.0,
            0.0,
            -29.9,
            179.95,
            19992657.1400,
            4.496427384,
            175.508078616,
        ),
        // Close to a pole, across it
        (89.0, 10.0, 89.0, -170.0, 223387.7298, 0.0, 180.0),
    ];

    #[test]
    fn test_geodesic_reference_values() {
        let wrap = |a: f64| (a + PI).rem_euclid(2.0 * PI) - PI;
        for (lat1, lon1, lat2, lon2, dist, az1, az2) in GEODESIC_REFERENCE {
            let p1 = ITRFCoord::from_geodetic_deg(lat1, lon1, 0.0);
            let p2 = ITRFCoord::from_geodetic_deg(lat2, lon2, 0.0);
            let (s, h1, h2) = p1.geodesic_distance(&p2);
            let what = format!("({lat1}, {lon1}) -> ({lat2}, {lon2})");
            assert_abs_diff_eq!(s, dist, epsilon = 1.0e-3);
            assert!(
                wrap(h1 - az1.to_radians()).abs() < 1.0e-7,
                "{what}: initial heading {} deg, expected {az1}",
                h1.to_degrees()
            );
            assert!(
                wrap(h2 - az2.to_radians()).abs() < 1.0e-7,
                "{what}: final heading {} deg, expected {az2}",
                h2.to_degrees()
            );
            // The same path travelled backwards
            let (sr, hr1, hr2) = p2.geodesic_distance(&p1);
            assert_abs_diff_eq!(sr, dist, epsilon = 1.0e-3);
            assert!(wrap(hr1 - h2 - PI).abs() < 1.0e-7, "{what}: reversed");
            assert!(wrap(hr2 - h1 - PI).abs() < 1.0e-7, "{what}: reversed");
        }
    }

    #[test]
    fn test_geodesic_antipodal() {
        // Exactly antipodal points: every meridian is a shortest path, of
        // length half the meridian ellipse (20003931.4586 m, GeographicLib).
        // The 5-iteration Vincenty returned 19903593.39 m for the first.
        const HALF_MERIDIAN: f64 = 20003931.4586;
        for (lat1, lon1, lat2, lon2) in [
            (0.0, 0.0, 0.0, 180.0),
            (10.0, 0.0, -10.0, 180.0),
            (-90.0, 0.0, 90.0, 0.0),
            (-33.0, 151.0, 33.0, -29.0),
        ] {
            let p1 = ITRFCoord::from_geodetic_deg(lat1, lon1, 0.0);
            let p2 = ITRFCoord::from_geodetic_deg(lat2, lon2, 0.0);
            let (s, h1, h2) = p1.geodesic_distance(&p2);
            assert_abs_diff_eq!(s, HALF_MERIDIAN, epsilon = 1.0e-3);
            assert!(h1.is_finite() && h2.is_finite());
        }
        // Equatorial points past (1 - f) * 180 deg of longitude: the shortest
        // path leaves the equator. Distance must grow with separation (the
        // old result for 179.5 deg was less than for 179 deg).
        let origin = ITRFCoord::from_geodetic_deg(0.0, 0.0, 0.0);
        let d = |lon: f64| origin.distance_to(&ITRFCoord::from_geodetic_deg(0.0, lon, 0.0));
        assert_abs_diff_eq!(d(179.5), 19980861.9089, epsilon = 1.0e-3);
        let mut last = 0.0;
        for i in 0..=400 {
            let dist = d(178.0 + 0.005 * i as f64);
            assert!(dist > last, "not increasing at step {i}");
            last = dist;
        }
        assert_abs_diff_eq!(last, HALF_MERIDIAN, epsilon = 1.0e-3);
    }

    #[test]
    fn test_geodesic_bisection_matches_vincenty() {
        // Where Vincenty's iteration converges, the bisection solver (used
        // only when it does not) must give the same geodesic.
        use rand::Rng;
        let mut rng = rand::rng();
        let wrap = |a: f64| (a + PI).rem_euclid(2.0 * PI) - PI;
        let mut compared = 0;
        for _ in 0..2000 {
            let lat1 = rng.random_range(-89.0..89.0_f64).to_radians();
            let lat2 = rng.random_range(-89.0..89.0_f64).to_radians();
            let londiff = rng.random_range(-179.0..179.0_f64).to_radians();
            let (su1, cu1) = ((1.0 - WGS84_F) * lat1.tan()).atan().sin_cos();
            let (su2, cu2) = ((1.0 - WGS84_F) * lat2.tan()).atan().sin_cos();
            let Some((s, h1, h2)) = vincenty_inverse(su1, cu1, su2, cu2, londiff) else {
                continue;
            };
            let (sb, hb1, hb2) = bisection_inverse(su1, cu1, su2, cu2, londiff);
            assert_abs_diff_eq!(s, sb, epsilon = 1.0e-4);
            // Headings are ill-conditioned for very short and nearly
            // antipodal paths
            if s > 1.0e3 && s < 19.0e6 {
                assert!(wrap(h1 - hb1).abs() < 1.0e-6, "h1 {h1} vs {hb1}");
                assert!(wrap(h2 - hb2).abs() < 1.0e-6, "h2 {h2} vs {hb2}");
            }
            compared += 1;
        }
        assert!(compared > 1500);
    }

    #[test]
    fn test_move_with_heading_long() {
        // Direct then inverse over long paths (5 fixed iterations of the
        // direct formula were not always enough)
        let start = ITRFCoord::from_geodetic_deg(-41.32, 174.81, 0.0);
        for (dist, heading_deg) in [
            (1.0e3, 10.0),
            (5.0e6, 123.0),
            (15.0e6, -80.0),
            (19.0e6, 40.0),
        ] {
            let heading = f64::to_radians(heading_deg);
            let end = start.move_with_heading(dist, heading);
            let (s, h1, _) = start.geodesic_distance(&end);
            assert_abs_diff_eq!(s, dist, epsilon = 1.0e-3);
            assert_abs_diff_eq!(h1, heading, epsilon = 1.0e-9);
        }
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

        let dvec = numeris::vector![-100.0, -200.0, 300.0];
        let itrf3 = itrf2 + itrf2.q_ned2itrf() * dvec;
        let nedvec = itrf3.to_ned(&itrf2);
        let itrf4 = itrf2 + itrf2.q_enu2itrf() * dvec;
        let enuvec = itrf4.to_enu(&itrf2);
        for x in 0..3 {
            assert!(((nedvec[x] - dvec[x]) / nedvec[x]).abs() < 1.0e-3);
            assert!(((enuvec[x] - dvec[x]) / nedvec[x]).abs() < 1.0e-3);
        }
        let itrf1 = ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, hae);

        // Go east 10 meters
        let itrf3 = itrf1 + itrf1.q_enu2itrf() * numeris::vector![10.0, 0.0, 0.0];
        let (lat3, lon3, _h3) = itrf3.to_geodetic_deg();
        assert!(((lat3 - lat_deg) / lat_deg).abs() < 1.0e-6);
        assert!(((lon3 - (lon_deg + 0.000129)) / lon_deg).abs() < 1.0e-6);

        // Go north 10 meters
        let itrf4 = itrf1 + itrf1.q_enu2itrf() * numeris::vector![0.0, 10.0, 0.0];
        let (lat4, lon4, _h4) = itrf4.to_geodetic_deg();
        assert!(((lat4 - (lat_deg + 0.000090)) / lat_deg).abs() < 1.0e-6);
        assert!(((lon4 - lon_deg) / lon_deg).abs() < 1.0e-6);

        // Go up 10 meters
        let itrf5 = itrf1 + itrf1.q_enu2itrf() * numeris::vector![0.0, 0.0, 10.0];
        let (lat5, lon5, h5) = itrf5.to_geodetic_deg();
        assert!(((lat5 - lat_deg) / lat_deg).abs() < 1.0e-6);
        assert!(((lon5 - lon_deg) / lon_deg).abs() < 1.0e-6);
        assert!(((h5 - (hae + 10.0)) / hae).abs() < 1.0e-6);
    }
}
