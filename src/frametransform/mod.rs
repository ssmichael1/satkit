//! Coordinate Frame Transformations
//!
//! This module provides quaternion-based rotations between reference frames
//! used in satellite astrodynamics.
//!
//! # Available Transforms
//!
//! | Function | From | To | Accuracy | Notes |
//! |---|---|---|---|---|
//! | [`qitrf2gcrf`] | ITRF | GCRF | Full IERS 2010 | Computationally expensive; requires EOP |
//! | [`qgcrf2itrf`] | GCRF | ITRF | Full IERS 2010 | Conjugate of `qitrf2gcrf` |
//! | [`qitrf2gcrf_approx`] | ITRF | GCRF | ~1 arcsec | IAU-76/FK5 approximation; fast |
//! | [`qgcrf2itrf_approx`] | GCRF | ITRF | ~1 arcsec | Conjugate of `qitrf2gcrf_approx` |
//! | [`qteme2itrf`] | TEME | ITRF | Exact | For SGP4 output; Vallado Eq. 3-90 |
//! | [`qteme2gcrf`] | TEME | GCRF | ~1 arcsec | Composes TEME→ITRF via `qitrf2gcrf_approx` |
//! | [`qmod2gcrf`] | MOD | GCRF | Full | Vallado Eqs. 3-88, 3-89 |
//! | [`qtirs2cirs`] | TIRS | CIRS | Full | Earth rotation angle only |
//!
//! # Frame Descriptions
//!
//! - **GCRF** (Geocentric Celestial Reference Frame): Inertial frame, IERS 2010
//! - **ITRF** (International Terrestrial Reference Frame): Earth-fixed frame
//! - **TEME** (True Equator Mean Equinox): Frame used by SGP4 propagator
//! - **TIRS** (Terrestrial Intermediate Reference System): IERS 2010 intermediate frame
//! - **CIRS** (Celestial Intermediate Reference System): IERS 2010 intermediate frame
//! - **MOD** (Mean of Date): Precession-only frame

mod ierstable;
mod qcirs2gcrs;

use crate::{TimeLike, TimeScale};
use std::f64::consts::PI;

use crate::mathtypes::*;

use super::earth_orientation_params;
pub use qcirs2gcrs::qcirs2gcrs;
pub use qcirs2gcrs::qcirs2gcrs_dxdy;


///
/// Greenwich Mean Sidereal Time
///
/// Vallado algorithm 15:
///
/// GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)
///
///
/// # Arguments
///
/// * `tm` - Instant object representing input time
///
/// # Returns
///
/// * `gmst` - in radians
///
pub fn gmst<T: TimeLike>(tm: &T) -> f64 {
    let tut1: f64 = (tm.as_mjd_with_scale(TimeScale::UT1) - 51544.5) / 36525.0;
    let mut gmst: f64 = tut1.mul_add(
        tut1.mul_add(
            tut1.mul_add(-6.2e-6, 0.093104),
            876600.0f64.mul_add(3600.0, 8640184.812866),
        ),
        67310.54841,
    );

    gmst = ((gmst % 86400.0) / 240.0).to_radians();
    gmst
}

/// Equation of Equinoxes
/// Equation of the equinoxes
pub fn eqeq<T: TimeLike>(tm: &T) -> f64 {
    let d: f64 = tm.as_mjd_with_scale(TimeScale::TT) - 51544.5;
    let omega = PI / 180.0 * 0.052954f64.mul_add(-d, 125.04);
    let l = 0.98565f64.mul_add(d, 280.47).to_radians();
    let epsilon = 0.0000004f64.mul_add(-d, 23.4393).to_radians();
    let d_psi = ((-0.000319f64).mul_add(f64::sin(omega), -(0.000024 * f64::sin(2.0 * l))) * 15.0)
        .to_radians();
    d_psi * f64::cos(epsilon)
}

/// Greenwich Apparent Sidereal Time
pub fn gast<T: TimeLike>(tm: &T) -> f64 {
    gmst(tm) + eqeq(tm)
}

///
/// Earth Rotation Angle
///
/// See
/// [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf?__blob=publicationFile&v=1)
///
/// Equation 5.15
///
///
/// # Arguments:
///
///   * `tm` - Instant at which to compute earth rotation angle
///
/// # Returns:
///
///  * Earth rotation angle, in radians
///
/// # Calculation Details
///
/// * Let t be UT1 Julian date
/// * let f be fractional component of t (fraction of day)
/// * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t − 2451545.0))
///
///
pub fn earth_rotation_angle<T: TimeLike>(tm: &T) -> f64 {
    let t = tm.as_jd_with_scale(TimeScale::UT1);
    let f = t % 1.0;
    2.0 * PI * (0.00273781191135448f64.mul_add(t - 2451545.0, 0.7790572732640 + f) % 1.0)
}

///
/// Rotation from International Terrestrial Reference Frame (ITRF)
/// to the Terrestrial Intermediate Reference System (TIRS)
///
/// # Arguments:
///  * `tm` - Time instant at which to compute rotation
///
/// # Return:
///
///  * Quaternion representing rotation from ITRF to TIRS
///
/// # Notes:
///
/// This function requires use of the Earth orentation parameters
/// (EOP) to compute the rotation. If the EOP are not outside of the
/// valid range of EOP data (1962 to current, predicts to current + ~ 4 months)
/// they will be set to zero, and a warning will be printed to stderr.
///
pub fn qitrf2tirs<T: TimeLike>(tm: &T) -> Quaternion {
    const ASEC2RAD: f64 = PI / 180.0 / 3600.0;
    // Get earth orientation parameters or set them all to zero if not available
    // (function will print warning to stderr if not available)
    let eop = earth_orientation_params::get(tm).unwrap_or([0.0; 6]);
    let xp = eop[1] * ASEC2RAD;
    let yp = eop[2] * ASEC2RAD;
    let t_tt = (tm.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;
    let sp = -47.0e-6 * ASEC2RAD * t_tt;
    Quaternion::rotz(sp) * Quaternion::roty(-xp) * Quaternion::rotx(-yp)
}

///
/// Rotation from True Equator Mean Equinox (TEME) frame
/// to International Terrestrial Reference Frame (ITRF)
///
/// # Arguments
///
/// * `tm` -  Time at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from TEME to ITRF
///
/// # Notes
///
/// * The TEME frame is the default frame output by the
///   SGP4 propagator
/// * This is Equation 3-90 in Vallado
///
pub fn qteme2itrf<T: TimeLike>(tm: &T) -> Quaternion {
    qitrf2tirs(tm).conjugate() * Quaternion::rotz(-gmst(tm))
}

///
/// Rotation from True Equator Mean Equinox (TEME) frame
/// to Geocentric Celestial Reference Frame (GCRF)
///
/// # Arguments
///
/// * `tm` - Time at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from TEME to GCRF
///
/// # Accuracy
///
/// **Approximate**: accurate to within ~1 arcsec (~30 m at Earth's surface).
/// Uses the approximate IAU-76/FK5 reduction internally via
/// [`qitrf2gcrf_approx`] composed with [`qteme2itrf`].
///
/// # Notes
///
/// * The TEME frame is the default frame output by the
///   SGP4 propagator
///
pub fn qteme2gcrf<T: TimeLike>(tm: &T) -> Quaternion {
    qitrf2gcrf_approx(tm) * qteme2itrf(tm)
}

///
/// Rotate from Mean Equinox of Date (MOD) coordinate frame
/// to Geocentric Celestial Reference Frame
///
/// # Arguments
///
/// * `tm` - Time at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from MOD to GCRF
///
/// # Notes
///
/// * Equations 3-88 and 3-89 in Vallado
///
pub fn qmod2gcrf<T: TimeLike>(tm: &T) -> Quaternion {
    const ASEC2RAD: f64 = PI / 180.0 / 3600.0;
    let tt = (tm.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;

    let zeta = tt.mul_add(
        tt.mul_add(
            tt.mul_add(
                tt.mul_add(tt.mul_add(-0.0000003173, -0.000005971), 0.01801828),
                0.2988499,
            ),
            2306.083227,
        ),
        2.650545,
    );
    let z = tt.mul_add(
        tt.mul_add(
            tt.mul_add(
                tt.mul_add(tt.mul_add(-0.0000002904, -0.000028596), 0.01826837),
                1.0927348,
            ),
            2306.077181,
        ),
        -2.650545,
    );
    let theta = tt
        * tt.mul_add(
            tt.mul_add(
                tt.mul_add(tt.mul_add(-0.0000001274, -0.000007089), -0.04182264),
                -0.42949342,
            ),
            2004.191903,
        );
    Quaternion::rotz(-zeta * ASEC2RAD) * Quaternion::roty(theta * ASEC2RAD) * Quaternion::rotz(-z * ASEC2RAD)
}

///
/// Approximate rotation from
/// Geocentric Celestial Reference Frame to
/// International Terrestrial Reference Frame
///
///
///  Arguments
///
/// * `tm` -  Time at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing approximate rotation from GCRF to ITRF
///
/// # Notes
///
/// * Accurate to approx. 1 arcsec
///
/// * This uses an approximation of the IAU-76/FK5 Reduction
///   See Vallado section 3.7.3
///
/// * For a reference, see "Explanatory Supplement to the
///   Astronomical Almanac", 2013, Ch. 6
///
/// # Note — velocity transforms
///
/// This quaternion rotates **position** vectors between GCRF and ITRF
/// correctly, but **is not sufficient for velocity** on its own. Because
/// ITRF is a rotating frame, the velocity transform picks up an extra
/// \\( \\vec{\\omega}_\\oplus \\times \\vec{r} \\) term (~470 m/s at LEO)
/// that this quaternion does not include. For full state transforms
/// (position + velocity) use [`gcrf_to_itrf_state`] /
/// [`itrf_to_gcrf_state`] instead.
pub fn qgcrf2itrf_approx<T: TimeLike>(tm: &T) -> Quaternion {
    // Neglecting polar motion
    let qitrf2tod_approx: Quaternion = Quaternion::rotz(gast(tm));

    (qmod2gcrf(tm) * qtod2mod_approx(tm) * qitrf2tod_approx).conjugate()
}

///
/// Approximate rotation from
/// International Terrestrial Reference Frame to
/// Geocentric Celestial Reference Frame
///
///
///  Arguments
///
/// * `tm` -  Time at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing approximate rotation from ITRF to GCRF
///
/// # Notes
///
/// * Accurate to approx. 1 arcsec
///
/// * This uses an approximation of the IAU-76/FK5 Reduction
///   See Vallado section 3.7.3
///
/// * For a reference, see "Explanatory Supplement to the
///   Astronomical Almanac", 2013, Ch. 6
///
/// # Note — velocity transforms
///
/// This quaternion rotates **position** vectors between ITRF and GCRF
/// correctly, but **is not sufficient for velocity** on its own. Because
/// ITRF is a rotating frame, the velocity transform picks up an extra
/// \\( \\vec{\\omega}_\\oplus \\times \\vec{r} \\) term (~470 m/s at LEO)
/// that this quaternion does not include. For full state transforms
/// (position + velocity) use [`itrf_to_gcrf_state`] /
/// [`gcrf_to_itrf_state`] instead.
pub fn qitrf2gcrf_approx<T: TimeLike>(tm: &T) -> Quaternion {
    qgcrf2itrf_approx(tm).conjugate()
}

/// Approximate rotation from
/// True of Date to Mean of Date
/// coordinate frame
///
/// See Vallado section 3.7.3
///
pub fn qtod2mod_approx<T: TimeLike>(tm: &T) -> Quaternion {
    let d = tm.as_mjd_with_scale(TimeScale::TT) - 51544.5;
    let t = d / 36525.0;

    const DEG2RAD: f64 = PI / 180.0;

    // Compute nutation rotation (accurate to ~ 1 arcsec)
    // This is where the approximation comes in
    let delta_psi = DEG2RAD
        * (-0.0048f64).mul_add(
            f64::sin(0.05295f64.mul_add(-d, 125.0) * DEG2RAD),
            -(0.0004 * f64::sin(1.97129f64.mul_add(d, 200.9) * DEG2RAD)),
        );
    let delta_epsilon = DEG2RAD
        * 0.0026f64.mul_add(
            f64::cos(0.05295f64.mul_add(-d, 125.0) * DEG2RAD),
            0.0002 * f64::cos(1.97129f64.mul_add(d, 200.9) * DEG2RAD),
        );
    let epsilon_a = DEG2RAD
        * t.mul_add(
            t.mul_add(
                t.mul_add(
                    t.mul_add(
                        -5.76e-7 / 3600.0 + t * -4.34E-8 / 3600.0,
                        0.00200340 / 3600.0,
                    ),
                    -0.0001831 / 3600.0,
                ),
                -46.836769 / 3600.0,
            ),
            23.0 + 26.0 / 60.0 + 21.406 / 3600.0,
        );
    let epsilon = epsilon_a + delta_epsilon;
    Quaternion::rotx(epsilon_a) * Quaternion::rotz(-delta_psi) * Quaternion::rotx(-epsilon)
}

///
/// Quaternion representing rotation from the
/// International Terrestrial Reference Frame (ITRF)
/// to the Geocentric Celestial Reference Frame (GCRF)
///
/// Performs full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
///
/// # Arguments
///
/// * `tm` - Time instant at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from ITRF to GCRF
///
/// # Notes:
///
///  * Uses the full IERS 2010 reduction, see
///    [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf)
///    Equation 5.1
///
///  * This is **very** computationally expensive; for most
///    applications, the approximate rotation will work just fine
///
/// * This computation **does not** include impact of the
///   Earth solid tides, but it does include polar motion,
///   precession, and nutation
///
/// * This function requires use of the Earth orientation parameters
///   (EOP) to compute the rotation. If the EOP are not outside of the
///   valid range of EOP data (1962 to current, predicts to current + ~ 4 months)
///   they will be set to zero, and a warning will be printed to stderr.
///
/// # Note — velocity transforms
///
/// This quaternion rotates **position** vectors between ITRF and GCRF
/// correctly, but **is not sufficient for velocity** on its own. Because
/// ITRF is a rotating frame, the velocity transform picks up an extra
/// \\( \\vec{\\omega}_\\oplus \\times \\vec{r} \\) term (~470 m/s at LEO)
/// that this quaternion does not include. For full state transforms
/// (position + velocity) use [`itrf_to_gcrf_state`] /
/// [`gcrf_to_itrf_state`] instead.
pub fn qitrf2gcrf<T: TimeLike>(tm: &T) -> Quaternion {
    // w is rotation from international terrestrial reference frame
    // to terrestrial intermediate reference frame
    let eop = earth_orientation_params::get(tm).unwrap_or([0.0; 6]);

    // Compute this here instead of using function above, so that
    // we only have to get earth orientation parameters once
    let w = {
        const ASEC2RAD: f64 = PI / 180.0 / 3600.0;
        let xp = eop[1] * ASEC2RAD;
        let yp = eop[2] * ASEC2RAD;
        let t_tt = (tm.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;
        let sp = -47.0e-6 * ASEC2RAD * t_tt;
        Quaternion::rotz(sp) * Quaternion::roty(-xp) * Quaternion::rotx(-yp)
    };
    let r = qtirs2cirs(tm);
    let q = qcirs2gcrs_dxdy(tm, Some((eop[4], eop[5])));
    q * r * w
}

///
/// Quaternion representing rotation from the
/// Geocentric Celestial Reference Frame (GCRF)
/// to the International Terrestrial Reference Frame (ITRF)
///
///
/// # Arguments
///
/// * `tm` - Time instant at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from GCRF to ITRF
///
/// # Notes:
///
///  * Uses the full IERS 2010 reduction, see
///    [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf?__blob=publicationFile&v=1)
///    Equation 5.1
///
///  * **Note** This is **very** computationally expensive; for most
///    applications, the approximate rotation will work just fine
///
/// # Note — velocity transforms
///
/// This quaternion rotates **position** vectors between GCRF and ITRF
/// correctly, but **is not sufficient for velocity** on its own. Because
/// ITRF is a rotating frame, the velocity transform picks up an extra
/// \\( \\vec{\\omega}_\\oplus \\times \\vec{r} \\) term (~470 m/s at LEO)
/// that this quaternion does not include. For full state transforms
/// (position + velocity) use [`gcrf_to_itrf_state`] /
/// [`itrf_to_gcrf_state`] instead.
pub fn qgcrf2itrf<T: TimeLike>(tm: &T) -> Quaternion {
    qitrf2gcrf(tm).conjugate()
}

///
/// Quaternion representing rotation from the
/// Terrestrial Intermediate Reference System
/// to the Celestial Intermediate Reference System
///
/// A rotation about zhat by -Earth Rotation Angle
///
/// # Arguments
///
/// * `tm` - Time instance at which to compute rotation
///
/// # Returns
///
/// * Quaternion representing rotation from TIRS to CIRS
///
///
/// See [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf)
/// Equation 5.5
///
#[inline]
pub fn qtirs2cirs<T: TimeLike>(tm: &T) -> Quaternion {
    Quaternion::rotz(earth_rotation_angle(tm))
}

/// Compute the RTN-to-GCRF rotation matrix from position and velocity.
///
/// RTN frame (CCSDS OEM/OMM convention, also known as RIC or RSW):
///   * R = radial (outward from Earth centre)
///   * T = tangential / in-track (perpendicular to R in the orbit plane)
///   * N = normal / cross-track (along angular momentum, h = r × v)
///
/// # Arguments
///
/// * `pos_gcrf` - Position vector in GCRF [m]
/// * `vel_gcrf` - Velocity vector in GCRF [m/s]
///
/// # Returns
///
/// 3x3 rotation matrix that transforms vectors from RTN to GCRF.
pub fn rtn_to_gcrf(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    let r_hat = pos_gcrf.normalize();
    let h = pos_gcrf.cross(vel_gcrf);
    let h_hat = h.normalize();
    let t_hat = h_hat.cross(&r_hat);
    let mut dcm = Matrix3::zeros();
    dcm.set_block(0, 0, &r_hat);
    dcm.set_block(0, 1, &t_hat);
    dcm.set_block(0, 2, &h_hat);
    dcm
}

/// Compute the GCRF-to-RTN rotation matrix from position and velocity.
///
/// Transpose of [`rtn_to_gcrf`].
pub fn gcrf_to_rtn(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    rtn_to_gcrf(pos_gcrf, vel_gcrf).transpose()
}

/// Backward-compatibility alias for [`rtn_to_gcrf`]. The `ric_to_gcrf`
/// name was the canonical spelling in earlier satkit versions; new code
/// should use `rtn_to_gcrf` (which matches the CCSDS OEM/OMM convention),
/// but this alias is kept so existing Rust callers don't break.
#[inline]
pub fn ric_to_gcrf(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    rtn_to_gcrf(pos_gcrf, vel_gcrf)
}

/// Backward-compatibility alias for [`gcrf_to_rtn`]. See [`ric_to_gcrf`]
/// for rationale.
#[inline]
pub fn gcrf_to_ric(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    gcrf_to_rtn(pos_gcrf, vel_gcrf)
}

/// Compute the NTW-to-GCRF rotation matrix from position and velocity.
///
/// NTW frame (Vallado §3.3):
/// * **N** (in-plane normal to velocity): T̂ × Ŵ. For circular orbits this
///   coincides with the outward radial direction; for eccentric orbits it
///   leans off-radial by the flight-path angle.
/// * **T** (tangent, along velocity): v̂ = v / |v|
/// * **W** (cross-track, along angular momentum): (r × v) / |r × v|
///
/// Unlike [`rtn_to_gcrf`], the tangent axis is parallel to the velocity
/// vector regardless of orbit eccentricity, so a +T delta-v adds that exact
/// magnitude to |v|. This makes NTW the natural frame for thrust-along-
/// velocity maneuver planning.
///
/// # Arguments
/// * `pos_gcrf` - Position vector in GCRF [m]
/// * `vel_gcrf` - Velocity vector in GCRF [m/s]
///
/// # Returns
/// 3x3 rotation matrix that transforms vectors from NTW to GCRF.
pub fn ntw_to_gcrf(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    let t_hat = vel_gcrf.normalize();
    let h = pos_gcrf.cross(vel_gcrf);
    let w_hat = h.normalize();
    let n_hat = t_hat.cross(&w_hat);
    let mut dcm = Matrix3::zeros();
    dcm.set_block(0, 0, &n_hat);
    dcm.set_block(0, 1, &t_hat);
    dcm.set_block(0, 2, &w_hat);
    dcm
}

/// Compute the GCRF-to-NTW rotation matrix. Transpose of [`ntw_to_gcrf`].
pub fn gcrf_to_ntw(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    ntw_to_gcrf(pos_gcrf, vel_gcrf).transpose()
}

/// Compute the LVLH-to-GCRF rotation matrix from position and velocity.
///
/// Local-Vertical / Local-Horizontal, the classical crewed-spaceflight and
/// GN&C "body-pointing" frame used on the ISS and most Earth-pointing
/// vehicles:
///
/// * **z** = −r̂ (nadir; pointing toward Earth centre)
/// * **y** = −ĥ (opposite angular momentum; "right" for a prograde mission)
/// * **x** = ŷ × ẑ = ĥ × r̂ / |ĥ × r̂| (completes right-handed system;
///   roughly velocity-aligned for circular orbits)
///
/// Geometrically the axes span the same orbital plane as [`rtn_to_gcrf`]
/// (RIC/RSW/RTN) but with different labels and sign conventions — LVLH +x
/// equals RIC +I, LVLH −z equals RIC +R, and LVLH −y equals RIC +C. For
/// eccentric orbits neither LVLH's x nor RIC's I is strictly along
/// velocity; use [`ntw_to_gcrf`] if you need that property.
///
/// # Arguments
/// * `pos_gcrf` - Position vector in GCRF [m]
/// * `vel_gcrf` - Velocity vector in GCRF [m/s]
///
/// # Returns
/// 3x3 rotation matrix that transforms vectors from LVLH to GCRF.
pub fn lvlh_to_gcrf(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    let r_hat = pos_gcrf.normalize();
    let h = pos_gcrf.cross(vel_gcrf);
    let h_hat = h.normalize();
    let z_hat = r_hat * -1.0;
    let y_hat = h_hat * -1.0;
    let x_hat = y_hat.cross(&z_hat);
    let mut dcm = Matrix3::zeros();
    dcm.set_block(0, 0, &x_hat);
    dcm.set_block(0, 1, &y_hat);
    dcm.set_block(0, 2, &z_hat);
    dcm
}

/// Compute the GCRF-to-LVLH rotation matrix. Transpose of [`lvlh_to_gcrf`].
pub fn gcrf_to_lvlh(pos_gcrf: &Vector3, vel_gcrf: &Vector3) -> Matrix3 {
    lvlh_to_gcrf(pos_gcrf, vel_gcrf).transpose()
}

/// Transform a satellite state (position and velocity) from ITRF
/// (Earth-fixed, rotating) to GCRF (inertial) at a given time.
///
/// This is the correct way to convert a state vector between ITRF and
/// GCRF: it accounts for both the rotation of position and the
/// Earth-rotation contribution to velocity. A plain quaternion rotation
/// (via [`qitrf2gcrf`] / [`qitrf2gcrf_approx`]) is fine for position but
/// gives an incorrect velocity because it ignores the
/// \\( \\vec{\\omega}_\\oplus \\times \\vec{r} \\) term (~470 m/s at LEO).
///
/// # Math
///
/// The IERS 2010 ITRF → GCRF reduction decomposes into three stages:
/// polar motion \\(W\\) (ITRF → TIRS), Earth rotation \\(R\\) (TIRS →
/// CIRS, a rotation about the CIO pole by the Earth Rotation Angle),
/// and precession-nutation \\(Q\\) (CIRS → GCRF):
///
/// \\[ \\vec{r}_\\mathrm{gcrf} = Q\\,R\\,W\\,\\vec{r}_\\mathrm{itrf} \\]
///
/// The only rapid time dependence is \\(R(t)\\). Taking the time
/// derivative and keeping only the dominant rotation-rate term:
///
/// \\[ \\vec{v}_\\mathrm{gcrf} = Q R W\\,\\vec{v}_\\mathrm{itrf}
///     + Q R\\,(\\vec{\\omega}_\\oplus^\\mathrm{tirs} \\times \\vec{r}_\\mathrm{tirs}) \\]
///
/// **The cross product must be evaluated in TIRS**, not in ITRF. In
/// TIRS, Earth's angular velocity is exactly along \\(\\hat z\\) by
/// definition — the Earth rotation matrix \\(R\\) is a rotation about
/// the CIO polar axis, and TIRS is defined such that that axis is its
/// \\(\\hat z\\). Computing \\(\\vec{\\omega}_\\oplus \\times \\vec{r}\\)
/// in ITRF would introduce a polar-motion-sized error (~0.3 arcsec,
/// hence ~11 mm/s at LEO velocities). Computing it in GCRF would be
/// wildly inaccurate because Earth's rotation axis drifts from GCRF
/// \\(+\\hat z\\) by the accumulated precession angle (tens of degrees
/// over a century).
///
/// # Implementation
///
/// 1. Apply polar motion to get the state in TIRS: `r_tirs = W r_itrf`, `v_tirs = W v_itrf`.
/// 2. Add the Earth-rotation sweep in TIRS: `v_tirs += omega_earth x r_tirs`, with `omega_earth = (0, 0, omega_e)` exactly.
/// 3. Rotate TIRS → CIRS → GCRF via the full IERS 2010 chain.
///
/// Uses the full IERS 2010 reduction (polar motion + Earth rotation +
/// precession-nutation with dX/dY corrections from EOP). This is
/// computationally expensive by comparison with the approximate
/// [`qitrf2gcrf_approx`]; if you need speed over correctness, roll your
/// own using the quaternion helpers.
///
/// # Arguments
///
/// * `pos_itrf` - Position in ITRF [m]
/// * `vel_itrf` - Velocity as observed in ITRF [m/s]. For a point at rest
///   on Earth, this is zero.
/// * `time` - Epoch of the state
///
/// # Returns
///
/// `(pos_gcrf, vel_gcrf)` — the state expressed in GCRF.
pub fn itrf_to_gcrf_state<T: TimeLike>(
    pos_itrf: &Vector3,
    vel_itrf: &Vector3,
    time: &T,
) -> (Vector3, Vector3) {
    // Pull EOP once so both polar motion and the CIRS→GCRF dX/dY
    // correction stay consistent.
    let eop = crate::earth_orientation_params::get(time).unwrap_or([0.0; 6]);

    // ITRF → TIRS via polar motion. Same construction as the inline
    // `w` in `qitrf2gcrf` to avoid a second EOP lookup.
    let q_itrf_to_tirs = {
        const ASEC2RAD: f64 = PI / 180.0 / 3600.0;
        let xp = eop[1] * ASEC2RAD;
        let yp = eop[2] * ASEC2RAD;
        let t_tt = (time.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;
        let sp = -47.0e-6 * ASEC2RAD * t_tt;
        Quaternion::rotz(sp) * Quaternion::roty(-xp) * Quaternion::rotx(-yp)
    };

    // Rotate state into TIRS.
    let pos_tirs = q_itrf_to_tirs * *pos_itrf;
    let vel_tirs = q_itrf_to_tirs * *vel_itrf;

    // Add the Earth-rotation sweep term in TIRS, where ω⊕ is exactly
    // along +z (the CIO polar axis).
    let omega_tirs: Vector3 = numeris::vector![0.0, 0.0, crate::consts::OMEGA_EARTH];
    let vel_tirs_swept = vel_tirs + omega_tirs.cross(&pos_tirs);

    // TIRS → CIRS → GCRF via the full IERS 2010 chain.
    let q_tirs_to_gcrf =
        qcirs2gcrs_dxdy(time, Some((eop[4], eop[5]))) * qtirs2cirs(time);

    let pos_gcrf = q_tirs_to_gcrf * pos_tirs;
    let vel_gcrf = q_tirs_to_gcrf * vel_tirs_swept;

    (pos_gcrf, vel_gcrf)
}

/// Approximate version of [`itrf_to_gcrf_state`] using the IAU-76/FK5
/// reduction (accurate to ~1 arcsec for position).
///
/// Neglects polar motion, so ITRF ≡ TIRS for the purposes of this
/// transform and the Earth-rotation sweep term `omega_earth x r` is
/// evaluated directly in ITRF (where ω⊕ is along +z to the same
/// approximation level). Uses [`qitrf2gcrf_approx`] for the rotation;
/// substantially cheaper than the full [`itrf_to_gcrf_state`] when the
/// IERS 2010 precision is not required.
pub fn itrf_to_gcrf_state_approx<T: TimeLike>(
    pos_itrf: &Vector3,
    vel_itrf: &Vector3,
    time: &T,
) -> (Vector3, Vector3) {
    let omega: Vector3 = numeris::vector![0.0, 0.0, crate::consts::OMEGA_EARTH];
    let vel_swept = vel_itrf + omega.cross(pos_itrf);

    let q = qitrf2gcrf_approx(time);
    (q * *pos_itrf, q * vel_swept)
}

/// Transform a satellite state (position and velocity) from GCRF
/// (inertial) to ITRF (Earth-fixed, rotating) at a given time.
///
/// Inverse of [`itrf_to_gcrf_state`]. See that function for the full
/// mathematical derivation. As for the forward transform, the
/// Earth-rotation sweep term \\(\\vec{\\omega}_\\oplus \\times \\vec{r}\\)
/// is computed in **TIRS** (where ω⊕ is exactly along \\(+\\hat z\\)),
/// not in ITRF or GCRF. Uses the full IERS 2010 reduction.
///
/// # Arguments
///
/// * `pos_gcrf` - Position in GCRF [m]
/// * `vel_gcrf` - Velocity in GCRF [m/s]
/// * `time` - Epoch of the state
///
/// # Returns
///
/// `(pos_itrf, vel_itrf)` — the state expressed in ITRF, with
/// `vel_itrf` being the velocity as observed from ITRF.
pub fn gcrf_to_itrf_state<T: TimeLike>(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    time: &T,
) -> (Vector3, Vector3) {
    let eop = crate::earth_orientation_params::get(time).unwrap_or([0.0; 6]);

    // GCRF → CIRS → TIRS (inverse of the forward chain above).
    let q_gcrf_to_tirs =
        (qcirs2gcrs_dxdy(time, Some((eop[4], eop[5]))) * qtirs2cirs(time)).conjugate();

    // Rotate state into TIRS.
    let pos_tirs = q_gcrf_to_tirs * *pos_gcrf;
    let vel_tirs_swept = q_gcrf_to_tirs * *vel_gcrf;

    // Remove the Earth-rotation sweep term in TIRS (same frame where we
    // added it in the forward transform).
    let omega_tirs: Vector3 = numeris::vector![0.0, 0.0, crate::consts::OMEGA_EARTH];
    let vel_tirs = vel_tirs_swept - omega_tirs.cross(&pos_tirs);

    // TIRS → ITRF via inverse polar motion.
    let q_tirs_to_itrf = {
        const ASEC2RAD: f64 = PI / 180.0 / 3600.0;
        let xp = eop[1] * ASEC2RAD;
        let yp = eop[2] * ASEC2RAD;
        let t_tt = (time.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;
        let sp = -47.0e-6 * ASEC2RAD * t_tt;
        (Quaternion::rotz(sp) * Quaternion::roty(-xp) * Quaternion::rotx(-yp)).conjugate()
    };

    let pos_itrf = q_tirs_to_itrf * pos_tirs;
    let vel_itrf = q_tirs_to_itrf * vel_tirs;

    (pos_itrf, vel_itrf)
}

/// Approximate version of [`gcrf_to_itrf_state`] using the IAU-76/FK5
/// reduction (accurate to ~1 arcsec for position). Inverse of
/// [`itrf_to_gcrf_state_approx`]; sweep term subtracted in ITRF
/// (polar motion neglected).
pub fn gcrf_to_itrf_state_approx<T: TimeLike>(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    time: &T,
) -> (Vector3, Vector3) {
    let q = qgcrf2itrf_approx(time);
    let pos_itrf = q * *pos_gcrf;
    let vel_swept = q * *vel_gcrf;

    let omega: Vector3 = numeris::vector![0.0, 0.0, crate::consts::OMEGA_EARTH];
    let vel_itrf = vel_swept - omega.cross(&pos_itrf);

    (pos_itrf, vel_itrf)
}

/// Unified rotation builder: return the DCM that transforms a 3-vector
/// from `frame` into GCRF at the current state.
///
/// This is the general-purpose dispatch for satellite-local orbital
/// frames. For arbitrary frame-to-frame rotations, compose with
/// [`from_gcrf`]:
///
/// ```ignore
/// // NTW -> RIC
/// let dcm = from_gcrf(Frame::RTN, &pos, &vel)? * to_gcrf(Frame::NTW, &pos, &vel)?;
/// ```
///
/// # Supported frames
///
/// * [`Frame::GCRF`] — returns the 3×3 identity (trivial case)
/// * [`Frame::LVLH`] — dispatches to [`lvlh_to_gcrf`]
/// * [`Frame::RTN`] (a.k.a. RSW, RIC) — dispatches to [`rtn_to_gcrf`]
/// * [`Frame::NTW`] — dispatches to [`ntw_to_gcrf`]
///
/// # Errors
///
/// Returns an error for frames that are not satellite-local orbital
/// frames (ITRF, TIRS, CIRS, TEME, EME2000, ICRF). Those frames require
/// a time argument for their rotation to GCRF and are handled by the
/// dedicated quaternion helpers ([`qteme2gcrf`], [`qitrf2gcrf`], etc.).
pub fn to_gcrf(
    frame: crate::Frame,
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
) -> anyhow::Result<Matrix3> {
    use crate::Frame;
    match frame {
        Frame::GCRF => Ok(Matrix3::eye()),
        Frame::LVLH => Ok(lvlh_to_gcrf(pos_gcrf, vel_gcrf)),
        Frame::RTN => Ok(rtn_to_gcrf(pos_gcrf, vel_gcrf)),
        Frame::NTW => Ok(ntw_to_gcrf(pos_gcrf, vel_gcrf)),
        Frame::ITRF
        | Frame::TIRS
        | Frame::CIRS
        | Frame::TEME
        | Frame::EME2000
        | Frame::ICRF => anyhow::bail!(
            "to_gcrf: frame {} is not a satellite-local orbital frame; use the \
             time-based quaternion helpers (qitrf2gcrf, qteme2gcrf, etc.) instead",
            frame
        ),
    }
}

/// Unified rotation builder: return the DCM that transforms a 3-vector
/// from GCRF into `frame` at the current state.
///
/// Transpose of [`to_gcrf`]. See that function's doc for supported
/// frames and error conditions.
pub fn from_gcrf(
    frame: crate::Frame,
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
) -> anyhow::Result<Matrix3> {
    Ok(to_gcrf(frame, pos_gcrf, vel_gcrf)?.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Duration, Instant, TimeScale};

    #[test]
    fn test_gmst() {
        // Vallado example 3-5
        let mut tm = Instant::from_datetime(1992, 8, 20, 12, 14, 0.0).unwrap();
        // Spoof this as UT1 value
        let tdiff = tm.as_mjd_with_scale(TimeScale::UT1) - tm.as_mjd_with_scale(TimeScale::UTC);
        tm -= Duration::from_days(tdiff);
        // Convert to UT1
        let gmval = gmst(&tm).to_degrees();
        let truth = -207.4212121875;
        assert!(((gmval - truth) / truth).abs() < 1.0e-6)
    }

    #[test]
    fn test_qitrf2gcrf_roundtrip() {
        let tm = Instant::from_datetime(2020, 6, 15, 12, 0, 0.0).unwrap();
        let v_itrf = numeris::vector![1000.0, 2000.0, 3000.0];
        let v_gcrf = qitrf2gcrf(&tm) * v_itrf;
        let v_back = qgcrf2itrf(&tm) * v_gcrf;
        assert!((v_back - v_itrf).norm() < 1.0e-12 * v_itrf.norm());
    }

    #[test]
    fn test_earth_rotation_angle() {
        // At J2000.0 epoch (2000-01-01 12:00:00 UT1), ERA should be ~280.46061837°
        // J2000.0 is JD 2451545.0
        let tm = Instant::from_jd_utc(2451545.0);
        let era = earth_rotation_angle(&tm).to_degrees().rem_euclid(360.0);
        assert!((era - 280.46061837).abs() < 0.01);
    }

    #[test]
    fn test_approx_vs_full() {
        let tm = Instant::from_datetime(2010, 3, 15, 6, 0, 0.0).unwrap();
        let v_itrf = numeris::vector![6378137.0, 0.0, 0.0];

        let v_approx = qitrf2gcrf_approx(&tm) * v_itrf;
        let v_full = qitrf2gcrf(&tm) * v_itrf;

        // Should agree within 1 arcsec (~30 m at Earth surface)
        let angle_rad = (v_approx.dot(&v_full) / (v_approx.norm() * v_full.norm())).acos();
        let angle_arcsec = angle_rad.to_degrees() * 3600.0;
        assert!(
            angle_arcsec < 1.0,
            "Approx vs full differ by {:.4} arcsec, expected < 1",
            angle_arcsec
        );
    }

    #[test]
    fn test_qteme2gcrf() {
        // Vallado Example 3-15: verify TEME→GCRF transform
        // Using the same time as test_gcrs2itrf (Vallado Example 3-14)
        let tm = &Instant::from_datetime(2004, 4, 6, 7, 51, 28.386009).unwrap();
        // TEME position from SGP4 output (Vallado example)
        let pitrf = numeris::vector![-1033.4793830, 7901.2952754, 6380.3565958];
        // Get GCRF via ITRF path (known good from test_gcrs2itrf)
        let pgcrf_via_itrf = qitrf2gcrf(tm) * pitrf;
        // Get GCRF via TEME path
        let pteme = qteme2itrf(tm).conjugate() * pitrf;
        let pgcrf_via_teme = qteme2gcrf(tm) * pteme;

        // Both paths should give the same GCRF result
        assert!((pgcrf_via_itrf - pgcrf_via_teme).norm() < 0.1);
    }

    #[test]
    fn test_gcrs2itrf() {
        // Example 3-14 from Vallado
        // With verification of intermediate calculations
        // Input time
        let tm = &Instant::from_datetime(2004, 4, 6, 7, 51, 28.386009).unwrap();
        // Input terrestrial location
        let pitrf = numeris::vector![-1033.4793830, 7901.2952754, 6380.3565958];
        let t_tt = (tm.as_jd_with_scale(TimeScale::TT) - 2451545.0) / 36525.0;
        assert!((t_tt - 0.0426236319).abs() < 1.0e-8);

        let dut1 =
            (tm.as_mjd_with_scale(TimeScale::UT1) - tm.as_mjd_with_scale(TimeScale::UTC)) * 86400.0;
        // We linearly interpolate dut1, so this won't match exactly
        assert!((dut1 + 0.4399619).abs() < 0.01);
        let delta_at =
            (tm.as_mjd_with_scale(TimeScale::TAI) - tm.as_mjd_with_scale(TimeScale::UTC)) * 86400.0;
        assert!((delta_at - 32.0).abs() < 1.0e-7);

        // Slight differences below are due to example using approximate
        // value for dut1 and polar wander, hence the larger than
        // expected errors (though still within ~ 1e-6)
        let ptirs = qitrf2tirs(tm) * pitrf;
        assert!((ptirs[0] + 1033.4750312).abs() < 1.0e-4);
        assert!((ptirs[1] - 7901.3055856).abs() < 1.0e-4);
        assert!((ptirs[2] - 6380.3445327).abs() < 1.0e-4);
        let era = earth_rotation_angle(tm);
        assert!((era.to_degrees() - 312.7552829).abs() < 1.0e-5);
        let pcirs = Quaternion::rotz(era) * ptirs;
        assert!((pcirs[0] - 5100.0184047).abs() < 1e-3);
        assert!((pcirs[1] - 6122.7863648).abs() < 1e-3);
        assert!((pcirs[2] - 6380.3446237).abs() < 1e-3);
        let pgcrf = qcirs2gcrs_dxdy(tm, None) * pcirs;
        assert!((pgcrf[0] - 5102.508959).abs() < 1e-3);
        assert!((pgcrf[1] - 6123.011403).abs() < 1e-3);
        assert!((pgcrf[2] - 6378.136925).abs() < 1e-3);
    }

    /// Vallado Example 3-14 (4th ed.) full state transform in a single
    /// call. The existing [`test_gcrs2itrf`] test verifies the *position*
    /// component by stepping through ITRF → TIRS → CIRS → GCRF; this
    /// test adds the velocity component and uses [`itrf_to_gcrf_state`]
    /// as a single high-level entry point.
    #[test]
    fn test_vallado_3_14_state_single_call() {
        let tm = Instant::from_datetime(2004, 4, 6, 7, 51, 28.386009).unwrap();
        // Vallado 3-14 input state (km, km/s — scale to SI below).
        let pos_itrf_km: Vector3 =
            numeris::vector![-1033.4793830, 7901.2952754, 6380.3565958];
        let vel_itrf_kms: Vector3 =
            numeris::vector![-3.225636520, -2.872451450, 5.531924446];
        let pos_itrf = pos_itrf_km * 1.0e3;
        let vel_itrf = vel_itrf_kms * 1.0e3;

        // Single-call state transform.
        let (pos_gcrf, vel_gcrf) = itrf_to_gcrf_state(&pos_itrf, &vel_itrf, &tm);

        // Expected GCRF state from Vallado Example 3-14 (SI units).
        // Position agrees with the existing test_gcrs2itrf case.
        let pos_gcrf_km = pos_gcrf / 1.0e3;
        let vel_gcrf_kms = vel_gcrf / 1.0e3;

        assert!((pos_gcrf_km[0] - 5102.508959).abs() < 1.0e-3);
        assert!((pos_gcrf_km[1] - 6123.011403).abs() < 1.0e-3);
        assert!((pos_gcrf_km[2] - 6378.136925).abs() < 1.0e-3);

        // Velocity: Vallado 3-14 expected values (km/s). Satkit matches
        // to sub-mm/s.
        assert!(
            (vel_gcrf_kms[0] - (-4.7432196)).abs() < 1.0e-6,
            "v_x = {}", vel_gcrf_kms[0]
        );
        assert!(
            (vel_gcrf_kms[1] - 0.7905366).abs() < 1.0e-6,
            "v_y = {}", vel_gcrf_kms[1]
        );
        assert!(
            (vel_gcrf_kms[2] - 5.5337561).abs() < 1.0e-6,
            "v_z = {}", vel_gcrf_kms[2]
        );

        // Round-trip: GCRF → ITRF → GCRF recovers the original to
        // well under a millimeter / nanometer per second.
        let (pos_back, vel_back) = gcrf_to_itrf_state(&pos_gcrf, &vel_gcrf, &tm);
        let pos_err = (pos_back - pos_itrf).norm();
        let vel_err = (vel_back - vel_itrf).norm();
        assert!(pos_err < 1.0e-6, "pos round-trip err = {} m", pos_err);
        assert!(vel_err < 1.0e-9, "vel round-trip err = {} m/s", vel_err);
    }

    /// Round-trip: GCRF → ITRF → GCRF should recover the original state
    /// to floating-point precision.
    #[test]
    fn test_itrf_gcrf_state_roundtrip() {
        let t = crate::Instant::from_datetime(2024, 3, 15, 12, 34, 56.0).unwrap();
        // A typical LEO state in GCRF
        let pos_gcrf: Vector3 = numeris::vector![6.878e6, 1.23e5, -4.56e5];
        let vel_gcrf: Vector3 = numeris::vector![-123.4, 7600.0, 89.0];

        let (pos_itrf, vel_itrf) = gcrf_to_itrf_state(&pos_gcrf, &vel_gcrf, &t);
        let (pos_back, vel_back) = itrf_to_gcrf_state(&pos_itrf, &vel_itrf, &t);

        let pos_err = (pos_gcrf - pos_back).norm();
        let vel_err = (vel_gcrf - vel_back).norm();
        assert!(pos_err < 1.0e-6, "position round-trip error = {} m", pos_err);
        assert!(vel_err < 1.0e-9, "velocity round-trip error = {} m/s", vel_err);
    }

    /// Geostationary sanity check: a satellite that is stationary in ITRF
    /// (parked over a fixed longitude on the equator) should have a GCRF
    /// velocity equal to the Earth-rotation sweep speed ω⊕·r. For a GEO
    /// orbit at ~42,164 km this is ~3074.7 m/s, which also happens to
    /// equal the GEO orbital speed (that's the whole point of GEO).
    #[test]
    fn test_itrf_gcrf_state_geo() {
        let t = crate::Instant::from_datetime(2024, 6, 1, 0, 0, 0.0).unwrap();
        let geo_r: f64 = 42_164.17e3; // m
        // Parked on equator at 0° longitude in ITRF, stationary
        let pos_itrf: Vector3 = numeris::vector![geo_r, 0.0, 0.0];
        let vel_itrf: Vector3 = numeris::vector![0.0, 0.0, 0.0];

        let (pos_gcrf, vel_gcrf) = itrf_to_gcrf_state(&pos_itrf, &vel_itrf, &t);

        // Position magnitude preserved
        assert!((pos_gcrf.norm() - geo_r).abs() < 1.0e-6);

        // Velocity magnitude = ω⊕ · r
        let v_expected = crate::consts::OMEGA_EARTH * geo_r;
        let v_mag = vel_gcrf.norm();
        assert!(
            (v_mag - v_expected).abs() < 1.0e-3,
            "GEO sweep speed {} m/s, expected {} m/s",
            v_mag,
            v_expected
        );

        // Velocity is perpendicular to position (pure tangential circular motion)
        let dot = pos_gcrf.dot(&vel_gcrf);
        assert!(dot.abs() / (pos_gcrf.norm() * vel_gcrf.norm()) < 1.0e-10);
    }

    /// Naive-vs-correct comparison: for a LEO state, rotating velocity via
    /// the raw quaternion (ignoring the Coriolis term) differs from the
    /// correct state transform by ~|ω⊕ × r| ≈ 470 m/s. This is the
    /// deliberate-misuse test that demonstrates why `itrf_to_gcrf_state`
    /// exists.
    #[test]
    fn test_naive_itrf_rotation_is_wrong_by_470mps() {
        let t = crate::Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
        // LEO position in ITRF, at rest (on Earth surface near equator)
        let earth_r = crate::consts::EARTH_RADIUS + 500.0e3;
        let pos_itrf: Vector3 = numeris::vector![earth_r, 0.0, 0.0];
        let vel_itrf_zero: Vector3 = numeris::vector![0.0, 0.0, 0.0];

        // Correct state transform
        let (_pos_gcrf, vel_gcrf_correct) =
            itrf_to_gcrf_state(&pos_itrf, &vel_itrf_zero, &t);

        // Naive transform: just rotate the (zero) velocity with the quaternion
        let q = qitrf2gcrf_approx(&t);
        let vel_gcrf_naive = q * vel_itrf_zero;
        assert!(vel_gcrf_naive.norm() < 1.0e-12, "naive zero stays zero");

        // Difference between correct and naive:
        // |v_correct - v_naive| ≈ |ω⊕ × r| = ω⊕ · r_equatorial
        let expected_sweep = crate::consts::OMEGA_EARTH * earth_r;
        let diff = (vel_gcrf_correct - vel_gcrf_naive).norm();
        let rel_err = (diff - expected_sweep).abs() / expected_sweep;
        assert!(
            rel_err < 1.0e-6,
            "naive-vs-correct velocity gap = {} m/s, expected {} m/s",
            diff,
            expected_sweep
        );
        // Sanity: for a 500 km equatorial state, this should be ~501 m/s
        assert!(
            (diff - 501.4).abs() < 1.0,
            "expected ~501 m/s Earth-rotation sweep at 500km equatorial, got {}",
            diff
        );
    }

    /// Round-trip for the approximate state transforms: GCRF → ITRF →
    /// GCRF should recover the original state to floating-point precision
    /// (the approximation error is in the absolute accuracy vs. IERS 2010,
    /// not in self-consistency).
    #[test]
    fn test_itrf_gcrf_state_approx_roundtrip() {
        let t = crate::Instant::from_datetime(2024, 3, 15, 12, 34, 56.0).unwrap();
        let pos_gcrf: Vector3 = numeris::vector![6.878e6, 1.23e5, -4.56e5];
        let vel_gcrf: Vector3 = numeris::vector![-123.4, 7600.0, 89.0];

        let (pos_itrf, vel_itrf) = gcrf_to_itrf_state_approx(&pos_gcrf, &vel_gcrf, &t);
        let (pos_back, vel_back) = itrf_to_gcrf_state_approx(&pos_itrf, &vel_itrf, &t);

        let pos_err = (pos_gcrf - pos_back).norm();
        let vel_err = (vel_gcrf - vel_back).norm();
        assert!(pos_err < 1.0e-6, "position round-trip error = {} m", pos_err);
        assert!(vel_err < 1.0e-9, "velocity round-trip error = {} m/s", vel_err);
    }

    /// Approximate transform should agree with the full IERS 2010 reduction
    /// to about an arcsecond on position (~30 m at LEO) and a small
    /// fraction of a m/s on velocity. This is the advertised accuracy of
    /// the IAU-76/FK5 approximation.
    #[test]
    fn test_itrf_gcrf_state_approx_vs_full() {
        let t = crate::Instant::from_datetime(2024, 3, 15, 12, 34, 56.0).unwrap();
        let pos_itrf: Vector3 = numeris::vector![-1.0334e6, 7.9013e6, 6.3804e6];
        let vel_itrf: Vector3 = numeris::vector![-3225.6, -2872.5, 5531.9];

        let (p_full, v_full) = itrf_to_gcrf_state(&pos_itrf, &vel_itrf, &t);
        let (p_approx, v_approx) = itrf_to_gcrf_state_approx(&pos_itrf, &vel_itrf, &t);

        let pos_diff = (p_full - p_approx).norm();
        let vel_diff = (v_full - v_approx).norm();

        // 1 arcsec at ~1 Earth radius ≈ 50 m; allow 100 m of headroom.
        assert!(pos_diff < 100.0, "approx vs full position diff = {} m", pos_diff);
        // Earth rotation takes ~470 m/s / Earth-radius; at 1 arcsec this
        // scales to well under 1 m/s.
        assert!(vel_diff < 1.0, "approx vs full velocity diff = {} m/s", vel_diff);
    }
}
