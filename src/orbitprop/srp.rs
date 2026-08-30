//! Empirical CODE Orbit Model (ECOM) solar radiation pressure.
//!
//! ECOM is the empirical SRP parameterization used by CODE and most IGS
//! analysis centers for GNSS precise orbit determination. It expresses the
//! non-gravitational acceleration in a Sun-oriented satellite frame (D, Y, B)
//! with constant and harmonic terms whose coefficients are *estimated* in
//! orbit determination rather than derived from a physical surface model.
//! satkit does not estimate them; it propagates with coefficients you
//! supply (from CODE products, your own fit, or a box-wing residual).
//!
//! Attach coefficients with [`SatPropertiesSimple::with_ecom`](crate::orbitprop::SatPropertiesSimple::with_ecom)
//! (constant over a propagation) or by implementing
//! [`SatProperties::srp_ecom`](crate::orbitprop::SatProperties::srp_ecom)
//! (time-varying). [`propagate`](crate::orbitprop::propagate) adds the ECOM
//! acceleration to the cannonball term `−ν·P☉·C_R A/m·ê_D`, so set
//! `craoverm = 0` for a pure ECOM model. The model contributes no partials
//! to the state transition matrix (like the cannonball). For a worked fit
//! against IGS GPS orbits see the *ECOM Solar Radiation Pressure* tutorial
//! in the documentation.
//!
//! # Frame (DYB)
//!
//! With `r` the satellite position, `s` the Sun position (both GCRF):
//!
//! * `ê_D = unit(s − r)` — satellite → Sun.
//! * `ê_Y = unit(ê_D × r̂)` — the solar-panel rotation axis for a
//!   nominally yaw-steering satellite.
//! * `ê_B = ê_D × ê_Y` — completes the right-handed triad.
//!
//! Sign conventions differ between publications; satkit fixes them as above.
//! Because `ê_D` points **at** the Sun, the physical radiation-pressure
//! acceleration has a **negative** `D0` — about −1e-7 m/s² for a GPS
//! satellite (C_R A/m ≈ 0.02 m²/kg), and 10–30 nm/s² when ECOM is applied as
//! a residual over an a-priori model. `Y0` and the B terms are typically
//! ~1e-9 m/s².
//!
//! # Model
//!
//! ```text
//! a = ν · [ D(φ)·ê_D + Y(φ)·ê_Y + B(φ)·ê_B ]
//!
//! D(φ) = D0 + Dc cos φ + Ds sin φ + D2c cos 2φ + D2s sin 2φ + D4c cos 4φ + D4s sin 4φ
//! Y(φ) = Y0 + Yc cos φ + Ys sin φ
//! B(φ) = B0 + Bc cos φ + Bs sin φ
//! ```
//!
//! where `ν ∈ [0, 1]` is the Earth-shadow factor (the same conical
//! umbra/penumbra function as the cannonball term). All three axes are
//! scaled by `ν`, matching CODE/Bernese, where "the acceleration due to the
//! solar radiation pressure is switched off when the satellite is in the
//! Earth's shadow" (Bernese GNSS Software v5.2 §2.2.2.3) — so coefficients
//! taken from CODE products keep their meaning. (Orekit's `ECOM2` applies
//! no shadow factor at all; a Y-bias persisting through eclipse, as proposed
//! by Sidorov et al. 2020, is not modelled.)
//!
//! # Coefficients
//!
//! | field | axis | term | model | typical (GPS, nm/s²) |
//! |---|---|---|---|---|
//! | `d0` | ê_D (toward Sun) | constant | all | −80 … −110 (≈ −P☉·C_R·A/m; negative) |
//! | `y0` | ê_Y (solar-panel axis) | constant | all | ~1 (attitude/thermal Y-bias) |
//! | `b0` | ê_B | constant | all | ~1–5, varies with β |
//! | `dc`, `ds` | ê_D | cos φ, sin φ | ECOM1 (φ = u) | ≲ 1 |
//! | `yc`, `ys` | ê_Y | cos φ, sin φ | ECOM1 | ≲ 1 |
//! | `bc`, `bs` | ê_B | cos φ, sin φ | reduced/ECOM1 (φ = u); ECOM2's B1c/B1s (φ = Δu) | ≲ 2 |
//! | `d2c`, `d2s` | ê_D | cos 2Δu, sin 2Δu | ECOM2 | few (eclipse seasons) |
//! | `d4c`, `d4s` | ê_D | cos 4Δu, sin 4Δu | ECOM2 | few (eclipse seasons) |
//!
//! All values are accelerations in m/s². Zero coefficients cost nothing, so
//! a 7-parameter ECOM2 is `ecom2(.., d4c: 0, d4s: 0)`.
//!
//! # Stability
//!
//! **Experimental.** This interface ([`EcomParams`], [`ecom_accel`],
//! `SatProperties::srp_ecom`) is new and may be reshaped in a minor release
//! (e.g. into a general empirical-acceleration hook); the physics and
//! conventions are stable.
//!
//! The angular argument `φ` depends on [`EcomParams::sun_relative`]:
//!
//! * `false` (ECOM1, Beutler et al. 1994; Springer et al. 1999): `φ = u`,
//!   the argument of latitude measured from the ascending node. For an
//!   equatorial orbit the node is undefined and the reference direction
//!   falls back to the projection of the GCRF x-axis into the orbit plane.
//! * `true` (ECOM2, Arnold et al. 2015): `φ = Δu = u − u_☉`, measured from
//!   *orbit noon* — the point of the orbit closest to the Sun's projection
//!   into the orbit plane. `Δu = 0` at noon, `π` at midnight. This is
//!   computed node-free and is regular at all inclinations.
//!
//! Named parameter sets:
//!
//! * **Reduced ECOM1** ([`EcomParams::reduced`]): `D0, Y0, B0, Bc, Bs` —
//!   CODE's long-standing operational GPS set.
//! * **ECOM1** ([`EcomParams::ecom1`]): the full 9-parameter once-per-rev set.
//! * **ECOM2** ([`EcomParams::ecom2`]): `D0, Y0, B0, B1c, B1s, D2c, D2s,
//!   D4c, D4s` in `Δu` — even D harmonics, odd B harmonics.
//!
//! # References
//!
//! * Beutler, G. et al. (1994), "Extended orbit modeling techniques at the
//!   CODE processing center of the IGS", Manuscripta Geodaetica 19, 367–386.
//! * Springer, T. A., Beutler, G., Rothacher, M. (1999), "A new solar
//!   radiation pressure model for GPS satellites", GPS Solutions 2(3), 50–62.
//! * Arnold, D. et al. (2015), "CODE's new solar radiation pressure model
//!   for GNSS orbit determination", J. Geodesy 89, 775–791.

use crate::mathtypes::Vector3;
use serde::{Deserialize, Serialize};

/// ECOM solar-radiation-pressure coefficients, all in m/s².
///
/// See the [module docs](self) for the frame, sign, and eclipse conventions.
/// A default (all-zero) instance contributes no acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EcomParams {
    /// Constant D (Sun-direction) term. Physically negative.
    pub d0: f64,
    /// Constant Y term (along the solar-panel axis).
    pub y0: f64,
    /// Constant B term.
    pub b0: f64,
    /// D cos φ.
    pub dc: f64,
    /// D sin φ.
    pub ds: f64,
    /// Y cos φ.
    pub yc: f64,
    /// Y sin φ.
    pub ys: f64,
    /// B cos φ.
    pub bc: f64,
    /// B sin φ.
    pub bs: f64,
    /// D cos 2φ (ECOM2).
    pub d2c: f64,
    /// D sin 2φ (ECOM2).
    pub d2s: f64,
    /// D cos 4φ (ECOM2).
    pub d4c: f64,
    /// D sin 4φ (ECOM2).
    pub d4s: f64,
    /// `true`: harmonics in `Δu` from orbit noon (ECOM2 convention);
    /// `false`: harmonics in the argument of latitude `u` (ECOM1).
    pub sun_relative: bool,
}

impl EcomParams {
    /// Reduced ECOM1: `D0, Y0, B0, Bc, Bs` (m/s²), harmonics in the argument
    /// of latitude (`sun_relative = false`). All other coefficients are zero.
    /// This is CODE's long-standing operational GPS parameter set.
    ///
    /// # Examples
    ///
    /// ```
    /// use satkit::orbitprop::{EcomParams, SatPropertiesSimple};
    ///
    /// // Coefficients from a fit to IGS orbits (nm/s² -> m/s²); D0 is
    /// // negative because ê_D points at the Sun.
    /// let ecom = EcomParams::reduced(-105.8e-9, 1.03e-9, -3.18e-9, 1.18e-9, 0.34e-9);
    /// assert!(!ecom.sun_relative);
    /// assert_eq!(ecom.dc, 0.0);
    ///
    /// // Pure ECOM: no cannonball term (craoverm = 0), then pass `&props`
    /// // as `satprops` to `satkit::orbitprop::propagate`.
    /// let props = SatPropertiesSimple::new(0.0, 0.0).with_ecom(ecom);
    /// assert_eq!(props.ecom, Some(ecom));
    /// ```
    pub const fn reduced(d0: f64, y0: f64, b0: f64, bc: f64, bs: f64) -> Self {
        Self {
            d0,
            y0,
            b0,
            bc,
            bs,
            dc: 0.0,
            ds: 0.0,
            yc: 0.0,
            ys: 0.0,
            d2c: 0.0,
            d2s: 0.0,
            d4c: 0.0,
            d4s: 0.0,
            sun_relative: false,
        }
    }

    /// Full 9-parameter ECOM1: `D0, Y0, B0, Dc, Ds, Yc, Ys, Bc, Bs` (m/s²),
    /// once-per-revolution harmonics in the argument of latitude
    /// (`sun_relative = false`).
    #[allow(clippy::too_many_arguments)]
    pub const fn ecom1(
        d0: f64,
        y0: f64,
        b0: f64,
        dc: f64,
        ds: f64,
        yc: f64,
        ys: f64,
        bc: f64,
        bs: f64,
    ) -> Self {
        Self {
            d0,
            y0,
            b0,
            dc,
            ds,
            yc,
            ys,
            bc,
            bs,
            d2c: 0.0,
            d2s: 0.0,
            d4c: 0.0,
            d4s: 0.0,
            sun_relative: false,
        }
    }

    /// ECOM2 (Arnold et al. 2015): `D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s`
    /// (m/s²), harmonics in `Δu` from orbit noon (`sun_relative = true`).
    /// Even harmonics on D, odd on B; `B1c, B1s` map to the `bc, bs` fields.
    /// For the 7-parameter variant (nD = 1) pass `d4c = d4s = 0`.
    #[allow(clippy::too_many_arguments)]
    pub const fn ecom2(
        d0: f64,
        y0: f64,
        b0: f64,
        b1c: f64,
        b1s: f64,
        d2c: f64,
        d2s: f64,
        d4c: f64,
        d4s: f64,
    ) -> Self {
        Self {
            d0,
            y0,
            b0,
            bc: b1c,
            bs: b1s,
            dc: 0.0,
            ds: 0.0,
            yc: 0.0,
            ys: 0.0,
            d2c,
            d2s,
            d4c,
            d4s,
            sun_relative: true,
        }
    }

    /// `true` if every coefficient is zero (the model contributes nothing).
    pub fn is_zero(&self) -> bool {
        self.d0 == 0.0
            && self.y0 == 0.0
            && self.b0 == 0.0
            && self.dc == 0.0
            && self.ds == 0.0
            && self.yc == 0.0
            && self.ys == 0.0
            && self.bc == 0.0
            && self.bs == 0.0
            && self.d2c == 0.0
            && self.d2s == 0.0
            && self.d4c == 0.0
            && self.d4s == 0.0
    }

    /// `true` if any harmonic (non-constant) coefficient is nonzero, i.e.
    /// the orbit angle needs to be evaluated.
    fn has_harmonics(&self) -> bool {
        self.dc != 0.0
            || self.ds != 0.0
            || self.yc != 0.0
            || self.ys != 0.0
            || self.bc != 0.0
            || self.bs != 0.0
            || self.d2c != 0.0
            || self.d2s != 0.0
            || self.d4c != 0.0
            || self.d4s != 0.0
    }
}

/// The ECOM (D, Y, B) unit vectors in GCRF.
///
/// * `ê_D = unit(sun − pos)` (satellite → Sun)
/// * `ê_Y = unit(ê_D × r̂)`
/// * `ê_B = ê_D × ê_Y`
///
/// The frame is undefined when the Sun is exactly radial (Sun in the orbit
/// plane and the satellite exactly at orbit noon or midnight); there
/// `ê_D × r̂` vanishes and this function falls back to the limiting
/// direction, `ê_Y = ±along-track` (`ĥ × r̂`, with `ĥ` from `vel_gcrf`),
/// instead of producing NaN.
pub fn dyb_basis(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    sun_gcrf: &Vector3,
) -> (Vector3, Vector3, Vector3) {
    let e_d = (*sun_gcrf - *pos_gcrf).normalize();
    let r_hat = pos_gcrf.normalize();
    let c = e_d.cross(&r_hat);
    // |ê_D × r̂| = sin(angle between Sun and radial); below ~1e-6 rad the
    // direction is numerically meaningless, so use the along-track limit.
    let e_y = if c.norm_squared() < 1e-12 {
        let h_hat = pos_gcrf.cross(vel_gcrf).normalize();
        h_hat.cross(&r_hat).normalize()
    } else {
        c.normalize()
    };
    let e_b = e_d.cross(&e_y);
    (e_d, e_y, e_b)
}

/// Angle of the satellite around its orbit, measured in the orbit plane from
/// a reference direction, positive in the direction of motion.
///
/// * `sun_relative = true`: from orbit noon (the Sun's projection into the
///   orbit plane) — `Δu`, zero at noon and `π` at midnight.
/// * `sun_relative = false`: from the ascending node — the classical
///   argument of latitude `u`. For an equatorial orbit (|n| < 1e-9) the
///   reference falls back to the GCRF x-axis projected into the plane.
///
/// Computed without inclination-dependent singularities: the reference
/// vector is projected into the plane spanned by `r` and `v`, then
/// `atan2(ĥ·(ref × r̂), ref·r̂)`.
pub fn orbit_angle(
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    sun_gcrf: &Vector3,
    sun_relative: bool,
) -> f64 {
    let h_hat = pos_gcrf.cross(vel_gcrf).normalize();
    let reference: Vector3 = if sun_relative {
        *sun_gcrf
    } else {
        let z: Vector3 = numeris::vector![0.0, 0.0, 1.0];
        let node = z.cross(&h_hat);
        if node.norm() < 1e-9 {
            numeris::vector![1.0, 0.0, 0.0]
        } else {
            node
        }
    };
    let in_plane = reference - h_hat * reference.dot(&h_hat);
    let r_hat = pos_gcrf.normalize();
    h_hat
        .dot(&in_plane.cross(&r_hat))
        .atan2(in_plane.dot(&r_hat))
}

/// ECOM acceleration in GCRF (m/s²).
///
/// * `pos_gcrf`, `vel_gcrf` — satellite state in GCRF (m, m/s).
/// * `sun_gcrf` — geocentric Sun position in GCRF (m).
/// * `shadow` — the Earth-shadow factor `ν ∈ [0, 1]` (1 = full sunlight,
///   0 = umbra; see [`crate::lpephem::sun`]). It scales the D and B axes
///   and Y alike (CODE/Bernese convention). Pass `1.0` to ignore eclipses.
///
/// [`crate::orbitprop::propagate`] calls this from its force model with the
/// same shadow factor as the cannonball term whenever
/// [`SatProperties::srp_ecom`](crate::orbitprop::SatProperties::srp_ecom)
/// returns `Some`; you only need it directly for custom force evaluations.
pub fn ecom_accel(
    p: &EcomParams,
    pos_gcrf: &Vector3,
    vel_gcrf: &Vector3,
    sun_gcrf: &Vector3,
    shadow: f64,
) -> Vector3 {
    let (e_d, e_y, e_b) = dyb_basis(pos_gcrf, vel_gcrf, sun_gcrf);
    let (mut d, mut y, mut b) = (p.d0, p.y0, p.b0);
    if p.has_harmonics() {
        let phi = orbit_angle(pos_gcrf, vel_gcrf, sun_gcrf, p.sun_relative);
        let (s1, c1) = phi.sin_cos();
        // cos 2φ, sin 2φ, cos 4φ, sin 4φ via double-angle identities.
        let c2 = c1 * c1 - s1 * s1;
        let s2 = 2.0 * s1 * c1;
        let c4 = c2 * c2 - s2 * s2;
        let s4 = 2.0 * s2 * c2;
        d += p.dc * c1 + p.ds * s1 + p.d2c * c2 + p.d2s * s2 + p.d4c * c4 + p.d4s * s4;
        y += p.yc * c1 + p.ys * s1;
        b += p.bc * c1 + p.bs * s1;
    }
    (e_d * d + e_y * y + e_b * b) * shadow
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const AU: f64 = 1.495978707e11;

    fn sample_geometry() -> (Vector3, Vector3, Vector3) {
        // Inclined circular-ish orbit, Sun off-plane.
        let pos: Vector3 = numeris::vector![1.5e7, 2.0e7, 1.0e7];
        let vel: Vector3 = numeris::vector![-2.5e3, 1.5e3, 1.0e3];
        let sun: Vector3 = numeris::vector![0.6 * AU, 0.7 * AU, 0.3 * AU];
        (pos, vel, sun)
    }

    #[test]
    fn dyb_is_orthonormal_and_right_handed() {
        let (pos, vel, sun) = sample_geometry();
        let (d, y, b) = dyb_basis(&pos, &vel, &sun);
        for e in [&d, &y, &b] {
            assert!((e.norm() - 1.0).abs() < 1e-14);
        }
        assert!(d.dot(&y).abs() < 1e-14);
        assert!(d.dot(&b).abs() < 1e-14);
        assert!(y.dot(&b).abs() < 1e-14);
        // Right-handed: D × Y = B.
        assert!((d.cross(&y) - b).norm() < 1e-14);
        // D points at the Sun; Y ⟂ r.
        assert!(d.dot(&(sun - pos)) > 0.0);
        assert!(y.dot(&pos).abs() / pos.norm() < 1e-14);
    }

    #[test]
    fn d0_only_is_constant_sun_direction() {
        let (pos, vel, sun) = sample_geometry();
        let p = EcomParams {
            d0: -1e-7,
            ..Default::default()
        };
        let a = ecom_accel(&p, &pos, &vel, &sun, 1.0);
        let (e_d, _, _) = dyb_basis(&pos, &vel, &sun);
        assert!((a - e_d * -1e-7).norm() < 1e-22);
        // Points away from the Sun.
        assert!(a.dot(&(sun - pos)) < 0.0);
    }

    /// CODE/Bernese convention: the whole ECOM acceleration is switched off
    /// in shadow, Y included; penumbra scales all three axes alike.
    #[test]
    fn all_axes_scaled_by_shadow() {
        let (pos, vel, sun) = sample_geometry();
        let p = EcomParams::reduced(-1e-7, 2e-9, 3e-9, 0.0, 0.0);
        let lit = ecom_accel(&p, &pos, &vel, &sun, 1.0);
        let half = ecom_accel(&p, &pos, &vel, &sun, 0.5);
        let dark = ecom_accel(&p, &pos, &vel, &sun, 0.0);
        assert!(dark.norm() < 1e-30);
        assert!((half - lit * 0.5).norm() < 1e-22);
    }

    /// Sun exactly radial (in the orbit plane, satellite at noon): ê_Y must be
    /// finite and along-track, not NaN.
    #[test]
    fn dyb_basis_sun_exactly_radial() {
        let sun: Vector3 = numeris::vector![AU, 0.0, 0.0];
        let pos: Vector3 = numeris::vector![2.66e7, 0.0, 0.0];
        let vel: Vector3 = numeris::vector![0.0, 3.9e3, 0.0];
        let (e_d, e_y, e_b) = dyb_basis(&pos, &vel, &sun);
        assert!(e_y.as_slice().iter().all(|v| v.is_finite()));
        // along-track limit: ĥ × r̂ = ẑ × x̂ = ŷ
        assert!((e_y - numeris::vector![0.0, 1.0, 0.0]).norm() < 1e-12);
        assert!((e_d.dot(&e_y)).abs() < 1e-12 && (e_b.dot(&e_y)).abs() < 1e-12);
    }

    /// Δu = 0 when r̂ lies along the Sun's in-plane projection, π opposite.
    #[test]
    fn delta_u_noon_and_midnight() {
        let sun: Vector3 = numeris::vector![AU, 0.0, 0.0];
        // Orbit in the x-z plane (polar-ish), moving +z at noon.
        let noon: Vector3 = numeris::vector![2.66e7, 0.0, 0.0];
        let vel_noon: Vector3 = numeris::vector![0.0, 0.0, 3.9e3];
        assert!(orbit_angle(&noon, &vel_noon, &sun, true).abs() < 1e-12);
        let midnight: Vector3 = numeris::vector![-2.66e7, 0.0, 0.0];
        let vel_mid: Vector3 = numeris::vector![0.0, 0.0, -3.9e3];
        assert!((orbit_angle(&midnight, &vel_mid, &sun, true).abs() - PI).abs() < 1e-12);
        // Quarter orbit after noon: +π/2 (positive in direction of motion).
        let q: Vector3 = numeris::vector![0.0, 0.0, 2.66e7];
        let vel_q: Vector3 = numeris::vector![-3.9e3, 0.0, 0.0];
        assert!((orbit_angle(&q, &vel_q, &sun, true) - PI / 2.0).abs() < 1e-12);
    }

    /// `sun_relative = false` reproduces the classical argument of latitude.
    #[test]
    fn argument_of_latitude_matches_kepler() {
        use crate::kepler::{Anomaly, Kepler};
        let inc = 55.0_f64.to_radians();
        let raan = 200.0_f64.to_radians();
        for (argp_deg, ta_deg) in [
            (30.0_f64, 100.0_f64),
            (250.0, 0.0),
            (0.0, 359.0),
            (170.0, 200.0),
        ] {
            let argp = argp_deg.to_radians();
            let ta = ta_deg.to_radians();
            let k = Kepler::new(2.656e7, 0.01, inc, raan, argp, Anomaly::True(ta));
            let (r, v) = k.to_pv();
            let sun: Vector3 = numeris::vector![AU, 0.0, 0.0];
            let u = orbit_angle(&r, &v, &sun, false);
            let expected = (argp + ta).rem_euclid(2.0 * PI);
            let got = u.rem_euclid(2.0 * PI);
            let diff = (got - expected + PI).rem_euclid(2.0 * PI) - PI;
            assert!(
                diff.abs() < 1e-9,
                "argp {argp_deg} ta {ta_deg}: got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn harmonics_are_periodic() {
        // Sample the B term around a full orbit built from Kepler; B(φ+2π)=B(φ)
        // is implied by the trig, so instead check antisymmetry: rotating the
        // satellite 180° about ĥ flips the sign of the cos/sin terms.
        let (pos, vel, sun) = sample_geometry();
        let h = pos.cross(&vel).normalize();
        let pos2 = -pos + h * (2.0 * pos.dot(&h));
        let vel2 = -vel + h * (2.0 * vel.dot(&h));
        let phi1 = orbit_angle(&pos, &vel, &sun, true);
        let phi2 = orbit_angle(&pos2, &vel2, &sun, true);
        let d = (phi2 - phi1 - PI + PI).rem_euclid(2.0 * PI) - PI;
        assert!(d.abs() < 1e-9, "phi1 {phi1} phi2 {phi2}");
    }

    /// A D0-only ECOM with `d0 = −P☉·C_R A/m` and `craoverm = 0` must
    /// reproduce the cannonball model: both are `−ν·P☉·C_R A/m` along the
    /// satellite→Sun line.
    #[test]
    fn d0_only_reproduces_cannonball() {
        use crate::orbitprop::{propagate, PropSettings, SatPropertiesSimple};
        use crate::{Duration, Instant};
        let cr_a_over_m = 0.02;
        let t0 = Instant::from_datetime(2024, 1, 15, 0, 0, 0.0).unwrap();
        let t1 = t0 + Duration::from_days(1.0);
        let settings = PropSettings {
            abs_error: 1e-12,
            rel_error: 1e-12,
            use_spaceweather: false,
            ..PropSettings::default()
        };
        // GPS-like state.
        let state: crate::orbitprop::SimpleState =
            numeris::vector![1.5e7, 2.0e7, 1.0e7, -2.5e3, 1.5e3, 1.0e3];
        let cannon = SatPropertiesSimple::new(0.0, cr_a_over_m);
        let ecom = SatPropertiesSimple::new(0.0, 0.0).with_ecom(EcomParams {
            d0: -4.56e-6 * cr_a_over_m,
            ..Default::default()
        });
        let a = propagate(&state, &t0, &t1, &settings, Some(&cannon)).unwrap();
        let b = propagate(&state, &t0, &t1, &settings, Some(&ecom)).unwrap();
        let dr = (a.state_end.block::<3, 1>(0, 0) - b.state_end.block::<3, 1>(0, 0)).norm();
        assert!(dr < 1e-3, "cannonball vs D0-only ECOM differ by {dr} m");
        // And the SRP actually did something (vs. no SRP at all).
        let none = SatPropertiesSimple::new(0.0, 0.0);
        let c = propagate(&state, &t0, &t1, &settings, Some(&none)).unwrap();
        let dr0 = (a.state_end.block::<3, 1>(0, 0) - c.state_end.block::<3, 1>(0, 0)).norm();
        assert!(dr0 > 10.0, "SRP had no effect: {dr0} m");
    }

    #[test]
    fn is_zero_and_constructors() {
        assert!(EcomParams::default().is_zero());
        let r = EcomParams::reduced(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9);
        assert!(!r.is_zero());
        assert!(!r.sun_relative);
        assert_eq!(r.bc, 3e-9);
        let e2 = EcomParams::ecom2(-1e-7, 0.0, 0.0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9);
        assert!(e2.sun_relative);
        assert_eq!((e2.bc, e2.bs, e2.d2c, e2.d4s), (1e-9, 2e-9, 3e-9, 6e-9));
        let e1 = EcomParams::ecom1(-1e-7, 0.0, 0.0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9);
        assert!(!e1.sun_relative);
        assert_eq!((e1.dc, e1.ys, e1.bs), (1e-9, 4e-9, 6e-9));
    }
}
