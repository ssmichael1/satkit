//! IERS 2010 §6.2 solid Earth tides.
//!
//! The solid body of the Earth deforms under lunar and solar gravitational
//! attraction. That deformation perturbs the Earth gravity field, producing
//! time-varying corrections to the Stokes coefficients C̄ₙₘ, S̄ₙₘ.
//!
//! Reference: IERS Conventions 2010 (IERS Technical Note 36), Chapter 6.
//!
//! Implementation status:
//! * [`TideModel::SolidStep1`] — frequency-independent Love-number response
//!   (§6.2.1, Eq. 6.6 and 6.7). Implemented.
//! * [`TideModel::SolidFull`] — Step 1 + frequency-dependent corrections
//!   from 71 tidal constituents (§6.2.2, Tables 6.5a/b/c). Currently
//!   falls back to Step 1 behavior. See issue #16.

use crate::consts;
use crate::mathtypes::*;

use serde::{Deserialize, Serialize};

/// Solid Earth tide model fidelity selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TideModel {
    /// No solid Earth tide correction.
    None,
    /// IERS 2010 §6.2.1 Step 1 — frequency-independent Love-number
    /// response. ≈99% of the total solid-tide signal at ~5% per-ydot
    /// overhead. This is the default.
    #[default]
    SolidStep1,
    /// Step 1 + IERS 2010 §6.2.2 Step 2 — frequency-dependent corrections
    /// from 71 tidal constituents (Tables 6.5a/b/c). Currently behaves
    /// as `SolidStep1`; Step 2 deferred.
    SolidFull,
}

/// Solid-tide perturbations to the fully-normalized Stokes coefficients.
///
/// Holds ΔC̄ₙₘ, ΔS̄ₙₘ for n ∈ {2,3,4}, m ∈ 0..=n. All other indices are zero.
/// The arrays are oversized (5×5) to allow plain `[n][m]` indexing.
#[derive(Debug, Clone, Copy, Default)]
pub struct TideDeltas {
    /// ΔC̄ₙₘ, indexed as `dc[n][m]`.
    pub dc: [[f64; 5]; 5],
    /// ΔS̄ₙₘ, indexed as `ds[n][m]`.
    pub ds: [[f64; 5]; 5],
}

// ---------------------------------------------------------------------------
// IERS Conventions 2010, Table 6.3 — nominal Love numbers (anelastic Earth).
// Index convention: K_RE[n][m], K_IM[n][m].
// Entries outside the (n,m) range listed in Table 6.3 are zero.
// ---------------------------------------------------------------------------
const K_RE: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.30190, 0.29830, 0.30102, 0.0],
    [0.09300, 0.09300, 0.09300, 0.09400],
];
const K_IM: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, -0.00144, -0.00130, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];
/// k_2m^(+) — degree-4 response of the field to the degree-2 tide
/// (IERS 2010 Eq. 6.7), indexed by m ∈ {0,1,2}.
const K_PLUS: [f64; 3] = [-0.00089, -0.00080, -0.00057];

// ---------------------------------------------------------------------------
// Fully-normalized associated-Legendre normalization factors
// N_nm = sqrt((2 - δ_0m)(2n+1)(n-m)!/(n+m)!). Constants chosen to allow
// the recursion-free closed-form evaluation of P̄_nm used here.
// ---------------------------------------------------------------------------
const N20: f64 = 2.2360679774997896; // sqrt(5)
const N21: f64 = 1.2909944487358056; // sqrt(5/3)
const N22: f64 = 0.6454972243679028; // sqrt(5/12)
const N30: f64 = 2.6457513110645907; // sqrt(7)
const N31: f64 = 1.0801234497346434; // sqrt(7/6)
const N32: f64 = 0.3415650255319866; // sqrt(7/60)
const N33: f64 = 0.13943375672974065; // sqrt(7/360)
const N40: f64 = 3.0; // sqrt(9)
const N41: f64 = 0.9486832980505138; // sqrt(9/10)
const N42: f64 = 0.22360679774997896; // sqrt(1/20)

/// Compute solid Earth tide ΔC̄ₙₘ, ΔS̄ₙₘ at the given epoch.
///
/// Implements IERS 2010 §6.2.1 Eq. 6.6 (n=2,3) and Eq. 6.7 (n=4 from
/// k_2m^(+)). All inputs in SI; the body positions must be in the
/// Earth-fixed (ITRF) frame because the formulas are written in
/// body-fixed spherical coordinates.
pub fn solid_tide_deltas(
    sun_itrf: &Vector3,
    moon_itrf: &Vector3,
    _time: &crate::Instant,
    model: TideModel,
) -> TideDeltas {
    let mut deltas = TideDeltas::default();
    if model == TideModel::None {
        return deltas;
    }
    // Step 2 (SolidFull) currently falls through to Step 1 only.

    let r_e = consts::EARTH_RADIUS;
    let mu_e = consts::MU_EARTH;

    for (body_pos, body_mu) in [(sun_itrf, consts::MU_SUN), (moon_itrf, consts::MU_MOON)] {
        accumulate_step1(body_pos, body_mu, mu_e, r_e, &mut deltas);
    }

    deltas
}

/// Per-body contribution to ΔC̄, ΔS̄ via IERS 2010 Eq. 6.6 and 6.7.
fn accumulate_step1(
    body_pos: &Vector3,
    body_mu: f64,
    mu_e: f64,
    r_e: f64,
    deltas: &mut TideDeltas,
) {
    let r_body = body_pos.norm();
    let mu_ratio = body_mu / mu_e;
    let sin_phi = body_pos[2] / r_body;
    let cos_phi = (1.0 - sin_phi * sin_phi).max(0.0).sqrt();
    let u = sin_phi;
    let u2 = u * u;
    let c2 = cos_phi * cos_phi;
    let c3 = c2 * cos_phi;

    // cos(mλ), sin(mλ) via the Chebyshev two-term recurrence — avoids
    // an atan2 and is robust when the body is nearly over the pole.
    let rxy = (body_pos[0] * body_pos[0] + body_pos[1] * body_pos[1]).sqrt();
    let (cos_l, sin_l) = if rxy > 0.0 {
        (body_pos[0] / rxy, body_pos[1] / rxy)
    } else {
        // Over the pole: λ undefined, but cos²φ = 0 zeros all m≥1 terms.
        (1.0, 0.0)
    };
    let mut cml = [0.0_f64; 4];
    let mut sml = [0.0_f64; 4];
    cml[0] = 1.0;
    cml[1] = cos_l;
    sml[1] = sin_l;
    cml[2] = 2.0 * cos_l * cml[1] - cml[0];
    sml[2] = 2.0 * cos_l * sml[1] - sml[0];
    cml[3] = 2.0 * cos_l * cml[2] - cml[1];
    sml[3] = 2.0 * cos_l * sml[2] - sml[1];

    // Fully-normalized P̄_nm(sin φ), n=2,3, closed-form.
    let p_bar: [[f64; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [
            N20 * 0.5 * (3.0 * u2 - 1.0),
            N21 * 3.0 * u * cos_phi,
            N22 * 3.0 * c2,
            0.0,
        ],
        [
            N30 * 0.5 * (5.0 * u2 * u - 3.0 * u),
            N31 * 1.5 * (5.0 * u2 - 1.0) * cos_phi,
            N32 * 15.0 * u * c2,
            N33 * 15.0 * c3,
        ],
    ];

    // Eq. 6.6: ΔC̄ - i ΔS̄ = (k_nm / (2n+1)) · (μ_j/μ_e) · (R_e/r_j)^{n+1} · P̄_nm · e^{-imλ}
    // Expand the complex Love number k = k_re + i k_im and exponential:
    //   ΔC̄ = a (k_re cos mλ + k_im sin mλ)
    //   ΔS̄ = a (k_re sin mλ − k_im cos mλ)
    // where a = (μ_j/μ_e)(R_e/r_j)^{n+1} P̄_nm / (2n+1).
    let re_over_r = r_e / r_body;
    let mut re_over_r_np1 = re_over_r * re_over_r * re_over_r; // (R/r)^3 for n=2
    for n in 2..=3 {
        let inv_2np1 = 1.0 / (2 * n + 1) as f64;
        for m in 0..=n {
            let a = mu_ratio * re_over_r_np1 * p_bar[n][m] * inv_2np1;
            let k_re = K_RE[n][m];
            let k_im = K_IM[n][m];
            deltas.dc[n][m] += a * (k_re * cml[m] + k_im * sml[m]);
            deltas.ds[n][m] += a * (k_re * sml[m] - k_im * cml[m]);
        }
        re_over_r_np1 *= re_over_r;
    }

    // Eq. 6.7: degree-4 corrections driven by the degree-2 tide. Uses
    // k_2m^(+), (R/r)^3, divisor 5, P̄_2m. k_2m^(+) is real, so no S-term
    // crossover from the imaginary part.
    let re_over_r_3 = (r_e / r_body).powi(3);
    let factor4 = mu_ratio * re_over_r_3 / 5.0;
    for m in 0..=2 {
        let a = factor4 * p_bar[2][m] * K_PLUS[m];
        deltas.dc[4][m] += a * cml[m];
        deltas.ds[4][m] += a * sml[m];
    }
}

/// Compute the ITRF acceleration on the satellite from a set of tide-induced
/// Stokes-coefficient corrections.
///
/// Implements Montenbruck & Gill Eq. 3.33 over the small (n ≤ 4) set of
/// nonzero coefficients in `deltas`. Returns the acceleration in m/s²
/// in the same frame as `pos_itrf`.
pub fn tide_accel(pos_itrf: &Vector3, deltas: &TideDeltas, mu_e: f64, r_e: f64) -> Vector3 {
    // Convert normalized ΔC̄, ΔS̄ to unnormalized for M&G's recursion.
    let n_fac: [[f64; 5]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [N20, N21, N22, 0.0, 0.0],
        [N30, N31, N32, N33, 0.0],
        [N40, N41, N42, 0.0, 0.0],
    ];
    let mut c = [[0.0_f64; 5]; 5];
    let mut s = [[0.0_f64; 5]; 5];
    for n in 2..=4 {
        let m_max = if n == 4 { 2 } else { n };
        for m in 0..=m_max {
            c[n][m] = n_fac[n][m] * deltas.dc[n][m];
            s[n][m] = n_fac[n][m] * deltas.ds[n][m];
        }
    }

    let (v, w) = cunningham_vw(pos_itrf, r_e);

    // M&G Eq. 3.33 summed over n=2,3,4.
    let mut ax = 0.0;
    let mut ay = 0.0;
    let mut az = 0.0;
    for n in 2..=4 {
        // m = 0 (zonal)
        let cn0 = c[n][0];
        ax += -cn0 * v[n + 1][1];
        ay += -cn0 * w[n + 1][1];
        az += -((n + 1) as f64) * cn0 * v[n + 1][0];

        // m ≥ 1
        let m_max = if n == 4 { 2 } else { n };
        for m in 1..=m_max {
            let cnm = c[n][m];
            let snm = s[n][m];
            let fac_a = ((n - m + 2) * (n - m + 1)) as f64;
            let fac_z = (n - m + 1) as f64;
            let vnp1mp1 = v[n + 1][m + 1];
            let wnp1mp1 = w[n + 1][m + 1];
            let vnp1mm1 = v[n + 1][m - 1];
            let wnp1mm1 = w[n + 1][m - 1];
            ax += 0.5 * (-cnm * vnp1mp1 - snm * wnp1mp1 + fac_a * (cnm * vnp1mm1 + snm * wnp1mm1));
            ay += 0.5 * (-cnm * wnp1mp1 + snm * vnp1mp1 + fac_a * (-cnm * wnp1mm1 + snm * vnp1mm1));
            az += fac_z * (-cnm * v[n + 1][m] - snm * w[n + 1][m]);
        }
    }

    let scale = mu_e / (r_e * r_e);
    numeris::vector![ax * scale, ay * scale, az * scale]
}

/// Unnormalized Cunningham V_nm, W_nm up to n=5, m=5.
///
/// Definition (M&G Eq. 3.29):
///   V_nm + i W_nm = (R_e/r)^{n+1} · P_nm(sinφ) · e^{imλ}
fn cunningham_vw(pos: &Vector3, r_e: f64) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
    let r2 = pos.norm_squared();
    let r = r2.sqrt();
    let inv_r2 = 1.0 / r2;
    let xf = pos[0] * r_e * inv_r2;
    let yf = pos[1] * r_e * inv_r2;
    let zf = pos[2] * r_e * inv_r2;
    let rf = r_e * r_e * inv_r2;

    let mut v = [[0.0_f64; 6]; 6];
    let mut w = [[0.0_f64; 6]; 6];
    v[0][0] = r_e / r;

    // Sectoral (n=m, m≥1): M&G Eq. 3.31.
    for m in 1..=5 {
        let mm1 = m - 1;
        let two = (2 * m - 1) as f64;
        v[m][m] = two * (xf * v[mm1][mm1] - yf * w[mm1][mm1]);
        w[m][m] = two * (xf * w[mm1][mm1] + yf * v[mm1][mm1]);
    }

    // Tesseral / zonal (n > m): M&G Eq. 3.30.
    // V_{m-1,m} and W_{m-1,m} are zero by convention.
    for m in 0..=5 {
        for n in (m + 1)..=5 {
            let nm = (n - m) as f64;
            let f1 = ((2 * n - 1) as f64) / nm;
            let (vnm2, wnm2) = if n >= m + 2 {
                (v[n - 2][m], w[n - 2][m])
            } else {
                (0.0, 0.0)
            };
            let f2 = if n >= m + 2 {
                ((n + m - 1) as f64) / nm
            } else {
                0.0
            };
            v[n][m] = f1 * zf * v[n - 1][m] - f2 * rf * vnm2;
            w[n][m] = f1 * zf * w[n - 1][m] - f2 * rf * wnm2;
        }
    }

    (v, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instant;

    fn epoch() -> Instant {
        Instant::from_datetime(2020, 1, 1, 0, 0, 0.0).unwrap()
    }

    #[test]
    fn zero_deltas_give_zero_accel() {
        let pos = numeris::vector![consts::GEO_R, 0.0, 0.0];
        let deltas = TideDeltas::default();
        let a = tide_accel(&pos, &deltas, consts::MU_EARTH, consts::EARTH_RADIUS);
        assert!(a.norm() < 1e-20, "tide_accel of zero deltas = {:?}", a);
    }

    #[test]
    fn none_model_yields_empty_deltas() {
        let sun = numeris::vector![1.49e11, 0.0, 0.0];
        let moon = numeris::vector![3.84e8, 0.0, 0.0];
        let deltas = solid_tide_deltas(&sun, &moon, &epoch(), TideModel::None);
        for n in 0..5 {
            for m in 0..5 {
                assert_eq!(deltas.dc[n][m], 0.0);
                assert_eq!(deltas.ds[n][m], 0.0);
            }
        }
    }

    #[test]
    fn body_over_pole_excites_only_zonal_terms() {
        // Sun and Moon both placed on +z. Sub-body latitude = 90°, so
        // cos φ = 0 zeros every P̄_nm for m ≥ 1. Only ΔC̄_n0 entries
        // (zonal) should be nonzero.
        let sun = numeris::vector![0.0, 0.0, 1.49e11];
        let moon = numeris::vector![0.0, 0.0, 3.84e8];
        let deltas = solid_tide_deltas(&sun, &moon, &epoch(), TideModel::SolidStep1);
        for n in 2..=4 {
            for m in 1..=n {
                assert!(
                    deltas.dc[n][m].abs() < 1e-25,
                    "ΔC̄[{}][{}] = {:e} (expected 0)",
                    n,
                    m,
                    deltas.dc[n][m]
                );
                assert!(
                    deltas.ds[n][m].abs() < 1e-25,
                    "ΔS̄[{}][{}] = {:e} (expected 0)",
                    n,
                    m,
                    deltas.ds[n][m]
                );
            }
        }
        // ΔC̄_20 should be clearly nonzero, and positive: P̄_20(1) = +√5,
        // and k20_re > 0 → contribution is positive.
        assert!(deltas.dc[2][0] > 0.0);
    }

    #[test]
    fn equator_x_body_gives_negative_dc20() {
        // Body on +x equator: sin φ = 0 → P̄_20 = -√5/2 < 0. With
        // k20_re > 0 and cos(0) = 1, ΔC̄_20 < 0. (This is the sign of
        // the classic "ocean bulge points toward the perturber" picture
        // expressed in the geopotential, modulo signs.)
        let moon = numeris::vector![3.84e8, 0.0, 0.0];
        let no_sun = numeris::vector![1e30, 0.0, 0.0]; // ~zero contribution
        let deltas = solid_tide_deltas(&no_sun, &moon, &epoch(), TideModel::SolidStep1);
        assert!(
            deltas.dc[2][0] < 0.0,
            "ΔC̄_20 from equatorial body = {:e}, expected < 0",
            deltas.dc[2][0]
        );
    }

    #[test]
    fn tide_accel_magnitude_at_geo_is_reasonable() {
        // Realistic Sun and Moon ITRF-frame positions (rough geometry,
        // both near the equator). Satellite at GEO. Expected solid-tide
        // perturbation: ~1e-9 to 1e-7 m/s².
        let sun = numeris::vector![1.0e11, 1.0e11, 0.0]; // ~150e6 km, equator
        let moon = numeris::vector![3.0e8, 2.0e8, 5.0e7]; // ~380,000 km, inclined
        let deltas = solid_tide_deltas(&sun, &moon, &epoch(), TideModel::SolidStep1);
        let pos = numeris::vector![consts::GEO_R, 0.0, 0.0];
        let a = tide_accel(&pos, &deltas, consts::MU_EARTH, consts::EARTH_RADIUS);
        let mag = a.norm();
        assert!(
            (1e-11..1e-6).contains(&mag),
            "tide accel at GEO = {:e}, expected ~1e-9 to 1e-7",
            mag
        );
    }

    #[test]
    fn cunningham_v00_is_re_over_r() {
        let pos = numeris::vector![1.0e7, 2.0e7, 3.0e7];
        let r = pos.norm();
        let (v, _) = cunningham_vw(&pos, consts::EARTH_RADIUS);
        let expected = consts::EARTH_RADIUS / r;
        assert!((v[0][0] - expected).abs() / expected < 1e-15);
    }

    #[test]
    fn solid_full_currently_matches_step1() {
        // Sanity: SolidFull is not yet implemented and must behave as
        // SolidStep1 until Step 2 is added.
        let sun = numeris::vector![1.4e11, 5e10, 2e10];
        let moon = numeris::vector![3.2e8, 1.5e8, 4e7];
        let d1 = solid_tide_deltas(&sun, &moon, &epoch(), TideModel::SolidStep1);
        let d2 = solid_tide_deltas(&sun, &moon, &epoch(), TideModel::SolidFull);
        for n in 2..=4 {
            for m in 0..=n.min(4) {
                assert_eq!(d1.dc[n][m], d2.dc[n][m]);
                assert_eq!(d1.ds[n][m], d2.ds[n][m]);
            }
        }
    }
}
