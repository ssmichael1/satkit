//! Frame-to-frame dispatch: take any two [`Frame`]s and return the
//! quaternion (or state transform) between them.
//!
//! Catches up to the convention that SPICE, Orekit, Astropy, and ANISE all
//! settled on long ago: a single function `rotation(from, to, t)` instead of
//! the per-pair named functions ([`qitrf2gcrf`], [`qteme2itrf`], …). The
//! per-pair functions remain canonical; this layer just dispatches into them.
//!
//! # Shortest-path dispatch
//!
//! Naive implementations pivot every transform through GCRF, paying the full
//! IERS 2010 precession/nutation reduction even for cheap ITRF↔TIRS pairs.
//! This module instead hand-codes the shortest path for each pair, so e.g.
//! `rotation(Frame::ITRF, Frame::TIRS, t)` only does polar motion, and
//! `rotation(Frame::ITRF, Frame::CIRS, t)` composes polar motion with the
//! Earth-rotation angle but skips the precession/nutation step.
//!
//! # Frame graph
//!
//! ```text
//! ICRF — GCRF — EME2000
//!         |
//!        CIRS
//!         |
//!        TIRS
//!         |
//!        ITRF — TEME
//! ```
//!
//! [`Frame::LVLH`], [`Frame::RTN`], and [`Frame::NTW`] are orbit-dependent
//! and not handled here — use [`to_gcrf`](super::to_gcrf) /
//! [`from_gcrf`](super::from_gcrf) for those.
//!
//! [`qitrf2gcrf`]: super::qitrf2gcrf
//! [`qteme2itrf`]: super::qteme2itrf

use std::f64::consts::PI;

use super::{
    gcrf_to_itrf_state, gcrf_to_itrf_state_approx, itrf_to_gcrf_state, itrf_to_gcrf_state_approx,
};
use super::{
    qcirs2gcrs, qitrf2gcrf, qitrf2gcrf_approx, qitrf2tirs, qteme2itrf, qtirs2cirs, Error, Result,
};
use crate::frames::Frame;
use crate::mathtypes::{Quaternion, Vector3};
use crate::TimeLike;

const ASEC2RAD: f64 = PI / 180.0 / 3600.0;

// ───── EME2000 frame bias ────────────────────────────────────────────────
//
// Constant rotation between EME2000 (J2000 mean dynamical equator + equinox)
// and GCRF (= GCRS). The three IERS 2010 canonical small Euler angles
// (Conventions 2010 §5.32):
//
//   dα0 = -0.014600 ± 0.000100 arcsec   RA offset of J2000 mean equinox
//   ξ0  = -0.016617 ± 0.000010 arcsec   obliquity-direction bias
//   η0  = -0.006819 ± 0.000010 arcsec   azimuth-direction bias
//
// The IERS bias matrix (eq. 5.36) is B = R1(-η0) · R2(ξ0) · R3(dα0), where
// R1/R2/R3 are *passive* (component-transformation) rotations. B transforms
// GCRS components to EME2000 components: v_EME2000 = B · v_GCRS.
//
// `numeris::Quaternion::rot{x,y,z}(θ)` is the *active* right-hand-rule
// rotation by +θ, which equals the passive R_i(−θ). So expressing IERS B
// in numeris terms requires negating each angle, and we further want the
// inverse B^T = R3(-dα0) · R2(-ξ0) · R1(η0) for EME2000 → GCRF:
const FRAME_BIAS_DALPHA0_AS: f64 = -0.014600;
const FRAME_BIAS_XI0_AS: f64 = -0.016617;
const FRAME_BIAS_ETA0_AS: f64 = -0.006819;

/// Constant quaternion: EME2000 → GCRF (≈ 17 milliarcsec frame bias).
///
/// Implements `B^T = R3(-dα0) · R2(-ξ0) · R1(η0)` in IERS notation. In
/// numeris' active-rotation convention this is `rotz(dα0) · roty(ξ0) ·
/// rotx(-η0)` (each axis-angle negated relative to the passive form).
fn qeme2000_to_gcrf() -> Quaternion {
    let dalpha0 = FRAME_BIAS_DALPHA0_AS * ASEC2RAD;
    let xi0 = FRAME_BIAS_XI0_AS * ASEC2RAD;
    let eta0 = FRAME_BIAS_ETA0_AS * ASEC2RAD;
    Quaternion::rotz(dalpha0) * Quaternion::roty(xi0) * Quaternion::rotx(-eta0)
}

// ───── Frame classification ──────────────────────────────────────────────

/// True for frames that rotate with Earth (state transforms to/from these
/// pick up an `ω⊕ × r` sweep term). Polar motion between ITRF and TIRS is
/// slow (~1.7e-9 rad/s) and treated as a static rotation.
fn is_earth_rotating(f: Frame) -> bool {
    match f {
        Frame::ITRF | Frame::TIRS => true,
        Frame::CIRS
        | Frame::GCRF
        | Frame::TEME
        | Frame::EME2000
        | Frame::ICRF
        | Frame::LVLH
        | Frame::RTN
        | Frame::NTW => false,
    }
}

/// True for frames whose axes are defined by an orbit's instantaneous
/// position and velocity — not handled by the time-only dispatch in this
/// module. Use [`to_gcrf`](super::to_gcrf) / [`from_gcrf`](super::from_gcrf).
fn is_orbit_dependent(f: Frame) -> bool {
    match f {
        Frame::LVLH | Frame::RTN | Frame::NTW => true,
        Frame::ITRF
        | Frame::TIRS
        | Frame::CIRS
        | Frame::GCRF
        | Frame::TEME
        | Frame::EME2000
        | Frame::ICRF => false,
    }
}

// ───── canonical ordering ────────────────────────────────────────────────

/// Position of each [`Frame`] in the canonical ordering used to normalise
/// the (from, to) pair so each unordered pair appears in the match once.
/// Adding a new variant forces this match to be updated.
fn frame_order(f: Frame) -> u8 {
    match f {
        Frame::ITRF => 0,
        Frame::TIRS => 1,
        Frame::CIRS => 2,
        Frame::GCRF => 3,
        Frame::TEME => 4,
        Frame::EME2000 => 5,
        Frame::ICRF => 6,
        Frame::LVLH => 7,
        Frame::RTN => 8,
        Frame::NTW => 9,
    }
}

/// Normalise `(from, to)` to a canonical ordered pair plus a `reversed` flag.
fn canonicalise(from: Frame, to: Frame) -> (Frame, Frame, bool) {
    if frame_order(from) <= frame_order(to) {
        (from, to, false)
    } else {
        (to, from, true)
    }
}

// ───── public API ────────────────────────────────────────────────────────

/// Quaternion rotating a vector from `from` to `to` at time `t`.
///
/// Uses the full IERS 2010 reduction for the Earth-frame chain. Every
/// time-parameterised pair is supported via the shortest path through the
/// frame graph; orbit-dependent frames ([`Frame::LVLH`], [`Frame::RTN`],
/// [`Frame::NTW`]) are not supported here — use [`to_gcrf`](super::to_gcrf)
/// for those.
///
/// # Examples
///
/// ```ignore
/// use satkit::{Frame, Instant};
/// use satkit::frametransform::rotation;
///
/// let t = Instant::from_datetime(2026, 5, 22, 12, 0, 0.0).unwrap();
/// let q = rotation(Frame::ITRF, Frame::GCRF, &t)?;
/// ```
pub fn rotation<T: TimeLike>(from: Frame, to: Frame, t: &T) -> Result<Quaternion> {
    if from == to {
        return Ok(Quaternion::identity());
    }
    let (a, b, reversed) = canonicalise(from, to);
    let q = canonical_rotation(a, b, t)?;
    Ok(if reversed { q.conjugate() } else { q })
}

/// Quaternion rotating a vector from `from` to `to` using the
/// IAU-76/FK5 approximate reduction (~1 arcsec, much cheaper than full
/// IERS 2010).
///
/// Only defined for pairs at the endpoints of the FK5 chain: [`Frame::ITRF`]
/// and the inertial cluster ([`Frame::GCRF`], [`Frame::EME2000`],
/// [`Frame::ICRF`], [`Frame::TEME`]). [`Frame::TIRS`] and [`Frame::CIRS`] are
/// defined by the IERS 2010 reduction and have no FK5 analogue — requests
/// involving them return [`Error::ApproxNotSupportedForFrame`].
pub fn rotation_approx<T: TimeLike>(from: Frame, to: Frame, t: &T) -> Result<Quaternion> {
    if from == to {
        return Ok(Quaternion::identity());
    }
    reject_for_approx(from)?;
    reject_for_approx(to)?;
    let (a, b, reversed) = canonicalise(from, to);
    let q = canonical_rotation_approx(a, b, t)?;
    Ok(if reversed { q.conjugate() } else { q })
}

/// State (position + velocity) transform from `from` to `to` at time `t`.
///
/// Uses the full IERS 2010 reduction. Properly handles the Earth-rotation
/// sweep term `ω⊕ × r` when transitioning between rotating ([`Frame::ITRF`],
/// [`Frame::TIRS`]) and inertial ([`Frame::GCRF`], [`Frame::EME2000`],
/// [`Frame::ICRF`], [`Frame::CIRS`], [`Frame::TEME`]) frames. ITRF↔TIRS is
/// treated as a static rotation — polar motion contributes ~1 mm/s at LEO
/// altitudes and is neglected here, matching the existing
/// [`itrf_to_gcrf_state`](super::itrf_to_gcrf_state) convention.
///
/// Orbit-dependent frames ([`Frame::LVLH`], [`Frame::RTN`], [`Frame::NTW`])
/// require orbit state to define their axes and are not handled here — use
/// [`to_gcrf`](super::to_gcrf) / [`from_gcrf`](super::from_gcrf) for those.
pub fn transform_state<T: TimeLike>(
    from: Frame,
    to: Frame,
    t: &T,
    pos: &Vector3,
    vel: &Vector3,
) -> Result<(Vector3, Vector3)> {
    if from == to {
        return Ok((*pos, *vel));
    }
    state_dispatch(from, to, t, pos, vel, /* approx = */ false)
}

/// State (position + velocity) transform using the IAU-76/FK5 approximate
/// reduction (~1 arcsec). Same domain restrictions as
/// [`rotation_approx`]: [`Frame::TIRS`] and [`Frame::CIRS`] are rejected
/// (no FK5 analogue); valid pairs are between [`Frame::ITRF`] and the
/// inertial cluster ([`Frame::GCRF`], [`Frame::EME2000`], [`Frame::ICRF`],
/// [`Frame::TEME`]), or within the inertial cluster.
pub fn transform_state_approx<T: TimeLike>(
    from: Frame,
    to: Frame,
    t: &T,
    pos: &Vector3,
    vel: &Vector3,
) -> Result<(Vector3, Vector3)> {
    if from == to {
        return Ok((*pos, *vel));
    }
    reject_for_approx(from)?;
    reject_for_approx(to)?;
    state_dispatch(from, to, t, pos, vel, /* approx = */ true)
}

// ───── internal dispatch ─────────────────────────────────────────────────

/// Reject TIRS / CIRS for approx-mode operations. Orbit frames are rejected
/// downstream by [`canonical_rotation_approx`] / [`state_dispatch`].
fn reject_for_approx(frame: Frame) -> Result<()> {
    match frame {
        Frame::TIRS | Frame::CIRS => Err(Error::ApproxNotSupportedForFrame { frame }),
        _ => Ok(()),
    }
}

/// Canonical-direction rotation for the full IERS 2010 reduction.
/// `from < to` per [`frame_order`].
fn canonical_rotation<T: TimeLike>(from: Frame, to: Frame, t: &T) -> Result<Quaternion> {
    use Frame::*;
    let q = match (from, to) {
        // ── 1-step direct edges ────────────────────────────────────────
        (ITRF, TIRS) => qitrf2tirs(t),
        (TIRS, CIRS) => qtirs2cirs(t),
        (CIRS, GCRF) => qcirs2gcrs(t),
        (ITRF, TEME) => qteme2itrf(t).conjugate(),
        (GCRF, EME2000) => qeme2000_to_gcrf().conjugate(),
        (GCRF, ICRF) => Quaternion::identity(),

        // ── existing amortised direct function ─────────────────────────
        (ITRF, GCRF) => qitrf2gcrf(t),

        // ── 2-step compositions (shortest path) ────────────────────────
        (ITRF, CIRS) => qtirs2cirs(t) * qitrf2tirs(t),
        (TIRS, GCRF) => qcirs2gcrs(t) * qtirs2cirs(t),
        // (TIRS, TEME): canonical pair wants q_{TIRS→TEME}. The expression
        // `qitrf2tirs * qteme2itrf` composes (applied to v) as TEME → ITRF →
        // TIRS, which is q_{TEME→TIRS}; conjugate to flip direction.
        (TIRS, TEME) => (qitrf2tirs(t) * qteme2itrf(t)).conjugate(),
        (TIRS, ICRF) => qcirs2gcrs(t) * qtirs2cirs(t),
        (CIRS, ICRF) => qcirs2gcrs(t),
        (EME2000, ICRF) => qeme2000_to_gcrf(),

        // ── 3-step compositions ────────────────────────────────────────
        // (CIRS, TEME): canonical pair wants q_{CIRS→TEME}. Same direction
        // flip as (TIRS, TEME) above.
        (CIRS, TEME) => (qtirs2cirs(t) * qitrf2tirs(t) * qteme2itrf(t)).conjugate(),
        (ITRF, EME2000) => qeme2000_to_gcrf().conjugate() * qitrf2gcrf(t),
        (ITRF, ICRF) => qitrf2gcrf(t),
        (TIRS, EME2000) => qeme2000_to_gcrf().conjugate() * qcirs2gcrs(t) * qtirs2cirs(t),
        (CIRS, EME2000) => qeme2000_to_gcrf().conjugate() * qcirs2gcrs(t),
        // (GCRF, TEME): canonical pair wants q_{GCRF→TEME}. We compose
        // through ITRF for full IERS 2010 (the existing `qteme2gcrf` uses
        // `qitrf2gcrf_approx` internally — that flavour belongs in
        // `rotation_approx`). The natural expression `qitrf2gcrf *
        // qteme2itrf` is q_{TEME→GCRF}; conjugate to flip direction.
        (GCRF, TEME) => (qitrf2gcrf(t) * qteme2itrf(t)).conjugate(),

        // ── 4+-step compositions ───────────────────────────────────────
        (TEME, EME2000) => qeme2000_to_gcrf().conjugate() * qitrf2gcrf(t) * qteme2itrf(t),
        (TEME, ICRF) => qitrf2gcrf(t) * qteme2itrf(t),

        // ── orbit-dependent frames need state ──────────────────────────
        (LVLH | RTN | NTW, _) | (_, LVLH | RTN | NTW) => {
            return Err(Error::OrbitFrameRequiresState { from, to });
        }

        // ── all remaining (from, to) pairs are not in canonical order ──
        // (canonicalise() guarantees from_order <= to_order)
        _ => unreachable!("non-canonical pair reached canonical_rotation: ({from}, {to})"),
    };
    Ok(q)
}

/// Canonical-direction rotation for the FK5 approximate reduction.
/// Only inertial-cluster + ITRF + TEME pairs are valid.
fn canonical_rotation_approx<T: TimeLike>(from: Frame, to: Frame, t: &T) -> Result<Quaternion> {
    use Frame::*;
    let q = match (from, to) {
        (ITRF, GCRF) => qitrf2gcrf_approx(t),
        (ITRF, TEME) => qteme2itrf(t).conjugate(),
        (ITRF, EME2000) => qeme2000_to_gcrf().conjugate() * qitrf2gcrf_approx(t),
        (ITRF, ICRF) => qitrf2gcrf_approx(t),

        (GCRF, EME2000) => qeme2000_to_gcrf().conjugate(),
        (GCRF, ICRF) => Quaternion::identity(),
        // (GCRF, TEME): same direction flip as in `canonical_rotation`.
        (GCRF, TEME) => (qitrf2gcrf_approx(t) * qteme2itrf(t)).conjugate(),

        (EME2000, ICRF) => qeme2000_to_gcrf(),
        (TEME, EME2000) => qeme2000_to_gcrf().conjugate() * qitrf2gcrf_approx(t) * qteme2itrf(t),
        (TEME, ICRF) => qitrf2gcrf_approx(t) * qteme2itrf(t),

        // TIRS / CIRS already rejected by reject_for_approx().
        // Orbit frames:
        (LVLH | RTN | NTW, _) | (_, LVLH | RTN | NTW) => {
            return Err(Error::OrbitFrameRequiresState { from, to });
        }

        _ => unreachable!("non-canonical pair reached canonical_rotation_approx: ({from}, {to})"),
    };
    Ok(q)
}

/// Common implementation for [`transform_state`] / [`transform_state_approx`].
///
/// Frame classification for state transforms:
///   - **Rotating** (relative to inertial space at Earth rotation rate):
///     [`Frame::ITRF`], [`Frame::TIRS`]. Polar motion between ITRF and TIRS
///     is treated as a static rotation (rate ~ 1.7e-9 rad/s × r is
///     sub-mm/s at LEO and is neglected, matching the existing
///     [`itrf_to_gcrf_state`] convention).
///   - **Inertial** (for state-transform purposes): [`Frame::GCRF`],
///     [`Frame::EME2000`], [`Frame::ICRF`], [`Frame::CIRS`],
///     [`Frame::TEME`]. CIRS's precession rate (~50"/year ≈ 7.7e-12 rad/s)
///     is negligible.
///
/// Dispatch:
///   - inertial ↔ inertial: just rotate pos and vel.
///   - rotating ↔ rotating (ITRF ↔ TIRS): just rotate (no sweep).
///   - rotating ↔ inertial: route via ITRF↔GCRF using the existing state
///     functions (which evaluate the `ω⊕ × r` sweep in TIRS where ω⊕ is
///     exactly along +ẑ), then chain a rotation on each side as needed.
fn state_dispatch<T: TimeLike>(
    from: Frame,
    to: Frame,
    t: &T,
    pos: &Vector3,
    vel: &Vector3,
    approx: bool,
) -> Result<(Vector3, Vector3)> {
    use Frame::*;

    // Orbit-dependent frames need orbit state to define their axes — not
    // handled by this state dispatch.
    if is_orbit_dependent(from) || is_orbit_dependent(to) {
        return Err(Error::OrbitFrameRequiresState { from, to });
    }

    let is_rotating = is_earth_rotating;

    // Case A: both inertial — straight rotation, no sweep term.
    if !is_rotating(from) && !is_rotating(to) {
        let q = if approx {
            rotation_approx(from, to, t)?
        } else {
            rotation(from, to, t)?
        };
        return Ok((q * *pos, q * *vel));
    }

    // Case B: both rotating (ITRF ↔ TIRS) — polar motion only, treated as
    // static, no sweep term.
    if is_rotating(from) && is_rotating(to) {
        // No approx variant: ITRF/TIRS aren't part of the FK5 chain. If
        // approx was requested for one of these, reject_for_approx() would
        // already have caught TIRS upstream.
        let q = rotation(from, to, t)?;
        return Ok((q * *pos, q * *vel));
    }

    // Case C: rotating ↔ inertial — route via ITRF↔GCRF.
    if is_rotating(from) {
        // Step 1: move pos/vel into ITRF basis (no sweep change; ITRF and
        // TIRS share angular velocity to the precision we model).
        let (p_itrf, v_itrf) = if from == ITRF {
            (*pos, *vel)
        } else {
            // from == TIRS
            let q = rotation(TIRS, ITRF, t)?;
            (q * *pos, q * *vel)
        };
        // Step 2: ITRF → GCRF, with the sweep term added in TIRS.
        let (p_gcrf, v_gcrf) = if approx {
            itrf_to_gcrf_state_approx(&p_itrf, &v_itrf, t)
        } else {
            itrf_to_gcrf_state(&p_itrf, &v_itrf, t)
        };
        // Step 3: rotate GCRF → target inertial frame.
        let q = if approx {
            rotation_approx(GCRF, to, t)?
        } else {
            rotation(GCRF, to, t)?
        };
        return Ok((q * p_gcrf, q * v_gcrf));
    }
    // Symmetric: inertial source, rotating target.
    // Step 1: rotate from source inertial frame to GCRF.
    let q = if approx {
        rotation_approx(from, GCRF, t)?
    } else {
        rotation(from, GCRF, t)?
    };
    let p_gcrf = q * *pos;
    let v_gcrf = q * *vel;
    // Step 2: GCRF → ITRF, with the sweep term subtracted in TIRS.
    let (p_itrf, v_itrf) = if approx {
        gcrf_to_itrf_state_approx(&p_gcrf, &v_gcrf, t)
    } else {
        gcrf_to_itrf_state(&p_gcrf, &v_gcrf, t)
    };
    // Step 3: move into target rotating basis.
    if to == ITRF {
        Ok((p_itrf, v_itrf))
    } else {
        // to == TIRS
        let q_itrf_to_tirs = rotation(ITRF, TIRS, t)?;
        Ok((q_itrf_to_tirs * p_itrf, q_itrf_to_tirs * v_itrf))
    }
}

// ───── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::qgcrf2itrf;
    use super::*;
    use crate::Instant;

    fn t() -> Instant {
        Instant::from_datetime(2026, 5, 22, 12, 0, 0.0).unwrap()
    }

    #[test]
    fn identity_pairs() {
        let tm = t();
        for f in [
            Frame::ITRF,
            Frame::TIRS,
            Frame::CIRS,
            Frame::GCRF,
            Frame::TEME,
            Frame::EME2000,
            Frame::ICRF,
        ] {
            let q = rotation(f, f, &tm).unwrap();
            assert!((q.w - 1.0).abs() < 1e-15, "{f}: w={}", q.w);
        }
    }

    #[test]
    fn matches_qitrf2gcrf() {
        let tm = t();
        let q_dispatch = rotation(Frame::ITRF, Frame::GCRF, &tm).unwrap();
        let q_direct = qitrf2gcrf(&tm);
        let v = numeris::vector![1000.0, 2000.0, 3000.0];
        let v_dispatch = q_dispatch * v;
        let v_direct = q_direct * v;
        assert!(
            (v_dispatch - v_direct).norm() < 1e-9,
            "dispatch={v_dispatch:?} direct={v_direct:?}"
        );
    }

    #[test]
    fn matches_qgcrf2itrf() {
        let tm = t();
        let q_dispatch = rotation(Frame::GCRF, Frame::ITRF, &tm).unwrap();
        let q_direct = qgcrf2itrf(&tm);
        let v = numeris::vector![1000.0, 2000.0, 3000.0];
        let v_dispatch = q_dispatch * v;
        let v_direct = q_direct * v;
        assert!((v_dispatch - v_direct).norm() < 1e-9);
    }

    #[test]
    fn matches_qitrf2tirs_direct_path() {
        // ITRF→TIRS is a single direct edge; should not pay precession cost.
        let tm = t();
        let q_dispatch = rotation(Frame::ITRF, Frame::TIRS, &tm).unwrap();
        let q_direct = qitrf2tirs(&tm);
        let v = numeris::vector![1000.0, 2000.0, 3000.0];
        assert!((q_dispatch * v - q_direct * v).norm() < 1e-12);
    }

    #[test]
    fn matches_qteme2itrf() {
        let tm = t();
        let q_dispatch = rotation(Frame::TEME, Frame::ITRF, &tm).unwrap();
        let q_direct = qteme2itrf(&tm);
        let v = numeris::vector![1000.0, 2000.0, 3000.0];
        assert!((q_dispatch * v - q_direct * v).norm() < 1e-12);
    }

    /// Direction pin for every TEME-involving pair. The roundtrip test
    /// passes regardless of direction (because `rotation(b, a)` is just
    /// the conjugate of `rotation(a, b)`), so this test pins the absolute
    /// direction by composing dispatch with a known-good reference. If a
    /// future change flips a sign, this fails.
    #[test]
    fn dispatch_teme_pairs_have_correct_direction() {
        use super::super::qteme2gcrf;
        let tm = t();
        let v = numeris::vector![7000e3_f64, 1000e3, 2000e3];

        // `qteme2gcrf` is the approximate TEME → GCRF rotation. Use it as
        // the reference for `rotation_approx` (which composes with the
        // same approximate ITRF↔GCRF). For full `rotation`, allow ~10 m
        // tolerance because dispatch uses the full IERS 2010 reduction
        // and qteme2gcrf is FK5-approx.
        let q_teme_to_gcrf_ref = qteme2gcrf(&tm);

        // rotation_approx(TEME, GCRF) should match qteme2gcrf to float
        // precision (both are the approximate reduction).
        let q_dispatch = rotation_approx(Frame::TEME, Frame::GCRF, &tm).unwrap();
        let lhs = q_dispatch * v;
        let rhs = q_teme_to_gcrf_ref * v;
        assert!(
            (lhs - rhs).norm() / v.norm() < 1e-12,
            "rotation_approx(TEME,GCRF) direction mismatch: dispatch={lhs:?} ref={rhs:?}"
        );

        // rotation_approx(GCRF, TEME) is the inverse.
        let q_dispatch = rotation_approx(Frame::GCRF, Frame::TEME, &tm).unwrap();
        let lhs = q_dispatch * v;
        let rhs = q_teme_to_gcrf_ref.conjugate() * v;
        assert!(
            (lhs - rhs).norm() / v.norm() < 1e-12,
            "rotation_approx(GCRF,TEME) direction mismatch: dispatch={lhs:?} ref={rhs:?}"
        );

        // Full rotation(TEME, GCRF): differs from qteme2gcrf by the
        // approx-vs-full reduction error (~1 arcsec). At |v|≈7300 km
        // that's ~35 m of position; check direction is right with a
        // loose tolerance.
        let q_dispatch = rotation(Frame::TEME, Frame::GCRF, &tm).unwrap();
        let lhs = q_dispatch * v;
        let rhs = q_teme_to_gcrf_ref * v;
        assert!(
            (lhs - rhs).norm() < 100.0,
            "rotation(TEME,GCRF) direction mismatch (>100 m): \
             dispatch={lhs:?} approx_ref={rhs:?}"
        );

        // Full rotation(GCRF, TEME): inverse direction, same tolerance.
        let q_dispatch = rotation(Frame::GCRF, Frame::TEME, &tm).unwrap();
        let lhs = q_dispatch * v;
        let rhs = q_teme_to_gcrf_ref.conjugate() * v;
        assert!(
            (lhs - rhs).norm() < 100.0,
            "rotation(GCRF,TEME) direction mismatch (>100 m): \
             dispatch={lhs:?} approx_ref={rhs:?}"
        );

        // (TIRS, TEME) and (CIRS, TEME): compose dispatch with the
        // direct functions to recover v_TEME from v_TEME and check
        // identity. Concretely: rotation(TIRS, TEME) ∘ rotation(TEME, TIRS)
        // = identity is the roundtrip (already tested) — but it doesn't
        // pin direction. Instead, take v_TEME, apply rotation(TEME, TIRS),
        // then qitrf2tirs(t) * qteme2itrf(t) * v_TEME should give the same
        // TIRS vector if both go TEME → ITRF → TIRS.
        let v_teme = v;
        let lhs = rotation(Frame::TEME, Frame::TIRS, &tm).unwrap() * v_teme;
        let rhs = qitrf2tirs(&tm) * (qteme2itrf(&tm) * v_teme);
        assert!(
            (lhs - rhs).norm() / v.norm() < 1e-12,
            "rotation(TEME,TIRS) direction mismatch: dispatch={lhs:?} ref={rhs:?}"
        );

        let lhs = rotation(Frame::TEME, Frame::CIRS, &tm).unwrap() * v_teme;
        let rhs = qtirs2cirs(&tm) * (qitrf2tirs(&tm) * (qteme2itrf(&tm) * v_teme));
        assert!(
            (lhs - rhs).norm() / v.norm() < 1e-12,
            "rotation(TEME,CIRS) direction mismatch: dispatch={lhs:?} ref={rhs:?}"
        );
    }

    #[test]
    fn roundtrip_all_pairs() {
        let tm = t();
        let v = numeris::vector![6378.0, 2000.0, 3000.0];
        let frames = [
            Frame::ITRF,
            Frame::TIRS,
            Frame::CIRS,
            Frame::GCRF,
            Frame::TEME,
            Frame::EME2000,
            Frame::ICRF,
        ];
        for &a in &frames {
            for &b in &frames {
                let q_ab = rotation(a, b, &tm).unwrap();
                let q_ba = rotation(b, a, &tm).unwrap();
                let v_round = q_ba * (q_ab * v);
                let err = (v_round - v).norm() / v.norm();
                assert!(err < 1e-12, "({a} → {b} → {a}) error {err}");
            }
        }
    }

    #[test]
    fn approx_rejects_intermediate_frames() {
        let tm = t();
        for f in [Frame::TIRS, Frame::CIRS] {
            let err = rotation_approx(f, Frame::GCRF, &tm).unwrap_err();
            assert!(matches!(err, Error::ApproxNotSupportedForFrame { frame } if frame == f));
            let err = rotation_approx(Frame::GCRF, f, &tm).unwrap_err();
            assert!(matches!(err, Error::ApproxNotSupportedForFrame { frame } if frame == f));
        }
    }

    #[test]
    fn approx_matches_qitrf2gcrf_approx() {
        let tm = t();
        let q_dispatch = rotation_approx(Frame::ITRF, Frame::GCRF, &tm).unwrap();
        let q_direct = qitrf2gcrf_approx(&tm);
        let v = numeris::vector![1000.0, 2000.0, 3000.0];
        assert!((q_dispatch * v - q_direct * v).norm() < 1e-9);
    }

    #[test]
    fn orbit_frames_rejected() {
        let tm = t();
        for of in [Frame::LVLH, Frame::RTN, Frame::NTW] {
            assert!(matches!(
                rotation(of, Frame::GCRF, &tm),
                Err(Error::OrbitFrameRequiresState { .. })
            ));
            assert!(matches!(
                rotation(Frame::GCRF, of, &tm),
                Err(Error::OrbitFrameRequiresState { .. })
            ));
        }
    }

    #[test]
    fn icrf_eme2000_constant_bias() {
        // ICRF↔EME2000 should be time-independent; check at two epochs.
        let t1 = Instant::from_datetime(2000, 1, 1, 0, 0, 0.0).unwrap();
        let t2 = Instant::from_datetime(2026, 5, 22, 0, 0, 0.0).unwrap();
        let q1 = rotation(Frame::ICRF, Frame::EME2000, &t1).unwrap();
        let q2 = rotation(Frame::ICRF, Frame::EME2000, &t2).unwrap();
        assert!((q1.w - q2.w).abs() < 1e-15);
    }

    #[test]
    fn eme2000_bias_matches_iers_2010() {
        // Pin the EME2000 → GCRF bias matrix to the IERS Conventions 2010
        // §5.32 small-Euler-angle reference (ξ0, η0, dα0). The matrix is
        // time-independent so a single epoch suffices.
        let t = Instant::from_datetime(2000, 1, 1, 12, 0, 0.0).unwrap();
        let q = rotation(Frame::EME2000, Frame::GCRF, &t).unwrap();
        let e1 = numeris::vector![1.0_f64, 0.0, 0.0];
        let e2 = numeris::vector![0.0_f64, 1.0, 0.0];
        let e3 = numeris::vector![0.0_f64, 0.0, 1.0];
        // Reference values from the IERS 2010 reference matrix (computed
        // off-line in numpy from the small-angle formula B^T = R3(-dα0) ·
        // R2(-ξ0) · R1(η0) with ξ0 = -0.016617", η0 = -0.006819",
        // dα0 = -0.014600"). See module-level doc comment.
        let c0 = q * e1;
        let c1 = q * e2;
        let c2 = q * e3;
        // First column: (1, dα0_rad, -ξ0_rad) to first order.
        assert!((c0[0] - 1.0).abs() < 1e-14);
        assert!((c0[1] - (-7.07827974e-8)).abs() < 1e-15);
        assert!((c0[2] - 8.05614894e-8).abs() < 1e-15);
        // Second column: (-dα0_rad, 1, η0_rad).
        assert!((c1[0] - 7.07827948e-8).abs() < 1e-15);
        assert!((c1[1] - 1.0).abs() < 1e-14);
        assert!((c1[2] - 3.30594449e-8).abs() < 1e-15);
        // Third column: (ξ0_rad, -η0_rad, 1).
        assert!((c2[0] - (-8.05614917e-8)).abs() < 1e-15);
        assert!((c2[1] - (-3.30594392e-8)).abs() < 1e-15);
        assert!((c2[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn transform_state_itrf_to_gcrf_matches_direct() {
        let tm = t();
        let p_itrf = numeris::vector![6378137.0, 0.0, 0.0];
        let v_itrf = numeris::vector![0.0, 0.0, 0.0];
        let (p_dispatch, v_dispatch) =
            transform_state(Frame::ITRF, Frame::GCRF, &tm, &p_itrf, &v_itrf).unwrap();
        let (p_direct, v_direct) = itrf_to_gcrf_state(&p_itrf, &v_itrf, &tm);
        assert!((p_dispatch - p_direct).norm() < 1e-9);
        assert!((v_dispatch - v_direct).norm() < 1e-12);
    }

    #[test]
    fn transform_state_roundtrip_itrf_gcrf() {
        let tm = t();
        let p = numeris::vector![6378137.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        let (p2, v2) = transform_state(Frame::ITRF, Frame::GCRF, &tm, &p, &v).unwrap();
        let (p3, v3) = transform_state(Frame::GCRF, Frame::ITRF, &tm, &p2, &v2).unwrap();
        assert!((p3 - p).norm() / p.norm() < 1e-10);
        assert!((v3 - v).norm() / v.norm() < 1e-10);
    }

    #[test]
    fn transform_state_all_non_orbit_pairs_roundtrip() {
        // Every pair of {ITRF, TIRS, CIRS, GCRF, EME2000, ICRF, TEME}
        // should roundtrip via transform_state in both directions.
        let tm = t();
        let p = numeris::vector![7000000.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        let frames = [
            Frame::ITRF,
            Frame::TIRS,
            Frame::CIRS,
            Frame::GCRF,
            Frame::TEME,
            Frame::EME2000,
            Frame::ICRF,
        ];
        for &a in &frames {
            for &b in &frames {
                let (pa, va) = transform_state(a, b, &tm, &p, &v).unwrap();
                let (pr, vr) = transform_state(b, a, &tm, &pa, &va).unwrap();
                let pos_err = (pr - p).norm() / p.norm();
                let vel_err = (vr - v).norm() / v.norm();
                assert!(pos_err < 1e-10, "({a}↔{b}) pos roundtrip err {pos_err}");
                assert!(vel_err < 1e-10, "({a}↔{b}) vel roundtrip err {vel_err}");
            }
        }
    }

    #[test]
    fn transform_state_inertial_pair_no_sweep() {
        // GCRF ↔ TEME: both inertial. Should be a pure rotation — v
        // magnitude preserved exactly (no sweep added or removed).
        let tm = t();
        let p = numeris::vector![7000000.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        let (_, v_teme) = transform_state(Frame::GCRF, Frame::TEME, &tm, &p, &v).unwrap();
        assert!(
            (v_teme.norm() - v.norm()).abs() < 1e-9,
            "inertial pair preserved |v|: |v|={}, |v_teme|={}",
            v.norm(),
            v_teme.norm()
        );
    }

    #[test]
    fn transform_state_tirs_via_itrf_chain() {
        // TIRS → GCRF should equal (ITRF → GCRF after rotating pos/vel
        // from TIRS into ITRF first). The dispatch routes via that path.
        let tm = t();
        let p_tirs = numeris::vector![7000000.0, 0.0, 0.0];
        let v_tirs = numeris::vector![0.0, 0.0, 0.0];
        let (p_a, v_a) = transform_state(Frame::TIRS, Frame::GCRF, &tm, &p_tirs, &v_tirs).unwrap();
        // Reference: do it by hand.
        let q_tirs_to_itrf = rotation(Frame::TIRS, Frame::ITRF, &tm).unwrap();
        let p_itrf = q_tirs_to_itrf * p_tirs;
        let v_itrf = q_tirs_to_itrf * v_tirs;
        let (p_b, v_b) = itrf_to_gcrf_state(&p_itrf, &v_itrf, &tm);
        assert!((p_a - p_b).norm() < 1e-9);
        assert!((v_a - v_b).norm() < 1e-12);
    }

    #[test]
    fn transform_state_itrf_tirs_no_sweep() {
        // ITRF ↔ TIRS is treated as static (polar motion only; no sweep).
        // |v| is preserved by the rotation.
        let tm = t();
        let p = numeris::vector![7000000.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        let (_, v_tirs) = transform_state(Frame::ITRF, Frame::TIRS, &tm, &p, &v).unwrap();
        assert!((v_tirs.norm() - v.norm()).abs() < 1e-9);
    }

    #[test]
    fn transform_state_approx_rejects_intermediates() {
        let tm = t();
        let p = numeris::vector![7000000.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        for f in [Frame::TIRS, Frame::CIRS] {
            assert!(matches!(
                transform_state_approx(f, Frame::GCRF, &tm, &p, &v),
                Err(Error::ApproxNotSupportedForFrame { .. })
            ));
        }
    }

    #[test]
    fn transform_state_orbit_frames_rejected() {
        let tm = t();
        let p = numeris::vector![7000000.0, 0.0, 0.0];
        let v = numeris::vector![0.0, 7600.0, 0.0];
        for of in [Frame::LVLH, Frame::RTN, Frame::NTW] {
            assert!(matches!(
                transform_state(of, Frame::GCRF, &tm, &p, &v),
                Err(Error::OrbitFrameRequiresState { .. })
            ));
        }
    }
}
