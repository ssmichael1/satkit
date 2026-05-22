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
// Constant rotation from EME2000 (J2000 mean dynamical equator + equinox)
// to GCRF. See IERS Technical Note 36, §5.4.4 and Vallado §3.7.2.
//
// Three small offsets (arcseconds at J2000):
//   dα0  = -0.0146     RA offset of the mean dynamical equator
//   ξ0   = -0.041775   obliquity-direction frame bias
//   η0   = -0.0068192  azimuth-direction frame bias
//
// The bias matrix is the product B = Rx(-η0) · Ry(ξ0) · Rz(dα0) (linearised
// for these milliarcsecond angles).
const FRAME_BIAS_DALPHA0_AS: f64 = -0.0146;
const FRAME_BIAS_XI0_AS: f64 = -0.041775;
const FRAME_BIAS_ETA0_AS: f64 = -0.0068192;

/// Constant quaternion: EME2000 → GCRF (≈ 17 milliarcsec frame bias).
fn qeme2000_to_gcrf() -> Quaternion {
    let dalpha0 = FRAME_BIAS_DALPHA0_AS * ASEC2RAD;
    let xi0 = FRAME_BIAS_XI0_AS * ASEC2RAD;
    let eta0 = FRAME_BIAS_ETA0_AS * ASEC2RAD;
    Quaternion::rotx(-eta0) * Quaternion::roty(xi0) * Quaternion::rotz(dalpha0)
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
/// sweep term `ω⊕ × r` when transitioning between rotating (ITRF) and
/// inertial frames.
///
/// **Currently supported pairs**: identity, ITRF↔{GCRF, EME2000, ICRF, TEME},
/// and within-inertial pairs (no rotating frame involved). Other pairs
/// (involving TIRS, CIRS as endpoints, or LVLH/RTN/NTW) return
/// [`Error::StateTransformNotSupported`]; for those, compose
/// [`itrf_to_gcrf_state`](super::itrf_to_gcrf_state) with the appropriate
/// constant rotation, or use the orbit-frame [`to_gcrf`](super::to_gcrf)
/// helper for satellite-local frames.
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
/// reduction. Same supported-pair set as [`transform_state`]; same error
/// behaviour for unsupported pairs.
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
        (TIRS, TEME) => qitrf2tirs(t) * qteme2itrf(t),
        (TIRS, ICRF) => qcirs2gcrs(t) * qtirs2cirs(t),
        (CIRS, ICRF) => qcirs2gcrs(t),
        (EME2000, ICRF) => qeme2000_to_gcrf(),

        // ── 3-step compositions ────────────────────────────────────────
        (CIRS, TEME) => qtirs2cirs(t) * qitrf2tirs(t) * qteme2itrf(t),
        (ITRF, EME2000) => qeme2000_to_gcrf().conjugate() * qitrf2gcrf(t),
        (ITRF, ICRF) => qitrf2gcrf(t),
        (TIRS, EME2000) => qeme2000_to_gcrf().conjugate() * qcirs2gcrs(t) * qtirs2cirs(t),
        (CIRS, EME2000) => qeme2000_to_gcrf().conjugate() * qcirs2gcrs(t),
        // GCRF↔TEME: compose via ITRF so we get full reduction (the existing
        // qteme2gcrf uses qitrf2gcrf_approx internally — keep that to
        // rotation_approx).
        (GCRF, TEME) => qitrf2gcrf(t) * qteme2itrf(t),

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
        (GCRF, TEME) => qitrf2gcrf_approx(t) * qteme2itrf(t),

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
/// Only handles pairs where at least one side is ITRF (the rotating frame)
/// or both sides are inertial.
fn state_dispatch<T: TimeLike>(
    from: Frame,
    to: Frame,
    t: &T,
    pos: &Vector3,
    vel: &Vector3,
    approx: bool,
) -> Result<(Vector3, Vector3)> {
    use Frame::*;

    // Reject orbit-dependent frames.
    if matches!(from, LVLH | RTN | NTW) || matches!(to, LVLH | RTN | NTW) {
        return Err(Error::OrbitFrameRequiresState { from, to });
    }

    let is_inertial = |f: Frame| matches!(f, GCRF | EME2000 | ICRF | TEME);

    // Case 1: ITRF ↔ inertial. Use the existing state-transform functions
    // (which handle the sweep term in TIRS) then rotate to/from the target.
    if from == ITRF && is_inertial(to) {
        let (p_gcrf, v_gcrf) = if approx {
            itrf_to_gcrf_state_approx(pos, vel, t)
        } else {
            itrf_to_gcrf_state(pos, vel, t)
        };
        // Rotate from GCRF to the requested inertial target.
        let q = if approx {
            rotation_approx(GCRF, to, t)?
        } else {
            rotation(GCRF, to, t)?
        };
        return Ok((q * p_gcrf, q * v_gcrf));
    }
    if to == ITRF && is_inertial(from) {
        // Source-inertial → GCRF first, then GCRF → ITRF state.
        let q = if approx {
            rotation_approx(from, GCRF, t)?
        } else {
            rotation(from, GCRF, t)?
        };
        let p_gcrf = q * *pos;
        let v_gcrf = q * *vel;
        let (p_itrf, v_itrf) = if approx {
            gcrf_to_itrf_state_approx(&p_gcrf, &v_gcrf, t)
        } else {
            gcrf_to_itrf_state(&p_gcrf, &v_gcrf, t)
        };
        return Ok((p_itrf, v_itrf));
    }

    // Case 2: both inertial. Just rotate — no sweep term.
    if is_inertial(from) && is_inertial(to) {
        let q = if approx {
            rotation_approx(from, to, t)?
        } else {
            rotation(from, to, t)?
        };
        return Ok((q * *pos, q * *vel));
    }

    // Anything else (involving TIRS, CIRS, or non-ITRF rotating-frame
    // bookkeeping) is out of scope for the first cut.
    Err(Error::StateTransformNotSupported { from, to })
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
}
