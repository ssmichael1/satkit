//! Property-based tests that need the Earth-orientation (EOP) data file:
//! UT1 and the Earth-fixed frame transforms.
//!
//! Companion to `tests/properties.rs` (data-independent; see its header for
//! the edge-biased generators and the "different routes" philosophy). Each
//! test here skips, with a note on stderr, when `finals2000A.all` is not
//! in the data directory (set `SATKIT_DATA`, or run
//! `python -m satkit.utils.update_datafiles`), the same way
//! `tests/earth_orientation_params_init.rs` does. No test here triggers a
//! download.
//!
//! Every epoch is drawn inside the loaded table's *observed* range, so the
//! properties do not depend on how fresh the file is.
//!
//! # Case counts
//!
//! The full IERS 2010 reduction is far costlier than the time arithmetic,
//! so the frame properties run `FRAME_CASES` (64) cases and the UT1
//! properties `UT1_CASES` (256); the file takes well under a second once
//! built. `PROPTEST_CASES` overrides both for a deeper run.

mod time_edges;

use std::sync::OnceLock;

use proptest::prelude::*;
use time_edges::*;

use satkit::earth_orientation_params as eop;
use satkit::frametransform::{
    qgcrf2itrf, qgcrf2itrf_approx, qitrf2gcrf, qitrf2gcrf_approx, qteme2gcrf, qteme2itrf, rotation,
    rotation_approx,
};
use satkit::{Duration, Frame, Instant, Quaternion, TimeScale};

const UT1_CASES: u32 = 256;
const FRAME_CASES: u32 = 64;

/// Arcseconds to radians.
const ASEC: f64 = std::f64::consts::PI / 180.0 / 3600.0;

/// Loaded-table bounds, `(first, last_observed)`, or `None` (with a
/// one-time note) when no EOP file is available locally.
fn eop_range() -> Option<(Instant, Instant)> {
    static RANGE: OnceLock<Option<(Instant, Instant)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        if satkit::utils::find_data_file(eop::FINALS2000A_FILE).is_none() {
            eprintln!(
                "skipping EOP properties: no finals2000A.all in the data \
                 directories; set SATKIT_DATA or run `python -m satkit.utils.update_datafiles`"
            );
            return None;
        }
        let cov = eop::coverage()?;
        // Keep two days of margin at both ends: interpolation needs a row
        // either side, and we evaluate up to a few hours past the sample.
        Some((
            cov.first + Duration::from_days(2.0),
            cov.last_observed - Duration::from_days(2.0),
        ))
    })
}

/// Map `frac ∈ [0, 1)` onto the loaded table's observed range.
fn in_range(frac: f64) -> Option<Instant> {
    let (a, b) = eop_range()?;
    Some(a + Duration::from_microseconds(((b - a).as_microseconds() as f64 * frac) as i64))
}

/// Whether leap day `i` (and a day either side) lies inside the table.
fn leap_day_covered(i: usize) -> bool {
    let Some((a, b)) = eop_range() else {
        return false;
    };
    let (y, m, d, _) = LEAP_DAYS[i];
    let day = Label::new(y, m, d, 0, 0, 0).instant();
    day >= a && day + Duration::from_days(2.0) <= b
}

/// UT1 in seconds (MJD × 86400).
fn ut1(t: &Instant) -> f64 {
    t.as_mjd_with_scale(TimeScale::UT1) * 86_400.0
}

/// Rotation angle of a unit quaternion, in `[0, π]`.
fn angle(q: &Quaternion) -> f64 {
    let v = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
    2.0 * v.atan2(q.w.abs())
}

/// Angle of the rotation taking `b` to `a`.
fn angle_between(a: &Quaternion, b: &Quaternion) -> f64 {
    angle(&(*a * b.conjugate()))
}

/// Bound on |LOD| (excess length of day) used for UT1 rate checks. The
/// observed range since 1962 is about −1.5 … +4.5 ms.
const LOD_MAX: f64 = 5.0e-3;

// ───────────────────────── UT1 ─────────────────────────

/// Deterministic sweep of every leap second inside the table: from 3 s
/// before the inserted second to 3 s after it, in 250 ms steps, UT1
/// advances by 1 s ± 1 ms per SI second. UT1 is continuous; only UTC
/// jumps. Catches UT1 − UTC being evaluated on the wrong side of the step
/// or UT1 freezing during `23:59:60`.
#[test]
fn ut1_one_second_steps_at_every_leap_second() {
    if eop_range().is_none() {
        return;
    }
    let mut checked = 0;
    for (i, &(y, mo, d, ins_us)) in LEAP_DAYS.iter().enumerate() {
        if !leap_day_covered(i) {
            continue;
        }
        let start = Label::new(y, mo, d, 23, 59, 57 * US).instant();
        for k in 0..=(4 * (ins_us + 6 * US) / US) {
            let t = start + Duration::from_microseconds(k * US / 4);
            let step = ut1(&(t + Duration::from_seconds(1.0))) - ut1(&t);
            assert!(
                (step - 1.0).abs() < 1e-3,
                "UT1 advanced {step} s in 1 s at {t} (leap second {y}-{mo:02}-{d:02})"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 20,
        "only {checked} leap seconds inside the EOP table"
    );
}

proptest! {
    #![proptest_config(cases(UT1_CASES))]

    /// UT1 advances at the SI rate to within the length-of-day excess,
    /// over 1 s and over 6 h windows, near leap seconds and anywhere in the
    /// table. The 6 h window catches a UT1 − UTC interpolation that smears
    /// the +1 s leap-second step over the preceding day (a rate error of
    /// 1 s/day, i.e. 0.25 s over 6 h against a 1.25 ms tolerance).
    #[test]
    fn ut1_rate_matches_si_rate(
        (i, off) in leap_offset(),
        frac in 0.0..1.0f64,
        day_frac in 0.0..1.0f64,
        which in 0..3u8,
    ) {
        if eop_range().is_none() {
            return Ok(()); // no EOP data: skip (see eop_range)
        }
        let t = match which {
            // Within seconds of a leap second (incl. inside it)
            0 => {
                prop_assume!(leap_day_covered(i));
                leap_label(i, off).instant()
            }
            // Anywhere on a leap-second day, or up to 6 h before it
            1 => {
                prop_assume!(leap_day_covered(i));
                let (y, m, d, _) = LEAP_DAYS[i];
                Label::new(y, m, d, 0, 0, 0).instant()
                    + Duration::from_seconds(day_frac * (86_400.0 + 21_600.0) - 21_600.0)
            }
            // Uniform over the table
            _ => in_range(frac).unwrap(),
        };
        for span in [1.0, 21_600.0] {
            let got = ut1(&(t + Duration::from_seconds(span))) - ut1(&t);
            let tol = span * LOD_MAX / 86_400.0 + 2e-6;
            prop_assert!(
                (got - span).abs() < tol,
                "UT1 advanced {got} s over {span} SI s from {t} (tolerance {tol})"
            );
        }
    }

    /// `from_mjd_with_scale(UT1)` inverts `as_mjd_with_scale(UT1)`, near
    /// and inside leap seconds and anywhere in the table.
    #[test]
    fn ut1_inverse((i, off) in leap_offset(), frac in 0.0..1.0f64, near_leap in any::<bool>()) {
        if eop_range().is_none() {
            return Ok(()); // no EOP data: skip (see eop_range)
        }
        let t = if near_leap {
            prop_assume!(leap_day_covered(i));
            leap_label(i, off).instant()
        } else {
            in_range(frac).unwrap()
        };
        let m = t.as_mjd_with_scale(TimeScale::UT1);
        let back = Instant::from_mjd_with_scale(m, TimeScale::UT1);
        let err = (back - t).as_microseconds().abs();
        prop_assert!(err <= 5, "{t} → UT1 MJD {m} → {back} ({err} µs)");
    }
}

// ───────────────────────── Frame transforms ─────────────────────────

proptest! {
    #![proptest_config(cases(FRAME_CASES))]

    /// Different routes through the frame graph agree: GCRF→ITRF is the
    /// inverse of ITRF→GCRF; the dispatcher matches the named functions;
    /// ITRF→TIRS→CIRS→GCRF composes to ITRF→GCRF; TEME→GCRF equals
    /// TEME→ITRF→GCRF and TEME→CIRS→GCRF; TEME↔GCRF round-trips.
    #[test]
    fn frame_routes_agree(frac in 0.0..1.0f64, (i, off) in leap_offset(), near_leap in any::<bool>()) {
        if eop_range().is_none() {
            return Ok(()); // no EOP data: skip (see eop_range)
        }
        let t = if near_leap {
            prop_assume!(leap_day_covered(i));
            leap_label(i, off).instant()
        } else {
            in_range(frac).unwrap()
        };
        let tol = 1e-12;
        let i2g = qitrf2gcrf(&t);
        prop_assert!(angle(&(qgcrf2itrf(&t) * i2g)) < tol);
        prop_assert!(angle_between(&rotation(Frame::ITRF, Frame::GCRF, &t).unwrap(), &i2g) < tol);
        prop_assert!(angle_between(&rotation(Frame::GCRF, Frame::ITRF, &t).unwrap(), &qgcrf2itrf(&t)) < tol);

        let r = |a, b| rotation(a, b, &t).unwrap();
        let chain = r(Frame::CIRS, Frame::GCRF) * r(Frame::TIRS, Frame::CIRS) * r(Frame::ITRF, Frame::TIRS);
        prop_assert!(angle_between(&chain, &i2g) < tol, "ITRF→TIRS→CIRS→GCRF differs");

        let teme2gcrf = r(Frame::TEME, Frame::GCRF);
        prop_assert!(angle_between(&teme2gcrf, &(i2g * qteme2itrf(&t))) < tol);
        prop_assert!(angle_between(&teme2gcrf, &(r(Frame::CIRS, Frame::GCRF) * r(Frame::TEME, Frame::CIRS))) < tol);
        prop_assert!(angle(&(r(Frame::GCRF, Frame::TEME) * teme2gcrf)) < tol);
        prop_assert!(angle(&(rotation_approx(Frame::GCRF, Frame::TEME, &t).unwrap()
            * rotation_approx(Frame::TEME, Frame::GCRF, &t).unwrap())) < tol);
    }

    /// The Earth-fixed rotation is continuous in time: over one SI second
    /// the ITRF→GCRF rotation (full and approximate) and TEME→ITRF turn by
    /// ω⊕ · 1 s (ω⊕ = 2π · 1.00273781191135448 / 86400 rad per UT1 second)
    /// to within 2e-8 rad (≈ 0.3 ms of UT1), across UTC midnights and leap
    /// seconds. A 1 s UT1 jump would show up as a 7.3e-5 rad error.
    #[test]
    fn earth_rotation_continuous_across_leap_seconds(
        (i, off) in leap_offset(),
        frac in 0.0..1.0f64,
        near_leap in any::<bool>(),
    ) {
        if eop_range().is_none() {
            return Ok(()); // no EOP data: skip (see eop_range)
        }
        let t = if near_leap {
            prop_assume!(leap_day_covered(i));
            leap_label(i, off).instant()
        } else {
            in_range(frac).unwrap()
        };
        let t2 = t + Duration::from_seconds(1.0);
        let omega = std::f64::consts::TAU * 1.002_737_811_911_354_5 / 86_400.0;
        for (name, a, b) in [
            ("qitrf2gcrf", qitrf2gcrf(&t), qitrf2gcrf(&t2)),
            ("qitrf2gcrf_approx", qitrf2gcrf_approx(&t), qitrf2gcrf_approx(&t2)),
            ("qteme2itrf", qteme2itrf(&t), qteme2itrf(&t2)),
        ] {
            let turned = angle_between(&b, &a);
            prop_assert!(
                (turned - omega).abs() < 2e-8,
                "{name} turned {turned:e} rad in 1 s at {t} (expected {omega:e})"
            );
        }
    }

    /// The approximate reduction agrees with the full IERS 2010 one to the
    /// documented "~1 arcsec" (src/frametransform/mod.rs module table;
    /// docs/api/frametransform.md). Measured worst case over 1973–2026 is
    /// ≈ 1.0 arcsec (up to 0.6″ of it polar motion, which the approximate
    /// chain neglects); the bound is 1.5 arcsec. `qteme2gcrf` involves no
    /// polar motion and is documented to 0.55″ (measured 0.545″): bound
    /// 0.6″ against the full TEME→GCRF dispatch, and it is exactly the
    /// approximate chain applied to PEF (TEME rotated by GMST82 alone).
    /// `qgcrf2itrf_approx` is the inverse of `qitrf2gcrf_approx`.
    #[test]
    fn approx_transforms_within_documented_accuracy(frac in 0.0..1.0f64) {
        if eop_range().is_none() {
            return Ok(()); // no EOP data: skip (see eop_range)
        }
        let t = in_range(frac).unwrap();
        let full = qitrf2gcrf(&t);
        let approx = qitrf2gcrf_approx(&t);
        let err = angle_between(&full, &approx) / ASEC;
        prop_assert!(err < 1.5, "approx vs full ITRF→GCRF: {err} arcsec at {t}");
        prop_assert!(angle(&(qgcrf2itrf_approx(&t) * approx)) < 1e-12);

        let err = angle_between(&qteme2gcrf(&t), &rotation(Frame::TEME, Frame::GCRF, &t).unwrap()) / ASEC;
        prop_assert!(err < 0.6, "qteme2gcrf vs full TEME→GCRF: {err} arcsec at {t}");
        let pm_free = approx * Quaternion::rotz(-satkit::frametransform::gmst(&t));
        prop_assert!(angle_between(&qteme2gcrf(&t), &pm_free) < 1e-12);
        prop_assert!(angle_between(&rotation_approx(Frame::TEME, Frame::GCRF, &t).unwrap(), &pm_free) < 1e-12);
    }
}
