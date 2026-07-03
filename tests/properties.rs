//! Property-based tests (proptest).
//!
//! Each test asserts an invariant over a *domain* of randomly generated
//! inputs rather than hand-picked examples. proptest over-weights boundary
//! values (0, endpoints), automatically shrinks any failure to a minimal
//! counterexample, and records failures in `proptest-regressions/` (checked
//! into git) so they replay as regression tests forever.
//!
//! Everything here is deliberately independent of the external data files
//! (no EOP / space weather / ephemerides), so the suite runs anywhere
//! `cargo test` does. UT1 conversions are excluded for that reason.

use proptest::prelude::*;

use satkit::kepler::Anomaly;
use satkit::{Duration, ITRFCoord, Instant, Kepler, Quaternion, TimeScale, TLE};

use std::f64::consts::{PI, TAU};

// ───────────────────────── Kepler ─────────────────────────

proptest! {
    /// from_pv ∘ to_pv round-trips for all elliptical orbits — including the
    /// exactly-circular and exactly-equatorial degeneracies that historically
    /// produced NaN (fixed via the Vallado special cases).
    #[test]
    fn kepler_pv_roundtrip(
        a in 6.6e6..5.0e7f64,
        e in 0.0..0.95f64,
        i in 0.0..PI,
        raan in 0.0..TAU,
        argp in 0.0..TAU,
        nu in 0.0..TAU,
    ) {
        let k = Kepler::new(a, e, i, raan, argp, Anomaly::True(nu));
        let (r, v) = k.to_pv();
        prop_assert!(r.norm().is_finite() && v.norm().is_finite());

        let k2 = Kepler::from_pv(r, v);
        prop_assert!(k2.is_ok(), "from_pv failed: {:?}", k2.err());
        let (r2, v2) = k2.unwrap().to_pv();
        prop_assert!((r - r2).norm() / r.norm() < 1e-9,            "position round-trip error {:e}", (r - r2).norm() / r.norm());
        prop_assert!((v - v2).norm() / v.norm() < 1e-9,
            "velocity round-trip error {:e}", (v - v2).norm() / v.norm());
    }

    /// Mean → (eccentric) → true → mean anomaly round-trips, and the Newton
    /// solve terminates, for every eccentricity below the parabolic limit.
    /// (mean2eccentric historically looped forever for e ≥ 1; the iteration
    /// is now capped — this pins the well-posed domain.)
    #[test]
    fn kepler_anomaly_roundtrip(
        e in 0.0..0.99f64,
        ma in 0.0..TAU,
    ) {
        let k = Kepler::with_mean_anomaly(7.0e6, e, 0.5, 0.0, 0.0, ma);
        let ma2 = k.mean_anomaly().rem_euclid(TAU);
        let diff = (ma2 - ma).abs();
        let diff = diff.min(TAU - diff); // wrap-around distance
        prop_assert!(diff < 1e-9, "mean anomaly round-trip error {:e} at e={}", diff, e);
    }

    /// Propagating forward by one period returns to the starting state.
    #[test]
    fn kepler_period_closure(
        a in 6.6e6..5.0e7f64,
        e in 0.0..0.9f64,
        nu in 0.0..TAU,
    ) {
        let k = Kepler::new(a, e, 0.6, 1.0, 2.0, Anomaly::True(nu));
        let k2 = k.propagate(&Duration::from_seconds(k.period()));
        let (r, _) = k.to_pv();
        let (r2, _) = k2.to_pv();
        prop_assert!((r - r2).norm() / r.norm() < 1e-6,
            "state after one period differs by {:e}", (r - r2).norm() / r.norm());
    }
}

// ───────────────────────── Geodesy (Vincenty) ─────────────────────────

proptest! {
    /// Geodesic distance is finite and non-negative for *all* coordinate
    /// pairs — including coincident points (historically NaN via 0/0) and
    /// near-antipodal pairs (where Vincenty loses accuracy but must never
    /// return NaN/inf or a negative distance).
    #[test]
    fn geodesic_distance_finite_nonnegative(
        lat1 in -90.0..90.0f64,
        lon1 in -180.0..180.0f64,
        lat2 in -90.0..90.0f64,
        lon2 in -180.0..180.0f64,
    ) {
        let p1 = ITRFCoord::from_geodetic_deg(lat1, lon1, 0.0);
        let p2 = ITRFCoord::from_geodetic_deg(lat2, lon2, 0.0);
        let (d, h1, h2) = p1.geodesic_distance(&p2);
        prop_assert!(d.is_finite(), "distance not finite for ({lat1},{lon1})→({lat2},{lon2})");
        prop_assert!(d >= 0.0, "negative distance {d}");
        prop_assert!(h1.is_finite() && h2.is_finite(), "headings not finite");
        // Half Earth's circumference is a hard upper bound (plus slack for
        // the documented near-antipodal inaccuracy).
        prop_assert!(d < 2.1e7, "distance {d} exceeds half circumference");
        // Coincident-point self distance is (near-)zero.
        prop_assert!(p1.distance_to(&p1) < 1e-6);
    }

    /// Geodetic → Cartesian → geodetic round-trips over the full globe and a
    /// wide altitude range.
    #[test]
    fn geodetic_roundtrip(
        lat in -89.999..89.999f64,
        lon in -179.999..179.999f64,
        alt in -5.0e3..1.0e7f64,
    ) {
        let c = ITRFCoord::from_geodetic_deg(lat, lon, alt);
        prop_assert!((c.latitude_deg() - lat).abs() < 1e-8, "lat error");
        prop_assert!((c.longitude_deg() - lon).abs() < 1e-8, "lon error");
        prop_assert!((c.hae() - alt).abs() < 1e-4, "alt error {:e}", (c.hae() - alt).abs());
    }
}

// ───────────────────────── Time ─────────────────────────

proptest! {
    /// MJD ↔ Instant round-trips in every data-independent time scale
    /// (UT1 needs Earth-orientation data and is excluded). Tolerance is a
    /// few µs — the f64 day representation quantizes at ~1 µs in this range.
    #[test]
    fn mjd_roundtrip(
        mjd in 41317.0..69807.0f64, // 1972 (leap-second era) … 2045
    ) {
        for scale in [TimeScale::UTC, TimeScale::TAI, TimeScale::TT, TimeScale::GPS, TimeScale::TDB] {
            let t = Instant::from_mjd_with_scale(mjd, scale);
            let mjd2 = t.as_mjd_with_scale(scale);
            let err_us = (mjd2 - mjd).abs() * 86400.0e6;
            prop_assert!(err_us < 5.0, "{scale} round-trip error {err_us} µs at mjd {mjd}");
        }
    }

    /// Unixtime ↔ Instant round-trips (unixtime ignores leap seconds; the
    /// mapping must still be self-consistent).
    #[test]
    fn unixtime_roundtrip(ut in 0.0..2.4e9f64) {
        let t = Instant::from_unixtime(ut);
        prop_assert!((t.as_unixtime() - ut).abs() < 5e-6,
            "unixtime round-trip error {:e}", (t.as_unixtime() - ut).abs());
    }

    /// Instant ± Duration arithmetic is exact at microsecond granularity.
    #[test]
    fn instant_duration_arithmetic(
        mjd in 50000.0..60000.0f64,
        dt_us in -1_000_000_000_000i64..1_000_000_000_000i64, // ±11.5 days
    ) {
        let t = Instant::from_mjd_with_scale(mjd, TimeScale::TAI);
        let d = Duration::from_seconds(dt_us as f64 * 1e-6);
        let t2 = t + d - d;
        prop_assert!(((t2 - t).as_seconds()).abs() < 1e-6);
    }
}

// ───────────────────────── TLE ─────────────────────────

proptest! {
    /// TLE formatting round-trips: every element written by to_2line is
    /// recovered by load_2line to within the TLE format's field precision.
    #[test]
    fn tle_format_roundtrip(
        sat_num in 1..99999i32,
        inclination in 0.0..180.0f64,
        raan in 0.0..360.0f64,
        eccen in 0.0..0.9999f64,
        argp in 0.0..360.0f64,
        ma in 0.0..360.0f64,
        mm in 0.5..17.0f64,
        bstar in -1.0e-2..1.0e-2f64,
        epoch_days in 0.0..10000.0f64, // 2000-01-01 + up to ~27 years
    ) {
        let mut tle = TLE::default();
        tle.sat_num = sat_num;
        tle.inclination = inclination;
        tle.raan = raan;
        tle.eccen = eccen;
        tle.arg_of_perigee = argp;
        tle.mean_anomaly = ma;
        tle.mean_motion = mm;
        tle.bstar = bstar;
        tle.epoch = Instant::from_date(2000, 1, 1).unwrap() + Duration::from_days(epoch_days);

        let lines = tle.to_2line();
        prop_assert!(lines.is_ok(), "to_2line failed: {:?}", lines.err());
        let lines = lines.unwrap();
        let tle2 = TLE::load_2line(&lines[0], &lines[1]);
        prop_assert!(tle2.is_ok(), "load_2line failed: {:?}\n{}\n{}", tle2.err(), lines[0], lines[1]);
        let tle2 = tle2.unwrap();

        prop_assert_eq!(tle2.sat_num, tle.sat_num);
        prop_assert!((tle2.inclination - tle.inclination).abs() < 1e-4);
        prop_assert!((tle2.raan - tle.raan).abs() < 1e-4);
        prop_assert!((tle2.eccen - tle.eccen).abs() < 1e-7);
        prop_assert!((tle2.arg_of_perigee - tle.arg_of_perigee).abs() < 1e-4);
        prop_assert!((tle2.mean_anomaly - tle.mean_anomaly).abs() < 1e-4);
        prop_assert!((tle2.mean_motion - tle.mean_motion).abs() < 1e-7);
        // bstar is stored as a 5-digit mantissa with implied exponent
        let bstar_err = (tle2.bstar - tle.bstar).abs();
        prop_assert!(bstar_err <= 1e-4 * tle.bstar.abs().max(1e-9),
            "bstar round-trip error {:e} for {:e}", bstar_err, tle.bstar);
        // epoch stored as day-of-year with 8 decimal places (~1 ms)
        prop_assert!((tle2.epoch - tle.epoch).as_seconds().abs() < 1e-2);
    }

    /// The TLE parser never panics on arbitrary (printable or not) input
    /// lines — it must return Ok or Err, not abort. This is the cheap
    /// in-process cousin of a fuzz target.
    #[test]
    fn tle_parser_never_panics(
        l1 in ".{0,90}",
        l2 in ".{0,90}",
    ) {
        let _ = TLE::load_2line(&l1, &l2);
        let _ = TLE::from_lines(&[l1, l2]);
    }
}

// ───────────────────────── Quaternion ─────────────────────────

proptest! {
    /// Rotations are rigid: |q·v| = |v| for any unit-axis rotation, and
    /// conjugation undoes the rotation.
    #[test]
    fn quaternion_rigid_rotation(
        ax in -1.0..1.0f64,
        ay in -1.0..1.0f64,
        az in -1.0..1.0f64,
        angle in -TAU..TAU,
        vx in -1.0e6..1.0e6f64,
        vy in -1.0e6..1.0e6f64,
        vz in -1.0e6..1.0e6f64,
    ) {
        let axis = numeris::vector![ax, ay, az];
        prop_assume!(axis.norm() > 1e-3); // skip degenerate axis
        let axis = axis * (1.0 / axis.norm());
        let q = Quaternion::from_axis_angle(axis, angle);
        let v = numeris::vector![vx, vy, vz];

        let rotated = q * v;
        prop_assert!((rotated.norm() - v.norm()).abs() <= 1e-9 * v.norm().max(1.0),
            "norm not preserved: {:e}", (rotated.norm() - v.norm()).abs());
        let back = q.conjugate() * rotated;
        prop_assert!((back - v).norm() <= 1e-9 * v.norm().max(1.0),
            "conjugate did not undo rotation: {:e}", (back - v).norm());
    }

    /// from_euler ∘ to_euler round-trips within the principal ranges
    /// (pitch restricted away from the ±90° gimbal singularity).
    #[test]
    fn quaternion_euler_roundtrip(
        roll in -3.1..3.1f64,
        pitch in -1.5..1.5f64,
        yaw in -3.1..3.1f64,
    ) {
        let q = Quaternion::from_euler(roll, pitch, yaw);
        let (r2, p2, y2) = q.to_euler();
        prop_assert!((r2 - roll).abs() < 1e-9, "roll error {:e}", (r2 - roll).abs());
        prop_assert!((p2 - pitch).abs() < 1e-9, "pitch error {:e}", (p2 - pitch).abs());
        prop_assert!((y2 - yaw).abs() < 1e-9, "yaw error {:e}", (y2 - yaw).abs());
    }
}
