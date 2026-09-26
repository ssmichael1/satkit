//! Property-based tests (proptest).
//!
//! Each test asserts an invariant over a *domain* of randomly generated
//! inputs rather than hand-picked examples. proptest over-weights boundary
//! values (0, endpoints), automatically shrinks any failure to a minimal
//! counterexample. proptest's own failure file
//! (`tests/properties.proptest-regressions`) is gitignored: a real
//! counterexample is pinned as a named `regression_*` test at the bottom of
//! this file instead, so it replays forever with an explanation attached.
//!
//! Everything here is deliberately independent of the external data files
//! (no EOP / space weather / ephemerides), so the suite runs anywhere
//! `cargo test` does. UT1 conversions and frame transforms need EOP data and
//! live in `tests/properties_data.rs`.
//!
//! # Case counts
//!
//! Properties run proptest's default 256 cases, except the cheap
//! edge-biased time properties, which run `TIME_CASES` (1024); the whole
//! file takes well under a second once built. Set `PROPTEST_CASES` to
//! override every count for a deeper run, e.g.
//! `PROPTEST_CASES=100000 cargo test --test properties` (about 2 s; the
//! weekly `deep-proptest` job in `.github/workflows/audit.yml` runs this).
//!
//! # Time properties and why they look the way they do
//!
//! The time-scale defects fixed in #217 were all invisible to the original
//! round-trip properties, for two reasons: an inverse pair shares a
//! symmetric bug (a wrong TDB − TT period cancels in MJD → Instant → MJD),
//! and uniform sampling over 1972–2045 essentially never lands near one of
//! the 28 leap-second boundaries or before 1970. The time sections below
//! therefore (a) draw from the edge-biased generators in `time_edges/`, and
//! (b) prefer properties that reach the same answer by *different routes*
//! (calendar components vs strings vs MJD vs unixtime) or check a physical
//! invariant against an independent source (day lengths from IERS
//! Bulletin C, the published TDB − TT series) rather than a self-inverse.

mod time_edges;

use proptest::prelude::*;
use time_edges::*;

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
        e in 0.0..0.95f64,
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

    /// The TLE parser never panics — it must return Ok or Err, not abort.
    /// This is the cheap in-process cousin of a fuzz target.
    ///
    /// Arbitrary strings almost never get past the length and ASCII checks
    /// (the previous `.{0,90}` strategy reached the field parsers in 0 of
    /// 100k cases), so the inputs here are printable-ASCII lines of data-line
    /// length and, mostly, real element sets with a few edits applied
    /// (`tle_line_pair`), which is what reaches the field parsers, the
    /// checksum check and the located-error paths of `TLE::records`.
    #[test]
    fn tle_parser_never_panics((l1, l2) in tle_line_pair()) {
        exercise_tle_parsers(&l1, &l2);
    }
}

/// Real element sets (valid checksums) the mutation strategy starts from:
/// a 5-digit and a space-padded catalogue number, a line 2 with trailing
/// content past column 69, an alpha-5 number and a negative-exponent
/// B*.
const SEED_TLES: &[(&str, &str)] = &[
    (
        "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
        "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615",
    ),
    (
        "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992",
        "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065",
    ),
    (
        "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
        "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.",
    ),
    (
        "1 A0000U 20037E   24323.73967089 -.00003818  12345-5 -31595-3 0  9993",
        "2 A0000  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239080",
    ),
];

/// One edit to a TLE line. Positions are taken modulo the line length.
#[derive(Clone, Debug)]
enum Edit {
    /// Overwrite one character
    Replace(usize, char),
    /// Insert a character (shifts the rest of the line right one column)
    Insert(usize, char),
    /// Delete a character (shifts the rest of the line left one column)
    Delete(usize),
    /// Swap two adjacent characters
    Swap(usize),
    /// Replace a digit with a different digit (a plausible typo that keeps
    /// the field parseable but breaks the checksum)
    Digit(usize, u8),
    /// Shift the line from here by n columns (spaces in, or characters out)
    Shift(usize, i8),
    /// Cut the line at this column
    Truncate(usize),
}

/// Characters an edit writes: the ones that carry meaning in a TLE field,
/// plus any printable ASCII and the occasional non-ASCII character.
fn tle_char() -> impl Strategy<Value = char> {
    prop_oneof![
        6 => prop::sample::select(
            "0123456789 +-.UCS".chars().collect::<Vec<_>>()
        ),
        3 => (0x20u8..0x7f).prop_map(char::from),
        1 => any::<char>(),
    ]
}

fn edit() -> impl Strategy<Value = Edit> {
    let pos = 0usize..90;
    prop_oneof![
        3 => (pos.clone(), tle_char()).prop_map(|(p, c)| Edit::Replace(p, c)),
        2 => (pos.clone(), tle_char()).prop_map(|(p, c)| Edit::Insert(p, c)),
        2 => pos.clone().prop_map(Edit::Delete),
        2 => pos.clone().prop_map(Edit::Swap),
        4 => (pos.clone(), 0u8..10).prop_map(|(p, d)| Edit::Digit(p, d)),
        1 => (pos.clone(), -3i8..=3).prop_map(|(p, n)| Edit::Shift(p, n)),
        1 => (60usize..75).prop_map(Edit::Truncate),
    ]
}

fn apply_edit(line: &str, e: &Edit) -> String {
    let mut c: Vec<char> = line.chars().collect();
    if c.is_empty() {
        return String::new();
    }
    let n = c.len();
    match *e {
        Edit::Replace(p, ch) => c[p % n] = ch,
        Edit::Insert(p, ch) => c.insert(p % (n + 1), ch),
        Edit::Delete(p) => {
            c.remove(p % n);
        }
        Edit::Swap(p) => {
            let p = p % n;
            if p + 1 < n {
                c.swap(p, p + 1);
            }
        }
        Edit::Digit(p, d) => {
            // The first digit at or after p (wrapping), so the edit lands in
            // a numeric field.
            if let Some(i) = (0..n).map(|k| (p + k) % n).find(|&i| c[i].is_ascii_digit()) {
                let old = c[i].to_digit(10).unwrap() as u8;
                let new = if d == old { (d + 1) % 10 } else { d };
                c[i] = char::from(b'0' + new);
            }
        }
        Edit::Shift(p, k) => {
            let p = p % n;
            if k > 0 {
                for _ in 0..k {
                    c.insert(p, ' ');
                }
            } else {
                for _ in 0..(-k) {
                    if p < c.len() {
                        c.remove(p);
                    }
                }
            }
        }
        Edit::Truncate(p) => c.truncate(p),
    }
    c.into_iter().collect()
}

/// A real TLE with 1–4 edits spread over its two lines.
fn mutated_tle() -> impl Strategy<Value = (String, String)> {
    (
        prop::sample::select(SEED_TLES),
        prop::collection::vec((any::<bool>(), edit()), 1..=4),
    )
        .prop_map(|((l1, l2), edits)| {
            let (mut l1, mut l2) = (l1.to_string(), l2.to_string());
            for (first, e) in &edits {
                if *first {
                    l1 = apply_edit(&l1, e);
                } else {
                    l2 = apply_edit(&l2, e);
                }
            }
            (l1, l2)
        })
}

/// Pairs of candidate TLE lines: mostly mutated real element sets, plus
/// random printable-ASCII lines of data-line length (with and without the
/// `"1 "` / `"2 "` prefix `TLE::records` groups on).
fn tle_line_pair() -> impl Strategy<Value = (String, String)> {
    prop_oneof![
        8 => mutated_tle(),
        1 => ("[ -~]{69,80}", "[ -~]{69,80}"),
        1 => ("1 [ -~]{67,78}", "2 [ -~]{67,78}"),
    ]
}

/// Run every TLE entry point over one candidate pair of lines; the callers
/// only care that none of them panics.
fn exercise_tle_parsers(l1: &str, l2: &str) -> Vec<satkit::tle::Result<TLE>> {
    let mut out = vec![TLE::load_2line(l1, l2), TLE::load_3line("0 FUZZ", l1, l2)];
    // The record reader: two-line, three-line (name line first) and with
    // checksum verification, whose located errors format the input.
    let lines = [l1, l2];
    let named = ["0 FUZZ", l1, l2, l1, l2];
    for rec in TLE::records(lines)
        .chain(TLE::records(named))
        .chain(TLE::records(named).check_checksums(true))
    {
        if let Err(e) = &rec {
            let _ = e.to_string();
        }
        out.push(rec);
    }
    let _ = TLE::from_lines(&[l1.to_string(), l2.to_string()]);
    for tle in out.iter().flatten() {
        // A TLE that parsed must also re-encode without panicking.
        let _ = tle.to_2line();
    }
    out
}

/// Guards the strategy itself: `tle_parser_never_panics` is only worth
/// running if its inputs reach past the length and ASCII checks. The
/// original `.{0,90}` strategy passed every run while never calling a field
/// parser. With a fixed seed, check that the generated pairs produce
/// successful parses, field-parse errors and checksum mismatches.
#[test]
fn tle_fuzz_strategy_reaches_field_parsers() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
    use satkit::tle::Error;

    // The seeds themselves are valid, checksums included.
    for (l1, l2) in SEED_TLES {
        for r in TLE::records([*l1, *l2]).check_checksums(true) {
            r.unwrap_or_else(|e| panic!("seed TLE does not parse: {e}\n{l1}\n{l2}"));
        }
    }

    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &[7; 32]),
    );
    let strat = tle_line_pair();
    let (mut ok, mut field, mut checksum, mut shape) = (0, 0, 0, 0);
    let n = 2000;
    for _ in 0..n {
        let (l1, l2) = strat.new_tree(&mut runner).unwrap().current();
        for r in exercise_tle_parsers(&l1, &l2) {
            let mut e = match r {
                Ok(_) => {
                    ok += 1;
                    continue;
                }
                Err(e) => e,
            };
            while let Error::Record { error, .. } = e {
                e = *error;
            }
            match e {
                Error::ChecksumMismatch { .. } => checksum += 1,
                Error::LineTooShort { .. }
                | Error::InvalidLineLengths { .. }
                | Error::NonAscii { .. } => shape += 1,
                _ => field += 1,
            }
        }
    }
    let total = ok + field + checksum + shape;
    eprintln!("TLE fuzz outcomes over {n} pairs: ok {ok}, field errors {field}, checksum {checksum}, length/ASCII {shape}");
    // Most results must come from past the up-front length/ASCII checks.
    assert!(ok * 10 > total, "too few successful parses: {ok}/{total}");
    assert!(
        field * 10 > total,
        "too few field-parse errors: {field}/{total}"
    );
    // (A mutated satellite number now fails the line 1 / line 2 match before
    // the checksum is checked, hence 1 in 40 rather than 1 in 20.)
    assert!(
        checksum * 40 > total,
        "too few checksum mismatches: {checksum}/{total}"
    );
    assert!(
        shape * 2 < total,
        "mostly rejected by the length/ASCII checks: {shape}/{total}"
    );
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

// ─────────────── Time: edge-biased generators (sanity) ───────────────

/// Cases for the (cheap) edge-biased time properties; `PROPTEST_CASES`
/// overrides.
const TIME_CASES: u32 = 1024;

proptest! {
    #![proptest_config(cases(TIME_CASES))]

    /// The independent calendar used by the generators round-trips, and
    /// agrees with satkit's Gregorian arithmetic on the date of every
    /// non-leap label (different algorithms, same answer). Covers 1900–2045
    /// including 1900 (not a leap year) and 2000 (a leap year).
    #[test]
    fn generator_calendar_matches_satkit(l in any_non_leap_label()) {
        prop_assert_eq!(civil_from_days(l.days()), (l.y, l.mo, l.d));
        let t = l.instant();
        let (y, mo, d, ..) = t.as_datetime();
        prop_assert_eq!((y, mo, d), (l.y, l.mo, l.d), "label {}", l.iso());
        // Unixtime day number from the independent calendar
        prop_assert_eq!(t.as_unixtime().div_euclid(86_400.0) as i64, l.days());
    }
}

// ─────────────── Time: construction paths agree ───────────────

/// `YYYY-MM-DD HH:MM:SS.ffffff` (space separator, no zone): not RFC 3339,
/// so `from_string` falls through to its tokenizer.
fn spaced(l: &Label) -> String {
    let (s, f) = l.sec_frac();
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:06}",
        l.y, l.mo, l.d, l.h, l.mi, s, f
    )
}

/// The local label `offset_min` minutes ahead of UTC label `l` (on the
/// 86400-s-per-day label axis, as RFC 3339 defines offsets), and its zone
/// suffix `±HH{sep}MM`.
fn offset_label(l: &Label, offset_min: i64, sep: &str) -> (Label, String) {
    let local = l.utc_basis_us() + offset_min * US_MIN;
    let ll = Label::from_day_tod(local.div_euclid(US_DAY), local.rem_euclid(US_DAY));
    let sign = if offset_min < 0 { '-' } else { '+' };
    let (h, m) = (offset_min.abs() / 60, offset_min.abs() % 60);
    (ll, format!("{sign}{h:02}{sep}{m:02}"))
}

/// `|a − b|` in microseconds.
fn diff_us(a: &Instant, b: &Instant) -> i64 {
    (*a - *b).as_microseconds().abs()
}

proptest! {
    #![proptest_config(cases(TIME_CASES))]

    /// Every way of entering the same UTC label yields the same instant:
    /// calendar components (`from_datetime`, `from_datetime_with_scale(UTC)`),
    /// RFC 3339 (`from_rfc3339`, `from_string`), the `from_string`
    /// tokenizer, `strptime`, and — where the label is representable —
    /// `from_mjd_with_scale(UTC)` and `from_unixtime`.
    ///
    /// Exclusions: unixtime and UTC MJD count 86400 s per day and cannot
    /// name `23:59:60.x`, so those two routes are skipped for leap labels.
    ///
    /// Tolerance: the reference is [`Label::instant`] (integer µs). The
    /// calendar and string routes are exact (float seconds are rounded to
    /// the nearest µs); the MJD / unixtime routes are allowed 1 µs, as an
    /// f64 MJD after 2038 resolves only ~1.3 µs, and from 2038 an f64 Unix
    /// time (≥ 2^31 s) resolves only ~0.5 µs, so from ~2041 (2^51 µs) about
    /// 10% of µs values come back 1 µs off.
    #[test]
    fn construction_paths_agree(l in any_label()) {
        let t = l.instant();
        let iso = l.iso();

        let dt = Instant::from_datetime(l.y, l.mo, l.d, l.h, l.mi, l.seconds_f64());
        prop_assert!(dt.is_ok(), "from_datetime rejected {iso}: {:?}", dt.err());
        let dt = dt.unwrap();
        prop_assert_eq!(dt, t, "from_datetime({}) = {}", iso, dt);

        let dts = Instant::from_datetime_with_scale(
            l.y, l.mo, l.d, l.h, l.mi, l.seconds_f64(), TimeScale::UTC,
        );
        prop_assert_eq!(dts.ok(), Some(dt), "from_datetime_with_scale(UTC) differs");

        for (route, s, parsed) in [
            ("from_rfc3339", iso.clone(), Instant::from_rfc3339(&iso)),
            ("from_string(rfc)", iso.clone(), Instant::from_string(&iso)),
            ("from_string(tokenizer)", spaced(&l), Instant::from_string(&spaced(&l))),
            (
                "strptime",
                spaced(&l),
                Instant::strptime(&spaced(&l), "%Y-%m-%d %H:%M:%S.%f"),
            ),
        ] {
            prop_assert!(parsed.is_ok(), "{route} rejected {s:?}: {:?}", parsed.err());
            let p = parsed.unwrap();
            prop_assert_eq!(p, t, "{}({:?}) = {} ≠ {}", route, s, p, t);
        }

        if !l.is_leap() {
            let basis = l.utc_basis_us();
            let unix = Instant::from_unixtime(basis as f64 * 1.0e-6);
            prop_assert!(diff_us(&unix, &t) <= 1, "from_unixtime → {unix} ≠ {t}");
            prop_assert_eq!(Instant::from_unixtime_microseconds(basis), t);
            prop_assert_eq!(t.as_unixtime_microseconds(), basis);
            // MJD epoch 1858-11-17 is 40587 days before the Unix epoch
            let mjd = (basis as f64 / US_DAY as f64) + 40_587.0;
            let m = Instant::from_mjd_with_scale(mjd, TimeScale::UTC);
            prop_assert!(diff_us(&m, &t) <= 1, "from_mjd_utc({mjd}) → {m} ≠ {t}");
        }
    }

    /// An RFC 3339 string with a UTC offset names the same instant as the
    /// equivalent `Z` string, exactly. The offset label is built
    /// independently (UTC label + offset on the 86400-s-per-day label
    /// axis), as RFC 3339 defines it, so an offset that spans a leap second
    /// is included (see `regression_rfc3339_offset_across_leap_second`).
    #[test]
    fn rfc3339_offset_matches_utc(
        l in any_non_leap_label(),
        offset_min in -14 * 60..=14 * 60i64,
    ) {
        let (ll, z) = offset_label(&l, offset_min, ":");
        let s = format!("{}{z}", ll.iso().trim_end_matches('Z'));
        let p = Instant::from_rfc3339(&s);
        prop_assert!(p.is_ok(), "rejected {s:?}: {:?}", p.err());
        let p = p.unwrap();
        prop_assert_eq!(p, l.instant(), "{:?} → {}, expected {}", s, p, l.iso());
    }

    /// `strptime`'s `%z` agrees with `from_rfc3339`'s offset handling (and
    /// hence with the `Z` label): `+HHMM` means local time is ahead of UTC.
    /// Exact, leap-second spans included.
    #[test]
    fn strptime_z_offset_sign(
        l in any_non_leap_label(),
        offset_min in -14 * 60..=14 * 60i64,
    ) {
        prop_assume!(offset_min != 0);
        let (ll, z) = offset_label(&l, offset_min, "");
        let s = format!("{}{z}", spaced(&ll));
        let p = Instant::strptime(&s, "%Y-%m-%d %H:%M:%S.%f%z");
        prop_assert!(p.is_ok(), "rejected {s:?}: {:?}", p.err());
        let p = p.unwrap();
        prop_assert_eq!(p, l.instant(), "{:?} → {}, expected {}", s, p, l.iso());
    }

    /// Passing seconds as `f64` loses no microseconds: `from_datetime` with
    /// `second = us · 1e-6` and `from_rfc3339` of a 6-digit fraction give
    /// exactly the label's instant, and `from_rfc3339(t.to_string())` is
    /// exactly `t`.
    #[test]
    fn from_datetime_float_seconds_exact(l in any_label()) {
        let t = l.instant();
        let dt = Instant::from_datetime(l.y, l.mo, l.d, l.h, l.mi, l.seconds_f64()).unwrap();
        prop_assert_eq!(dt, t, "from_datetime({}) off by {} µs", l.iso(), (dt - t).as_microseconds());
        let (y, mo, d, h, mi, s) = t.as_datetime();
        let back = Instant::from_datetime(y, mo, d, h, mi, s).unwrap();
        prop_assert_eq!(back, t, "from_datetime(as_datetime({}))", l.iso());
        let p = Instant::from_rfc3339(&t.to_string()).unwrap();
        prop_assert_eq!(p, t, "from_rfc3339({})", t);
    }
}

// ─────────────── Time: calendar round-trips and formatting ───────────────

proptest! {
    #![proptest_config(cases(TIME_CASES))]

    /// Calendar label → instant → calendar label is the identity over the
    /// edge-biased domain (pre-1970, leap seconds, `23:59:60.x`), the
    /// fields are in range, and the formatted string (`Display`,
    /// `as_iso8601`, `as_rfc3339`) is exactly the label and re-parses to the
    /// instant.
    ///
    /// Range: `0 ≤ h < 24`, `0 ≤ m < 60`, `0 ≤ s < 60` except
    /// `s < 60 + inserted` in the last minute of a day that ends in inserted
    /// time: 61 for a leap second, 60.1 / 60.107758 for the pre-1972 steps,
    /// 61.422818 on 1960-12-31.
    #[test]
    fn calendar_roundtrip_and_format(l in any_label()) {
        let t = l.instant();
        let (y, mo, d, h, mi, s) = t.as_datetime();
        prop_assert_eq!((y, mo, d, h, mi), (l.y, l.mo, l.d, l.h, l.mi), "label {}", l.iso());
        prop_assert_eq!((s * 1.0e6).round() as i64, l.us, "seconds field for {}", l.iso());

        prop_assert!((0..24).contains(&h) && (0..60).contains(&mi), "{h}:{mi}");
        let ins = inserted_us(l.days());
        let s_max = if ins > 0 && (h, mi) == (23, 59) {
            60.0 + ins as f64 * 1.0e-6
        } else {
            60.0
        };
        prop_assert!((0.0..s_max).contains(&s), "second {s} out of [0, {s_max}) for {}", l.iso());

        let iso = l.iso();
        prop_assert_eq!(t.to_string(), iso.clone());
        prop_assert_eq!(t.as_iso8601(), iso.clone());
        prop_assert_eq!(t.as_rfc3339(), iso.clone());
        let p = Instant::from_rfc3339(&t.to_string()).unwrap();
        prop_assert_eq!(p, t, "re-parse of {}", iso);
    }

    /// The ISO label is monotonic: a later instant never gets a
    /// lexicographically smaller label (years 1900–2045 are all four
    /// digits, so string order is time order). Checked for nearby pairs
    /// (which cross the leap-second boundaries the generator targets) and
    /// for arbitrary pairs.
    #[test]
    fn iso_label_monotonic(
        a in any_instant(),
        b in any_instant(),
        dt_us in 1..20_000_000i64,
    ) {
        // In the rubber-second era a UTC microsecond is 1 + (1.3 … 3.0)e-8
        // SI microseconds, so about one TAI microsecond in 3e7 shares its
        // neighbour's label: equal labels are allowed 1 µs apart only
        let a2 = a + Duration::from_microseconds(dt_us);
        let (s, s2) = (a.to_string(), a2.to_string());
        prop_assert!(s < s2 || (dt_us == 1 && s == s2), "{a} !< {a2} (dt {dt_us} µs)");
        let (sa, sb) = (a.to_string(), b.to_string());
        if sa == sb {
            prop_assert!(diff_us(&a, &b) <= 1, "{} vs {}", a, b);
        } else {
            prop_assert_eq!(a.cmp(&b), sa.cmp(&sb), "{} vs {}", a, b);
        }
    }
}

/// Deterministic sweep of *every* leap second and positive pre-1972 step:
/// from 3 s before the inserted interval to 3 s after it, in 250 ms steps
/// plus the microseconds either side of each boundary, the formatted label
/// is exactly the independently computed one and strictly increasing.
///
/// Offsets are SI microseconds from the start of the inserted interval. In
/// the rubber-second era the labels either side advance by 1 − (1.3 … 3.0)e-8
/// UTC seconds per SI second, i.e. < 0.1 µs over 3 s, below the label's
/// rounding.
#[test]
fn iso_label_monotonic_at_every_leap_second() {
    for (i, &(y, mo, d, ins_us)) in LEAP_DAYS.iter().enumerate() {
        let start = leap_label(i, 0).instant();
        let mut offsets: Vec<i64> = (-12..=(4 * ins_us / US + 12)).map(|k| k * US / 4).collect();
        offsets.extend([-1, 0, 1, ins_us - 1, ins_us, ins_us + 1]);
        offsets.sort_unstable();
        offsets.dedup();
        let mut prev = String::new();
        for off in offsets {
            let t = start + Duration::from_microseconds(off);
            let s = t.to_string();
            assert_eq!(
                s,
                leap_label(i, off).iso(),
                "leap day {y}-{mo}-{d} offset {off} µs"
            );
            assert!(prev < s, "labels not increasing: {prev} then {s}");
            prev = s;
        }
    }
}

// ─────────────── Time: physical invariants ───────────────

/// UTC day length, exhaustively for 1900–2050: `00:00` of the next day
/// minus `00:00` of this day is 86400 s plus the change in TAI − UTC: the
/// inserted second on the 27 leap-second days (86401 s), and before 1972
/// the day's drift (1.1–2.6 ms) plus any step (+0.1 s, −0.05 s, −0.1 s,
/// +0.107758 s on 1971-12-31, and satkit's +1.422818 s on 1960-12-31). The
/// expected lengths come from IERS Bulletin C and USNO `tai-utc.dat`
/// (`tai_minus_utc_us`), not from satkit. `add_utc_days(1.0)` agrees with
/// the calendar route.
#[test]
fn utc_day_length() {
    let first = days_from_civil(1900, 1, 1);
    let last = days_from_civil(2051, 1, 1);
    let mut t0 = Label::from_day_tod(first, 0).instant();
    for day in first..last {
        let t1 = Label::from_day_tod(day + 1, 0).instant();
        let (y, m, d) = civil_from_days(day);
        let expected =
            US_DAY + tai_minus_utc_us((day + 1) * US_DAY) - tai_minus_utc_us(day * US_DAY);
        assert_eq!(
            (t1 - t0).as_microseconds(),
            expected,
            "length of {y:04}-{m:02}-{d:02}"
        );
        assert_eq!(
            t0.add_utc_days(1.0),
            t1,
            "add_utc_days(1) from {y:04}-{m:02}-{d:02}"
        );
        t0 = t1;
    }
}

proptest! {
    #![proptest_config(cases(TIME_CASES))]

    /// `add_utc_days(n)` keeps the time-of-day label and moves the date by
    /// `n` calendar days, whatever leap seconds lie in between. Exact: a
    /// whole number of days is an integer count of µs on the UTC basis.
    #[test]
    fn add_utc_days_keeps_label(
        l in any_non_leap_label(),
        n in -3000..3000i64,
    ) {
        let expected = Label::from_day_tod(l.days() + n, l.utc_basis_us().rem_euclid(US_DAY));
        let got = l.instant().add_utc_days(n as f64);
        prop_assert_eq!(
            got, expected.instant(),
            "{} + {} d → {}, expected {}", l.iso(), n, got, expected.iso()
        );
    }

    /// Elapsed SI time from `23:59:00` on a leap-second day to any label
    /// near the boundary is `60 s + offset`: the inserted second(s) are
    /// contiguous with the minute before and the day after. Catches both a
    /// midnight that lands on the start of the leap second instead of its
    /// end and a table entry keyed to the wrong second.
    ///
    /// Before 1972 the minute's drift of TAI − UTC (≤ 1.8 µs) is added from
    /// the independent table, and 1 µs is allowed for the rounding of TAI −
    /// UTC and of the fraction (added as an SI `Duration` to a UTC second).
    #[test]
    fn elapsed_time_across_leap_second((i, off) in leap_offset()) {
        let (y, mo, d, ins) = LEAP_DAYS[i];
        let base_label = Label::new(y, mo, d, 23, 59, 0);
        let base = base_label.instant();
        let l = leap_label(i, off);
        let t = Instant::from_datetime(l.y, l.mo, l.d, l.h, l.mi, (l.us / US) as f64).unwrap()
            + Duration::from_microseconds(l.us % US);
        // Independent TAI count of a label (relative to the label axis): the
        // UTC-basis count plus TAI − UTC, and inside the inserted interval
        // the old offset carried to midnight plus the offset into it
        let tai = |l: &Label| {
            if l.is_leap() {
                let midnight = (l.days() + 1) * US_DAY;
                midnight + tai_minus_utc_us(midnight) - ins + (l.us - US_MIN)
            } else {
                l.utc_basis_us() + tai_minus_utc_us(l.utc_basis_us())
            }
        };
        let expected = tai(&l) - tai(&base_label);
        let tol = if in_rubber_era(base_label.days()) { 1 } else { 0 };
        prop_assert!(
            ((t - base).as_microseconds() - expected).abs() <= tol,
            "label {}: {} µs, expected {expected}", l.iso(), (t - base).as_microseconds()
        );
        if tol == 0 {
            prop_assert_eq!(expected, US_MIN + off);
        }
    }

    /// TT − TAI = 32.184 s and TAI − GPS = 19 s exactly, for every instant
    /// and in both directions: the same MJD read in two scales differs by
    /// the offset (integer µs), and an instant read back in two scales
    /// differs by it to f64 MJD resolution.
    #[test]
    fn fixed_scale_offsets(mjd in 15_020.0..69_807.0f64, t in any_instant()) {
        let tai = Instant::from_mjd_with_scale(mjd, TimeScale::TAI);
        let tt = Instant::from_mjd_with_scale(mjd, TimeScale::TT);
        let gps = Instant::from_mjd_with_scale(mjd, TimeScale::GPS);
        prop_assert_eq!((tai - tt).as_microseconds(), 32_184_000);
        prop_assert_eq!((gps - tai).as_microseconds(), 19_000_000);

        let m = |s| t.as_mjd_with_scale(s) * 86_400.0;
        prop_assert!((m(TimeScale::TT) - m(TimeScale::TAI) - 32.184).abs() < 2e-6);
        prop_assert!((m(TimeScale::TAI) - m(TimeScale::GPS) - 19.0).abs() < 2e-6);
    }

    /// GPS week/second, GPS MJD and the calendar all agree: the GPS epoch is
    /// 1980-01-06T00:00:00 UTC (when TAI − UTC was 19 s, so GPS = UTC), and
    /// a GPS week is exactly 604800 SI seconds.
    #[test]
    fn gps_week_routes_agree(week in 0..3000i32, sow_us in 0..604_800_000_000i64) {
        prop_assert_eq!(Instant::GPS_EPOCH, Instant::from_date(1980, 1, 6).unwrap());
        let sow = sow_us as f64 * 1.0e-6;
        let t = Instant::from_gps_week_and_second(week, sow);
        let direct = Instant::GPS_EPOCH
            + Duration::from_microseconds(week as i64 * 604_800_000_000 + sow_us);
        prop_assert_eq!(t, direct, "week {} sow {}", week, sow);
        let mjd = 44_244.0 + 7.0 * week as f64 + sow / 86_400.0;
        let m = Instant::from_mjd_with_scale(mjd, TimeScale::GPS);
        prop_assert!(diff_us(&m, &direct) <= 1, "GPS MJD {mjd}: {m} vs {direct}");
    }

    /// TAI − UTC matches IERS Bulletin C and USNO `tai-utc.dat`: 0 before
    /// 1961 (satkit's convention), the drifting rubber-second value to 1971,
    /// then an integer number of seconds equal to 10 + the number of leap
    /// seconds so far. Read two ways: TAI count minus unixtime (integer µs),
    /// and the TAI and UTC MJDs (f64). Labels inside a leap second are
    /// excluded (UTC MJD / unixtime cannot name them).
    #[test]
    fn tai_minus_utc_matches_table(l in any_non_leap_label()) {
        let t = l.instant();
        let expected = tai_minus_utc_us(l.utc_basis_us());
        let tai_1970 = Instant::from_mjd_with_scale(40_587.0, TimeScale::TAI);
        let raw_minus_unix = (t - tai_1970).as_microseconds()
            - (t.as_unixtime() * 1.0e6).round() as i64;
        prop_assert_eq!(raw_minus_unix, expected, "{}", l.iso());
        let via_mjd =
            (t.as_mjd_with_scale(TimeScale::TAI) - t.as_mjd_with_scale(TimeScale::UTC)) * 86_400.0;
        prop_assert!((via_mjd - expected as f64 * 1.0e-6).abs() < 2e-6, "{}: {via_mjd}", l.iso());
    }

    /// TAI − UTC never decreases (UTC never repeats a label), except at the
    /// two negative pre-1972 steps, where it drops by the removed time.
    #[test]
    fn tai_minus_utc_nondecreasing(a in any_instant(), b in any_instant()) {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let tai_1970 = Instant::from_mjd_with_scale(40_587.0, TimeScale::TAI);
        let off = |t: Instant| (t - tai_1970).as_seconds() - t.as_unixtime();
        let day = |t: Instant| t.as_unixtime().div_euclid(86_400.0) as i64;
        let removed: i64 = REMOVED_DAYS
            .iter()
            .filter(|(y, m, d, _)| (day(a)..day(b)).contains(&days_from_civil(*y, *m, *d)))
            .map(|e| e.3)
            .sum();
        prop_assert!(
            off(a) <= off(b) + removed as f64 * 1.0e-6 + 1e-6,
            "{a}: {} > {b}: {}", off(a), off(b)
        );
    }

    /// Instant ± Duration is exact at microsecond resolution across leap
    /// seconds and before 1970: `(t + d) − t == d` and `t + d − d == t`.
    /// `Duration::from_seconds` of the same span in `f64` seconds is exactly
    /// `d`: it rounds to the nearest µs, and below 2^31 s (here ≤ 1.5e9 s)
    /// the f64 seconds and their µs product are both within 0.125 µs.
    #[test]
    fn duration_arithmetic_exact_across_leap_seconds(
        t in any_instant(),
        d_us in prop_oneof![
            -20 * US..20 * US,
            -1_000_000 * US..1_000_000 * US,
            -1_500_000_000 * US..1_500_000_000 * US,
        ],
    ) {
        let d = Duration::from_microseconds(d_us);
        prop_assert_eq!(((t + d) - t).as_microseconds(), d_us);
        prop_assert_eq!(t + d - d, t);
        prop_assert_eq!(Duration::from_seconds(d_us as f64 * 1e-6), d);
    }
}

// ─────────────── Time: TDB − TT ───────────────

/// Mean period of the TDB − TT series satkit implements (Vallado Eq. 3-50,
/// argument `628.3076 T + 6.2401` rad, `T` in Julian centuries of TT):
/// 2π / 628.3076 centuries ≈ 365.256 days.
const TDB_PERIOD_S: f64 = std::f64::consts::TAU / 628.3076 * 36_525.0 * 86_400.0;

fn tdb_minus_tt(t: &Instant) -> f64 {
    (t.as_mjd_with_scale(TimeScale::TDB) - t.as_mjd_with_scale(TimeScale::TT)) * 86_400.0
}

proptest! {
    #![proptest_config(cases(TIME_CASES))]

    /// TDB − TT is bounded (|x| < 1.7 ms), periodic with one (anomalistic)
    /// year, and actually swings through its ±1.657 ms amplitude within
    /// that year. Also matches the independent USNO approximation
    /// `0.001657 sin g + 0.000014 sin 2g`, g = 357.53° + 0.98560028° d,
    /// to 4e-5 s (satkit omits the 2g term and uses a slightly different
    /// rate; the difference over ±1 century is ~1e-5 s). A wrong period
    /// (the #217 `PI/180` bug stretched it to ~57 years) fails all four.
    #[test]
    fn tdb_minus_tt_bounded_and_annual(t in uniform_label(1900, 2100).prop_map(|l| l.instant())) {
        let x = tdb_minus_tt(&t);
        prop_assert!(x.abs() < 1.7e-3, "TDB − TT = {x} s at {t}");

        let t_next = t + Duration::from_seconds(TDB_PERIOD_S);
        let x_next = tdb_minus_tt(&t_next);
        prop_assert!((x - x_next).abs() < 1e-5, "not periodic: {x} at {t}, {x_next} a period later");

        let samples: Vec<f64> = (0..16)
            .map(|k| tdb_minus_tt(&(t + Duration::from_seconds(TDB_PERIOD_S * k as f64 / 16.0))))
            .collect();
        let span = samples.iter().cloned().fold(f64::MIN, f64::max)
            - samples.iter().cloned().fold(f64::MAX, f64::min);
        prop_assert!(span > 3.0e-3, "TDB − TT spans only {span} s over one year from {t}");

        let d = t.as_jd_with_scale(TimeScale::TT) - 2_451_545.0;
        let g = (357.53 + 0.985_600_28 * d).to_radians();
        let usno = 0.001_657 * g.sin() + 0.000_014 * (2.0 * g).sin();
        prop_assert!((x - usno).abs() < 4e-5, "TDB − TT {x} vs USNO {usno} at {t}");
    }

    /// TDB MJD → Instant → TDB MJD round-trips (the inverse evaluates the
    /// periodic term at TDB instead of TT; the difference is ~1e-13 s).
    #[test]
    fn tdb_inverse(t in any_instant()) {
        let m = t.as_mjd_with_scale(TimeScale::TDB);
        let back = Instant::from_mjd_with_scale(m, TimeScale::TDB);
        prop_assert!(diff_us(&back, &t) <= 2, "{t} → TDB {m} → {back}");
    }
}

// ───────────────────── Pinned regressions ─────────────────────

/// CI-found counterexample (2026-07-03, ubuntu runner): high-eccentricity
/// one-period closure failed because `mean2eccentric` received an unwrapped
/// mean anomaly (M + 2π); the naive `E₀ = M ± e` Newton guess turned chaotic
/// at e ≈ 0.88 and exhausted the iteration cap. Fixed via range reduction +
/// Danby's initial guess. Pinned here so it can never regress.
#[test]
fn regression_high_eccen_period_closure() {
    let k = Kepler::new(
        11568493.745532092,
        0.8828701159267661,
        0.6,
        1.0,
        2.0,
        Anomaly::True(2.879249859070718),
    );
    let k2 = k.propagate(&Duration::from_seconds(k.period()));
    let (r, _) = k.to_pv();
    let (r2, _) = k2.to_pv();
    let rel = (r - r2).norm() / r.norm();
    assert!(rel < 1e-6, "one-period closure error {rel:e}");
}

// Counterexamples for defects found by the edge-biased time properties
// (2026-09-25), fixed in the same release; kept as regression tests.

/// `from_datetime` truncates `second * 1e6` instead of rounding
/// (src/time/instant.rs:780, and :769 for `23:59:60.x`). Microsecond
/// values are rarely exact in `f64`: `0.000249 * 1e6 = 248.99999999999997`
/// → 248 µs. About 1.2% of 6-digit fractions written as decimal literals
/// or RFC 3339 strings (`from_rfc3339` builds `S + f / 1e6`), and about 29%
/// of the seconds values `as_datetime()` returns, come back 1 µs early.
/// proptest's minimal counterexample: seconds `45773591 as f64 * 1e-6`
/// on 1972-01-01T05:30.
#[test]
fn regression_from_datetime_truncates_microseconds() {
    let midnight = Instant::from_date(2024, 1, 1).unwrap();
    let t = Instant::from_datetime(2024, 1, 1, 0, 0, 0.000249).unwrap();
    assert_eq!(
        (t - midnight).as_microseconds(),
        249,
        "from_datetime(…, 0.000249)"
    );

    let s = "2024-01-01T00:00:00.000249Z";
    assert_eq!(
        Instant::from_rfc3339(s).unwrap().to_string(),
        s,
        "from_rfc3339"
    );

    let exact = Instant::from_datetime(1972, 1, 1, 5, 30, 45.0).unwrap()
        + Duration::from_microseconds(773_591);
    let (y, mo, d, h, mi, sec) = exact.as_datetime();
    let back = Instant::from_datetime(y, mo, d, h, mi, sec).unwrap();
    assert_eq!(
        back,
        exact,
        "from_datetime(as_datetime(t)) off by {} µs",
        (back - exact).as_microseconds()
    );
}

/// `strptime`'s `%z` adds the UTC offset instead of subtracting it
/// (src/time/instantparse.rs:339): `+HHMM` means local time is *ahead* of
/// UTC, so `12:00+0100` is `11:00Z`. proptest's minimal counterexample:
/// `1972-01-02 00:01:00.000000+0001` → `00:02Z` instead of `00:00Z`.
#[test]
fn regression_strptime_z_offset_sign() {
    let t = Instant::strptime("2024-01-01T12:00:00+0100", "%Y-%m-%dT%H:%M:%S%z").unwrap();
    assert_eq!(
        t,
        Instant::from_datetime(2024, 1, 1, 11, 0, 0.0).unwrap(),
        "got {t}"
    );
}

/// `from_rfc3339` applies a UTC offset as an SI-seconds `Duration`
/// (src/time/instantparse.rs:390), so an offset that spans a leap second
/// is 1 s off. RFC 3339 offsets act on labels:
/// `2017-01-01T00:30:00+01:00` is `2016-12-31T23:30:00Z`.
#[test]
fn regression_rfc3339_offset_across_leap_second() {
    let p = Instant::from_rfc3339("2017-01-01T00:30:00+01:00").unwrap();
    let expected = Instant::from_datetime(2016, 12, 31, 23, 30, 0.0).unwrap();
    assert_eq!(p, expected, "got {p}");
}
