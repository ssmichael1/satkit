//! UTC from 1961-01-01 to 1972-01-01: the "rubber second" era.
//!
//! Before 1972, UTC was steered to UT2 by changing the length of the UTC
//! second (a frequency offset, so TAI − UTC drifted linearly) and by
//! occasional fractional-second steps. TAI − UTC is piecewise linear in the
//! UTC Modified Julian Date:
//!
//! ```text
//! TAI − UTC = A + (MJD_UTC − MJD₀) × rate      [seconds]
//! ```
//!
//! with the segments below, each starting at 00:00:00 UTC on the first of a
//! month and ending where the next starts. From 1972-01-01 TAI − UTC is a
//! whole number of seconds (the leap-second table in `instant.rs`).
//!
//! # Sources
//!
//! * USNO, `tai-utc.dat` (<https://maia.usno.navy.mil/ser7/tai-utc.dat>), the
//!   1961-01-01 to 1968-02-01 lines.
//! * IERS / SOFA / ERFA `dat` (the `drift` and `changes` arrays of `dat.c`),
//!   which carries the same coefficients.
//! * Urban & Seidelmann (eds.), *Explanatory Supplement to the Astronomical
//!   Almanac*, 3rd ed. (2013), chapter on time, for the definition of
//!   pre-1972 UTC.
//!
//! Every coefficient was checked against `pyerfa` (`erfa.dat`) at each
//! segment's start, end and midpoint and at 20,000 random dates in
//! 1961–1971; the largest difference was 2e-15 s.
//!
//! # Before 1961
//!
//! satkit models UTC from 1961-01-01, the first line of USNO's `tai-utc.dat`.
//! Earlier labels are TAI-aligned: TAI − UTC = 0, as before this model was
//! added. This deliberately differs from ERFA, whose `dat` also has a 1960
//! entry (1.4178180 s + (MJD − 37300) × 0.001296 s) and flags years before
//! 1960 as dubious. At 1961-01-01 TAI − UTC therefore steps from 0 to
//! 1.422818 s; the step is an inserted interval like the others (below).
//!
//! # Steps
//!
//! At a segment boundary `B` (00:00:00 UTC), TAI − UTC changes from the old
//! segment's value at `B` to the new one's:
//!
//! * **Positive step** (1961-01-01 +1.422818 s; +0.1 s on 1963-11-01,
//!   1964-04-01, 1964-09-01, 1965-01-01, 1965-03-01, 1965-07-01, 1965-09-01;
//!   +0.107758 s on 1972-01-01): time is inserted. The inserted interval is
//!   labelled `23:59:60.x` on the preceding day, from `23:59:60.0` to
//!   `23:59:60 + step`, exactly like a post-1972 leap second (e.g.
//!   `1963-10-31T23:59:60.05`). ERFA's `dtf2d` / `utctai` do the same
//!   (they stretch the day by the step). ERFA's `d2dtf` does not: it only
//!   treats a step over 0.5 s as a leap second, so on the days ending in a
//!   fractional step it prints labels up to `|step|` off from those its own
//!   `dtf2d` accepts. satkit follows `dtf2d` / `utctai`.
//! * **Negative step** (1961-08-01 −0.05 s, 1968-02-01 −0.1 s): time is
//!   removed, so the UTC labels in the last `|step|` seconds of the preceding
//!   day (e.g. `1961-07-31T23:59:59.95` to `23:59:59.999999`) never occurred.
//!   Constructing such a label is not an error: as in ERFA (`dtf2d` accepts
//!   it with the warning "time is after end of day", then `utctai`), TAI −
//!   UTC is taken from the preceding day's segment, so the label lands on the
//!   same instants as the first `|step|` seconds of the next day. Converting
//!   back gives the next-day label, so these labels do not round-trip.
//! * **Zero step** (1962-01-01, 1964-01-01, 1966-01-01): only the rate
//!   changes; TAI − UTC is continuous.
//!
//! # Precision
//!
//! TAI − UTC is evaluated in integer arithmetic and rounded to the nearest
//! microsecond (the rates are exact multiples of 0.1 µs per day). A UTC label
//! on the microsecond grid converts to the internal TAI count and back
//! exactly; because a pre-1972 UTC second is slightly longer than an SI
//! second, about one TAI microsecond in (3–8)×10⁷ has no UTC label of its
//! own and reads back as its neighbour.

/// MJD of 1970-01-01, the origin of the UTC-basis microsecond count
const MJD_UNIX: i64 = 40587;

/// Microseconds per day
const DAY_US: i64 = 86_400_000_000;

/// Rates are stored in units of 0.1 µs per day; this converts a UTC-basis
/// microsecond span times such a rate into microseconds.
const RATE_DEN: i128 = 10 * DAY_US as i128;

/// First day of the model: 1961-01-01
pub(super) const START_MJD: i64 = 37300;

/// First day of the integer leap-second table: 1972-01-01
pub(super) const END_MJD: i64 = 41317;

/// One segment of TAI − UTC, in force from 00:00:00 UTC on `start_mjd` until
/// the next segment's start: `TAI − UTC = offset + (MJD_UTC − ref_mjd) × rate`.
struct Segment {
    start_mjd: i64,
    /// `A`, in microseconds
    offset_us: i64,
    /// `MJD₀`
    ref_mjd: i64,
    /// Drift, in units of 0.1 µs per day (0.001296 s/day = 12960)
    rate: i64,
}

/// USNO `tai-utc.dat` / ERFA `dat`, 1961-01-01 through 1968-02-01
const SEGMENTS: [Segment; 13] = [
    seg(37300, 1_422_818, 37300, 12_960), // 1961-01-01  1.4228180 + (MJD − 37300) × 0.001296
    seg(37512, 1_372_818, 37300, 12_960), // 1961-08-01  1.3728180 + (MJD − 37300) × 0.001296
    seg(37665, 1_845_858, 37665, 11_232), // 1962-01-01  1.8458580 + (MJD − 37665) × 0.0011232
    seg(38334, 1_945_858, 37665, 11_232), // 1963-11-01  1.9458580 + (MJD − 37665) × 0.0011232
    seg(38395, 3_240_130, 38761, 12_960), // 1964-01-01  3.2401300 + (MJD − 38761) × 0.001296
    seg(38486, 3_340_130, 38761, 12_960), // 1964-04-01  3.3401300 + (MJD − 38761) × 0.001296
    seg(38639, 3_440_130, 38761, 12_960), // 1964-09-01  3.4401300 + (MJD − 38761) × 0.001296
    seg(38761, 3_540_130, 38761, 12_960), // 1965-01-01  3.5401300 + (MJD − 38761) × 0.001296
    seg(38820, 3_640_130, 38761, 12_960), // 1965-03-01  3.6401300 + (MJD − 38761) × 0.001296
    seg(38942, 3_740_130, 38761, 12_960), // 1965-07-01  3.7401300 + (MJD − 38761) × 0.001296
    seg(39004, 3_840_130, 38761, 12_960), // 1965-09-01  3.8401300 + (MJD − 38761) × 0.001296
    seg(39126, 4_313_170, 39126, 25_920), // 1966-01-01  4.3131700 + (MJD − 39126) × 0.002592
    seg(39887, 4_213_170, 39126, 25_920), // 1968-02-01  4.2131700 + (MJD − 39126) × 0.002592
];

const fn seg(start_mjd: i64, offset_us: i64, ref_mjd: i64, rate: i64) -> Segment {
    Segment {
        start_mjd,
        offset_us,
        ref_mjd,
        rate,
    }
}

/// UTC-basis (unixtime-style, leap-second-free) microsecond count of
/// 00:00:00 UTC on the given MJD
const fn utc_of_mjd(mjd: i64) -> i64 {
    (mjd - MJD_UNIX) * DAY_US
}

/// `n / d` rounded to the nearest integer (halves away from zero), `d > 0`
const fn div_round(n: i128, d: i128) -> i64 {
    if n >= 0 {
        ((n + d / 2) / d) as i64
    } else {
        -((-n + d / 2) / d) as i64
    }
}

impl Segment {
    const fn start_utc(&self) -> i64 {
        utc_of_mjd(self.start_mjd)
    }

    /// TAI − UTC, in microseconds, at the UTC-basis count `utc`
    const fn dat_us(&self, utc: i64) -> i64 {
        let span = (utc - utc_of_mjd(self.ref_mjd)) as i128;
        self.offset_us + div_round(span * self.rate as i128, RATE_DEN)
    }

    /// Internal TAI count of the UTC-basis count `utc`
    const fn raw_of(&self, utc: i64) -> i64 {
        utc + self.dat_us(utc)
    }

    /// Inverse of [`Self::raw_of`]: with `x = utc − ref`, `raw − ref − A =
    /// x (1 + rate/RATE_DEN)` up to the rounding of TAI − UTC, which the
    /// division by `1 + rate/RATE_DEN` cannot push past half a microsecond,
    /// so `utc_of(raw_of(u)) == u` for every `u`.
    const fn utc_of(&self, raw: i64) -> i64 {
        let r = utc_of_mjd(self.ref_mjd);
        let x = (raw - r - self.offset_us) as i128;
        r + div_round(x * RATE_DEN, RATE_DEN + self.rate as i128)
    }

    /// TAI − UTC, in seconds, at a floating-point UTC MJD
    fn dat_seconds(&self, mjd_utc: f64) -> f64 {
        (mjd_utc - self.ref_mjd as f64)
            .mul_add(self.rate as f64 * 1.0e-7, self.offset_us as f64 * 1.0e-6)
    }
}

/// UTC-basis count of the end (exclusive) of segment `k`
const fn end_utc(k: usize) -> i64 {
    if k + 1 < SEGMENTS.len() {
        SEGMENTS[k + 1].start_utc()
    } else {
        utc_of_mjd(END_MJD)
    }
}

/// TAI − UTC, in microseconds, on the last segment extended to
/// 1972-01-01 00:00:00 UTC (9.892242 s): the offset in force just before the
/// +0.107758 s step to the 10 s of the leap-second table.
pub(super) const DAT_US_AT_END: i64 = SEGMENTS[SEGMENTS.len() - 1].dat_us(utc_of_mjd(END_MJD));

/// Internal count at which the 1972-01-01 inserted interval starts; from
/// here on the leap-second table applies.
const END_RAW: i64 = utc_of_mjd(END_MJD) + DAT_US_AT_END;

/// TAI − UTC, in microseconds, at the UTC-basis count `utc`, or `None`
/// outside 1961-01-01 .. 1972-01-01. A label in a negative step's missing
/// seconds uses the preceding day's segment (see the module docs).
pub(super) fn dat_us(utc: i64) -> Option<i64> {
    if !(utc_of_mjd(START_MJD)..utc_of_mjd(END_MJD)).contains(&utc) {
        return None;
    }
    SEGMENTS
        .iter()
        .rev()
        .find(|s| utc >= s.start_utc())
        .map(|s| s.dat_us(utc))
}

/// TAI − UTC, in seconds, at a floating-point UTC MJD, or `None` outside
/// 1961-01-01 .. 1972-01-01
pub(super) fn dat_seconds(mjd_utc: f64) -> Option<f64> {
    if !(START_MJD as f64..END_MJD as f64).contains(&mjd_utc) {
        return None;
    }
    SEGMENTS
        .iter()
        .rev()
        .find(|s| mjd_utc >= s.start_mjd as f64)
        .map(|s| s.dat_seconds(mjd_utc))
}

/// UTC-basis count of an internal count `raw` before the 1972-01-01
/// inserted interval, and, if `raw` lies in an inserted interval (positive
/// step), the microseconds since the interval began.
///
/// Inside an inserted interval the UTC-basis count repeats the last
/// `step` seconds of the day (the new offset is already applied), as for
/// post-1972 leap seconds. Returns `None` from the 1972-01-01 inserted
/// interval on, where the leap-second table takes over.
pub(super) fn raw_to_utc(raw: i64) -> Option<(i64, Option<i64>)> {
    if raw >= END_RAW {
        return None;
    }
    for (k, s) in SEGMENTS.iter().enumerate().rev() {
        if raw < s.raw_of(s.start_utc()) {
            continue;
        }
        let end = end_utc(k);
        let end_raw = s.raw_of(end);
        if raw >= end_raw {
            // Only reachable for a positive step into segment k + 1 (the last
            // segment ends at END_RAW, and for a zero or negative step raw
            // would already be in k + 1)
            let dat_new = SEGMENTS[k + 1].dat_us(end);
            return Some((raw - dat_new, Some(raw - end_raw)));
        }
        return Some((s.utc_of(raw).min(end - 1), None));
    }
    // Before 1961-01-01 TAI − UTC = 0; the step to 1.422818 s is an inserted
    // interval starting at the UTC-basis count of 1961-01-01.
    let b = SEGMENTS[0].start_utc();
    if raw >= b {
        Some((raw - SEGMENTS[0].dat_us(b), Some(raw - b)))
    } else {
        Some((raw, None))
    }
}

/// If a positive step (inserted interval) ends at the UTC-basis count `utc`
/// (00:00:00 UTC of a segment start, 1961-01-01 through 1968-02-01), return
/// the internal count at which it starts and its length in microseconds.
/// The 1972-01-01 step is in the leap-second table.
pub(super) fn inserted_interval_ending_at(utc: i64) -> Option<(i64, i64)> {
    let k = SEGMENTS.iter().position(|s| s.start_utc() == utc)?;
    let old_end_raw = if k == 0 {
        utc
    } else {
        SEGMENTS[k - 1].raw_of(utc)
    };
    let new_start_raw = SEGMENTS[k].raw_of(utc);
    (new_start_raw > old_end_raw).then_some((old_end_raw, new_start_raw - old_end_raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instant, TimeScale};

    /// Every step, in seconds, at each boundary (verified against
    /// `erfa.dat` just before and at the boundary)
    #[test]
    fn steps() {
        let mut got = vec![];
        let mut prev: Option<&Segment> = None;
        for s in SEGMENTS.iter() {
            let b = s.start_utc();
            // TAI − UTC = 0 before 1961
            let old = prev.map_or(0, |p| p.dat_us(b));
            got.push(s.dat_us(b) - old);
            prev = Some(s);
        }
        got.push(10_000_000 - DAT_US_AT_END);
        assert_eq!(
            got,
            [
                1_422_818, -50_000, 0, 100_000, 0, 100_000, 100_000, 100_000, 100_000, 100_000,
                100_000, 0, -100_000, 107_758
            ]
        );
        assert_eq!(DAT_US_AT_END, 9_892_242);
    }

    /// `utc_of` inverts `raw_of` exactly, and `raw_of` is strictly increasing
    #[test]
    fn inverse_is_exact() {
        for (k, s) in SEGMENTS.iter().enumerate() {
            let (a, b) = (s.start_utc(), end_utc(k));
            let mut u = a;
            let step = 7_777_777_777i64;
            while u < b {
                for d in 0..40 {
                    let x = u + d;
                    assert_eq!(s.utc_of(s.raw_of(x)), x, "segment {k}, utc {x}");
                    assert!(s.raw_of(x + 1) > s.raw_of(x));
                }
                u += step;
            }
            for x in [a, a + 1, b - 1, b] {
                assert_eq!(s.utc_of(s.raw_of(x)), x);
            }
        }
    }

    /// TAI − UTC from `erfa.dat(iy, im, id, fd)`, in seconds, generated with
    /// pyerfa 2.0.1.5:
    ///
    /// ```python
    /// for (y, m, d, fd) in cases: print(repr(erfa.dat(y, m, d, fd)))
    /// ```
    #[test]
    fn dat_matches_erfa() {
        const CASES: [(i64, f64, f64); 12] = [
            // (MJD of the date, fraction of day, erfa.dat)
            (37300, 0.0, 1.422818),                // 1961-01-01
            (37511, 0.999999, 1.6975699987039994), // 1961-07-31
            (37512, 0.0, 1.6475700000000002),      // 1961-08-01
            (37665, 0.5, 1.8464196),               // 1962-01-01
            (38333, 0.75, 2.596998),               // 1963-10-31
            (38334, 0.0, 2.6972788),               // 1963-11-01
            (38395, 0.0, 2.765794),                // 1964-01-01
            (39004, 0.0, 4.1550579999999995),      // 1965-09-01
            (39126, 0.25, 4.313818),               // 1966-01-01
            (39886, 0.5, 6.2843860000000005),      // 1968-01-31
            (39887, 0.0, 6.185682),                // 1968-02-01
            (41316, 0.999999, 9.892241997408),     // 1971-12-31
        ];
        for (mjd, fd, expected) in CASES {
            let utc = utc_of_mjd(mjd) + (fd * DAY_US as f64).round() as i64;
            let got = dat_us(utc).unwrap() as f64 * 1.0e-6;
            assert!(
                (got - expected).abs() <= 0.5e-6,
                "MJD {mjd} + {fd}: {got} vs {expected}"
            );
            let got_s = dat_seconds(mjd as f64 + fd).unwrap();
            assert!(
                (got_s - expected).abs() < 1.0e-9,
                "MJD {mjd} + {fd}: {got_s}"
            );
        }
        assert_eq!(dat_us(utc_of_mjd(START_MJD) - 1), None);
        assert_eq!(dat_us(utc_of_mjd(END_MJD)), None);
        assert_eq!(dat_seconds(START_MJD as f64 - 1.0e-9), None);
        assert_eq!(dat_seconds(END_MJD as f64), None);
    }

    /// UTC label -> internal TAI count, against ERFA. Generated with pyerfa
    /// 2.0.1.5 as the TAI microseconds since 1970-01-01 00:00:00 TAI:
    ///
    /// ```python
    /// d1, d2 = erfa.utctai(*erfa.dtf2d("UTC", y, mo, d, h, mi, s))
    /// round(((d1 - 2400000.5 - 40587) + d2) * 86400e6)
    /// ```
    ///
    /// ERFA's two-part JD carries ~0.1 µs, so agreement is to 1 µs.
    #[test]
    fn labels_match_erfa() {
        type Label = (i32, i32, i32, i32, i32, f64);
        const CASES: [(Label, i64); 20] = [
            ((1961, 1, 1, 0, 0, 0.0), -283_996_798_577_182),
            ((1961, 1, 1, 12, 0, 0.0), -283_953_598_576_534),
            ((1961, 7, 31, 23, 59, 59.9), -265_679_998_402_430),
            // Missing label (negative step): lands 0.02 s into 1961-08-01
            ((1961, 7, 31, 23, 59, 59.97), -265_679_998_332_430),
            ((1961, 8, 1, 0, 0, 0.0), -265_679_998_352_430),
            ((1961, 8, 1, 0, 0, 0.02), -265_679_998_332_430),
            ((1963, 10, 31, 23, 59, 59.5), -194_659_197_902_721),
            ((1963, 10, 31, 23, 59, 60.0), -194_659_197_402_721),
            ((1963, 10, 31, 23, 59, 60.05), -194_659_197_352_721),
            ((1963, 11, 1, 0, 0, 0.0), -194_659_197_302_721),
            ((1965, 6, 1, 12, 30, 15.25), -144_674_980_913_499),
            ((1968, 1, 31, 23, 59, 59.95), -60_479_993_764_318),
            ((1968, 2, 1, 0, 0, 0.0), -60_479_993_814_318),
            ((1969, 12, 31, 23, 59, 59.5), 7_500_082),
            ((1970, 1, 1, 0, 0, 0.0), 8_000_082),
            ((1971, 12, 31, 23, 59, 59.0), 63_072_008_892_242),
            ((1971, 12, 31, 23, 59, 60.0), 63_072_009_892_242),
            ((1971, 12, 31, 23, 59, 60.1), 63_072_009_992_242),
            ((1972, 1, 1, 0, 0, 0.0), 63_072_010_000_000),
            ((1966, 7, 4, 3, 21, 7.123456), -110_320_728_086_084),
        ];
        for (c, erfa_raw) in CASES {
            let t = Instant::from_datetime(c.0, c.1, c.2, c.3, c.4, c.5).unwrap();
            assert!(
                (t.raw - erfa_raw).abs() <= 1,
                "{c:?}: satkit {} vs ERFA {erfa_raw} ({} us)",
                t.raw,
                t.raw - erfa_raw
            );
        }
    }

    /// Before 1961 labels are TAI-aligned (TAI − UTC = 0); the +1.422818 s
    /// step at 1961-01-01 is labelled 1960-12-31T23:59:60.0 .. 23:59:61.422817
    #[test]
    fn step_1961() {
        let b = Instant::from_date(1961, 1, 1).unwrap();
        assert_eq!(b.raw, utc_of_mjd(START_MJD) + 1_422_818);
        let before = Instant::from_datetime(1960, 12, 31, 23, 59, 59.0).unwrap();
        assert_eq!(before.raw, utc_of_mjd(START_MJD) - 1_000_000);
        assert_eq!((b - before).as_microseconds(), 2_422_818);
        let t = Instant::from_datetime(1960, 12, 31, 23, 59, 61.4).unwrap();
        assert_eq!(t.raw, utc_of_mjd(START_MJD) + 1_400_000);
        assert_eq!(t.to_string(), "1960-12-31T23:59:61.400000Z");
        assert_eq!(
            Instant::new(b.raw - 1).to_string(),
            "1960-12-31T23:59:61.422817Z"
        );
        assert!(Instant::from_datetime(1960, 12, 31, 23, 59, 61.422818).is_err());
        let early = Instant::from_datetime(1950, 6, 1, 12, 0, 0.0).unwrap();
        assert_eq!(
            early.as_mjd_with_scale(TimeScale::TAI),
            early.as_mjd_with_scale(TimeScale::UTC)
        );
    }

    /// Labels on a negative step's missing seconds are accepted and land on
    /// the next day's instants (ERFA `utctai`), and read back as those
    #[test]
    fn negative_step_gap() {
        let gap = Instant::from_datetime(1968, 1, 31, 23, 59, 59.95).unwrap();
        let next = Instant::from_date(1968, 2, 1).unwrap();
        assert_eq!(gap.raw, next.raw + 50_000);
        assert_eq!(gap.to_string(), "1968-02-01T00:00:00.050000Z");
        // The last real label of 1968-01-31 is 0.1 s earlier than midnight
        let last = Instant::new(next.raw - 1);
        assert_eq!(last.to_string(), "1968-01-31T23:59:59.899999Z");
        // 23:59:60 is not a label on a negative-step day
        assert!(Instant::from_datetime(1968, 1, 31, 23, 59, 60.0).is_err());
    }

    /// Label -> raw -> label round trips, and raw -> label is monotonic:
    /// every microsecond within 20 ms of each step (both ends of an inserted
    /// interval), a stride through the 1.5 s before each step, and a coarse
    /// sweep across 1960–1973. Labels of instants 2 µs or more apart are
    /// strictly increasing; adjacent microseconds may share a label (a UTC
    /// microsecond is slightly longer than an SI one), so there only
    /// non-decreasing is required.
    #[test]
    fn round_trip_and_monotonic() {
        let label = |raw: i64| {
            let g = Instant::new(raw).as_datetime();
            (g.0, g.1, g.2, g.3, g.4, (g.5 * 1.0e6).round() as i64)
        };
        let check = |raw: i64, step: usize, prev: &mut (i32, i32, i32, i32, i32, i64)| {
            let l = label(raw);
            assert!(
                l > *prev || (step == 1 && l == *prev),
                "{raw}: {prev:?} -> {l:?}"
            );
            *prev = l;
            let back =
                Instant::from_datetime(l.0, l.1, l.2, l.3, l.4, l.5 as f64 * 1.0e-6).unwrap();
            // A TAI microsecond with no UTC label of its own reads back as
            // its neighbour; a label always round-trips exactly
            assert!((back.raw - raw).abs() <= 1, "{l:?}: {} vs {raw}", back.raw);
            assert_eq!(label(back.raw), l);
        };
        let mut boundaries: Vec<i64> = SEGMENTS.iter().map(|s| s.start_mjd).collect();
        boundaries.push(END_MJD);
        for mjd in boundaries {
            // First instant of the new day, and of the preceding step
            let new_start = Instant::from_mjd_utc(mjd as f64).raw;
            let old_end = super::super::instant::inserted_interval_ending_at(utc_of_mjd(mjd))
                .map_or(new_start, |(start, _)| start);
            let mut windows = vec![
                (old_end - 20_000, old_end + 20_000, 1),
                (new_start - 20_000, new_start + 20_000, 1),
                (new_start - 1_500_000, new_start, 997),
            ];
            windows.sort();
            for (a, b, step) in windows {
                let mut prev = label(a - 1);
                for raw in (a..b).step_by(step) {
                    check(raw, step, &mut prev);
                }
            }
        }
        let mut raw = Instant::from_date(1960, 6, 1).unwrap().raw;
        let end = Instant::from_date(1973, 1, 1).unwrap().raw;
        let mut prev = label(raw - 1);
        while raw < end {
            check(raw, 2, &mut prev);
            // Away from inserted intervals the UTC MJD maps back too (an f64
            // MJD near 40000 resolves ~0.6 us)
            if prev.5 < 60_000_000 {
                let back = Instant::from_mjd_utc(Instant::new(raw).as_mjd_utc());
                assert!(
                    (back.raw - raw).abs() <= 2,
                    "{raw}: from MJD {}",
                    back.raw - raw
                );
            }
            raw += 3_917_171_113;
        }
    }
}
