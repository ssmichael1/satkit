//! Edge-biased time generators shared by the property suites
//! (`tests/properties.rs`, `tests/properties_data.rs`).
//!
//! Uniform sampling over 1972–2045 lands within a few seconds of one of the
//! 28 leap-second boundaries with probability ~1e-6 per case, never produces
//! a `23:59:60` label, and never goes before 1970. Every time-scale defect
//! fixed in #217 lived in exactly those places, so the strategies here mix
//! uniform sampling with:
//!
//! * labels within a few seconds of every leap second, including the
//!   inserted `23:59:60.x` itself, and of the positive pre-1972 UTC steps
//!   (0.1 s, 0.107758 s at 1972-01-01, and satkit's 1.422818 s at
//!   1961-01-01, labelled `23:59:60.x` the same way);
//! * exact edge values (1961-01-01, the pre-1972 steps, 1970-01-01,
//!   1972-01-01, J2000, the last leap second, the Unix-epoch neighbourhood);
//! * pre-1970 dates back to 1900, including the pre-1972 "rubber second"
//!   era (1961–1971), where TAI − UTC drifts.
//!
//! The calendar arithmetic here (`days_from_civil` / `civil_from_days`) is
//! deliberately independent of satkit's own Gregorian code, so properties
//! that compare the two reach the same answer by different routes.
//!
//! Not every item is used by every test binary that includes this module.
#![allow(dead_code)]

use proptest::prelude::*;
use satkit::{Duration, Instant};

/// Microseconds per second / minute / day.
pub const US: i64 = 1_000_000;
pub const US_MIN: i64 = 60 * US;
pub const US_DAY: i64 = 86_400 * US;

/// Every UTC day that ends with inserted time, as `(year, month, day,
/// inserted microseconds)`, oldest first.
///
/// Hard-coded from IERS Bulletin C (1972 on) and USNO `tai-utc.dat`
/// (before 1972), not read back from satkit, so a defect in satkit's tables
/// cannot hide itself. Before 1972 the rows are the positive steps of
/// rubber-second UTC, from the old segment's TAI − UTC at midnight to the
/// new one's; the first row is satkit's convention for the start of its UTC
/// model (TAI − UTC = 0 before 1961-01-01, 1.422818 s from it). Each step is
/// labelled `23:59:60.x` on the day it ends.
pub const LEAP_DAYS: [(i32, i32, i32, i64); 36] = [
    (1960, 12, 31, 1_422_818),
    (1963, 10, 31, 100_000),
    (1964, 3, 31, 100_000),
    (1964, 8, 31, 100_000),
    (1964, 12, 31, 100_000),
    (1965, 2, 28, 100_000),
    (1965, 6, 30, 100_000),
    (1965, 8, 31, 100_000),
    (1971, 12, 31, 107_758),
    (1972, 6, 30, US),
    (1972, 12, 31, US),
    (1973, 12, 31, US),
    (1974, 12, 31, US),
    (1975, 12, 31, US),
    (1976, 12, 31, US),
    (1977, 12, 31, US),
    (1978, 12, 31, US),
    (1979, 12, 31, US),
    (1981, 6, 30, US),
    (1982, 6, 30, US),
    (1983, 6, 30, US),
    (1985, 6, 30, US),
    (1987, 12, 31, US),
    (1989, 12, 31, US),
    (1990, 12, 31, US),
    (1992, 6, 30, US),
    (1993, 6, 30, US),
    (1994, 6, 30, US),
    (1995, 12, 31, US),
    (1997, 6, 30, US),
    (1998, 12, 31, US),
    (2005, 12, 31, US),
    (2008, 12, 31, US),
    (2012, 6, 30, US),
    (2015, 6, 30, US),
    (2016, 12, 31, US),
];

/// The UTC days that end in a negative pre-1972 step, as `(year, month,
/// day, removed microseconds)`: the last `removed` µs of labels on these
/// days never occurred (USNO `tai-utc.dat`).
pub const REMOVED_DAYS: [(i32, i32, i32, i64); 2] = [(1961, 7, 31, 50_000), (1968, 1, 31, 100_000)];

/// Pre-1972 TAI − UTC from USNO `tai-utc.dat`, as `(year, month, A [µs],
/// MJD₀, rate [1e-7 s/day])`: from 00:00 UTC on the 1st of the month,
/// TAI − UTC = A + (MJD_UTC − MJD₀) × rate.
pub const TAI_UTC_PRE1972: [(i32, i32, i64, i64, i64); 13] = [
    (1961, 1, 1_422_818, 37_300, 12_960),
    (1961, 8, 1_372_818, 37_300, 12_960),
    (1962, 1, 1_845_858, 37_665, 11_232),
    (1963, 11, 1_945_858, 37_665, 11_232),
    (1964, 1, 3_240_130, 38_761, 12_960),
    (1964, 4, 3_340_130, 38_761, 12_960),
    (1964, 9, 3_440_130, 38_761, 12_960),
    (1965, 1, 3_540_130, 38_761, 12_960),
    (1965, 3, 3_640_130, 38_761, 12_960),
    (1965, 7, 3_740_130, 38_761, 12_960),
    (1965, 9, 3_840_130, 38_761, 12_960),
    (1966, 1, 4_313_170, 39_126, 25_920),
    (1968, 2, 4_213_170, 39_126, 25_920),
];

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`; valid for all `i32` years).
pub fn days_from_civil(y: i32, m: i32, d: i32) -> i64 {
    let y = y as i64 - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`].
pub fn civil_from_days(z: i64) -> (i32, i32, i32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    ((y + i64::from(m <= 2)) as i32, m as i32, d as i32)
}

/// Inserted microseconds at the end of the UTC day `days` (since
/// 1970-01-01).
pub fn inserted_us(days: i64) -> i64 {
    LEAP_DAYS
        .iter()
        .find(|(y, m, d, _)| days_from_civil(*y, *m, *d) == days)
        .map_or(0, |e| e.3)
}

/// Whether the UTC day `days` ends in inserted or removed time.
pub fn ends_in_step(days: i64) -> bool {
    inserted_us(days) != 0
        || REMOVED_DAYS
            .iter()
            .any(|(y, m, d, _)| days_from_civil(*y, *m, *d) == days)
}

/// Whether the UTC day `days` is in the rubber-second era
/// (1961-01-01 .. 1971-12-31), where a UTC second is not an SI second.
pub fn in_rubber_era(days: i64) -> bool {
    (days_from_civil(1961, 1, 1)..days_from_civil(1972, 1, 1)).contains(&days)
}

/// `n / d` rounded to the nearest integer, halves away from zero (`d > 0`).
fn div_round(n: i128, d: i128) -> i64 {
    (if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }) as i64
}

/// TAI − UTC, in microseconds (rounded), at the UTC-basis count `basis`
/// (microseconds since the label 1970-01-01T00:00:00, 86400 s per day), from
/// the hard-coded tables: 0 before 1961, `TAI_UTC_PRE1972` to 1971, then
/// 10 s plus the leap seconds so far. A step applies from 00:00:00 of the
/// day after it.
pub fn tai_minus_utc_us(basis: i64) -> i64 {
    let days = basis.div_euclid(US_DAY);
    if days >= days_from_civil(1972, 1, 1) {
        let passed = LEAP_DAYS
            .iter()
            .filter(|(y, m, d, _)| *y >= 1972 && days > days_from_civil(*y, *m, *d))
            .count() as i64;
        return (10 + passed) * US;
    }
    TAI_UTC_PRE1972
        .iter()
        .rev()
        .find(|(y, m, ..)| days >= days_from_civil(*y, *m, 1))
        .map_or(0, |&(_, _, a, mjd0, rate)| {
            // µs since MJD₀ (MJD 40587 is 1970-01-01); rate × 1e-7 s/day is
            // rate / 864e9 µs per µs
            let span = basis as i128 - (mjd0 - 40_587) as i128 * US_DAY as i128;
            a + div_round(span * rate as i128, 864_000_000_000)
        })
}

/// A UTC calendar label with microsecond resolution.
///
/// `us` counts microseconds into the minute, so `us ≥ 60 s` is a
/// leap-second label `23:59:60.x` (up to `23:59:61.422817` on 1960-12-31).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label {
    pub y: i32,
    pub mo: i32,
    pub d: i32,
    pub h: i32,
    pub mi: i32,
    pub us: i64,
}

impl Label {
    pub fn new(y: i32, mo: i32, d: i32, h: i32, mi: i32, us: i64) -> Self {
        Self {
            y,
            mo,
            d,
            h,
            mi,
            us,
        }
    }

    /// Label for `days` since 1970-01-01 plus `tod_us` into the day
    /// (`0 ≤ tod_us < 86400 s`).
    pub fn from_day_tod(days: i64, tod_us: i64) -> Self {
        let (y, mo, d) = civil_from_days(days);
        Self::new(
            y,
            mo,
            d,
            (tod_us / 3_600_000_000) as i32,
            ((tod_us / US_MIN) % 60) as i32,
            tod_us % US_MIN,
        )
    }

    /// Days since 1970-01-01.
    pub fn days(&self) -> i64 {
        days_from_civil(self.y, self.mo, self.d)
    }

    /// Whether this is an inserted-second label (`:60` or later).
    pub fn is_leap(&self) -> bool {
        self.us >= US_MIN
    }

    /// Whole seconds and microsecond fraction of the seconds field.
    pub fn sec_frac(&self) -> (i64, i64) {
        (self.us / US, self.us % US)
    }

    /// Leap-second-free ("UTC basis", unixtime-style) microseconds since
    /// 1970-01-01T00:00:00. Not meaningful for leap labels.
    pub fn utc_basis_us(&self) -> i64 {
        self.days() * US_DAY + self.h as i64 * 3_600 * US + self.mi as i64 * US_MIN + self.us
    }

    /// `YYYY-MM-DDTHH:MM:SS.ffffffZ`, the format `Display` / `as_iso8601`
    /// produce.
    pub fn iso(&self) -> String {
        let (s, f) = self.sec_frac();
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            self.y, self.mo, self.d, self.h, self.mi, s, f
        )
    }

    /// Seconds as the `f64` a caller would pass to `from_datetime`.
    pub fn seconds_f64(&self) -> f64 {
        self.us as f64 * 1.0e-6
    }

    /// The instant for this label, built without any float rounding: whole
    /// seconds through `from_datetime` (exact in `f64`) plus the fraction as
    /// an integer [`Duration`]. Inside a single second there is no leap
    /// boundary, so this is exact, except in the rubber-second era, where a
    /// UTC second is 1 + (1.3 … 3.0)e-8 SI seconds: there (outside an
    /// inserted interval) the seconds go through `from_datetime` as `f64`,
    /// which rounds to the microsecond.
    pub fn instant(&self) -> Instant {
        if in_rubber_era(self.days()) && !self.is_leap() {
            return Instant::from_datetime(
                self.y,
                self.mo,
                self.d,
                self.h,
                self.mi,
                self.seconds_f64(),
            )
            .unwrap_or_else(|e| panic!("from_datetime rejected {self:?}: {e}"));
        }
        let (s, f) = self.sec_frac();
        Instant::from_datetime(self.y, self.mo, self.d, self.h, self.mi, s as f64)
            .unwrap_or_else(|e| panic!("from_datetime rejected {self:?}: {e}"))
            + Duration::from_microseconds(f)
    }
}

/// Uniform labels over `[y0-01-01, y1-01-01)`.
pub fn uniform_label(y0: i32, y1: i32) -> impl Strategy<Value = Label> {
    (
        days_from_civil(y0, 1, 1)..days_from_civil(y1, 1, 1),
        0..US_DAY,
    )
        .prop_map(|(days, tod)| Label::from_day_tod(days, tod))
}

/// Label at `offset_us` from the start of the inserted interval of
/// `LEAP_DAYS[idx]` (negative: before it; `0 ≤ offset < inserted`: inside
/// it, i.e. `23:59:60.x`; beyond: after the following midnight).
pub fn leap_label(idx: usize, offset_us: i64) -> Label {
    let (y, mo, d, ins_us) = LEAP_DAYS[idx];
    if offset_us < ins_us {
        // 23:59:(60 + offset). Offsets are within a minute of the boundary.
        assert!(offset_us >= -US_MIN);
        Label::new(y, mo, d, 23, 59, US_MIN + offset_us)
    } else {
        Label::from_day_tod(days_from_civil(y, mo, d) + 1, offset_us - ins_us)
    }
}

/// `(leap-day index, offset)` near a leap-second boundary: uniform within
/// ±3 s of the inserted interval, plus the exact boundary microseconds.
pub fn leap_offset() -> impl Strategy<Value = (usize, i64)> {
    (0..LEAP_DAYS.len()).prop_flat_map(|i| {
        let ins = LEAP_DAYS[i].3;
        (
            Just(i),
            prop_oneof![
                3 => -3 * US..ins + 3 * US,
                1 => prop::sample::select(vec![-US, -1, 0, 1, US / 2, ins - 1, ins, ins + 1]),
            ],
        )
    })
}

/// Labels within a few seconds of a leap second, `:60` labels included.
pub fn leap_edge_label() -> impl Strategy<Value = Label> {
    leap_offset().prop_map(|(i, off)| leap_label(i, off))
}

/// Hand-picked edge labels.
pub fn fixed_edge_labels() -> Vec<Label> {
    vec![
        Label::new(1900, 1, 1, 0, 0, 0),
        Label::new(1900, 3, 1, 0, 0, 0),
        Label::new(1960, 2, 29, 12, 0, 0),
        Label::new(1960, 12, 31, 23, 59, 61_422_817),
        Label::new(1961, 1, 1, 0, 0, 0),
        Label::new(1961, 8, 1, 0, 0, 0),
        Label::new(1963, 10, 31, 23, 59, 60_050_000),
        Label::new(1968, 1, 31, 23, 59, 59_899_999),
        Label::new(1968, 2, 1, 0, 0, 0),
        Label::new(1969, 12, 31, 23, 59, 59_999_999),
        Label::new(1970, 1, 1, 0, 0, 0),
        Label::new(1970, 1, 1, 0, 0, 1),
        Label::new(1971, 12, 31, 23, 59, 59_999_999),
        Label::new(1971, 12, 31, 23, 59, 60_000_000),
        Label::new(1971, 12, 31, 23, 59, 60_107_757),
        Label::new(1972, 1, 1, 0, 0, 0),
        // J2000 = 2000-01-01T12:00:00 TT = 11:58:55.816 UTC
        Label::new(2000, 1, 1, 11, 58, 55_816_000),
        Label::new(2000, 2, 29, 23, 59, 59_999_999),
        Label::new(2016, 12, 31, 23, 59, 59_999_999),
        Label::new(2016, 12, 31, 23, 59, 60_000_000),
        Label::new(2016, 12, 31, 23, 59, 60_999_999),
        Label::new(2017, 1, 1, 0, 0, 0),
        Label::new(2044, 12, 31, 23, 59, 59_999_999),
    ]
}

/// The main edge-biased label strategy: uniform over the leap-second era,
/// labels at every leap second, pre-1970 dates, and fixed edges.
pub fn any_label() -> impl Strategy<Value = Label> {
    prop_oneof![
        4 => uniform_label(1972, 2045),
        4 => leap_edge_label(),
        2 => uniform_label(1900, 1972),
        1 => prop::sample::select(fixed_edge_labels()),
    ]
}

/// Like [`any_label`] but without `:60` labels (for representations that
/// cannot express them: unixtime, UTC MJD, Python `datetime`).
pub fn any_non_leap_label() -> impl Strategy<Value = Label> {
    any_label().prop_filter("leap-second label", |l| !l.is_leap())
}

/// Edge-biased instants (the instants of [`any_label`]).
pub fn any_instant() -> impl Strategy<Value = Instant> {
    any_label().prop_map(|l| l.instant())
}

/// A proptest config with `default` cases, overridable with the standard
/// `PROPTEST_CASES` environment variable (e.g. `PROPTEST_CASES=100000
/// cargo test --test properties` for a deep run). The global-reject budget
/// scales with the case count, so properties that `prop_assume!` away a
/// few percent of inputs still work at deep counts.
pub fn cases(default: u32) -> ProptestConfig {
    let n = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default);
    ProptestConfig {
        max_global_rejects: n.max(1024),
        max_local_rejects: (n * 4).max(65_536),
        ..ProptestConfig::with_cases(n)
    }
}
