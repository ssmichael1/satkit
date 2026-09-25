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
//!   inserted `23:59:60.x` itself (and `23:59:60`–`23:59:69.x` for the 10 s
//!   step satkit models at 1972-01-01);
//! * exact edge values (1970-01-01, 1972-01-01, J2000, the last leap second,
//!   the Unix-epoch neighbourhood);
//! * pre-1970 dates back to 1900.
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
/// inserted seconds)`, oldest first.
///
/// Hard-coded from IERS Bulletin C (not read back from satkit, so a defect
/// in satkit's table cannot hide itself). The first row is satkit's
/// modelling convention rather than a real leap second: TAI − UTC is taken
/// as 0 before 1972 and the 10 s offset UTC started with is represented as
/// a single 10 s inserted interval labelled `1971-12-31T23:59:60` …
/// `23:59:69.999999` (see `LEAP_SECOND_TABLE` in `src/time/instant.rs`).
/// TAI − UTC after row `k` is `10 + k` seconds.
pub const LEAP_DAYS: [(i32, i32, i32, i64); 28] = [
    (1971, 12, 31, 10),
    (1972, 6, 30, 1),
    (1972, 12, 31, 1),
    (1973, 12, 31, 1),
    (1974, 12, 31, 1),
    (1975, 12, 31, 1),
    (1976, 12, 31, 1),
    (1977, 12, 31, 1),
    (1978, 12, 31, 1),
    (1979, 12, 31, 1),
    (1981, 6, 30, 1),
    (1982, 6, 30, 1),
    (1983, 6, 30, 1),
    (1985, 6, 30, 1),
    (1987, 12, 31, 1),
    (1989, 12, 31, 1),
    (1990, 12, 31, 1),
    (1992, 6, 30, 1),
    (1993, 6, 30, 1),
    (1994, 6, 30, 1),
    (1995, 12, 31, 1),
    (1997, 6, 30, 1),
    (1998, 12, 31, 1),
    (2005, 12, 31, 1),
    (2008, 12, 31, 1),
    (2012, 6, 30, 1),
    (2015, 6, 30, 1),
    (2016, 12, 31, 1),
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

/// Inserted seconds at the end of the UTC day `days` (since 1970-01-01).
pub fn inserted_seconds(days: i64) -> i64 {
    LEAP_DAYS
        .iter()
        .find(|(y, m, d, _)| days_from_civil(*y, *m, *d) == days)
        .map_or(0, |e| e.3)
}

/// TAI − UTC, in seconds, on the UTC day `days`, from the hard-coded table
/// (the offset that applies from 00:00:00 of that day).
pub fn tai_minus_utc_on_day(days: i64) -> i64 {
    let passed = LEAP_DAYS
        .iter()
        .filter(|(y, m, d, _)| days > days_from_civil(*y, *m, *d))
        .count() as i64;
    if passed == 0 {
        0
    } else {
        9 + passed
    }
}

/// A UTC calendar label with microsecond resolution.
///
/// `us` counts microseconds into the minute, so `us ≥ 60 s` is a
/// leap-second label `23:59:60.x` (up to `23:59:69.x` on 1971-12-31).
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
    /// boundary, so this is exact.
    pub fn instant(&self) -> Instant {
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
    let (y, mo, d, ins) = LEAP_DAYS[idx];
    let ins_us = ins * US;
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
        let ins = LEAP_DAYS[i].3 * US;
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
        Label::new(1969, 12, 31, 23, 59, 59_999_999),
        Label::new(1970, 1, 1, 0, 0, 0),
        Label::new(1970, 1, 1, 0, 0, 1),
        Label::new(1971, 12, 31, 23, 59, 59_999_999),
        Label::new(1971, 12, 31, 23, 59, 60_000_000),
        Label::new(1971, 12, 31, 23, 59, 69_999_999),
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
