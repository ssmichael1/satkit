use super::utc_pre1972;
use super::{InstantError, TimeScale};
use serde::{Deserialize, Serialize};

/// Local result alias used by [`Instant`] constructors.
type Result<T> = std::result::Result<T, InstantError>;

// Days in the month, neglecting leap years
const MDAYS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// A module for handling time and date conversions.  Time is stored natively as
/// a continuous count of TAI microseconds since 1970-01-01 00:00:00 TAI, so
/// leap seconds (and the pre-1972 UTC offsets) are accounted for.
///
/// The Instant struct provides methods for converting to and from Unix time, GPS time,
/// Julian Date, Modified Julian Date, and Gregorian calendar date.
///
/// Why do we need another structure that handles time?
///
/// This structure is necessary as it is time scale aware, i.e. it can
/// handle different time scales such as UTC, TAI, TT, UT1, GPS, etc.
/// This is necessary for high-precision coordinate transforms and orbit propagation.
///
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Instant {
    /// TAI microseconds since 1970-01-01 00:00:00 TAI (the TAI MJD is
    /// `40587 + raw / 86400e6`). This is not Unix time: 1970-01-01 00:00:00
    /// UTC is `raw = 8_000_082`, since TAI − UTC was 8.000082 s then.
    pub raw: i64,
}

/// For conversion between Julian day and
/// Gregorian calendar date
/// See: <https://en.wikipedia.org/wiki/Julian_day>
/// or Expl. Suppl. Astron. Almanac, P. 619
mod gregorian_coefficients {
    #[allow(non_upper_case_globals)]
    pub const y: i64 = 4716;
    #[allow(non_upper_case_globals)]
    pub const j: i64 = 1401;
    #[allow(non_upper_case_globals)]
    pub const m: i64 = 2;
    #[allow(non_upper_case_globals)]
    pub const n: i64 = 12;
    #[allow(non_upper_case_globals)]
    pub const r: i64 = 4;
    #[allow(non_upper_case_globals)]
    pub const p: i64 = 1461;
    #[allow(non_upper_case_globals)]
    pub const v: i64 = 3;
    #[allow(non_upper_case_globals)]
    pub const u: i64 = 5;
    #[allow(non_upper_case_globals)]
    pub const s: i64 = 153;
    #[allow(non_upper_case_globals)]
    pub const t: i64 = 2;
    #[allow(non_upper_case_globals)]
    pub const w: i64 = 2;
    pub const A: i64 = 184;
    pub const B: i64 = 274_277;
    pub const C: i64 = -38;
}

/// Leap second table, newest first.
///
/// Each entry is `(t, ls)`:
///
/// * `t` is the internal (continuous) `raw` count, in microseconds, at which
///   the inserted time *begins* — i.e. the instant labelled `23:59:60` on the
///   last day before the new offset takes effect. Equivalently,
///   `t = u + ls_prev`, where `u` is the UTC-basis (leap-second-free, unixtime)
///   microsecond count of the following 00:00:00 UTC and `ls_prev` is the
///   previous entry's offset.
/// * `ls` is TAI − UTC, in microseconds, from `t` onwards.
///
/// The inserted interval is `[t, t + (ls - ls_prev))`: one second for every
/// real leap second.
///
/// The oldest entry is the 1972-01-01 step from pre-1972 UTC (see
/// `utc_pre1972`), whose drifting TAI − UTC had reached 9.892242 s at
/// 1972-01-01 00:00:00 UTC. Its `ls_prev` is that value, so the step is a
/// 0.107758 s inserted interval labelled `1971-12-31T23:59:60.0` to
/// `23:59:60.107757`, and 1971-12-31 is 86,400.107758 SI seconds plus that
/// day's drift (0.002592 s) long.
const LEAP_SECOND_TABLE: [(i64, i64); 28] = [
    (1483228836000000, 37000000), // 2017-01-01
    (1435708835000000, 36000000), // 2015-07-01
    (1341100834000000, 35000000), // 2012-07-01
    (1230768033000000, 34000000), // 2009-01-01
    (1136073632000000, 33000000), // 2006-01-01
    (915148831000000, 32000000),  // 1999-01-01
    (867715230000000, 31000000),  // 1997-07-01
    (820454429000000, 30000000),  // 1996-01-01
    (773020828000000, 29000000),  // 1994-07-01
    (741484827000000, 28000000),  // 1993-07-01
    (709948826000000, 27000000),  // 1992-07-01
    (662688025000000, 26000000),  // 1991-01-01
    (631152024000000, 25000000),  // 1990-01-01
    (567993623000000, 24000000),  // 1988-01-01
    (489024022000000, 23000000),  // 1985-07-01
    (425865621000000, 22000000),  // 1983-07-01
    (394329620000000, 21000000),  // 1982-07-01
    (362793619000000, 20000000),  // 1981-07-01
    (315532818000000, 19000000),  // 1980-01-01
    (283996817000000, 18000000),  // 1979-01-01
    (252460816000000, 17000000),  // 1978-01-01
    (220924815000000, 16000000),  // 1977-01-01
    (189302414000000, 15000000),  // 1976-01-01
    (157766413000000, 14000000),  // 1975-01-01
    (126230412000000, 13000000),  // 1974-01-01
    (94694411000000, 12000000),   // 1973-01-01
    (78796810000000, 11000000),   // 1972-07-01
    // 1972-01-01: +0.107758 s from pre-1972 UTC (see above)
    (63072000000000 + utc_pre1972::DAT_US_AT_END, 10000000),
];

/// Iterate the leap-second table as `(t, ls, ls_prev)`, newest first, where
/// `ls_prev` is the offset in effect before `t` (for the oldest entry, the
/// pre-1972 TAI − UTC at 1972-01-01 00:00:00 UTC).
fn leap_entries() -> impl Iterator<Item = (i64, i64, i64)> {
    LEAP_SECOND_TABLE.iter().enumerate().map(|(i, &(t, ls))| {
        let ls_prev = LEAP_SECOND_TABLE
            .get(i + 1)
            .map_or(utc_pre1972::DAT_US_AT_END, |e| e.1);
        (t, ls, ls_prev)
    })
}

/// TAI − UTC, in microseconds, at the internal `raw` count.
///
/// Inside an inserted (leap-second) interval this is already the new offset,
/// so `raw - microleapseconds(raw)` repeats the last second(s) of the day
/// (e.g. `23:59:59.x` for the whole of `23:59:60.x`), which keeps UTC MJD,
/// unixtime and the UTC day number on the day the leap second belongs to.
/// Before 1972 the offset drifts (see `utc_pre1972`); before 1961 it is zero.
fn microleapseconds(raw: i64) -> i64 {
    for (t, ls) in LEAP_SECOND_TABLE.iter() {
        if raw >= *t {
            return *ls;
        }
    }
    utc_pre1972::raw_to_utc(raw).map_or(0, |(utc, _)| raw - utc)
}

/// TAI − UTC, in microseconds, in effect at the UTC-basis (leap-second-free,
/// unixtime-style) microsecond count `utc`. The new offset applies from
/// 00:00:00 UTC of the day after the leap second.
fn utc_microleapseconds(utc: i64) -> i64 {
    for (t, ls, ls_prev) in leap_entries() {
        if utc >= t - ls_prev {
            return ls;
        }
    }
    utc_pre1972::dat_us(utc).unwrap_or(0)
}

/// Fold leap seconds into a raw count that was built on a UTC basis (i.e. as if
/// leap seconds did not exist), producing the internal continuous-time raw
/// count. Applied when constructing an [`Instant`] from unixtime, a UTC MJD, or
/// a Gregorian date. A UTC-basis value can never land inside an inserted
/// interval: 00:00:00 of the day after a leap second maps to the end of the
/// leap second, not its start.
fn add_leapseconds(utc: i64) -> i64 {
    // Saturating: `utc` may already be pegged at the i64 boundaries (the
    // MJD/unixtime constructors saturate out-of-range inputs)
    utc.saturating_add(utc_microleapseconds(utc))
}

/// If `raw` falls inside an inserted (leap-second) interval, return the
/// microseconds elapsed since the start of that interval.
fn leap_interval_offset(raw: i64) -> Option<i64> {
    leap_entries()
        .find(|(t, ls, ls_prev)| raw >= *t && raw - *t < ls - ls_prev)
        .map(|(t, _, _)| raw - t)
        .or_else(|| utc_pre1972::raw_to_utc(raw).and_then(|(_, offset)| offset))
}

/// If an inserted interval (a leap second, or a positive pre-1972 step) ends
/// at the UTC-basis count `utc` (a 00:00:00 UTC), return the `raw` count at
/// which it starts and its length in microseconds.
pub(super) fn inserted_interval_ending_at(utc: i64) -> Option<(i64, i64)> {
    leap_entries()
        .find(|(t, _, ls_prev)| utc == t - ls_prev)
        .map(|(t, ls, ls_prev)| (t, ls - ls_prev))
        .or_else(|| utc_pre1972::inserted_interval_ending_at(utc))
}

/// TAI − UTC, in seconds, in effect at the given UTC MJD (new offset from
/// 00:00:00 UTC of the day after each leap second; before 1972 the drifting
/// offset of `utc_pre1972`, zero before 1961). Used to route UT1 conversions
/// and the UT1 − UTC interpolation through the continuous UT1 − TAI.
pub(crate) fn tai_minus_utc_at_mjd_utc(mjd_utc: f64) -> f64 {
    // Compare in MJD (as the EOP table lookup does) rather than rounding the
    // query to microseconds, so both switch at exactly the same value.
    for (t, ls, ls_prev) in leap_entries() {
        let boundary = (t - ls_prev - Instant::MJD_EPOCH.raw) as f64 / 86_400_000_000.0;
        if mjd_utc >= boundary {
            return ls as f64 * 1.0e-6;
        }
    }
    utc_pre1972::dat_seconds(mjd_utc).unwrap_or(0.0)
}

/// Argument, in radians, of the periodic TDB − TT term (Vallado Eq. 3-50):
/// `628.3076 T + 6.2401`, with `T` in Julian centuries of TT from J2000.
/// One revolution per year.
#[inline]
fn tdb_minus_tt_arg(ttc: f64) -> f64 {
    628.3076f64.mul_add(ttc, 6.2401)
}

impl Instant {
    /// Construct a new Instant from raw microseconds
    ///
    /// # Arguments
    /// * `raw` - TAI microseconds since 1970-01-01 00:00:00 TAI (see
    ///   [`Instant::raw`]; not Unix time)
    ///
    /// # Returns
    /// A new Instant object
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::Instant;
    /// let now = Instant::new(1234567890);
    /// ```
    pub const fn new(raw: i64) -> Self {
        Self { raw }
    }

    /// Construct a new Instant from GPS week and second of week
    ///
    /// # Arguments
    /// * `week` - The GPS week number
    /// * `sow` - The second of week
    ///
    /// # Returns
    /// A new Instant object
    ///
    pub fn from_gps_week_and_second(week: i32, sow: f64) -> Self {
        // Saturating: an absurd week count clamps instead of overflowing
        // (which would panic in debug builds and wrap in release)
        let raw = (week as i64)
            .saturating_mul(604_800_000_000)
            .saturating_add((sow * 1.0e6) as i64)
            .saturating_add(Self::GPS_EPOCH.raw);
        Self { raw }
    }

    /// Construct a new Instant from Unix time
    ///
    /// # Arguments
    /// * `unixtime` - The Unix time in seconds
    ///
    /// # Returns
    /// A new Instant object representing the input Unix time
    ///
    /// # Note:
    /// Unixtime is the number of non-leap seconds since Jan 1 1970 00:00:00 UTC
    /// (Leap seconds are ignored!!)
    pub fn from_unixtime(unixtime: f64) -> Self {
        // unixtime is the UTC-basis count (it ignores leap seconds and the
        // pre-1972 offsets); fold TAI − UTC in.
        Self {
            raw: add_leapseconds((unixtime * 1.0e6).round() as i64),
        }
    }

    /// Convert Instant to Unix time
    ///
    /// # Returns
    /// The Unix time in seconds (since 1970-01-01 00:00:00 UTC)
    ///
    /// # Note
    /// Unixtime is the number of non-leap seconds since
    /// 1970-01-01 00:00:00 UTC.
    pub fn as_unixtime(&self) -> f64 {
        // Subtract TAI − UTC since unixtime ignores it
        (self.raw - microleapseconds(self.raw)) as f64 * 1.0e-6
    }

    /// J2000 epoch is 2000-01-01 12:00:00 TT
    /// TT (Terrestrial Time) is 32.184 seconds ahead of TAI
    pub const J2000: Self = Self {
        raw: 946727967816000,
    };

    /// Unix epoch is 1970-01-01 00:00:00 UTC. TAI − UTC was 8.000082 s then
    /// (pre-1972 UTC), so its `raw` is 8,000,082 µs, not zero.
    pub const UNIX_EPOCH: Self = Self { raw: 8_000_082 };

    /// GPS epoch is 1980-01-06 00:00:00 UTC
    pub const GPS_EPOCH: Self = Self {
        raw: 315964819000000,
    };

    pub const INVALID: Self = Self { raw: i64::MIN };

    /// Modified Julian day epoch is
    /// 1858-11-17 00:00:00 UTC
    ///
    /// satkit takes TAI − UTC = 0 before 1961, so this is also 1858-11-17
    /// 00:00:00 TAI, and its `raw` (−40587 days) doubles as the UTC-basis
    /// (unixtime-style) microsecond count of MJD 0 in the UTC conversions.
    pub const MJD_EPOCH: Self = Self {
        raw: -3506716800000000,
    };

    /// Return the day of the week
    /// 0 = Sunday, 1 = Monday, ..., 6 = Saturday
    ///
    /// See: <https://en.wikipedia.org/wiki/Determination_of_the_day_of_the_week>
    pub fn day_of_week(&self) -> super::Weekday {
        let jd = self.as_jd_utc();
        // `(jd + 1.5) mod 7` is always in [0, 7), so the floor is 0..=6 and the
        // conversion never fails; fall back to `Invalid` defensively.
        super::Weekday::try_from(((jd + 1.5) % 7.0).floor() as i32)
            .unwrap_or(super::Weekday::Invalid)
    }

    /// As Modified Julian Date (UTC)
    /// Days since 1858-11-17 00:00:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    #[deprecated(note = "Use as_mjd_utc() for explicit scale, or as_mjd_with_scale()")]
    pub fn as_mjd(&self) -> f64 {
        self.as_mjd_utc()
    }

    /// As Modified Julian Date (UTC)
    /// Days since 1858-11-17 00:00:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    pub fn as_mjd_utc(&self) -> f64 {
        self.as_mjd_with_scale(TimeScale::UTC)
    }

    /// UTC calendar day as an integer Modified Julian Day number.
    /// Same leap-second convention as `as_mjd_utc`, but integer arithmetic only.
    pub(crate) fn utc_day_number(&self) -> i64 {
        (self.raw - Self::MJD_EPOCH.raw - microleapseconds(self.raw)).div_euclid(86_400_000_000)
    }

    /// Create Instant from Modified Julian Date (UTC)
    ///
    /// # Arguments
    /// * `mjd` - Modified Julian Date (UTC)
    ///
    /// # Returns
    /// A new Instant object representing the given MJD
    #[deprecated(note = "Use from_mjd_utc() for explicit scale, or from_mjd_with_scale()")]
    pub fn from_mjd(mjd: f64) -> Self {
        Self::from_mjd_utc(mjd)
    }

    /// Create Instant from Modified Julian Date (UTC)
    ///
    /// # Arguments
    /// * `mjd` - Modified Julian Date (UTC)
    ///
    /// # Returns
    /// A new Instant object representing the given MJD
    pub fn from_mjd_utc(mjd: f64) -> Self {
        Self::from_mjd_with_scale(mjd, TimeScale::UTC)
    }

    /// Create Instant from Julian Date (UTC)
    ///
    /// # Arguments
    /// * `jd` - Julian Date (UTC)
    ///
    /// # Returns
    /// A new Instant object representing the given JD
    #[deprecated(note = "Use from_jd_utc() for explicit scale, or from_jd_with_scale()")]
    pub fn from_jd(jd: f64) -> Self {
        Self::from_jd_utc(jd)
    }

    /// Create Instant from Julian Date (UTC)
    ///
    /// # Arguments
    /// * `jd` - Julian Date (UTC)
    ///
    /// # Returns
    /// A new Instant object representing the given JD
    pub fn from_jd_utc(jd: f64) -> Self {
        Self::from_mjd_utc(jd - 2400000.5)
    }

    /// Create Instant from Julian Date with given time scale
    /// (UTC, TAI, TT, UT1, GPS)
    /// Days since 4713 BC January 1, 12:00 UTC
    ///
    /// # Arguments
    /// * `jd` - Julian Date
    /// * `scale` - The time scale to use
    ///
    /// # Returns
    /// A new Instant object representing the given JD at given time scale
    pub fn from_jd_with_scale(jd: f64, scale: TimeScale) -> Self {
        Self::from_mjd_with_scale(jd - 2400000.5, scale)
    }

    /// Construct an instant from a given Modified Julian Date
    /// and time scale
    ///
    /// # Arguments
    /// * `mjd` - The Modified Julian Date
    /// * `scale` - The time scale to use
    ///
    /// # Returns
    /// A new Instant object representing the given MJD at given time scale
    pub fn from_mjd_with_scale(mjd: f64, scale: TimeScale) -> Self {
        // The float-to-i64 cast saturates for out-of-range MJD values; the
        // subsequent epoch-offset additions must saturate as well or they
        // overflow (debug panic / release wrap) at the i64 boundaries.
        match scale {
            TimeScale::UTC => {
                let raw = ((mjd * 86_400_000_000.0) as i64).saturating_add(Self::MJD_EPOCH.raw);
                Self {
                    raw: add_leapseconds(raw),
                }
            }
            TimeScale::TAI => {
                let raw = ((mjd * 86_400_000_000.0) as i64).saturating_add(Self::MJD_EPOCH.raw);
                Self { raw }
            }
            TimeScale::TT => {
                let raw = ((mjd * 86_400_000_000.0) as i64)
                    .saturating_add(Self::MJD_EPOCH.raw)
                    .saturating_sub(32_184_000);
                Self { raw }
            }
            TimeScale::UT1 => {
                // Go through UT1 − TAI, which is continuous across leap
                // seconds (UT1 − UTC is not). The EOP table is indexed by UTC,
                // so it is evaluated at the UT1 value itself; |UT1 − UTC| < 0.9 s
                // and UT1 − TAI changes by ~ns over that span, so the
                // approximation is exact for practical purposes, including
                // inside a leap second.
                let dut1 = crate::earth_orientation_params::eop_from_mjd_utc_or_zero(mjd)[0];
                let ut1_minus_tai = dut1 - tai_minus_utc_at_mjd_utc(mjd);
                Self::from_mjd_with_scale(mjd - ut1_minus_tai / 86_400.0, TimeScale::TAI)
            }
            TimeScale::GPS => {
                // GPS = TAI - 19 seconds
                let raw = ((mjd * 86_400_000_000.0) as i64)
                    .saturating_add(Self::MJD_EPOCH.raw)
                    .saturating_add(19_000_000);
                Self { raw }
            }
            TimeScale::Invalid => Self::INVALID,
            TimeScale::TDB => {
                // Inverse of the TT -> TDB series in `as_mjd_with_scale`.
                // The periodic term is evaluated at TDB instead of TT; the
                // two differ by < 2 ms, which moves the term by ~1e-13 s.
                let ttc: f64 = (mjd - (2451545.0 - 2400000.5)) / 36525.0;
                let mjd = (0.001657f64 / 86400.0f64).mul_add(-tdb_minus_tt_arg(ttc).sin(), mjd)
                    - 32.184 / 86400.0;
                Self::from_mjd_with_scale(mjd, TimeScale::TAI)
            }
        }
    }

    /// As Julian Date (UTC)
    /// Days since 4713 BC January 1, 12:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    #[deprecated(note = "Use as_jd_utc() for explicit scale, or as_jd_with_scale()")]
    #[allow(deprecated)]
    pub fn as_jd(&self) -> f64 {
        self.as_mjd() + 2400000.5
    }

    /// As Julian Date (UTC)
    /// Days since 4713 BC January 1, 12:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    pub fn as_jd_utc(&self) -> f64 {
        self.as_mjd_utc() + 2400000.5
    }

    /// As Julian Date with given time scale
    /// Days since 4713 BC January 1, 12:00 UTC
    ///
    /// # Arguments
    /// * `scale` - The time scale to use
    ///
    /// # Returns
    /// The Julian Date in the given time scale
    ///
    pub fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
        self.as_mjd_with_scale(scale) + 2400000.5
    }

    /// Add given floating-point number of days to Instant instance,
    /// and return new instance representing new time.
    ///
    /// Days are defined in this case to have exactly 86400.0 seconds
    /// In other words, this will ignore leap seconds and the integer
    /// part of the floating point will increment the number of days and
    /// the decimal part will increment the fractions of a day.
    ///
    /// So, for example, adding 1.0 to a day with a leap second will
    /// increment by a full day
    ///
    /// # Arguments
    /// * `days` - The number of days to add
    ///
    /// # Returns
    /// A new Instant object representing the new time
    ///
    pub fn add_utc_days(&self, days: f64) -> Self {
        let mut utc = self.as_mjd_with_scale(TimeScale::UTC);
        utc += days;
        Self::from_mjd_with_scale(utc, TimeScale::UTC)
    }

    /// As Modified Julian Date with given time scale
    /// Days since 1858-11-17 00:00:00 UTC
    ///
    /// # Arguments
    /// * `scale` - The time scale to use
    ///
    /// # Returns
    /// The Modified Julian Date in the given time scale
    ///
    pub fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
        // Saturating: raw values near the i64 boundaries (e.g. saturated
        // extreme constructions, Instant::INVALID) would overflow the plain
        // epoch-offset subtraction, which panics in debug builds
        match scale {
            TimeScale::UTC => {
                (self
                    .raw
                    .saturating_sub(Self::MJD_EPOCH.raw)
                    .saturating_sub(microleapseconds(self.raw))) as f64
                    / 86_400_000_000.0
            }
            TimeScale::TT => {
                (self
                    .raw
                    .saturating_sub(Self::MJD_EPOCH.raw)
                    .saturating_add(32_184_000)) as f64
                    / 86_400_000_000.0
            }
            TimeScale::UT1 => {
                // UT1 = UTC + (UT1 − UTC), except that inside a leap second
                // the UTC MJD repeats the last second of the day while UT1
                // keeps running. Using the TAI − UTC of the UTC *day*
                // (rather than of `raw`) makes this UT1 = TAI + (UT1 − TAI),
                // continuous through the leap second; elsewhere it is the
                // plain UTC MJD plus UT1 − UTC.
                let mjd_utc = self.as_mjd_utc();
                let dut1 = crate::earth_orientation_params::eop_from_mjd_utc_or_zero(mjd_utc)[0];
                let utc = self.raw.saturating_sub(microleapseconds(self.raw));
                let mjd_utc_day_basis = (self
                    .raw
                    .saturating_sub(Self::MJD_EPOCH.raw)
                    .saturating_sub(utc_microleapseconds(utc)))
                    as f64
                    / 86_400_000_000.0;
                mjd_utc_day_basis + dut1 / 86_400.0
            }
            TimeScale::TAI => {
                self.raw.saturating_sub(Self::MJD_EPOCH.raw) as f64 / 86_400_000_000.0
            }
            TimeScale::GPS => {
                // GPS = TAI - 19 seconds
                (self
                    .raw
                    .saturating_sub(Self::MJD_EPOCH.raw)
                    .saturating_sub(19_000_000)) as f64
                    / 86_400_000_000.0
            }
            TimeScale::TDB => {
                let tt: f64 = self.as_mjd_with_scale(TimeScale::TT);
                let ttc: f64 = (tt - (2451545.0f64 - 2400000.5f64)) / 36525.0;
                // Vallado Eq. 3-50: TDB − TT ≈ 0.001657 s · sin(628.3076 T + 6.2401),
                // T in Julian centuries of TT; the argument is in radians
                // (annual period, mean anomaly of the Earth).
                (0.001657f64 / 86400.0f64).mul_add(tdb_minus_tt_arg(ttc).sin(), tt)
            }
            // Return NaN rather than 0.0 (a perfectly valid MJD, 1858-11-17) so
            // that using an Invalid time scale visibly poisons downstream math
            // instead of silently producing a plausible date.
            TimeScale::Invalid => f64::NAN,
        }
    }

    /// Return the Gregorian date and time
    ///
    /// # Returns
    /// (year, month, day, hour, minute, second), UTC
    pub fn as_datetime(&self) -> (i32, i32, i32, i32, i32, f64) {
        // UTC-basis (leap-second-free) microseconds since the Unix epoch.
        // Inside a leap second this repeats 23:59:59.x of the same day; the
        // label is patched to 23:59:60.x below. Euclidean division keeps the
        // time of day in [0, 86400 s) for instants before 1970.
        let utc = self.raw.saturating_sub(microleapseconds(self.raw));
        let unix_day = utc.div_euclid(86_400_000_000);
        let utc_usec_of_day = utc.rem_euclid(86_400_000_000);

        let mut hour = utc_usec_of_day / 3_600_000_000;
        let mut minute = (utc_usec_of_day % 3_600_000_000) / 60_000_000;
        let mut second = (utc_usec_of_day % 60_000_000) as f64 * 1.0e-6;

        // Inside an inserted interval: label as 23:59:60.x (up to
        // 23:59:61.422817 for the 1.422818 s step at 1961-01-01; see
        // utc_pre1972)
        if let Some(offset) = leap_interval_offset(self.raw) {
            hour = 23;
            minute = 59;
            second = 60.0 + offset as f64 * 1.0e-6;
        }

        /// See: https://en.wikipedia.org/wiki/Julian_day
        /// or Expl. Suppl. Astron. Almanac, P. 619
        use gregorian_coefficients as gc;
        // Julian Day Number of the UTC calendar day (1970-01-01 is JDN 2440588)
        let jd = unix_day + 2_440_588;
        let f = jd + gc::j + (((4 * jd + gc::B) / 146097) * 3) / 4 + gc::C;
        let e = gc::r * f + gc::v;
        let g = (e % gc::p) / gc::r;
        let h = gc::u * g + gc::w;
        let day = ((h % gc::s) / gc::u) + 1;
        let month = ((h / gc::s + gc::m) % gc::n) + 1;
        let year = (e / gc::p) - gc::y + (gc::n + gc::m - month) / gc::n;

        (
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second,
        )
    }

    /// Construct an instant from a given UTC date
    ///
    /// # Arguments
    /// * `year` - The year
    /// * `month` - The month
    /// * `day` - The day
    ///
    /// # Returns
    /// A new Instant object representing the given date
    pub fn from_date(year: i32, month: i32, day: i32) -> Result<Self> {
        Self::from_datetime(year, month, day, 0, 0, 0.0)
    }

    /// Return the day of the year (1-based, Gregorian); leap-year aware.
    ///
    /// # Returns
    /// The day of the year (1-based, Gregorian); leap-year aware.
    ///
    /// # Notes:
    ///
    /// * Gregorian Jan 1 = 1
    ///
    /// # Example
    ///
    /// ```rust
    /// // Examples checked against google query
    ///
    /// let thedate = satkit::Instant::from_date(2023, 1, 1).unwrap();
    /// assert_eq!(thedate.day_of_year(), 1);
    ///
    /// let thedate = satkit::Instant::from_date(2025, 8, 16).unwrap();
    /// assert_eq!(thedate.day_of_year(), 228);
    /// ```
    pub fn day_of_year(&self) -> u32 {
        let (y, m, d, _, _, _) = self.as_datetime();
        let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
        let mut doy = d;
        for mm in 1..m {
            doy += if mm == 2 && leap {
                29i32
            } else {
                MDAYS[(mm - 1) as usize] as i32
            };
        }
        doy as u32
    }

    /// Convenience alias for `from_datetime`
    ///
    /// Construct an instant from a given Gregorian UTC date and time
    ///
    /// # Arguments
    /// * `year` - The year
    /// * `month` - The month (1-12)
    /// * `day` - The day (1-31)
    /// * `hour` - The hour (0-23)
    /// * `minute` - The minute (0-59)
    /// * `second` - The second (0.0-60.0)
    ///
    /// # Returns
    /// A new Instant object representing the given date and time, or error if invalid
    pub fn utc(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self> {
        Self::from_datetime(year, month, day, hour, minute, second)
    }

    /// Construct an instant from a given Gregorian UTC date and time
    ///
    /// # Arguments
    /// * `year` - The year
    /// * `month` - The month
    /// * `day` - The day
    /// * `hour` - The hour
    /// * `minute` - The minute
    /// * `second` - The second
    ///
    /// # Returns
    /// A new Instant object representing the given date and time, or error if invalid
    /// or error if the month, day, hour, minute, or second are out of bounds
    ///
    /// # Leap seconds
    /// `second` in `[60, 61)` is accepted only as the label of a real UTC leap
    /// second, e.g. `2016-12-31 23:59:60.5`, and is otherwise an error. Before
    /// 1972 the positive UTC steps are labelled the same way: 0.1 s (e.g.
    /// `1963-10-31 23:59:60.05`), 0.107758 s on 1971-12-31, and 1.422818 s on
    /// 1960-12-31 (`[60, 61.422818)`).
    ///
    /// # Pre-1972 UTC
    /// From 1961 to 1972 TAI − UTC drifted ("rubber seconds") and stepped by
    /// fractions of a second; it is evaluated at the label, as in ERFA `dat`
    /// and `utctai`. UTC stepped back on 1961-08-01 (0.05 s) and 1968-02-01
    /// (0.1 s), so the last 0.05 s / 0.1 s of labels on the preceding days
    /// never occurred; they are accepted and, as in ERFA, land on the same
    /// instants as the first 0.05 s / 0.1 s of the next day. Before 1961,
    /// TAI − UTC is taken as 0.
    pub fn from_datetime(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Result<Self> {
        let mut check_leapsecond: bool = false;

        // Bounds checking on input
        if !(1..=12).contains(&month) {
            return Err(InstantError::InvalidMonth(month));
        }
        let max_day = if month == 2 {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        } else {
            MDAYS[(month - 1) as usize]
        };
        if day < 1 || day > max_day as i32 {
            return Err(InstantError::InvalidDay(day));
        }
        if !(0..=23).contains(&hour) {
            return Err(InstantError::InvalidHour(hour));
        }
        if !(0..=59).contains(&minute) {
            return Err(InstantError::InvalidMinute(minute));
        }
        if !(0.0..60.0).contains(&second) {
            // Check for rare case of leap second. Ordinary leap seconds allow
            // [60, 61); the longest inserted interval, the 1.422818 s step at
            // 1961-01-01, allows up to 61.422818 (checked exactly below).
            if (60.0..62.0).contains(&second) {
                check_leapsecond = true;
            } else {
                return Err(InstantError::InvalidSecondF(second));
            }
        }

        use gregorian_coefficients as gc;
        let h = month as i64 - gc::m;
        let g = year as i64 + gc::y - (gc::n - h) / gc::n;
        let f = (h - 1 + gc::n) % gc::n;
        let e = (gc::p * g) / gc::r + day as i64 - 1 - gc::j;
        let mut jd = e + (gc::s * f + gc::t) / gc::u;
        jd = jd - (3 * ((g + gc::A) / 100)) / 4 - gc::C;

        // Note, JD is the given julian day at noon on given date,
        // so we subtract an additional 0.5 to get midnight
        let jd = jd as f64 - 0.5;
        let mjd = jd - 2400000.5;

        // A leap-second label 23:59:60.x is built as the start of the next
        // minute (i.e. 00:00:00 of the next day, on the UTC basis) plus an
        // offset into the inserted interval; everything else directly.
        let (minute_second, leap_offset) = if check_leapsecond {
            (60.0, ((second - 60.0) * 1_000_000.0).round() as i64)
        } else {
            (second, 0)
        };

        // Checked: an extreme year overflows the i64 microsecond count
        // (panicking in debug builds, silently wrapping in release)
        let utc = (mjd as i64)
            .checked_mul(86_400_000_000)
            .and_then(|v| v.checked_add(hour as i64 * 3_600_000_000))
            .and_then(|v| v.checked_add(minute as i64 * 60_000_000))
            .and_then(|v| v.checked_add((minute_second * 1_000_000.0).round() as i64))
            .and_then(|v| v.checked_add(Self::MJD_EPOCH.raw))
            .ok_or(InstantError::InvalidYear(year))?;

        if check_leapsecond {
            // Valid only if `utc` is the 00:00:00 UTC at which an inserted
            // interval ends and the offset lies inside that interval
            return inserted_interval_ending_at(utc)
                .filter(|(_, len)| leap_offset < *len)
                .map(|(t, _)| Self {
                    raw: t + leap_offset,
                })
                .ok_or(InstantError::InvalidLeapSecond);
        }

        let raw = add_leapseconds(utc);
        Ok(Self { raw })
    }

    /// Construct an instant from a given Gregorian date and time interpreted
    /// in the specified time scale
    ///
    /// For `TimeScale::UTC` this is identical to [`Self::from_datetime`],
    /// including leap-second handling.  For the uniform (non-leap-second) time
    /// scales the Gregorian components are interpreted directly in that scale.
    ///
    /// # Arguments
    /// * `year` - The year
    /// * `month` - The month
    /// * `day` - The day
    /// * `hour` - The hour
    /// * `minute` - The minute
    /// * `second` - The second
    /// * `scale` - The time scale in which the components are expressed
    ///
    /// # Returns
    /// A new Instant object representing the given date and time, or error if invalid
    pub fn from_datetime_with_scale(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
        scale: TimeScale,
    ) -> Result<Self> {
        // UTC preserves the exact existing behavior, including leap-second handling
        if scale == TimeScale::UTC {
            return Self::from_datetime(year, month, day, hour, minute, second);
        }

        // Bounds checking on input.  Uniform time scales have no leap seconds,
        // so the second must be in [0, 60).
        if !(1..=12).contains(&month) {
            return Err(InstantError::InvalidMonth(month));
        }
        let max_day = if month == 2 {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        } else {
            MDAYS[(month - 1) as usize]
        };
        if day < 1 || day > max_day as i32 {
            return Err(InstantError::InvalidDay(day));
        }
        if !(0..=23).contains(&hour) {
            return Err(InstantError::InvalidHour(hour));
        }
        if !(0..=59).contains(&minute) {
            return Err(InstantError::InvalidMinute(minute));
        }
        if !(0.0..60.0).contains(&second) {
            return Err(InstantError::InvalidSecondF(second));
        }

        use gregorian_coefficients as gc;
        let h = month as i64 - gc::m;
        let g = year as i64 + gc::y - (gc::n - h) / gc::n;
        let f = (h - 1 + gc::n) % gc::n;
        let e = (gc::p * g) / gc::r + day as i64 - 1 - gc::j;
        let mut jd = e + (gc::s * f + gc::t) / gc::u;
        jd = jd - (3 * ((g + gc::A) / 100)) / 4 - gc::C;

        // Note, JD is the given julian day at noon on given date,
        // so we subtract an additional 0.5 to get midnight
        let jd = jd as f64 - 0.5;
        let mjd = jd - 2400000.5 + (hour as f64 * 3600.0 + minute as f64 * 60.0 + second) / 86400.0;

        Ok(Self::from_mjd_with_scale(mjd, scale))
    }

    /// Current time
    ///
    /// # Returns
    /// The current time as an Instant object
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::Instant;
    /// let now = Instant::now();
    /// ```
    ///
    pub fn now() -> Self {
        let now = std::time::SystemTime::now();
        let since_epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        let raw = since_epoch.as_micros() as i64;
        Self {
            raw: add_leapseconds(raw),
        }
    }
}

impl std::fmt::Display for Instant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (year, month, day, hour, minute, second) = self.as_datetime();
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:09.6}Z",
            year, month, day, hour, minute, second
        )
    }
}

impl std::fmt::Debug for Instant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (year, month, day, hour, minute, second) = self.as_datetime();
        write!(
            f,
            "Instant {{ year: {}, month: {}, day: {}, hour: {}, minute: {}, second: {:06.3} }}",
            year, month, day, hour, minute, second
        )
    }
}
