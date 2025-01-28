use super::TimeScale;
use serde::{Deserialize, Serialize};

/// A module for handling time and date conversions.  Time is stored natively as
/// the number of microseconds since the Unix epoch (1970-01-01 00:00:00 UTC)
/// with leap seconds accounted for.
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
    /// The number of microseconds since
    /// Unix epoch (1970-01-01 00:00:00 UTC)
    pub(crate) raw: i64,
}

/// For conversian between Julian day and
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

/// Leap second table
/// The first element is the number of microseconds since unixtime epoch
/// The second element is the number of leap seconds to add as microseconds
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
    (63072009000000, 10000000),   // 1972-01-01
];

/// Return the number of leap "micro" seconds at "raw" time,
/// which is microseconds since unixtime epoch
fn microleapseconds(raw: i64) -> i64 {
    for (t, ls) in LEAP_SECOND_TABLE.iter() {
        if raw > *t {
            return *ls;
        }
    }
    0
}

impl Instant {
    /// Construct a new Instant from raw microseconds
    ///
    /// # Arguments
    /// * `raw` - The number of microseconds since unixtime epoch
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
        let week = week as i64;
        let raw = week * 14_515_200_000_000 + (sow * 1.0e6) as i64 + Self::GPS_EPOCH.raw;
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
        let mut raw = (unixtime * 1.0e6) as i64 + Self::UNIX_EPOCH.raw;

        // Add leapseconds since unixtime ignores them
        let ls = microleapseconds(raw);
        raw += ls;
        // Make sure adding the leapseconds didn't cross another
        // leapsecond boundary
        raw += microleapseconds(raw) - ls;
        Self { raw }
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
        // Subtract leap seconds since unixtime ignores them
        (self.raw - Self::UNIX_EPOCH.raw - microleapseconds(self.raw)) as f64 * 1.0e-6
    }

    /// J2000 epoch is 2000-01-01 12:00:00 TT
    /// TT (Terristrial Time) is 32.184 seconds ahead of TAI
    pub const J2000: Self = Self {
        raw: 946728064184000,
    };

    /// Unix epoch is 1970-01-01 00:00:00 UTC
    pub const UNIX_EPOCH: Self = Self { raw: 0 };

    /// GPS epoch is 1980-01-06 00:00:00 UTC
    pub const GPS_EPOCH: Self = Self {
        raw: 315964819000000,
    };

    pub const INVALID: Self = Self { raw: i64::MIN };

    /// Modified Julian day epoch is
    /// 1858-11-17 00:00:00 UTC
    pub const MJD_EPOCH: Self = Self {
        raw: -3506716800000000,
    };

    /// Return the day of the week
    /// 0 = Sunday, 1 = Monday, ..., 6 = Saturday
    ///
    /// See: <https://en.wikipedia.org/wiki/Determination_of_the_day_of_the_week>
    pub fn day_of_week(&self) -> super::Weekday {
        let jd = self.as_jd();
        super::Weekday::from(((jd + 1.5) % 7.0).floor() as i32)
    }

    /// As Modified Julian Date (UTC)
    /// Days since 1858-11-17 00:00:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    pub fn as_mjd(&self) -> f64 {
        // Make sure to account for leap seconds
        self.as_mjd_with_scale(TimeScale::UTC)
    }

    /// Create Instant from Modified Julian Date (UTC)
    ///
    /// # Arguments
    /// * `mjd` - Modified Julian Date
    ///
    /// # Returns
    /// A new Instant object representing the given MJD
    pub fn from_mjd(mjd: f64) -> Self {
        Self::from_mjd_with_scale(mjd, TimeScale::UTC)
    }

    /// Create Instant from Julian Date (UTC)
    ///
    /// # Arguments
    /// * `jd` - Julian Date
    ///
    /// # Returns
    /// A new Instant object representing the given JD
    pub fn from_jd(jd: f64) -> Self {
        Self::from_mjd(jd - 2400000.5)
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
        match scale {
            TimeScale::UTC => {
                let raw = (mjd * 86_400_000_000.0) as i64 + Self::MJD_EPOCH.raw;
                let ls = microleapseconds(raw);
                let raw = raw + ls;
                // Make sure adding the leapseconds didn't cross another
                // leapsecond boundary
                let raw = raw + microleapseconds(raw) - ls;
                Self { raw }
            }
            TimeScale::TAI => {
                let raw = (mjd * 86_400_000_000.0) as i64 + Self::MJD_EPOCH.raw;
                Self { raw }
            }
            TimeScale::TT => {
                let raw = (mjd * 86_400_000_000.0) as i64 + Self::MJD_EPOCH.raw - 32_184_000;
                Self { raw }
            }
            TimeScale::UT1 => {
                // This will be approximately correct for computing ut1
                let eop =
                    crate::earth_orientation_params::eop_from_mjd_utc(mjd).unwrap_or([0.0; 6]);
                let dut1 = eop[0] as f64;
                Self::from_mjd_with_scale(mjd - dut1 / 86_400.0, TimeScale::UTC)
            }
            TimeScale::GPS => {
                if mjd >= 44244.0 {
                    let raw = (mjd * 86_400_000_000.0) as i64 + Self::GPS_EPOCH.raw + 19_000_000;
                    Self { raw }
                } else {
                    let raw = (mjd * 86_400_000_000.0) as i64 + Self::MJD_EPOCH.raw;
                    Self { raw }
                }
            }
            TimeScale::Invalid => Self::INVALID,
            TimeScale::TDB => {
                let ttc: f64 = (mjd - (2451545.0 - 2400000.4)) / 36525.0;
                let mjd = (0.01657f64 / 86400.0f64).mul_add(
                    -(std::f64::consts::PI / 180.0 * 628.3076f64.mul_add(ttc, 6.2401)).sin(),
                    mjd,
                ) - 32.184 / 86400.0;
                Self::from_mjd_with_scale(mjd, TimeScale::TAI)
            }
        }
    }

    /// As Julian Date (UTC)
    /// Days since 4713 BC January 1, 12:00 UTC
    /// where each day is 86,400 seconds
    /// (no leap seconds)
    pub fn as_jd(&self) -> f64 {
        self.as_mjd() + 2400000.5
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
        match scale {
            TimeScale::UTC => {
                (self.raw - Self::MJD_EPOCH.raw - microleapseconds(self.raw)) as f64
                    / 86_400_000_000.0
            }
            TimeScale::TT => {
                (self.raw - Self::MJD_EPOCH.raw + 32_184_000) as f64 / 86_400_000_000.0
            }
            TimeScale::UT1 => {
                let mjd_utc = self.as_mjd();
                let eop =
                    crate::earth_orientation_params::eop_from_mjd_utc(mjd_utc).unwrap_or([0.0; 6]);
                let dut1 = eop[0] as f64;
                mjd_utc + dut1 / 86_400.0
            }
            TimeScale::TAI => (self.raw - Self::MJD_EPOCH.raw) as f64 / 86_400_000_000.0,
            TimeScale::GPS => {
                if self > &Self::GPS_EPOCH {
                    (self.raw - Self::GPS_EPOCH.raw - 19_000_000) as f64 / 86_400_000_000.0
                } else {
                    (self.raw - Self::MJD_EPOCH.raw) as f64 / 86_400_000_000.0
                }
            }
            TimeScale::TDB => {
                let tt: f64 = self.as_mjd_with_scale(TimeScale::TT);
                let ttc: f64 = (tt - (2451545.0f64 - 2400000.4f64)) / 36525.0;
                (0.001657f64 / 86400.0f64).mul_add(
                    (std::f64::consts::PI / 180.0 * 628.3076f64.mul_add(ttc, 6.2401)).sin(),
                    tt,
                )
            }
            TimeScale::Invalid => 0.0,
        }
    }

    /// Return the Gregorian date and time
    ///
    /// # Returns
    /// (year, month, day, hour, minute, second), UTC
    pub fn as_datetime(&self) -> (i32, i32, i32, i32, i32, f64) {
        // Fractional part of UTC day, accounting for leapseconds and TT - TAI
        let utc_usec_of_day = (self.raw - microleapseconds(self.raw)) % 86_400_000_000;
        let mut jdadd: i64 = 0;

        let mut hour = utc_usec_of_day / 3_600_000_000;
        if hour < 12 {
            jdadd += 1
        }
        let mut minute = (utc_usec_of_day - (hour * 3_600_000_000)) / 60_000_000;
        let mut second =
            (utc_usec_of_day - (hour * 3_600_000_000) - (minute * 60_000_000)) as f64 * 1.0e-6;

        // Rare case where we are in the middle of a leap-second
        for (t, _) in LEAP_SECOND_TABLE.iter() {
            if self.raw >= *t && self.raw - *t < 1_000_000 {
                hour = 23;
                minute = 59;
                if second == 0.0 {
                    second = 60.0;
                    jdadd -= 1;
                } else {
                    second += 1.0;
                }
            }
        }

        /// See: https://en.wikipedia.org/wiki/Julian_day
        /// or Expl. Suppl. Astron. Almanac, P. 619
        use gregorian_coefficients as gc;
        let mut jd = self.as_jd().floor() as i64;
        jd += jdadd;
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
    pub fn from_date(year: i32, month: i32, day: i32) -> Self {
        Self::from_datetime(year, month, day, 0, 0, 0.0)
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
    /// A new Instant object representing the given date and time
    pub fn from_datetime(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Self {
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

        let mut raw = mjd as i64 * 86_400_000_000
            + (hour as i64 * 3_600_000_000)
            + (minute as i64 * 60_000_000)
            + (second * 1_000_000.0) as i64
            + Self::MJD_EPOCH.raw;
        // Account for additional leap seconds if needed
        let ls = microleapseconds(raw);
        raw += ls;
        // Make sure adding the leapseconds didn't cross another
        // leapsecond boundary
        raw = raw + microleapseconds(raw) - ls;

        Self { raw }
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
        let mut raw = since_epoch.as_micros() as i64;
        let ls = microleapseconds(raw);
        raw += ls;
        // Make sure adding the leapseconds didn't cross another
        // leapsecond boundary
        raw += microleapseconds(raw) - ls;
        Self { raw }
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
