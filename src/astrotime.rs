//!
//! # AstroTime
//!
//! A representation of time that allows for conversion between
//! various scales or epochs. Epoch conversion is an often necessary for
//! calculation of astronomic phenomenon, e.g.the exact rotation between
//! Earth-centered inertial and fixed coordinate frames.
//!
//! The conversion between epochs necessitates the use of this struct rather
//! than using one from the standard rust or chrono
//!
//! ## Scales include:
//!
//! * `UTC`` - Universal Time Coordinate. In common use.  Local times are
//!   generally UTC time with a timezone offset
//!
//! * `TAI` - International Atomic Time.  A monotonically increasing epoch
//!   that differsfrom UTC in that it does not include leap seconds.
//!
//! * `UT1`` - Universal Time.  This is defined by the Earth's rotation, with
//!   correction for polar wander
//!
//! * `TT` - Terrestrial Time.  Defined from 2001 on. Leads TAI by
//!    a constant 32.184 seconds.
//!
//! * `GPS` - GPS Time.  Defined for global positining system. Trails TAI by a
//!   constant 19 seconds after GPS epoch of Jan 6 1980.  Typically reported in
//!   weeks since midnight Jan 1 1980 and seconds of week.
//!
//! * `TDB` - Barycentric Dynamical time.  Used as time scale when dealing
//!     with solar system ephemerides in solar system barycentric coordinate
//!     system.
//!
//! ## Additional Info
//!
//! For a good description, see [here](https://www.stjarnhimlen.se/comp/time.html)
//!
//!

#[derive(PartialEq, PartialOrd, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Instant {
    pub(crate) mjd_tai: f64,
}

use crate::utils::{skerror, SKResult};
use crate::Duration;

use super::earth_orientation_params as eop;

extern crate chrono;

use std::f64::consts::PI;

use std::time::{SystemTime, UNIX_EPOCH};

const UTC1970: f64 = 40587.;
const UTC1972: f64 = 41317.;
const TAI1972: f64 = UTC1972 + 10. / 86400.;
const UTCGPS0: f64 = 44244.; // 1980-01-06
const TAIGPS0: f64 = UTCGPS0 + 19. / 86400.;

/// Conversion from Julian Date to Modified Julian Date
pub const JD2MJD: f64 = -2400000.5;

/// Conversion from Modified Julian Date to Julian Date
pub const MJD2JD: f64 = 2400000.5;

/*
const DELTAAT_OLD: [[f64; 4]; 15] = [
    [36204., 0., 36204., 0.],
    [36934., 1.4178180, 37300., 0.001296],
    [37300., 1.4228180, 37300., 0.001296],
    [37512., 1.3728180, 37300., 0.001296],
    [37665., 1.8458580, 37665., 0.0011232],
    [38334., 1.9458580, 37665., 0.0011232],
    [38395., 3.2401300, 38761., 0.001296],
    [38486., 3.3401300, 38761., 0.001296],
    [38639., 3.4401300, 38761., 0.001296],
    [38761., 3.5401300, 38761., 0.001296],
    [38820., 3.6401300, 38761., 0.001296],
    [38942., 3.7401300, 38761., 0.001296],
    [39004., 3.8401300, 38761., 0.001296],
    [39126., 4.3131700, 39126., 0.002592],
    [39887., 4.2131700, 39126., 0.002592],
];
*/

/// Time Scales
///
/// # Enum Values:
///
/// * `UTC` - Univeral Time Coordiante
/// * `TT` - Terrestrial Time
/// * `UT1` - UT1
/// * `TAI` - International Atomic Time
/// * `GPS` - Global Positioning System
/// * `TDB` - Barycentric Dynamical Time
/// * `INVALID` - Invalid
///    
#[derive(PartialEq, Debug)]
pub enum Scale {
    /// Invalid
    INVALID = -1,
    /// Universal Time Coordinate
    UTC = 1,
    /// Terrestrial Time
    TT = 2,
    /// UT1
    UT1 = 3,
    /// International Atomic Time
    TAI = 4,
    /// Global Positioning System
    GPS = 5,
    /// Barycentric Dynamical Time
    TDB = 6,
}

// Delta AT values (leap seconds)
const DELTAAT: [[u64; 2]; 28] = [
    [3692217600, 37],
    [3644697600, 36],
    [3550089600, 35],
    [3439756800, 34],
    [3345062400, 33],
    [3124137600, 32],
    [3076704000, 31],
    [3029443200, 30],
    [2982009600, 29],
    [2950473600, 28],
    [2918937600, 27],
    [2871676800, 26],
    [2840140800, 25],
    [2776982400, 24],
    [2698012800, 23],
    [2634854400, 22],
    [2603318400, 21],
    [2571782400, 20],
    [2524521600, 19],
    [2492985600, 18],
    [2461449600, 17],
    [2429913600, 16],
    [2398291200, 15],
    [2366755200, 14],
    [2335219200, 13],
    [2303683200, 12],
    [2287785600, 11],
    [2272060800, 10],
];

impl std::fmt::Display for Instant {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (mut year, mut mon, mut day, mut hour, mut min, mut sec) = self.as_datetime();

        // Prevent edge case where seconds is displayed as 60
        if sec > 59.999 {
            let t = *self + 5e-4 / 86400.0;
            (year, mon, day, hour, min, sec) = t.as_datetime();
        }

        write!(
            f,
            "{}-{:02}-{:02} {:02}:{:02}:{:06.3}Z",
            year, mon, day, hour, min, sec
        )
    }
}

impl std::ops::Add<f64> for Instant {
    type Output = Self;
    #[inline]
    fn add(self, other: f64) -> Self::Output {
        Instant::from_mjd(self.mjd_tai + other, Scale::TAI)
    }
}

impl std::ops::Add<Duration> for Instant {
    type Output = Self;
    #[inline]
    fn add(self, other: Duration) -> Self::Output {
        Self {
            mjd_tai: self.mjd_tai + other.days(),
        }
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Self;
    #[inline]
    fn sub(self, other: Duration) -> Self::Output {
        Self {
            mjd_tai: self.mjd_tai - other.days(),
        }
    }
}

/// Take the difference between two AstroTime time instances
/// returning the floating-point number of days between
/// the instances.
impl std::ops::Sub<f64> for Instant {
    type Output = Self;
    fn sub(self, other: f64) -> Self::Output {
        Self {
            mjd_tai: self.mjd_tai - other,
        }
    }
}

impl std::ops::Sub<f64> for &Instant {
    type Output = Instant;
    fn sub(self, other: f64) -> Self::Output {
        Instant {
            mjd_tai: self.mjd_tai - other,
        }
    }
}

impl std::ops::Sub<Instant> for &Instant {
    type Output = Duration;
    fn sub(self, other: Instant) -> Duration {
        Duration::Days(self.mjd_tai - other.mjd_tai)
    }
}

impl std::ops::Sub<&Instant> for &Instant {
    type Output = Duration;
    fn sub(self, other: &Instant) -> Duration {
        Duration::Days(self.mjd_tai - other.mjd_tai)
    }
}

impl std::ops::Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, other: Instant) -> Duration {
        Duration::Days(self.mjd_tai - other.mjd_tai)
    }
}

impl std::ops::Sub<&Instant> for Instant {
    type Output = Duration;
    fn sub(self, other: &Instant) -> Duration {
        Duration::Days(self.mjd_tai - other.mjd_tai)
    }
}

impl std::ops::Add<&Vec<Duration>> for Instant {
    type Output = Vec<Self>;
    fn add(self, other: &Vec<Duration>) -> Self::Output {
        other.iter().map(|x| self + x.days()).collect()
    }
}

impl std::ops::Add<&Vec<f64>> for Instant {
    type Output = Vec<Self>;
    fn add(self, other: &Vec<f64>) -> Self::Output {
        other.iter().map(|x| self + *x).collect()
    }
}

impl<T: chrono::TimeZone> std::convert::From<chrono::DateTime<T>> for Instant {
    fn from(c: chrono::DateTime<T>) -> Self {
        let mut t = Instant::from_unixtime(c.timestamp() as f64);
        t = t + c.timestamp_subsec_micros() as f64 / 86400.0e6;
        t
    }
}

impl std::convert::From<&chrono::NaiveDateTime> for Instant {
    fn from(c: &chrono::NaiveDateTime) -> Self {
        let mut t = Instant::from_unixtime(c.and_utc().timestamp() as f64);
        t = t + c.and_utc().timestamp_subsec_micros() as f64 / 86400.0e6;
        t
    }
}

impl std::convert::From<Instant> for chrono::NaiveDateTime {
    fn from(s: Instant) -> chrono::NaiveDateTime {
        let secs: i64 = s.as_unixtime() as i64;
        let nsecs: u32 = (((s.as_mjd_with_scale(TimeScale::UTC) * 86400.0) % 1.0) * 1.0e9) as u32;
        chrono::DateTime::from_timestamp(secs, nsecs)
            .unwrap()
            .naive_utc()
    }
}

impl TryFrom<std::time::SystemTime> for Instant {
    type Error = &'static str;
    fn try_from(st: std::time::SystemTime) -> Result<Self, Self::Error> {
        let val = st.duration_since(std::time::SystemTime::UNIX_EPOCH);
        match val {
            Ok(v) => Ok(Instant::from_unixtime(v.as_secs() as f64)),
            Err(_) => Err("Invalid system time"),
        }
    }
}

impl Default for Instant {
    fn default() -> Self {
        Self::new()
    }
}

impl Instant {
    #[inline]
    /// Construct new astrotime object,
    /// representing time at TAI epoch of modified Julian date
    pub fn new() -> Instant {
        Instant { mjd_tai: JD2MJD }
    }

    /// Construct new AstroTime object, representing given unixtime
    /// (seconds since midnight Jan 1 1970, UTC)
    ///
    /// # Arguments:
    ///
    /// * `t` - The unixtime (seconds since midnight, Jan 1, 1970, UTC)
    pub fn from_unixtime(t: f64) -> Instant {
        Instant::from_mjd(t / 86400.0 + UTC1970, Scale::UTC)
    }

    /// Construt new AstroTime object, representing
    /// current date and time
    pub fn now() -> SKResult<Instant> {
        let now = SystemTime::now();
        match now.duration_since(UNIX_EPOCH) {
            Ok(v) => Ok(Instant::from_mjd(
                v.as_millis() as f64 / 86400000.0 + UTC1970,
                Scale::UTC,
            )),
            Err(v) => skerror!("Cannot get current time: {}", v),
        }
    }

    /// Convert to unixtime (seconds since midnight Jan 1 1970, UTC)
    pub fn as_unixtime(&self) -> f64 {
        (self.as_mjd_with_scale(TimeScale::UTC) - UTC1970) * 86400.0
    }

    /// Add given floating-point number of days to AstroTime instance,
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
    pub fn add_utc_days(&self, days: f64) -> Instant {
        let mut utc = self.as_mjd_with_scale(TimeScale::UTC);
        utc += days;
        Instant::from_mjd(utc, Scale::UTC)
    }

    /// Construct new AstroTime object representing time at
    /// given Modified Julian Day (MJD) and time scale
    ///
    /// Modified Julian Day is the days (including fractional) since
    /// midnight on Nov 17, 1858.  The MJD can be computed by
    /// subtracting 2400000.5 from the Julian day
    /// # Arguments:
    ///
    /// * `mjd` - The modified Julian day
    /// * `scale` - The time scale of the input date
    ///
    /// # Returns
    ///
    /// * AstroTime object
    pub fn from_mjd(val: f64, scale: Scale) -> Instant {
        match scale {
            Scale::TAI => Instant { mjd_tai: val },
            Scale::UTC => Instant {
                mjd_tai: val + mjd_utc2tai_seconds(val) / 86400.0,
            },
            Scale::TT => Instant {
                mjd_tai: val - 32.184 / 86400.0,
            },
            Scale::GPS => Instant {
                mjd_tai: {
                    if val >= UTCGPS0 - f64::EPSILON {
                        val + 19.0 / 86400.0
                    } else {
                        val
                    }
                },
            },
            Scale::TDB => Instant {
                mjd_tai: {
                    let ttc: f64 = (val - (2451545.0 - 2400000.4)) / 36525.0;
                    val - 0.01657 / 86400.0 * (PI / 180.0 * (628.3076 * ttc + 6.2401)).sin()
                        - 32.184 / 86400.0
                },
            },
            Scale::UT1 => {
                let utc: f64 = val - eop::eop_from_mjd_utc(val).unwrap()[0] / 86400.0;
                Instant {
                    mjd_tai: utc + mjd_utc2tai_seconds(val) / 86400.0,
                }
            }
            Scale::INVALID => Instant { mjd_tai: JD2MJD },
        }
    }

    /// Construct AstroTime object from given UTC Gregorian Date
    ///
    /// # Arguments
    ///
    /// * `year` - the year
    /// * `month` - the month, 1 based (1=January, 2=February, ...)
    /// * `day` - Day of month, starting from 1
    ///
    /// # Returns
    ///
    /// * AstroTime Object
    pub fn from_date(year: i32, month: u32, day: u32) -> Instant {
        Instant::from_mjd(date2mjd_utc(year, month, day) as f64, Scale::UTC)
    }

    /// Convert AstroTime to UTC Gregorian date
    ///
    /// # Returns
    ///
    /// * Tuple with following values:
    ///   * year - the year
    ///   * month - the month, 1 based (1=January, 2=February, ...)
    ///   * day - Day of month, starting from 1
    pub fn to_date(&self) -> (u32, u32, u32) {
        mjd_utc2date(self.as_mjd_with_scale(TimeScale::UTC))
    }

    /// Convert AstroTime to Gregorian date and timewith given scale
    ///
    /// # Arguments
    ///
    /// * `scale` - Time scale of returned Gregorian date, e.g. UTC, GPS,
    ///
    /// # Returns
    ///
    /// * 6-element tuple with following values:
    ///   * `year` - the year
    ///   * `month` - the month, 1 based (1=January, 2=February, ...)
    ///   * `day` - Day of month, starting from 1
    ///   * `hour` - Hour of day, in range \[0,23\]
    ///   * `min` - Minute of hour, in range \[0,59\]
    ///   * `sec` - Second of minute, including fractions for subsecond, in range \[0,1)
    ///
    pub fn as_datetime_with_scale(&self, scale: Scale) -> (u32, u32, u32, u32, u32, f64) {
        let mjd_utc = self.as_mjd_with_scale(TimeScale);
        let (year, month, day) = mjd_utc2date(mjd_utc);
        let fracofday: f64 = mjd_utc - mjd_utc.floor();
        let mut sec: f64 = fracofday * 86400.0;
        let hour: u32 = std::cmp::min((sec / 3600.0).floor() as u32, 23);
        let min: u32 = std::cmp::min((sec as u32 - hour * 3600) / 60_u32, 59);
        sec = sec - hour as f64 * 3600.0 - min as f64 * 60.0;

        (year, month, day, hour, min, sec)
    }

    /// Convert AstroTime to UTC Gregorian date and time
    ///
    /// # Returns
    ///
    /// * 6-element tuple with following values:
    ///   * `year` - the year
    ///   * `month` - the month, 1 based (1=January, 2=February, ...)
    ///   * `day` - Day of month, starting from 1
    ///   * `hour` - Hour of day, in range \[0,23\]
    ///   * `min` - Minute of hour, in range \[0,59\]
    ///   * `sec` - Second of minute, including fractions for subsecond, in range \[0,1)
    ///
    #[inline]
    pub fn as_datetime(&self) -> (u32, u32, u32, u32, u32, f64) {
        self.as_datetime_with_scale(Scale::UTC)
    }

    /// Convert UTC Gregorian date and time to AstroTime
    ///
    /// # Arguments
    ///
    /// * `year` - the year (u32)
    /// * `month` - the month, 1 based (1=January, 2=February, ...)
    /// * `day` - Day of month, starting from 1
    /// * `hour` - Hour of day, in range \[0,23\]
    /// * `min` - Minute of hour, in range \[0,59\]
    /// * `sec` - Second of minute, including fractions for subsecond, in range \[0,1)
    ///
    /// # Return
    ///
    /// * AstroTime object
    #[inline]
    pub fn from_datetime(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
        sec: f64,
    ) -> Instant {
        Instant::from_datetime_with_scale(year, month, day, hour, min, sec, Scale::UTC)
    }

    /// Convert UTC Gregorian date and time to AstroTime
    ///
    /// # Arguments
    ///
    /// * `year` - the year (u32)
    /// * `month` - the month, 1 based (1=January, 2=February, ...)
    /// * `day` - Day of month, starting from 1
    /// * `hour` - Hour of day, in range \[0,23\]
    /// * `min` - Minute of hour, in range \[0,59\]
    /// * `sec` - Second of minute, including fractions for subsecond, in range \[0,1)
    /// * `scale` - Time Scale represented by input time, e.g. UTC, GPS
    ///
    /// # Return
    ///
    /// * AstroTime object
    pub fn from_datetime_with_scale(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
        sec: f64,
        scale: Scale,
    ) -> Instant {
        let mut mjd: f64 = date2mjd_utc(year, month, day) as f64;
        mjd += (((min + (hour * 60)) * 60) as f64 + sec) / 86400.0;
        Instant::from_mjd(mjd, scale)
    }

    /// Convert to modified Julian day (MJD), with given scale
    ///
    /// Modified Julian Day is the days (including fractional) since
    /// midnight on Nov 17, 1858.  The MJD can be computed by
    /// subtracting 2400000.5 from the Julian day
    ///
    pub fn to_mjd(&self, scale: Scale) -> f64 {
        match scale {
            Scale::TAI => self.mjd_tai,
            Scale::GPS => {
                // GPS tracks TAI - 19 seconds
                // after Jan 1 1980, & prior is
                // undefined, but we'll just set it to UTC
                if self.mjd_tai > TAIGPS0 {
                    self.mjd_tai - 19.0 / 86400.0
                } else {
                    self.mjd_tai + mjd_tai2utc_seconds(self.mjd_tai) / 86400.0
                }
            }
            Scale::TT => self.mjd_tai + 32.184 / 86400.0,
            Scale::UT1 => {
                // First convert to UTC
                let utc: f64 = self.mjd_tai + mjd_tai2utc_seconds(self.mjd_tai) / 86400.0;

                // Then convert to UT1
                // First earth-orientation parameter is dut1
                // which is (UT1 - UTC) in seconds
                utc + eop::eop_from_mjd_utc(utc).unwrap()[0] / 86400.0
            }

            Scale::UTC => self.mjd_tai + mjd_tai2utc_seconds(self.mjd_tai) / 86400.0,
            Scale::INVALID => -1.0,
            Scale::TDB => {
                let tt: f64 = self.mjd_tai + 32.184 / 86400.0;
                let ttc: f64 = (tt - (2451545.0 - 2400000.4)) / 36525.0;
                tt + 0.001657 / 86400.0 * (PI / 180.0 * (628.3076 * ttc + 6.2401)).sin()
            }
        }
    }

    /// Convert to Julian day, with given scale
    /// Julian day is total number of days (including fraction)
    /// since given epoch
    ///
    /// Jan 1 2000, 12pm is defined as Julian day 2451545.0
    pub fn to_jd(&self, scale: Scale) -> f64 {
        self.as_mjd_with_scale(TimeScale) + MJD2JD
    }

    /// Convert from Julian day, with given scale
    /// Julian day is total number of days (including fraction)
    /// since given epoch
    ///
    /// Jan 1 2000, 12pm is defined as Julian day 2451545.0
    pub fn from_jd(jd: f64, scale: Scale) -> Instant {
        Instant::from_mjd(jd + JD2MJD, scale)
    }

    /// Convert from string with given format
    /// using strftime format
    ///
    /// # Notes:
    /// * For format, see: <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>
    ///
    /// # Arguments
    ///
    /// * `s` - string to format
    /// * `format` - strftime format string
    ///
    /// # Returns
    /// * Result with AstroTime object
    ///
    pub fn strftime(s: &str, format: &str) -> SKResult<Instant> {
        let dt = chrono::NaiveDateTime::parse_from_str(s, format);
        match dt {
            Ok(v) => Ok(Instant::from(&v)),
            Err(e) => skerror!("Cannot parse datetime: {}", e),
        }
    }

    /// Create time from string with format guessed
    ///
    /// # Arguments
    ///
    /// * `s` - string to format
    ///
    /// # Returns
    ///
    /// * Result with AstroTime object
    ///
    /// # Exceptions
    ///
    /// * If string cannot be parsed
    ///
    pub fn from_string(s: &str) -> SKResult<Instant> {
        // Attempt to guess format
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S%.fZ",
            "%Y-%m-%d %H:%M:%S%.f %Z",
            "%Y-%m-%d %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S%.f %z",
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y-%j",
            "%Y-%j %H:%M:%S",
            "%Y-%j %H:%M:%S%.f",
            "%Y-%j %H:%M:%S%.fZ",
            "%Y-%j %H:%M:%S %Z",
            "%Y-%j %H:%M:%S %z",
            "%Y-%j %H:%M:%S%.f %z",
            "%Y-%j %H:%M:%S%.f",
            "%Y-%j %H:%M:%S",
            "%Y-%j %H:%M",
            "%Y-%j",
            "%Y-%m-%d %H:%M:%S%.f %Z",
            "%Y-%m-%d %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S%.f %z",
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y-%j",
            "%Y-%j %H:%M:%S",
            "%Y-%j %H:%M:%S%.f",
            "%Y-%j %H:%M:%S%.fZ",
            "%Y-%j %H:%M:%S %Z",
            "%Y-%j %H:%M:%S %z",
            "%Y-%j %H:%M:%S%.f %z",
            "%Y-%j %H:%M:%S%.f",
            "%Y-%j %H:%M:%S",
            "%Y-%j %H:%M",
            "%Y-%j",
            "%Y-%m-%d %H:%M:%S%.f %Z",
            "%Y-%m-%d %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S%.f %z",
            "%Y/%m/%d %H:%M:%S %z",
            "%Y/%m/%d %H:%M:%S%.f %z",
            "%Y/%m/%d %H:%M:%S %Z",
            "%Y/%m/%d %H:%M:%S%.f %Z",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S%.f",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%Y/%j",
            "%Y/%j %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S%.f",
            "%m/%d/%Y %H:%M:%S %Z",
            "%m/%d/%Y %H:%M:%S%.f %Z",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S%.f",
            "%m/%d/%Y %H:%M",
        ];

        for f in formats.iter() {
            let dt = chrono::NaiveDateTime::parse_from_str(s, f);
            if let Ok(dt) = dt {
                return Ok(Instant::from(&dt));
            }
        }
        skerror!("Cannot parse datetime")
    }

    /// Convert from string with RFC3339 format (overlaps with ISO8601)
    ///
    /// # Arguments
    ///
    /// * `s` - string to format
    ///
    /// # Returns
    ///
    /// * Result with AstroTime object
    ///
    /// # Exceptions
    ///
    /// * If string cannot be parsed
    ///
    pub fn from_rfc3339(s: &str) -> SKResult<Instant> {
        let dt = chrono::DateTime::parse_from_rfc3339(s);
        match dt {
            Ok(v) => Ok(Instant::from(&v.naive_utc())),
            Err(e) => skerror!("Cannot parse datetime: {}", e),
        }
    }
}

/// (TAI - UTC) in seconds for an UTC input modified Julian date
fn mjd_utc2tai_seconds(mjd_utc: f64) -> f64 {
    if mjd_utc > UTC1972 {
        let utc1900: u64 = (mjd_utc as u64 - 15020) * 86400;
        let val = DELTAAT.iter().find(|&&x| x[0] < utc1900);
        val.unwrap_or(&[0, 0])[1] as f64
    } else {
        0.0
    }
}

fn mjd_tai2utc_seconds(mjd_tai: f64) -> f64 {
    if mjd_tai > TAI1972 {
        let tai1900: u64 = (mjd_tai as u64 - 15020) * 86400;
        let val = DELTAAT.iter().find(|&&x| (x[0] + x[1]) < tai1900);
        -(val.unwrap_or(&[0, 0])[1] as f64)
    } else {
        0.0
    }
}

fn mjd_utc2date(mjd_utc: f64) -> (u32, u32, u32) {
    // Chapter 15 "Calendars" section 15.11.3 of the
    // Explanatory Suppliment to the Astronomical Almanac
    const Y: i32 = 4716;
    const J: i32 = 1401;
    const M: i32 = 2;
    const N: i32 = 12;
    const R: i32 = 4;
    const P: i32 = 1461;
    const V: i32 = 3;
    const U: i32 = 5;
    const S: i32 = 153;
    const W: i32 = 2;
    const B: i32 = 274277;
    const C: i32 = -38;

    let jd: i32 = (0.5 + mjd_utc + MJD2JD) as i32;
    let mut f: i32 = jd + J;
    f = f + (((4 * jd + B) / 146097) * 3) / 4 + C;
    let e: i32 = R * f + V;
    let g: i32 = (e % P) / R;
    let h: i32 = U * g + W;

    let day: i32 = (h % S) / U + 1;
    let month: i32 = ((h / S + M) % N) + 1;
    let year: i32 = e / P - Y + (N + M - month) / N;

    (year as u32, month as u32, day as u32)
}

fn date2mjd_utc(year: i32, month: u32, day: u32) -> i32 {
    // Chapter 15 "Calendars" section 15.11.3 of the
    // Explanatory Suppliment to the Astronomical Almanac
    // Algorithm 3
    const Y: i32 = 4716;
    const J: i32 = 1401;
    const M: i32 = 2;
    const N: i32 = 12;
    const R: i32 = 4;
    const P: i32 = 1461;
    const Q: i32 = 0;
    const U: i32 = 5;
    const S: i32 = 153;
    const T: i32 = 2;
    const A: i32 = 184;
    const C: i32 = -38;

    let h: i32 = month as i32 - M;
    let g: i32 = year + Y - (N - h) / N;
    let f: i32 = (h - 1 + N) % N;
    let e: i32 = (P * g + Q) / R + (day as i32) - 1 - J;

    let mut jdn: i32 = e + (S * f + T) / U;
    jdn = jdn - (3 * ((g + A) / 100)) / 4 - C;

    jdn - 2400001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date2mjd_test() {
        // Truth is pulled from leap-seconds file
        // for these examples
        assert_eq!(59219, date2mjd_utc(2021, 1, 5));
        // Try a "leap day"
        assert_eq!(58908, date2mjd_utc(2020, 2, 29));
    }

    #[test]
    fn testchrono() {
        use chrono::prelude::*;

        //let dt = Utc.ymd(2014, 7, 8).and_hms(9, 10, 11); // `2014-07-08T09:10:11Z`
        let dt = Utc.with_ymd_and_hms(2014, 7, 8, 9, 10, 11).unwrap();

        let ts = Instant::from(dt);
        let (year, mon, day, hour, min, sec) = ts.as_datetime();
        assert_eq!(year, 2014);
        assert_eq!(mon, 7);
        assert_eq!(day, 8);
        assert_eq!(hour, 9);
        assert_eq!(min, 10);
        assert!(((sec - 11.0) / 11.0).abs() < 1.0e-5);
        let dt2 = chrono::NaiveDateTime::from(ts);
        println!("{}", dt2);
        println!("{}", ts);
    }

    #[test]
    fn testdatetime() {
        let tm: Instant = Instant::from_datetime(2021, 3, 4, 12, 45, 33.0);
        let (year, mon, day, hour, min, sec) = tm.as_datetime();
        assert_eq!(year, 2021);
        assert_eq!(mon, 3);
        assert_eq!(day, 4);
        assert_eq!(hour, 12);
        assert_eq!(min, 45);
        assert!(((sec - 33.0) / 33.0).abs() < 1.0e-5);
    }

    #[test]
    fn test_rfctime() {
        let tm = Instant::from_rfc3339("2021-03-04T12:45:33Z").unwrap();
        let (year, mon, day, hour, min, sec) = tm.as_datetime();
        assert_eq!(year, 2021);
        assert_eq!(mon, 3);
        assert_eq!(day, 4);
        assert_eq!(hour, 12);
        assert_eq!(min, 45);
        assert!(((sec - 33.0) / 33.0).abs() < 1.0e-5);
    }

    #[test]
    fn sub_test() {
        let tm1 = Instant::from_datetime(2024, 2, 3, 22, 0, 0.0);
        let tm2 = Instant::from_datetime(2024, 2, 3, 11, 0, 0.0);
        let diff = tm1 - tm2;
        println!("diff = {}", diff);
        let diff2 = tm2 - tm1;
        println!("diff2 = {}", diff2);
        println!("diff2 seconds = {}", diff2.seconds());
    }

    #[test]
    fn add_test() {
        let tm = Instant::from_datetime(2021, 3, 4, 11, 20, 41.0);
        let delta: f64 = 0.5;
        let tm2 = tm + delta;
        let (year, mon, day, hour, min, sec) = tm2.as_datetime();
        assert_eq!(year, 2021);
        assert_eq!(mon, 3);
        assert_eq!(day, 4);
        assert_eq!(hour, 23);
        assert_eq!(min, 20);
        assert!(((sec - 41.0) / 41.0).abs() < 1.0e-5);

        let dcalc: f64 = (tm2 - tm).days();
        assert!(((dcalc - delta) / delta).abs() < 1.0e-5);
    }

    #[test]
    fn test_vec() {
        let tm = Instant::from_datetime(2004, 4, 6, 7, 51, 28.386009);
        let v: &Vec<f64> = &vec![0.0, 1.0, 2.0, 3.0];
        println!("time vec = {:?}", tm + v);
    }

    #[test]
    fn test_deltaat() {
        // Pulled from Vallado example 3-14
        let tm = &Instant::from_datetime(2004, 4, 6, 7, 51, 28.386009);
        println!("tm = {}", tm);
        let dut1 =
            (tm.as_mjd_with_scale(TimeScale::UT1) - tm.as_mjd_with_scale(TimeScale::UTC)) * 86400.0;
        println!("dut1 = {} sec", dut1);
        let delta_at =
            (tm.as_mjd_with_scale(TimeScale::TAI) - tm.as_mjd_with_scale(TimeScale::UTC)) * 86400.0;
        println!("delta_at = {} sec", delta_at);
        println!(
            "delta at 2 = {}",
            mjd_utc2tai_seconds(tm.as_mjd_with_scale(TimeScale::UTC))
        );
    }

    #[test]
    fn mjd_utc2date_test() {
        let (year, mon, day) = mjd_utc2date(59219.0);
        assert_eq!(year, 2021);
        assert_eq!(mon, 1);
        assert_eq!(day, 5);

        let (year2, mon2, day2) = mjd_utc2date(58908.0);
        assert_eq!(year2, 2020);
        assert_eq!(mon2, 2);
        assert_eq!(day2, 29);
    }
}
