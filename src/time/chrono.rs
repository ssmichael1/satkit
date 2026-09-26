//! Implement chrono interoperability
//!

use chrono::TimeZone;

use crate::Instant;

/// Exact conversion (to the nearest microsecond) through integer Unix
/// microseconds rather than an f64 Unix time, which resolves only ~0.2 µs
/// today and loses a microsecond to truncation about 1% of the time.
///
/// chrono writes a leap second as `23:59:59` with a nanosecond field of
/// 1e9 or more; the whole second goes through the Unix basis and the
/// sub-second part is elapsed time, which lands inside the leap second.
/// (`timestamp()` is the floor, and the nanoseconds are non-negative, also
/// before 1970.)
#[inline]
fn datetime_to_instant<Tz>(dt: &chrono::DateTime<Tz>) -> Instant
where
    Tz: chrono::TimeZone,
{
    let whole = Instant::from_unixtime_microseconds(dt.timestamp().saturating_mul(1_000_000));
    let us = (dt.timestamp_subsec_nanos() as i64 + 500) / 1000;
    whole + crate::Duration::from_microseconds(us)
}

/// Exact conversion from integer Unix microseconds, split with Euclidean
/// division so that instants before 1970 keep their (non-negative)
/// fraction of a second. (Splitting an f64 Unix time with `trunc()` /
/// `fract()` gave a negative fraction there, which saturated to zero:
/// `1969-12-31T23:59:59.5` became `1970-01-01T00:00:00`.)
///
/// Inside a leap second the Unix time repeats `23:59:59.x`, and so does
/// the result, as with the Python bindings' `time.to_datetime()`.
#[inline]
fn instant_to_datetime(inst: &Instant) -> chrono::DateTime<chrono::Utc> {
    let us = inst.as_unixtime_microseconds();
    let secs = us.div_euclid(1_000_000);
    let nsecs = (us.rem_euclid(1_000_000) * 1000) as u32;
    // chrono can only represent years within about +/- 262,000; `From`
    // cannot fail, so saturate instants outside that range rather than
    // panicking on the unwrap.
    chrono::Utc
        .timestamp_opt(secs, nsecs)
        .single()
        .unwrap_or(if secs < 0 {
            chrono::DateTime::<chrono::Utc>::MIN_UTC
        } else {
            chrono::DateTime::<chrono::Utc>::MAX_UTC
        })
}

impl From<Instant> for chrono::DateTime<chrono::Utc> {
    fn from(inst: Instant) -> Self {
        instant_to_datetime(&inst)
    }
}

impl From<&Instant> for chrono::DateTime<chrono::Utc> {
    fn from(inst: &Instant) -> Self {
        instant_to_datetime(inst)
    }
}

impl<TZ> From<chrono::DateTime<TZ>> for Instant
where
    TZ: chrono::TimeZone,
{
    fn from(dt: chrono::DateTime<TZ>) -> Self {
        datetime_to_instant(&dt)
    }
}

impl<TZ> From<&chrono::DateTime<TZ>> for Instant
where
    TZ: chrono::TimeZone,
{
    fn from(dt: &chrono::DateTime<TZ>) -> Self {
        datetime_to_instant(dt)
    }
}

mod chrono_impls {
    use super::datetime_to_instant;
    use crate::{Instant, TimeLike, TimeScale};

    impl<Tz> TimeLike for chrono::DateTime<Tz>
    where
        Tz: chrono::TimeZone,
    {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            datetime_to_instant(self).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            datetime_to_instant(self).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            datetime_to_instant(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{TimeLike, TimeScale};

    use super::*;

    #[test]
    fn test_extreme_instant_saturates() {
        // Instants beyond chrono's representable range (about +/- 262,000
        // years) must saturate rather than panic in the From impl
        let dt: chrono::DateTime<chrono::Utc> = Instant::new(i64::MAX).into();
        assert_eq!(dt, chrono::DateTime::<chrono::Utc>::MAX_UTC);
        let dt: chrono::DateTime<chrono::Utc> = Instant::new(i64::MIN + 1).into();
        assert_eq!(dt, chrono::DateTime::<chrono::Utc>::MIN_UTC);
    }

    fn utc(
        y: i32,
        mo: u32,
        d: u32,
        h: u32,
        mi: u32,
        s: u32,
        us: u32,
    ) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .with_ymd_and_hms(y, mo, d, h, mi, s)
            .unwrap()
            .checked_add_signed(chrono::TimeDelta::microseconds(us as i64))
            .unwrap()
    }

    /// Conversions both ways are exact to the microsecond (no f64 Unix
    /// time), including fractional seconds before 1970.
    #[test]
    fn test_chrono_exact() {
        let base = Instant::from_datetime(2039, 2, 25, 21, 42, 35.0).unwrap();
        for us in [0, 1, 249, 990_070, 999_999] {
            let t = base + crate::Duration::from_microseconds(us);
            let dt: chrono::DateTime<chrono::Utc> = t.into();
            assert_eq!(dt, utc(2039, 2, 25, 21, 42, 35, us as u32));
            assert_eq!(Instant::from(dt), t);
        }
        // Before 1970 the f64 fraction was negative and saturated to zero
        for (t, dt) in [
            (
                Instant::from_datetime(1969, 12, 31, 23, 59, 59.5).unwrap(),
                utc(1969, 12, 31, 23, 59, 59, 500_000),
            ),
            (
                Instant::from_datetime(1960, 6, 1, 12, 0, 0.75).unwrap(),
                utc(1960, 6, 1, 12, 0, 0, 750_000),
            ),
            (
                Instant::from_datetime(1969, 12, 31, 23, 59, 59.999999).unwrap(),
                utc(1969, 12, 31, 23, 59, 59, 999_999),
            ),
            (
                Instant::from_datetime(1900, 1, 1, 0, 0, 0.000001).unwrap(),
                utc(1900, 1, 1, 0, 0, 0, 1),
            ),
        ] {
            let got: chrono::DateTime<chrono::Utc> = t.into();
            assert_eq!(got, dt, "{t}");
            assert_eq!(Instant::from(dt), t, "{dt}");
            assert_eq!(dt.as_instant(), t, "{dt}");
        }
    }

    /// A leap second comes back as `23:59:59.x` (Unix-time convention, as
    /// the Python `to_datetime()`); chrono's own leap-second form
    /// (nanos >= 1e9) converts into the leap second.
    #[test]
    fn test_chrono_leap_second() {
        let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = leap.into();
        assert_eq!(dt, utc(2016, 12, 31, 23, 59, 59, 500_000));
        let before = Instant::from_datetime(2016, 12, 31, 23, 59, 59.5).unwrap();
        assert_eq!(Instant::from(dt), before);
        let chrono_leap = chrono::NaiveDate::from_ymd_opt(2016, 12, 31)
            .unwrap()
            .and_hms_micro_opt(23, 59, 59, 1_500_000)
            .unwrap()
            .and_utc();
        assert_eq!(Instant::from(chrono_leap), leap);
        let after = Instant::from_date(2017, 1, 1).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = after.into();
        assert_eq!(dt, utc(2017, 1, 1, 0, 0, 0, 0));
    }

    /// Random round trip, 1900–2100, microsecond resolution
    #[test]
    fn test_chrono_random_roundtrip() {
        use rand::Rng;
        let mut rng = rand::rng();
        let lo = Instant::from_date(1900, 1, 1)
            .unwrap()
            .as_unixtime_microseconds();
        let hi = Instant::from_date(2100, 1, 1)
            .unwrap()
            .as_unixtime_microseconds();
        for _ in 0..20_000 {
            let us = rng.random_range(lo..hi);
            let t = Instant::from_unixtime_microseconds(us);
            let dt: chrono::DateTime<chrono::Utc> = t.into();
            assert_eq!(dt.timestamp_micros(), us);
            assert_eq!(Instant::from(dt), t);
        }
    }

    #[test]
    fn test_instant_chrono_conversion() {
        let inst = Instant::from_datetime(2024, 1, 1, 12, 0, 0.0).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst);
        let inst_converted = Instant::from(dt);
        assert!((inst.as_unixtime() - inst_converted.as_unixtime()).abs() < 1.0e-9);
        let inst2 = dt.as_instant();
        assert!((inst.as_unixtime() - inst2.as_unixtime()).abs() < 1.0e-9);
    }

    #[test]
    fn test_timelike_trait_mjd_conversion() {
        // Test that Instant and chrono::DateTime produce the same MJD values
        let inst = Instant::from_datetime(2024, 6, 15, 18, 30, 45.5).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst);

        // Test UTC scale
        let mjd_instant = inst.as_mjd_with_scale(TimeScale::UTC);
        let mjd_chrono = dt.as_mjd_with_scale(TimeScale::UTC);
        assert!(
            (mjd_instant - mjd_chrono).abs() < 1.0e-9,
            "MJD UTC mismatch: {} vs {}",
            mjd_instant,
            mjd_chrono
        );

        // Test TAI scale
        let mjd_instant_tai = inst.as_mjd_with_scale(TimeScale::TAI);
        let mjd_chrono_tai = dt.as_mjd_with_scale(TimeScale::TAI);
        assert!(
            (mjd_instant_tai - mjd_chrono_tai).abs() < 1.0e-9,
            "MJD TAI mismatch: {} vs {}",
            mjd_instant_tai,
            mjd_chrono_tai
        );

        // Test TT scale
        let mjd_instant_tt = inst.as_mjd_with_scale(TimeScale::TT);
        let mjd_chrono_tt = dt.as_mjd_with_scale(TimeScale::TT);
        assert!(
            (mjd_instant_tt - mjd_chrono_tt).abs() < 1.0e-9,
            "MJD TT mismatch: {} vs {}",
            mjd_instant_tt,
            mjd_chrono_tt
        );
    }

    #[test]
    fn test_timelike_trait_jd_conversion() {
        // Test that Instant and chrono::DateTime produce the same JD values
        let inst = Instant::from_datetime(2000, 1, 1, 12, 0, 0.0).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst);

        // Test UTC scale
        let jd_instant = inst.as_jd_with_scale(TimeScale::UTC);
        let jd_chrono = dt.as_jd_with_scale(TimeScale::UTC);
        assert!(
            (jd_instant - jd_chrono).abs() < 1.0e-9,
            "JD UTC mismatch: {} vs {}",
            jd_instant,
            jd_chrono
        );

        // Test that JD = MJD + 2400000.5
        let mjd_instant = inst.as_mjd_with_scale(TimeScale::UTC);
        assert!(
            (jd_instant - (mjd_instant + 2400000.5)).abs() < 1.0e-12,
            "JD-MJD relationship incorrect"
        );
    }

    #[test]
    fn test_timelike_trait_as_instant() {
        // Test conversion back to Instant
        let inst1 = Instant::from_datetime(2024, 12, 25, 6, 30, 15.123).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst1);
        let inst2 = dt.as_instant();

        assert!(
            (inst1.as_unixtime() - inst2.as_unixtime()).abs() < 1.0e-9,
            "as_instant() conversion failed"
        );
    }

    #[test]
    fn test_timelike_trait_with_timezone() {
        // Test with different timezone (should produce same results since we convert to UTC)
        let dt_utc = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 12, 0, 0).unwrap();
        let dt_fixed = chrono::DateTime::<chrono::FixedOffset>::from(dt_utc);

        let mjd_utc = dt_utc.as_mjd_with_scale(TimeScale::UTC);
        let mjd_fixed = dt_fixed.as_mjd_with_scale(TimeScale::UTC);

        assert!(
            (mjd_utc - mjd_fixed).abs() < 1.0e-9,
            "MJD mismatch between timezones: {} vs {}",
            mjd_utc,
            mjd_fixed
        );
    }
}
