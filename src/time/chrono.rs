//! Implement chrono interoperability
//!

use chrono::TimeZone;

use crate::Instant;

#[inline]
fn datetime_to_unixtime<Tz>(dt: &chrono::DateTime<Tz>) -> f64
where
    Tz: chrono::TimeZone,
{
    dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1.0e-9
}

impl From<Instant> for chrono::DateTime<chrono::Utc> {
    fn from(inst: Instant) -> Self {
        let unixtime = inst.as_unixtime();
        let secs = unixtime.trunc() as i64;
        let nsecs = ((unixtime.fract()) * 1.0e9) as u32;
        chrono::Utc.timestamp_opt(secs, nsecs).unwrap()
    }
}

impl From<&Instant> for chrono::DateTime<chrono::Utc> {
    fn from(inst: &Instant) -> Self {
        let unixtime = inst.as_unixtime();
        let secs = unixtime.trunc() as i64;
        let nsecs = ((unixtime.fract()) * 1.0e9) as u32;
        chrono::Utc.timestamp_opt(secs, nsecs).unwrap()
    }
}

impl<TZ> From<chrono::DateTime<TZ>> for Instant
where
    TZ: chrono::TimeZone,
{
    fn from(dt: chrono::DateTime<TZ>) -> Self {
        Instant::from_unixtime(datetime_to_unixtime(&dt))
    }
}

impl<TZ> From<&chrono::DateTime<TZ>> for Instant
where
    TZ: chrono::TimeZone,
{
    fn from(dt: &chrono::DateTime<TZ>) -> Self {
        Instant::from_unixtime(datetime_to_unixtime(&dt))
    }
}

mod chrono_impls {
    use crate::{Instant, TimeLike, TimeScale};
    use super::datetime_to_unixtime;

    impl<Tz> TimeLike for chrono::DateTime<Tz>
    where
        Tz: chrono::TimeZone,
    {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            let unixtime = datetime_to_unixtime(&self);
            Instant::from_unixtime(unixtime).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            let unixtime = datetime_to_unixtime(&self);
            Instant::from_unixtime(unixtime).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            Instant::from_unixtime(datetime_to_unixtime(&self))
        }
    }
}



#[cfg(test)]
mod tests {
    use crate::{TimeLike, TimeScale};

    use super::*;

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
        assert!((mjd_instant - mjd_chrono).abs() < 1.0e-9,
                "MJD UTC mismatch: {} vs {}", mjd_instant, mjd_chrono);

        // Test TAI scale
        let mjd_instant_tai = inst.as_mjd_with_scale(TimeScale::TAI);
        let mjd_chrono_tai = dt.as_mjd_with_scale(TimeScale::TAI);
        assert!((mjd_instant_tai - mjd_chrono_tai).abs() < 1.0e-9,
                "MJD TAI mismatch: {} vs {}", mjd_instant_tai, mjd_chrono_tai);

        // Test TT scale
        let mjd_instant_tt = inst.as_mjd_with_scale(TimeScale::TT);
        let mjd_chrono_tt = dt.as_mjd_with_scale(TimeScale::TT);
        assert!((mjd_instant_tt - mjd_chrono_tt).abs() < 1.0e-9,
                "MJD TT mismatch: {} vs {}", mjd_instant_tt, mjd_chrono_tt);
    }

    #[test]
    fn test_timelike_trait_jd_conversion() {
        // Test that Instant and chrono::DateTime produce the same JD values
        let inst = Instant::from_datetime(2000, 1, 1, 12, 0, 0.0).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst);

        // Test UTC scale
        let jd_instant = inst.as_jd_with_scale(TimeScale::UTC);
        let jd_chrono = dt.as_jd_with_scale(TimeScale::UTC);
        assert!((jd_instant - jd_chrono).abs() < 1.0e-9,
                "JD UTC mismatch: {} vs {}", jd_instant, jd_chrono);

        // Test that JD = MJD + 2400000.5
        let mjd_instant = inst.as_mjd_with_scale(TimeScale::UTC);
        assert!((jd_instant - (mjd_instant + 2400000.5)).abs() < 1.0e-12,
                "JD-MJD relationship incorrect");
    }

    #[test]
    fn test_timelike_trait_as_instant() {
        // Test conversion back to Instant
        let inst1 = Instant::from_datetime(2024, 12, 25, 6, 30, 15.123).unwrap();
        let dt: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(inst1);
        let inst2 = dt.as_instant();

        assert!((inst1.as_unixtime() - inst2.as_unixtime()).abs() < 1.0e-9,
                "as_instant() conversion failed");
    }

    #[test]
    fn test_timelike_trait_with_timezone() {
        // Test with different timezone (should produce same results since we convert to UTC)
        let dt_utc = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 12, 0, 0).unwrap();
        let dt_fixed = chrono::DateTime::<chrono::FixedOffset>::from(dt_utc);

        let mjd_utc = dt_utc.as_mjd_with_scale(TimeScale::UTC);
        let mjd_fixed = dt_fixed.as_mjd_with_scale(TimeScale::UTC);

        assert!((mjd_utc - mjd_fixed).abs() < 1.0e-9,
                "MJD mismatch between timezones: {} vs {}", mjd_utc, mjd_fixed);
    }

}
