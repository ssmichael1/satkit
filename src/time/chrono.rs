//! Implement chrono interoperability
//!

use chrono::TimeZone;

use crate::Instant;

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
        let unixtime = dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1.0e-9;
        Instant::from_unixtime(unixtime)
    }
}

impl<TZ> From<&chrono::DateTime<TZ>> for Instant
where
    TZ: chrono::TimeZone,
{
    fn from(dt: &chrono::DateTime<TZ>) -> Self {
        let unixtime = dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1.0e-9;
        Instant::from_unixtime(unixtime)
    }
}

mod chrono_impls {
    use crate::{Instant, TimeLike, TimeScale};

    impl<Tz> TimeLike for chrono::DateTime<Tz>
    where
        Tz: chrono::TimeZone,
    {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            let unixtime = self.timestamp() as f64 + self.timestamp_subsec_nanos() as f64 * 1.0e-9;
            Instant::from_unixtime(unixtime).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            let unixtime = self.timestamp() as f64 + self.timestamp_subsec_nanos() as f64 * 1.0e-9;
            Instant::from_unixtime(unixtime).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            let unixtime = self.timestamp() as f64 + self.timestamp_subsec_nanos() as f64 * 1.0e-9;
            Instant::from_unixtime(unixtime)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::TimeLike;

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
}
