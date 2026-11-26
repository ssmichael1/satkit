//! Implement jiff interoperability
//!

use crate::Instant;

/// Exact conversion (to the nearest microsecond) through integer Unix
/// nanoseconds rather than an f64 Unix time.
///
/// `div_euclid` rounds toward negative infinity, so instants before 1970
/// round the same way as those after (half a microsecond rounds up).
/// jiff has no leap seconds: a timestamp is Unix time.
#[inline]
fn timestamp_to_instant(ts: jiff::Timestamp) -> Instant {
    // jiff's range is years -9999 to 9999, well inside i64 microseconds
    let us = (ts.as_nanosecond() + 500).div_euclid(1000);
    Instant::from_unixtime_microseconds(us as i64)
}

/// Exact conversion from integer Unix microseconds.
///
/// Inside a leap second the Unix time repeats `23:59:59.x`, and so does
/// the result, as with the Python bindings' `time.to_datetime()`.
#[inline]
fn instant_to_timestamp(inst: &Instant) -> jiff::Timestamp {
    let us = inst.as_unixtime_microseconds();
    // jiff represents years -9999 to 9999; `From` cannot fail, so saturate
    // instants outside that range.
    jiff::Timestamp::from_microsecond(us).unwrap_or(if us < 0 {
        jiff::Timestamp::MIN
    } else {
        jiff::Timestamp::MAX
    })
}

impl From<Instant> for jiff::Timestamp {
    fn from(inst: Instant) -> Self {
        instant_to_timestamp(&inst)
    }
}

impl From<&Instant> for jiff::Timestamp {
    fn from(inst: &Instant) -> Self {
        instant_to_timestamp(inst)
    }
}

impl From<jiff::Timestamp> for Instant {
    fn from(ts: jiff::Timestamp) -> Self {
        timestamp_to_instant(ts)
    }
}

impl From<&jiff::Timestamp> for Instant {
    fn from(ts: &jiff::Timestamp) -> Self {
        timestamp_to_instant(*ts)
    }
}

impl From<jiff::ZonedDateTime> for Instant {
    fn from(zdt: jiff::ZonedDateTime) -> Self {
        timestamp_to_instant(zdt.timestamp())
    }
}

impl From<&jiff::ZonedDateTime> for Instant {
    fn from(zdt: &jiff::ZonedDateTime) -> Self {
        timestamp_to_instant(zdt.timestamp())
    }
}

mod jiff_impls {
    use super::timestamp_to_instant;
    use crate::{Instant, TimeLike, TimeScale};

    impl TimeLike for jiff::Timestamp {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            timestamp_to_instant(*self).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            timestamp_to_instant(*self).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            timestamp_to_instant(*self)
        }
    }

    impl TimeLike for jiff::ZonedDateTime {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            timestamp_to_instant(self.timestamp()).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            timestamp_to_instant(self.timestamp()).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            timestamp_to_instant(self.timestamp())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{TimeLike, TimeScale};

    use super::*;

    fn ts(s: &str) -> jiff::Timestamp {
        s.parse().unwrap()
    }

    #[test]
    fn test_extreme_instant_saturates() {
        // Instants beyond jiff's range (years -9999 to 9999) must saturate
        // rather than panic in the From impl
        let t: jiff::Timestamp = Instant::new(i64::MAX).into();
        assert_eq!(t, jiff::Timestamp::MAX);
        let t: jiff::Timestamp = Instant::new(i64::MIN + 1).into();
        assert_eq!(t, jiff::Timestamp::MIN);
    }

    /// Conversions both ways are exact to the microsecond, including
    /// fractional seconds before 1970.
    #[test]
    fn test_jiff_exact() {
        for (t, j) in [
            (
                Instant::from_datetime(2039, 2, 25, 21, 42, 35.0).unwrap()
                    + crate::Duration::from_microseconds(990_070),
                ts("2039-02-25T21:42:35.990070Z"),
            ),
            (
                Instant::from_datetime(1969, 12, 31, 23, 59, 59.5).unwrap(),
                ts("1969-12-31T23:59:59.5Z"),
            ),
            (
                Instant::from_datetime(1960, 6, 1, 12, 0, 0.75).unwrap(),
                ts("1960-06-01T12:00:00.75Z"),
            ),
            (
                Instant::from_datetime(1969, 12, 31, 23, 59, 59.999999).unwrap(),
                ts("1969-12-31T23:59:59.999999Z"),
            ),
            (
                Instant::from_datetime(1900, 1, 1, 0, 0, 0.000001).unwrap(),
                ts("1900-01-01T00:00:00.000001Z"),
            ),
        ] {
            let got: jiff::Timestamp = t.into();
            assert_eq!(got, j, "{t}");
            assert_eq!(Instant::from(j), t, "{j}");
            assert_eq!(Instant::from(&j), t, "{j}");
            assert_eq!(j.as_instant(), t, "{j}");
        }
    }

    /// jiff resolves nanoseconds; satkit rounds to the nearest microsecond,
    /// the same way on both sides of 1970.
    #[test]
    fn test_jiff_rounds_nanoseconds() {
        let base = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0).unwrap();
        for (ns, us) in [(499, 0), (500, 1), (1499, 1), (1500, 2)] {
            let j = ts("2024-06-15T12:00:00Z")
                .checked_add(jiff::SignedDuration::from_nanos(ns))
                .unwrap();
            assert_eq!(
                Instant::from(j),
                base + crate::Duration::from_microseconds(us),
                "{ns} ns"
            );
        }
        let base = Instant::from_datetime(1960, 6, 1, 12, 0, 0.0).unwrap();
        for (ns, us) in [(499, 0), (500, 1), (-500, 0), (-501, -1)] {
            let j = ts("1960-06-01T12:00:00Z")
                .checked_add(jiff::SignedDuration::from_nanos(ns))
                .unwrap();
            assert_eq!(
                Instant::from(j),
                base + crate::Duration::from_microseconds(us),
                "{ns} ns"
            );
        }
    }

    /// A leap second comes back as `23:59:59.x` (Unix-time convention, as
    /// the Python `to_datetime()`).
    #[test]
    fn test_jiff_leap_second() {
        let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5).unwrap();
        let j: jiff::Timestamp = leap.into();
        assert_eq!(j, ts("2016-12-31T23:59:59.5Z"));
        let before = Instant::from_datetime(2016, 12, 31, 23, 59, 59.5).unwrap();
        assert_eq!(Instant::from(j), before);
        let after = Instant::from_date(2017, 1, 1).unwrap();
        let j: jiff::Timestamp = after.into();
        assert_eq!(j, ts("2017-01-01T00:00:00Z"));
    }

    /// Random round trip, 1900–2100, microsecond resolution
    #[test]
    fn test_jiff_random_roundtrip() {
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
            let j: jiff::Timestamp = t.into();
            assert_eq!(j.as_microsecond(), us);
            assert_eq!(Instant::from(j), t);
        }
    }

    #[test]
    fn test_timelike_trait_matches_instant() {
        let inst = Instant::from_datetime(2024, 6, 15, 18, 30, 45.5).unwrap();
        let j: jiff::Timestamp = inst.into();
        for scale in [TimeScale::UTC, TimeScale::TAI, TimeScale::TT] {
            assert_eq!(
                j.as_mjd_with_scale(scale),
                inst.as_mjd_with_scale(scale),
                "{scale:?}"
            );
            assert_eq!(
                TimeLike::as_jd_with_scale(&j, scale),
                inst.as_jd_with_scale(scale),
                "{scale:?}"
            );
        }
    }

    /// A zoned datetime converts by its instant, whatever its time zone
    #[test]
    fn test_zoned_datetime() {
        let j = ts("2024-03-15T12:00:00Z");
        let inst = Instant::from_datetime(2024, 3, 15, 12, 0, 0.0).unwrap();
        for tz in [
            jiff::tz::TimeZone::UTC,
            jiff::tz::TimeZone::fixed(jiff::tz::Offset::constant(-5)),
        ] {
            let zdt = j.to_zoned_datetime(tz);
            assert_eq!(Instant::from(&zdt), inst);
            assert_eq!(zdt.as_instant(), inst);
            assert_eq!(
                zdt.as_mjd_with_scale(TimeScale::TT),
                inst.as_mjd_with_scale(TimeScale::TT)
            );
            assert_eq!(Instant::from(zdt), inst);
        }
    }
}
