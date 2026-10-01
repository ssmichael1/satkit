//! Implement hifitime interoperability
//!
//! Conversions go through TAI, a continuous count of SI seconds in both
//! crates, so they never touch either crate's leap-second table: an
//! `Instant` and the `Epoch` it converts to are the same physical instant,
//! whatever UTC label each crate gives it. The labels agree from 1972 on;
//! before 1972 hifitime's UTC is TAI (IERS leap seconds only), up to
//! 9.89 s from satkit's rubber-second UTC. hifitime's TDB differs from
//! satkit's by up to ~50 µs over 1900–2100. Those models apply only when
//! an `Epoch` is built or read in those scales; the [`TimeLike`] impl
//! below always uses satkit's. An `Epoch` inside a leap second should stay
//! in TAI (as converted): moved to hifitime's UTC scale, it reads
//! `23:59:59.x` and converts back to TAI a second early.

use crate::Instant;

/// Nanoseconds from hifitime's TAI reference epoch (1900-01-01 00:00:00
/// TAI) to satkit's (1970-01-01 00:00:00 TAI): 25,567 days.
const NS_1900_TO_1970: i128 = 25_567 * 86_400 * 1_000_000_000;

/// Exact: satkit's ±292,000-year range fits well inside hifitime's
/// ±3.2 million years.
#[inline]
fn instant_to_epoch(inst: &Instant) -> hifitime::Epoch {
    hifitime::Epoch::from_tai_duration(hifitime::Duration::from_total_nanoseconds(
        inst.raw as i128 * 1000 + NS_1900_TO_1970,
    ))
}

/// To the nearest microsecond (half a microsecond rounds up, also before
/// 1970), saturating outside satkit's range rather than wrapping or
/// producing [`Instant::INVALID`].
#[inline]
fn epoch_to_instant(epoch: &hifitime::Epoch) -> Instant {
    let ns = epoch.to_tai_duration().total_nanoseconds() - NS_1900_TO_1970;
    let us = (ns + 500).div_euclid(1000);
    Instant::new(us.clamp(i64::MIN as i128 + 1, i64::MAX as i128) as i64)
}

impl From<Instant> for hifitime::Epoch {
    fn from(inst: Instant) -> Self {
        instant_to_epoch(&inst)
    }
}

impl From<&Instant> for hifitime::Epoch {
    fn from(inst: &Instant) -> Self {
        instant_to_epoch(inst)
    }
}

impl From<hifitime::Epoch> for Instant {
    fn from(epoch: hifitime::Epoch) -> Self {
        epoch_to_instant(&epoch)
    }
}

impl From<&hifitime::Epoch> for Instant {
    fn from(epoch: &hifitime::Epoch) -> Self {
        epoch_to_instant(epoch)
    }
}

mod hifitime_impls {
    use super::epoch_to_instant;
    use crate::{Instant, TimeLike, TimeScale};

    /// Converts to an [`Instant`] first, so UT1 (satkit's Earth orientation
    /// table) and TDB (satkit's series) are satkit's, not hifitime's.
    impl TimeLike for hifitime::Epoch {
        #[inline]
        fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
            epoch_to_instant(self).as_mjd_with_scale(scale)
        }

        #[inline]
        fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
            epoch_to_instant(self).as_jd_with_scale(scale)
        }

        #[inline]
        fn as_instant(&self) -> Instant {
            epoch_to_instant(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Duration, TimeLike, TimeScale};
    use hifitime::Epoch;

    /// The reference epochs line up: 1970-01-01 00:00:00 TAI is raw 0.
    #[test]
    fn test_hifitime_epoch_alignment() {
        let e: Epoch = Instant::new(0).into();
        assert_eq!(e, Epoch::from_gregorian_tai_at_midnight(1970, 1, 1), "{e}");
        assert_eq!(Instant::from(e), Instant::new(0));
    }

    /// After 1972 both crates' leap-second tables agree, so UTC calendar
    /// labels match exactly.
    #[test]
    fn test_hifitime_utc_labels() {
        for (y, mo, d, h, mi, s, ns) in [
            (1972, 1, 1, 0, 0, 0, 0),
            (1999, 12, 31, 23, 59, 59, 999_999_000),
            (2016, 12, 31, 23, 59, 59, 500_000_000),
            (2017, 1, 1, 0, 0, 0, 0),
            (2024, 6, 15, 12, 0, 0, 250_000_000),
            (2039, 2, 25, 21, 42, 35, 990_070_000),
        ] {
            let t = Instant::from_datetime(y, mo, d, h, mi, s as f64 + ns as f64 * 1e-9).unwrap();
            let e = Epoch::from_gregorian_utc(y, mo as u8, d as u8, h as u8, mi as u8, s, ns);
            assert_eq!(Epoch::from(t), e, "{t}");
            assert_eq!(Instant::from(e), t, "{e}");
        }
    }

    /// Before 1972 hifitime's UTC is TAI (it counts IERS leap seconds
    /// only), while satkit models the 1961–1971 rubber-second UTC. The
    /// conversion keeps the physical instant, so the UTC labels differ by
    /// satkit's TAI − UTC.
    #[test]
    fn test_hifitime_pre1972_utc() {
        for (y, mo, d, dat) in [
            (1955, 1, 1, 0.0),
            (1961, 1, 1, 1.422818),
            (1965, 6, 1, 3.835826),
            (1971, 12, 31, 9.889650),
        ] {
            let t = Instant::from_date(y, mo, d).unwrap();
            let e = Epoch::from_gregorian_utc_at_midnight(y, mo as u8, d as u8);
            assert_eq!(
                e,
                Epoch::from_gregorian_tai_at_midnight(y, mo as u8, d as u8)
            );
            assert_eq!(Instant::from(e) - t, Duration::from_seconds(-dat), "{t}");
        }
    }

    /// An instant inside a leap second is the same physical instant in
    /// both crates.
    #[test]
    fn test_hifitime_leap_second() {
        let before = Instant::from_datetime(2016, 12, 31, 23, 59, 59.0).unwrap();
        let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5).unwrap();
        let after = Instant::from_date(2017, 1, 1).unwrap();
        let (eb, el, ea) = (Epoch::from(before), Epoch::from(leap), Epoch::from(after));
        assert_eq!(el - eb, hifitime::Duration::from_milliseconds(1500.0));
        assert_eq!(ea - eb, hifitime::Duration::from_seconds(2.0));
        assert_eq!(Instant::from(el), leap);
    }

    /// The uniform scales agree exactly.
    #[test]
    fn test_hifitime_uniform_scales() {
        let t = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0).unwrap();
        let e = Epoch::from(t);
        let tai_utc = e.to_tai_seconds() - e.to_utc_seconds();
        assert_eq!(tai_utc, 37.0);
        let gps = Instant::from_gps_week_and_second(2318, 561_618.0);
        assert_eq!(gps, t);
        assert_eq!(
            Epoch::from(gps),
            Epoch::from_gpst_seconds(2318.0 * 604_800.0 + 561_618.0)
        );
        assert_eq!(Instant::from(Epoch::from_tt_seconds(e.to_tt_seconds())), t,);
    }

    /// Nanoseconds round to the nearest microsecond, before 1970 too.
    #[test]
    fn test_hifitime_rounding() {
        let base = Instant::from_datetime(1969, 12, 31, 23, 59, 59.0).unwrap();
        let eb = Epoch::from(base);
        for (ns, us) in [
            (0, 0),
            (499, 0),
            (500, 1),
            (1_499, 1),
            (-499, 0),
            (-501, -1),
        ] {
            let e = eb + hifitime::Duration::from_total_nanoseconds(ns);
            assert_eq!(
                Instant::from(e),
                base + Duration::from_microseconds(us),
                "{ns} ns"
            );
        }
    }

    /// Epochs beyond satkit's range saturate rather than wrap or become
    /// `Instant::INVALID`; satkit's whole range converts exactly.
    #[test]
    fn test_hifitime_range() {
        assert_eq!(
            Instant::from(Epoch::from(Instant::new(i64::MAX))).raw,
            i64::MAX
        );
        assert_eq!(
            Instant::from(Epoch::from(Instant::new(i64::MIN + 1))).raw,
            i64::MIN + 1
        );
        let far = Epoch::from_tai_duration(hifitime::Duration::MAX);
        assert_eq!(Instant::from(far).raw, i64::MAX);
        let far = Epoch::from_tai_duration(hifitime::Duration::MIN);
        assert_eq!(Instant::from(far).raw, i64::MIN + 1);
    }

    /// Random round trip, 1900–2100, microsecond resolution
    #[test]
    fn test_hifitime_random_roundtrip() {
        use rand::Rng;
        let mut rng = rand::rng();
        let lo = Instant::from_date(1900, 1, 1).unwrap().raw;
        let hi = Instant::from_date(2100, 1, 1).unwrap().raw;
        for _ in 0..20_000 {
            let t = Instant::new(rng.random_range(lo..hi));
            let e = Epoch::from(t);
            assert_eq!(e.to_tai_duration().total_nanoseconds() % 1000, 0);
            assert_eq!(Instant::from(e), t);
            assert_eq!(e.as_instant(), t);
        }
    }

    /// `TimeLike` uses satkit's models for every scale, including TDB and UT1.
    #[test]
    fn test_hifitime_timelike() {
        let t = Instant::from_datetime(2024, 6, 15, 18, 30, 45.5).unwrap();
        let e = Epoch::from(t);
        for scale in [
            TimeScale::UTC,
            TimeScale::TAI,
            TimeScale::TT,
            TimeScale::GPS,
            TimeScale::TDB,
            TimeScale::UT1,
        ] {
            assert_eq!(
                e.as_mjd_with_scale(scale),
                t.as_mjd_with_scale(scale),
                "{scale}"
            );
            assert_eq!(
                e.as_jd_with_scale(scale),
                t.as_jd_with_scale(scale),
                "{scale}"
            );
        }
    }
}
