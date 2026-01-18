use super::{Instant, TimeScale};

/// Trait for time-like types usable by satkit APIs that require time scale conversions.
///
/// Implementors provide Modified Julian Date (MJD) and Julian Date (JD) conversions
/// for a given `TimeScale`, plus a conversion to satkit's `Instant` in UTC.
///
/// This trait enables interoperability with external time types (for example
/// `chrono::DateTime`) while preserving satkit's time-scale aware calculations.
///
/// # Examples
///
/// ```
/// use satkit::{TimeLike, TimeScale, Instant};
///
/// let t = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
/// let mjd = t.as_mjd_with_scale(TimeScale::UTC);
/// let jd = t.as_jd_with_scale(TimeScale::UTC);
/// assert!((jd - (mjd + 2400000.5)).abs() < 1.0e-12);
/// ```
pub trait TimeLike {
    /// Modified Julian Date with the provided time scale.
    fn as_mjd_with_scale(&self, scale: TimeScale) -> f64;

    /// Julian Date with the provided time scale.
    fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
        self.as_mjd_with_scale(scale) + 2400000.5
    }

    /// Convert to a satkit `Instant` in UTC.
    ///
    /// The default implementation converts through MJD (UTC).
    ///
    /// Note: this is needed, as other time-like types may not accurately
    /// keep track of leap seconds, (e.g., for TAI or TT), so usage in
    /// functions such as sgp4 may be inaccurate
    fn as_instant(&self) -> Instant;
}

impl TimeLike for Instant {
    #[inline]
    fn as_mjd_with_scale(&self, scale: TimeScale) -> f64 {
        Instant::as_mjd_with_scale(self, scale)
    }

    #[inline]
    fn as_jd_with_scale(&self, scale: TimeScale) -> f64 {
        Instant::as_jd_with_scale(self, scale)
    }

    #[inline]
    fn as_instant(&self) -> Instant {
        *self
    }
}

#[cfg(feature = "chrono")]
mod chrono_impls {
    use super::{Instant, TimeLike, TimeScale};

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
