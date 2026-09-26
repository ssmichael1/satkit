//! Arithmetic on [`Instant`] and [`Duration`].
//!
//! Both are `i64` microsecond counts, good for about ±292,000 years. The
//! operators (`+`, `-`, `+=`, `-=`) **saturate** at the ends of that range,
//! in debug and release builds alike: they never panic and never wrap
//! around, matching the MJD / unixtime / day-count constructors, which
//! saturate too. A saturated value is not a meaningful time; to detect
//! overflow instead, use [`Instant::checked_add`], [`Instant::checked_sub`],
//! [`Instant::checked_duration_since`], [`Duration::checked_add`] and
//! [`Duration::checked_sub`], which return `None` (the Python bindings use
//! them and raise `OverflowError`).

use super::Duration;
use super::Instant;

impl std::ops::Add<Duration> for Instant {
    type Output = Self;

    fn add(self, other: Duration) -> Self {
        Self {
            raw: self.raw.saturating_add(other.usec),
        }
    }
}

impl std::ops::Add<Duration> for &Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        Instant {
            raw: self.raw.saturating_add(other.usec),
        }
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Self;

    fn sub(self, other: Duration) -> Self {
        Self {
            raw: self.raw.saturating_sub(other.usec),
        }
    }
}

impl std::ops::Sub<Duration> for &Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        Instant {
            raw: self.raw.saturating_sub(other.usec),
        }
    }
}

impl std::ops::Sub<Instant> for &Instant {
    type Output = Duration;

    fn sub(self, other: Instant) -> Duration {
        Duration {
            usec: self.raw.saturating_sub(other.raw),
        }
    }
}

impl std::ops::Sub<Self> for Instant {
    type Output = Duration;

    fn sub(self, other: Self) -> Duration {
        Duration {
            usec: self.raw.saturating_sub(other.raw),
        }
    }
}

impl std::ops::AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        self.raw = self.raw.saturating_add(other.usec);
    }
}

impl std::ops::SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        self.raw = self.raw.saturating_sub(other.usec);
    }
}

/// Add two durations together
impl std::ops::Add<Self> for Duration {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            usec: self.usec.saturating_add(other.usec),
        }
    }
}

impl std::ops::AddAssign<Self> for Duration {
    fn add_assign(&mut self, other: Self) {
        self.usec = self.usec.saturating_add(other.usec);
    }
}

impl std::ops::SubAssign<Self> for Duration {
    fn sub_assign(&mut self, other: Self) {
        self.usec = self.usec.saturating_sub(other.usec);
    }
}

/// Subtract two durations
impl std::ops::Sub<Self> for Duration {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            usec: self.usec.saturating_sub(other.usec),
        }
    }
}

impl Instant {
    /// `self + d`, or `None` if the result is outside the representable
    /// range (about ±292,000 years around 2000). The `+` operator saturates
    /// there instead.
    pub const fn checked_add(&self, d: Duration) -> Option<Self> {
        match self.raw.checked_add(d.usec) {
            Some(raw) => Some(Self { raw }),
            None => None,
        }
    }

    /// `self - d`, or `None` if the result is outside the representable
    /// range. The `-` operator saturates there instead.
    pub const fn checked_sub(&self, d: Duration) -> Option<Self> {
        match self.raw.checked_sub(d.usec) {
            Some(raw) => Some(Self { raw }),
            None => None,
        }
    }

    /// `self - earlier` as a [`Duration`], or `None` if the difference does
    /// not fit (only possible for instants near the ends of the range, such
    /// as [`Instant::INVALID`]). The `-` operator saturates there instead.
    pub const fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
        match self.raw.checked_sub(earlier.raw) {
            Some(usec) => Some(Duration { usec }),
            None => None,
        }
    }
}

impl Duration {
    /// `self + other`, or `None` on overflow (beyond about ±292,000 years).
    /// The `+` operator saturates there instead.
    pub const fn checked_add(&self, other: Self) -> Option<Self> {
        match self.usec.checked_add(other.usec) {
            Some(usec) => Some(Self { usec }),
            None => None,
        }
    }

    /// `self - other`, or `None` on overflow. The `-` operator saturates
    /// there instead.
    pub const fn checked_sub(&self, other: Self) -> Option<Self> {
        match self.usec.checked_sub(other.usec) {
            Some(usec) => Some(Self { usec }),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The operators saturate and the checked forms report overflow; neither
    /// wraps. (`+` used to wrap in release builds: adding 1e8 days twice to
    /// 2024 gave year -34949.)
    #[test]
    fn overflow_saturates_or_is_reported() {
        let t = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
        let big = Duration::from_days(1.0e8);
        let far = t + big;
        assert!(far > t);
        assert_eq!((far + big).raw, i64::MAX);
        assert_eq!(far.checked_add(big), None);
        assert_eq!(t.checked_add(big), Some(far));
        assert_eq!(((t - big) - big).raw, i64::MIN);
        assert_eq!((t - big).checked_sub(big), None);
        let mut m = far;
        m += big;
        assert_eq!(m.raw, i64::MAX);
        m -= big;
        assert!(m > t);

        let max = Duration::from_microseconds(i64::MAX);
        let one = Duration::from_microseconds(1);
        assert_eq!((max + one).usec, i64::MAX);
        assert_eq!(max.checked_add(one), None);
        assert_eq!(big.checked_add(big), None);
        assert_eq!(big.checked_sub(big), Some(Duration::zero()));
        let min = Duration::from_microseconds(i64::MIN);
        assert_eq!((min - one).usec, i64::MIN);
        assert_eq!(min.checked_sub(one), None);
        let mut d = max;
        d += one;
        assert_eq!(d, max);
        d = min;
        d -= one;
        assert_eq!(d, min);

        assert_eq!((t - Instant::INVALID).usec, i64::MAX);
        assert_eq!(t.checked_duration_since(Instant::INVALID), None);
        assert_eq!(far.checked_duration_since(t), Some(big));
        assert_eq!(far - t, big);
    }
}
