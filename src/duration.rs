use crate::AstroTime;

/// Enum representing durations of times, allowing for representation
/// via common measures of duration (years, days, hours, minutes, seconds)
///
/// This enum can be added to and subtracted from "astrotime" objects to
/// represent new "astrotime" objects, and is also returned when
/// two "astrotime" objects are subtracted from one anothre
#[derive(Clone, Debug)]
pub enum Duration {
    Days(f64),
    Seconds(f64),
    Years(f64),
    Minutes(f64),
    Hours(f64),
}

impl Duration {
    #[inline]
    /// Return duration represented as days
    /// Note: a day is defined as being exactly equal to 86400 seconds
    /// (leap-seconds are neglected)
    pub fn days(&self) -> f64 {
        match self {
            Duration::Days(v) => *v,
            Duration::Seconds(v) => *v / 86400.0,
            Duration::Years(v) => *v * 365.25,
            Duration::Minutes(v) => *v / 1440.0,
            Duration::Hours(v) => *v / 24.0,
        }
    }

    /// Return duration represented as seconds
    #[inline]
    pub fn seconds(&self) -> f64 {
        match self {
            Duration::Days(v) => *v * 86400.0,
            Duration::Seconds(v) => *v,
            Duration::Years(v) => *v * 86400.0 * 365.25,
            Duration::Minutes(v) => *v * 60.0,
            Duration::Hours(v) => *v * 3600.0,
        }
    }

    /// Return duration represented as hours (3600 seconds)
    #[inline]
    pub fn hours(&self) -> f64 {
        match self {
            Duration::Days(v) => *v * 24.0,
            Duration::Seconds(v) => *v / 3600.0,
            Duration::Minutes(v) => *v / 60.0,
            Duration::Hours(v) => *v,
            Duration::Years(v) => *v * 24.0 * 365.25,
        }
    }

    // Return duration represented as minutes (60 seconds)
    #[inline]
    pub fn minutes(&self) -> f64 {
        match self {
            Duration::Days(v) => *v * 1440.0,
            Duration::Seconds(v) => *v / 60.0,
            Duration::Minutes(v) => *v,
            Duration::Hours(v) => *v * 60.0,
            Duration::Years(v) => *v * 1440.0 * 365.25,
        }
    }

    /// String representation of duration
    pub fn to_string(&self) -> String {
        let mut secs = self.seconds();
        let mut sign = String::from("");
        if secs < 0.0 {
            sign = String::from("-");
            secs = -1.0 * secs;
        }
        // add 5e-4 seconds so that display will round correctly,
        // e.g. if seconds = 59.999..., rather than round up and
        // display "seconds" in second field, show 0 and increment
        // minutes...
        if secs % 60.0 > 59.9995 {
            secs = secs + 5.0e-4;
        }

        if secs < 1.0 {
            format!("Duration: {}{:.3} microseconds", sign, (secs % 1.0) * 1.0e6)
        } else {
            let days = (secs / 86400.0) as usize;
            let hours = ((secs % 86400.0) / 3600.0) as usize;
            let minutes = ((secs % 3600.0) / 60.0) as usize;
            secs = secs % 60.0;
            let mut s = String::from("Duration: ");
            s.push_str(&sign);
            if days > 0 {
                s.push_str(format!("{} days, ", days).as_str());
            }
            if hours > 0 || days > 0 {
                s.push_str(format!("{} hours, ", hours).as_str());
            }
            if minutes > 0 || hours > 0 || days > 0 {
                s.push_str(format!("{} minutes, ", minutes).as_str());
            }
            s.push_str(format!("{:.3} seconds", secs).as_str());
            s
        }
    }
}

impl std::ops::Add<AstroTime> for Duration {
    type Output = AstroTime;
    #[inline]
    fn add(self, other: AstroTime) -> AstroTime {
        other + self.days()
    }
}

impl std::ops::Add<&AstroTime> for Duration {
    type Output = AstroTime;
    #[inline]
    fn add(self, other: &AstroTime) -> AstroTime {
        *other + self.days()
    }
}

impl std::ops::Add<Duration> for Duration {
    type Output = Duration;
    #[inline]
    fn add(self, other: Duration) -> Self::Output {
        Duration::Seconds(self.seconds() + other.seconds())
    }
}

impl std::ops::Add<&Duration> for Duration {
    type Output = Duration;
    #[inline]
    fn add(self, other: &Duration) -> Self::Output {
        Duration::Seconds(self.seconds() + other.seconds())
    }
}

impl std::ops::Add<Duration> for &Duration {
    type Output = Duration;
    #[inline]
    fn add(self, other: Duration) -> Self::Output {
        Duration::Seconds(self.seconds() + other.seconds())
    }
}

impl std::ops::Sub<Duration> for Duration {
    type Output = Duration;
    #[inline]
    fn sub(self, other: Duration) -> Self::Output {
        Duration::Seconds(self.seconds() - other.seconds())
    }
}

impl std::ops::Sub<&Duration> for Duration {
    type Output = Duration;
    #[inline]
    fn sub(self, other: &Duration) -> Self::Output {
        Duration::Seconds(self.seconds() - other.seconds())
    }
}

impl std::ops::Sub<Duration> for &Duration {
    type Output = Duration;
    #[inline]
    fn sub(self, other: Duration) -> Self::Output {
        Duration::Seconds(self.seconds() - other.seconds())
    }
}

impl std::ops::Sub<&Duration> for &Duration {
    type Output = Duration;
    #[inline]
    fn sub(self, other: &Duration) -> Self::Output {
        Duration::Seconds(self.seconds() - other.seconds())
    }
}

impl std::fmt::Display for Duration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn testdur() {
        // A random assortment of duration tests...
        assert!(Duration::Seconds(1.0).seconds() == 1.0);
        assert!(Duration::Minutes(1.0).seconds() == 60.0);
        assert!(Duration::Hours(1.0).minutes() == 60.0);
        assert!(Duration::Hours(1.0).seconds() == 3600.0);
        assert!(Duration::Days(1.0).hours() == 24.0);
        assert!(Duration::Days(1.0).seconds() == 86400.0);
        assert!((Duration::Days(1.0) + Duration::Days(1.0)).seconds() == 2.0 * 86400.0);
    }
}
