use super::Duration;
use super::Instant;

impl std::ops::Add<Duration> for Instant {
    type Output = Self;

    fn add(self, other: Duration) -> Self {
        Self {
            raw: self.raw + other.usec,
        }
    }
}

impl std::ops::Add<&Duration> for Instant {
    type Output = Self;

    fn add(self, other: &Duration) -> Self {
        Self {
            raw: self.raw + other.usec,
        }
    }
}

impl std::ops::Add<Duration> for &Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        Instant {
            raw: self.raw + other.usec,
        }
    }
}

impl std::ops::Add<&Duration> for &Instant {
    type Output = Instant;

    fn add(self, other: &Duration) -> Instant {
        Instant {
            raw: self.raw + other.usec,
        }
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Self;

    fn sub(self, other: Duration) -> Self {
        Self {
            raw: self.raw - other.usec,
        }
    }
}

impl std::ops::Sub<&Duration> for Instant {
    type Output = Self;

    fn sub(self, other: &Duration) -> Self {
        Self {
            raw: self.raw - other.usec,
        }
    }
}

impl std::ops::Sub<Duration> for &Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        Instant {
            raw: self.raw - other.usec,
        }
    }
}

impl std::ops::Sub<&Duration> for &Instant {
    type Output = Instant;

    fn sub(self, other: &Duration) -> Instant {
        Instant {
            raw: self.raw - other.usec,
        }
    }
}

impl std::ops::Sub<Instant> for &Instant {
    type Output = Duration;

    fn sub(self, other: Instant) -> Duration {
        Duration {
            usec: self.raw - other.raw,
        }
    }
}

impl std::ops::Sub<Self> for Instant {
    type Output = Duration;

    fn sub(self, other: Self) -> Duration {
        Duration {
            usec: self.raw - other.raw,
        }
    }
}

impl std::ops::Sub<&Self> for Instant {
    type Output = Duration;

    fn sub(self, other: &Self) -> Duration {
        Duration {
            usec: self.raw - other.raw,
        }
    }
}

impl std::ops::Sub<&Instant> for &Instant {
    type Output = Duration;

    fn sub(self, other: &Instant) -> Duration {
        Duration {
            usec: self.raw - other.raw,
        }
    }
}

impl std::ops::AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        self.raw += other.usec;
    }
}

impl std::ops::SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        self.raw -= other.usec;
    }
}

impl std::ops::AddAssign<&Duration> for Instant {
    fn add_assign(&mut self, other: &Duration) {
        self.raw += other.usec;
    }
}

impl std::ops::SubAssign<&Duration> for Instant {
    fn sub_assign(&mut self, other: &Duration) {
        self.raw -= other.usec;
    }
}

impl std::ops::AddAssign<Duration> for &mut Instant {
    fn add_assign(&mut self, other: Duration) {
        self.raw += other.usec;
    }
}

impl std::ops::SubAssign<Duration> for &mut Instant {
    fn sub_assign(&mut self, other: Duration) {
        self.raw -= other.usec;
    }
}

impl std::ops::AddAssign<&Duration> for &mut Instant {
    fn add_assign(&mut self, other: &Duration) {
        self.raw += other.usec;
    }
}

impl std::ops::SubAssign<&Duration> for &mut Instant {
    fn sub_assign(&mut self, other: &Duration) {
        self.raw -= other.usec;
    }
}

impl std::cmp::PartialEq for Instant {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl std::cmp::PartialOrd for Instant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Eq for Instant {}

impl std::cmp::Ord for Instant {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

/// Add two durations together
impl std::ops::Add<Self> for Duration {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            usec: self.usec + other.usec,
        }
    }
}

impl std::ops::AddAssign<Self> for Duration {
    fn add_assign(&mut self, other: Self) {
        self.usec += other.usec;
    }
}

impl std::ops::AddAssign<&Self> for Duration {
    fn add_assign(&mut self, other: &Self) {
        self.usec += other.usec;
    }
}

impl std::ops::SubAssign<Self> for Duration {
    fn sub_assign(&mut self, other: Self) {
        self.usec -= other.usec;
    }
}

impl std::ops::SubAssign<&Self> for Duration {
    fn sub_assign(&mut self, other: &Self) {
        self.usec -= other.usec;
    }
}

impl std::cmp::PartialEq for Duration {
    fn eq(&self, other: &Self) -> bool {
        self.usec == other.usec
    }
}

impl std::cmp::PartialOrd for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Eq for Duration {}

impl std::cmp::Ord for Duration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.usec.cmp(&other.usec)
    }
}

/// Subtract two durations
impl std::ops::Sub<Self> for Duration {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            usec: self.usec - other.usec,
        }
    }
}
