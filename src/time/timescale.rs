/// Time Scales
///
/// # Enum Values:
///
/// * `UTC` - Universal Time Coordinated
/// * `TT` - Terrestrial Time
/// * `UT1` - Universal Time 1
/// * `TAI` - International Atomic Time
/// * `GPS` - Global Positioning System
/// * `TDB` - Barycentric Dynamical Time
/// * `Invalid` - Invalid
///    
#[derive(PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum TimeScale {
    /// Invalid
    Invalid = -1,
    /// Universal Time Coordinate
    UTC = 1,
    /// Terrestrial Time
    TT = 2,
    /// Universal Time 1
    UT1 = 3,
    /// International Atomic Time
    TAI = 4,
    /// Global Positioning System
    GPS = 5,
    /// Barycentric Dynamical Time
    TDB = 6,
}

/// Error returned when converting an out-of-range integer to a [`TimeScale`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid TimeScale value: {0} (expected 1..=6 for UTC, TT, UT1, TAI, GPS, TDB)")]
pub struct InvalidTimeScale(pub i32);

impl TryFrom<i32> for TimeScale {
    type Error = InvalidTimeScale;

    /// Convert an integer to a [`TimeScale`], validating the value. Unlike the
    /// previous infallible `From<i32>`, an out-of-range integer is a hard error
    /// rather than being silently mapped to `Invalid`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::UTC),
            2 => Ok(Self::TT),
            3 => Ok(Self::UT1),
            4 => Ok(Self::TAI),
            5 => Ok(Self::GPS),
            6 => Ok(Self::TDB),
            _ => Err(InvalidTimeScale(value)),
        }
    }
}

impl std::fmt::Display for TimeScale {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::UTC => "Coordinated Universal Time",
                Self::TT => "Terrestrial Time",
                Self::UT1 => "Universal Time 1",
                Self::TAI => "International Atomic Time",
                Self::GPS => "Global Positioning System",
                Self::TDB => "Barycentric Dynamical Time",
                Self::Invalid => "Invalid",
            }
        )
    }
}
