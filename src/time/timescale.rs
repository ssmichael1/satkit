/// Time Scales
///
/// # Enum Values:
///
/// * `UTC` - Univeral Time Coordiante
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

impl From<i32> for TimeScale {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::UTC,
            2 => Self::TT,
            3 => Self::UT1,
            4 => Self::TAI,
            5 => Self::GPS,
            6 => Self::TDB,
            _ => Self::Invalid,
        }
    }
}

impl std::fmt::Display for TimeScale {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::UTC => "Coordinate Univeral Time",
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
