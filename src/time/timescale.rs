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
#[derive(PartialEq, Debug)]
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
            1 => TimeScale::UTC,
            2 => TimeScale::TT,
            3 => TimeScale::UT1,
            4 => TimeScale::TAI,
            5 => TimeScale::GPS,
            6 => TimeScale::TDB,
            _ => TimeScale::Invalid,
        }
    }
}

impl std::fmt::Display for TimeScale {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TimeScale::UTC => "Coordinate Univeral Time",
                TimeScale::TT => "Terrestrial Time",
                TimeScale::UT1 => "Universal Time 1",
                TimeScale::TAI => "International Atomic Time",
                TimeScale::GPS => "Global Positioning System",
                TimeScale::TDB => "Barycentric Dynamical Time",
                TimeScale::Invalid => "Invalid",
            }
        )
    }
}
