/// Time Scales
///
/// # Enum Values:
///
/// * `UTC` - Univeral Time Coordiante
/// * `TT` - Terrestrial Time
/// * `UT1` - UT1
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
    /// UT1
    UT1 = 3,
    /// International Atomic Time
    TAI = 4,
    /// Global Positioning System
    GPS = 5,
    /// Barycentric Dynamical Time
    TDB = 6,
}
