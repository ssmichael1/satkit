/// Day of week
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weekday {
    Sunday = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6,
    Invalid = 7,
}

/// Error returned when converting an out-of-range integer to a [`Weekday`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid Weekday value: {0} (expected 0..=6, Sunday..=Saturday)")]
pub struct InvalidWeekday(pub i32);

impl TryFrom<i32> for Weekday {
    type Error = InvalidWeekday;

    /// Convert an integer to a [`Weekday`], validating the value. Unlike the
    /// previous infallible `From<i32>`, an out-of-range integer is a hard error
    /// rather than being silently mapped to `Invalid`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Sunday),
            1 => Ok(Self::Monday),
            2 => Ok(Self::Tuesday),
            3 => Ok(Self::Wednesday),
            4 => Ok(Self::Thursday),
            5 => Ok(Self::Friday),
            6 => Ok(Self::Saturday),
            _ => Err(InvalidWeekday(value)),
        }
    }
}

impl From<Weekday> for i32 {
    fn from(value: Weekday) -> Self {
        match value {
            Weekday::Sunday => 0,
            Weekday::Monday => 1,
            Weekday::Tuesday => 2,
            Weekday::Wednesday => 3,
            Weekday::Thursday => 4,
            Weekday::Friday => 5,
            Weekday::Saturday => 6,
            Weekday::Invalid => -1,
        }
    }
}

impl std::fmt::Display for Weekday {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Sunday => "Sunday",
                Self::Monday => "Monday",
                Self::Tuesday => "Tuesday",
                Self::Wednesday => "Wednesday",
                Self::Thursday => "Thursday",
                Self::Friday => "Friday",
                Self::Saturday => "Saturday",
                Self::Invalid => "Invalid",
            }
        )
    }
}
