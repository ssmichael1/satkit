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

impl From<i32> for Weekday {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Sunday,
            1 => Self::Monday,
            2 => Self::Tuesday,
            3 => Self::Wednesday,
            4 => Self::Thursday,
            5 => Self::Friday,
            6 => Self::Saturday,
            _ => Self::Invalid,
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
