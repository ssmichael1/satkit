#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weekday {
    Sunday = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6,
}

impl From<i32> for Weekday {
    fn from(value: i32) -> Self {
        match value {
            0 => Weekday::Sunday,
            1 => Weekday::Monday,
            2 => Weekday::Tuesday,
            3 => Weekday::Wednesday,
            4 => Weekday::Thursday,
            5 => Weekday::Friday,
            6 => Weekday::Saturday,
            _ => panic!("Invalid weekday value: {}", value),
        }
    }
}

impl std::fmt::Display for Weekday {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Weekday::Sunday => "Sunday",
                Weekday::Monday => "Monday",
                Weekday::Tuesday => "Tuesday",
                Weekday::Wednesday => "Wednesday",
                Weekday::Thursday => "Thursday",
                Weekday::Friday => "Friday",
                Weekday::Saturday => "Saturday",
            }
        )
    }
}
