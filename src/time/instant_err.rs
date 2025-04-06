use thiserror::Error;

#[derive(Error, Debug)]
pub enum InstantError {
    #[error("Invalid Month String: {0}")]
    InvalidMonthString(String),
    #[error("Invalid Month: {0}")]
    InvalidMonth(i32),
    #[error("Invalid Day: {0}")]
    InvalidDay(i32),
    #[error("Invalid Hour: {0}")]
    InvalidHour(i32),
    #[error("Invalid Minute: {0}")]
    InvalidMinute(i32),
    #[error("Invalid Second: {0}")]
    InvalidSecond(i32),
    #[error("Invalid Microsecond: {0}")]
    InvalidMicrosecond(i32),
    #[error("Invalid String: {0}")]
    InvalidString(String),
    #[error("Invalid Format Character: {0}")]
    InvalidFormat(char),
    #[error("Missing Format Character")]
    MissingFormat,
}
