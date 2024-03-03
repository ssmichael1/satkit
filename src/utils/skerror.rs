use std::error::Error;
use std::fmt;

pub type SKResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
pub struct SKErr {
    details: String,
}

impl SKErr {
    pub fn new(msg: &str) -> SKErr {
        SKErr {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for SKErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SKErr: {}", self.details)
    }
}

impl std::error::Error for SKErr {
    fn description(&self) -> &str {
        &self.details
    }
}

#[macro_export]
macro_rules! skerror {
    ($($args:tt),*) => {{
        Err(crate::utils::SKErr::new(format!($($args),*).as_str()).into())
    }};
}

pub(crate) use skerror;
