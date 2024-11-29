mod duration;
mod instant;
mod instant_ops;
mod instantparse;
mod timescale;

pub use duration::Duration;
pub use instant::Instant;
pub use timescale::TimeScale;

#[cfg(test)]
mod tests;
