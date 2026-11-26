//! Converting between `satkit::Instant` and `jiff::Timestamp`.
//!
//! Needs the `jiff` feature (off by default):
//!
//! ```text
//! cargo run --example jiff_interop --features jiff
//! ```

use jiff::{tz, Timestamp};
use satkit::{Duration, Instant, TimeScale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // jiff -> satkit, to the nearest microsecond
    let ts: Timestamp = "2024-06-15T12:00:00Z".parse()?;
    let t = Instant::from(ts);
    println!("from jiff: {t}");

    // satkit -> jiff
    let back: Timestamp = (t + Duration::from_hours(1.5)).into();
    println!("to jiff:   {back}");

    // A zoned datetime converts by its instant, whatever its time zone
    let zoned = ts.to_zoned_datetime(tz::TimeZone::fixed(tz::Offset::constant(-4)));
    println!("zoned:     {zoned} -> {}", Instant::from(&zoned));

    // Every satkit function that takes a time is generic over `TimeLike`,
    // which `jiff::Timestamp` and `jiff::ZonedDateTime` implement, so
    // either can be passed as is
    let tt = satkit::TimeLike::as_mjd_with_scale(&ts, TimeScale::TT);
    let pos = satkit::lpephem::sun::pos_gcrf(&zoned);
    println!("MJD (TT) = {tt:.6}, Sun distance = {:.4e} m", pos.norm());

    // jiff has no leap seconds: 23:59:60 maps to 23:59:59, as Unix time does
    let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5)?;
    let leap_jiff: Timestamp = leap.into();
    println!("{leap} -> {leap_jiff}");

    Ok(())
}
