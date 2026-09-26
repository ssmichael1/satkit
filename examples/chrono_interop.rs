//! Converting between `satkit::Instant` and `chrono::DateTime`.
//!
//! Needs the `chrono` feature (off by default):
//!
//! ```text
//! cargo run --example chrono_interop --features chrono
//! ```

use chrono::{DateTime, TimeZone, Utc};
use satkit::{Duration, Instant, TimeScale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // chrono -> satkit, exact to the microsecond (any time zone)
    let dt = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let t = Instant::from(dt);
    println!("from chrono: {t}");

    // satkit -> chrono
    let back: DateTime<Utc> = (t + Duration::from_hours(1.5)).into();
    println!("to chrono:   {back}");

    // Every satkit function that takes a time is generic over `TimeLike`,
    // which `chrono::DateTime<Utc>` implements, so it can be passed as is
    let tt = satkit::TimeLike::as_mjd_with_scale(&dt, TimeScale::TT);
    let pos = satkit::lpephem::sun::pos_gcrf(&dt);
    println!("MJD (TT) = {tt:.6}, Sun distance = {:.4e} m", pos.norm());

    // chrono has no leap seconds: 23:59:60 maps to 23:59:59, as Unix time does
    let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5)?;
    let leap_chrono: DateTime<Utc> = leap.into();
    println!("{leap} -> {leap_chrono}");

    Ok(())
}
