//! Converting between `satkit::Instant` and `hifitime::Epoch`.
//!
//! Needs the `hifitime` feature (off by default):
//!
//! ```text
//! cargo run --example hifitime_interop --features hifitime
//! ```

use hifitime::{Epoch, TimeScale as HifiScale};
use satkit::{Duration, Instant, TimeScale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // hifitime -> satkit, to the nearest microsecond, in any hifitime scale
    let e = Epoch::from_gregorian_utc_hms(2024, 6, 15, 12, 0, 0);
    let t = Instant::from(e);
    println!("from hifitime: {t}");

    // satkit -> hifitime, exact; the Epoch is in TAI, and displays in
    // whatever scale it is moved to
    let back: Epoch = (t + Duration::from_hours(1.5)).into();
    println!(
        "to hifitime:   {back} = {}",
        back.to_time_scale(HifiScale::UTC)
    );

    // `hifitime::Epoch` implements `TimeLike`, so it can be passed to any
    // satkit function that takes a time; UT1 and TDB use satkit's models
    let tt = satkit::TimeLike::as_mjd_with_scale(&e, TimeScale::TT);
    let pos = satkit::lpephem::sun::pos_gcrf(&e);
    println!("MJD (TT) = {tt:.6}, Sun distance = {:.4e} m", pos.norm());

    // Both crates count leap seconds, so an instant inside one survives
    // the round trip. (Moving that Epoch to hifitime's UTC scale does not:
    // it reads 23:59:59.5 there and converts back a second early.)
    let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5)?;
    let leap_e = Epoch::from(leap);
    println!("{leap} -> {leap_e} -> {}", Instant::from(leap_e));

    Ok(())
}
