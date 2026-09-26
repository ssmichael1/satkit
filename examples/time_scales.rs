//! Time scales, arithmetic, parsing and formatting with `satkit::Instant`.
//!
//! Runs offline: the leap-second table is compiled in. UT1 needs the Earth
//! orientation file `finals2000A.all` (downloaded on first use); without it
//! satkit warns once and uses UT1 = UTC.
//!
//! ```text
//! cargo run --example time_scales
//! ```

use satkit::{Duration, Instant, TimeScale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Calendar input is UTC; the seconds are rounded to the microsecond
    let t = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0)?;
    println!("t = {t}");

    // The same instant expressed in each scale, as seconds ahead of UTC
    let mjd_utc = t.as_mjd_with_scale(TimeScale::UTC);
    for scale in [
        TimeScale::TAI,
        TimeScale::TT,
        TimeScale::GPS,
        TimeScale::UT1,
        TimeScale::TDB,
    ] {
        let ahead = (t.as_mjd_with_scale(scale) - mjd_utc) * 86_400.0;
        println!("  {scale:?} - UTC = {ahead:+.4} s");
    }

    // Calendar components interpreted in another scale
    let t_tt = Instant::from_datetime_with_scale(2024, 6, 15, 12, 0, 0.0, TimeScale::TT)?;
    let dt: Duration = t - t_tt; // exact, in integer microseconds
    println!("12:00 UTC - 12:00 TT = {} us", dt.as_microseconds());

    // Arithmetic is exact and counts leap seconds
    let t0 = Instant::from_datetime(2016, 12, 31, 23, 59, 59.0)?;
    let t1 = t0 + Duration::from_seconds(1.0);
    let t2 = t0 + Duration::from_seconds(2.0);
    println!("leap second: {t0} -> {t1} -> {t2}");
    let jan1 = Instant::from_date(2017, 1, 1)?;
    let dec31 = Instant::from_date(2016, 12, 31)?;
    println!("2016-12-31 lasted {} s", (jan1 - dec31).as_seconds());

    // Parsing: RFC 3339, free-form strings, and explicit formats
    let a = Instant::from_rfc3339("2024-06-15T12:00:00.25Z")?;
    let b = Instant::from_string("June 15 2024 13:30")?;
    let c = Instant::strptime("15/06/2024 14:45:10", "%d/%m/%Y %H:%M:%S")?;
    println!("parsed: {a}, {b}, {c}");

    // Formatting
    println!("strftime: {}", c.strftime("%A %d %B %Y, %H:%M")?);
    println!("day of year {}, {}", c.day_of_year(), c.day_of_week());

    // Other representations
    println!("JD (UTC)  = {:.6}", t.as_jd_with_scale(TimeScale::UTC));
    println!("MJD (TT)  = {:.6}", t.as_mjd_with_scale(TimeScale::TT));
    println!("Unix time = {}", t.as_unixtime());
    let g = Instant::from_gps_week_and_second(2318, 561_618.0);
    println!("GPS week 2318, 561618 s = {g}");

    Ok(())
}
