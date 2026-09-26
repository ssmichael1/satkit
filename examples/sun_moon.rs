//! Low-precision (analytic) Sun and Moon: positions, sunrise and sunset,
//! Moon phase, and the Earth-shadow function.
//!
//! Runs offline with no data files.
//!
//! ```text
//! cargo run --example sun_moon
//! ```

use satkit::lpephem::{self, moon, sun};
use satkit::{ITRFCoord, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = Instant::from_datetime(2024, 6, 20, 12, 0, 0.0)?;

    // Geocentric positions in GCRF, meters
    let psun = sun::pos_gcrf(&t);
    let pmoon = moon::pos_gcrf(&t);
    println!("Sun  distance = {:.6} AU", psun.norm() / satkit::consts::AU);
    println!("Moon distance = {:.0} km", pmoon.norm() / 1e3);

    // Moon phase: Sun-Moon elongation in ecliptic longitude
    println!(
        "Moon phase {:.1} deg ({}), {:.1}% illuminated",
        moon::phase(&t).to_degrees(),
        moon::phase_name(&t).name(),
        100.0 * moon::illumination(&t)
    );

    // Sunrise and sunset (UTC) on the input's UTC calendar date. `None`
    // is the standard 90 deg 50' zenith angle; 96, 102 and 108 deg give
    // civil, nautical and astronomical twilight.
    let boston = ITRFCoord::from_geodetic_deg(42.3601, -71.0589, 0.0);
    let (rise, set) = sun::riseset(&t, &boston, None)?;
    println!(
        "Boston sunrise {} UTC, sunset {} UTC",
        rise.strftime("%H:%M")?,
        set.strftime("%H:%M")?
    );
    let (dawn, dusk) = sun::riseset(&t, &boston, Some(96.0))?;
    println!(
        "civil twilight {} to {} UTC",
        dawn.strftime("%H:%M")?,
        dusk.strftime("%H:%M")?
    );

    // Midsummer above the Arctic Circle: no sunset, reported as an error
    let tromso = ITRFCoord::from_geodetic_deg(69.65, 18.96, 0.0);
    match sun::riseset(&t, &tromso, None) {
        Ok((rise, set)) => println!("Tromso: sunrise {rise}, sunset {set}"),
        Err(lpephem::Error::NoSunriseOrSunset) => println!("Tromso: midnight Sun, no sunset"),
        Err(e) => return Err(e.into()),
    }

    // Fraction of the Sun's disc visible from a satellite (0 = umbra)
    let r = 6_878_137.0;
    let sunward = psun * (r / psun.norm());
    let behind = -sunward;
    println!(
        "shadow function: sunward {:.1}, behind the Earth {:.1}",
        sun::shadowfunc(&psun, &sunward),
        sun::shadowfunc(&psun, &behind)
    );

    Ok(())
}
