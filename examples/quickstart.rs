//! Where is the ISS? Parse a TLE, run SGP4, and convert to latitude,
//! longitude and altitude.
//!
//! SGP4 needs no data files; the TEME -> ITRF rotation uses Earth
//! orientation parameters, downloaded on first use.
//!
//! ```text
//! cargo run --example quickstart
//! ```

use satkit::frametransform::qteme2itrf;
use satkit::sgp4::sgp4;
use satkit::{Duration, ITRFCoord, Vector3, TLE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tle = TLE::load_3line(
        "ISS (ZARYA)",
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009",
        "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299350",
    )?;

    // Half an orbit after the element-set epoch
    let t = tle.epoch + Duration::from_minutes(46.0);

    // SGP4 gives TEME position (m) and velocity (m/s), one column per time
    let state = sgp4(&mut tle, &[t])?;
    let p_teme = Vector3::from_slice(state.pos.col_slice(0));

    // Rotate to the Earth-fixed frame and convert to geodetic
    let itrf = ITRFCoord::from(qteme2itrf(&t) * p_teme);
    println!("{} at {t}", tle.name);
    println!(
        "  latitude {:.3} deg, longitude {:.3} deg, altitude {:.1} km",
        itrf.latitude_deg(),
        itrf.longitude_deg(),
        itrf.hae() / 1e3
    );

    Ok(())
}
