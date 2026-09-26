//! JPL planetary ephemerides, compared with the low-precision models.
//!
//! Needs the JPL DE440 ephemeris `linux_p1550p2650.440` (102 MB), which is
//! downloaded and SHA-256 verified on first use, then cached in the data
//! directory. With `SATKIT_OFFLINE=1` and no copy on disk the queries fail
//! with `jplephem::Error`.
//!
//! ```text
//! cargo run --example ephemeris
//! ```

use satkit::{jplephem, lpephem, Instant, SolarSystem};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = Instant::from_datetime(2024, 6, 20, 12, 0, 0.0)?;

    // Geocentric position (m) and velocity (m/s) of the Moon, in the GCRF
    let (p, v) = jplephem::geocentric_state(SolarSystem::Moon, &t)?;
    println!(
        "Moon: {:.0} km away, moving at {:.3} km/s",
        p.norm() / 1e3,
        v.norm() / 1e3
    );

    // Positions relative to the solar-system barycenter
    for body in [SolarSystem::Venus, SolarSystem::Mars, SolarSystem::Jupiter] {
        let p = jplephem::barycentric_pos(body, &t)?;
        println!(
            "{body:?}: {:.4} AU from the barycenter",
            p.norm() / satkit::consts::AU
        );
    }

    // The low-precision Sun and Moon are much faster; how far off are they?
    let angle = |a: satkit::Vector3, b: satkit::Vector3| {
        (a.dot(&b) / (a.norm() * b.norm()))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
            * 3600.0
    };
    let sun_jpl = jplephem::geocentric_pos(SolarSystem::Sun, &t)?;
    let sun_lp = lpephem::sun::pos_gcrf(&t);
    println!(
        "low-precision Sun:  {:.0} arcsec from DE440",
        angle(sun_jpl, sun_lp)
    );
    let moon_lp = lpephem::moon::pos_gcrf(&t);
    println!(
        "low-precision Moon: {:.0} arcsec from DE440",
        angle(p, moon_lp)
    );

    // Constants from the ephemeris file header
    if let Some(au_km) = jplephem::consts("AU") {
        println!("AU in the DE440 header: {au_km} km");
    }

    Ok(())
}
