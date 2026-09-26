//! Keplerian elements, two-body propagation, and Lambert targeting.
//!
//! Runs offline with no data files.
//!
//! ```text
//! cargo run --example kepler_lambert
//! ```

use satkit::consts::{EARTH_RADIUS, MU_EARTH};
use satkit::kepler::{Anomaly, Kepler};
use satkit::lambert::lambert;
use satkit::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Elements: a (m), e, i, RAAN, argument of perigee (radians), anomaly
    let k = Kepler::try_new(
        7000.0e3,
        0.001,
        98.0_f64.to_radians(),
        45.0_f64.to_radians(),
        0.0,
        Anomaly::Mean(30.0_f64.to_radians()),
    )?;
    println!(
        "a = {:.1} km, period = {:.2} min, perigee alt = {:.1} km",
        k.a / 1e3,
        k.period() / 60.0,
        (k.periapsis() - EARTH_RADIUS) / 1e3
    );

    // Elements -> Cartesian state (inertial, m and m/s) -> elements
    let (r, v) = k.to_pv();
    let k2 = Kepler::from_pv(r, v)?;
    println!(
        "|r| = {:.3} km, |v| = {:.4} km/s, round trip da = {:.1e} m",
        r.norm() / 1e3,
        v.norm() / 1e3,
        (k2.a - k.a).abs()
    );

    // Two-body propagation by a quarter period
    let k3 = k.propagate(&Duration::from_seconds(k.period() / 4.0));
    println!(
        "mean anomaly after T/4: {:.3} deg (true anomaly {:.3} deg)",
        k3.mean_anomaly().to_degrees(),
        k3.nu.to_degrees()
    );

    // Lambert: the transfer from r1 to r2 in a given time of flight, here
    // from 7000 km to 12000 km radius, 150 deg further along
    let r1 = satkit::Vector3::from_array([7000.0e3, 0.0, 0.0]);
    let theta = 150.0_f64.to_radians();
    let r2 = satkit::Vector3::from_array([12000.0e3 * theta.cos(), 12000.0e3 * theta.sin(), 0.0]);
    let tof = 3900.0;
    let solutions = lambert(&r1, &r2, tof, MU_EARTH, true)?;
    let (v1, v2) = solutions[0]; // the zero-revolution solution comes first
    println!(
        "Lambert ({} solution(s)): v1 = {:.1} m/s, v2 = {:.1} m/s",
        solutions.len(),
        v1,
        v2
    );

    // Check: propagate the departure state and compare with r2
    let transfer = Kepler::from_pv(r1, v1)?;
    let (r_arrive, _) = transfer.propagate(&Duration::from_seconds(tof)).to_pv();
    println!(
        "transfer: a = {:.1} km, e = {:.4}, perigee alt = {:.1} km, arrival miss = {:.2e} m",
        transfer.a / 1e3,
        transfer.eccen,
        (transfer.periapsis() - EARTH_RADIUS) / 1e3,
        (r_arrive - r2).norm()
    );

    Ok(())
}
