//! High-precision numerical propagation of a low-Earth orbit: force-model
//! settings, drag, dense-output interpolation, impulsive maneuvers, and
//! covariance propagation through the state transition matrix.
//!
//! Needs the JPL ephemeris for the Sun and Moon (`linux_p1550p2650.440`,
//! 102 MB, downloaded on first use), Earth orientation parameters, and the
//! space-weather tables for drag (both fetched on first use and refreshed by
//! `satkit::utils::update_datafiles`).
//!
//! ```text
//! cargo run --release --example propagate_leo
//! ```

use satkit::frametransform::gcrf_to_rtn;
use satkit::kepler::{Anomaly, Kepler};
use satkit::orbitprop::{
    propagate, ImpulsiveManeuver, PropSettings, SatPropertiesSimple, SatState, StateCov,
};
use satkit::{Duration, Frame, Instant, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::from_datetime(2024, 6, 15, 0, 0, 0.0)?;
    let t1 = t0 + Duration::from_days(1.0);

    // A 500 km, 97.4 deg orbit from Keplerian elements
    let kep = Kepler::new(
        satkit::consts::EARTH_RADIUS + 500.0e3,
        0.001,
        97.4_f64.to_radians(),
        30.0_f64.to_radians(),
        0.0,
        Anomaly::Mean(0.0),
    );
    let (r0, v0) = kep.to_pv();
    // The propagator state is a 6-vector: GCRF position (m), velocity (m/s)
    let mut pv0 = satkit::mathtypes::Vector6::zeros();
    pv0.set_block(0, 0, &r0);
    pv0.set_block(3, 0, &v0);

    // Force-model settings: the defaults include Sun and Moon gravity,
    // solid tides, relativity, drag with space weather, and SRP
    let mut settings = PropSettings::default();
    settings.set_gravity(16, 16)?;

    // Cd * A / m and Cr * A / m, in m^2/kg
    let props = SatPropertiesSimple::new(2.2 * 1.0 / 100.0, 1.3 * 1.0 / 100.0);

    let res = propagate(&pv0, &t0, &t1, &settings, Some(&props))?;
    println!(
        "{} accepted steps, {} function evaluations",
        res.accepted_steps, res.num_eval
    );
    let r1 = res.state_end.block::<3, 1>(0, 0);
    println!(
        "after 1 day: |r| = {:.3} km, |v| = {:.4} km/s",
        r1.norm() / 1e3,
        res.state_end.block::<3, 1>(3, 0).norm() / 1e3
    );

    // Dense output: the state at any time within the propagation
    let mid = res.interp(&(t0 + Duration::from_hours(12.0)))?;
    println!(
        "interpolated at 12 h: |r| = {:.3} km",
        mid.block::<3, 1>(0, 0).norm() / 1e3
    );

    // Drag lowers the orbit: compare with a drag-free run
    let no_drag = propagate(&pv0, &t0, &t1, &settings, None)?;
    let sep = (res.state_end - no_drag.state_end).block::<3, 1>(0, 0);
    println!("drag vs no drag after 1 day: {:.1} m apart", sep.norm());

    // `SatState` wraps the same propagation with maneuvers and covariance
    let mut sat = SatState::from_pv(&t0, &r0, &v0);
    sat.add_maneuver(ImpulsiveManeuver::prograde(
        t0 + Duration::from_hours(6.0),
        1.0,
    ));
    // 1-sigma position uncertainty: 10 m radial, 50 m in-track, 5 m cross-track
    sat.set_pos_uncertainty(&Vector3::from_array([10.0, 50.0, 5.0]), Frame::RTN)?;
    sat.set_vel_uncertainty(&Vector3::from_array([0.01, 0.01, 0.01]), Frame::RTN)?;
    let sat1 = sat.propagate(&t1, Some(&settings), Some(&props))?;
    // Offset from the unmaneuvered orbit, in its radial/in-track/cross-track
    // frame: the higher orbit is slower, so the satellite falls behind
    let v1 = res.state_end.block::<3, 1>(3, 0);
    let dr_rtn = gcrf_to_rtn(&r1, &v1) * (sat1.pos_gcrf() - r1);
    println!(
        "1 m/s prograde burn at 6 h: RTN offset after 1 day = {:.1} km",
        dr_rtn / 1e3
    );

    // Rotate the propagated position covariance into RTN
    if let StateCov::PVCov(cov) = sat1.cov() {
        let d = gcrf_to_rtn(&sat1.pos_gcrf(), &sat1.vel_gcrf());
        let c_rtn = d * cov.block::<3, 3>(0, 0) * d.transpose();
        println!(
            "1-sigma after 1 day (RTN): {:.0} m, {:.0} m, {:.0} m",
            c_rtn[(0, 0)].sqrt(),
            c_rtn[(1, 1)].sqrt(),
            c_rtn[(2, 2)].sqrt()
        );
    }

    Ok(())
}
