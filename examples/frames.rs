//! Coordinate frames: geodetic coordinates, ITRF <-> GCRF / TEME rotations,
//! state transforms, and local (ENU) and orbit-local (RTN) frames.
//!
//! The IERS nutation tables are compiled in, so this runs offline. The
//! Earth-fixed rotations also use Earth orientation parameters from
//! `finals2000A.all` (downloaded on first use, refreshed by
//! `satkit::utils::update_datafiles`); without that file satkit warns once
//! and uses zero polar motion and UT1 = UTC, which is off by up to ~12".
//!
//! ```text
//! cargo run --example frames
//! ```

use satkit::earth_orientation_params as eop;
use satkit::frametransform::{self, rotation, rotation_approx, transform_state};
use satkit::{Frame, ITRFCoord, Instant, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0)?;

    // Earth orientation parameters in effect at `t`
    println!("EOP status: {:?}", eop::status(&t));
    if let Some([dut1, xp, yp, ..]) = eop::get(&t) {
        println!("UT1-UTC = {dut1:.4} s, polar motion = ({xp:.4}\", {yp:.4}\")");
    }

    // A ground station from geodetic coordinates (degrees, meters)
    let station = ITRFCoord::from_geodetic_deg(42.3601, -71.0589, 20.0);
    println!("station: {station}");
    println!("  ITRF = {:.1} m", station.itrf);

    // Rotation between frames, as a quaternion: full IERS 2010 reduction
    let q = rotation(Frame::ITRF, Frame::GCRF, &t)?;
    let p_gcrf = q * station.itrf;
    println!("  GCRF = {p_gcrf:.1} m");

    // The approximate reduction is cheaper and good to about an arcsecond
    let q_approx = rotation_approx(Frame::ITRF, Frame::GCRF, &t)?;
    let (_, angle) = (q_approx.conjugate() * q).to_axis_angle();
    println!(
        "full vs approximate reduction: {:.3} arcsec",
        angle.to_degrees() * 3600.0
    );

    // State transform: velocity picks up the Earth-rotation term w x r
    let (p, v) = transform_state(
        Frame::ITRF,
        Frame::GCRF,
        &t,
        &station.itrf,
        &Vector3::zeros(),
    )?;
    println!(
        "station inertial speed = {:.1} m/s at {:.1} km",
        v.norm(),
        p.norm() / 1e3
    );

    // TEME (the SGP4 output frame) to ITRF, and back to geodetic
    let p_teme = Vector3::from_array([6_778_137.0, 0.0, 0.0]);
    let sat = ITRFCoord::from(rotation(Frame::TEME, Frame::ITRF, &t)? * p_teme);
    println!("TEME x-axis point at 400 km: {sat}");

    // Local East-North-Up vector from the station to another point
    let target = ITRFCoord::from_geodetic_deg(40.7128, -74.0060, 10.0);
    let enu = target.to_enu(&station);
    let azimuth = enu[0].atan2(enu[1]).to_degrees().rem_euclid(360.0);
    let (dist, heading, _) = station.geodesic_distance(&target);
    println!(
        "to target: ENU = {:.1} km, azimuth {azimuth:.2} deg",
        enu / 1e3
    );
    println!(
        "  geodesic distance {:.1} km, initial heading {:.2} deg",
        dist / 1e3,
        heading.to_degrees().rem_euclid(360.0)
    );

    // Orbit-local frames (RTN, NTW, LVLH) need the orbit state
    let r = Vector3::from_array([6_778_137.0, 0.0, 0.0]);
    let vel = Vector3::from_array([0.0, 5_000.0, 5_000.0]);
    let q_rtn = frametransform::rotation_with_state(Frame::GCRF, Frame::RTN, &t, &r, &vel)?;
    println!("velocity in RTN = {:.1} m/s", q_rtn * vel);

    Ok(())
}
