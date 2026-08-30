//! What satkit can do with **no data directory and no network**.
//!
//! The core data (IERS nutation tables, gravity models to degree 70) is
//! compiled into the library, so everything below must work in a process
//! whose data directory is empty and where `SATKIT_OFFLINE=1` forbids
//! downloads. CI runs this file exactly that way (see `build.yml`); in a
//! normal developer run the data directory is populated and the same tests
//! simply exercise the on-disk path.
//!
//! Not offline-capable, by design: the JPL ephemeris (downloaded on first
//! use, SHA-256 verified) and the Earth-orientation / space-weather files
//! (refreshed from CelesTrak). `offline_missing_ephemeris_is_typed_error`
//! checks that asking for the ephemeris in that state is a clean error, not
//! a hang or a panic.

use satkit::earthgravity::{self, GravityModel};
use satkit::frametransform::ierstable;
use satkit::{Duration, Instant, TimeScale};

fn offline_env() -> bool {
    std::env::var("SATKIT_OFFLINE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Earth gravity at degree 20 from the compiled-in (or on-disk) EGM96.
#[test]
fn offline_gravity_acceleration() {
    let pos_itrf = numeris::vector![7000.0e3, 0.0, 0.0];
    let a = earthgravity::accel(&pos_itrf, 20, 20, GravityModel::EGM96);
    let mag = a.norm();
    // ~ GM / r^2 = 3.986e14 / (7e6)^2 = 8.13 m/s^2
    assert!((mag - 8.135).abs() < 0.02, "|a| = {mag}");
    assert!(a[0] < 0.0, "points toward Earth");
    // All four models must be available without a data directory.
    for model in [
        GravityModel::JGM3,
        GravityModel::JGM2,
        GravityModel::ITUGrace16,
    ] {
        let b = earthgravity::accel(&pos_itrf, 12, 12, model);
        assert!((b.norm() - mag).abs() < 0.01, "{model:?}");
    }
}

/// The IERS 2010 nutation / CIO tables load from the compiled-in copies, and
/// the precession-nutation quaternion (which needs them, and only zeros from
/// the absent EOP table for dX/dY) evaluates to a sane rotation.
#[test]
fn offline_iers_tables_and_precession_nutation() {
    ierstable::preload().expect("IERS tables available without data files");
    let t = Instant::from_datetime(2024, 6, 1, 0, 0, 0.0).unwrap();
    let q = satkit::frametransform::qcirs2gcrs(&t);
    let (_axis, angle) = q.to_axis_angle();
    // The CIP has moved ~2004"/century × 0.24 century ≈ 480" ≈ 0.13° since
    // J2000 (the X/Y series), so the CIRS→GCRS rotation is ~0.13°.
    let angle = angle.abs().min(std::f64::consts::TAU - angle.abs());
    assert!(
        angle > 0.10f64.to_radians() && angle < 0.20f64.to_radians(),
        "{angle}"
    );
}

/// Time scales that do not need UT1 (TAI/TT/GPS/TDB) are pure arithmetic.
#[test]
fn offline_time_scales() {
    let t = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
    let tai = t.as_mjd_with_scale(TimeScale::TAI);
    let utc = t.as_mjd_with_scale(TimeScale::UTC);
    assert!(((tai - utc) * 86400.0 - 37.0).abs() < 1e-6);
    let tt = t.as_mjd_with_scale(TimeScale::TT);
    assert!(((tt - tai) * 86400.0 - 32.184).abs() < 1e-6);
    let _ = t + Duration::from_days(1.5);
}

/// SGP4 needs no data files at all.
#[test]
fn offline_sgp4() {
    let lines = [
        "ISS (ZARYA)".to_string(),
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927".to_string(),
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537".to_string(),
    ];
    let mut tle = satkit::TLE::from_lines(&lines)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let t = tle.epoch + Duration::from_hours(1.0);
    let s = satkit::sgp4::sgp4(&mut tle, &[t]).unwrap();
    let r = (s.pos[(0, 0)].powi(2) + s.pos[(1, 0)].powi(2) + s.pos[(2, 0)].powi(2)).sqrt();
    assert!((r - 6.78e6).abs() < 5e4, "r = {r}");
    let v = (s.vel[(0, 0)].powi(2) + s.vel[(1, 0)].powi(2) + s.vel[(2, 0)].powi(2)).sqrt();
    assert!((v - 7.66e3).abs() < 1e2, "v = {v}");
}

/// Keplerian propagation and the Lambert solver are pure math.
#[test]
fn offline_kepler_and_lambert() {
    use satkit::kepler::Anomaly;
    let k = satkit::Kepler::new(7000.0e3, 0.001, 0.9, 0.1, 0.2, Anomaly::True(0.3));
    let (r0, v0) = k.to_pv();
    let (r1, _v1) = k.propagate(&Duration::from_seconds(600.0)).to_pv();
    assert!((r0.norm() - 7000.0e3).abs() < 2e4);
    let sols = satkit::lambert::lambert(&r0, &r1, 600.0, satkit::consts::MU_EARTH, true).unwrap();
    let (v1, _v2) = sols[0];
    // The Lambert velocity at r0 must reproduce the Kepler velocity.
    assert!((v1 - v0).norm() < 0.5, "{}", (v1 - v0).norm());
}

/// With no ephemeris on disk and downloads forbidden, asking for a
/// geocentric planet position is a typed error that names the file and its
/// sources — never a hang, a panic, or a silent zero.
///
/// Only meaningful in the CI offline job (`SATKIT_DATA=<empty dir>`,
/// `SATKIT_OFFLINE=1`); in a provisioned developer environment the
/// ephemeris is present and the assertion is skipped.
#[test]
fn offline_missing_ephemeris_is_typed_error() {
    if !offline_env() || satkit::utils::data_found() {
        eprintln!("skipped: ephemeris present or SATKIT_OFFLINE not set");
        return;
    }
    let t = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
    let err = satkit::jplephem::geocentric_pos(satkit::SolarSystem::Moon, &t)
        .expect_err("no ephemeris and no network must be an error");
    let msg = err.to_string();
    assert!(msg.contains("SATKIT_OFFLINE"), "{msg}");
    assert!(msg.contains("linux_p1550p2650.440"), "{msg}");
    assert!(
        msg.contains("https://"),
        "should list the manifest sources: {msg}"
    );
}
