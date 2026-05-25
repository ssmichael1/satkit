//! Integration test for the earthgravity bytes-loading entry points.
//!
//! Lives in `tests/` (separate binary, fresh singleton state) so we can
//! exercise the init-before-anything-else path without contention from the
//! in-source tests that lazy-load via the per-model accessors.

use satkit::earthgravity::{self, GravityModel};
use satkit::utils::datadir;

fn gravity_bytes(filename: &str) -> Option<Vec<u8>> {
    let path = datadir().ok()?.join(filename);
    std::fs::read(path).ok()
}

#[test]
fn init_from_bytes_then_query_and_double_init_errors() {
    let Some(bytes) = gravity_bytes("EGM96.gfc") else {
        eprintln!(
            "skipping: EGM96.gfc not available in datadir(); \
             run `python -m satkit.utils.update_datafiles` or set SATKIT_DATA"
        );
        return;
    };

    // 1. First init wins.
    earthgravity::init_from_bytes(GravityModel::EGM96, &bytes)
        .expect("init_from_bytes(EGM96) should succeed on first call");

    // 2. Acceleration query goes through the just-installed EGM96 singleton.
    //    Equatorial radius ≈ 6378 km; g ≈ 9.8 m/s² there.
    let pos = satkit::mathtypes::Vector3::from_array([6378.137e3, 0.0, 0.0]);
    let a = earthgravity::accel(&pos, 20, 20, GravityModel::EGM96);
    let g = a.norm();
    assert!(
        (9.5..10.0).contains(&g),
        "Equatorial gravity out of expected range: {g} m/s²"
    );

    // 3. Second init for the same model is rejected.
    let err = earthgravity::init_from_bytes(GravityModel::EGM96, &bytes)
        .expect_err("second init_from_bytes(EGM96) should return AlreadyInitialized");
    assert!(matches!(
        err,
        earthgravity::Error::AlreadyInitialized(GravityModel::EGM96)
    ));
}
