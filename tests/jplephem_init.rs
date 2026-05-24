//! Integration test for the JPL ephemeris bytes-loading entry points.
//!
//! Lives in `tests/` (separate binary, fresh singleton state) so we can
//! exercise the init-before-anything-else path without contention from the
//! in-source tests that lazy-load via `jplephem_singleton()`.

use satkit::jplephem;
use satkit::utils::datadir;
use satkit::{Instant, SolarSystem};

fn legacy_file_bytes() -> Option<Vec<u8>> {
    let path = datadir().ok()?.join("linux_p1550p2650.440");
    std::fs::read(path).ok()
}

#[test]
fn init_from_bytes_then_query_and_double_init_errors() {
    let Some(bytes) = legacy_file_bytes() else {
        eprintln!(
            "skipping: linux_p1550p2650.440 not available in datadir(); \
             run `python -m satkit.utils.update_datafiles` or set SATKIT_DATA"
        );
        return;
    };

    // 1. First init wins.
    jplephem::init_from_bytes(&bytes).expect("init_from_bytes should succeed on first call");

    // 2. Position query goes through the just-installed singleton.
    let tm = Instant::from_datetime(2024, 3, 1, 12, 0, 0.0).unwrap();
    let pos = jplephem::geocentric_pos(SolarSystem::Moon, &tm).expect("geocentric_pos");
    // Sanity: Moon is ~3.8e8 m from Earth.
    let r = pos.norm();
    assert!(
        (3.0e8..5.0e8).contains(&r),
        "Moon distance out of expected range: {r}"
    );

    // 3. Second init is rejected.
    let err = jplephem::init_from_bytes(&bytes)
        .expect_err("second init_from_bytes should return AlreadyInitialized");
    assert!(matches!(err, jplephem::Error::AlreadyInitialized));
}
