//! Integration test for the earth_orientation_params bytes-loading entry
//! points. Refresh-in-place semantics: `init_from_bytes` always succeeds
//! and replaces.

use satkit::earth_orientation_params as eop;
use satkit::utils::datadir;

fn eop_bytes() -> Option<Vec<u8>> {
    let path = datadir().ok()?.join("EOP-All.csv");
    std::fs::read(path).ok()
}

#[test]
fn init_from_bytes_replaces_and_query_works() {
    let Some(bytes) = eop_bytes() else {
        eprintln!(
            "skipping: EOP-All.csv not available in datadir(); \
             run `python -m satkit.utils.update_datafiles` or set SATKIT_DATA"
        );
        return;
    };

    // 1. First init populates the singleton.
    eop::init_from_bytes(&bytes).expect("init_from_bytes should succeed on first call");

    // 2. Query against the just-installed records (known truth value from
    //    the in-source test).
    let v = eop::eop_from_mjd_utc(59464.00).expect("EOP for MJD 59464");
    let truth: [f64; 4] = [-0.1145667, 0.241155, 0.317274, -0.0002255];
    for (a, b) in v.iter().zip(truth.iter()) {
        assert!(
            ((a - b) / b).abs() < 1.0e-3,
            "EOP mismatch after bytes init: got {a}, expected {b}"
        );
    }

    // 3. Second init succeeds (refresh-in-place semantics) and the query
    //    still works.
    eop::init_from_bytes(&bytes)
        .expect("second init_from_bytes should succeed (refreshable subsystem)");
    let v2 = eop::eop_from_mjd_utc(59464.00).expect("EOP for MJD 59464 after reload");
    assert_eq!(v, v2);
}
