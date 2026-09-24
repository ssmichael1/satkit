//! Integration test for the spaceweather bytes-loading entry points.
//!
//! Lives in `tests/` (separate binary, fresh singleton state) so we can
//! exercise the init path without contention from the in-source test that
//! lazy-loads via `get()`.
//!
//! Unlike the static subsystems (jplephem, gravity, IERS tables),
//! `spaceweather` is intentionally refreshable: `init_from_bytes` always
//! succeeds and replaces.

use satkit::spaceweather;
use satkit::utils::datadir;
use satkit::Instant;

/// The GFZ table if provisioned (what the default loader reads now), else a
/// cached CelesTrak `SW-All.csv`: `init_from_bytes` detects either format.
fn sw_bytes() -> Option<Vec<u8>> {
    let dir = datadir().ok()?;
    ["Kp_ap_Ap_SN_F107_since_1932.txt", "SW-All.csv"]
        .iter()
        .find_map(|name| std::fs::read(dir.join(name)).ok())
}

#[test]
fn init_from_bytes_replaces_and_query_works() {
    let Some(bytes) = sw_bytes() else {
        eprintln!(
            "skipping: no space-weather file in datadir(); \
             run `python -m satkit.utils.update_datafiles` or set SATKIT_DATA"
        );
        return;
    };

    // 1. First init populates the singleton.
    spaceweather::init_from_bytes(&bytes).expect("init_from_bytes should succeed on first call");

    // 2. Query against the just-installed records.
    let tm = Instant::from_datetime(2023, 11, 14, 0, 0, 0.0).unwrap();
    let r1 = spaceweather::get(&tm).expect("get() should find a record for 2023-11-14");

    // 3. Second init succeeds (refresh-in-place semantics) and yields the
    //    same record for the same query.
    spaceweather::init_from_bytes(&bytes)
        .expect("second init_from_bytes should succeed (refreshable subsystem)");
    let r2 = spaceweather::get(&tm).expect("get() after reload");

    assert_eq!(r1.date, r2.date);
    assert_eq!(r1.f10p7_obs, r2.f10p7_obs);
}
