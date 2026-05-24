//! Integration test for the IERS-table bytes-loading entry points.
//!
//! Lives in `tests/` (separate binary, fresh singleton state) so we can
//! exercise the init-before-anything-else path without contention from the
//! in-source tests / qcirs2gcrs queries that would lazy-load.

use satkit::frametransform::ierstable::{self, IersTableId};
use satkit::utils::datadir;

fn iers_bytes(filename: &str) -> Option<Vec<u8>> {
    let path = datadir().ok()?.join(filename);
    std::fs::read(path).ok()
}

#[test]
fn init_all_three_tables_from_bytes_then_run_qcirs2gcrs() {
    let triples = [
        (IersTableId::Tab5A, "tab5.2a.txt"),
        (IersTableId::Tab5B, "tab5.2b.txt"),
        (IersTableId::Tab5D, "tab5.2d.txt"),
    ];

    let mut all_bytes = Vec::new();
    for (id, fname) in &triples {
        let Some(bytes) = iers_bytes(fname) else {
            eprintln!("skipping: {fname} not available in datadir()");
            return;
        };
        all_bytes.push((*id, bytes));
    }

    // 1. First init wins for all three.
    for (id, bytes) in &all_bytes {
        ierstable::init_from_bytes(*id, bytes)
            .unwrap_or_else(|e| panic!("init_from_bytes({id:?}) should succeed: {e}"));
    }

    // 2. qcirs2gcrs goes through the just-installed singletons; the
    //    quaternion should be near identity at J2000 (rotation magnitudes
    //    from CIO precession-nutation are at most a few arcseconds there).
    use satkit::{Instant, TimeScale};
    let t = Instant::from_jd_with_scale(2451545.0, TimeScale::TT);
    let q = satkit::frametransform::qcirs2gcrs(&t);
    let w = q.w.abs();
    assert!(
        w > 0.99999,
        "qcirs2gcrs at J2000 should be near-identity (|w| > 0.99999), got {w}"
    );

    // 3. Second init for any table is rejected.
    let (id, bytes) = &all_bytes[0];
    let err = ierstable::init_from_bytes(*id, bytes)
        .expect_err("second init_from_bytes should return IersTableAlreadyInitialized");
    assert!(matches!(
        err,
        satkit::frametransform::Error::IersTableAlreadyInitialized { id: e_id } if e_id == *id
    ));
}
