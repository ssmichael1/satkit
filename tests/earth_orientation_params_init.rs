//! Integration test for the earth_orientation_params bytes-loading entry
//! points. Refresh-in-place semantics: `init_from_bytes` always succeeds
//! and replaces.

use satkit::earth_orientation_params as eop;
use satkit::utils::datadir;

/// Either EOP file from the data directory (the IERS file first, as the
/// loader prefers it).
fn eop_bytes() -> Option<Vec<u8>> {
    let dir = datadir().ok()?;
    [eop::FINALS2000A_FILE, eop::CELESTRAK_FILE]
        .iter()
        .find_map(|name| std::fs::read(dir.join(name)).ok())
}

#[test]
fn init_from_bytes_replaces_and_query_works() {
    let Some(bytes) = eop_bytes() else {
        eprintln!(
            "skipping: no EOP file (finals2000A.all / EOP-All.csv) in datadir(); \
             run `python -m satkit.utils.update_datafiles` or set SATKIT_DATA"
        );
        return;
    };

    // 1. First init populates the singleton.
    eop::init_from_bytes(&bytes).expect("init_from_bytes should succeed on first call");

    // 2. Query against the just-installed records (known truth value from
    //    the in-source test; the IERS and CelesTrak files agree to well
    //    within the tolerance on UT1-UTC and polar motion, LOD less so).
    assert!(eop::source().is_some());
    let v = eop::eop_from_mjd_utc(59464.00).expect("EOP for MJD 59464");
    let truth: [f64; 3] = [-0.1145667, 0.241155, 0.317274];
    for (a, b) in v.iter().zip(truth.iter()) {
        assert!(
            ((a - b) / b).abs() < 1.0e-3,
            "EOP mismatch after bytes init: got {a}, expected {b}"
        );
    }
    assert!((v[3] - -0.0002255).abs() < 5.0e-5, "LOD {}", v[3]);

    // 3. Second init succeeds (refresh-in-place semantics) and the query
    //    still works.
    eop::init_from_bytes(&bytes)
        .expect("second init_from_bytes should succeed (refreshable subsystem)");
    let v2 = eop::eop_from_mjd_utc(59464.00).expect("EOP for MJD 59464 after reload");
    assert_eq!(v, v2);

    // 4. Coverage / status on the real table.
    let cov = eop::coverage().expect("coverage after init");
    assert!(cov.first < cov.last_observed && cov.last_observed <= cov.last);
    assert_eq!(eop::status(&cov.first), eop::EopStatus::Observed);
    assert_eq!(
        eop::status(&(cov.last + satkit::Duration::from_days(1.0))),
        eop::EopStatus::Extrapolated
    );

    // 5. An empty table (header only) counts as *not loaded*: no coverage,
    //    queries return None, and the propagator refuses to build its
    //    ephemeris table rather than run with zero EOP.
    let header = b"DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n";
    eop::init_from_bytes(header).expect("header-only init parses");
    assert!(eop::coverage().is_none());
    assert_eq!(eop::status(&cov.first), eop::EopStatus::NotLoaded);
    assert!(eop::eop_from_mjd_utc(59464.00).is_none());
    let t0 = satkit::Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
    let t1 = t0 + satkit::Duration::from_hours(1.0);
    match satkit::orbitprop::Precomputed::new(&t0, &t1) {
        Err(satkit::orbitprop::Error::EopUnavailable) => {}
        other => panic!("expected EopUnavailable with an empty EOP table, got {other:?}"),
    }

    // 6. Restore the real table so nothing else in this process is affected.
    eop::init_from_bytes(&bytes).expect("restore");
    assert!(eop::coverage().is_some());
}
