//! End-to-end verification that the JPL ephemeris parser handles DE421
//! correctly.
//!
//! DE421 ships under JPL's older `lnxp*.421` naming and uses the smaller
//! `n_con == 400` constants block — a different code path through the
//! parser than the DE440 file the rest of the test suite exercises.
//! This test loads DE421 via [`init_from_path`] and checks predicted
//! positions against JPL's own `testpo.421` truth values.
//!
//! It is also the check that the public API evaluates the ephemeris at
//! TDB on a second file (`tests/jplephem_tdb.rs` covers DE440 without test
//! vectors).
//!
//! # Running it
//!
//! The DE421 binary (`lnxp1900p2053.421`) and `testpo.421` are not yet in
//! the test-vector bucket, so the test is `#[ignore]`d: `cargo test` lists
//! it as ignored rather than passing it silently. Run it with
//!
//! ```text
//! cargo test --test jplephem_de421_testpo -- --ignored
//! ```
//!
//! and it FAILS if either file is missing under
//! `$SATKIT_TESTVEC_ROOT/jplephem/`. CI runs it that way as soon as both
//! files are in the restored test vectors, and otherwise posts a warning
//! annotation on the run (the "DE421" step of the rust job in
//! `.github/workflows/build.yml`).

use satkit::jplephem;
use satkit::{Instant, SolarSystem, TimeScale};

/// Locate the testvecs root the same way the in-source `testvecs` test
/// does: `SATKIT_TESTVEC_ROOT` if set, else `satkit-testvecs/` beside the
/// crate root.
fn testvec_root() -> std::path::PathBuf {
    match std::env::var("SATKIT_TESTVEC_ROOT") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("satkit-testvecs"),
    }
}

/// IAU 2012 definition; matches the value JPL bakes into the testpo files.
const AU_M: f64 = 149_597_870_700.0;

#[test]
#[ignore = "needs jplephem/lnxp1900p2053.421 and jplephem/testpo.421 under SATKIT_TESTVEC_ROOT \
            (not yet in the test-vector bucket); run with --ignored"]
fn de421_loads_and_matches_testpo_positions() {
    let testdir = testvec_root();
    let bin = testdir.join("jplephem").join("lnxp1900p2053.421");
    let testpo = testdir.join("jplephem").join("testpo.421");
    let missing: Vec<String> = [&bin, &testpo]
        .iter()
        .filter(|p| !p.is_file())
        .map(|p| p.display().to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "DE421 test requested but its files are missing: {}. Fetch them from \
         https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de421/ into \
         $SATKIT_TESTVEC_ROOT/jplephem/ (the test-vector bucket does not carry them yet).",
        missing.join(", ")
    );

    jplephem::init_from_path(&bin).expect("DE421 should load via init_from_path");

    let testpo_text = std::fs::read_to_string(&testpo).unwrap();
    let mut checked = 0usize;
    let mut failed: Vec<String> = Vec::new();

    for line in testpo_text.lines().skip(14) {
        let s: Vec<&str> = line.split_whitespace().collect();
        if s.len() < 7 {
            continue;
        }
        let jd: f64 = s[2].parse().unwrap();
        let tar: i32 = s[3].parse().unwrap();
        let src: i32 = s[4].parse().unwrap();
        let coord: usize = s[5].parse().unwrap();
        let truth: f64 = s[6].parse().unwrap();

        // Scope of this test:
        //   * positions only (coord 1..=3) — velocity rows would require the
        //     AU/day unit conversion the in-source `testvecs` test handles
        //   * skip rows touching Earth (testpo index 3) — Earth ≠ EMB and
        //     the conversion needs `emrat`, which isn't on the public API
        //   * skip Sun/SSB/EMB (testpo 11/12/13) — same shape as the
        //     in-source test
        if !(1..=3).contains(&coord) {
            continue;
        }
        if tar == 3 || src == 3 {
            continue;
        }
        if !(1..=10).contains(&tar) || !(1..=10).contains(&src) {
            continue;
        }

        // testpo epochs are JD in T_eph (TDB)
        let tm = Instant::from_jd_with_scale(jd, TimeScale::TDB);
        let tbody = SolarSystem::try_from(tar - 1).expect("valid solar body index");
        let sbody = SolarSystem::try_from(src - 1).expect("valid solar body index");
        // testpo.421 includes rows at the very edge of DE421's span; the
        // parser correctly rejects out-of-range queries with
        // `InvalidJulianDate`. Skip those — they're not parser bugs.
        let (tpos, _) = match jplephem::geocentric_state(tbody, &tm) {
            Ok(v) => v,
            Err(jplephem::Error::InvalidJulianDate(_)) => continue,
            Err(e) => panic!("geocentric_state for target: {e:?}"),
        };
        let (spos, _) = match jplephem::geocentric_state(sbody, &tm) {
            Ok(v) => v,
            Err(jplephem::Error::InvalidJulianDate(_)) => continue,
            Err(e) => panic!("geocentric_state for source: {e:?}"),
        };
        let diff_au = (tpos - spos) / AU_M;
        let got = diff_au[coord - 1];

        let rel = (got - truth).abs() / truth.abs().max(1.0);
        if rel > 1.0e-10 {
            failed.push(format!(
                "tar={tar} src={src} coord={coord} jd={jd}: got {got:.15} expected {truth:.15} \
                 (rel err {rel:.2e})"
            ));
        }
        checked += 1;
    }

    if !failed.is_empty() {
        for line in failed.iter().take(5) {
            eprintln!("MISMATCH: {line}");
        }
        panic!(
            "DE421 testpo validation: {} of {} vectors failed",
            failed.len(),
            checked
        );
    }
    assert!(
        checked >= 50,
        "expected to validate >=50 testpo rows, got {checked}"
    );
    eprintln!("DE421 testpo: {checked} position vectors passed");
}
