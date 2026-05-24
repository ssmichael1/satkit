//! End-to-end verification that the JPL ephemeris parser handles DE421
//! correctly.
//!
//! DE421 ships under JPL's older `lnxp*.421` naming and uses the smaller
//! `n_con == 400` constants block — a different code path through the
//! parser than the DE440 file the rest of the test suite exercises.
//! This test loads DE421 via [`init_from_path`] and checks predicted
//! positions against JPL's own `testpo.421` truth values.
//!
//! Skipped silently when the DE421 binary or test-vector file aren't
//! present in the testvecs directory (since they aren't part of the
//! default `satkit-testvecs` bundle).

use satkit::jplephem;
use satkit::{Instant, SolarSystem, TimeScale};

/// Locate the testvecs root the same way the in-source `testvecs` test
/// does: `SATKIT_TESTVEC_ROOT` if set, else `satkit-testvecs/` beside the
/// crate root.
fn testvec_root() -> Option<std::path::PathBuf> {
    if let Ok(v) = std::env::var("SATKIT_TESTVEC_ROOT") {
        return Some(std::path::PathBuf::from(v));
    }
    Some(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("satkit-testvecs"))
}

/// IAU 2012 definition; matches the value JPL bakes into the testpo files.
const AU_M: f64 = 149_597_870_700.0;

#[test]
fn de421_loads_and_matches_testpo_positions() {
    let Some(testdir) = testvec_root() else {
        eprintln!("skipping: SATKIT_TESTVEC_ROOT unset and no satkit-testvecs/ alongside");
        return;
    };
    let bin = testdir.join("jplephem").join("lnxp1900p2053.421");
    let testpo = testdir.join("jplephem").join("testpo.421");
    if !bin.is_file() || !testpo.is_file() {
        eprintln!(
            "skipping: DE421 binary or testpo.421 not present at {}",
            testdir.join("jplephem").display()
        );
        return;
    }

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

        let tm = Instant::from_jd_with_scale(jd, TimeScale::TT);
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
