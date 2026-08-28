//! Regression tests against NASA GMAT reference trajectories.
//!
//! GMAT cannot run in CI, so the references are generated offline by
//! `tests/gmat/generate.py` (GMAT `GmatConsole`, RK89 at 1e-14, SPICE
//! DE440, `EarthICRF`) and committed as JSON under `tests/gmat/cases/`.
//! Each file carries the epoch, force model, GMAT metadata, an hourly state
//! history, and the tolerance that gates the test.
//!
//! The test re-propagates *segment by segment* from its own previous state
//! to each GMAT sample time and compares position/velocity.  A failure prints
//! the per-sample residual table so the CI log shows *when* the divergence
//! begins (e.g. a lunar perigee passage) rather than only that it happened.
//!
//! Adding a case: append to `tests/gmat/cases.py`, run `generate.py`,
//! then add the name to `gmat_cases!` below.  Tolerances live in `cases.py`
//! (`generate.py --update-tolerances` rewrites the JSON without re-running
//! GMAT); tightening one is a reviewed change tied to a model improvement.

use std::path::{Path, PathBuf};

use satkit::earthgravity::GravityModel;
use satkit::orbitprop::{propagate, Integrator, PropSettings, SimpleState, TideModel};
use satkit::{Duration, Instant};
use serde::Deserialize;

#[derive(Deserialize)]
struct ForceModel {
    gravity_model: String,
    gravity_degree: u16,
    gravity_order: u16,
    sun: bool,
    moon: bool,
    tides: String,
    relativity: bool,
}

#[derive(Deserialize)]
struct Tolerance {
    pos_m: f64,
    vel_mps: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    epoch_utc: String,
    force_model: ForceModel,
    tolerance: Tolerance,
    /// `[elapsed_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms]` in EarthICRF.
    samples: Vec<[f64; 7]>,
}

fn case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gmat/cases")
}

/// Directory used by the `report` diagnostic (see below): defaults to the
/// committed corpus, overridable with `SATKIT_GMAT_CASE_DIR` so ad-hoc
/// cases generated into a scratch directory can be evaluated without
/// touching `tests/gmat/cases` or the `gmat_cases!` list.
fn report_dir() -> PathBuf {
    std::env::var_os("SATKIT_GMAT_CASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(case_dir)
}

fn load_case(name: &str) -> Case {
    load_case_from(&case_dir().join(format!("{name}.json")), name)
}

fn load_case_from(path: &Path, name: &str) -> Case {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let case: Case = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()));
    assert_eq!(case.name, name, "case name mismatch in {}", path.display());
    assert!(case.samples.len() >= 2, "{name}: need at least two samples");
    case
}

fn parse_epoch(iso: &str) -> Instant {
    // "YYYY-MM-DDTHH:MM:SS[.fff]" (UTC)
    let (date, time) = iso.split_once('T').expect("epoch_utc must be ISO-8601");
    let d: Vec<i32> = date.split('-').map(|s| s.parse().unwrap()).collect();
    let t: Vec<&str> = time.split(':').collect();
    Instant::from_datetime(
        d[0],
        d[1],
        d[2],
        t[0].parse().unwrap(),
        t[1].parse().unwrap(),
        t[2].parse().unwrap(),
    )
    .expect("valid epoch")
}

fn settings_for(fm: &ForceModel) -> PropSettings {
    // Explicit matches: an unknown string is a corpus error, not a default.
    let gravity_model = match fm.gravity_model.as_str() {
        "EGM96" => GravityModel::EGM96,
        "JGM3" => GravityModel::JGM3,
        "JGM2" => GravityModel::JGM2,
        "ITUGrace16" => GravityModel::ITUGrace16,
        other => panic!("unknown gravity_model {other:?}"),
    };
    let tide_model = match fm.tides.as_str() {
        "None" => TideModel::None,
        "SolidStep1" => TideModel::SolidStep1,
        "SolidFull" => TideModel::SolidFull,
        other => panic!("unknown tides {other:?}"),
    };
    PropSettings {
        gravity_model,
        gravity_degree: fm.gravity_degree,
        gravity_order: fm.gravity_order,
        use_sun_gravity: fm.sun,
        use_moon_gravity: fm.moon,
        tide_model,
        use_relativistic_correction: fm.relativity,
        use_spaceweather: false,
        integrator: Integrator::RKV98NoInterp,
        abs_error: 1e-13,
        rel_error: 1e-13,
        enable_interp: false,
        ..PropSettings::default()
    }
}

/// GMAT sample (km, km/s) -> satkit state (m, m/s).
fn to_state(s: &[f64; 7]) -> SimpleState {
    let mut st = SimpleState::zeros();
    for i in 0..6 {
        st[i] = s[i + 1] * 1e3;
    }
    st
}

struct Residual {
    elapsed_s: f64,
    pos_m: f64,
    vel_mps: f64,
}

/// Propagate a case and return the per-sample residuals vs GMAT.
fn evaluate(case: &Case) -> Vec<Residual> {
    let name = &case.name;
    let epoch = parse_epoch(&case.epoch_utc);
    let settings = settings_for(&case.force_model);

    let mut state = to_state(&case.samples[0]);
    let mut t_prev = epoch + Duration::from_seconds(case.samples[0][0]);
    let mut residuals = Vec::with_capacity(case.samples.len());

    for sample in &case.samples[1..] {
        let t = epoch + Duration::from_seconds(sample[0]);
        let res = propagate(&state, &t_prev, &t, &settings, None)
            .unwrap_or_else(|e| panic!("{name}: propagate failed at {} s: {e}", sample[0]));
        state = res.state_end;
        t_prev = t;

        let truth = to_state(sample);
        let dr = ((0..3).map(|i| (state[i] - truth[i]).powi(2)).sum::<f64>()).sqrt();
        let dv = ((3..6).map(|i| (state[i] - truth[i]).powi(2)).sum::<f64>()).sqrt();
        residuals.push(Residual {
            elapsed_s: sample[0],
            pos_m: dr,
            vel_mps: dv,
        });
    }
    residuals
}

fn print_table(name: &str, residuals: &[Residual]) {
    eprintln!("{name}: residuals vs GMAT (elapsed s, |dr| m, |dv| m/s):");
    for r in residuals {
        eprintln!(
            "  {:>9.0}  {:>12.5}  {:>12.4e}",
            r.elapsed_s, r.pos_m, r.vel_mps
        );
    }
}

fn run_case(name: &str) {
    let case = load_case(name);
    let residuals = evaluate(&case);
    let worst_pos = residuals
        .iter()
        .max_by(|a, b| a.pos_m.total_cmp(&b.pos_m))
        .unwrap();
    let worst_vel = residuals
        .iter()
        .max_by(|a, b| a.vel_mps.total_cmp(&b.vel_mps))
        .unwrap();
    let last = residuals.last().unwrap();
    println!(
        "{name}: max |dr| = {:.4} m @ {:.0} s (tol {} m); max |dv| = {:.3e} m/s @ {:.0} s (tol {:.1e} m/s); final |dr| = {:.4} m",
        worst_pos.pos_m, worst_pos.elapsed_s, case.tolerance.pos_m,
        worst_vel.vel_mps, worst_vel.elapsed_s, case.tolerance.vel_mps, last.pos_m
    );

    let ok = worst_pos.pos_m <= case.tolerance.pos_m && worst_vel.vel_mps <= case.tolerance.vel_mps;
    if !ok {
        print_table(name, &residuals);
        panic!(
            "{name}: exceeds GMAT tolerance (pos {:.4} m > {} m or vel {:.3e} > {:.1e} m/s)",
            worst_pos.pos_m, case.tolerance.pos_m, worst_vel.vel_mps, case.tolerance.vel_mps
        );
    }
}

/// Diagnostic, not a gate: evaluate every JSON in `SATKIT_GMAT_CASE_DIR`
/// (default: the committed corpus) and print the full residual table.
///
/// ```text
/// SATKIT_GMAT_CASE_DIR=/path/to/cases cargo test --test gmat_regression -- --ignored report --nocapture
/// ```
#[test]
#[ignore]
fn report() {
    let dir = report_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let case = load_case_from(&path, &name);
        let residuals = evaluate(&case);
        print_table(&name, &residuals);
    }
}

/// One `#[test]` per case so they run in parallel and fail individually.
macro_rules! gmat_cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() { run_case(stringify!($name)); }
        )*
        const CASE_NAMES: &[&str] = &[$(stringify!($name)),*];
    };
}

gmat_cases!(
    leo_iss_j2,
    leo_iss_full,
    leo_iss_gr,
    sso_800_j2,
    sso_800_full,
    meo_gps_j2,
    meo_gps_full,
    molniya_j2,
    molniya_full,
    geo_j2,
    geo_full,
    tess_j2,
    tess_full,
    tess_gr,
    cislunar_j2,
    cislunar_full,
    cislunar_gr,
);

/// Every JSON file in the corpus must be wired to a test above, and vice
/// versa, so a case can't be silently added without a gate.
#[test]
fn every_case_file_has_a_test() {
    let mut on_disk: Vec<String> = std::fs::read_dir(case_dir())
        .expect("tests/gmat/cases exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = CASE_NAMES.iter().map(|s| s.to_string()).collect();
    listed.sort();
    assert_eq!(
        on_disk, listed,
        "tests/gmat/cases/*.json and gmat_cases!(...) are out of sync"
    );
}
