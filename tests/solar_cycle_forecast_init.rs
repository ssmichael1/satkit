//! Integration test for the solar_cycle_forecast bytes-loading entry
//! points. Refresh-in-place semantics: `init_from_bytes` always succeeds
//! and replaces.

use satkit::solar_cycle_forecast;
use satkit::Instant;

const FORECAST_A: &[u8] = br#"[
    {"time-tag": "2027-01", "predicted_ssn": 100.0, "predicted_f10.7": 140.0},
    {"time-tag": "2027-07", "predicted_ssn": 90.0, "predicted_f10.7": 120.0}
]"#;

const FORECAST_B: &[u8] = br#"[
    {"time-tag": "2027-01", "predicted_ssn": 200.0, "predicted_f10.7": 200.0},
    {"time-tag": "2027-07", "predicted_ssn": 180.0, "predicted_f10.7": 180.0}
]"#;

#[test]
fn init_from_bytes_replaces() {
    let mid = Instant::from_date(2027, 4, 15).unwrap();

    // 1. Install forecast A; midpoint should interpolate to ~130.
    solar_cycle_forecast::init_from_bytes(FORECAST_A).expect("first init");
    let f_a = solar_cycle_forecast::get_predicted_f107(&mid).expect("A midpoint");
    assert!(
        (f_a - 130.0).abs() < 2.0,
        "forecast A midpoint should be ~130, got {f_a}"
    );

    // 2. Replace with forecast B; same midpoint should now be ~190.
    solar_cycle_forecast::init_from_bytes(FORECAST_B)
        .expect("second init should succeed (refreshable subsystem)");
    let f_b = solar_cycle_forecast::get_predicted_f107(&mid).expect("B midpoint");
    assert!(
        (f_b - 190.0).abs() < 2.0,
        "forecast B midpoint should be ~190, got {f_b}"
    );
    assert!(
        (f_b - f_a).abs() > 50.0,
        "replace should have meaningfully changed the value"
    );
}
