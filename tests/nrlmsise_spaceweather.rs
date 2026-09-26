//! NRLMSISE-00 against a small synthetic GFZ table loaded with
//! `init_from_bytes`, so the result does not depend on the live file.
//!
//! Lives in `tests/` (separate binary) because it replaces the process-wide
//! space-weather table.

use satkit::nrlmsise::nrlmsise;
use satkit::spaceweather;
use satkit::{Duration, Instant};

/// One GFZ `Kp_ap_Ap_SN_F107_since_1932.txt` row: all eight 3-hourly ap and
/// the daily Ap equal to `ap`, observed and adjusted F10.7 equal to `f107`
/// (`-1.0` for a missing value, as GFZ writes it).
fn gfz_row(date: Instant, ap: i32, f107: f64) -> String {
    let (y, m, d, _, _, _) = date.as_datetime();
    let kp = " 3.000".repeat(8);
    let aps = format!(" {ap:4}").repeat(8);
    format!("{y:4} {m:02} {d:02} 0 0.5 2600 1{kp}{aps} {ap:5} 100 {f107:8.1} {f107:8.1} 2")
}

/// `days` consecutive rows from `start`; F10.7 is `f107(i)` on day `i`.
fn gfz_table(start: Instant, days: usize, ap: i32, f107: impl Fn(usize) -> f64) -> Vec<u8> {
    let mut s = String::from("# synthetic GFZ table\n");
    for i in 0..days {
        s += &gfz_row(start + Duration::from_days(i as f64), ap, f107(i));
        s.push('\n');
    }
    s.into_bytes()
}

/// Density at 400 km with the loaded table, and with the model defaults
/// (F10.7 = F10.7A = 150, Ap = 4).
fn with_and_without(tm: &Instant) -> (f64, f64) {
    let (sw, _) = nrlmsise(400.0, Some(30.0), Some(40.0), Some(tm), true);
    let (def, _) = nrlmsise(400.0, Some(30.0), Some(40.0), Some(tm), false);
    (sw, def)
}

#[test]
fn missing_flux_keeps_the_rest_of_the_space_weather() {
    spaceweather::disable_space_weather_time_warning();

    // A storm-level Ap = 50 table with 2025-02-12 missing its F10.7 (a real
    // gap in the GFZ record). The day after used to drop every index and run
    // on the quiet defaults — 35-39 % low in density.
    let start = Instant::from_date(2024, 12, 1).unwrap();
    let gap = (Instant::from_date(2025, 2, 12).unwrap() - start).as_days() as usize;
    let table = gfz_table(start, 150, 50, |i| if i == gap { -1.0 } else { 150.0 });
    spaceweather::init_from_bytes(&table).unwrap();

    let after_gap = Instant::from_datetime(2025, 2, 13, 12, 0, 0.0).unwrap();
    let (sw, def) = with_and_without(&after_gap);
    assert!(
        sw > 1.2 * def,
        "Ap = 50 was dropped: {sw:e} vs defaults {def:e}"
    );
    // With F10.7 = F10.7A = 150 filled from the neighbouring days, the only
    // difference from the day before the gap is the day of year.
    let (before_gap, _) = with_and_without(&(after_gap - Duration::from_days(2.0)));
    assert!(
        (sw / before_gap - 1.0).abs() < 0.02,
        "{sw:e} vs {before_gap:e}"
    );

    // Before 1947 GFZ has Ap but no F10.7 at all: F10.7 takes the default,
    // the measured Ap is still used.
    let start = Instant::from_date(1932, 1, 1).unwrap();
    spaceweather::init_from_bytes(&gfz_table(start, 120, 50, |_| -1.0)).unwrap();
    let rec = spaceweather::get(&Instant::from_date(1932, 2, 15).unwrap()).unwrap();
    assert_eq!(
        rec.f10p7_obs_c81, -1.0,
        "81-day average of no measured days"
    );
    let (sw, def) = with_and_without(&Instant::from_datetime(1932, 2, 15, 12, 0, 0.0).unwrap());
    assert!(
        sw > 1.2 * def,
        "Ap = 50 was dropped: {sw:e} vs defaults {def:e}"
    );

    // Before the table: the defaults, no panic.
    let (sw, def) = with_and_without(&Instant::from_datetime(1931, 6, 1, 12, 0, 0.0).unwrap());
    assert_eq!(sw, def);
}
