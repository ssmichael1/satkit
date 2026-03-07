//! Solar Cycle Forecast data from NOAA/SWPC
//!
//! Provides predicted F10.7 solar flux values for future dates,
//! sourced from the NOAA Space Weather Prediction Center's
//! solar cycle prediction JSON endpoint.
//!
//! Used as a fallback when historical space weather data is not
//! available (i.e., for future propagation dates).

use crate::utils::{datadir, download_to_string};
use crate::{Instant, TimeLike};
use anyhow::{bail, Result};

use std::path::PathBuf;
use std::sync::RwLock;

use once_cell::sync::OnceCell;

#[derive(Debug, Clone)]
pub struct ForecastRecord {
    pub date: Instant,
    pub predicted_f107: f64,
}

fn forecast_path() -> Result<PathBuf> {
    Ok(datadir()?.join("predicted-solar-cycle.json"))
}

fn load_forecast() -> Result<Vec<ForecastRecord>> {
    let path = forecast_path()?;
    if !path.is_file() {
        bail!("Solar cycle forecast file not found");
    }
    let contents = std::fs::read_to_string(&path)?;
    parse_forecast_json(&contents)
}

fn parse_forecast_json(contents: &str) -> Result<Vec<ForecastRecord>> {
    let parsed = json::parse(contents)?;
    if !parsed.is_array() {
        bail!("Expected JSON array in solar cycle forecast");
    }

    let mut records = Vec::new();
    for entry in parsed.members() {
        let time_tag = entry["time-tag"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing time-tag"))?;

        // Parse "YYYY-MM" format
        let parts: Vec<&str> = time_tag.split('-').collect();
        if parts.len() != 2 {
            continue;
        }
        let year: i32 = parts[0].parse()?;
        let month: i32 = parts[1].parse()?;
        // Use the 15th of each month as the representative date
        let date = Instant::from_date(year, month, 15)?;

        let f107 = entry["predicted_f10.7"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("Missing predicted_f10.7"))?;

        records.push(ForecastRecord {
            date,
            predicted_f107: f107,
        });
    }

    records.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
    Ok(records)
}

fn forecast_singleton() -> &'static RwLock<Option<Vec<ForecastRecord>>> {
    static INSTANCE: OnceCell<RwLock<Option<Vec<ForecastRecord>>>> = OnceCell::new();
    INSTANCE.get_or_init(|| RwLock::new(load_forecast().ok()))
}

/// Get predicted F10.7 solar flux for a given time.
///
/// Linearly interpolates between monthly forecast entries.
///
/// Returns `None` if no forecast data is available or the time
/// is outside the forecast range.
pub fn get_predicted_f107<T: TimeLike>(tm: &T) -> Option<f64> {
    let tm = tm.as_instant();
    let lock = forecast_singleton().read().unwrap();
    let records = lock.as_ref()?;

    if records.is_empty() {
        return None;
    }

    // Before first entry
    if tm < records[0].date {
        return None;
    }

    // After last entry — use last value
    if tm >= records[records.len() - 1].date {
        return Some(records[records.len() - 1].predicted_f107);
    }

    // Find bracketing entries and interpolate
    let idx = records.partition_point(|r| r.date <= tm);
    if idx == 0 {
        return Some(records[0].predicted_f107);
    }

    let r0 = &records[idx - 1];
    let r1 = &records[idx];
    let frac = (tm - r0.date).as_seconds() / (r1.date - r0.date).as_seconds();
    Some(r0.predicted_f107 + frac * (r1.predicted_f107 - r0.predicted_f107))
}

/// Download the latest solar cycle forecast from NOAA/SWPC.
pub fn update() -> Result<()> {
    let url = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json";
    let contents = download_to_string(url)?;

    // Validate before saving
    let records = parse_forecast_json(&contents)?;
    if records.is_empty() {
        bail!("Downloaded forecast contains no records");
    }

    let path = forecast_path()?;
    std::fs::write(&path, &contents)?;

    // Reload singleton
    *forecast_singleton().write().unwrap() = Some(records);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json() {
        let json = r#"[
            {"time-tag": "2026-01", "predicted_ssn": 100.0, "predicted_f10.7": 145.0},
            {"time-tag": "2026-06", "predicted_ssn": 90.0, "predicted_f10.7": 135.0},
            {"time-tag": "2027-01", "predicted_ssn": 80.0, "predicted_f10.7": 125.0}
        ]"#;
        let records = parse_forecast_json(json).unwrap();
        assert_eq!(records.len(), 3);
        assert!((records[0].predicted_f107 - 145.0).abs() < 1e-6);
        assert!((records[2].predicted_f107 - 125.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolation() {
        let json = r#"[
            {"time-tag": "2026-01", "predicted_ssn": 100.0, "predicted_f10.7": 140.0},
            {"time-tag": "2026-07", "predicted_ssn": 90.0, "predicted_f10.7": 120.0}
        ]"#;
        let records = parse_forecast_json(json).unwrap();
        *forecast_singleton().write().unwrap() = Some(records);

        // Midpoint should be ~130
        let mid = Instant::from_date(2026, 4, 15).unwrap();
        let f107 = get_predicted_f107(&mid).unwrap();
        assert!((f107 - 130.0).abs() < 2.0);

        // Before range
        let early = Instant::from_date(2025, 1, 1).unwrap();
        assert!(get_predicted_f107(&early).is_none());
    }

    #[test]
    fn test_download_and_parse() {
        let url = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json";
        let contents = crate::utils::download_to_string(url).unwrap();
        let records = parse_forecast_json(&contents).unwrap();
        assert!(
            records.len() > 10,
            "Expected at least 10 forecast records, got {}",
            records.len()
        );
        // F10.7 values should be physically reasonable (50-400)
        for r in &records {
            assert!(
                r.predicted_f107 > 50.0 && r.predicted_f107 < 400.0,
                "Unreasonable F10.7 value: {}",
                r.predicted_f107
            );
        }
    }
}
