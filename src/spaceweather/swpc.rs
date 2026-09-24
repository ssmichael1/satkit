//! NOAA/SWPC 45-day Ap and F10.7 forecast — daily resolution across the
//! near-term window.
//!
//! MSAFE already spans this period at monthly cadence, so this is not filling
//! a coverage hole: it supplies day-scale F10.7 and Ap across the only forward
//! window where geomagnetic activity is meaningfully predictable. US
//! Government work, public domain.
//!
//! Served at <https://services.swpc.noaa.gov/text/45-day-forecast.txt>.

use super::{Error, Result, SpaceWeatherDataType, SpaceWeatherRecord};
use crate::Instant;
use std::collections::BTreeMap;

/// `24Sep26` → 2026-09-24.
fn parse_date(s: &str) -> Option<Instant> {
    if s.len() != 7 {
        return None;
    }
    let day: i32 = s[0..2].parse().ok()?;
    let month = super::msafe::month_number(&s[2..5])?;
    let yy: i32 = s[5..7].parse().ok()?;
    Instant::from_date(2000 + yy, month, day).ok()
}

/// Parse the 45-day forecast: two sections, `45-DAY AP FORECAST` and
/// `45-DAY F10.7 CM FLUX FORECAST`, each a run of `DDMonYY value` pairs.
pub fn parse(text: &str) -> Result<Vec<SpaceWeatherRecord>> {
    let mut ap: BTreeMap<i64, (Instant, i32)> = BTreeMap::new();
    let mut f107: BTreeMap<i64, f64> = BTreeMap::new();
    let mut section = 0; // 0 none, 1 ap, 2 f107
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with("45-DAY AP") {
            section = 1;
            continue;
        }
        if l.starts_with("45-DAY F10.7") {
            section = 2;
            continue;
        }
        if section == 0 || l.is_empty() || l.starts_with(':') || l.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = l.split_whitespace().collect();
        // The data block ends at the first line that does not open with a
        // date: the `FORECASTER:` footer and the `99999` terminator.
        if f.first().and_then(|s| parse_date(s)).is_none() {
            section = 0;
            continue;
        }
        for pair in f.chunks(2) {
            if pair.len() != 2 {
                return Err(Error::InvalidEntry);
            }
            let Some(d) = parse_date(pair[0]) else {
                return Err(Error::InvalidNumber("swpc date"));
            };
            let v: f64 = pair[1]
                .parse()
                .map_err(|_| Error::InvalidNumber("swpc value"))?;
            let key = d.utc_day_number();
            match section {
                1 => {
                    ap.insert(key, (d, v.round() as i32));
                }
                _ => {
                    f107.insert(key, v);
                }
            }
        }
    }
    let out: Vec<SpaceWeatherRecord> = ap
        .into_iter()
        .filter_map(|(k, (date, a))| {
            let flux = *f107.get(&k)?;
            let (cp, c9) = super::gfz::cp_c9(8 * a);
            Some(SpaceWeatherRecord {
                date,
                bsrn: -1,
                nd: -1,
                data_type: SpaceWeatherDataType::PredictedDaily,
                kp: [-1; 8],
                kp_sum: -1,
                ap: [a; 8],
                ap_avg: a,
                cp,
                c9,
                isn: -1,
                f10p7_obs: flux,
                f10p7_adj: flux,
                f10p7_obs_c81: -1.0,
                f10p7_obs_l81: -1.0,
                f10p7_adj_c81: -1.0,
                f10p7_adj_l81: -1.0,
            })
        })
        .collect();
    if out.is_empty() {
        return Err(Error::InvalidEntry);
    }
    Ok(out)
}

/// Check that the file at `path` parses as a 45-day forecast.
pub(crate) fn validate_file(path: &std::path::Path) -> std::result::Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse(&text)
        .map(|_| ())
        .map_err(|e| format!("not a parsable SWPC 45-day forecast ({e})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
:Product: 45 Day AP and F10.7cm Flux Forecast  45-day-forecast.txt
:Issued: 2026 Sep 24 0000 UTC
# Prepared by Dept. of Commerce, NOAA, Space Weather Prediction Center.
#
45-DAY AP FORECAST
24Sep26 024 25Sep26 016 26Sep26 012 27Sep26 008 28Sep26 012
29Sep26 008
45-DAY F10.7 CM FLUX FORECAST
24Sep26 120 25Sep26 120 26Sep26 118 27Sep26 114 28Sep26 110
29Sep26 108
FORECASTER:  AUTOMATED - SWPC Forecasting System
99999
";

    #[test]
    fn test_parse_joins_sections_by_date() {
        let r = parse(SAMPLE).unwrap();
        assert_eq!(r.len(), 6);
        assert_eq!(r[0].date, Instant::from_date(2026, 9, 24).unwrap());
        assert_eq!(r[0].ap_avg, 24);
        assert_eq!(r[0].ap, [24; 8]);
        assert_eq!(r[0].f10p7_obs, 120.0);
        assert_eq!(r[0].data_type, SpaceWeatherDataType::PredictedDaily);
        assert!(r[0].has_geomagnetic());
        assert_eq!(r[5].date, Instant::from_date(2026, 9, 29).unwrap());
        assert_eq!(r[5].ap_avg, 8);
        assert_eq!(r[5].f10p7_obs, 108.0);
    }

    #[test]
    fn test_date_format() {
        assert_eq!(parse_date("24Sep26"), Instant::from_date(2026, 9, 24).ok());
        assert_eq!(parse_date("01Jan27"), Instant::from_date(2027, 1, 1).ok());
        assert_eq!(parse_date("bogus"), None);
    }
}
