//! NASA MSFC Marshall Solar Activity Future Estimation (MSAFE) — the monthly
//! forecast.
//!
//! The long-range source: monthly 13-month-smoothed F10.7 **and Ap** for the
//! balance of the current solar cycle plus a mean cycle beyond it, each with
//! 95 / 50 / 5 percentile bands. US Government work, public domain. This is
//! what Orekit consumes through `MarshallSolarActivityFutureEstimation`, and
//! what CelesTrak's monthly rows lack: they carry F10.7 only, which is why
//! satkit used to fall back to a quiet-time `Ap = 4` past the daily data.
//!
//! The record fed to NRLMSISE-00 uses the 50 % (median) band. The full
//! forecast, with both bands, is kept on [`Forecast`] for lifetime studies
//! that want to bracket rather than point-estimate.
//!
//! Published at
//! <https://www.nasa.gov/solar-cycle-progression-and-forecast/archived-forecast/>.

use super::{Error, Result, SpaceWeatherDataType, SpaceWeatherRecord};
use crate::utils::download::{self, RefreshOutcome};
use crate::Instant;
use std::path::Path;

/// One month of the MSAFE forecast, with its percentile bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsafeMonth {
    /// First day of the month.
    pub date: Instant,
    /// F10.7 at the 95th / 50th / 5th percentile, sfu.
    pub f107: [f64; 3],
    /// Daily Ap at the 95th / 50th / 5th percentile.
    pub ap: [f64; 3],
}

/// The parsed MSAFE table.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    pub months: Vec<MsafeMonth>,
}

pub(super) fn month_number(name: &str) -> Option<i32> {
    Some(match name.to_ascii_uppercase().as_str() {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    })
}

/// Parse an MSAFE `*f10-prd.txt` (or older `*f10.txt`) table.
///
/// Data rows look like
/// ` 2026.5837   AUG   138.2     126.1     113.9      19.8      15.0      10.1`
/// — decimal year, month name, then F10.7 and Ap at 95 / 50 / 5 %. Header and
/// caption lines are skipped by shape.
pub fn parse(text: &str) -> Result<Forecast> {
    let mut months = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 8 {
            continue;
        }
        let Ok(decimal_year) = f[0].parse::<f64>() else {
            continue;
        };
        let Some(month) = month_number(f[1]) else {
            continue;
        };
        let num = |s: &str| -> Result<f64> {
            s.parse::<f64>()
                .map_err(|_| Error::InvalidNumber("msafe value"))
        };
        // The decimal year is mid-month-ish; the integer part is the year
        // (Dec 2026 = 2026.917, Jan 2027 = 2027.000).
        let year = decimal_year.floor() as i32;
        months.push(MsafeMonth {
            date: Instant::from_date(year, month, 1)?,
            f107: [num(f[2])?, num(f[3])?, num(f[4])?],
            ap: [num(f[5])?, num(f[6])?, num(f[7])?],
        });
    }
    if months.is_empty() {
        return Err(Error::InvalidEntry);
    }
    months.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
    Ok(Forecast { months })
}

impl Forecast {
    /// Monthly records for the space-weather table, from the 50 % band.
    ///
    /// Ap fills all eight 3-hourly slots (a smoothed climatology has no
    /// intra-day structure), Kp is not provided and stays `-1`, Cp/C9 come
    /// from the Bartels table on `8 × Ap`, and F10.7 is used for both the
    /// observed and 1 AU-adjusted fields — the ±3 % seasonal adjustment is
    /// below the resolution of a 13-month-smoothed climatology, and
    /// NRLMSISE-00 reads the observed value.
    pub fn records(&self) -> Vec<SpaceWeatherRecord> {
        self.months
            .iter()
            .map(|m| {
                SpaceWeatherRecord::forecast(
                    m.date,
                    SpaceWeatherDataType::PredictedMonthly,
                    m.ap[1].round() as i32,
                    m.f107[1],
                )
            })
            .collect()
    }
}

/// Where NASA publishes the monthly forecasts.
#[cfg(feature = "download")]
pub(crate) const NASA_UPLOADS: &str = "https://www.nasa.gov/wp-content/uploads/";

/// The URL, under `base` ([`NASA_UPLOADS`] in production), of the forecast
/// issued in a given month.
#[cfg(feature = "download")]
pub(crate) fn nasa_url(base: &str, year: i32, month: i32) -> String {
    const MON: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    format!(
        "{base}{year}/{month:02}/{}{year}f10-prd.txt",
        MON[(month - 1) as usize]
    )
}

/// Bring the MSAFE file in `dir` up to date, as [`MSAFE_FILE`](super::MSAFE_FILE).
///
/// NASA publishes one file per month under a month-specific name with no
/// stable "latest" URL, so this walks back from the current month until a
/// file answers (six months is more than the series has ever gone without
/// an issue). The copy on disk is kept under one stable name; its
/// `.http-cache` sidecar carries the age gate, so a copy checked within the
/// last week is reported current with no request.
///
/// Without the `download` feature this is [`Error::FeatureDisabled`](download::Error::FeatureDisabled).
#[cfg(feature = "download")]
pub fn refresh_into(dir: &Path, force: bool) -> download::Result<RefreshOutcome> {
    refresh_from(NASA_UPLOADS, dir, force).map(|(outcome, _)| outcome)
}

/// [`refresh_into`] from the monthly files under `base` ([`NASA_UPLOADS`]
/// in production), also returning the URL of the file downloaded (`None`
/// when the copy on disk was current and no request was made).
#[cfg(feature = "download")]
pub(crate) fn refresh_from(
    base: &str,
    dir: &Path,
    force: bool,
) -> download::Result<(RefreshOutcome, Option<String>)> {
    use download::{read_refresh_marker, write_refresh_marker};
    let dest = dir.join(super::MSAFE_FILE);
    if !force && dest.is_file() {
        if let Some((checked_at, _)) = read_refresh_marker(&dest) {
            let age = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
                .saturating_sub(checked_at);
            if age < download::refresh_min_age_secs(super::MSAFE_FILE) {
                return Ok((RefreshOutcome::Fresh { age_secs: age }, None));
            }
        }
    }
    download::check_online(super::MSAFE_FILE)?;
    let (mut y, mut m, ..) = Instant::now().as_datetime();
    let mut attempts = Vec::new();
    for _ in 0..6 {
        let url = nasa_url(base, y, m);
        match download::download_to_string(&url) {
            Ok(text) => {
                // Validate before it replaces a good file.
                if let Err(e) = parse(&text) {
                    attempts.push(format!("{url}: {e}"));
                } else {
                    download::write_atomic(&mut std::io::Cursor::new(text), &dest, &url)?;
                    write_refresh_marker(&dest, None);
                    return Ok((RefreshOutcome::Downloaded, Some(url)));
                }
            }
            Err(e) => attempts.push(format!("{url}: {e}")),
        }
        if m == 1 {
            y -= 1;
            m = 12;
        } else {
            m -= 1;
        }
    }
    Err(download::Error::AllSourcesFailed {
        name: super::MSAFE_FILE.to_string(),
        attempts,
        hint: Some(
            "NASA publishes the MSAFE forecast monthly at \
             https://www.nasa.gov/solar-cycle-progression-and-forecast/archived-forecast/"
                .to_string(),
        ),
    })
}

#[cfg(not(feature = "download"))]
pub fn refresh_into(_dir: &Path, _force: bool) -> download::Result<RefreshOutcome> {
    Err(download::Error::FeatureDisabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
  TABLE 3 ESTIMATES OF 13-MONTH SMOOTH SOLAR ACTIVITY FOR
  BALANCE OF CYCLE 25 WITH A MEAN CYCLE GIVEN FOR CYCLE 26

    TIME         10.7 CM SOLAR FLUX   (F10.7)      GEOMAGNETIC INDEX   (Ap)
                         PERCENTILE                    PERCENTILE
                  95.0%       50%      5.0%     95.0%       50%      5.0% 

 2026.9170   DEC   135.1     117.3     105.8      21.7      15.7      11.7
 2027.0003   JAN   132.1     115.5     103.0      22.3      16.0      11.4
 2026.5837   AUG   138.2     126.1     113.9      19.8      15.0      10.1
";

    #[test]
    fn test_parse_and_sort() {
        let f = parse(SAMPLE).unwrap();
        assert_eq!(f.months.len(), 3);
        // sorted, and the year boundary handled by the integer part
        assert_eq!(f.months[0].date, Instant::from_date(2026, 8, 1).unwrap());
        assert_eq!(f.months[1].date, Instant::from_date(2026, 12, 1).unwrap());
        assert_eq!(f.months[2].date, Instant::from_date(2027, 1, 1).unwrap());
        assert_eq!(f.months[0].f107, [138.2, 126.1, 113.9]);
        assert_eq!(f.months[0].ap, [19.8, 15.0, 10.1]);
    }

    #[test]
    fn test_records_carry_ap() {
        // The whole point: monthly rows with geomagnetic data.
        let r = parse(SAMPLE).unwrap().records();
        assert_eq!(r[0].data_type, SpaceWeatherDataType::PredictedMonthly);
        assert_eq!(r[0].ap_avg, 15);
        assert_eq!(r[0].ap, [15; 8]);
        assert!(r[0].has_geomagnetic());
        assert_eq!(r[0].f10p7_obs, 126.1);
        assert_eq!(r[0].kp_sum, -1);
    }

    #[test]
    fn test_empty_is_error() {
        assert!(parse("no data here\n").is_err());
    }
}
