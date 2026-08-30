use std::cmp::Ordering;
use std::path::PathBuf;

use crate::utils::{datadir, download_file, download_if_not_exist, RefreshableSingleton};
use crate::Instant;
use crate::TimeLike;
use thiserror::Error;

/// Errors produced by the [`spaceweather`](crate::spaceweather) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A field in the CSV space-weather record could not be parsed as the
    /// expected numeric type.
    #[error("Invalid number in file: {0}")]
    InvalidNumber(&'static str),

    /// A line in the space-weather CSV has fewer than the expected number of
    /// comma-separated fields (a truncated or corrupt file).
    #[error("Invalid entry in space weather file: too few fields")]
    InvalidEntry,

    /// No space-weather record exists for the requested time.
    #[error("No space weather record found for date")]
    NoRecordForDate,

    /// The configured data directory is read-only and cannot receive an
    /// updated space-weather file.
    #[error(
        "Data directory is read-only. Try setting the environment variable SATKIT_DATA \
         to a writeable directory and re-starting or explicitly set data directory to \
         a writeable directory"
    )]
    DataDirReadOnly,

    /// Bytes passed to [`init_from_bytes`] were not valid UTF-8 — the
    /// space-weather file is a CSV text format.
    #[error("space-weather byte buffer is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the `spaceweather` module.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct SpaceWeatherRecord {
    /// Date of record
    pub date: Instant,
    /// Bartels Solar Radiation Number.
    /// A sequence of 27-day intervals counted continuously from 1832 February 8
    pub bsrn: i32,
    /// Number of day within the bsrn
    pub nd: i32,
    /// Kp
    pub kp: [i32; 8],
    pub kp_sum: i32,
    pub ap: [i32; 8],
    pub ap_avg: i32,
    /// Planetary daily character figure
    pub cp: f64,
    /// Scale cp to \[0, 9\]
    pub c9: i32,
    /// International Sunspot Number
    pub isn: i32,
    pub f10p7_obs: f64,
    pub f10p7_adj: f64,
    pub f10p7_obs_c81: f64,
    pub f10p7_obs_l81: f64,
    pub f10p7_adj_c81: f64,
    pub f10p7_adj_l81: f64,
}

fn str2num<T: core::str::FromStr>(
    s: &str,
    sidx: usize,
    eidx: usize,
    field: &'static str,
) -> Result<T> {
    s.chars()
        .skip(sidx)
        .take(eidx - sidx)
        .collect::<String>()
        .trim()
        .parse()
        .map_err(|_| Error::InvalidNumber(field))
}

impl PartialEq for SpaceWeatherRecord {
    fn eq(&self, other: &Self) -> bool {
        self.date == other.date
    }
}

impl PartialOrd for SpaceWeatherRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.date.partial_cmp(&other.date)
    }
}

impl PartialEq<Instant> for SpaceWeatherRecord {
    fn eq(&self, other: &Instant) -> bool {
        self.date == *other
    }
}

impl PartialOrd<Instant> for SpaceWeatherRecord {
    fn partial_cmp(&self, other: &Instant) -> Option<Ordering> {
        self.date.partial_cmp(other)
    }
}

/// Parse a `SW-All.csv` text buffer into space-weather records.
fn parse_csv(text: &str) -> Result<Vec<SpaceWeatherRecord>> {
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| -> Result<SpaceWeatherRecord> {
            let lvals: Vec<&str> = line.split(",").collect();
            // The record reads fixed column indices up to 30; bail on a
            // truncated line rather than panicking on out-of-bounds indexing.
            if lvals.len() < 31 {
                return Err(Error::InvalidEntry);
            }

            let year: u32 = str2num(lvals[0], 0, 4, "year")?;
            let mon: u32 = str2num(lvals[0], 5, 7, "month")?;
            let day: u32 = str2num(lvals[0], 8, 10, "day of month")?;

            Ok(SpaceWeatherRecord {
                date: (Instant::from_date(year as i32, mon as i32, day as i32)?),
                bsrn: lvals[1].parse().unwrap_or(-1),
                nd: lvals[2].parse().unwrap_or(-1),
                kp: {
                    let mut kparr: [i32; 8] = [-1, -1, -1, -1, -1, -1, -1, -1];
                    for idx in 0..8 {
                        kparr[idx] = lvals[idx + 3].parse().unwrap_or(-1);
                    }
                    kparr
                },
                kp_sum: lvals[11].parse().unwrap_or(-1),
                ap: {
                    let mut aparr: [i32; 8] = [-1, -1, -1, -1, -1, -1, -1, -1];
                    for idx in 0..8 {
                        aparr[idx] = lvals[12 + idx].parse().unwrap_or(-1)
                    }
                    aparr
                },
                ap_avg: lvals[20].parse().unwrap_or(-1),
                cp: lvals[21].parse().unwrap_or(-1.0),
                c9: lvals[22].parse().unwrap_or(-1),
                isn: lvals[23].parse().unwrap_or(-1),
                f10p7_obs: lvals[24].parse().unwrap_or(-1.0),
                f10p7_adj: lvals[25].parse().unwrap_or(-1.0),
                f10p7_obs_c81: lvals[27].parse().unwrap_or(-1.0),
                f10p7_obs_l81: lvals[28].parse().unwrap_or(-1.0),
                f10p7_adj_c81: lvals[29].parse().unwrap_or(-1.0),
                f10p7_adj_l81: lvals[30].parse().unwrap_or(-1.0),
            })
        })
        .collect()
}

fn load_default_path() -> Result<PathBuf> {
    // Found in any search directory, else downloaded into the write location.
    Ok(crate::utils::datadir::path_for("SW-All.csv")?)
}

/// Lazy default load from `SW-All.csv` under [`datadir`], with auto-download.
fn load_space_weather_csv() -> Result<Vec<SpaceWeatherRecord>> {
    let path = load_default_path()?;
    download_if_not_exist(&path, Some("https://celestrak.org/SpaceData/"))?;
    parse_csv(&std::fs::read_to_string(&path)?)
}

/// Module-scope refreshable singleton. The lazy default load (best-effort,
/// silent on failure) runs at most once; [`init_from_bytes`] /
/// [`init_from_path`] / [`update`] replace any current contents.
static SPACE_WEATHER: RefreshableSingleton<Vec<SpaceWeatherRecord>> = RefreshableSingleton::new();

/// Initialize the space-weather singleton from an in-memory byte buffer.
///
/// The bytes must be a valid `SW-All.csv` text file (UTF-8). Unlike the
/// static-data subsystems, this *always* succeeds and replaces any
/// previously loaded data — space-weather records update daily and the
/// refresh-in-place semantics are intentional.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    SPACE_WEATHER.set(parse_csv(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the space-weather singleton from a file at `path`.
///
/// Same semantics as [`init_from_bytes`] but reads the file from disk.
/// Always replaces any previously loaded data.
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    SPACE_WEATHER.set(parse_csv(&std::fs::read_to_string(path)?)?);
    Ok(())
}

/// Best-effort default load on first read. The full load (which may attempt a
/// download) runs at most once — previously it re-attempted a blocking load on
/// *every* `get`, which for a drag propagation with no cached file meant an
/// HTTP attempt per ODE step. If that first attempt failed, later calls retry
/// **from disk only** when the file has since appeared (a cheap stat, no
/// network), so a process that started before the data directory was populated
/// recovers once `update_datafiles` (or anything else) writes the file.
fn ensure_default_loaded() {
    SPACE_WEATHER.ensure_default_loaded(|| load_space_weather_csv().ok());
    if SPACE_WEATHER.read().is_none() {
        let Ok(path) = load_default_path() else {
            return;
        };
        if path.is_file() {
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(records) = parse_csv(&text) {
                    if !records.is_empty() {
                        SPACE_WEATHER.set(records);
                    }
                }
            }
        }
    }
}

///
/// Return full Space Weather record from Space Weather file,
/// as a function of requested instant in time.
///
/// Returns the record for the same day when present, otherwise the most
/// recent prior record (no interpolation). For dates beyond the last record
/// the final record is returned; note that predicted rows may carry `-1`
/// sentinel values in fields celestrak has not filled in.
///
/// # Arguments
///
/// * `tm` - time instant at which to retrieve space weather record
///
/// # Returns
///
/// * Full space weather record
///
/// # Notes:
///
/// * Space weather is updated daily in a file: SW-All.csv
pub fn get<T: TimeLike>(tm: &T) -> Result<SpaceWeatherRecord> {
    let tm = tm.as_instant();
    ensure_default_loaded();
    let guard = SPACE_WEATHER.read();
    let sw = guard.as_ref().ok_or(Error::NoRecordForDate)?;
    // Guard empty data (e.g. a header-only CSV) so the indexing below can't
    // panic; treat it the same as "not loaded".
    let first = sw.first().ok_or(Error::NoRecordForDate)?;

    // First, try simple indexing
    let idx = (tm - first.date).as_days().floor() as usize;
    if idx < sw.len() && (tm - sw[idx].date).as_days().abs() < 1.0 {
        return Ok(sw[idx].clone());
    }

    sw.iter()
        .rev()
        .find(|x| x.date <= tm)
        .cloned()
        .ok_or(Error::NoRecordForDate)
}

/// Download new Space Weather file, and load it.
pub fn update() -> Result<()> {
    // Get data directory
    let d = datadir()?;
    if d.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    // Download most-recent SW file. This must be the same file the loader
    // parses (`SW-All.csv`); downloading `sw19571001.txt` here left the loader
    // reading stale data and made this a silent no-op.
    let url = "https://celestrak.org/SpaceData/SW-All.csv";
    download_file(url, &d, true)?;

    SPACE_WEATHER.set(load_space_weather_csv()?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let tm: Instant = Instant::from_datetime(2023, 11, 14, 0, 0, 0.0).unwrap();
        let r = get(&tm);
        println!("r = {:?}", r);
        println!("rdate = {}", r.unwrap().date);
    }

    #[test]
    fn test_parse_truncated_line_errors_not_panics() {
        // A truncated data line (fewer than the required fields) must return a
        // clean error rather than panicking on out-of-bounds indexing.
        let csv = "HEADER\n2023-11-14,1,2,3\n";
        assert!(matches!(parse_csv(csv), Err(Error::InvalidEntry)));
    }

    #[test]
    fn test_parse_skips_blank_trailing_line() {
        // A trailing blank line should be skipped, not treated as a truncated
        // record. (Build one full 31-field row.)
        let mut row = String::from("2023-11-14");
        for _ in 0..30 {
            row.push_str(",0");
        }
        let csv = format!("HEADER\n{row}\n\n");
        let recs = parse_csv(&csv).unwrap();
        assert_eq!(recs.len(), 1);
    }
}
