pub mod assemble;
pub mod cssi;
pub mod gfz;
pub mod msafe;
pub mod swpc;

use std::cmp::Ordering;
use std::path::PathBuf;

use crate::utils::{datadir, download_if_not_exist, RefreshableSingleton};
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

/// Provenance of a space-weather row, from the `F10.7_DATA_TYPE` column of
/// `SW-All.csv` (the one-digit `Q` field in the fixed-width `.txt` twin).
///
/// The distinction matters for drag: only [`Observed`](Self::Observed) rows
/// are measurements, and [`PredictedMonthly`](Self::PredictedMonthly) rows
/// carry no geomagnetic data at all — every `kp`/`ap` field is the `-1`
/// sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceWeatherDataType {
    /// Measured and definitive (`OBS` in `SW-All.csv`; GFZ `D = 2`).
    Observed,
    /// Measured but still preliminary — GFZ's `D = 0`, replaced by a
    /// definitive value once it is available. `SW-All.csv` has no equivalent.
    ObservedPreliminary,
    /// `INT` — interpolated across a gap in the measured record.
    Interpolated,
    /// `PRD` — daily prediction (NOAA/SWPC 45-day forecast). Kp/ap present.
    PredictedDaily,
    /// `PRM` — monthly prediction. F10.7 only; Kp/ap are all `-1`.
    PredictedMonthly,
    /// The column was empty or held an unrecognised value.
    Unknown,
}

impl SpaceWeatherDataType {
    /// Parse the `F10.7_DATA_TYPE` column.
    fn parse(s: &str) -> Self {
        match s.trim() {
            "OBS" => Self::Observed,
            "INT" => Self::Interpolated,
            "PRD" => Self::PredictedDaily,
            "PRM" => Self::PredictedMonthly,
            _ => Self::Unknown,
        }
    }

    /// Whether the row is a measurement (`OBS` or `INT`) rather than a
    /// prediction.
    pub fn is_observed(&self) -> bool {
        matches!(
            self,
            Self::Observed | Self::ObservedPreliminary | Self::Interpolated
        )
    }

    /// Whether the row is at daily cadence — measured, interpolated or the
    /// daily forecast — as opposed to a monthly row.
    pub fn is_daily(&self) -> bool {
        !matches!(self, Self::PredictedMonthly | Self::Unknown)
    }

    /// The column text this variant was parsed from.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Observed => "OBS",
            Self::ObservedPreliminary => "OBS-P",
            Self::Interpolated => "INT",
            Self::PredictedDaily => "PRD",
            Self::PredictedMonthly => "PRM",
            Self::Unknown => "",
        }
    }
}

/// Where an epoch falls relative to the loaded space-weather table.
///
/// Returned by [`status`]; see [`coverage`] for the table bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceWeatherStatus {
    /// On or before the last measured row.
    Observed,
    /// After the last measured row, inside the daily predictions: F10.7 and
    /// the geomagnetic indices are the NOAA/SWPC 45-day forecast.
    PredictedDaily,
    /// Past the daily predictions. Only monthly F10.7 is available; the
    /// record returned is the most recent month's, and it carries **no**
    /// geomagnetic data, so NRLMSISE-00 falls back to a quiet-time
    /// `Ap = 4`. Density can be wrong by a factor of two during a storm.
    PredictedMonthly,
    /// After the last row of the table: that row's values are returned
    /// unchanged.
    Extrapolated,
    /// Before the first row of the table (1957 for `SW-All.csv`).
    BeforeTable,
    /// No space-weather table is loaded at all.
    NotLoaded,
}

/// Time bounds of the loaded space-weather table (UTC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeatherCoverage {
    /// Epoch of the first row.
    pub first: Instant,
    /// Epoch of the last measured row (`OBS`/`INT`).
    pub last_observed: Instant,
    /// Epoch of the last row at daily cadence (`OBS`/`INT`/`PRD`). After
    /// this, only monthly rows remain and no geomagnetic data is available.
    pub last_daily: Instant,
    /// Epoch of the last row of any kind.
    pub last: Instant,
}

#[derive(Debug, Clone)]
pub struct SpaceWeatherRecord {
    /// Date of record
    pub date: Instant,
    /// Bartels Solar Radiation Number.
    /// A sequence of 27-day intervals counted continuously from 1832 February 8
    pub bsrn: i32,
    /// Number of day within the bsrn
    pub nd: i32,
    /// 3-hourly planetary Kp (×10 as tabulated by CelesTrak, e.g. 33 = 3+),
    /// one per UT interval: `[0]` 00–03 UT, …, `[7]` 21–24 UT. `-1` when the
    /// field is empty (e.g. monthly predicted rows).
    pub kp: [i32; 8],
    pub kp_sum: i32,
    /// 3-hourly planetary ap (`AP1..AP8`), one per UT interval: `[0]` covers
    /// 00:00–03:00 UT, `[1]` 03:00–06:00, …, `[7]` 21:00–24:00. `-1` when
    /// the field is empty (monthly predicted rows). Daily predicted rows
    /// carry the same value in all eight slots.
    pub ap: [i32; 8],
    /// Daily Ap (`AP_AVG`, the mean of the eight 3-hourly values); `-1` when
    /// empty.
    pub ap_avg: i32,
    /// Planetary daily character figure
    pub cp: f64,
    /// Scale cp to \[0, 9\]
    pub c9: i32,
    /// Provenance of this row — measured, interpolated, or predicted.
    /// See [`SpaceWeatherDataType`].
    pub data_type: SpaceWeatherDataType,
    /// International Sunspot Number
    pub isn: i32,
    pub f10p7_obs: f64,
    pub f10p7_adj: f64,
    pub f10p7_obs_c81: f64,
    pub f10p7_obs_l81: f64,
    pub f10p7_adj_c81: f64,
    pub f10p7_adj_l81: f64,
}

impl SpaceWeatherRecord {
    /// Whether this row carries geomagnetic data NRLMSISE-00 can use.
    ///
    /// False when the daily Ap is the `-1` sentinel — CelesTrak's monthly
    /// predicted rows — in which case the model falls back to a quiet-time
    /// `Ap = 4`. MSAFE monthly rows carry a climatological Ap and return
    /// true.
    pub fn has_geomagnetic(&self) -> bool {
        self.ap_avg >= 0
    }
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

fn load_default_path() -> Result<PathBuf> {
    // Found in any search directory, else downloaded into the write location.
    Ok(crate::utils::datadir::path_for("SW-All.csv")?)
}

/// Lazy default load from `SW-All.csv` under [`datadir`], with auto-download.
fn load_space_weather_csv() -> Result<Vec<SpaceWeatherRecord>> {
    let path = load_default_path()?;
    download_if_not_exist(&path, Some("https://celestrak.org/SpaceData/"))?;
    cssi::parse_csv(&std::fs::read_to_string(&path)?)
}

/// UTC day number of the last day of the month containing `date`.
pub(crate) fn month_end_day(date: Instant) -> i64 {
    let (y, m, _, _, _, _) = date.as_datetime();
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    Instant::from_date(ny, nm, 1).unwrap().utc_day_number() - 1
}

/// Module-scope refreshable singleton. The lazy default load (best-effort,
/// silent on failure) runs at most once; [`init_from_bytes`] /
/// [`init_from_path`] / [`update`] replace any current contents.
static SPACE_WEATHER: RefreshableSingleton<Vec<SpaceWeatherRecord>> = RefreshableSingleton::new();

/// One-time warning latches; see [`disable_space_weather_time_warning`].
static MONTHLY_WARNING_SHOWN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static EXTRAP_WARNING_SHOWN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static NOT_LOADED_WARNING_SHOWN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Initialize the space-weather singleton from an in-memory byte buffer.
///
/// The bytes must be a valid `SW-All.csv` text file (UTF-8). Unlike the
/// static-data subsystems, this *always* succeeds and replaces any
/// previously loaded data — space-weather records update daily and the
/// refresh-in-place semantics are intentional.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    SPACE_WEATHER.set(cssi::parse_csv(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the space-weather singleton from a file at `path`.
///
/// Same semantics as [`init_from_bytes`] but reads the file from disk.
/// Always replaces any previously loaded data.
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    SPACE_WEATHER.set(cssi::parse_csv(&std::fs::read_to_string(path)?)?);
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
                if let Ok(records) = cssi::parse_csv(&text) {
                    if !records.is_empty() {
                        SPACE_WEATHER.set(records);
                    }
                }
            }
        }
    }
}

/// Disable the warnings about out-of-range or missing space-weather data.
///
/// Three one-time warnings exist: an epoch past the daily predictions (only
/// monthly F10.7, no geomagnetic data), an epoch past the end of the table,
/// and no table loaded at all. Each is shown at most once per process; call
/// this to suppress all of them.
///
/// # Example
///
/// ```rust
/// satkit::spaceweather::disable_space_weather_time_warning();
/// ```
pub fn disable_space_weather_time_warning() {
    use std::sync::atomic::Ordering;
    MONTHLY_WARNING_SHOWN.store(true, Ordering::Relaxed);
    EXTRAP_WARNING_SHOWN.store(true, Ordering::Relaxed);
    NOT_LOADED_WARNING_SHOWN.store(true, Ordering::Relaxed);
}

/// Time bounds of the loaded space-weather table, or `None` if no table is
/// loaded (file missing and download failed, or an empty table installed).
///
/// `last_daily` is the boundary that matters for drag: past it the table
/// holds only monthly rows, which carry no geomagnetic data at all.
///
/// # Example
///
/// ```rust
/// if let Some(c) = satkit::spaceweather::coverage() {
///     println!("observed through {}, daily through {}", c.last_observed, c.last_daily);
/// }
/// ```
pub fn coverage() -> Option<SpaceWeatherCoverage> {
    ensure_default_loaded();
    let guard = SPACE_WEATHER.read();
    coverage_of(guard.as_ref()?)
}

/// Pure core of [`coverage`], over a table slice.
fn coverage_of(sw: &[SpaceWeatherRecord]) -> Option<SpaceWeatherCoverage> {
    let first = sw.first()?;
    let last = sw.last()?;
    let last_observed = sw
        .iter()
        .rev()
        .find(|r| r.data_type.is_observed())
        .unwrap_or(first);
    let last_daily = sw
        .iter()
        .rev()
        .find(|r| r.data_type.is_daily())
        .unwrap_or(last_observed);
    Some(SpaceWeatherCoverage {
        first: first.date,
        last_observed: last_observed.date,
        last_daily: last_daily.date,
        last: last.date,
    })
}

/// Classify an epoch against the loaded space-weather table — see
/// [`SpaceWeatherStatus`].
///
/// Useful before a long drag propagation: a result of
/// [`SpaceWeatherStatus::PredictedMonthly`] means NRLMSISE-00 is running on
/// a quiet-time `Ap = 4` with no storm information, and
/// [`SpaceWeatherStatus::Extrapolated`] means the file should be refreshed.
pub fn status<T: TimeLike>(tm: &T) -> SpaceWeatherStatus {
    let tm = tm.as_instant();
    let Some(c) = coverage() else {
        return SpaceWeatherStatus::NotLoaded;
    };
    status_in(&c, tm)
}

/// Pure core of [`status`], against known table bounds.
fn status_in(c: &SpaceWeatherCoverage, tm: Instant) -> SpaceWeatherStatus {
    let day = tm.utc_day_number();
    if day < c.first.utc_day_number() {
        return SpaceWeatherStatus::BeforeTable;
    }
    if day > c.last.utc_day_number() {
        return SpaceWeatherStatus::Extrapolated;
    }
    if day <= c.last_observed.utc_day_number() {
        SpaceWeatherStatus::Observed
    } else if day <= c.last_daily.utc_day_number() {
        SpaceWeatherStatus::PredictedDaily
    } else {
        SpaceWeatherStatus::PredictedMonthly
    }
}

///
/// Return full Space Weather record from Space Weather file,
/// as a function of requested instant in time.
///
/// Returns the record for the same day when present, otherwise the most
/// recent prior record (no interpolation). For dates beyond the last record
/// the final record is returned. Predicted rows may carry `-1`
/// sentinel values in fields CelesTrak has not filled in; monthly
/// predicted rows carry no geomagnetic data at all. See [`status`] and
/// [`coverage`] to classify an epoch before relying on the result.
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
    use std::sync::atomic::Ordering;
    let tm = tm.as_instant();
    let mut guard = SPACE_WEATHER.read();
    if guard.is_none() {
        drop(guard);
        ensure_default_loaded();
        guard = SPACE_WEATHER.read();
    }
    let Some(sw) = guard.as_ref().filter(|s| !s.is_empty()) else {
        if !NOT_LOADED_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: no space-weather table is loaded; NRLMSISE-00 is running on its \
                 defaults (F10.7 = F10.7A = 150, Ap = 4), which can be wrong by a factor of \
                 two in atmospheric density.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download SW-All.csv, or set SATKIT_DATA to a directory containing it.\n\
                 To disable: `satkit::spaceweather::disable_space_weather_time_warning()` \
                 (Python: `satkit.spaceweather.disable_space_weather_time_warning()`)"
            );
        }
        return Err(Error::NoRecordForDate);
    };
    // Guard empty data (e.g. a header-only CSV) so the indexing below can't
    // panic; treat it the same as "not loaded".
    let first = sw.first().ok_or(Error::NoRecordForDate)?;
    let last = sw.last().ok_or(Error::NoRecordForDate)?;

    // Past the final row every query returns that row unchanged.
    if tm.utc_day_number() > last.date.utc_day_number()
        && !EXTRAP_WARNING_SHOWN.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "Warning: the space-weather table ends at {}; the request for {tm} and all later \
             epochs return that row's values unchanged.\n\
             Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
             to download the most recent space-weather file.\n\
             To disable: `satkit::spaceweather::disable_space_weather_time_warning()` \
                 (Python: `satkit.spaceweather.disable_space_weather_time_warning()`)",
            last.date
        );
    }

    // Index by UTC calendar day. Instants count leap seconds, so a
    // continuous-day index lands on the next record in the last seconds of a
    // day.
    let day = tm.utc_day_number();
    let first_day = first.date.utc_day_number();
    let found = if day >= first_day
        && ((day - first_day) as usize) < sw.len()
        && sw[(day - first_day) as usize].date.utc_day_number() == day
    {
        Some(&sw[(day - first_day) as usize])
    } else {
        sw.iter().rev().find(|x| x.date <= tm)
    };
    let rec = found.ok_or(Error::NoRecordForDate)?;

    // A monthly-predicted row carries no geomagnetic data at all: every
    // kp/ap field is -1, so NRLMSISE-00 falls back to a quiet-time Ap = 4.
    if !rec.has_geomagnetic() && !MONTHLY_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
        eprintln!(
            "Warning: the space-weather record for {tm} is a monthly prediction ({}); it \
             carries F10.7 but no Kp/ap, so NRLMSISE-00 runs on a quiet-time Ap = 4 with no \
             storm information. Density can be wrong by a factor of two during a geomagnetic \
             storm (measured 1.9x at 400 km for the 2024-05-11 event).\n\
             Daily data ends shortly after the last observed day; see \
             `satkit::spaceweather::coverage()` / `status()`.\n\
             To disable: `satkit::spaceweather::disable_space_weather_time_warning()` \
                 (Python: `satkit.spaceweather.disable_space_weather_time_warning()`)",
            rec.date
        );
    }
    Ok(rec.clone())
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
    //
    // CelesTrak publishes space weather every 3 hours and asks clients to
    // download it once per update, so a copy younger than that is reused
    // without contacting the server and an unchanged one costs a `304`.
    let url = "https://celestrak.org/SpaceData/SW-All.csv";
    crate::utils::refresh_file(url, &d, false)?;

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
    fn test_get_uses_utc_day() {
        let noon = Instant::from_datetime(2023, 11, 14, 12, 0, 0.0).unwrap();
        let late = Instant::from_datetime(2023, 11, 14, 23, 59, 50.0).unwrap();
        assert_eq!(get(&noon).unwrap().date, get(&late).unwrap().date);
    }

    #[test]
    fn test_data_type_parsed_from_column_27() {
        // One full row per data type; the column is index 26 (0-based).
        let mut rows = String::from("HEADER\n");
        for (day, ty) in [(1, "OBS"), (2, "INT"), (3, "PRD"), (4, "PRM"), (5, "")] {
            let mut f: Vec<String> = vec![format!("2023-11-0{day}")];
            for i in 1..31 {
                f.push(if i == 26 {
                    ty.to_string()
                } else {
                    "0".to_string()
                });
            }
            rows.push_str(&f.join(","));
            rows.push('\n');
        }
        let recs = cssi::parse_csv(&rows).unwrap();
        let got: Vec<SpaceWeatherDataType> = recs.iter().map(|r| r.data_type).collect();
        assert_eq!(
            got,
            vec![
                SpaceWeatherDataType::Observed,
                SpaceWeatherDataType::Interpolated,
                SpaceWeatherDataType::PredictedDaily,
                SpaceWeatherDataType::PredictedMonthly,
                SpaceWeatherDataType::Unknown,
            ]
        );
        assert!(recs[0].data_type.is_observed());
        assert!(recs[1].data_type.is_observed());
        assert!(!recs[2].data_type.is_observed());
        // Only the monthly rows are off daily cadence.
        assert!(recs[2].data_type.is_daily());
        assert!(!recs[3].data_type.is_daily());
    }

    /// Build a synthetic table: OBS days 1-3, PRD day 4, PRM day 5.
    fn synthetic_table() -> String {
        let mut rows = String::from("HEADER\n");
        for (day, ty) in [(1, "OBS"), (2, "OBS"), (3, "OBS"), (4, "PRD"), (5, "PRM")] {
            let mut f: Vec<String> = vec![format!("2023-11-0{day}")];
            for i in 1..31 {
                f.push(if i == 26 {
                    ty.to_string()
                } else {
                    "0".to_string()
                });
            }
            rows.push_str(&f.join(","));
            rows.push('\n');
        }
        rows
    }

    #[test]
    fn test_coverage_and_status_boundaries() {
        // Pure core, so the shared singleton is left alone.
        let sw = cssi::parse_csv(&synthetic_table()).unwrap();
        let c = coverage_of(&sw).unwrap();
        let d = |n| Instant::from_date(2023, 11, n).unwrap();
        assert_eq!(c.first, d(1));
        assert_eq!(c.last_observed, d(3));
        assert_eq!(c.last_daily, d(4));
        assert_eq!(c.last, d(5));

        assert_eq!(status_in(&c, d(1)), SpaceWeatherStatus::Observed);
        assert_eq!(status_in(&c, d(3)), SpaceWeatherStatus::Observed);
        assert_eq!(status_in(&c, d(4)), SpaceWeatherStatus::PredictedDaily);
        assert_eq!(status_in(&c, d(5)), SpaceWeatherStatus::PredictedMonthly);
        assert_eq!(
            status_in(&c, Instant::from_date(2023, 12, 1).unwrap()),
            SpaceWeatherStatus::Extrapolated
        );
        assert_eq!(
            status_in(&c, Instant::from_date(2000, 1, 1).unwrap()),
            SpaceWeatherStatus::BeforeTable
        );
    }

    #[test]
    fn test_coverage_with_no_predictions() {
        // An all-observed table: every boundary collapses onto the last row.
        let mut rows = String::from("HEADER\n");
        for day in 1..=3 {
            let mut f: Vec<String> = vec![format!("2023-11-0{day}")];
            for i in 1..31 {
                f.push(if i == 26 {
                    "OBS".to_string()
                } else {
                    "0".to_string()
                });
            }
            rows.push_str(&f.join(","));
            rows.push('\n');
        }
        let sw = cssi::parse_csv(&rows).unwrap();
        let c = coverage_of(&sw).unwrap();
        assert_eq!(c.last_observed, c.last_daily);
        assert_eq!(c.last_daily, c.last);
    }

    #[test]
    fn test_parse_truncated_line_errors_not_panics() {
        // A truncated data line (fewer than the required fields) must return a
        // clean error rather than panicking on out-of-bounds indexing.
        let csv = "HEADER\n2023-11-14,1,2,3\n";
        assert!(matches!(cssi::parse_csv(csv), Err(Error::InvalidEntry)));
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
        let recs = cssi::parse_csv(&csv).unwrap();
        assert_eq!(recs.len(), 1);
    }
}
