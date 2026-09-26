pub mod assemble;
pub mod cssi;
pub mod gfz;
pub mod msafe;
pub mod swpc;

/// GFZ Potsdam observed record (CC BY 4.0).
pub const GFZ_FILE: &str = "Kp_ap_Ap_SN_F107_since_1932.txt";
/// NOAA/SWPC 45-day Ap and F10.7 forecast.
pub const SWPC_FILE: &str = "45-day-forecast.txt";
/// NASA MSFC MSAFE monthly forecast, stored under a stable name (NASA's own
/// file name changes every month).
pub const MSAFE_FILE: &str = "msafe-f10-prd.txt";
/// CelesTrak's `SW-All.csv`: no longer downloaded, still readable through
/// [`init_from_path`] for a cached copy or the file GMAT and Orekit read.
pub const CSSI_FILE: &str = "SW-All.csv";

/// The refresh URL for one of the daily feeds, from the manifest's
/// `refresh` list.
fn refresh_url(file: &str) -> Option<String> {
    crate::utils::manifest::embedded()
        .refresh
        .iter()
        .find(|u| u.ends_with(file))
        .cloned()
}

/// The directory part of [`refresh_url`], for `download_if_not_exist`.
fn refresh_base(file: &str) -> Option<String> {
    refresh_url(file)?.strip_suffix(file).map(str::to_string)
}

use std::cmp::Ordering;
use std::sync::atomic::AtomicBool;

use crate::utils::{datadir, diag, download_if_not_exist, RefreshableSingleton};
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

    /// The data directory cannot receive updated space-weather files:
    /// read-only filesystem, no write permission, or owned by another user.
    #[error(
        "Data directory {path} is not writable ({reason}). Set the environment variable \
         SATKIT_DATA to a writable directory and restart, or call set_datadir \
         (Python: satkit.utils.set_datadir) with one"
    )]
    DataDirReadOnly { path: String, reason: String },

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

/// Provenance of a space-weather row: which of the three sources it came
/// from (GFZ observed record, NOAA/SWPC 45-day forecast, NASA MSAFE monthly
/// forecast) and, for GFZ, whether the value is definitive yet. For a
/// CelesTrak `SW-All.csv` it is the `F10.7_DATA_TYPE` column (the one-digit
/// `Q` field in the fixed-width `.txt` twin).
///
/// The distinction matters for drag: only the observed variants are
/// measurements, and [`PredictedMonthly`](Self::PredictedMonthly) rows have
/// no 3-hourly structure — a single climatological daily Ap from MSAFE, or
/// no geomagnetic data at all (every `kp`/`ap` field `-1`) from `SW-All.csv`.
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
    /// Past the daily predictions, inside the MSAFE monthly forecast: the
    /// record returned is that month's 13-month-smoothed F10.7 and
    /// climatological daily Ap — the expected activity for that point in the
    /// solar cycle, with no storm timing and no 3-hourly structure. (A
    /// hand-loaded `SW-All.csv` carries **no** geomagnetic data here, so
    /// NRLMSISE-00 falls back to a quiet-time `Ap = 4`, up to a factor of
    /// two low in density during a storm; a one-time warning says so.)
    PredictedMonthly,
    /// After the last row of the table: that row's values are returned
    /// unchanged.
    Extrapolated,
    /// Before the first row of the table (1932 for the GFZ record, 1957 for
    /// `SW-All.csv`).
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
    /// A forecast row: daily `ap` (in all eight 3-hour slots, with the
    /// matching Cp / C9) and F10.7 `flux` (observed and adjusted); every
    /// other field is the `-1` "not given" sentinel.
    pub(crate) fn forecast(
        date: Instant,
        data_type: SpaceWeatherDataType,
        ap: i32,
        flux: f64,
    ) -> Self {
        let (cp, c9) = gfz::cp_c9(8 * ap);
        Self {
            date,
            bsrn: -1,
            nd: -1,
            data_type,
            kp: [-1; 8],
            kp_sum: -1,
            ap: [ap; 8],
            ap_avg: ap,
            cp,
            c9,
            isn: -1,
            f10p7_obs: flux,
            f10p7_adj: flux,
            f10p7_obs_c81: -1.0,
            f10p7_obs_l81: -1.0,
            f10p7_adj_c81: -1.0,
            f10p7_adj_l81: -1.0,
        }
    }

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

/// UTC day number of the last row of a parsed file, for
/// [`datadir::freshest_of`].
fn last_row_day(rows: &[SpaceWeatherRecord]) -> Option<i64> {
    rows.last().map(|r| r.date.utc_day_number())
}

/// The path to read `name` from — the freshest copy across the search
/// directories (see [`datadir::freshest_of`]) — or, when there is none, where it
/// would be written.
fn freshest_path_for(
    name: &str,
    last_day: impl Fn(&str) -> Option<i64>,
) -> Result<std::path::PathBuf> {
    match datadir::freshest_of(datadir::find_all(name), last_day) {
        Some(p) => Ok(p),
        None => Ok(datadir()?.join(name)),
    }
}

/// Lazy default load: the three primary sources under [`datadir`],
/// fetched on first use, assembled into one table. When a file exists in
/// more than one search directory the copy with the latest last row is
/// used, so a stale read-only copy cannot shadow a fresh download. A
/// `SW-All.csv` already in a search directory with no GFZ file anywhere is
/// read instead, so an existing cache or a provisioned copy keeps working.
fn load_default() -> Result<Vec<SpaceWeatherRecord>> {
    use crate::utils::datadir::path_for;
    let gfz_path = freshest_path_for(GFZ_FILE, |t| last_row_day(&gfz::parse(t).ok()?))?;
    if !gfz_path.is_file() {
        if let Ok(csv) = path_for(CSSI_FILE) {
            if csv.is_file() {
                return cssi::parse_csv(&std::fs::read_to_string(&csv)?);
            }
        }
        let base = refresh_base(GFZ_FILE)
            .unwrap_or_else(|| "https://www-app3.gfz-potsdam.de/kp_index/".to_string());
        download_if_not_exist(&gfz_path, Some(&base))?;
    }
    // The two forecasts are best-effort: a failed fetch leaves an
    // observed-only table.
    let swpc_path = freshest_path_for(SWPC_FILE, |t| last_row_day(&swpc::parse(t).ok()?))?;
    if !swpc_path.is_file() {
        if let Some(base) = refresh_base(SWPC_FILE) {
            let _ = download_if_not_exist(&swpc_path, Some(&base));
        }
    }
    let msafe_path = freshest_path_for(MSAFE_FILE, |t| {
        last_row_day(&msafe::parse(t).ok()?.records())
    })?;
    if !msafe_path.is_file() {
        if let Ok(dir) = datadir() {
            let _ = msafe::refresh_into(&dir, false);
        }
    }
    assemble_from_paths(&gfz_path, &swpc_path, &msafe_path)
}

/// Assemble the table from the three source files on disk, no download.
/// The GFZ record is required; the two forecasts are best-effort, since an
/// observed-only table is still a table and `status()` says where it ends.
fn assemble_from_paths(
    gfz_path: &std::path::Path,
    swpc_path: &std::path::Path,
    msafe_path: &std::path::Path,
) -> Result<Vec<SpaceWeatherRecord>> {
    let observed = gfz::parse(&std::fs::read_to_string(gfz_path)?)?;
    let daily = std::fs::read_to_string(swpc_path)
        .ok()
        .and_then(|s| swpc::parse(&s).ok())
        .unwrap_or_default();
    let monthly = std::fs::read_to_string(msafe_path)
        .ok()
        .and_then(|s| msafe::parse(&s).ok())
        .map(|f| f.records())
        .unwrap_or_default();
    Ok(assemble::assemble(observed, daily, monthly))
}

/// Assemble the table from the files in one directory, disk only: the three
/// primary sources when the GFZ record is there, otherwise the CSSI table.
fn assemble_from_dir(dir: &std::path::Path) -> Result<Vec<SpaceWeatherRecord>> {
    let gfz_path = dir.join(GFZ_FILE);
    if gfz_path.is_file() {
        return assemble_from_paths(&gfz_path, &dir.join(SWPC_FILE), &dir.join(MSAFE_FILE));
    }
    let csv = dir.join(CSSI_FILE);
    if csv.is_file() {
        return cssi::parse_csv(&std::fs::read_to_string(&csv)?);
    }
    Err(Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("no space-weather table in {}", dir.display()),
    )))
}

/// Load the space-weather table from the files in `dir`, replacing any
/// current contents. Disk only — nothing is downloaded. This is what
/// [`update_datafiles`](crate::utils::update_datafiles) calls after
/// refreshing into `dir`, so the in-memory table follows the files just
/// written rather than whatever was loaded first.
pub fn load_from_dir(dir: &std::path::Path) -> Result<()> {
    SPACE_WEATHER.set(assemble_from_dir(dir)?);
    Ok(())
}

/// Parse a space-weather text buffer, detecting the format from its first
/// line: CelesTrak's `SW-All.csv` (`DATE,BSRN,...`) or the GFZ table (`#`
/// header). A GFZ buffer becomes an observed-only table.
fn parse_any(text: &str) -> Result<Vec<SpaceWeatherRecord>> {
    let first = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    if first.starts_with("DATE,") {
        cssi::parse_csv(text)
    } else {
        let observed = gfz::parse(text)?;
        Ok(assemble::assemble(observed, Vec::new(), Vec::new()))
    }
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
static MONTHLY_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static EXTRAP_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static NOT_LOADED_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static DEFAULTS_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// One-time warning that NRLMSISE-00 is running on its default F10.7
/// (`f107`) and/or Ap (`ap`) at `tm` because the loaded table cannot supply
/// it: before the table starts, or before 1947 in the GFZ record, which has
/// Ap but no F10.7. Silent when no table is loaded at all, and for an Ap
/// missing from a row that exists (a monthly predicted row): [`get`] has
/// already reported both.
pub(crate) fn warn_model_defaults(tm: &Instant, f107: bool, ap: bool) {
    use std::sync::atomic::Ordering;
    if !(f107 || ap) || DEFAULTS_WARNING_SHOWN.load(Ordering::Relaxed) {
        return;
    }
    let Some(c) = coverage() else {
        return;
    };
    let before = status_in(&c, *tm) == SpaceWeatherStatus::BeforeTable;
    if !(f107 || before) || DEFAULTS_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
        return;
    }
    let what = match (f107, ap) {
        (true, true) => "F10.7 or Ap",
        (true, false) => "F10.7",
        _ => "Ap",
    };
    let reason = if before {
        format!(
            "the epoch is before the first row of the table, {}",
            c.first
        )
    } else {
        "the table has no measured F10.7 within three days of it and no 81-day average".to_string()
    };
    diag::warn!(
        "no {what} for {tm} in the space-weather table ({reason}); NRLMSISE-00 \
         uses its default for it (F10.7 = F10.7A = 150, Ap = 4), which can be wrong by a \
         factor of two in atmospheric density.\n\
         To disable: `satkit::spaceweather::disable_space_weather_time_warning()` \
         (Python: `satkit.spaceweather.disable_space_weather_time_warning()`)"
    );
}

/// Initialize the space-weather singleton from an in-memory byte buffer.
///
/// Either CelesTrak's `SW-All.csv` or the GFZ `Kp_ap_Ap_SN_F107_since_1932.txt`
/// table (UTF-8), detected from the content; a GFZ buffer yields an
/// observed-only table. Always succeeds and replaces any previously loaded
/// data — space weather updates daily and refresh-in-place is intended.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    SPACE_WEATHER.set(parse_any(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the space-weather singleton from a file at `path` (either
/// format — see [`init_from_bytes`]). Always replaces.
///
/// This is also the way to feed satkit the CSSI file GMAT and Orekit
/// read, when a comparison must pin against the same input.
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    SPACE_WEATHER.set(parse_any(&std::fs::read_to_string(path)?)?);
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
    SPACE_WEATHER.ensure_default_loaded(|| load_default().ok());
    if SPACE_WEATHER.read().is_none() {
        // Retry from disk only (no network) once the files have appeared.
        let on_disk = [GFZ_FILE, CSSI_FILE]
            .iter()
            .any(|f| datadir::path_for(f).is_ok_and(|p| p.is_file()));
        if on_disk {
            if let Ok(records) = load_default() {
                if !records.is_empty() {
                    SPACE_WEATHER.set(records);
                }
            }
        }
    }
}

/// Disable the warnings about out-of-range or missing space-weather data.
///
/// Four one-time warnings exist: an epoch past the daily predictions (only
/// monthly F10.7, no geomagnetic data), an epoch past the end of the table,
/// no table loaded at all, and an index NRLMSISE-00 has to take its default
/// for (an epoch before the table starts, or before 1947, when F10.7 was not
/// yet measured). Each is shown at most once per process; call this to
/// suppress all of them.
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
    DEFAULTS_WARNING_SHOWN.store(true, Ordering::Relaxed);
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
/// a smoothed monthly F10.7 and a climatological Ap with no storm
/// information, and [`SpaceWeatherStatus::Extrapolated`] means the files
/// should be refreshed.
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
/// Return the full space-weather record for the requested instant.
///
/// Returns the record for the same day when present, otherwise the most
/// recent prior record (no interpolation). For dates beyond the last record
/// the final record is returned. Forecast rows carry `-1` in the fields
/// their source does not publish — `kp` on every forecast row, `isn` on the
/// whole default table (see [`gfz`]) — and the monthly rows of a hand-loaded
/// `SW-All.csv` carry no geomagnetic data at all. See [`status`] and
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
/// * The table is assembled from the three files named by [`GFZ_FILE`],
///   [`SWPC_FILE`] and [`MSAFE_FILE`], refreshed by [`update`] or
///   [`update_datafiles`](crate::utils::update_datafiles).
pub fn get<T: TimeLike>(tm: &T) -> Result<SpaceWeatherRecord> {
    let tm = tm.as_instant();
    // Warnings are logged after the table's read lock is released.
    diag::deferred(|| get_locked(tm))
}

/// [`get`] under the table lock.
fn get_locked(tm: Instant) -> Result<SpaceWeatherRecord> {
    use std::sync::atomic::Ordering;
    let mut guard = SPACE_WEATHER.read();
    if guard.is_none() {
        drop(guard);
        ensure_default_loaded();
        guard = SPACE_WEATHER.read();
    }
    let Some(sw) = guard.as_ref().filter(|s| !s.is_empty()) else {
        if !NOT_LOADED_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            diag::warn!(
                "no space-weather table is loaded; NRLMSISE-00 is running on its \
                 defaults (F10.7 = F10.7A = 150, Ap = 4), which can be wrong by a factor of \
                 two in atmospheric density.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download the space-weather files, or set SATKIT_DATA to a directory \
                 containing them.\n\
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
        // A table that ends in the past is stale and a refresh extends it;
        // one that ends in the future already runs to the end of the
        // long-range forecast, which no refresh moves.
        let advice = if last.date < Instant::now() {
            "The table ends in the past, so it is out of date: run \
             `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
             to download the current space-weather files."
        } else {
            "The table already reaches past today to the end of its long-range forecast \
             (the default NASA MSAFE forecast runs about 15 years ahead); refreshing the data \
             files will not move that end."
        };
        diag::warn!(
            "the space-weather table ends at {}; the request for {tm} and all later \
             epochs return that row's values unchanged.\n\
             {advice}\n\
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
        diag::warn!(
            "the space-weather record for {tm} is a monthly prediction ({}); it \
             carries F10.7 but no Kp/ap, so NRLMSISE-00 runs on a quiet-time Ap = 4 with no \
             storm information. Density can be wrong by a factor of two during a geomagnetic \
             storm (measured 1.9x at 400 km for the 2024-05-11 event).\n\
             Daily data ends shortly after the last observed day; see \
             `satkit::spaceweather::coverage()` / `status()` (Python: \
             `satkit.spaceweather.coverage()` / `status()`).\n\
             To disable: `satkit::spaceweather::disable_space_weather_time_warning()` \
             (Python: `satkit.spaceweather.disable_space_weather_time_warning()`)",
            rec.date
        );
    }
    Ok(rec.clone())
}

/// Bring the three space-weather files in the data directory up to date and
/// load them.
///
/// GFZ and SWPC go through [`refresh_file`](crate::utils::refresh_file) —
/// a copy fetched within its publication cadence is not re-requested, an
/// older one costs a conditional GET — and MSAFE through
/// [`msafe::refresh_into`], which walks back from the current month to the
/// newest file NASA has published.
pub fn update() -> Result<()> {
    let d = datadir()?;
    datadir::check_writable(&d, |path, reason| Error::DataDirReadOnly { path, reason })?;
    for file in [GFZ_FILE, SWPC_FILE] {
        if let Some(url) = refresh_url(file) {
            crate::utils::refresh_file(&url, &d, false)?;
        }
    }
    msafe::refresh_into(&d, false)?;
    SPACE_WEATHER.set(load_default()?);
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

    /// `update_datafiles` reloads the singleton through `load_from_dir`: with
    /// the GFZ record beside a stale cached CSSI file the primary sources win,
    /// and the CSSI file is only read when no GFZ record is there.
    #[test]
    fn test_assemble_from_dir_prefers_primary_sources() {
        let dir = std::env::temp_dir().join(format!("satkit_sw_dir_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let gfz = "# comment\n\
2026 09 12 34588 34588.5 2633 11  1.333  2.333  2.333  2.000  1.667  1.333  1.000  0.667    5    9    9    7    6    5    4    3     6  92  109.9  111.8 2\n";
        let mut csv = String::from("HEADER\n");
        let mut f: Vec<String> = vec!["2023-11-01".to_string()];
        f.extend((1..31).map(|i| {
            if i == 26 {
                "OBS".to_string()
            } else {
                "0".to_string()
            }
        }));
        csv.push_str(&f.join(","));
        csv.push('\n');
        std::fs::write(dir.join(GFZ_FILE), gfz).unwrap();
        std::fs::write(dir.join(CSSI_FILE), &csv).unwrap();

        let recs = assemble_from_dir(&dir).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].date, Instant::from_date(2026, 9, 12).unwrap());
        assert_eq!(recs[0].isn, -1, "GFZ table leaves isn unset");

        std::fs::remove_file(dir.join(GFZ_FILE)).unwrap();
        let recs = assemble_from_dir(&dir).unwrap();
        assert_eq!(recs[0].date, Instant::from_date(2023, 11, 1).unwrap());

        std::fs::remove_file(dir.join(CSSI_FILE)).unwrap();
        assert!(assemble_from_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
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
