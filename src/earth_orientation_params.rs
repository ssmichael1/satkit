//! Earth Orientation Parameters (EOP) module
//!
//! This module provides access to Earth Orientation Parameters (EOP) data,
//! which are essential for accurate satellite orbit predictions and transformations
//! between different reference frames.
//!
//! The EOP data includes parameters such as polar motion, UT1-UTC, and length of day (LOD),
//! which are crucial for precise calculations in satellite tracking and navigation.
//!
//! # Sources
//!
//! Two on-disk formats are read, and the loader uses whichever is fresher
//! (later last *observed* row) when both are present in the data directories:
//!
//! * **`finals2000A.all`** — the IERS Rapid Service / Prediction Centre's
//!   Bulletin A combined file: observed values from 1973 plus about a year of
//!   predictions, updated daily. This is the primary source; the refresh
//!   fetches it from the USNO and IERS mirrors. The Bulletin A columns are
//!   used throughout (never the Bulletin B columns, which end earlier and
//!   would introduce a splice).
//! * **`EOP-All.csv`** — CelesTrak's repackaging of the IERS series, which
//!   reaches back to 1962 and carries about six months of predictions. It is
//!   the fallback when both IERS mirrors are unreachable, and it is still read
//!   when present (a hand-provisioned data directory, the `satkit-data`
//!   bundle). When `finals2000A.all` is the fresher table and an `EOP-All.csv`
//!   is also present, the CSV's rows before 1973 are kept so 1962–1972
//!   coverage is not lost.
//!
//! The source order lives in the embedded data manifest (`data/manifest.json`,
//! `eop` section); [`source`] reports which file the loaded table came from.
//!
//! Both files are published once a day, so [`refresh_into`] leaves a copy
//! fetched within the last 24 h alone without contacting anyone, and past
//! that sends a conditional request that costs a `304` when the file has not
//! changed (see [`refresh_file`](crate::utils::refresh_file)).
//!
//! This module also provides a way to disable warnings about out-of-range EOP data,
//! allowing users to suppress these warnings if they are aware of the limitations of the data.
//!
//! See: <https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html> for details on EOP data,
//! and <https://maia.usno.navy.mil/ser7/readme.finals2000A> for the `finals2000A.all` format.
//!

use crate::utils::datadir;
use crate::utils::download::{self, refresh_file};
use crate::utils::manifest::RefreshSource;
use crate::utils::RefreshableSingleton;
use crate::{Instant, TimeLike, TimeScale};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

/// File name of the IERS Bulletin A combined file (the primary source).
pub const FINALS2000A_FILE: &str = "finals2000A.all";
/// File name of the CelesTrak EOP file (fallback source, legacy on-disk format).
pub const CELESTRAK_FILE: &str = "EOP-All.csv";

/// Errors produced by the
/// [`earth_orientation_params`](crate::earth_orientation_params) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A line in the EOP CSV file has fewer than the expected 12 fields.
    #[error("Invalid entry in EOP file")]
    InvalidEntry,

    /// A `finals2000A.all` line could not be parsed (a flag other than `I`/`P`,
    /// or a numeric column that is neither blank nor a number).
    #[error("Invalid finals2000A.all line {line}: {reason}")]
    InvalidFinalsLine { line: usize, reason: String },

    /// Neither `finals2000A.all` nor `EOP-All.csv` could be read from the
    /// data directory (after a refresh that reported success, or when
    /// loading a directory explicitly).
    #[error(
        "No Earth orientation file ({FINALS2000A_FILE} or {CELESTRAK_FILE}) readable in {dir}"
    )]
    NoEopFile { dir: String },

    /// The configured data directory is read-only and cannot receive an
    /// updated EOP file.
    #[error(
        "Data directory is read-only. Try setting the environment variable SATKIT_DATA \
         to a writeable directory and re-starting or explicitly set data directory"
    )]
    DataDirReadOnly,

    /// Bytes passed to [`init_from_bytes`] were not valid UTF-8 — both EOP
    /// file formats are text.
    #[error("EOP byte buffer is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseFloat(#[from] ParseFloatError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the
/// `earth_orientation_params` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Which file the loaded EOP table came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopSource {
    /// IERS Bulletin A combined file `finals2000A.all` (Bulletin A columns).
    IersFinals2000A,
    /// CelesTrak `EOP-All.csv`.
    CelesTrak,
}

impl EopSource {
    /// The on-disk file name this source is read from.
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::IersFinals2000A => FINALS2000A_FILE,
            Self::CelesTrak => CELESTRAK_FILE,
        }
    }

    /// The source read from a file of this name, if it is one of the two.
    pub fn from_file_name(name: &str) -> Option<Self> {
        match name {
            FINALS2000A_FILE => Some(Self::IersFinals2000A),
            CELESTRAK_FILE => Some(Self::CelesTrak),
            _ => None,
        }
    }
}

impl std::fmt::Display for EopSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IersFinals2000A => write!(f, "IERS finals2000A.all"),
            Self::CelesTrak => write!(f, "CelesTrak EOP-All.csv"),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
struct EOPEntry {
    mjd_utc: f64,
    /// Polar motion, arcsec.
    xp: f64,
    yp: f64,
    /// UT1−UTC, seconds.
    dut1: f64,
    /// Excess length of day, seconds per day.
    lod: f64,
    /// Celestial pole offsets wrt IAU 2000A, milliarcsec.
    dX: f64,
    dY: f64,
    /// `true` for an observed row (`I` flag in `finals2000A.all`, `O` in
    /// `EOP-All.csv`), `false` for a predicted (`P`) row.
    observed: bool,
}

/// The loaded table: entries sorted by MJD, plus where they came from.
#[derive(Debug)]
struct EopTable {
    entries: Vec<EOPEntry>,
    source: EopSource,
}

/// Where a given epoch falls relative to the loaded EOP table.
///
/// Returned by [`status`]; see [`coverage`] for the table bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopStatus {
    /// Inside the table, on or before the last observed row.
    Observed,
    /// Inside the table, after the last observed row: IERS predictions.
    Predicted,
    /// After the last row of the table: the last row's values are held
    /// constant. Accuracy degrades with distance from the table end
    /// (polar motion drifts ~0.1″ and UT1−UTC by ~10 ms over a few
    /// months) — refresh the data with
    /// [`update`] / `satkit::utils::update_datafiles()`.
    Extrapolated,
    /// Before the first row of the table (1973 for `finals2000A.all`, 1962
    /// for `EOP-All.csv`): no EOP available, [`get`] returns `None` and the
    /// frame transforms use zeros.
    BeforeTable,
    /// No EOP table is loaded at all (file missing and download failed,
    /// or an empty table was installed): [`get`] returns `None` and the
    /// frame transforms use zeros.
    NotLoaded,
}

/// Time bounds of the loaded EOP table (UTC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EopCoverage {
    /// Epoch of the first row.
    pub first: Instant,
    /// Epoch of the last observed row; rows after it are IERS
    /// predictions.
    pub last_observed: Instant,
    /// Epoch of the last row (observed or predicted). Queries after it
    /// return this row's values unchanged.
    pub last: Instant,
}

/// Parse an `EOP-All.csv` text buffer into EOP entries.
///
/// CelesTrak's `DX`/`DY` columns are in arcsec; they are stored in
/// milliarcsec, the unit the nutation correction takes.
fn parse_csv(text: &str) -> Result<Vec<EOPEntry>> {
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| -> Result<EOPEntry> {
            let lvals: Vec<&str> = line.split(",").collect();
            if lvals.len() < 12 {
                return Err(Error::InvalidEntry);
            }
            Ok(EOPEntry {
                mjd_utc: lvals[1].parse()?,
                xp: lvals[2].parse()?,
                yp: lvals[3].parse()?,
                dut1: lvals[4].parse()?,
                lod: lvals[5].parse()?,
                dX: lvals[8].parse::<f64>()? * 1.0e3,
                dY: lvals[9].parse::<f64>()? * 1.0e3,
                observed: lvals[11].trim() != "P",
            })
        })
        .collect()
}

/// The 1-based fixed-width column range `[start, start + len)` of `line`,
/// or an empty string where the line is too short (trailing blanks are
/// routinely trimmed by editors and by the IERS file itself for the
/// prediction rows that have no nutation values).
fn cols(line: &str, start: usize, len: usize) -> &str {
    let bytes = line.as_bytes();
    let begin = (start - 1).min(bytes.len());
    let end = (begin + len).min(bytes.len());
    // The file is ASCII; a non-ASCII line would already have been rejected.
    std::str::from_utf8(&bytes[begin..end]).unwrap_or("")
}

/// Parse an IERS `finals2000A.all` text buffer (Bulletin A columns) into
/// EOP entries.
///
/// Column layout (1-based, from `readme.finals2000A`): MJD 8–15; polar
/// motion flag 17 (`I` observed, `P` predicted), x 19–27 and y 38–46
/// (arcsec); UT1−UTC flag 58, value 59–68 (s); LOD 80–86 (ms, not always
/// filled); nutation flag 96, dX 98–106 and dY 117–125 (mas, not always
/// filled). Rows whose flag is blank (the tail of the file) end the table.
///
/// A blank LOD is filled by the finite difference of UT1−UTC across the
/// following day (the excess length of day is −d(UT1−UTC)/dt), which is
/// what the prediction rows carry in the CelesTrak file; a blank dX/dY is
/// zero (no correction to the IAU 2000A model).
fn parse_finals2000a(text: &str) -> Result<Vec<EOPEntry>> {
    let invalid = |line: usize, reason: String| Error::InvalidFinalsLine { line, reason };
    let num = |line: usize, s: &str, what: &str| -> Result<f64> {
        s.trim()
            .parse::<f64>()
            .map_err(|e| invalid(line, format!("{what} {s:?}: {e}")))
    };
    let opt = |line: usize, s: &str, what: &str| -> Result<Option<f64>> {
        if s.trim().is_empty() {
            Ok(None)
        } else {
            num(line, s, what).map(Some)
        }
    };

    let mut rows: Vec<EOPEntry> = Vec::new();
    let mut lod_missing: Vec<usize> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        if !line.is_ascii() {
            return Err(invalid(lineno, "non-ASCII text".into()));
        }
        let observed = match cols(line, 17, 1) {
            "I" => true,
            "P" => false,
            // The file ends with rows that carry a date but no values.
            " " | "" => continue,
            other => return Err(invalid(lineno, format!("polar motion flag {other:?}"))),
        };
        let mjd_utc = num(lineno, cols(line, 8, 8), "MJD")?;
        let xp = num(lineno, cols(line, 19, 9), "x pole")?;
        let yp = num(lineno, cols(line, 38, 9), "y pole")?;
        // Polar motion without UT1−UTC would leave a row that cannot be
        // used; it has not happened in the file's history, but the flag is
        // separate, so treat such a row as the end of usable data.
        let Some(dut1) = opt(lineno, cols(line, 59, 10), "UT1-UTC")? else {
            continue;
        };
        let lod = opt(lineno, cols(line, 80, 7), "LOD")?.map(|ms| ms * 1.0e-3);
        let dx = opt(lineno, cols(line, 98, 9), "dX")?.unwrap_or(0.0);
        let dy = opt(lineno, cols(line, 117, 9), "dY")?.unwrap_or(0.0);
        if lod.is_none() {
            lod_missing.push(rows.len());
        }
        rows.push(EOPEntry {
            mjd_utc,
            xp,
            yp,
            dut1,
            lod: lod.unwrap_or(0.0),
            dX: dx,
            dY: dy,
            observed,
        });
    }
    rows.sort_by(|a, b| a.mjd_utc.total_cmp(&b.mjd_utc));
    rows.dedup_by(|a, b| a.mjd_utc == b.mjd_utc);
    for &i in &lod_missing {
        // Forward difference where there is a next row, else the backward
        // one; a jump of about a second is a leap second, not a rate.
        let pair = if i + 1 < rows.len() {
            Some((i, i + 1))
        } else if i > 0 {
            Some((i - 1, i))
        } else {
            None
        };
        if let Some((a, b)) = pair {
            let d = rows[b].dut1 - rows[a].dut1;
            let dt = rows[b].mjd_utc - rows[a].mjd_utc;
            if dt > 0.0 && d.abs() < 0.5 {
                rows[i].lod = -d / dt;
            }
        }
    }
    Ok(rows)
}

/// The format of an EOP text buffer, from its first line: CelesTrak's CSV
/// starts with its header, anything else is taken as `finals2000A.all`.
fn detect_source(text: &str) -> EopSource {
    if text.trim_start_matches('\u{feff}').starts_with("DATE,") {
        EopSource::CelesTrak
    } else {
        EopSource::IersFinals2000A
    }
}

/// Parse either EOP format, detected from the content.
fn parse_any(text: &str) -> Result<EopTable> {
    let source = detect_source(text);
    let entries = match source {
        EopSource::CelesTrak => parse_csv(text)?,
        EopSource::IersFinals2000A => parse_finals2000a(text)?,
    };
    Ok(EopTable { entries, source })
}

/// Check that the file at `path` is a parsable EOP file (either format),
/// without touching the loaded table.
///
/// Used by the downloader to reject a response that is not the file it
/// claims to be — a proxy notice page served with `200 OK`, or a truncated
/// transfer — before it replaces a good table on disk. The check is the real
/// parser, so anything that would later fail to load fails here instead,
/// while the previous file is still in place.
pub(crate) fn validate_file(path: &Path) -> std::result::Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    match parse_any(&text) {
        Ok(t) if t.entries.is_empty() => Err("the file holds no EOP rows".to_string()),
        Ok(_) => Ok(()),
        Err(e) => Err(format!("not a parsable EOP file ({e})")),
    }
}

/// MJD of the last observed row, or −∞ for a table with none.
fn last_observed_mjd(rows: &[EOPEntry]) -> f64 {
    rows.iter()
        .rev()
        .find(|e| e.observed)
        .map_or(f64::NEG_INFINITY, |e| e.mjd_utc)
}

/// Choose between the two tables that may be on disk: the one whose
/// observed record runs later wins (ties go to the IERS file). When the IERS
/// file wins and a CelesTrak table is also present, the CelesTrak rows
/// before the IERS file's first row (1962–1972) are kept in front of it.
fn select_table(finals: Option<Vec<EOPEntry>>, csv: Option<Vec<EOPEntry>>) -> Option<EopTable> {
    let finals = finals.filter(|r| !r.is_empty());
    let csv = csv.filter(|r| !r.is_empty());
    match (finals, csv) {
        (None, None) => None,
        (Some(entries), None) => Some(EopTable {
            entries,
            source: EopSource::IersFinals2000A,
        }),
        (None, Some(entries)) => Some(EopTable {
            entries,
            source: EopSource::CelesTrak,
        }),
        (Some(finals), Some(csv)) => {
            if last_observed_mjd(&csv) > last_observed_mjd(&finals) {
                return Some(EopTable {
                    entries: csv,
                    source: EopSource::CelesTrak,
                });
            }
            let first = finals[0].mjd_utc;
            let mut entries: Vec<EOPEntry> =
                csv.into_iter().filter(|e| e.mjd_utc < first).collect();
            entries.extend(finals);
            Some(EopTable {
                entries,
                source: EopSource::IersFinals2000A,
            })
        }
    }
}

/// Parse the file at `path`, or `None` (with a warning) if it is unreadable
/// or holds no rows — a corrupt copy of one file must not hide the other.
fn read_table(path: Option<&Path>) -> Option<Vec<EOPEntry>> {
    let p = path?;
    let parsed = std::fs::read_to_string(p)
        .map_err(Error::from)
        .and_then(|t| parse_any(&t));
    match parsed {
        Ok(t) if !t.entries.is_empty() => Some(t.entries),
        Ok(_) => {
            eprintln!("Warning: {} holds no EOP rows; ignoring it", p.display());
            None
        }
        Err(e) => {
            eprintln!("Warning: could not read {}: {e}; ignoring it", p.display());
            None
        }
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

/// The table assembled from the two files at the given paths (either may be
/// absent), per [`select_table`]. Does not touch the loaded table.
fn load_from_paths(finals: Option<PathBuf>, csv: Option<PathBuf>) -> Option<EopTable> {
    select_table(read_table(finals.as_deref()), read_table(csv.as_deref()))
}

/// Load the EOP table from the files in `dir` (`finals2000A.all` and/or
/// `EOP-All.csv`), replacing any loaded table. Used after a refresh into
/// an explicit directory; the lazy default load searches all data
/// directories instead.
pub fn load_from_dir(dir: &Path) -> Result<()> {
    let table = load_from_paths(
        existing(dir.join(FINALS2000A_FILE)),
        existing(dir.join(CELESTRAK_FILE)),
    )
    .ok_or_else(|| Error::NoEopFile {
        dir: dir.display().to_string(),
    })?;
    EOP.set(table);
    Ok(())
}

/// Lazy default load: the two files as found in the data search
/// directories; when neither exists, refresh into the write location first.
fn load_default() -> Result<EopTable> {
    let mut finals = datadir::find_file(FINALS2000A_FILE);
    let mut csv = datadir::find_file(CELESTRAK_FILE);
    if finals.is_none() && csv.is_none() {
        let dir = datadir::datadir()?;
        refresh_into(&dir, false)?;
        finals = existing(dir.join(FINALS2000A_FILE));
        csv = existing(dir.join(CELESTRAK_FILE));
    }
    load_from_paths(finals.clone(), csv.clone()).ok_or_else(|| Error::NoEopFile {
        dir: finals
            .or(csv)
            .and_then(|p| p.parent().map(|d| d.display().to_string()))
            .unwrap_or_else(|| "the data directories".to_string()),
    })
}

/// What a refresh settled on: which source, the URL it was checked against,
/// and whether that cost a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshOutcome {
    /// The file that is now current on disk.
    pub source: EopSource,
    /// The URL it was fetched from, or would have been: for a copy still
    /// inside its publication cadence no request was made.
    pub url: String,
    /// Whether the file was transferred, answered `304`, or not requested.
    pub fetch: download::RefreshOutcome,
}

/// Bring the EOP file in `dir` up to date, trying the sources in the order
/// the embedded data manifest lists them: `finals2000A.all` from the USNO
/// and IERS mirrors, then CelesTrak's `EOP-All.csv`.
///
/// Each URL goes through [`refresh_file`](crate::utils::refresh_file), so a
/// copy fetched within the last 24 h is reported current without a request
/// and an older one costs a conditional GET (`304` when unchanged); `force`
/// transfers the file unconditionally. The first source that answers is
/// kept (an HTML notice page or a truncated transfer is rejected and the
/// next URL tried); a fallback past the primary source is reported on
/// stderr. Does not change the loaded table — call [`load_from_dir`] or
/// [`update`] for that.
///
/// Fails with [`download::Error::Offline`] under offline mode without any
/// network I/O, and with [`download::Error::AllSourcesFailed`] listing every
/// URL and its error when nothing could be fetched.
pub fn refresh_into(dir: &Path, force: bool) -> download::Result<RefreshOutcome> {
    refresh_into_with_sources(dir, &crate::utils::manifest::embedded().eop, force)
}

/// [`refresh_into`] with an explicit source list (the manifest's `eop`
/// section in production; test servers in tests).
pub(crate) fn refresh_into_with_sources(
    dir: &Path,
    sources: &[RefreshSource],
    force: bool,
) -> download::Result<RefreshOutcome> {
    download::check_online("Earth orientation parameters")?;
    let mut attempts: Vec<String> = Vec::new();
    for (rank, src) in sources.iter().enumerate() {
        let Some(source) = EopSource::from_file_name(&src.name) else {
            continue;
        };
        for url in &src.urls {
            match refresh_file(url, dir, force) {
                Ok(fetch) => {
                    if rank > 0 && fetch == download::RefreshOutcome::Downloaded {
                        eprintln!(
                            "Warning: the primary Earth orientation source was unreachable; \
                             using {source} from {url} instead.\n  {}",
                            attempts.join("\n  ")
                        );
                    }
                    return Ok(RefreshOutcome {
                        source,
                        url: url.clone(),
                        fetch,
                    });
                }
                Err(e) => attempts.push(format!("{url}: {e}")),
            }
        }
    }
    Err(download::Error::AllSourcesFailed {
        name: "Earth orientation parameters".to_string(),
        attempts,
        hint: None,
    })
}

/// `true` when `mjd_utc` lies strictly after the last table row; a query at
/// exactly the last epoch is inside the table.
fn beyond_table(mjd_utc: f64, last: &EOPEntry) -> bool {
    mjd_utc > last.mjd_utc
}

static WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static EXTRAP_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static NOT_LOADED_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// Module-scope refreshable singleton. The lazy default load (best-effort,
/// silent on failure) runs at most once; [`init_from_bytes`] /
/// [`init_from_path`] / [`update`] replace any current contents.
static EOP: RefreshableSingleton<EopTable> = RefreshableSingleton::new();

/// Best-effort default load on first read. Failures are silent — if EOP
/// can't be loaded, the singleton stays empty and queries fall through
/// to the "no data" branch.
fn ensure_default_loaded() {
    EOP.ensure_default_loaded(|| load_default().ok());
}

/// Initialize the EOP singleton from an in-memory byte buffer.
///
/// The bytes must be a valid `finals2000A.all` or CelesTrak `EOP-All.csv`
/// text file (UTF-8); the format is detected from the content. Always
/// succeeds and replaces any previously loaded data — IERS publishes new
/// EOP daily and refresh-in-place is the intended model.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    EOP.set(parse_any(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the EOP singleton from a file at `path` (either format).
///
/// Same semantics as [`init_from_bytes`]; always replaces.
pub fn init_from_path(path: &Path) -> Result<()> {
    EOP.set(parse_any(&std::fs::read_to_string(path)?)?);
    Ok(())
}

///
/// Disable the warnings about out-of-range or missing EOP data.
///
/// Three one-time warnings exist: epoch before the table, epoch after the
/// table (values held constant), and no table loaded at all (zeros used).
/// Each is shown at most once per process; call this to suppress all of
/// them.
///
/// # Example
///
/// ```rust
/// satkit::earth_orientation_params::disable_eop_time_warning();
/// ```
///
pub fn disable_eop_time_warning() {
    WARNING_SHOWN.store(true, Ordering::Relaxed);
    EXTRAP_WARNING_SHOWN.store(true, Ordering::Relaxed);
    NOT_LOADED_WARNING_SHOWN.store(true, Ordering::Relaxed);
}

/// Time bounds of the loaded EOP table, or `None` if no table is loaded
/// (file missing and download failed, or an empty table was installed).
///
/// # Example
///
/// ```rust
/// if let Some(c) = satkit::earth_orientation_params::coverage() {
///     println!("EOP observed through {}, predicted through {}", c.last_observed, c.last);
/// }
/// ```
pub fn coverage() -> Option<EopCoverage> {
    ensure_default_loaded();
    let guard = EOP.read();
    let eop = &guard.as_ref()?.entries;
    let first = eop.first()?;
    let last = eop.last()?;
    let last_observed = eop.iter().rev().find(|e| e.observed).unwrap_or(first);
    Some(EopCoverage {
        first: Instant::from_mjd_utc(first.mjd_utc),
        last_observed: Instant::from_mjd_utc(last_observed.mjd_utc),
        last: Instant::from_mjd_utc(last.mjd_utc),
    })
}

/// Which file the loaded EOP table came from, or `None` if no table is
/// loaded. A table assembled from both files (IERS rows with the
/// CelesTrak file's pre-1973 history in front) reports
/// [`EopSource::IersFinals2000A`].
pub fn source() -> Option<EopSource> {
    ensure_default_loaded();
    let guard = EOP.read();
    guard
        .as_ref()
        .filter(|t| !t.entries.is_empty())
        .map(|t| t.source)
}

/// Classify an epoch against the loaded EOP table — see [`EopStatus`].
///
/// Useful before a long propagation or a precision frame transform: a
/// result of [`EopStatus::Extrapolated`] means the data file should be
/// refreshed, and [`EopStatus::NotLoaded`] means every EOP-dependent
/// transform is using zeros.
pub fn status<T: TimeLike>(tm: &T) -> EopStatus {
    let mjd_utc = tm.as_mjd_with_scale(TimeScale::UTC);
    ensure_default_loaded();
    let guard = EOP.read();
    let Some(eop) = guard
        .as_ref()
        .map(|t| t.entries.as_slice())
        .filter(|e| !e.is_empty())
    else {
        return EopStatus::NotLoaded;
    };
    if mjd_utc < eop[0].mjd_utc {
        return EopStatus::BeforeTable;
    }
    if mjd_utc > eop[eop.len() - 1].mjd_utc {
        return EopStatus::Extrapolated;
    }
    if mjd_utc <= last_observed_mjd(eop) {
        EopStatus::Observed
    } else {
        EopStatus::Predicted
    }
}

/// Bring the Earth Orientation Parameters file in the data directory up to
/// date (see [`refresh_into`] for the source order and the once-a-day
/// cadence; a copy fetched within the last 24 h is not re-requested), and
/// load it.
pub fn update() -> Result<()> {
    let d = datadir::datadir()?;
    if d.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }
    refresh_into(&d, false)?;
    load_from_dir(&d)
}

///
/// Get Earth Orientation Parameters at given Modified Julian Date (UTC)
/// Returns None if no data is available for the given date
///
/// # Arguments:
///
/// * `mjd_utc` - Modified Julian Date (UTC)
///
/// # Returns:
///
/// * If time is valid within file, Vector [f64; 6] with following elements:
///     * 0 : (UT1 - UTC) in seconds
///     * 1 : X polar motion in arcsecs
///     * 2 : Y polar motion in arcsecs
///     * 3 : LOD: excess length of day (−d(UT1−UTC)/dt), seconds per day
///     * 4 : dX wrt IAU 2000A nutation, milli-arcsecs
///     * 5 : dY wrt IAU 2000A nutation, milli-arcsecs
///
/// * If time is before range of file, returns None and prints warning to stderr
///   (but only once per library load)
/// * If time is after range of file, returns the last entry's values (constant
///   extrapolation) and prints a warning to stderr the first time this happens
/// * If no table is loaded at all, returns None and prints a warning to stderr
///   the first time this happens
///
/// Use [`status`] / [`coverage`] to check which regime an epoch is in without
/// relying on the warnings; [`disable_eop_time_warning`] suppresses them.
///
pub fn eop_from_mjd_utc(mjd_utc: f64) -> Option<[f64; 6]> {
    ensure_default_loaded();
    let guard = EOP.read();
    let Some(eop) = guard
        .as_ref()
        .map(|t| t.entries.as_slice())
        .filter(|e| !e.is_empty())
    else {
        if !NOT_LOADED_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: no Earth Orientation Parameters (EOP) table is loaded; polar motion, \
                 UT1-UTC and nutation corrections are being treated as zero, which biases \
                 Earth-fixed frame transforms and orbit propagation by metres.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download finals2000A.all, or set SATKIT_DATA to a directory containing it \
                 (or a CelesTrak EOP-All.csv).\n\
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()`"
            );
        }
        return None;
    };

    // Binary search: find first entry with mjd_utc > query (O(log n) vs O(n) linear scan)
    let idx = eop.partition_point(|x| x.mjd_utc <= mjd_utc);

    if idx == 0 {
        if !WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: EOP data not available for MJD UTC = {mjd_utc} (too early).\n\
                 Run `satkit::utils::update_datafiles()` to download the most recent data.\n\
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()`"
            );
        }
        return None;
    }

    // At or beyond the last row, use the last entry's values. A query at
    // exactly the last epoch is still inside the table: no warning.
    if idx >= eop.len() {
        let last = &eop[eop.len() - 1];
        if beyond_table(mjd_utc, last) && !EXTRAP_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: EOP data ends at {} (MJD {}); the request for MJD UTC = {mjd_utc} and \
                 all later epochs use the last entry's values held constant. Polar motion and \
                 UT1-UTC drift by ~0.1 arcsec / ~10 ms over a few months, i.e. metres at LEO.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download the most recent Earth orientation file.\n\
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()`",
                Instant::from_mjd_utc(last.mjd_utc),
                last.mjd_utc
            );
        }
        return Some([last.dut1, last.xp, last.yp, last.lod, last.dX, last.dY]);
    }

    // Linear interpolation between bracketing entries
    let v0 = &eop[idx - 1];
    let v1 = &eop[idx];
    let g1 = (mjd_utc - v0.mjd_utc) / (v1.mjd_utc - v0.mjd_utc);
    let g0 = 1.0 - g1;
    Some([
        g0.mul_add(v0.dut1, g1 * v1.dut1),
        g0.mul_add(v0.xp, g1 * v1.xp),
        g0.mul_add(v0.yp, g1 * v1.yp),
        g0.mul_add(v0.lod, g1 * v1.lod),
        g0.mul_add(v0.dX, g1 * v1.dX),
        g0.mul_add(v0.dY, g1 * v1.dY),
    ])
}

///
/// Get Earth Orientation Parameters at given instant
///
/// # Arguments:
///
/// * tm: Instant at which to query parameters
///
/// # Returns:
///
/// * Vector [f64; 6] with following elements:
///   * 0 : (UT1 - UTC) in seconds
///   * 1 : X polar motion in arcsecs
///   * 2 : Y polar motion in arcsecs
///   * 3 : LOD: excess length of day (−d(UT1−UTC)/dt), seconds per day
///   * 4 : dX wrt IAU 2000A nutation, milli-arcsecs
///   * 5 : dY wrt IAU 2000A nutation, milli-arcsecs
///
///
/// # Example:
///
/// ```rust
/// let tm = satkit::Instant::from_rfc3339("2006-04-16T17:52:50.805408Z").unwrap();
/// let eop = satkit::earth_orientation_params::get(&tm);
/// ```
///
#[inline]
pub fn get<T: crate::TimeLike>(tm: &T) -> Option<[f64; 6]> {
    eop_from_mjd_utc(tm.as_mjd_with_scale(crate::TimeScale::UTC))
}

/// Same as [`get`], but returns all-zero parameters when EOP data is
/// unavailable — the standard fallback used by the frame transforms.
#[inline]
pub fn get_or_zero<T: crate::TimeLike>(tm: &T) -> [f64; 6] {
    get(tm).unwrap_or([0.0; 6])
}

/// Same as [`eop_from_mjd_utc`], but returns all-zero parameters when EOP
/// data is unavailable — the standard fallback used by the frame transforms.
#[inline]
pub fn eop_from_mjd_utc_or_zero(mjd_utc: f64) -> [f64; 6] {
    eop_from_mjd_utc(mjd_utc).unwrap_or([0.0; 6])
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Six real lines of `finals2000A.all` (trailing blanks trimmed, as an
    /// editor would): an early observed row with a filled zero LOD and
    /// predicted nutation; a fully observed row; the last observed row of a
    /// file, with a blank LOD; a predicted row with nutation; a predicted row
    /// without nutation; and a date-only tail row.
    const FINALS_SAMPLE: &str = "\
73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    -0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667
92 3 2 48683.00 I  0.006416 0.000153  0.116395 0.000159  I-0.2719443 0.0000093  2.3575 0.0080  I     0.054    0.168     0.085    0.150   .006400   .116200  -.2719420      0.054     0.085
26 917 61300.00 I  0.190054 0.000090  0.329163 0.000090  I-0.0086337 0.0000267                 P     0.084    0.128     0.235    0.160
26 918 61301.00 P  0.189180 0.000600  0.329137 0.000401  P-0.0091919 0.0001080                 P     0.094    0.128     0.236    0.160
27 925 61673.00 P  0.235938 0.017545  0.302527 0.028482  P-0.1313246 0.0254096
27 926 61674.00
";

    const CSV_SAMPLE: &str = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n\
        1962-01-01,37665,-0.012700,0.213000,0.0326338,0.0017230,0.064261,0.006067,0.000000,0.000000,2,O\n\
        1972-12-31,41682,0.125800,0.125100,0.8135920,0.0026960,0.046960,0.003620,0.000000,0.000000,11,O\n\
        1973-01-02,41684,0.123500,0.123000,0.8078584,0.0027100,0.046923,0.003514,0.000000,0.000000,12,O\n\
        2021-09-07,59464,0.241182,0.317273,-0.1145667,-0.0002255,-0.118552,-0.009274,-0.000102,-0.000150,37,O\n\
        2026-09-18,61301,0.187672,0.328316,-0.0073956,0.0000148,-0.124590,-0.011105,0.000295,-0.000027,37,P\n";

    /// Check that data is loaded
    #[test]
    fn loaded() {
        ensure_default_loaded();
        let guard = EOP.read();
        let eop = guard
            .as_ref()
            .expect("default EOP load should succeed in tests");
        assert!(eop.entries[0].mjd_utc >= 0.0);
    }

    #[test]
    fn parse_finals_columns_flags_and_units() {
        let rows = parse_finals2000a(FINALS_SAMPLE).unwrap();
        assert_eq!(rows.len(), 5, "the date-only tail row is not a table row");

        let r = &rows[0];
        assert_eq!(r.mjd_utc, 41684.0);
        assert!(r.observed);
        assert!((r.xp - 0.120733).abs() < 1e-12);
        assert!((r.yp - 0.136966).abs() < 1e-12);
        assert!((r.dut1 - 0.8084178).abs() < 1e-12);
        // LOD is filled (0.0000 ms) and dX/dY are the Bulletin A predictions, in mas.
        assert_eq!(r.lod, 0.0);
        assert!((r.dX - -0.766).abs() < 1e-12);
        assert!((r.dY - -0.720).abs() < 1e-12);

        let r = &rows[1];
        assert!(r.observed);
        assert!((r.dut1 - -0.2719443).abs() < 1e-12);
        assert!((r.lod - 2.3575e-3).abs() < 1e-15, "LOD ms -> s");
        assert!((r.dX - 0.054).abs() < 1e-12);
        assert!((r.dY - 0.085).abs() < 1e-12);

        // Last observed row: blank LOD is the forward difference of UT1-UTC.
        let r = &rows[2];
        assert!(r.observed);
        let expected_lod = -(-0.0091919 - -0.0086337) / 1.0;
        assert!((r.lod - expected_lod).abs() < 1e-12, "{}", r.lod);

        let r = &rows[3];
        assert!(!r.observed);
        assert!((r.dX - 0.094).abs() < 1e-12);

        // Prediction without nutation: zero correction, LOD from the
        // backward difference (it is the last row).
        let r = &rows[4];
        assert!(!r.observed);
        assert_eq!((r.dX, r.dY), (0.0, 0.0));
        let expected_lod = -(-0.1313246 - -0.0091919) / (61673.0 - 61301.0);
        assert!((r.lod - expected_lod).abs() < 1e-12, "{}", r.lod);
    }

    #[test]
    fn parse_finals_rejects_garbage_but_skips_blank_lines() {
        let bad =
            "73 1 2 41684.00 X  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710\n";
        assert!(matches!(
            parse_finals2000a(bad),
            Err(Error::InvalidFinalsLine { line: 1, .. })
        ));
        let bad =
            "73 1 2 41684.00 I  0.12x733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710\n";
        assert!(matches!(
            parse_finals2000a(bad),
            Err(Error::InvalidFinalsLine { line: 1, .. })
        ));
        let with_blanks = format!("\n\n{FINALS_SAMPLE}\n\n");
        assert_eq!(parse_finals2000a(&with_blanks).unwrap().len(), 5);
        assert!(parse_finals2000a("").unwrap().is_empty());
    }

    #[test]
    fn format_is_detected_from_content() {
        assert_eq!(detect_source(CSV_SAMPLE), EopSource::CelesTrak);
        assert_eq!(detect_source("\u{feff}DATE,MJD"), EopSource::CelesTrak);
        assert_eq!(detect_source(FINALS_SAMPLE), EopSource::IersFinals2000A);
        let t = parse_any(CSV_SAMPLE).unwrap();
        assert_eq!(t.source, EopSource::CelesTrak);
        assert_eq!(t.entries.len(), 5);
        let t = parse_any(FINALS_SAMPLE).unwrap();
        assert_eq!(t.source, EopSource::IersFinals2000A);
        assert_eq!(t.entries.len(), 5);
        // An HTML notice page is neither.
        assert!(parse_any("<!DOCTYPE html><html></html>").is_err());
    }

    /// The CSV's DX/DY are arcsec; the table (and the nutation correction)
    /// use milliarcsec, matching the finals2000A.all columns.
    #[test]
    fn csv_pole_offsets_are_converted_to_mas() {
        let rows = parse_csv(CSV_SAMPLE).unwrap();
        let r = &rows[3];
        assert_eq!(r.mjd_utc, 59464.0);
        assert!((r.dX - -0.102).abs() < 1e-9);
        assert!((r.dY - -0.150).abs() < 1e-9);
        assert!(rows[0].observed);
        assert!(!rows[4].observed);
    }

    #[test]
    fn fresher_table_wins_and_history_is_kept() {
        let finals = parse_finals2000a(FINALS_SAMPLE).unwrap();
        let csv = parse_csv(CSV_SAMPLE).unwrap();

        // Finals observed through 61300, CSV through 59464: finals win, with
        // the CSV rows before 1973-01-02 in front and nothing after.
        let t = select_table(Some(finals.clone()), Some(csv.clone())).unwrap();
        assert_eq!(t.source, EopSource::IersFinals2000A);
        let mjds: Vec<f64> = t.entries.iter().map(|e| e.mjd_utc).collect();
        assert_eq!(
            mjds,
            vec![37665.0, 41682.0, 41684.0, 48683.0, 61300.0, 61301.0, 61673.0]
        );
        assert!(mjds.windows(2).all(|w| w[0] < w[1]));

        // A CSV observed later than the finals file wins outright.
        let mut newer = csv.clone();
        newer.push(EOPEntry {
            mjd_utc: 61400.0,
            observed: true,
            ..csv[4].clone()
        });
        let t = select_table(Some(finals.clone()), Some(newer)).unwrap();
        assert_eq!(t.source, EopSource::CelesTrak);
        assert_eq!(t.entries.len(), 6);

        // Only one file present.
        assert_eq!(
            select_table(Some(finals.clone()), None).unwrap().source,
            EopSource::IersFinals2000A
        );
        assert_eq!(
            select_table(None, Some(csv.clone())).unwrap().source,
            EopSource::CelesTrak
        );
        // Empty tables count as absent.
        assert!(select_table(Some(vec![]), None).is_none());
        assert_eq!(
            select_table(Some(vec![]), Some(csv)).unwrap().source,
            EopSource::CelesTrak
        );
        assert!(select_table(None, None).is_none());
    }

    /// The on-disk loader: both files present picks the fresher observed
    /// record (with the CSV's early rows kept), one file present uses it, a
    /// corrupt file is skipped with a warning, and an empty directory is a
    /// typed error. Nothing here touches the loaded table.
    #[test]
    fn load_from_paths_picks_the_fresher_file() {
        let dir = std::env::temp_dir().join(format!("satkit_eop_load_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let finals = dir.join(FINALS2000A_FILE);
        let csv = dir.join(CELESTRAK_FILE);
        std::fs::write(&finals, FINALS_SAMPLE).unwrap();
        std::fs::write(&csv, CSV_SAMPLE).unwrap();

        let t = load_from_paths(existing(finals.clone()), existing(csv.clone())).unwrap();
        assert_eq!(t.source, EopSource::IersFinals2000A);
        assert_eq!(t.entries.first().unwrap().mjd_utc, 37665.0);
        assert_eq!(t.entries.last().unwrap().mjd_utc, 61673.0);

        let t = load_from_paths(None, existing(csv.clone())).unwrap();
        assert_eq!(t.source, EopSource::CelesTrak);
        let t = load_from_paths(existing(finals.clone()), None).unwrap();
        assert_eq!(t.source, EopSource::IersFinals2000A);
        assert_eq!(t.entries.len(), 5);

        // A corrupt IERS file does not hide the CSV.
        std::fs::write(&finals, "73 1 2 41684.00 X garbage\n").unwrap();
        let t = load_from_paths(existing(finals.clone()), existing(csv.clone())).unwrap();
        assert_eq!(t.source, EopSource::CelesTrak);

        std::fs::remove_file(&finals).unwrap();
        std::fs::remove_file(&csv).unwrap();
        assert!(load_from_paths(existing(finals), existing(csv)).is_none());
        assert!(matches!(load_from_dir(&dir), Err(Error::NoEopFile { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_names_round_trip() {
        for s in [EopSource::IersFinals2000A, EopSource::CelesTrak] {
            assert_eq!(EopSource::from_file_name(s.file_name()), Some(s));
        }
        assert_eq!(EopSource::from_file_name("SW-All.csv"), None);
    }

    #[test]
    fn test_time_bound() {
        // Future dates should return last entry's values (constant extrapolation)
        let tm = crate::Instant::from_rfc3339("2056-04-16T17:52:50.805408Z").unwrap();
        let eop = eop_from_mjd_utc(tm.as_mjd_with_scale(crate::TimeScale::UTC));
        assert!(eop.is_some());

        // Past dates before file start should return None
        let tm = crate::Instant::from_rfc3339("1950-04-16T17:52:50.805408Z").unwrap();
        let eop = eop_from_mjd_utc(tm.as_mjd_with_scale(crate::TimeScale::UTC));
        assert!(eop.is_none());
    }

    #[test]
    fn coverage_and_status() {
        let c = coverage().expect("EOP table loaded in tests");
        assert!(c.first < c.last_observed);
        assert!(c.last_observed <= c.last);
        assert!(source().is_some());

        // A well-observed historical epoch.
        let t = crate::Instant::from_rfc3339("2006-04-16T17:52:50.805408Z").unwrap();
        assert_eq!(status(&t), EopStatus::Observed);
        assert_eq!(status(&c.first), EopStatus::Observed);
        assert_eq!(status(&c.last_observed), EopStatus::Observed);
        // Past the end of the table: held constant.
        let late = c.last + crate::Duration::from_days(10.0);
        assert_eq!(status(&late), EopStatus::Extrapolated);
        assert!(eop_from_mjd_utc(late.as_mjd_utc()).is_some());
        // Predictions, when the file carries any.
        if c.last_observed < c.last {
            let mid = c.last_observed + crate::Duration::from_days(1.0);
            assert_eq!(status(&mid), EopStatus::Predicted);
        }
        // Before 1962.
        let early = crate::Instant::from_rfc3339("1950-04-16T00:00:00Z").unwrap();
        assert_eq!(status(&early), EopStatus::BeforeTable);
    }

    /// The last row of the table is inside the table: a query at exactly its
    /// epoch is not extrapolation (and must not print the out-of-range
    /// warning); anything later is.
    #[test]
    fn last_row_epoch_is_inside_table() {
        let csv = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n\
                   2024-01-01,60310,0.1,0.2,0.01,0.001,0,0,0.3,0.4,37,O\n\
                   2024-01-02,60311,0.5,0.6,0.02,0.002,0,0,0.7,0.8,37,P\n";
        let table = parse_csv(csv).unwrap();
        let last = &table[1];
        assert!(!beyond_table(last.mjd_utc, last));
        assert!(!beyond_table(last.mjd_utc - 0.5, last));
        assert!(beyond_table(last.mjd_utc + 1e-9, last));
    }

    #[test]
    fn parse_retains_data_type() {
        let text = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n\
                    2024-01-10,60319,0.119289,0.206294,0.0074355,-0.0004170,-0.112002,-0.006175,0.000248,-0.000168,37,O\n\
                    2024-01-11,60320,0.118000,0.207000,0.0075000,-0.0004000,-0.112000,-0.006100,0.000240,-0.000160,37,P\n";
        let rows = parse_csv(text).unwrap();
        assert!(rows[0].observed);
        assert!(!rows[1].observed);
    }

    /// Check value against the IERS/CelesTrak values for 2021-09-07. The two
    /// sources differ at the few-µs / 0.1 mas level (Bulletin A rapid vs the
    /// final series), so the tolerances accept either; LOD differs more
    /// between them (rapid vs final analysis) and is checked loosely.
    #[test]
    fn checkval() {
        let tm = crate::Instant::from_rfc3339("2006-04-16T17:52:50.805408Z").unwrap();
        let v: Option<[f64; 6]> = eop_from_mjd_utc(tm.as_mjd_utc());
        assert!(v.is_some());

        let v = eop_from_mjd_utc(59464.00).unwrap();
        const TRUTH: [f64; 3] = [-0.1145667, 0.241155, 0.317274];
        for (a, b) in v.iter().zip(TRUTH.iter()) {
            assert!(((a - b) / b).abs() < 1.0e-3, "{a} vs {b}");
        }
        assert!((v[3] - -0.0002255).abs() < 5.0e-5, "LOD {}", v[3]);
    }

    /// Interpolation between two table rows is linear in every column.
    #[test]
    fn checkinterp() {
        let mjd0: f64 = 57909.00;
        let v0 = eop_from_mjd_utc(mjd0).unwrap();
        let v1 = eop_from_mjd_utc(mjd0 + 1.0).unwrap();
        for x in 0..101 {
            let dt: f64 = x as f64 / 100.0;
            let vt = eop_from_mjd_utc(mjd0 + dt).unwrap();
            for (v, (a, b)) in vt.iter().zip(v0.iter().zip(v1.iter())) {
                let vtest = (1.0 - dt) * a + dt * b;
                assert!(
                    (v - vtest).abs() < 1.0e-9 * (1.0 + v.abs()),
                    "{v} vs {vtest}"
                );
            }
        }
    }
}
