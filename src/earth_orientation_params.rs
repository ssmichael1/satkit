//! Earth Orientation Parameters (EOP) module
//!
//! This module provides access to Earth Orientation Parameters (EOP) data,
//! which are essential for accurate satellite orbit predictions and transformations
//! between different reference frames.
//!
//! The EOP data includes parameters such as polar motion, UT1-UTC, and length of day (LOD),
//! which are crucial for precise calculations in satellite tracking and navigation.
//!
//! # Source
//!
//! The table is the IERS Rapid Service / Prediction Centre's Bulletin A
//! combined file, **`finals2000A.all`**: observed values from 1973-01-02
//! plus about a year of predictions, updated daily. The refresh fetches it
//! from the USNO and IERS mirrors. The Bulletin A columns are used
//! throughout (never the Bulletin B columns, which end earlier and would
//! introduce a splice).
//!
//! Before the table's first row there is no EOP: [`get`] returns `None`,
//! the frame transforms use zeros (so UT1 = UTC, as in ERFA), and a
//! one-time warning is printed.
//!
//! When the file has copies in more than one search directory, the copy
//! with the latest last observed row is read, so a stale copy in an earlier
//! directory (an `add_search_dir` directory, a system-wide copy) does not
//! shadow a fresh download in the write location. A copy that cannot be
//! parsed is skipped with a warning.
//!
//! The download URLs live in the embedded data manifest
//! (`data/manifest.json`, `eop` section).
//!
//! The file is published once a day, so [`refresh_into`] leaves a copy
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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

/// File name of the IERS Bulletin A combined file, the EOP table.
pub const FINALS2000A_FILE: &str = "finals2000A.all";

/// Errors produced by the
/// [`earth_orientation_params`](crate::earth_orientation_params) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A `finals2000A.all` line could not be parsed (a flag other than `I`/`P`,
    /// or a numeric column that is neither blank nor a number).
    #[error("Invalid finals2000A.all line {line}: {reason}")]
    InvalidFinalsLine { line: usize, reason: String },

    /// The data given is CelesTrak's `EOP-All.csv`, which is not read:
    /// the EOP table is IERS `finals2000A.all`.
    #[error(
        "CelesTrak EOP-All.csv is not supported: Earth orientation is read from IERS \
         {FINALS2000A_FILE} only. Run `satkit::utils::update_datafiles()` \
         (Python: `satkit.utils.update_datafiles()`) to download it"
    )]
    UnsupportedCsv,

    /// `finals2000A.all` could not be read from the data directory (after a
    /// refresh that reported success, or when loading a directory
    /// explicitly).
    #[error("No Earth orientation file ({FINALS2000A_FILE}) readable in {dir}")]
    NoEopFile { dir: String },

    /// The data directory cannot receive an updated EOP file: read-only
    /// filesystem, no write permission, or owned by another user.
    #[error(
        "Data directory {path} is not writable ({reason}). Set the environment variable \
         SATKIT_DATA to a writable directory and restart, or call set_datadir \
         (Python: satkit.utils.set_datadir) with one"
    )]
    DataDirReadOnly { path: String, reason: String },

    /// Bytes passed to [`init_from_bytes`] were not valid UTF-8 — the EOP
    /// file is text.
    #[error("EOP byte buffer is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the
/// `earth_orientation_params` module.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
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
    /// `true` for an observed (`I`) row, `false` for a predicted (`P`) row.
    observed: bool,
}

/// Where a given epoch falls relative to the loaded EOP table.
///
/// Returned by [`status`]; see [`coverage`] for the table bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopStatus {
    /// Inside the table, on or before the last observed row.
    Observed,
    /// Inside the table, after the last observed row: IERS predictions.
    /// When that row is more than 30 days before the current date (the
    /// file has not been refreshed), the first lookup here prints a
    /// one-time warning.
    Predicted,
    /// After the last row of the table: the last row's values are held
    /// constant. Accuracy degrades with distance from the table end
    /// (polar motion drifts ~0.1″ and UT1−UTC by ~10 ms over a few
    /// months) — refresh the data with
    /// [`update`] / `satkit::utils::update_datafiles()`.
    Extrapolated,
    /// Before the first row of the table (1973-01-02 for
    /// `finals2000A.all`): no EOP available, [`get`] returns `None` and the
    /// frame transforms use zeros (UT1 = UTC).
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
/// filled). Rows whose flag is blank (the tail of the file) carry only a
/// date and are not table rows.
///
/// A line cut short — the last line of a truncated transfer — is an error,
/// not a shorter table: a row with an `I`/`P` flag must hold the whole
/// UT1−UTC field (at least 68 columns, with its decimal point in column
/// 61), a row with a nutation flag must hold dX and dY, no line may end
/// inside a numeric field that is read, and a line without a flag must be a
/// complete date and nothing else.
///
/// Rows are sorted by date. A date that appears more than once (a repeated
/// line, two copies of the file concatenated) keeps a single row: the
/// observed (`I`) one over a predicted (`P`) one, and among rows of the same
/// kind the one later in the file (in a concatenation, the newer copy).
///
/// A blank LOD is filled, after sorting and de-duplication, by the finite
/// difference of UT1−UTC across the following day in the final table (the
/// excess length of day is −d(UT1−UTC)/dt); a blank dX/dY is zero (no
/// correction to the IAU 2000A model).
///
/// Malformed input is an [`Error::InvalidFinalsLine`], never a panic: that
/// includes non-finite numbers and an MJD outside 0–100000.
///
/// CelesTrak's `EOP-All.csv` (recognised by its `DATE,` header) is rejected
/// with [`Error::UnsupportedCsv`].
fn parse_finals2000a(text: &str) -> Result<Vec<EOPEntry>> {
    // A UTF-8 byte-order mark (a Windows editor) is not part of the data.
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if text.starts_with("DATE,") {
        return Err(Error::UnsupportedCsv);
    }
    let invalid = |line: usize, reason: String| Error::InvalidFinalsLine { line, reason };
    // Numbers are right-justified in their fields and trailing blanks are
    // trimmed, so a line that ends strictly inside a field has lost the end
    // of that number (a truncated transfer: "-0.1478001" cut to "-0.").
    fn field<'a>(
        line: &'a str,
        lineno: usize,
        start: usize,
        len: usize,
        what: &str,
    ) -> Result<&'a str> {
        let n = line.len();
        if n > start - 1 && n < start - 1 + len {
            return Err(Error::InvalidFinalsLine {
                line: lineno,
                reason: format!(
                    "line ends at column {n}, inside the {what} field (columns {start}-{}): \
                     the file is truncated",
                    start + len - 1
                ),
            });
        }
        Ok(cols(line, start, len))
    }
    // Rust's float parser also accepts "NaN", "inf" and exponents; none of
    // them is a finals2000A.all value, and a non-finite number would reach
    // the sort and the time conversions.
    let num = |line: usize, s: &str, what: &str| -> Result<f64> {
        let v = s
            .trim()
            .parse::<f64>()
            .map_err(|e| invalid(line, format!("{what} {s:?}: {e}")))?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(invalid(line, format!("{what} {s:?} is not finite")))
        }
    };
    let opt = |line: usize, s: &str, what: &str| -> Result<Option<f64>> {
        if s.trim().is_empty() {
            Ok(None)
        } else {
            num(line, s, what).map(Some)
        }
    };

    // Each row keeps its LOD as read (`None` where the column is blank) and
    // its line number until the table is sorted and de-duplicated; the blank
    // LODs are filled only then, from the neighbours in the final table.
    struct Parsed {
        entry: EOPEntry,
        lod: Option<f64>,
        lineno: usize,
    }
    let mut rows: Vec<Parsed> = Vec::new();
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
            // The file ends with rows that carry a complete date (columns
            // 1-15) and nothing else; anything shorter is a cut line.
            " " | "" => {
                if line.len() < 15 {
                    return Err(invalid(
                        lineno,
                        format!("line ends at column {}: the file is truncated", line.len()),
                    ));
                }
                if line.len() > 16 {
                    return Err(invalid(lineno, "values without an I/P flag".into()));
                }
                num(lineno, cols(line, 8, 8), "MJD")?;
                continue;
            }
            other => return Err(invalid(lineno, format!("polar motion flag {other:?}"))),
        };
        let mjd_utc = num(lineno, cols(line, 8, 8), "MJD")?;
        // MJD 0 is 1858-11-17 and 100000 is 2132-09-01: anything outside is
        // not a date this file can hold.
        if !(0.0..=100_000.0).contains(&mjd_utc) {
            return Err(invalid(lineno, format!("MJD {mjd_utc} out of range")));
        }
        // Every data row carries UT1−UTC (F10.7, |UT1−UTC| < 0.9 s, so the
        // decimal point is always in column 61). A row that stops before its
        // end, or has it misplaced, is truncated or not this format.
        if line.len() < 68 || cols(line, 61, 1) != "." {
            return Err(invalid(
                lineno,
                format!(
                    "no complete UT1-UTC value in columns 59-68 (line has {} columns): \
                     truncated or misaligned",
                    line.len()
                ),
            ));
        }
        let xp = num(lineno, cols(line, 19, 9), "x pole")?;
        let yp = num(lineno, cols(line, 38, 9), "y pole")?;
        let dut1 = num(lineno, cols(line, 59, 10), "UT1-UTC")?;
        // A nutation flag promises both dX and dY; a line that stops before
        // them lost them.
        if cols(line, 96, 1).trim() != "" && line.len() < 125 {
            return Err(invalid(
                lineno,
                format!(
                    "nutation flag set but the line ends at column {}, before dX/dY \
                     (columns 98-125): the file is truncated",
                    line.len()
                ),
            ));
        }
        let lod = opt(lineno, field(line, lineno, 80, 7, "LOD")?, "LOD")?.map(|ms| ms * 1.0e-3);
        let dx = opt(lineno, field(line, lineno, 98, 9, "dX")?, "dX")?.unwrap_or(0.0);
        let dy = opt(lineno, field(line, lineno, 117, 9, "dY")?, "dY")?.unwrap_or(0.0);
        rows.push(Parsed {
            entry: EOPEntry {
                mjd_utc,
                xp,
                yp,
                dut1,
                lod: 0.0,
                dX: dx,
                dY: dy,
                observed,
            },
            lod,
            lineno,
        });
    }
    // Sort by date; for a date that appears more than once (a line repeated,
    // two files concatenated) keep one row: an observed row over a
    // predicted one, and otherwise the one later in the file. The sort puts
    // that row first in each run of equal dates and `dedup_by` keeps the
    // first.
    rows.sort_by(|a, b| {
        a.entry
            .mjd_utc
            .total_cmp(&b.entry.mjd_utc)
            .then(b.entry.observed.cmp(&a.entry.observed))
            .then(b.lineno.cmp(&a.lineno))
    });
    rows.dedup_by(|later, kept| later.entry.mjd_utc == kept.entry.mjd_utc);
    let lods: Vec<Option<f64>> = rows.iter().map(|r| r.lod).collect();
    let mut table: Vec<EOPEntry> = rows.into_iter().map(|r| r.entry).collect();
    for (i, lod) in lods.into_iter().enumerate() {
        table[i].lod = match lod {
            Some(v) => v,
            None => lod_from_neighbours(&table, i),
        };
    }
    Ok(table)
}

/// Excess length of day at row `i` of a sorted, de-duplicated table from
/// UT1−UTC: the forward difference where there is a next row, else the
/// backward one, else zero. A jump of about a second is a leap second, not a
/// rate, and gives zero.
fn lod_from_neighbours(table: &[EOPEntry], i: usize) -> f64 {
    let (a, b) = if i + 1 < table.len() {
        (i, i + 1)
    } else if i > 0 {
        (i - 1, i)
    } else {
        return 0.0;
    };
    let d = table[b].dut1 - table[a].dut1;
    let dt = table[b].mjd_utc - table[a].mjd_utc;
    if dt > 0.0 && d.abs() < 0.5 {
        -d / dt
    } else {
        0.0
    }
}

/// Check that the file at `path` is a parsable `finals2000A.all`, without
/// touching the loaded table.
///
/// Used by the downloader to reject a response that is not the file it
/// claims to be — a proxy notice page served with `200 OK`, or a truncated
/// transfer — before it replaces a good table on disk. The check is the real
/// parser, so anything that would later fail to load fails here instead,
/// while the previous file is still in place; on top of that the table must
/// end in predicted rows, as every published `finals2000A.all` does.
///
/// A transfer cut inside a line is rejected by the parser (see
/// [`parse_finals2000a`]); one cut at a line boundary in the observed part
/// has no predicted rows and is rejected here. A cut at a line boundary in
/// the predicted part is not detected: every row read is complete, the
/// predictions just end earlier.
///
/// A parser panic (a bug; malformed input is an error) is reported as a
/// rejection rather than unwinding through the refresh thread, which would
/// leave the partial download behind.
// Called by the downloader; the parser tests use it in every feature set.
#[cfg_attr(not(feature = "download"), allow(dead_code))]
pub(crate) fn validate_file(path: &Path) -> std::result::Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    match std::panic::catch_unwind(|| parse_finals2000a(&text)) {
        Ok(Ok(t)) if t.is_empty() => Err("the file holds no EOP rows".to_string()),
        Ok(Ok(t)) if t.last().is_some_and(|e| e.observed) => Err(
            "the file has no predicted rows after the last observed one: it is truncated"
                .to_string(),
        ),
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(format!("not a parsable EOP file ({e})")),
        Err(_) => Err("the EOP parser failed on this file".to_string()),
    }
}

/// MJD of the last observed row, or −∞ for a table with none.
fn last_observed_mjd(rows: &[EOPEntry]) -> f64 {
    rows.iter()
        .rev()
        .find(|e| e.observed)
        .map_or(f64::NEG_INFINITY, |e| e.mjd_utc)
}

/// Parse the file at `path`, or `None` (with a warning) if it is absent,
/// unreadable or holds no rows.
fn read_table(path: Option<&Path>) -> Option<Vec<EOPEntry>> {
    let p = path?;
    let parsed = std::fs::read_to_string(p)
        .map_err(Error::from)
        .and_then(|t| parse_finals2000a(&t));
    match parsed {
        Ok(t) if !t.is_empty() => Some(t),
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

/// Load the EOP table from `finals2000A.all` in `dir`, replacing any loaded
/// table. Used after a refresh into an explicit directory; the lazy default
/// load searches all data directories instead.
pub fn load_from_dir(dir: &Path) -> Result<()> {
    let table = read_table(existing(dir.join(FINALS2000A_FILE)).as_deref()).ok_or_else(|| {
        Error::NoEopFile {
            dir: dir.display().to_string(),
        }
    })?;
    EOP.set(table);
    Ok(())
}

/// The freshest readable copy of `finals2000A.all` in `dirs`, parsed, with
/// its path: the one whose last observed row is latest, ties keeping search
/// order. A copy that cannot be read or parsed is skipped with a warning
/// (see [`read_table`]), so one corrupt copy anywhere in the search path
/// never stops a good one from loading.
fn freshest_table(dirs: &[PathBuf]) -> Option<(PathBuf, Vec<EOPEntry>)> {
    let mut best: Option<(PathBuf, Vec<EOPEntry>)> = None;
    for p in datadir::find_all_in(dirs, FINALS2000A_FILE) {
        let Some(t) = read_table(Some(&p)) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(_, b)| last_observed_mjd(&t) > last_observed_mjd(b))
        {
            best = Some((p, t));
        }
    }
    best
}

/// Lazy default load: the freshest readable copy of `finals2000A.all`
/// across the data search directories; when there is no copy at all,
/// refresh into the write location first.
fn load_default() -> Result<Vec<EOPEntry>> {
    let dirs = datadir::search_dirs();
    let copies = datadir::find_all_in(&dirs, FINALS2000A_FILE);
    if !copies.is_empty() {
        return freshest_table(&dirs)
            .map(|(_, t)| t)
            .ok_or_else(|| Error::NoEopFile {
                dir: copies
                    .iter()
                    .filter_map(|p| p.parent())
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
    }
    let dir = datadir::datadir()?;
    refresh_into(&dir, false)?;
    read_table(existing(dir.join(FINALS2000A_FILE)).as_deref()).ok_or_else(|| Error::NoEopFile {
        dir: dir.display().to_string(),
    })
}

/// What a refresh settled on: the URL it was checked against, and whether
/// that cost a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshOutcome {
    /// The URL `finals2000A.all` was fetched from, or would have been: for
    /// a copy still inside its publication cadence no request was made.
    pub url: String,
    /// Whether the file was transferred, answered `304`, or not requested.
    pub fetch: download::RefreshOutcome,
}

/// Bring `finals2000A.all` in `dir` up to date from the mirrors the
/// embedded data manifest lists (USNO, then the IERS data centre).
///
/// Each URL goes through [`refresh_file`](crate::utils::refresh_file), so a
/// copy fetched within the last 24 h is reported current without a request
/// and an older one costs a conditional GET (`304` when unchanged); `force`
/// transfers the file unconditionally. The first mirror that answers is
/// kept (an HTML notice page, or a transfer cut inside a line or before the
/// predicted rows, is rejected and the next URL tried; a cut exactly at a
/// line boundary inside the predictions cannot be seen, but leaves only
/// complete rows). Does not change the loaded table — call
/// [`load_from_dir`] or [`update`] for that.
///
/// Fails with [`download::Error::RefreshOffline`] under offline mode without
/// any network I/O (it says whether a copy exists, which is left as it is,
/// and lists the mirrors for fetching the file by hand), and with [`download::Error::AllSourcesFailed`] listing every
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
    let urls: Vec<&String> = sources
        .iter()
        .filter(|s| s.name == FINALS2000A_FILE)
        .flat_map(|s| &s.urls)
        .collect();
    if let Some(reason) = download::offline_reason() {
        // The copy a refresh would replace, else whichever copy the search
        // directories hold: the message names it as left unchanged, not as
        // the one in use (the default load picks the freshest copy).
        let copy = existing(dir.join(FINALS2000A_FILE))
            .or_else(|| datadir::find_all(FINALS2000A_FILE).into_iter().next());
        return Err(download::Error::RefreshOffline {
            name: format!("Earth orientation parameters ({FINALS2000A_FILE})"),
            reason,
            existing: copy.map(|p| p.display().to_string()),
            urls: urls.into_iter().cloned().collect(),
        });
    }
    let mut attempts: Vec<String> = Vec::new();
    for url in urls {
        match refresh_file(url, dir, force) {
            Ok(fetch) => {
                return Ok(RefreshOutcome {
                    url: url.clone(),
                    fetch,
                })
            }
            Err(e) => attempts.push(format!("{url}: {e}")),
        }
    }
    Err(download::Error::AllSourcesFailed {
        name: "Earth orientation parameters".to_string(),
        attempts,
        hint: None,
    })
}

/// MJD of the first row of `finals2000A.all` (1973-01-02).
const FINALS_FIRST_MJD: f64 = 41684.0;

/// The advice line (with its newline, or empty) of the "too early" warning
/// for a table starting at `first_mjd`.
///
/// A table that starts where `finals2000A.all` does gets the note that there
/// is no EOP data before it; any other start (a table loaded with
/// [`init_from_path`] / [`init_from_bytes`], a truncated file) gets no
/// advice, since the warning already says where the loaded table starts.
fn too_early_advice(first_mjd: f64) -> &'static str {
    if first_mjd == FINALS_FIRST_MJD {
        "finals2000A.all has no EOP data before 1973-01-02; refreshing the data files does \
         not change this.\n"
    } else {
        ""
    }
}

/// `true` when `mjd_utc` lies strictly after the last table row; a query at
/// exactly the last epoch is inside the table.
fn beyond_table(mjd_utc: f64, last: &EOPEntry) -> bool {
    mjd_utc > last.mjd_utc
}

static WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static EXTRAP_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static NOT_LOADED_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);
static STALE_PREDICTION_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// Days after which the observed part of the table counts as stale: a query
/// in the predicted range of a table whose last observed row is older than
/// this (by the wall clock) gets a one-time warning. IERS updates the file
/// daily; month-old predictions are already off by ~0.1″ and ~10–20 ms.
const STALE_OBSERVED_DAYS: f64 = 30.0;

/// For a query at `mjd_utc` after the last observed row of `eop`, when that
/// row is more than [`STALE_OBSERVED_DAYS`] before `now_mjd`: the row's MJD
/// and its age in days. Otherwise (a query on observed data, a recent
/// table, a table with no observed rows) `None`.
///
/// The observed rows precede the predicted ones, so the boundary is a
/// binary search: this runs on the lookup path.
fn stale_prediction_age(eop: &[EOPEntry], mjd_utc: f64, now_mjd: f64) -> Option<(f64, f64)> {
    let n_observed = eop.partition_point(|e| e.observed);
    let last_observed = eop.get(n_observed.checked_sub(1)?)?.mjd_utc;
    let age = now_mjd - last_observed;
    (mjd_utc > last_observed && age > STALE_OBSERVED_DAYS).then_some((last_observed, age))
}

/// Module-scope refreshable singleton. The lazy default load (best-effort,
/// silent on failure) runs at most once; [`init_from_bytes`] /
/// [`init_from_path`] / [`update`] replace any current contents.
static EOP: RefreshableSingleton<Vec<EOPEntry>> = RefreshableSingleton::new();

/// Best-effort default load on first read. Failures are silent — if EOP
/// can't be loaded, the singleton stays empty and queries fall through
/// to the "no data" branch.
fn ensure_default_loaded() {
    EOP.ensure_default_loaded(|| load_default().ok());
}

/// Initialize the EOP singleton from an in-memory byte buffer.
///
/// The bytes must be the text of an IERS `finals2000A.all` file (UTF-8);
/// CelesTrak's `EOP-All.csv` is rejected with [`Error::UnsupportedCsv`].
/// Replaces any previously loaded data — IERS publishes new EOP daily and
/// refresh-in-place is the intended model.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    EOP.set(parse_finals2000a(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the EOP singleton from a `finals2000A.all` file at `path`.
///
/// Same semantics as [`init_from_bytes`]; always replaces.
pub fn init_from_path(path: &Path) -> Result<()> {
    EOP.set(parse_finals2000a(&std::fs::read_to_string(path)?)?);
    Ok(())
}

///
/// Disable the warnings about out-of-range or missing EOP data.
///
/// Four one-time warnings exist: epoch before the table, epoch after the
/// table (values held constant), epoch in the predictions of a table whose
/// observed data ended more than 30 days ago (stale predictions), and no
/// table loaded at all (zeros used). Each is shown at most once per
/// process; call this to suppress all of them.
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
    STALE_PREDICTION_WARNING_SHOWN.store(true, Ordering::Relaxed);
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
    let eop = guard.as_ref()?;
    let first = eop.first()?;
    let last = eop.last()?;
    let last_observed = eop.iter().rev().find(|e| e.observed).unwrap_or(first);
    Some(EopCoverage {
        first: Instant::from_mjd_utc(first.mjd_utc),
        last_observed: Instant::from_mjd_utc(last_observed.mjd_utc),
        last: Instant::from_mjd_utc(last.mjd_utc),
    })
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
    let Some(eop) = guard.as_deref().filter(|e| !e.is_empty()) else {
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
/// date (see [`refresh_into`] for the mirrors and the once-a-day
/// cadence; a copy fetched within the last 24 h is not re-requested), and
/// load it.
pub fn update() -> Result<()> {
    let d = datadir::datadir()?;
    datadir::check_writable(&d, |path, reason| Error::DataDirReadOnly { path, reason })?;
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
    let Some(eop) = guard.as_deref().filter(|e| !e.is_empty()) else {
        if !NOT_LOADED_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: no Earth Orientation Parameters (EOP) table is loaded; polar motion, \
                 UT1-UTC and nutation corrections are being treated as zero, which biases \
                 Earth-fixed frame transforms by up to ~12 arcsec (UT1-UTC up to 0.9 s \
                 plus polar motion up to ~0.5 arcsec), i.e. hundreds of metres at LEO.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download finals2000A.all, or set SATKIT_DATA to a directory containing it.\n\
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()` \
                 (Python: `satkit.frametransform.disable_eop_time_warning()`)"
            );
        }
        return None;
    };

    // Binary search: find first entry with mjd_utc > query (O(log n) vs O(n) linear scan)
    let idx = eop.partition_point(|x| x.mjd_utc <= mjd_utc);

    if idx == 0 {
        if !WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: EOP data not available for MJD UTC = {mjd_utc} (too early): the \
                 loaded table starts at {} (MJD {}), and polar motion, UT1-UTC and nutation \
                 corrections are treated as zero (UT1 = UTC) before it.\n\
                 {}\
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()` \
                 (Python: `satkit.frametransform.disable_eop_time_warning()`)",
                Instant::from_mjd_utc(eop[0].mjd_utc),
                eop[0].mjd_utc,
                too_early_advice(eop[0].mjd_utc)
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
                 To disable: `satkit::earth_orientation_params::disable_eop_time_warning()` \
                 (Python: `satkit.frametransform.disable_eop_time_warning()`)",
                Instant::from_mjd_utc(last.mjd_utc),
                last.mjd_utc
            );
        }
        return Some([last.dut1, last.xp, last.yp, last.lod, last.dX, last.dY]);
    }

    // Inside the predictions of a file that has not been refreshed for a
    // month or more: the values are old forecasts, not measurements. The
    // wall clock is read only for a query past an observed row, until the
    // warning has been shown.
    if !eop[idx].observed && !STALE_PREDICTION_WARNING_SHOWN.load(Ordering::Relaxed) {
        let now = Instant::now().as_mjd_utc();
        if let Some((last_observed, age)) = stale_prediction_age(eop, mjd_utc, now) {
            if !STALE_PREDICTION_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "Warning: EOP for MJD UTC = {mjd_utc} comes from IERS predictions made \
                     {age:.0} days ago: the loaded table's observed data ends at {} \
                     (MJD {last_observed}), and the file has not been refreshed since. \
                     Months-old predictions are off by ~0.3-0.6 arcsec in UT1 \
                     (10-20 m at LEO).\n\
                     Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                     to download the current Earth orientation file.\n\
                     To disable: `satkit::earth_orientation_params::disable_eop_time_warning()` \
                     (Python: `satkit.frametransform.disable_eop_time_warning()`)",
                    Instant::from_mjd_utc(last_observed)
                );
            }
        }
    }

    // Linear interpolation between bracketing entries
    let v0 = &eop[idx - 1];
    let v1 = &eop[idx];
    let g1 = (mjd_utc - v0.mjd_utc) / (v1.mjd_utc - v0.mjd_utc);
    let g0 = 1.0 - g1;
    // UT1 − UTC jumps wherever TAI − UTC does: by a leap second at 00:00 UTC
    // of a row, and before 1972 also by the fractional UTC steps and the
    // daily drift of TAI − UTC. Interpolate the continuous UT1 − TAI instead,
    // i.e. re-reference both rows to the TAI − UTC at the query:
    // UT1 − UTC = UT1 − TAI (interpolated) + TAI − UTC (query). After 1972,
    // within a day, the corrections are exactly zero except for the row past
    // a leap second.
    let dat = crate::time::tai_minus_utc_at_mjd_utc(mjd_utc);
    let dut1_0 = v0.dut1 + (dat - crate::time::tai_minus_utc_at_mjd_utc(v0.mjd_utc));
    let dut1_1 = v1.dut1 + (dat - crate::time::tai_minus_utc_at_mjd_utc(v1.mjd_utc));
    Some([
        g0.mul_add(dut1_0, g1 * dut1_1),
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

    /// The first lines of a CelesTrak `EOP-All.csv`, which is not read.
    const CSV_SAMPLE: &str = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n\
        1962-01-01,37665,-0.012700,0.213000,0.0326338,0.0017230,0.064261,0.006067,0.000000,0.000000,2,O\n";

    /// Check that data is loaded
    #[test]
    fn loaded() {
        ensure_default_loaded();
        let guard = EOP.read();
        let eop = guard
            .as_ref()
            .expect("default EOP load should succeed in tests");
        assert!(eop[0].mjd_utc >= 0.0);
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
        // An HTML notice page is not an EOP file.
        assert!(parse_finals2000a("<!DOCTYPE html><html></html>").is_err());
    }

    /// `text`'s lines in the given order, newline-terminated.
    fn reorder(text: &str, order: &[usize]) -> String {
        let lines: Vec<&str> = text.lines().collect();
        order.iter().map(|&i| format!("{}\n", lines[i])).collect()
    }

    /// A repeated line, the file concatenated with itself, and the rows in
    /// reverse or shuffled order all parse to the same table as the file
    /// itself — including the LOD filled for the blank-LOD rows, which is
    /// computed from the neighbours in the sorted table. (A repeated line
    /// used to panic with an out-of-bounds index; unsorted rows used to get
    /// LODs filled on the wrong rows.)
    #[test]
    fn parse_finals_duplicates_and_order_do_not_matter() {
        let reference = parse_finals2000a(FINALS_SAMPLE).unwrap();
        assert_eq!(reference.len(), 5);
        let variants = [
            // Line 3 (blank LOD) repeated, and line 5 (blank LOD, the last row).
            reorder(FINALS_SAMPLE, &[0, 1, 2, 2, 3, 4, 4, 5]),
            format!("{FINALS_SAMPLE}{FINALS_SAMPLE}"),
            reorder(FINALS_SAMPLE, &[5, 4, 3, 2, 1, 0]),
            reorder(FINALS_SAMPLE, &[3, 0, 4, 2, 5, 1]),
            reorder(FINALS_SAMPLE, &[4, 2, 0, 3, 1, 2, 4]),
        ];
        for text in &variants {
            let t = parse_finals2000a(text).unwrap();
            assert_eq!(t, reference, "{text}");
        }
        // The validator accepts them too.
        let dir = std::env::temp_dir().join(format!("satkit_eop_dup_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(FINALS2000A_FILE);
        std::fs::write(&p, &variants[0]).unwrap();
        assert_eq!(validate_file(&p), Ok(()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// For a date that appears more than once the observed row wins over a
    /// predicted one wherever they are in the file, and among rows of the
    /// same kind the later one in the file wins.
    #[test]
    fn parse_finals_dedup_rule() {
        let lines: Vec<&str> = FINALS_SAMPLE.lines().collect();
        let observed = lines[2]; // 61300, observed
        let predicted = observed.replacen(" I  ", " P  ", 1);
        assert_eq!(cols(&predicted, 17, 1), "P");
        for text in [
            format!("{observed}\n{predicted}\n"),
            format!("{predicted}\n{observed}\n"),
        ] {
            let t = parse_finals2000a(&text).unwrap();
            assert_eq!(t.len(), 1);
            assert!(t[0].observed, "{text}");
        }
        // Two observed rows for one date (a revised value): the later wins.
        let revised = observed.replacen("0.190054", "0.190999", 1);
        let t = parse_finals2000a(&format!("{observed}\n{revised}\n")).unwrap();
        assert_eq!(t.len(), 1);
        assert!((t[0].xp - 0.190999).abs() < 1e-12);
        let t = parse_finals2000a(&format!("{revised}\n{observed}\n")).unwrap();
        assert!((t[0].xp - 0.190054).abs() < 1e-12);
    }

    /// The same on the real file: concatenated with itself, or with every
    /// row reversed, it parses to the table the file itself gives (the
    /// duplicated file used to panic with "index out of bounds: the len is
    /// 19997 but the index is 19997").
    #[test]
    fn parse_real_finals_duplicated_and_reversed() {
        let path = datadir::find_all(FINALS2000A_FILE)
            .into_iter()
            .next()
            .expect("finals2000A.all present in tests");
        let text = std::fs::read_to_string(path).unwrap();
        let reference = parse_finals2000a(&text).unwrap();
        assert!(reference.len() > 19_000);
        let doubled = parse_finals2000a(&format!("{text}{text}")).unwrap();
        assert_eq!(doubled, reference);
        let reversed: String = text.lines().rev().map(|l| format!("{l}\n")).collect();
        assert_eq!(parse_finals2000a(&reversed).unwrap(), reference);
        // One line repeated in the middle of the file.
        let mut lines: Vec<&str> = text.lines().collect();
        lines.insert(1000, lines[999]);
        let dup: String = lines.iter().map(|l| format!("{l}\n")).collect();
        assert_eq!(parse_finals2000a(&dup).unwrap(), reference);
    }

    /// Values Rust's float parser accepts but no EOP file holds (NaN,
    /// infinities, an absurd MJD) are errors, not table rows.
    #[test]
    fn parse_finals_rejects_non_finite_and_out_of_range() {
        let good = FINALS_SAMPLE.lines().nth(1).unwrap();
        for (from, to) in [
            ("48683.00", "     NaN"),
            ("48683.00", "     inf"),
            ("48683.00", "   1e300"),
            ("48683.00", "-4868.00"),
            (" 0.006416", "      NaN"),
            ("-0.2719443", "      -inf"),
        ] {
            let bad = good.replacen(from, to, 1);
            assert_ne!(bad, good);
            assert!(
                matches!(
                    parse_finals2000a(&bad),
                    Err(Error::InvalidFinalsLine { line: 1, .. })
                ),
                "{bad}"
            );
        }
    }

    /// A copy in an earlier search directory that is duplicated (formerly a
    /// parser panic inside the default load, which poisoned the singleton
    /// for the rest of the process) or corrupt does not stop the default
    /// load: the duplicated copy loads as the table it holds, the corrupt
    /// one is skipped with a warning, and a singleton loaded through the
    /// same path answers every later read.
    #[test]
    fn bad_copy_in_earlier_search_dir_does_not_poison_load() {
        let root = std::env::temp_dir().join(format!("satkit_eop_poison_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (early, late) = (root.join("early"), root.join("late"));
        std::fs::create_dir_all(&early).unwrap();
        std::fs::create_dir_all(&late).unwrap();
        // The early copy is the sample with lines repeated and shuffled but
        // cut after its last observed row, the late one the full sample.
        std::fs::write(
            early.join(FINALS2000A_FILE),
            reorder(FINALS_SAMPLE, &[2, 1, 0, 1, 2]),
        )
        .unwrap();
        std::fs::write(late.join(FINALS2000A_FILE), FINALS_SAMPLE).unwrap();
        let dirs = vec![early.clone(), late.clone()];
        let reference = parse_finals2000a(FINALS_SAMPLE).unwrap();

        // Both have the same last observed row: search order wins.
        let (p, t) = freshest_table(&dirs).unwrap();
        assert_eq!(p, early.join(FINALS2000A_FILE));
        assert_eq!(t.len(), 3);
        assert_eq!(t[..2], reference[..2]);

        let single: RefreshableSingleton<Vec<EOPEntry>> = RefreshableSingleton::new();
        single.ensure_default_loaded(|| freshest_table(&dirs).map(|(_, t)| t));
        for _ in 0..3 {
            single.ensure_default_loaded(|| unreachable!("the default load runs once"));
            assert_eq!(single.read().as_ref().map(Vec::len), Some(3));
        }

        // A corrupt early copy is skipped for the good late one.
        std::fs::write(early.join(FINALS2000A_FILE), "73 1 2 41684.00 X garbage\n").unwrap();
        let (p, t) = freshest_table(&dirs).unwrap();
        assert_eq!(p, late.join(FINALS2000A_FILE));
        assert_eq!(t, reference);
        // Nothing readable at all: no table, and no panic.
        std::fs::write(late.join(FINALS2000A_FILE), "<html></html>\n").unwrap();
        assert!(freshest_table(&dirs).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A table whose observed data ends months before today (the real file
    /// with every row after a past date re-flagged as predicted) makes a
    /// query in its predictions stale; a query on observed data, or a table
    /// observed up to a recent date, is not.
    #[test]
    fn stale_predictions_are_detected() {
        let path = datadir::find_all(FINALS2000A_FILE)
            .into_iter()
            .next()
            .expect("finals2000A.all present in tests");
        let mut t = parse_finals2000a(&std::fs::read_to_string(path).unwrap()).unwrap();
        let now = Instant::now().as_mjd_utc();
        let cutoff = (now - 200.0).floor();
        for r in t.iter_mut().filter(|r| r.mjd_utc > cutoff) {
            r.observed = false;
        }
        // Six months of predictions made 200 days ago.
        let (last, age) = stale_prediction_age(&t, now, now).expect("stale");
        assert_eq!(last, cutoff);
        assert!((age - (now - cutoff)).abs() < 1e-9, "{age}");
        assert!(stale_prediction_age(&t, cutoff + 10.5, now).is_some());
        // Observed data is never stale, however old the file.
        assert!(stale_prediction_age(&t, cutoff - 10.0, now).is_none());
        assert!(stale_prediction_age(&t, cutoff, now).is_none());
        // The same table seen from shortly after the cutoff: current.
        assert!(stale_prediction_age(&t, cutoff + 10.5, cutoff + 20.0).is_none());
        assert!(stale_prediction_age(&t, cutoff + 10.5, cutoff + 31.0).is_some());
        // No observed rows at all: nothing to date the predictions by.
        for r in t.iter_mut() {
            r.observed = false;
        }
        assert!(stale_prediction_age(&t, now, now).is_none());
        assert!(stale_prediction_age(&[], now, now).is_none());
    }

    /// A truncated transfer of the real file is rejected by the download
    /// validator wherever it is cut in an observed row, and the parser
    /// never turns the cut line into a row with a partial value. (A cut
    /// inside the UT1-UTC field used to parse "-0." as 0.0, and a cut line
    /// that lost its flag was skipped, both leaving a table without
    /// predictions that replaced the good file on disk.)
    #[test]
    fn truncated_file_is_rejected() {
        let path = datadir::find_all(FINALS2000A_FILE)
            .into_iter()
            .next()
            .expect("finals2000A.all present in tests");
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let row = |mjd: &str| lines.iter().position(|l| cols(l, 8, 8) == mjd).unwrap();
        let dir = std::env::temp_dir().join(format!("satkit_eop_trunc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(FINALS2000A_FILE);
        let prefix = |upto: usize, cut: &str| -> String {
            let mut t = lines[..upto].join("\n");
            t.push('\n');
            t.push_str(cut);
            t
        };

        // The whole file is fine, with or without a byte-order mark.
        std::fs::write(&file, &text).unwrap();
        assert_eq!(validate_file(&file), Ok(()));
        let reference = parse_finals2000a(&text).unwrap();
        assert_eq!(
            parse_finals2000a(&format!("\u{feff}{text}")).unwrap(),
            reference
        );

        // Observed row of MJD 60000, cut at every column: the parser either
        // rejects the line or reads only complete values, and the validator
        // always rejects the file (no predictions after the cut).
        let idx = row("60000.00");
        let line = lines[idx].trim_end();
        let full = parse_finals2000a(line).unwrap();
        for cut in 0..line.len() {
            let piece = &line[..cut];
            if let Ok(t) = parse_finals2000a(&format!("{}\n{piece}", lines[idx - 1])) {
                if let Some(r) = t.iter().find(|r| r.mjd_utc == 60000.0) {
                    let f = &full[0];
                    assert_eq!((r.xp, r.yp, r.dut1), (f.xp, f.yp, f.dut1), "cut {cut}");
                    assert!(r.dX == f.dX || r.dX == 0.0, "cut {cut}");
                    assert!(r.dY == f.dY || r.dY == 0.0, "cut {cut}");
                }
            }
        }
        for cut in [
            0, 5, 12, 15, 16, 20, 30, 59, 61, 62, 63, 67, 68, 70, 83, 100, 120,
        ] {
            std::fs::write(&file, prefix(idx, &line[..cut])).unwrap();
            assert!(validate_file(&file).is_err(), "cut at column {cut}");
        }
        // Where the cut line is incomplete, the parser itself says so.
        for cut in [5, 12, 20, 30, 59, 61, 62, 63, 67, 83, 100, 120] {
            let err = parse_finals2000a(&prefix(idx, &line[..cut])).unwrap_err();
            assert!(
                matches!(err, Error::InvalidFinalsLine { line, .. } if line == idx + 1),
                "cut at column {cut}: {err}"
            );
        }

        // A predicted row cut inside a value is an error too; cut at a line
        // boundary inside the predictions, the file is accepted (every row
        // read is complete, the predictions just end earlier).
        let idx = lines.iter().rposition(|l| cols(l, 17, 1) == "P").unwrap() - 100;
        let line = lines[idx].trim_end();
        for cut in [30, 62, 66] {
            std::fs::write(&file, prefix(idx, &line[..cut])).unwrap();
            assert!(validate_file(&file).is_err(), "predicted row cut at {cut}");
        }
        std::fs::write(&file, prefix(idx, "")).unwrap();
        assert_eq!(validate_file(&file), Ok(()));

        // The date-only rows at the end of the file are accepted; a cut one,
        // or one carrying values without a flag, is not.
        let tail = lines.iter().rposition(|l| cols(l, 17, 1) == " ").unwrap();
        assert_eq!(validate_file(&file), Ok(()));
        let date_only = lines[tail].trim_end();
        assert_eq!(date_only.len(), 15);
        assert_eq!(parse_finals2000a(date_only).unwrap(), vec![]);
        assert!(parse_finals2000a(&date_only[..12]).is_err());
        assert!(parse_finals2000a(&format!("{date_only}    0.123456")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CelesTrak's `EOP-All.csv` is refused with an error that names
    /// `finals2000A.all` and `update_datafiles()`, from bytes, from a path
    /// and by the download validator.
    #[test]
    fn celestrak_csv_is_rejected_with_a_clear_error() {
        for text in [CSV_SAMPLE.to_string(), format!("\u{feff}{CSV_SAMPLE}")] {
            let err = parse_finals2000a(&text).unwrap_err();
            assert!(matches!(err, Error::UnsupportedCsv), "{err}");
            let msg = err.to_string();
            assert!(msg.contains("finals2000A.all"), "{msg}");
            assert!(msg.contains("update_datafiles()"), "{msg}");
        }
        let dir = std::env::temp_dir().join(format!("satkit_eop_csv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("EOP-All.csv");
        std::fs::write(&csv, CSV_SAMPLE).unwrap();
        assert!(validate_file(&csv).unwrap_err().contains("EOP-All.csv"));
        // `init_from_path` fails before touching the loaded table.
        assert!(matches!(init_from_path(&csv), Err(Error::UnsupportedCsv)));
        assert!(coverage().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The directory loader reads `finals2000A.all`; a directory holding
    /// only a CelesTrak `EOP-All.csv` has no EOP file, and a corrupt file is
    /// skipped with a warning. Nothing here replaces the loaded table.
    #[test]
    fn read_table_uses_finals_only() {
        let dir = std::env::temp_dir().join(format!("satkit_eop_load_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let finals = dir.join(FINALS2000A_FILE);
        std::fs::write(dir.join("EOP-All.csv"), CSV_SAMPLE).unwrap();
        assert!(read_table(existing(finals.clone()).as_deref()).is_none());
        assert!(matches!(load_from_dir(&dir), Err(Error::NoEopFile { .. })));
        assert!(freshest_copy(std::slice::from_ref(&dir)).is_none());

        std::fs::write(&finals, FINALS_SAMPLE).unwrap();
        let t = read_table(existing(finals.clone()).as_deref()).unwrap();
        assert_eq!(t.len(), 5);
        assert_eq!(t[0].mjd_utc, FINALS_FIRST_MJD);

        std::fs::write(&finals, "73 1 2 41684.00 X garbage\n").unwrap();
        assert!(read_table(existing(finals.clone()).as_deref()).is_none());
        assert!(matches!(load_from_dir(&dir), Err(Error::NoEopFile { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Path of the copy [`freshest_table`] picks.
    fn freshest_copy(dirs: &[PathBuf]) -> Option<PathBuf> {
        freshest_table(dirs).map(|(p, _)| p)
    }

    /// A stale copy of `finals2000A.all` in an earlier search directory (an
    /// `add_search_dir` directory, a system-wide copy) must not shadow
    /// the fresh copy in a later one (the write location): the default load
    /// reads the copy with the latest observed row.
    #[test]
    fn stale_copy_in_earlier_search_dir_does_not_shadow_fresh_one() {
        let root = std::env::temp_dir().join(format!("satkit_eop_shadow_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (early, late) = (root.join("system"), root.join("write"));
        std::fs::create_dir_all(&early).unwrap();
        std::fs::create_dir_all(&late).unwrap();
        // Stale: the file truncated after its 1992 row.
        let stale: String = FINALS_SAMPLE
            .lines()
            .take(2)
            .map(|l| format!("{l}\n"))
            .collect();
        std::fs::write(early.join(FINALS2000A_FILE), &stale).unwrap();
        std::fs::write(late.join(FINALS2000A_FILE), FINALS_SAMPLE).unwrap();
        let dirs = vec![early.clone(), late.clone()];

        // First-match lookup (the old behaviour) would read the stale copy.
        assert_eq!(
            datadir::find_all_in(&dirs, FINALS2000A_FILE)[0],
            early.join(FINALS2000A_FILE)
        );
        let finals = freshest_copy(&dirs);
        assert_eq!(
            finals.as_deref(),
            Some(late.join(FINALS2000A_FILE).as_path())
        );
        let t = read_table(finals.as_deref()).unwrap();
        assert_eq!(last_observed_mjd(&t), 61300.0);

        // Reversed search order: the fresh copy still wins.
        let dirs = vec![late.clone(), early.clone()];
        assert_eq!(
            freshest_copy(&dirs).as_deref(),
            Some(late.join(FINALS2000A_FILE).as_path())
        );
        // A corrupt copy is passed over for a readable one.
        std::fs::write(early.join(FINALS2000A_FILE), "73 1 2 41684.00 X garbage\n").unwrap();
        let dirs = vec![early.clone(), late.clone()];
        assert_eq!(
            freshest_copy(&dirs).as_deref(),
            Some(late.join(FINALS2000A_FILE).as_path())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The "too early" warning says there is no earlier data only for a
    /// table that starts where `finals2000A.all` does; a custom or truncated
    /// table just gets its start (in the main message).
    #[test]
    fn too_early_advice_only_for_the_default_file() {
        assert!(too_early_advice(FINALS_FIRST_MJD).contains("no EOP data before 1973-01-02"));
        assert_eq!(too_early_advice(50000.0), "");
        assert_eq!(too_early_advice(37665.0), "");
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
        let early = crate::Instant::from_rfc3339("1950-04-16T00:00:00Z").unwrap();
        assert_eq!(status(&early), EopStatus::BeforeTable);
    }

    /// The default table is `finals2000A.all`, which starts on 1973-01-02;
    /// before that there is no EOP, so UT1 = UTC and the frame transforms
    /// use zeros.
    #[test]
    fn table_starts_1973_and_ut1_is_utc_before() {
        let c = coverage().expect("EOP table loaded in tests");
        assert_eq!(c.first.as_mjd_utc(), FINALS_FIRST_MJD);
        let t = crate::Instant::from_rfc3339("1972-06-01T12:00:00Z").unwrap();
        assert_eq!(status(&t), EopStatus::BeforeTable);
        assert!(get(&t).is_none());
        assert_eq!(get_or_zero(&t), [0.0; 6]);
        let ut1 = t.as_mjd_with_scale(TimeScale::UT1);
        let utc = t.as_mjd_with_scale(TimeScale::UTC);
        assert!((ut1 - utc).abs() * 86400.0 < 1.0e-6, "{ut1} vs {utc}");
        // The first row is inside the table.
        assert!(get(&c.first).is_some());
    }

    /// The last row of the table is inside the table: a query at exactly its
    /// epoch is not extrapolation (and must not print the out-of-range
    /// warning); anything later is.
    #[test]
    fn last_row_epoch_is_inside_table() {
        let table = parse_finals2000a(FINALS_SAMPLE).unwrap();
        let last = table.last().unwrap();
        assert!(!beyond_table(last.mjd_utc, last));
        assert!(!beyond_table(last.mjd_utc - 0.5, last));
        assert!(beyond_table(last.mjd_utc + 1e-9, last));
    }

    /// Check values against the IERS values for 2021-09-07. The tolerances
    /// accept the Bulletin A rapid values as well as the final series (they
    /// differ at the few-µs / 0.1 mas level); LOD differs more between them
    /// and is checked loosely.
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

    /// Between the rows that bracket a leap second (2016-12-31 and
    /// 2017-01-01), UT1 − UTC is interpolated without the +1 s step, which
    /// only takes effect at the second row.
    #[test]
    fn interp_across_leap_second() {
        let v0 = eop_from_mjd_utc(57753.0).unwrap()[0];
        let v1 = eop_from_mjd_utc(57754.0).unwrap()[0];
        assert!((v1 - v0 - 1.0).abs() < 0.01, "step {v0} -> {v1}");
        for x in 0..100 {
            let g = x as f64 / 100.0;
            let v = eop_from_mjd_utc(57753.0 + g).unwrap()[0];
            let expected = (1.0 - g) * v0 + g * (v1 - 1.0);
            assert!((v - expected).abs() < 1.0e-9, "{g}: {v} vs {expected}");
        }
        assert_eq!(eop_from_mjd_utc(57754.0).unwrap()[0], v1);
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
