//! Earth Orientation Parameters (EOP) module
//!
//! This module provides access to Earth Orientation Parameters (EOP) data,
//! which are essential for accurate satellite orbit predictions and transformations
//! between different reference frames.
//!
//! It includes functionality to load EOP data from a CSV file, retrieve EOP parameters for a given Modified Julian Date (MJD),
//! and update the EOP data by downloading the latest file from a specified URL.
//!
//! The EOP data includes parameters such as polar motion, UT1-UTC, and length of day (LOD),
//! which are crucial for precise calculations in satellite tracking and navigation.
//!
//! This module also provides a way to disable warnings about out-of-range EOP data,
//! allowing users to suppress these warnings if they are aware of the limitations of the data.
//!
//! See: https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html for details on EOP data
//!

use crate::utils::RefreshableSingleton;
use std::num::ParseFloatError;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::utils::datadir;
use crate::utils::{download_file, download_if_not_exist};
use crate::{Instant, TimeLike, TimeScale};

use thiserror::Error;

/// Errors produced by the
/// [`earth_orientation_params`](crate::earth_orientation_params) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A line in the EOP CSV file has fewer than the expected 12 fields.
    #[error("Invalid entry in EOP file")]
    InvalidEntry,

    /// The configured data directory is read-only and cannot receive an
    /// updated EOP file.
    #[error(
        "Data directory is read-only. Try setting the environment variable SATKIT_DATA \
         to a writeable directory and re-starting or explicitly set data directory"
    )]
    DataDirReadOnly,

    /// Bytes passed to [`init_from_bytes`] were not valid UTF-8 — the
    /// EOP file is a CSV text format.
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

#[derive(Debug)]
#[allow(non_snake_case)]
struct EOPEntry {
    mjd_utc: f64,
    xp: f64,
    yp: f64,
    dut1: f64,
    lod: f64,
    dX: f64,
    dY: f64,
    /// `true` for an observed (`O`) row, `false` for a predicted (`P`) row
    /// (CelesTrak `DATA_TYPE` column).
    observed: bool,
}

/// Where a given epoch falls relative to the loaded EOP table.
///
/// Returned by [`status`]; see [`coverage`] for the table bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopStatus {
    /// Inside the table, on or before the last observed (`O`) row.
    Observed,
    /// Inside the table, after the last observed row: IERS predictions.
    Predicted,
    /// After the last row of the table: the last row's values are held
    /// constant. Accuracy degrades with distance from the table end
    /// (polar motion drifts ~0.1″ and UT1−UTC by ~10 ms over a few
    /// months) — refresh the data with
    /// [`update`] / `satkit::utils::update_datafiles()`.
    Extrapolated,
    /// Before the first row of the table (before 1962): no EOP available,
    /// [`get`] returns `None` and the frame transforms use zeros.
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
    /// Epoch of the last observed (`O`) row; rows after it are IERS
    /// predictions.
    pub last_observed: Instant,
    /// Epoch of the last row (observed or predicted). Queries after it
    /// return this row's values unchanged.
    pub last: Instant,
}

/// Parse an `EOP-All.csv` text buffer into EOP entries.
fn parse_csv(text: &str) -> Result<Vec<EOPEntry>> {
    text.lines()
        .skip(1)
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
                dX: lvals[8].parse()?,
                dY: lvals[9].parse()?,
                observed: lvals[11].trim() != "P",
            })
        })
        .collect()
}

/// Check that the file at `path` is a parsable `EOP-All.csv`, without
/// touching the loaded table.
///
/// Used by the downloader to reject a response that is not the file it
/// claims to be — a proxy notice page served with `200 OK`, or a truncated
/// transfer — before it replaces a good table on disk. The check is the real
/// parser, so anything that would later fail to load fails here instead,
/// while the previous file is still in place.
pub(crate) fn validate_file(path: &std::path::Path) -> std::result::Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    match parse_csv(&text) {
        Ok(rows) if rows.is_empty() => Err("the file holds no EOP rows".to_string()),
        Ok(_) => Ok(()),
        Err(e) => Err(format!("not a parsable EOP-All.csv ({e})")),
    }
}

/// Lazy default load from `EOP-All.csv` under [`datadir`], with auto-download.
fn load_eop_file_csv() -> Result<Vec<EOPEntry>> {
    // Found in any search directory, else downloaded into the write location.
    let path = crate::utils::datadir::path_for("EOP-All.csv")?;
    download_if_not_exist(&path, Some("https://celestrak.org/SpaceData/"))?;
    parse_csv(&std::fs::read_to_string(&path)?)
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
static EOP: RefreshableSingleton<Vec<EOPEntry>> = RefreshableSingleton::new();

/// Best-effort default load on first read. Failures are silent — if EOP
/// can't be loaded, the singleton stays empty and queries fall through
/// to the "no data" branch.
fn ensure_default_loaded() {
    EOP.ensure_default_loaded(|| load_eop_file_csv().ok());
}

/// Initialize the EOP singleton from an in-memory byte buffer.
///
/// The bytes must be a valid CelesTrak `EOP-All.csv` text file (UTF-8).
/// Always succeeds and replaces any previously loaded data — IERS
/// publishes new EOP daily and refresh-in-place is the intended model.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    EOP.set(parse_csv(std::str::from_utf8(bytes)?)?);
    Ok(())
}

/// Initialize the EOP singleton from a file at `path`.
///
/// Same semantics as [`init_from_bytes`]; always replaces.
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    EOP.set(parse_csv(&std::fs::read_to_string(path)?)?);
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
    let Some(eop) = guard.as_ref().filter(|e| !e.is_empty()) else {
        return EopStatus::NotLoaded;
    };
    if mjd_utc < eop[0].mjd_utc {
        return EopStatus::BeforeTable;
    }
    if mjd_utc > eop[eop.len() - 1].mjd_utc {
        return EopStatus::Extrapolated;
    }
    let last_observed = eop
        .iter()
        .rev()
        .find(|e| e.observed)
        .map_or(-1.0, |e| e.mjd_utc);
    if mjd_utc <= last_observed {
        EopStatus::Observed
    } else {
        EopStatus::Predicted
    }
}

/// Download new Earth Orientation Parameters file, and load it.
pub fn update() -> Result<()> {
    let d = datadir()?;
    if d.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    let url = "https://celestrak.org/SpaceData/EOP-All.csv";
    download_file(url, &d, true)?;

    EOP.set(load_eop_file_csv()?);
    Ok(())
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
///     * 3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
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
    let Some(eop) = guard.as_ref().filter(|e| !e.is_empty()) else {
        if !NOT_LOADED_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
            eprintln!(
                "Warning: no Earth Orientation Parameters (EOP) table is loaded; polar motion, \
                 UT1-UTC and nutation corrections are being treated as zero, which biases \
                 Earth-fixed frame transforms and orbit propagation by metres.\n\
                 Run `satkit::utils::update_datafiles()` (Python: `satkit.utils.update_datafiles()`) \
                 to download EOP-All.csv, or set SATKIT_DATA to a directory containing it.\n\
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
                 to download the most recent EOP-All.csv.\n\
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
///   * 3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
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

    /// Check value against manual value from file
    #[test]
    fn checkval() {
        let tm = crate::Instant::from_rfc3339("2006-04-16T17:52:50.805408Z").unwrap();
        let v: Option<[f64; 6]> = eop_from_mjd_utc(tm.as_mjd_utc());
        assert!(v.is_some());

        let v = eop_from_mjd_utc(59464.00).unwrap();
        const TRUTH: [f64; 4] = [-0.1145667, 0.241155, 0.317274, -0.0002255];
        for it in v.iter().zip(TRUTH.iter()) {
            let (a, b) = it;
            assert!(((a - b) / b).abs() < 1.0e-3);
        }
    }

    /// Check interpolation between points
    #[test]
    fn checkinterp() {
        let mjd0: f64 = 57909.00;
        const TRUTH0: [f64; 4] = [0.3754421, 0.102693, 0.458455, 0.0011699];
        const TRUTH1: [f64; 4] = [0.3743358, 0.104031, 0.458373, 0.0010383];
        for x in 0..101 {
            let dt: f64 = x as f64 / 100.0;
            let vt = eop_from_mjd_utc(mjd0 + dt).unwrap();
            let g0: f64 = 1.0 - dt;
            let g1: f64 = dt;
            for it in vt.iter().zip(TRUTH0.iter().zip(TRUTH1.iter())) {
                let (v, (v0, v1)) = it;
                let vtest: f64 = g0 * v0 + g1 * v1;
                assert!(((v - vtest) / v).abs() < 1.0e-5);
            }
        }
    }
}
