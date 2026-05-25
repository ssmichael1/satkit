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

use std::fs::File;
use std::io::{self, BufRead};
use std::num::ParseFloatError;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, RwLock};

use crate::utils::datadir;
use crate::utils::{download_file, download_if_not_exist};

use thiserror::Error;

/// Errors produced by the
/// [`earth_orientation_params`](crate::earth_orientation_params) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A line in the EOP CSV file has fewer than the expected 12 fields.
    #[error("Invalid entry in EOP file")]
    InvalidEntry,

    /// The legacy `finals2000A.all` file could not be located.
    #[error("Cannot open earth orientation parameters file: {0}")]
    LegacyFileMissing(String),

    /// Failed to open the legacy `finals2000A.all` file.
    #[error("Couldn't open {path}: {source}")]
    LegacyOpenFailed {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse a numeric field from the legacy bulletin file.
    #[error("Could not extract {field} from file")]
    LegacyFieldParse {
        field: &'static str,
        #[source]
        source: ParseFloatError,
    },

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
            })
        })
        .collect()
}

/// Lazy default load from `EOP-All.csv` under [`datadir`], with auto-download.
fn load_eop_file_csv() -> Result<Vec<EOPEntry>> {
    let path = datadir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("EOP-All.csv");
    download_if_not_exist(&path, Some("http://celestrak.org/SpaceData/"))?;
    parse_csv(&std::fs::read_to_string(&path)?)
}

#[allow(dead_code)]
fn load_eop_file_legacy(filename: Option<PathBuf>) -> Result<Vec<EOPEntry>> {
    let path: PathBuf = filename.unwrap_or_else(|| {
        datadir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("finals2000A.all")
    });

    if !path.is_file() {
        return Err(Error::LegacyFileMissing(
            path.to_str().unwrap_or_default().to_string(),
        ));
    }

    let file = match File::open(&path) {
        Err(why) => {
            return Err(Error::LegacyOpenFailed {
                path: path.display().to_string(),
                source: why,
            });
        }
        Ok(file) => file,
    };

    let mut eopvec = Vec::<EOPEntry>::new();
    for line in io::BufReader::new(file).lines() {
        match &line.unwrap() {
            v if v.len() < 100 => (),
            v if !v.is_ascii() => (),
            v if {
                let c: String = v.chars().skip(16).take(1).collect();
                c != "I" && c != "P"
            } => {}
            v => {
                // Pull from "Bulliten A"
                let mjd_str: String = v.chars().skip(7).take(8).collect();
                let xp_str: String = v.chars().skip(18).take(9).collect();
                let yp_str: String = v.chars().skip(37).take(9).collect();
                let dut1_str: String = v.chars().skip(58).take(10).collect();
                let lod_str: String = v.chars().skip(49).take(7).collect();
                let dx_str: String = v.chars().skip(97).take(9).collect();
                let dy_str: String = v.chars().skip(116).take(9).collect();

                eopvec.push(EOPEntry {
                    mjd_utc: mjd_str
                        .trim()
                        .parse()
                        .map_err(|source| Error::LegacyFieldParse {
                            field: "MJD",
                            source,
                        })?,
                    xp: xp_str
                        .trim()
                        .parse()
                        .map_err(|source| Error::LegacyFieldParse {
                            field: "X polar motion",
                            source,
                        })?,
                    yp: yp_str
                        .trim()
                        .parse()
                        .map_err(|source| Error::LegacyFieldParse {
                            field: "Y polar motion",
                            source,
                        })?,
                    dut1: dut1_str
                        .trim()
                        .parse()
                        .map_err(|source| Error::LegacyFieldParse {
                            field: "delta UT1",
                            source,
                        })?,
                    lod: lod_str.trim().parse().unwrap_or(0.0),
                    dX: dx_str.trim().parse().unwrap_or(0.0),
                    dY: dy_str.trim().parse().unwrap_or(0.0),
                })
            }
        }
    }
    Ok(eopvec)
}

static WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// Module-scope refreshable singleton. Const-initialized as `None`;
/// the lazy default load (best-effort, silent on failure) runs at most
/// once via [`DEFAULT_LOAD_ONCE`]. [`init_from_bytes`] /
/// [`init_from_path`] / [`update`] replace any current contents.
static EOP: RwLock<Option<Vec<EOPEntry>>> = RwLock::new(None);
static DEFAULT_LOAD_ONCE: Once = Once::new();

/// Best-effort default load on first read. Failures are silent — if EOP
/// can't be loaded, the singleton stays `None` and queries fall through
/// to the "no data" branch.
fn ensure_default_loaded() {
    DEFAULT_LOAD_ONCE.call_once(|| {
        if let Ok(records) = load_eop_file_csv() {
            *EOP.write().unwrap() = Some(records);
        }
    });
}

/// Initialize the EOP singleton from an in-memory byte buffer.
///
/// The bytes must be a valid CelesTrak `EOP-All.csv` text file (UTF-8).
/// Always succeeds and replaces any previously loaded data — IERS
/// publishes new EOP daily and refresh-in-place is the intended model.
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    let records = parse_csv(std::str::from_utf8(bytes)?)?;
    DEFAULT_LOAD_ONCE.call_once(|| {});
    *EOP.write().unwrap() = Some(records);
    Ok(())
}

/// Initialize the EOP singleton from a file at `path`.
///
/// Same semantics as [`init_from_bytes`]; always replaces.
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    let records = parse_csv(&std::fs::read_to_string(path)?)?;
    DEFAULT_LOAD_ONCE.call_once(|| {});
    *EOP.write().unwrap() = Some(records);
    Ok(())
}

///
/// Disable warning about out-of-range EOP data.
///
/// Warning is shown only once, but to prevent it from being shown,
/// run this function.
///
/// # Example
///
/// ```rust
/// satkit::earth_orientation_params::disable_eop_time_warning();
/// ```
///
pub fn disable_eop_time_warning() {
    WARNING_SHOWN.store(true, Ordering::Relaxed);
}

/// Download new Earth Orientation Parameters file, and load it.
pub fn update() -> Result<()> {
    let d = datadir()?;
    if d.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    let url = "http://celestrak.org/SpaceData/EOP-All.csv";
    download_file(url, &d, true)?;

    let records = load_eop_file_csv()?;
    DEFAULT_LOAD_ONCE.call_once(|| {});
    *EOP.write().unwrap() = Some(records);

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
/// * If time is after range of file, returns the last entry's values (constant extrapolation)
///
pub fn eop_from_mjd_utc(mjd_utc: f64) -> Option<[f64; 6]> {
    ensure_default_loaded();
    let guard = EOP.read().unwrap();
    let eop = guard.as_ref()?;

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

    // For dates beyond the file, use the last entry's values
    if idx >= eop.len() {
        let last = &eop[eop.len() - 1];
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

#[cfg(test)]
mod tests {

    use super::*;

    /// Check that data is loaded
    #[test]
    fn loaded() {
        ensure_default_loaded();
        let guard = EOP.read().unwrap();
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
