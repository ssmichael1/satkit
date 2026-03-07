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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Once;

use crate::utils::datadir;
use crate::utils::{download_file, download_if_not_exist};

use anyhow::{bail, Context, Result};

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

fn load_eop_file_csv(filename: Option<PathBuf>) -> Result<Vec<EOPEntry>> {
    let path: PathBuf = filename.map_or_else(
        || {
            datadir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("EOP-All.csv")
        },
        |pb| pb,
    );
    // Download EOP data from celetrak.org
    download_if_not_exist(&path, Some("http://celestrak.org/SpaceData/"))?;

    let file: File = File::open(&path)?;

    io::BufReader::new(file)
        .lines()
        .skip(1)
        .map(|rline| -> Result<EOPEntry> {
            let line = rline.unwrap();
            let lvals: Vec<&str> = line.split(",").collect();
            if lvals.len() < 12 {
                bail!("Invalid entry in EOP file");
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

#[allow(dead_code)]
fn load_eop_file_legacy(filename: Option<PathBuf>) -> Result<Vec<EOPEntry>> {
    let path: PathBuf = filename.map_or_else(
        || {
            datadir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("finals2000A.all")
        },
        |pb| pb,
    );

    if !path.is_file() {
        bail!(
            "Cannot open earth orientation parameters file: {}",
            path.to_str().unwrap()
        );
    }

    let file = match File::open(&path) {
        Err(why) => bail!("Couldn't open {}: {}", path.display(), why),
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
                        .context("Could not extract MJD from file")?,
                    xp: xp_str
                        .trim()
                        .parse()
                        .context("Could not extract X polar motion from file")?,
                    yp: yp_str
                        .trim()
                        .parse()
                        .context("Could not extract Y polar motion from file")?,
                    dut1: dut1_str
                        .trim()
                        .parse()
                        .context("Could not extract delta UT1 from file")?,
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
static EOP_INIT: Once = Once::new();
static EOP_PTR: AtomicPtr<Vec<EOPEntry>> = AtomicPtr::new(std::ptr::null_mut());

/// Ensure EOP data is loaded (runs at most once).
fn ensure_eop_loaded() {
    EOP_INIT.call_once(|| {
        let data = Box::new(load_eop_file_csv(None).unwrap_or_default());
        EOP_PTR.store(Box::into_raw(data), Ordering::Release);
    });
}

/// Get a lock-free reference to the EOP data.
///
/// SAFETY: After `ensure_eop_loaded()`, `EOP_PTR` is non-null and points to
/// a valid `Vec<EOPEntry>`. The data is never freed — old allocations are
/// intentionally leaked on `update()` since other threads may reference them.
fn eop_data() -> &'static Vec<EOPEntry> {
    ensure_eop_loaded();
    unsafe { &*EOP_PTR.load(Ordering::Acquire) }
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
        bail!(
            r#"Data directory is read-only.
             Try setting the environment variable SATKIT_DATA
             to a writeable directory and re-starting or explicitly set
             data directory"#
        );
    }

    let url = "http://celestrak.org/SpaceData/EOP-All.csv";
    download_file(url, &d, true)?;

    // Ensure initial pointer is valid, then swap in new data.
    // Old allocation is intentionally leaked (other threads may reference it).
    ensure_eop_loaded();
    let new_data = Box::into_raw(Box::new(load_eop_file_csv(None)?));
    EOP_PTR.store(new_data, Ordering::Release);

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
///     * 4 : dX wrt IAU-2000 Nutation, milli-arcsecs
///     * 5 : dY wrt IAU-2000 Nutation, milli-arcsecs
///
/// * If time is before range of file, returns None and prints warning to stderr
///   (but only once per library load)
/// * If time is after range of file, returns the last entry's values (constant extrapolation)
///
pub fn eop_from_mjd_utc(mjd_utc: f64) -> Option<[f64; 6]> {
    let eop = eop_data();

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
///   * 4 : dX wrt IAU-2000 Nutation, milli-arcsecs
///   * 5 : dY wrt IAU-2000 Nutation, milli-arcsecs
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
        assert!(eop_data()[0].mjd_utc >= 0.0);
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
