use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::IntoPyObjectExt;

use std::path::PathBuf;

use anyhow::Result;

///
/// Download data files needed for computation
///
/// Not required for normal use: the IERS nutation tables and gravity models
/// are compiled into satkit, the JPL ephemeris is downloaded on first use,
/// and the Earth-orientation / space-weather files are fetched on first use.
/// Call this to provision everything up front (a container image, a machine
/// that will later be offline) or to refresh the daily files.
///
/// Args:
///     overwrite (bool): Download and overwrite files if they already exist
///     dir (str): Target directory for files.  Uses ``datadir()`` if not specified
///
///
/// Files downloaded:
///
/// * linux_p1550p2650.440 :: JPL Ephemeris version 440 (~ 100 MB)
/// * finals2000A.all :: Earth orientation parameters (IERS Bulletin A), updated daily;
///   CelesTrak's EOP-All.csv is fetched instead when both IERS mirrors are unreachable
/// * Kp_ap_Ap_SN_F107_since_1932.txt, 45-day-forecast.txt, msafe-f10-prd.txt :: Space weather (GFZ observed record, SWPC and MSAFE forecasts)
///
/// The IERS nutation tables (tab5.2a/b/d.txt) and the gravity models
/// (EGM96, EGM2008, JGM2, JGM3 — to degree 70) are compiled into satkit
/// and are not downloaded; ITU_GRACE16 (CC BY 4.0) is fetched on first use
/// of ``gravmodel.itugrace16``. A full-degree gravity file or an updated
/// IERS table placed in the data directory still takes precedence over the
/// compiled-in copy.
///
/// Static files are fetched per the compiled-in data manifest
/// (`data/manifest.json`): `SATKIT_DATA_URL` mirror first if set, then the
/// GitHub release asset, the origin server, and the legacy bucket, and are
/// only kept when size and SHA-256 match. Files already present and verified
/// are skipped unless `overwrite=True`.
///
/// The daily EOP and space-weather files follow CelesTrak's usage policy
/// rather than being re-downloaded every call: no request is made while the
/// local copy is inside the file's publication cadence (3 hours for
/// the GFZ record, 24 hours for the SWPC forecast and the Earth-orientation file, a week for MSAFE), and past that the request is
/// conditional, so an unchanged file costs a `304` and no transfer.
/// `overwrite=True` forces a full re-fetch of these too.
///
#[pyfunction]
#[pyo3(signature=(**kwds))]
fn update_datafiles(kwds: Option<&Bound<'_, PyDict>>) -> Result<()> {
    let overwrite_files = match kwds {
        None => false,
        Some(u) => match u.get_item("overwrite")? {
            Some(v) => v.extract::<bool>()?,
            None => false,
        },
    };
    let datadir = match kwds {
        None => None,
        Some(u) => match u.get_item("dir")? {
            Some(v) => Some(PathBuf::from(v.extract::<String>()?)),
            None => None,
        },
    };

    satkit::utils::update_datafiles(datadir, overwrite_files)?;
    Ok(())
}

/// Directory where downloaded data files are written
///
/// The core data (IERS nutation tables, gravity models to degree 70) is
/// compiled into satkit, so a data directory is only needed for the JPL
/// ephemeris (downloaded on first use, SHA-256 verified) and the regularly
/// refreshed Earth-orientation / space-weather files.
///
/// Files are *looked up* across several locations (see ``data_search_dirs``),
/// but downloads go to exactly one place: ``SATKIT_DATA`` if set, else the
/// directory given to ``set_datadir``, else the platform user-data directory:
///
/// * macOS: ``~/Library/Application Support/satkit-data``
/// * Linux: ``$XDG_DATA_HOME/satkit-data`` (default ``~/.local/share/satkit-data``)
/// * Windows: ``%LOCALAPPDATA%\satkit-data``
///
/// satkit never writes next to its own shared library or inside
/// ``site-packages``. Set ``SATKIT_OFFLINE=1`` to forbid downloads entirely.
///
/// Returns:
///     str: Directory downloads are written to (created on first use), or
///     None if none could be determined
#[pyfunction]
fn datadir() -> PyResult<Py<PyAny>> {
    pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
        match satkit::utils::datadir() {
            // to_string_lossy: a non-UTF-8 path (settable via SATKIT_DATA)
            // must not panic
            Ok(v) => v.to_string_lossy().into_py_any(py),
            Err(_) => pyo3::types::PyNone::get(py).into_py_any(py),
        }
    })
}

/// Directories searched for data files, in order
///
/// A file is used from the first directory that contains it; any of these
/// may be read-only (a system-wide directory, the optional ``satkit-data``
/// package inside ``site-packages``). Downloads go only to ``datadir()``.
///
/// 1. ``SATKIT_DATA`` (environment; also the write location)
/// 2. the directory given to ``set_datadir`` (also the write location)
/// 3. directories added with ``add_search_dir``
/// 4. ``<dir of the satkit extension>/satkit-data``
/// 5. ``<site-packages>/satkit_data/data`` (the ``satkit-data`` pip package)
/// 6. the platform user-data directory (the default write location)
/// 7. ``~/.satkit-data`` (legacy)
/// 8. ``/usr/share/satkit-data`` (not on Windows)
/// 9. macOS: ``/Library/Application Support/satkit-data``
///
/// Returns:
///     list[str]: search directories in order
#[pyfunction]
fn data_search_dirs() -> Vec<String> {
    satkit::utils::data_search_dirs()
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// Add a read-only directory to the data-file search list
///
/// Tried after ``SATKIT_DATA`` / ``set_datadir`` and before the platform
/// locations. Downloads are never written here. Used by the ``satkit``
/// package itself to register the optional ``satkit_data`` bundle.
///
/// Args:
///    path (str): Directory to search
#[pyfunction]
fn add_search_dir(path: String) {
    satkit::utils::add_search_dir(&PathBuf::from(path));
}

/// Forbid (or re-allow) downloads for this process
///
/// Offline mode blocks *downloads only*: the explicit ``update_datafiles()``
/// and every lazy first-use fetch (the JPL ephemeris, the Earth-orientation
/// and space-weather refresh, any non-embedded file). It does not change
/// where files are searched, and the compiled-in core data (IERS nutation
/// tables, gravity models) is unaffected. A blocked download raises
/// ``RuntimeError`` naming the file and its sources — the same error a
/// build without the ``download`` feature gives.
///
/// Precedence: the last call to ``set_offline`` wins; if it was never
/// called, the ``SATKIT_OFFLINE`` environment variable is consulted.
///
/// Args:
///    enabled (bool): True to forbid downloads, False to allow them
#[pyfunction]
fn set_offline(enabled: bool) {
    satkit::utils::set_offline(enabled);
}

/// Whether downloads are currently forbidden
///
/// Reflects ``set_offline`` if it was ever called, else the
/// ``SATKIT_OFFLINE`` environment variable.
///
/// Returns:
///    bool: True if downloads are forbidden
#[pyfunction]
fn is_offline() -> bool {
    satkit::utils::is_offline()
}

/// Set the data directory
///
/// The directory becomes the first search location (after ``SATKIT_DATA``)
/// and the location downloads are written to.
///
/// Args:
///    datadir (str): Path to the data directory
///
/// Returns:
///   None
///
/// Raises:
///  RuntimeError: If the directory does not exist
///
#[pyfunction]
fn set_datadir(datadir: String) -> Result<()> {
    let d = PathBuf::from(datadir);
    Ok(satkit::utils::set_datadir(&d)?)
}

/// Check if data files are found
///
/// Returns:
///   bool: True if data files are found
///        False if data files are not found
///
#[pyfunction]
fn datafiles_exist() -> bool {
    satkit::utils::data_found()
}

/// Git hash of compiled library
///
/// Returns:
///     str: Git hash of compiled library
#[pyfunction]
fn githash() -> String {
    String::from(satkit::utils::githash())
}

/// Version of satkit
///
/// Returns:
///    str: Version of satkit
#[pyfunction]
fn version() -> String {
    String::from(satkit::utils::gittag())
}

/// Location of the compiled library
///
/// Returns:
///     str: Path to the compiled library
#[pyfunction]
fn dylib_path() -> Result<String> {
    process_path::get_dylib_path()
        .and_then(|v| v.to_str().map(|s| s.to_string()))
        .ok_or_else(|| anyhow::anyhow!("Failed to get dylib path"))
}

/// Build date of compiled
///
/// Returns:
///     str: Build date of compiled library
#[pyfunction]
fn build_date() -> PyResult<String> {
    Ok(String::from(satkit::utils::build_date()))
}

/// Astro utility functions
#[pymodule]
pub fn utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(datadir, m)?).unwrap();
    m.add_function(wrap_pyfunction!(set_datadir, m)?).unwrap();
    m.add_function(wrap_pyfunction!(data_search_dirs, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(add_search_dir, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(set_offline, m)?).unwrap();
    m.add_function(wrap_pyfunction!(is_offline, m)?).unwrap();
    m.add_function(wrap_pyfunction!(datafiles_exist, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(dylib_path, m)?).unwrap();
    m.add_function(wrap_pyfunction!(update_datafiles, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(githash, m)?).unwrap();
    m.add_function(wrap_pyfunction!(version, m)?).unwrap();
    m.add_function(wrap_pyfunction!(build_date, m)?).unwrap();
    Ok(())
}
