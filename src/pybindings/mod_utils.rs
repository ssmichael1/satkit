use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::IntoPyObjectExt;

use std::path::PathBuf;

///
/// Download data files needed for computation
///
/// Args:
///     overwrite (bool): Download and overwrite files if they already exist
///     dir (str): Target directory for files.  Uses existing data directory if not specified
///
///
/// Files include:
///
/// * EGM96.gfc :: EGM-96 Gravity Model Coefficients
/// * JGM3.gfc :: JGM-3 Gravity Model Coefficients
/// * JGM2.gfc :: JGM-2 Gravity Model Coefficients
/// * ITU_GRACE16.gfc :: ITU Grace 16 Gravity Model Coefficients
/// * tab5.2a.txt :: Coefficients for GCRS to GCRF conversion
/// * tab5.2b.txt :: Coefficients for GCRS to GCRF conversion
/// * tab5.2d.txt :: Coefficients for GCRS to GCRF conversion
/// * SW-All.csv :: Space weather data, updated daily
/// * leap-seconds.txt :: Leap seconds (UTC vs TAI)
/// * EOP_All.csv :: Earth orientation parameters,  updated daily
/// * linux_p1550p2650.440 :: JPL Ephemeris version 440 (~ 100 MB)
///
/// Note: Files update daily will always be downloaded independet of
/// overwrite flag
///
#[pyfunction]
#[pyo3(signature=(**kwds))]
fn update_datafiles(kwds: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
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

    match crate::utils::update_datafiles(datadir, overwrite_files) {
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        Ok(_) => Ok(()),
    }
}

/// Get directory where astronomy data is stored
///
/// Tries the following paths in order, and stops when the
/// files are found
///
/// *  "SATKIT_DATA" environment variable
/// *  ${DYLIB}/satkit-data where ${DYLIB} is location of compiled python library
/// *  ${HOME}/Library/Application Support/satkit-data (on MacOS Only)
/// *  ${HOME}/.satkit-data
/// *  ${HOME}
/// *  /usr/share/satkit-data
/// * /Library/Application Support/satkit-data (on MacOS Only)
///
/// Returns:
///     str: Directory where files are stored
#[pyfunction]
fn datadir() -> PyResult<PyObject> {
    pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
        match crate::utils::datadir() {
            Ok(v) => v.to_str().unwrap().into_py_any(py),
            Err(_) => pyo3::types::PyNone::get(py).into_py_any(py),
        }
    })
}

/// Set the data directory
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
fn set_datadir(datadir: String) -> PyResult<()> {
    let d = PathBuf::from(datadir);
    match crate::utils::set_datadir(&d) {
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        Ok(_) => Ok(()),
    }
}

/// Check if data files are found
///
/// Returns:
///   bool: True if data files are found
///        False if data files are not found
///
#[pyfunction]
fn datafiles_exist() -> PyResult<bool> {
    Ok(crate::utils::data_found())
}

/// Git hash of compiled library
///
/// Returns:
///     str: Git hash of compiled library
#[pyfunction]
fn githash() -> PyResult<String> {
    Ok(String::from(crate::utils::githash()))
}

/// Version of satkit
///
/// Returns:
///    str: Version of satkit
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(String::from(crate::utils::gittag()))
}

/// Location of the compiled library
///
/// Returns:
///     str: Path to the compiled library
#[pyfunction]
fn dylib_path() -> PyResult<PyObject> {
    pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
        process_path::get_dylib_path().map_or_else(
            || pyo3::types::PyNone::get(py).into_py_any(py),
            |v| v.to_str().unwrap().into_py_any(py),
        )
    })
}

/// Build date of compiled
///
/// Returns:
///     str: Build date of compiled library
#[pyfunction]
fn build_date() -> PyResult<String> {
    Ok(String::from(crate::utils::build_date()))
}

/// Astro utility functions
#[pymodule]
pub fn utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(datadir, m)?).unwrap();
    m.add_function(wrap_pyfunction!(set_datadir, m)?).unwrap();
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
