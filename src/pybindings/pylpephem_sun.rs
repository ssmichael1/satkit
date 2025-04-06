use super::pyinstant::PyInstant;
use super::pyitrfcoord::PyITRFCoord;
use super::pyutils;
use crate::lpephem::sun;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use anyhow::Result;

/// Sun position in the Geocentric Celestial Reference Frame (GCRF)
///
/// Notes:
//     * Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated from MOD to GCRF via Equations 3-88 and 3-89 in Vallado.
///    * Valid with accuracy of .01 degrees from 1950 to 2050
///
/// Args:
///     time (satkit.time, numpy array, or list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element array or Nx3 array representing sun position in GCRF frame at input time[s]
#[pyfunction]
pub fn pos_gcrf(time: &Bound<'_, PyAny>) -> Result<PyObject> {
    pyutils::py_vec3_of_time_arr(&sun::pos_gcrf, time)
}

/// Sun position in the Mean-of-Date Frame
///
/// Notes:
///    * Algorithm 29 from Vallado for sun in Mean of Date (MOD)
///    * Valid with accuracy of .01 degrees from 1950 to 2050
/// Args:
///     time (Instant, numpy array, or list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element array or Nx3 array representing sun position in MOD frame at input time[s]
#[pyfunction]
pub fn pos_mod(time: &Bound<'_, PyAny>) -> Result<PyObject> {
    pyutils::py_vec3_of_time_arr(&sun::pos_mod, time)
}

/// Sunrise and sunset times on the day given by input time and at the given location.  
///
/// Notes:
///     * Vallado Algorithm 30
///     * Time is at location, and should have hours, minutes, and seconds set to zero
///     * Sigma is the angle between noon and rise/set.  Common values:
///         * "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
///         * "Civil Twilight": 96 deg
///         * "Nautical Twilight": 102 deg
///         * "Astronomical Twilight": 108 deg
///
/// Args:
///     time (satkit.time): time at which to compute sunrise and sunset
///     coord (satkit.ITRFCoord): location at which to compute sunrise and sunset
///     sigma (float, optional): angle in degrees between noon and rise/set.  Default is 90.0+50.0/60.0 (Standard)
///
/// Returns:
///     (satkit.time, satkit.time): tuple of sunrise and sunset times
#[pyfunction(signature=(time, coord, sigma=None))]
pub fn rise_set(
    time: &PyInstant,
    coord: &PyITRFCoord,
    sigma: Option<f64>,
) -> PyResult<(PyObject, PyObject)> {
    match sun::riseset(&time.0, &coord.0, sigma) {
        Ok((rise, set)) => {
            pyo3::Python::with_gil(|py| Ok((rise.into_py_any(py)?, set.into_py_any(py)?)))
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

#[pyfunction(signature=(sunpos, satpos))]
pub fn shadowfunc(sunpos: Bound<'_, PyAny>, satpos: Bound<'_, PyAny>) -> PyResult<f64> {
    let satpos = pyutils::py_to_smatrix::<3, 1>(&satpos)?;
    let sunpos = pyutils::py_to_smatrix::<3, 1>(&sunpos)?;
    Ok(sun::shadowfunc(&sunpos, &satpos))
}
