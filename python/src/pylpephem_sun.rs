use crate::pyinstant::{PyInstant, TimeArg};
use crate::pyitrfcoord::PyITRFCoord;
use crate::pyutils;
use anyhow::Result;
use pyo3::prelude::*;
use satkit::lpephem::sun;

/// Sun position in the Geocentric Celestial Reference Frame (GCRF)
///
/// Notes:
///    * Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated from MOD to GCRF via Equations 3-88 and 3-89 in Vallado.
///    * Valid with accuracy of .01 degrees from 1950 to 2050
///
/// Args:
///     time (satkit.time, numpy array, or list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element array or Nx3 array representing sun position in GCRF frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_gcrf(time: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(&|t| Ok(sun::pos_gcrf(t)), time)
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
///     numpy.ndarray: 3-element array or Nx3 array representing sun position in MOD frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_mod(time: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(&|t| Ok(sun::pos_mod(t)), time)
}

/// Sunrise and sunset times on a calendar date at the given location.
///
/// The input selects the UTC calendar date (its time of day is ignored);
/// the returned sunrise and sunset are those of that date at the location's
/// longitude, as UTC times.  At far-west longitudes the sunset can fall on
/// the next UTC date (and at far-east longitudes the sunrise on the
/// previous one).
///
/// To get the events of a *local* date, pass a timezone-aware datetime at
/// local noon: for time zones UTC-11 to UTC+11 local noon falls on the same
/// UTC date, while local midnight falls on the previous UTC date east of
/// Greenwich.
///
/// Notes:
///     * Vallado Algorithm 30
///     * Sigma is the angle between noon and rise/set.  Common values:
///         * "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
///         * "Civil Twilight": 96 deg
///         * "Nautical Twilight": 102 deg
///         * "Astronomical Twilight": 108 deg
///
/// Args:
///     time (satkit.time|datetime.datetime): time whose UTC calendar date selects the day
///     coord (satkit.itrfcoord): location at which to compute sunrise and sunset
///     sigma (float, optional): angle in degrees between noon and rise/set.  Default is 90.0+50.0/60.0 (Standard)
///
/// Returns:
///     (satkit.time, satkit.time): tuple of sunrise and sunset times
///
/// Example:
///
/// ```python
/// from datetime import datetime, timedelta, timezone
///
/// # Sunrise and sunset on 2024-10-14 in Honolulu: pass local noon on that
/// # date (a ZoneInfo("Pacific/Honolulu") tzinfo works the same way)
/// honolulu = timezone(timedelta(hours=-10))
/// coord = satkit.itrfcoord(latitude_deg=21.31, longitude_deg=-157.86)
/// noon = datetime(2024, 10, 14, 12, tzinfo=honolulu)
/// sunrise, sunset = satkit.sun.rise_set(noon, coord)
/// print(f"Sunrise: {sunrise.to_datetime().astimezone(honolulu)}")
/// print(f"Sunset:  {sunset.to_datetime().astimezone(honolulu)}")
/// ```
#[pyfunction(signature=(time, coord, sigma=None))]
pub fn rise_set(
    time: TimeArg,
    coord: &PyITRFCoord,
    sigma: Option<f64>,
) -> PyResult<(PyInstant, PyInstant)> {
    let (rise, set) = sun::riseset(&time.0, &coord.0, sigma)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyInstant(rise), PyInstant(set)))
}

/// Is satellite in Earth shadow given sun position
///
/// Notes:
///     * See algorithm in Section 3.4.2 of Montenbruck and Gill for calculation
///     * Beyond ~1.4 million km on the anti-Sun side the Earth's disc is smaller
///       than the Sun's, and on the shadow axis the eclipse is annular:
///       1 - b^2/a^2, with a and b the apparent radii of the Sun and the Earth
///     * A position at or below the Earth's surface is lit when the Sun is above
///       its local horizon plane; the Earth's center returns 0
///
/// Args:
///     sunpos (numpy.ndarray): 3-element geocentric Sun position, meters
///     satpos (numpy.ndarray): 3-element geocentric satellite position, meters
///
/// Returns:
///     float: unitless number in range [0,1] indicating no sun (0) or full sun (1, no occlusion) hitting satellite
#[pyfunction(signature=(sunpos, satpos))]
pub fn shadowfunc(sunpos: Bound<'_, PyAny>, satpos: Bound<'_, PyAny>) -> Result<f64> {
    let satpos = pyutils::py_to_smatrix::<3, 1>(&satpos)?;
    let sunpos = pyutils::py_to_smatrix::<3, 1>(&sunpos)?;
    Ok(sun::shadowfunc(&sunpos, &satpos))
}
