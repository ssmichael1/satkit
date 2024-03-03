use super::pyastrotime::PyAstroTime;
use super::pyitrfcoord::PyITRFCoord;
use super::pyutils;
use crate::lpephem::sun;
use pyo3::prelude::*;

///
/// Sun position in the Geocentric Celestial Reference Frame (GCRF)
///
/// Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
/// from MOD to GCRF via Equations 3-88 and 3-89 in Vallado
///
/// Input:
///
///    time:  astro.time object, list, or numpy array
///           for which to compute position
///
/// Output:
///
///    Nx3 numpy array representing sun position in GCRF frame
///    at given time[s].  Units are meters
///
///
///  From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
///
#[pyfunction]
pub fn pos_gcrf(time: &PyAny) -> PyResult<PyObject> {
    pyutils::py_vec3_of_time_arr(&sun::pos_gcrf, time)
}

///
/// Sun position in the Mean-of-Date Frame
///
/// Algorithm 29 from Vallado for sun in Mean of Date (MOD)
///
/// Input:
///
///    time:  astro.time object, numpy array, or list
///           for which to compute position
///
/// Output:
///
///    Nx3 numpy array representing sun position in MOD frame
///    at given time[s].  Units are meters
///
/// From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
///
#[pyfunction]
pub fn pos_mod(time: &PyAny) -> PyResult<PyObject> {
    pyutils::py_vec3_of_time_arr(&sun::pos_mod, time)
}

///
/// Sunrise and sunset times on the day given by input time
/// and at the given location.  
///
/// Time is at location, and should have hours, minutes, and seconds
/// set to zero
///
/// Vallado Algorithm 30
///
/// Inputs:
///
///      Time:   astro.time representing date for which to compute
///              sunrise & sunset
///
///     coord:   astro.itrfcoord representing location for which to compute
///              sunrise & sunset
///
///     sigma:   optional angle in degrees between noon & rise/set:
///              Common Values:
///                           "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
///                     "Civil Twilight": 96 deg
///                  "Nautical Twilight": 102 deg
///              "Astronomical Twilight": 108 deg
///
///              If not passed in, "Standard" is used (90.0 + 50.0/60.0)
///
/// Returns Tuple:
///
///    (sunrise: AstroTime, sunset: AstroTime)
///
///
#[pyfunction]
pub fn rise_set(
    tm: &PyAstroTime,
    coord: &PyITRFCoord,
    sigma: Option<f64>,
) -> PyResult<(PyObject, PyObject)> {
    match sun::riseset(&tm.inner, &coord.inner, sigma) {
        Ok((rise, set)) => pyo3::Python::with_gil(|py| Ok((rise.into_py(py), set.into_py(py)))),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}
