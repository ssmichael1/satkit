use super::pyutils;
use crate::lpephem::moon;
use pyo3::prelude::*;

#[pyfunction]
///
/// Approximate Moon position in the GCRF Frame
///
/// From Vallado Algorithm 31
///
/// Input:
///
///    time:  astro.time object, list, or numpy array
///           for which to compute position
///
/// Output:
///
///    Nx3 numpy array representing moon position in GCRF frame
///    at given time[s].  Units are meters
///
/// Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
/// and 1275 km in range
///
pub fn pos_gcrf(time: &PyAny) -> PyResult<PyObject> {
    pyutils::py_vec3_of_time_arr(&moon::pos_gcrf, time)
}
