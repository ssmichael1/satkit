use super::pyutils;
use crate::lpephem::moon;
use pyo3::prelude::*;

/// Approximate Moon position in the GCRF Frame
///
/// Notes:
///   * From Vallado Algorithm 31
///   * Valid with accuracy of 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude, and 1275 km in range
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element numpy array or Nx3 numpy array representing moon position in GCRF frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_gcrf(time: &Bound<'_, PyAny>) -> anyhow::Result<PyObject> {
    pyutils::py_vec3_of_time_arr(&moon::pos_gcrf, time)
}
