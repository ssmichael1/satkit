use numpy::PyArray1;
use pyo3::prelude::*;

use crate::pyutils::to_vector3;

type LambertSolution = (Py<PyArray1<f64>>, Py<PyArray1<f64>>);

/// Solve Lambert's problem: find the transfer orbit between two positions.
///
/// Given two position vectors and a time of flight, compute the departure and
/// arrival velocity vectors for the transfer orbit connecting them.
///
/// Args:
///     r1 (array-like): Initial position vector, meters: any real numeric
///         3-element array-like (numpy array, list or tuple)
///     r2 (array-like): Final position vector, meters (likewise)
///     tof (float): Time of flight in seconds (must be positive)
///     mu (float, optional): Gravitational parameter in m³/s² (default: Earth μ)
///     prograde (bool, optional): If True (default), use prograde transfer
///         (angular momentum h_z >= 0); if False, use retrograde transfer
///         (h_z <= 0). This picks the short or long way around.
///
/// Returns:
///     list[tuple[numpy.ndarray, numpy.ndarray]]: List of (v1, v2) solution pairs.
///         Each v1 is the departure velocity and v2 is the arrival velocity
///         (3-element numpy arrays in m/s). The first element is the
///         zero-revolution solution.
///
/// Raises:
///     TypeError: If ``r1`` or ``r2`` is not a real numeric array-like, or
///         ``tof`` / ``mu`` is not a number
///     ValueError: If inputs are invalid (``r1`` or ``r2`` not 3 elements,
///         non-finite values, tof or mu not positive, zero position,
///         r1 == r2), or if the solver fails to converge
///
/// Example:
///     >>> import satkit
///     >>> import numpy as np
///     >>> r1 = np.array([7000e3, 0, 0])
///     >>> r2 = np.array([0, 7000e3, 0])
///     >>> solutions = satkit.lambert(r1, r2, 3600.0)
///     >>> v1, v2 = solutions[0]
///
#[pyfunction]
#[pyo3(signature = (r1, r2, tof, mu=None, prograde=None))]
pub fn lambert(
    py: Python,
    r1: &Bound<'_, PyAny>,
    r2: &Bound<'_, PyAny>,
    tof: f64,
    mu: Option<f64>,
    prograde: Option<bool>,
) -> PyResult<Vec<LambertSolution>> {
    // Any real numeric 3-element array-like (list, tuple, integer or
    // non-contiguous array): `TypeError` for a non-numeric input,
    // `ValueError` for the wrong shape
    let r1_vec = to_vector3(r1, "r1")?;
    let r2_vec = to_vector3(r2, "r2")?;

    let mu_val = mu.unwrap_or(satkit::consts::MU_EARTH);
    let prograde_val = prograde.unwrap_or(true);

    let solutions = satkit::lambert::lambert(&r1_vec, &r2_vec, tof, mu_val, prograde_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(solutions
        .iter()
        .map(|(v1, v2)| {
            (
                PyArray1::from_slice(py, v1.as_slice()).unbind(),
                PyArray1::from_slice(py, v2.as_slice()).unbind(),
            )
        })
        .collect())
}
