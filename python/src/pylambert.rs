use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use satkit::mathtypes::Vector3;

type LambertSolution = (Py<PyArray1<f64>>, Py<PyArray1<f64>>);

/// Solve Lambert's problem: find the transfer orbit between two positions.
///
/// Given two position vectors and a time of flight, compute the departure and
/// arrival velocity vectors for the transfer orbit connecting them.
///
/// Args:
///     r1 (numpy.ndarray): Initial position vector, 3-element array (meters)
///     r2 (numpy.ndarray): Final position vector, 3-element array (meters)
///     tof (float): Time of flight in seconds (must be positive)
///     mu (float, optional): Gravitational parameter in m³/s² (default: Earth μ)
///     prograde (bool, optional): If True (default), use prograde transfer;
///         if False, use retrograde transfer.
///
/// Returns:
///     list[tuple[numpy.ndarray, numpy.ndarray]]: List of (v1, v2) solution pairs.
///         Each v1 is the departure velocity and v2 is the arrival velocity
///         (3-element numpy arrays in m/s). The first element is the
///         zero-revolution solution.
///
/// Raises:
///     ValueError: If inputs are invalid (negative tof, zero position, etc.)
///     RuntimeError: If the solver fails to converge
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
    r1: &Bound<'_, PyArray1<f64>>,
    r2: &Bound<'_, PyArray1<f64>>,
    tof: f64,
    mu: Option<f64>,
    prograde: Option<bool>,
) -> PyResult<Vec<LambertSolution>> {
    if r1.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r1 must be a 3-element array",
        ));
    }
    if r2.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r2 must be a 3-element array",
        ));
    }

    let r1_slice = unsafe { r1.as_slice()? };
    let r2_slice = unsafe { r2.as_slice()? };

    let r1_vec = Vector3::from_slice(r1_slice);
    let r2_vec = Vector3::from_slice(r2_slice);

    let mu_val = mu.unwrap_or(satkit::consts::MU_EARTH);
    let prograde_val = prograde.unwrap_or(true);

    let solutions = satkit::lambert::lambert(&r1_vec, &r2_vec, tof, mu_val, prograde_val)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Python::attach(|py| {
        let result: Vec<LambertSolution> = solutions
            .iter()
            .map(|(v1, v2)| {
                let v1_arr = PyArray1::from_slice(py, v1.as_slice()).unbind();
                let v2_arr = PyArray1::from_slice(py, v2.as_slice()).unbind();
                (v1_arr, v2_arr)
            })
            .collect();
        Ok(result)
    })
}
