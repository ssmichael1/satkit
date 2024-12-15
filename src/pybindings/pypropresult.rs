use pyo3::prelude::*;

use super::pyinstant::PyInstant;
use super::pyutils::*;

use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use numpy::PyArrayMethods;
use numpy::{self as np, ToPyArray};

use crate::orbitprop::PropagationResult;
use crate::types::*;
use crate::Instant;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PyPropResultType {
    R1(Box<PropagationResult<1>>),
    R7(Box<PropagationResult<7>>),
}

/// Propagation statistics
///
/// This class holds statistics about the result of a high-precision orbit propagation
///
#[pyclass(name = "propstats", module = "satkit")]
pub struct PyPropStats {
    /// Number of function evaluations
    ///
    /// Returns:
    ///     int: number of derivative function evalations used in propagation
    #[pyo3(get)]
    num_eval: u32,

    /// Number of accepted steps
    ///
    /// Returns:
    ///     int: number of accepted steps in propagation
    #[pyo3(get)]
    num_accept: u32,

    /// Number of rejected steps
    ///
    /// Returns:
    ///    int: number of rejected steps in propagation
    #[pyo3(get)]
    num_reject: u32,
}

#[pymethods]
impl PyPropStats {
    fn __str__(&self) -> String {
        format!("Propagation Statistics:\n  Function Evals: {}\n  Accepted Steps: {}\n  Rejected Steps: {}",
    self.num_eval, self.num_accept, self.num_reject)
    }
}

/// Propagation result
///
/// This class holds the result of a high-precision orbit propagation
///
/// The result includes the final state of the satellite, the time at which the state was computed,
/// and statistics about the propagation
///
/// The result may also include a dense ODE solution that can be used for interpolation of states
/// between the start and stop times
///
/// Attributes:
///
///    time_start: satkit.time object representing the time at which the propagation began
///          time: satkit.time object representing the time at which the propagation ended
///         stats: satkit.propstats object with statistics about the propagation
///           pos: 3-element numpy array representing the final position of the satellite in GCRF meters
///           vel: 3-element numpy array representing the final velocity of the satellite in GCRF m/s
///         state: 6-element numpy array representing the final state of the satellite in GCRF,
///                a concatenation of pos and vel
///           phi: 6x6 numpy array representing the state transition matrix between
///                the start and stop times, if requested
///    can_interp: boolean indicating whether the result includes a dense ODE
///                solution that can be used for interpolation
///                of states between the start and stop times
///
#[pyclass(name = "propresult", module = "satkit")]
#[derive(Debug, Clone)]
pub struct PyPropResult(pub PyPropResultType);

fn to_string<const T: usize>(r: &PropagationResult<T>) -> String {
    let mut s = "Propagation Results\n".to_string();
    s.push_str(format!("  Time: {}\n", r.time_end).as_str());
    s.push_str(
        format!(
            "   Pos: [{:.3}, {:.3}, {:.3}] km\n",
            r.state_end[0] * 1.0e-3,
            r.state_end[1] * 1.0e-3,
            r.state_end[2] * 1.0e-3
        )
        .as_str(),
    );
    s.push_str(
        format!(
            "   Vel: [{:.3}, {:.3}, {:.3}] m/s\n",
            r.state_end[3], r.state_end[4], r.state_end[5]
        )
        .as_str(),
    );
    s.push_str("  Stats:\n");
    s.push_str(format!("       Function Evaluations: {}\n", r.num_eval).as_str());
    s.push_str(format!("             Accepted Steps: {}\n", r.accepted_steps).as_str());
    s.push_str(format!("             Rejected Steps: {}\n", r.rejected_steps).as_str());
    s.push_str(format!("   Can Interp: {}\n", r.odesol.is_some()).as_str());
    if r.odesol.is_some() {
        s.push_str(format!("        Start Time: {}", r.time_start).as_str());
    }
    s
}

#[pymethods]
impl PyPropResult {
    #[new]
    /// This should never be called and is here only for pickle support
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(PyPropResultType::R1(Box::new(PropagationResult::<1> {
            time_start: Instant::INVALID,
            state_start: Vector::<6>::zeros(),
            time_end: Instant::INVALID,
            state_end: Vector::<6>::zeros(),
            num_eval: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            odesol: None,
        })))
    }

    // Get start time
    #[getter]
    fn time_start(&self) -> PyInstant {
        PyInstant(match &self.0 {
            PyPropResultType::R1(r) => r.time_start,
            PyPropResultType::R7(r) => r.time_start,
        })
    }

    /// Get the stop time
    #[getter]
    fn time(&self) -> PyInstant {
        PyInstant(match &self.0 {
            PyPropResultType::R1(r) => r.time_end,
            PyPropResultType::R7(r) => r.time_end,
        })
    }

    /// Get the stop time
    #[getter]
    fn time_end(&self) -> PyInstant {
        PyInstant(match &self.0 {
            PyPropResultType::R1(r) => r.time_end,
            PyPropResultType::R7(r) => r.time_end,
        })
    }

    #[getter]
    fn stats(&self) -> PyPropStats {
        match &self.0 {
            PyPropResultType::R1(r) => PyPropStats {
                num_eval: r.num_eval,
                num_accept: r.accepted_steps,
                num_reject: r.rejected_steps,
            },
            PyPropResultType::R7(r) => PyPropStats {
                num_eval: r.num_eval,
                num_accept: r.accepted_steps,
                num_reject: r.rejected_steps,
            },
        }
    }

    #[getter]
    fn pos(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(r) => np::ndarray::arr1(&r.state_end.as_slice()[0..3])
                    .to_pyarray(py)
                    .into_py_any(py),
                PyPropResultType::R7(r) => np::ndarray::arr1(&r.state_end.as_slice()[0..3])
                    .to_pyarray(py)
                    .into_py_any(py),
            }
        })
    }

    #[getter]
    fn vel(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(r) => np::ndarray::arr1(&r.state_end.as_slice()[3..6])
                    .to_pyarray(py)
                    .into_py_any(py),
                PyPropResultType::R7(r) => {
                    np::ndarray::arr1(&r.state_end.column(0).as_slice()[3..6])
                        .to_pyarray(py)
                        .into_py_any(py)
                }
            }
        })
    }

    #[getter]
    fn state(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(r) => np::ndarray::arr1(r.state_end.as_slice())
                    .to_pyarray(py)
                    .into_py_any(py),
                PyPropResultType::R7(r) => np::ndarray::arr1(&r.state_end.as_slice()[0..6])
                    .to_pyarray(py)
                    .into_py_any(py),
            }
        })
    }

    #[getter]
    fn state_end(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(r) => np::ndarray::arr1(r.state_end.as_slice())
                    .to_pyarray(py)
                    .into_py_any(py),
                PyPropResultType::R7(r) => np::ndarray::arr1(&r.state_end.as_slice()[0..6])
                    .to_pyarray(py)
                    .into_py_any(py),
            }
        })
    }

    #[getter]
    fn state_start(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(r) => np::ndarray::arr1(r.state_start.as_slice())
                    .to_pyarray(py)
                    .into_py_any(py),
                PyPropResultType::R7(r) => np::ndarray::arr1(&r.state_start.as_slice()[0..6])
                    .to_pyarray(py)
                    .into_py_any(py),
            }
        })
    }

    #[getter]
    fn phi(&self) -> PyResult<PyObject> {
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            match &self.0 {
                PyPropResultType::R1(_r) => Ok(py.None()),
                PyPropResultType::R7(r) => {
                    let phi = unsafe { np::PyArray2::<f64>::new(py, [6, 6], false) };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            r.state_end.as_ptr().offset(6),
                            phi.as_raw_array_mut().as_mut_ptr(),
                            36,
                        );
                    }
                    phi.into_py_any(py)
                }
            }
        })
    }

    fn __str__(&self) -> String {
        match &self.0 {
            PyPropResultType::R1(r) => to_string::<1>(r),
            PyPropResultType::R7(r) => to_string::<7>(r),
        }
    }

    #[getter]
    const fn can_interp(&self) -> bool {
        match &self.0 {
            PyPropResultType::R1(r) => r.odesol.is_some(),
            PyPropResultType::R7(r) => r.odesol.is_some(),
        }
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::empty(py);
        (tp, d)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let s = state.as_bytes(py);

        self.0 = serde_pickle::from_slice(s, serde_pickle::DeOptions::default()).unwrap();
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        let p = serde_pickle::to_vec(&self.0, serde_pickle::SerOptions::default()).unwrap();
        PyBytes::new(py, p.as_slice()).into_py_any(py)
    }

    #[pyo3(signature=(time, output_phi=false))]
    fn interp(&self, time: PyInstant, output_phi: bool) -> PyResult<PyObject> {
        match &self.0 {
            PyPropResultType::R1(r) => match r.interp(&time.0) {
                Ok(res) => pyo3::Python::with_gil(|py| -> PyResult<PyObject> { vec2py(py, &res) }),
                Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
            },
            PyPropResultType::R7(r) => match r.interp(&time.0) {
                Ok(res) => {
                    if !output_phi {
                        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                            slice2py1d(py, &res.as_slice()[0..6])
                        })
                    } else {
                        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                            (
                                slice2py1d(py, &res.as_slice()[0..6])?,
                                slice2py2d(py, &res.as_slice()[6..42], 6, 6)?,
                            )
                                .into_py_any(py)
                        })
                    }
                }
                Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ser() {
        let sol = PyPropResult::new();
        println!("sol = {:?}", sol);
        let v = serde_pickle::to_vec(&sol.0, serde_pickle::SerOptions::default()).unwrap();
        //print!("v = {:?}", v);
        let sol2 = PyPropResult(
            serde_pickle::from_slice(v.as_slice(), serde_pickle::DeOptions::default()).unwrap(),
        );
        println!("sol2 = {:?}", sol2);
    }
}
