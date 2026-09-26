use pyo3::prelude::*;

use crate::pyinstant::{PyInstant, ToTimeVec};
use crate::pyutils::*;

use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;
use satkit::orbitprop::PropagationResult;
use satkit::Instant;

use serde::{Deserialize, Serialize};

/// Evaluate `$body` with `$r` bound to the inner result, whichever variant
macro_rules! each {
    ($e:expr, $r:ident => $body:expr) => {
        match $e {
            PyPropResultType::R1($r) => $body,
            PyPropResultType::R7($r) => $body,
        }
    };
}

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

    /// Rebuild the statistics from their three counts. Used internally by
    /// `__reduce__`; not part of the public API.
    #[staticmethod]
    const fn _from_pickle(num_eval: u32, num_accept: u32, num_reject: u32) -> Self {
        Self {
            num_eval,
            num_accept,
            num_reject,
        }
    }

    /// Pickle (and `copy.deepcopy`) support. `propstats` has no `__new__`,
    /// so it is rebuilt through the private `_from_pickle` staticmethod from
    /// all three counts.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (u32, u32, u32))> {
        let ctor = py.get_type::<Self>().getattr("_from_pickle")?;
        Ok((ctor, (self.num_eval, self.num_accept, self.num_reject)))
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
/// between the begin and end times
///
/// Attributes:
///
///    time_begin: satkit.time object representing the time at which the propagation began
///          time: satkit.time object representing the time at which the propagation ended
///         stats: satkit.propstats object with statistics about the propagation
///           pos: 3-element numpy array representing the final position of the satellite in GCRF meters
///           vel: 3-element numpy array representing the final velocity of the satellite in GCRF m/s
///         state: 6-element numpy array representing the final state of the satellite in GCRF,
///                a concatenation of pos (meters) and vel (m/s)
///           phi: 6x6 numpy array representing the state transition matrix between
///                the begin and end times, if requested (maps a begin-state perturbation
///                in meters, m/s to the end state in meters, m/s)
///    can_interp: boolean indicating whether the result includes a dense ODE
///                solution that can be used for interpolation
///                of states between the begin and end times
///
#[pyclass(name = "propresult", module = "satkit", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPropResult(pub PyPropResultType);

fn to_string<const T: usize>(r: &PropagationResult<T>) -> String {
    let se = r.state_end.as_slice();
    let mut s = "Propagation Results\n".to_string();
    s.push_str(format!("  Time: {}\n", r.time_end).as_str());
    s.push_str(
        format!(
            "   Pos: [{:.3}, {:.3}, {:.3}] km\n",
            se[0] * 1.0e-3,
            se[1] * 1.0e-3,
            se[2] * 1.0e-3
        )
        .as_str(),
    );
    s.push_str(format!("   Vel: [{:.3}, {:.3}, {:.3}] m/s\n", se[3], se[4], se[5]).as_str());
    s.push_str("  Stats:\n");
    s.push_str(format!("       Function Evaluations: {}\n", r.num_eval).as_str());
    s.push_str(format!("             Accepted Steps: {}\n", r.accepted_steps).as_str());
    s.push_str(format!("             Rejected Steps: {}\n", r.rejected_steps).as_str());
    s.push_str(format!("   Can Interp: {}\n", r.odesol.is_some()).as_str());
    if r.odesol.is_some() {
        s.push_str(format!("        Begin Time: {}", r.time_begin).as_str());
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
            time_begin: Instant::INVALID,
            state_begin: Vector::<6>::zeros(),
            time_end: Instant::INVALID,
            state_end: Vector::<6>::zeros(),
            num_eval: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            next_step_secs: 0.0,
            odesol: None,
            gj_dense: None,
            integrator: satkit::orbitprop::Integrator::default(),
        })))
    }

    /// Get the begin time (time at which state_begin is valid)
    #[getter]
    fn time_begin(&self) -> PyInstant {
        PyInstant(each!(&self.0, r => r.time_begin))
    }

    /// Get the end time
    #[getter]
    fn time(&self) -> PyInstant {
        self.time_end()
    }

    /// Get the end time
    #[getter]
    fn time_end(&self) -> PyInstant {
        PyInstant(each!(&self.0, r => r.time_end))
    }

    /// Statistics of the propagation
    ///
    /// Returns:
    ///     satkit.propstats: function-evaluation and step counts
    #[getter]
    fn stats(&self) -> PyPropStats {
        each!(&self.0, r => PyPropStats {
            num_eval: r.num_eval,
            num_accept: r.accepted_steps,
            num_reject: r.rejected_steps,
        })
    }

    /// Step the integrator would take next, seconds: its working stride at
    /// ``time_end`` (the controller's last unclamped proposal for the adaptive
    /// integrators, the fixed step for ``integrator.gauss_jackson8``, 0 for a
    /// zero-duration propagation). Signed like the propagation direction.
    /// Pass it as ``propsettings.initial_step_secs`` to continue this arc
    /// without the start-up ramp.
    ///
    /// Returns:
    ///     float: next integrator step, seconds
    #[getter]
    fn next_step_secs(&self) -> f64 {
        each!(&self.0, r => r.next_step_secs)
    }

    /// GCRF position of satellite at end of propagation
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element GCRF position, meters
    #[getter]
    fn pos(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, &self.end6()[0..3])
    }

    /// GCRF velocity of satellite at end of propagation
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element GCRF velocity, meters/second
    #[getter]
    fn vel(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, &self.end6()[3..6])
    }

    /// 6-element GCRF state (pos + vel) at end of propagation (same as state_end)
    ///
    /// Returns:
    ///     numpy.ndarray: [x, y, z, vx, vy, vz] in meters and meters/second
    #[getter]
    fn state(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, self.end6())
    }

    /// 6-element GCRF state (pos + vel) at end of propagation
    ///
    /// Returns:
    ///     numpy.ndarray: [x, y, z, vx, vy, vz] in meters and meters/second
    #[getter]
    fn state_end(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, self.end6())
    }

    /// 6-element GCRF state (pos + vel) at begin of propagation
    ///
    /// Returns:
    ///     numpy.ndarray: [x, y, z, vx, vy, vz] in meters and meters/second
    #[getter]
    fn state_begin(&self, py: Python) -> PyResult<Py<PyAny>> {
        slice2py1d(py, each!(&self.0, r => &r.state_begin.as_slice()[0..6]))
    }

    /// State transition matrix between begin and end times
    ///
    /// Returns:
    ///     numpy.ndarray | None: 6x6 state transition matrix, or None if not computed.
    ///     Maps a perturbation of the begin state (meters, m/s) to the end state
    ///     (meters, m/s), so blocks are unitless, seconds, 1/seconds, unitless.
    #[getter]
    fn phi(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            PyPropResultType::R1(_) => Ok(py.None()),
            PyPropResultType::R7(r) => slice2py2d(
                py,
                r.state_end.block::<6, 6>(0, 1).transpose().as_slice(),
                6,
                6,
            ),
        }
    }

    fn __str__(&self) -> String {
        match &self.0 {
            PyPropResultType::R1(r) => to_string::<1>(r),
            PyPropResultType::R7(r) => to_string::<7>(r),
        }
    }

    /// Whether this result supports interpolation (dense output is available)
    #[getter]
    const fn can_interp(&self) -> bool {
        each!(&self.0, r => r.odesol.is_some() || r.gj_dense.is_some())
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::empty(py);
        (tp, d)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        self.0 = serde_pickle_from_slice(state.as_bytes(py), "propresult")?;
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let p = serde_pickle_to_vec(&self.0, "propresult")?;
        PyBytes::new(py, p.as_slice()).into_py_any(py)
    }

    /// Interpolate the GCRF state at one or more times between the begin and end times
    ///
    /// Args:
    ///     time (satkit.time | datetime.datetime | list): time(s) at which to interpolate
    ///     output_phi (bool): also return the 6x6 state transition matrix. Default False
    ///
    /// Returns:
    ///     numpy.ndarray: 6-element state [x, y, z, vx, vy, vz] in meters and m/s
    ///     for a single time; for a list / array of N times, one (N, 6) array
    ///     ((0, 6) for an empty list). With output_phi=True, a (state, phi) tuple,
    ///     where phi is the 6x6 state transition matrix, or a list of them for a
    ///     list of times
    ///
    /// Raises:
    ///     ValueError: if output_phi is True but the propagation did not compute
    ///         the state transition matrix (``propagate(..., output_phi=True)``),
    ///         or a time is outside the interpolation range
    #[pyo3(signature=(time, output_phi=false))]
    fn interp(&self, py: Python, time: Bound<'_, PyAny>, output_phi: bool) -> PyResult<Py<PyAny>> {
        // Without the STM there is no phi to return; this used to hand back
        // the bare state instead of the promised (state, phi) tuple
        if output_phi && matches!(self.0, PyPropResultType::R1(_)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "output_phi=True needs the state transition matrix, which this result \
                 does not have: propagate with output_phi=True",
            ));
        }
        let is_list = time.is_instance_of::<pyo3::types::PyList>()
            || time.is_instance_of::<numpy::PyArray1<Py<PyAny>>>();

        let times = (&time).to_time_vec()?;

        if is_list && !output_phi {
            // Batch interpolation — returns Nx6 numpy array
            let (flat, n): (Vec<f64>, usize) = each!(&self.0, r => {
                let results = r
                    .interp_batch(&times)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let flat = results
                    .iter()
                    .flat_map(|r| r.as_slice().iter().copied().take(6))
                    .collect();
                (flat, results.len())
            });
            slice2py2d(py, &flat, n, 6)
        } else if is_list {
            // Fallback for output_phi=true — need per-element processing
            times
                .iter()
                .map(|t| self.interp_at(py, t, output_phi))
                .collect::<PyResult<Vec<_>>>()?
                .into_py_any(py)
        } else {
            self.interp_at(py, &times[0], output_phi)
        }
    }
}

impl PyPropResult {
    /// End state, position and velocity (the first column for a result
    /// propagated with the state transition matrix)
    fn end6(&self) -> &[f64] {
        each!(&self.0, r => &r.state_end.as_slice()[0..6])
    }

    fn interp_at(&self, py: Python, time: &Instant, output_phi: bool) -> PyResult<Py<PyAny>> {
        let err =
            |e: satkit::orbitprop::Error| pyo3::exceptions::PyValueError::new_err(e.to_string());
        match &self.0 {
            PyPropResultType::R1(r) => vec2py(py, &r.interp(time).map_err(err)?),
            PyPropResultType::R7(r) => {
                let res = r.interp(time).map_err(err)?;
                let state = slice2py1d(py, &res.as_slice()[0..6])?;
                if !output_phi {
                    return Ok(state);
                }
                let phi = res.block::<6, 6>(0, 1).transpose();
                (state, slice2py2d(py, phi.as_slice(), 6, 6)?).into_py_any(py)
            }
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
        let sol2 = PyPropResult(
            serde_pickle::from_slice(v.as_slice(), serde_pickle::DeOptions::default()).unwrap(),
        );
        println!("sol2 = {:?}", sol2);
    }
}
