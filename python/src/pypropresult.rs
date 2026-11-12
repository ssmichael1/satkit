use pyo3::prelude::*;

use crate::pyinstant::{PyInstant, ToTimeVec};
use crate::pyutils::*;

use pyo3::types::PyBytes;
use pyo3::IntoPyObjectExt;

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
/// The step and force-evaluation counts of a high-precision orbit propagation,
/// for whichever integrator ran it (the adaptive Runge-Kutta pairs, RODAS4 or
/// the fixed-step Gauss-Jackson 8). Returned by ``propresult.stats``; not
/// constructed directly.
///
#[pyclass(name = "propstats", module = "satkit")]
pub struct PyPropStats {
    /// Number of force-model (derivative) evaluations
    ///
    /// Returns:
    ///     int: number of derivative function evaluations used in propagation
    #[pyo3(get)]
    num_eval: u32,

    /// Number of accepted integrator steps (for the fixed-step
    /// ``integrator.gauss_jackson8``, the number of steps taken)
    ///
    /// Returns:
    ///     int: number of accepted steps in propagation
    #[pyo3(get)]
    num_accept: u32,

    /// Number of steps the adaptive step-size control rejected and retried
    /// (always 0 for the fixed-step ``integrator.gauss_jackson8``)
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
/// The result also includes dense output that can be used for interpolation of states
/// between the begin and end times (``interp``), unless ``propsettings.enable_interp``
/// was False or the integrator has none (``integrator.rodas4``,
/// ``integrator.rkv98_nointerp``); ``can_interp`` says which.
///
/// Returned by ``satkit.propagate``; not constructed directly.
///
/// Attributes:
///
///    time_begin: satkit.time object representing the time at which the propagation began
///          time: satkit.time object representing the time at which the propagation ended
///                (same as time_end)
///         stats: satkit.propstats object with statistics about the propagation
///           pos: 3-element numpy array representing the final position of the satellite in GCRF meters
///           vel: 3-element numpy array representing the final velocity of the satellite in GCRF m/s
///         state: 6-element numpy array representing the final state of the satellite in GCRF,
///                a concatenation of pos (meters) and vel (m/s)
///           phi: 6x6 numpy array representing the state transition matrix between
///                the begin and end times, if requested (maps a begin-state perturbation
///                in meters, m/s to the end state in meters, m/s)
///    can_interp: boolean indicating whether the result includes dense output,
///                so that ``interp`` works between the begin and end times
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
    s.push_str(format!("   Can Interp: {}\n", r.can_interp()).as_str());
    s.push_str(format!("        Begin Time: {}", r.time_begin).as_str());
    s
}

#[pymethods]
impl PyPropResult {
    /// Time at which the propagation began
    ///
    /// Returns:
    ///     satkit.time: time at which ``state_begin`` is valid
    #[getter]
    fn time_begin(&self) -> PyInstant {
        PyInstant(each!(&self.0, r => r.time_begin))
    }

    /// Time of the end state (same as time_end)
    ///
    /// Returns:
    ///     satkit.time: time at which ``state`` / ``pos`` / ``vel`` are valid
    #[getter]
    fn time(&self) -> PyInstant {
        self.time_end()
    }

    /// Time of the end state
    ///
    /// Returns:
    ///     satkit.time: time at which ``state_end`` is valid
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

    /// Whether ``interp`` works on this result: True when the propagation
    /// stored dense output.
    ///
    /// False when ``propsettings.enable_interp`` was False, for
    /// ``integrator.rodas4`` and ``integrator.rkv98_nointerp`` (which have no
    /// interpolant) and for a zero-duration propagation; ``interp`` then
    /// raises ``ValueError``.
    ///
    /// Returns:
    ///     bool: whether dense output is available for interpolation
    #[getter]
    fn can_interp(&self) -> bool {
        each!(&self.0, r => r.can_interp())
    }

    /// Rebuild a result from its pickled byte state. Used internally by
    /// `__reduce__`; not part of the public API.
    #[staticmethod]
    fn _from_pickle(state: &[u8]) -> PyResult<Self> {
        Ok(Self(serde_pickle_from_slice(state, "propresult")?))
    }

    /// Pickle (and `copy.deepcopy`) support. `propresult` has no `__new__`
    /// (a result only comes from `propagate`), so it is rebuilt through the
    /// private `_from_pickle` staticmethod from a serde byte payload.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = serde_pickle_to_vec(&self.0, "propresult")?;
        let ctor = py.get_type::<Self>().getattr("_from_pickle")?;
        Ok((ctor, (PyBytes::new(py, &bytes),)))
    }

    /// Interpolate the GCRF state at one or more times between the begin and end times
    ///
    /// Args:
    ///     time (satkit.time | datetime.datetime | numpy.datetime64 | list | numpy.ndarray): time(s) at which to interpolate
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
    ///     ValueError: if the result has no dense output (``can_interp`` is
    ///         False), a time is outside the begin-to-end range, or output_phi
    ///         is True but the propagation did not compute the state transition
    ///         matrix (``propagate(..., output_phi=True)``)
    ///     TypeError: if time is not a time or a list / array of times (a
    ///         ``RuntimeError`` for a tuple, a number or a string)
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
        // A list or array of times (including a datetime64 array, which
        // used to be read as a single time) gives one result per time
        let input = (&time).to_time_input()?;
        // `can_interp` and `interp` agree: without dense output nothing is
        // interpolated, not even the begin time (which the core hands back
        // as a special case) or an empty list
        if !self.can_interp() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "this result has no dense output to interpolate (can_interp is False): \
                 propagate over a non-zero span with propsettings.enable_interp=True and an \
                 integrator other than rodas4 or rkv98_nointerp",
            ));
        }
        let times = input.times;

        if input.scalar {
            self.interp_at(py, &times[0], output_phi)
        } else if !output_phi {
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
        } else {
            // Fallback for output_phi=true — need per-element processing
            times
                .iter()
                .map(|t| self.interp_at(py, t, output_phi))
                .collect::<PyResult<Vec<_>>>()?
                .into_py_any(py)
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
        let sol = PyPropResult(PyPropResultType::R1(Box::new(PropagationResult::<1> {
            time_begin: Instant::INVALID,
            state_begin: satkit::mathtypes::Vector::<6>::zeros(),
            time_end: Instant::INVALID,
            state_end: satkit::mathtypes::Vector::<6>::zeros(),
            num_eval: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            next_step_secs: 0.0,
            odesol: None,
            gj_dense: None,
            integrator: satkit::orbitprop::Integrator::default(),
        })));
        println!("sol = {:?}", sol);
        let v = serde_pickle::to_vec(&sol.0, serde_pickle::SerOptions::default()).unwrap();
        let sol2 = PyPropResult(
            serde_pickle::from_slice(v.as_slice(), serde_pickle::DeOptions::default()).unwrap(),
        );
        println!("sol2 = {:?}", sol2);
    }
}
