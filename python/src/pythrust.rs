use pyo3::prelude::*;

use crate::pyframes::PyFrame;
use crate::pyinstant::PyInstant;
use crate::pyutils::py_to_smatrix;

use satkit::mathtypes::*;
use satkit::orbitprop::{ContinuousThrust, ThrustProfile};
use satkit::Frame;

use anyhow::Result;

/// Python wrapper for ContinuousThrust
#[pyclass(name = "thrust", module = "satkit", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyThrust(pub ContinuousThrust);

#[pymethods]
impl PyThrust {
    /// Create a constant thrust acceleration.
    ///
    /// Args:
    ///     accel (array-like): 3-element acceleration vector [m/s^2]
    ///     start (satkit.time): Start time of thrust arc
    ///     end (satkit.time): End time of thrust arc
    ///     frame (satkit.frame): Coordinate frame. Supported values:
    ///
    ///         - ``frame.GCRF`` — inertial Cartesian
    ///         - ``frame.RTN`` — radial / tangential / normal (a.k.a. RSW, RIC)
    ///         - ``frame.NTW`` — normal-to-velocity / tangent / cross-track
    ///           (use this for thrust along the velocity vector)
    ///         - ``frame.LVLH`` — Local Vertical / Local Horizontal
    ///
    ///     The frame argument is required — there is no default.
    #[staticmethod]
    fn constant(
        accel: &Bound<PyAny>,
        start: PyInstant,
        end: PyInstant,
        frame: PyFrame,
    ) -> Result<Self> {
        let accel_vec: Vector3 = py_to_smatrix(accel)?;
        let rust_frame: Frame = frame.into();
        // Frame validation happens in ContinuousThrust::new.
        Ok(Self(ContinuousThrust::new(
            accel_vec, rust_frame, start.0, end.0,
        )?))
    }

    /// Get the acceleration vector
    ///
    /// Returns:
    ///     list[float]: 3-element acceleration vector in the thrust's frame [m/s^2]
    #[getter]
    fn get_accel(&self) -> [f64; 3] {
        [self.0.accel[0], self.0.accel[1], self.0.accel[2]]
    }

    /// Get the frame
    #[getter]
    fn get_frame(&self) -> PyFrame {
        PyFrame::from(self.0.frame)
    }

    /// Get the start time
    #[getter]
    fn get_start(&self) -> PyInstant {
        PyInstant(self.0.start)
    }

    /// Get the end time
    #[getter]
    fn get_end(&self) -> PyInstant {
        PyInstant(self.0.end)
    }

    fn __repr__(&self) -> String {
        format!(
            "thrust(accel=[{:.6e}, {:.6e}, {:.6e}], frame={}, start={}, end={})",
            self.0.accel[0],
            self.0.accel[1],
            self.0.accel[2],
            self.0.frame,
            self.0.start,
            self.0.end,
        )
    }

    /// Reconstruct a thrust arc from its pickled byte state. Used internally by
    /// `__reduce__`; not part of the public API.
    #[staticmethod]
    fn _from_pickle(state: &[u8]) -> PyResult<Self> {
        let ct: ContinuousThrust =
            serde_pickle::from_slice(state, Default::default()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid thrust pickle: {e}"))
            })?;
        // serde deserializes the Frame field without the construction-time
        // validation in ContinuousThrust::new; reject unsupported frames here
        // so a doctored pickle can't smuggle in an Earth frame that would
        // panic during propagation.
        crate::pyutils::maneuver_frame_to_u8(ct.frame)?;
        Ok(Self(ct))
    }

    /// Pickle support. `thrust` has no `__new__`, so we reconstruct via the
    /// private `_from_pickle` staticmethod with a self-describing byte payload
    /// (serde), avoiding any dependence on the picklability of the frame/time
    /// wrapper types.
    fn __reduce__(&self, py: Python) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        use pyo3::IntoPyObjectExt;
        let bytes = serde_pickle::to_vec(&self.0, Default::default()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to serialize thrust: {e}"))
        })?;
        let ctor = py.get_type::<Self>().getattr("_from_pickle")?;
        let args = (pyo3::types::PyBytes::new(py, &bytes),).into_py_any(py)?;
        Ok((ctor.into_py_any(py)?, args))
    }
}

/// Convert a Python list of PyThrust objects to a ThrustProfile
pub fn py_thrusts_to_profile(thrusts: Vec<PyThrust>) -> ThrustProfile {
    ThrustProfile::new(thrusts.into_iter().map(|t| t.0).collect())
}
