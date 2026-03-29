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
    /// Create a constant thrust acceleration
    ///
    /// Args:
    ///     accel (array-like): 3-element acceleration vector [m/s^2]
    ///     start (satkit.time): Start time of thrust arc
    ///     end (satkit.time): End time of thrust arc
    ///     frame (satkit.frame): Coordinate frame - frame.GCRF or frame.RIC (default: frame.GCRF)
    ///
    /// RIC components are [radial, in-track, cross-track]
    #[staticmethod]
    #[pyo3(signature = (accel, start, end, frame=PyFrame::GCRF))]
    fn constant(
        accel: &Bound<PyAny>,
        start: PyInstant,
        end: PyInstant,
        frame: PyFrame,
    ) -> Result<Self> {
        let accel_vec: Vector3 = py_to_smatrix(accel)?;
        let rust_frame: Frame = frame.into();

        match rust_frame {
            Frame::GCRF | Frame::RIC => {}
            _ => anyhow::bail!(
                "Invalid frame for thrust: {}. Must be frame.GCRF or frame.RIC",
                rust_frame
            ),
        }

        Ok(Self(ContinuousThrust::new(
            accel_vec,
            rust_frame,
            start.0,
            end.0,
        )))
    }

    /// Get the acceleration vector
    #[getter]
    fn get_accel(&self) -> [f64; 3] {
        [self.0.accel[0], self.0.accel[1], self.0.accel[2]]
    }

    /// Get the frame
    #[getter]
    fn get_frame(&self) -> PyFrame {
        PyFrame::from(self.0.frame.clone())
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
}

/// Convert a Python list of PyThrust objects to a ThrustProfile
pub fn py_thrusts_to_profile(thrusts: Vec<PyThrust>) -> ThrustProfile {
    ThrustProfile::new(thrusts.into_iter().map(|t| t.0).collect())
}
