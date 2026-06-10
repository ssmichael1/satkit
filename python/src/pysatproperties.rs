use satkit::orbitprop::SatPropertiesSimple;

use crate::pythrust::{py_thrusts_to_profile, PyThrust};
use crate::pyutils::{kwargs_or_default, reject_unused_kwargs};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

#[pyclass(name = "satproperties", module = "satkit", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySatProperties(pub SatPropertiesSimple);

#[pymethods]
impl PySatProperties {
    ///
    /// Create a static sat properties object
    /// setting satellite susceptibility to
    /// drag & radiation pressure
    ///
    /// With Cr A / m (m^2/kg),  radiation pressure
    /// and Cd A / m (m^2/kg), drag pressure
    /// passed in as arguments in that order, or set explicitly
    /// via the "craoverm" and "cdaoverm" keyword arguments
    ///
    /// Optionally, set continuous thrust arcs via the "thrusts"
    /// keyword argument, which takes a list of satkit.thrust objects
    ///
    /// If these are not set, default is 0
    ///
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(args: &Bound<PyTuple>, mut kwargs: Option<&Bound<'_, PyDict>>) -> Result<Self> {
        let mut craoverm: f64 = 0.0;
        let mut cdaoverm: f64 = 0.0;

        if args.len() > 0 {
            craoverm = args.get_item(0)?.extract::<f64>()?;
        }
        if args.len() > 1 {
            cdaoverm = args.get_item(1)?.extract::<f64>()?;
        }

        if kwargs.is_some() {
            craoverm = kwargs_or_default(&mut kwargs, "craoverm", craoverm)?;
            cdaoverm = kwargs_or_default(&mut kwargs, "cdaoverm", cdaoverm)?;
        }

        let mut props = SatPropertiesSimple::new(cdaoverm, craoverm);

        // Handle thrusts keyword
        if let Some(kw) = kwargs {
            if let Some(thrusts_obj) = kw.get_item("thrusts")? {
                let thrusts: Vec<PyThrust> = thrusts_obj.extract()?;
                props = props.with_thrust(py_thrusts_to_profile(thrusts));
                kw.del_item("thrusts")?;
            }
            reject_unused_kwargs(kw)?;
        }

        Ok(Self(props))
    }

    /// Get the satellite's susceptibility to radiation pressure
    ///
    /// Returns:
    ///     float: Cr A / m (m^2/kg)
    #[getter]
    const fn get_craoverm(&self) -> f64 {
        self.0.craoverm
    }

    /// Get the satellite's susceptibility to drag
    ///
    /// Returns:
    ///     float: Cd A / m (m^2/kg)
    #[getter]
    const fn get_cdaoverm(&self) -> f64 {
        self.0.cdaoverm
    }

    /// Set the satellite's susceptibility to radiation pressure
    ///
    /// Args:
    ///     craoverm (float): Cr A / m (m^2/kg)
    #[setter]
    fn set_craoverm(&mut self, craoverm: f64) {
        self.0.craoverm = craoverm;
    }

    /// Set the satellite's susceptibility to drag
    ///
    /// Args:
    ///     cdaoverm (float): Cd A / m (m^2/kg)
    #[setter]
    fn set_cdaoverm(&mut self, cdaoverm: f64) {
        self.0.cdaoverm = cdaoverm;
    }

    /// Get the list of thrust arcs
    ///
    /// Returns:
    ///     list[satkit.thrust]: List of continuous thrust arcs
    #[getter]
    fn get_thrusts(&self) -> Vec<PyThrust> {
        self.0
            .thrust
            .thrusts
            .iter()
            .map(|t| PyThrust(t.clone()))
            .collect()
    }

    /// Set the thrust arcs
    ///
    /// Args:
    ///     thrusts (list[satkit.thrust]): List of continuous thrust arcs
    #[setter]
    fn set_thrusts(&mut self, thrusts: Vec<PyThrust>) {
        self.0.thrust = py_thrusts_to_profile(thrusts);
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> Result<()> {
        let state = state.as_bytes(py);
        if state.len() < 16 {
            bail!("Invalid serialization length");
        }
        let craoverm = f64::from_le_bytes(state[0..8].try_into()?);
        let cdaoverm = f64::from_le_bytes(state[8..16].try_into()?);
        self.0.cdaoverm = cdaoverm;
        self.0.craoverm = craoverm;
        self.0.thrust = satkit::orbitprop::ThrustProfile::default();

        // Each thrust arc: 24 (accel) + 1 (frame tag) + 8 (start) + 8 (end) = 41 bytes
        let mut offset = 16;
        while offset + 41 <= state.len() {
            let accel = satkit::mathtypes::Vector3::from_slice(unsafe {
                std::slice::from_raw_parts(state[offset..].as_ptr() as *const f64, 3)
            });
            offset += 24;
            let frame = match state[offset] {
                1 => satkit::Frame::RTN,
                _ => satkit::Frame::GCRF,
            };
            offset += 1;
            let start = satkit::Instant::from_mjd_with_scale(
                f64::from_le_bytes(state[offset..offset + 8].try_into()?),
                satkit::TimeScale::TAI,
            );
            offset += 8;
            let end = satkit::Instant::from_mjd_with_scale(
                f64::from_le_bytes(state[offset..offset + 8].try_into()?),
                satkit::TimeScale::TAI,
            );
            offset += 8;
            self.0
                .thrust
                .thrusts
                .push(satkit::orbitprop::ContinuousThrust::new(
                    accel, frame, start, end,
                ));
        }
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        // Each thrust arc: 24 (accel) + 1 (frame tag) + 8 (start) + 8 (end) = 41 bytes
        let thrust_len = self.0.thrust.thrusts.len() * 41;
        let mut raw = vec![0u8; 16 + thrust_len];
        raw[0..8].clone_from_slice(&self.0.craoverm.to_le_bytes());
        raw[8..16].clone_from_slice(&self.0.cdaoverm.to_le_bytes());
        let mut offset = 16;
        for t in &self.0.thrust.thrusts {
            unsafe {
                raw[offset..offset + 24].clone_from_slice(std::slice::from_raw_parts(
                    t.accel.as_slice().as_ptr() as *const u8,
                    24,
                ));
            }
            offset += 24;
            raw[offset] = match t.frame {
                satkit::Frame::RTN => 1,
                _ => 0,
            };
            offset += 1;
            raw[offset..offset + 8].clone_from_slice(
                &t.start
                    .as_mjd_with_scale(satkit::TimeScale::TAI)
                    .to_le_bytes(),
            );
            offset += 8;
            raw[offset..offset + 8].clone_from_slice(
                &t.end
                    .as_mjd_with_scale(satkit::TimeScale::TAI)
                    .to_le_bytes(),
            );
            offset += 8;
        }
        pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}
