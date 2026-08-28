use satkit::orbitprop::SatPropertiesSimple;

use crate::pyecom::{ecom_from_block, encode_ecom_block, PyEcomParams, ECOM_BLOCK_LEN};
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
    /// keyword argument, which takes a list of satkit.thrust objects,
    /// and ECOM empirical solar-radiation-pressure coefficients via the
    /// "ecom" keyword argument (a satkit.ecomparams, added to the
    /// cannonball term; use craoverm=0 for a pure ECOM model)
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
            if let Some(ecom_obj) = kw.get_item("ecom")? {
                if !ecom_obj.is_none() {
                    let ecom = ecom_obj
                        .cast::<PyEcomParams>()
                        .map_err(|e| anyhow::anyhow!("ecom must be a satkit.ecomparams: {e}"))?
                        .borrow()
                        .0;
                    props = props.with_ecom(ecom);
                }
                kw.del_item("ecom")?;
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

    /// Get the ECOM solar-radiation-pressure coefficients, or None
    ///
    /// When set, the ECOM acceleration (see satkit.ecomparams for the DYB
    /// frame, sign and eclipse conventions) is added to the cannonball term
    /// craoverm; use craoverm=0 for a pure ECOM model. The "ECOM Solar
    /// Radiation Pressure" tutorial shows how to fit the coefficients to
    /// IGS GPS orbits.
    ///
    /// Returns:
    ///     satkit.ecomparams | None
    #[getter]
    fn get_ecom(&self) -> Option<PyEcomParams> {
        self.0.ecom.map(PyEcomParams)
    }

    /// Set (or clear, with None) the ECOM solar-radiation-pressure coefficients
    ///
    /// Args:
    ///     ecom (satkit.ecomparams | None)
    #[setter]
    fn set_ecom(&mut self, ecom: Option<PyEcomParams>) {
        self.0.ecom = ecom.map(|e| e.0);
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> Result<()> {
        let state = state.as_bytes(py);
        // Self-describing format:
        //   [0]       version byte (1 or 2)
        //   [1..9]    craoverm (f64)
        //   [9..17]   cdaoverm (f64)
        //   [17..21]  thrust-arc count (u32 little-endian)
        //   [..]      count * 41-byte arcs: 24 accel, 1 frame tag, 8 start, 8 end
        // v2 appends:
        //   [..]      has_ecom (u8); if 1, an ECOM block (see pyecom.rs)
        const HEADER: usize = 1 + 8 + 8 + 4;
        if state.len() < HEADER {
            bail!("invalid satproperties pickle: truncated header");
        }
        let version = state[0];
        if version != 1 && version != 2 {
            bail!(
                "unsupported satproperties pickle version {} (expected 1 or 2)",
                version
            );
        }
        let read_f64 = |at: usize| f64::from_le_bytes(state[at..at + 8].try_into().unwrap());

        self.0.craoverm = read_f64(1);
        self.0.cdaoverm = read_f64(9);
        self.0.thrust = satkit::orbitprop::ThrustProfile::default();

        let count = u32::from_le_bytes(state[17..21].try_into()?) as usize;
        let thrust_end = HEADER + count * 41;
        if state.len() < thrust_end {
            bail!("invalid satproperties pickle: thrust block length mismatch");
        }
        let mut offset = HEADER;
        for _ in 0..count {
            let mut accel = [0.0f64; 3];
            for (i, v) in accel.iter_mut().enumerate() {
                *v = read_f64(offset + i * 8);
            }
            offset += 24;
            let frame = crate::pyutils::maneuver_frame_from_u8(state[offset])?;
            offset += 1;
            let start =
                satkit::Instant::from_mjd_with_scale(read_f64(offset), satkit::TimeScale::TAI);
            offset += 8;
            let end =
                satkit::Instant::from_mjd_with_scale(read_f64(offset), satkit::TimeScale::TAI);
            offset += 8;
            self.0.thrust.thrusts.push(
                satkit::orbitprop::ContinuousThrust::new(
                    satkit::mathtypes::Vector3::from_slice(&accel),
                    frame,
                    start,
                    end,
                )
                .map_err(|e| anyhow::anyhow!("invalid thrust in pickle: {e}"))?,
            );
        }

        self.0.ecom = None;
        let tail = &state[thrust_end..];
        match version {
            1 => {
                if !tail.is_empty() {
                    bail!("invalid satproperties pickle: trailing bytes in v1 format");
                }
            }
            _ => {
                if tail.is_empty() {
                    bail!("invalid satproperties pickle: missing ECOM flag");
                }
                match tail[0] {
                    0 => {
                        if tail.len() != 1 {
                            bail!("invalid satproperties pickle: trailing bytes after ECOM flag");
                        }
                    }
                    1 => {
                        if tail.len() != 1 + ECOM_BLOCK_LEN {
                            bail!("invalid satproperties pickle: ECOM block length mismatch");
                        }
                        self.0.ecom = Some(ecom_from_block(&tail[1..])?);
                    }
                    other => bail!("invalid satproperties pickle: bad ECOM flag {other}"),
                }
            }
        }
        Ok(())
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        // See `__setstate__` for the format. Values are written little-endian via
        // `to_le_bytes`, so there is no alignment assumption on the buffer.
        let mut raw: Vec<u8> =
            Vec::with_capacity(21 + self.0.thrust.thrusts.len() * 41 + 1 + ECOM_BLOCK_LEN);
        raw.push(2u8); // version
        raw.extend_from_slice(&self.0.craoverm.to_le_bytes());
        raw.extend_from_slice(&self.0.cdaoverm.to_le_bytes());
        raw.extend_from_slice(&(self.0.thrust.thrusts.len() as u32).to_le_bytes());
        for t in &self.0.thrust.thrusts {
            for v in t.accel.as_slice() {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            raw.push(crate::pyutils::maneuver_frame_to_u8(t.frame)?);
            raw.extend_from_slice(
                &t.start
                    .as_mjd_with_scale(satkit::TimeScale::TAI)
                    .to_le_bytes(),
            );
            raw.extend_from_slice(
                &t.end
                    .as_mjd_with_scale(satkit::TimeScale::TAI)
                    .to_le_bytes(),
            );
        }
        match &self.0.ecom {
            Some(e) => {
                raw.push(1u8);
                raw.extend_from_slice(&encode_ecom_block(e));
            }
            None => raw.push(0u8),
        }
        pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}
