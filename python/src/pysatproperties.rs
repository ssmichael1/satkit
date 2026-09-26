use satkit::orbitprop::SatPropertiesSimple;

use crate::pyecom::{ecom_from_block, encode_ecom_block, PyEcomParams, ECOM_BLOCK_LEN};
use crate::pythrust::{py_thrusts_to_profile, PyThrust};
use crate::pyutils::invalid_value;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

crate::arg_extractor!(cdaoverm_arg: f64, |_| invalid_value("cdaoverm"));
crate::arg_extractor!(craoverm_arg: f64, |_| invalid_value("craoverm"));
crate::arg_extractor!(ecom_arg: Option<PyEcomParams>, |e| {
    pyo3::exceptions::PyRuntimeError::new_err(format!("ecom must be a satkit.ecomparams: {e}"))
});
// `satproperties=` of `propagate` and `satstate.propagate`
crate::arg_extractor!(pub(crate) satproperties_arg: Option<PySatProperties>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid satproperties: {e}"))
});

/// Satellite properties relevant for drag, radiation pressure, and thrust
///
/// This class lets the satellite radiation pressure, drag,
/// and thrust parameters be set for duration of propagation.
///
/// Attributes:
///     cdaoverm (float): Coefficient of drag times area over mass in m^2/kg
///     craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
///     thrusts (list[thrust]): List of continuous thrust arcs
///     ecom (ecomparams | None): ECOM empirical solar-radiation-pressure
///         coefficients, added to the cannonball term (use ``craoverm=0``
///         for a pure ECOM model). See :class:`ecomparams` for the
///         conventions and the "ECOM Solar Radiation Pressure" tutorial
///         for a fit against IGS GPS orbits.
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
    /// All arguments are keyword-only:
    ///
    /// Keyword Args:
    ///     cdaoverm (float): Cd A / m (m^2/kg), susceptibility to drag. Default 0
    ///     craoverm (float): Cr A / m (m^2/kg), susceptibility to radiation
    ///         pressure. Default 0
    ///     thrusts (list[satkit.thrust] | None): continuous thrust arcs. Default None
    ///     ecom (satkit.ecomparams | None): ECOM empirical solar-radiation-pressure
    ///         coefficients, added to the cannonball term (use craoverm=0 for a
    ///         pure ECOM model). Default None
    ///
    /// Raises:
    ///     TypeError: if any argument is passed positionally. (Releases before
    ///         this change read positional arguments as (craoverm, cdaoverm),
    ///         the reverse of the documented order, so positional calls are
    ///         refused rather than reinterpreted.)
    ///
    #[new]
    // `*args` only catches positional calls to refuse them with the message
    // below; `text_signature` hides it.
    #[pyo3(
        signature=(*args, cdaoverm=0.0, craoverm=0.0, thrusts=None, ecom=None),
        text_signature = "(*, cdaoverm=0.0, craoverm=0.0, thrusts=None, ecom=None)"
    )]
    fn new(
        args: &Bound<PyTuple>,
        #[pyo3(from_py_with = cdaoverm_arg)] cdaoverm: f64,
        #[pyo3(from_py_with = craoverm_arg)] craoverm: f64,
        thrusts: Option<Vec<PyThrust>>,
        #[pyo3(from_py_with = ecom_arg)] ecom: Option<PyEcomParams>,
    ) -> PyResult<Self> {
        if !args.is_empty() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "satproperties() takes keyword arguments only: cdaoverm=, craoverm=, \
                 thrusts=, ecom= ({} positional argument{} given; earlier releases \
                 read positional arguments as (craoverm, cdaoverm), the reverse of \
                 the documented order)",
                args.len(),
                if args.len() == 1 { "" } else { "s" }
            )));
        }
        let mut props = SatPropertiesSimple::new(cdaoverm, craoverm);
        if let Some(thrusts) = thrusts {
            props = props.with_thrust(py_thrusts_to_profile(thrusts));
        }
        if let Some(ecom) = ecom {
            props = props.with_ecom(ecom.0);
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
        //   [0]       version byte (1, 2 or 3)
        //   [1..9]    craoverm (f64)
        //   [9..17]   cdaoverm (f64)
        //   [17..21]  thrust-arc count (u32 little-endian)
        //   [..]      count * 41-byte arcs: 24 accel, 1 frame tag, 8 start, 8 end
        // v2 and v3 append:
        //   [..]      has_ecom (u8); if 1, an ECOM block (see pyecom.rs)
        // Thrust start/end times: v3 stores the Instant's raw i64
        // microseconds (exact); v1/v2 (satkit <= 0.23) a TAI MJD as f64,
        // which lost a microsecond ~1% of the time, still read (rounded).
        const HEADER: usize = 1 + 8 + 8 + 4;
        if state.len() < HEADER {
            bail!("invalid satproperties pickle: truncated header");
        }
        let version = state[0];
        if !(1..=3).contains(&version) {
            bail!(
                "unsupported satproperties pickle version {} (expected 1, 2 or 3)",
                version
            );
        }
        let read_f64 = |at: usize| f64::from_le_bytes(state[at..at + 8].try_into().unwrap());
        let read_time = |at: usize| match version {
            1 | 2 => satkit::Instant::from_mjd_with_scale(read_f64(at), satkit::TimeScale::TAI),
            _ => satkit::Instant::new(i64::from_le_bytes(state[at..at + 8].try_into().unwrap())),
        };

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
            let start = read_time(offset);
            offset += 8;
            let end = read_time(offset);
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
        raw.push(3u8); // version
        raw.extend_from_slice(&self.0.craoverm.to_le_bytes());
        raw.extend_from_slice(&self.0.cdaoverm.to_le_bytes());
        raw.extend_from_slice(&(self.0.thrust.thrusts.len() as u32).to_le_bytes());
        for t in &self.0.thrust.thrusts {
            for v in t.accel.as_slice() {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            raw.push(crate::pyutils::maneuver_frame_to_u8(t.frame)?);
            raw.extend_from_slice(&t.start.raw.to_le_bytes());
            raw.extend_from_slice(&t.end.raw.to_le_bytes());
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
