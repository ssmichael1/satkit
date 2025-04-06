use super::pyduration::PyDuration;
use super::pyinstant::PyInstant;
use super::pypropresult::{PyPropResult, PyPropResultType};
use super::pypropsettings::PyPropSettings;
use super::pysatproperties::PySatProperties;
use super::pyutils::*;
use pyo3::IntoPyObjectExt;

use nalgebra as na;

use crate::orbitprop::SatProperties;
use crate::orbitprop::SatPropertiesStatic;
use crate::types::*;
use crate::Duration;
use crate::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use anyhow::{bail, Result};

/// High-precision orbit propagator
///
/// Propagate statellite ephemeris (position, velocity in gcrs & time) to new time via adaptive Runge-Kutta 9/8 ordinary differential equation (ODE) integration
///
/// Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
///
/// Inputs:
///   
///      state0 (npt.ArrayLike[float], optional): 6-element numpy array representing satellite position & velocity
///      start (satkit.time, optional): Start time of propagation, time of "state0"
///       stop (satkit.time, optional): Stop time of propagation
///
///
/// Optional keyword arguments:
///
///
/// 4 ways of setting propagation end:
/// (one of these must be used)
///   
///             stop: (satkit.time, optional): instant at which new position and
///                   velocity will be computed
///    duration_secs: (float, optional): duration in seconds from "tm" for at which new
///                   position and velocity will be computed.  
///    duration_days: (float, optional): duration in days from "tm" at which new position and
///                   velocity will be computed.  
///         duration: (satkit.duration, optional): An astro.duration object setting duration
///                   from "tm" at which new position & velocity will be computed.
///
///  Other keywords:
///
///
///       output_phi (bool): boolean inticating Output 6x6 state transition matrix
///                   between "starttime" and "stoptime"
///                   default is False
///     propsettings (satkit.propsettings): Settings for
///                   the propagation. if left out, default will be used.
///    satproperties (satkit.satproperties_static): object with drag and
///                   radiation pressure succeptibility of satellite.
///                   If left out, drag and radiation pressure are neglected
///                   Dynamic drag & radiation pressure models are not
///                   yet implemented
///     output_dense (bool): boolean indicacting output dense ODE solution that can
///                   be used for interpolation of state between
///                  "starttime" and "stoptime".  Default is False
///           
///
/// Returns:
///
///    satkit.propresult: object with new position and velocity, and possibly
///                       state transition matrix between "starttime" and "stoptime",
///                       and dense ODE solution that allow for interpolation, if requested
///
/// Raises:
///
///   RuntimeError: If "pos" or "vel" are not 3-element numpy arrays
///   RuntimeError: If neither "stoptime", "duration", "duration_secs", or "duration_days" are set
///   RuntimeError: If extraneous keyword arguments are passed
///
///
///    Notes:
///        * Propagator uses advanced Runga-Kutta integrators and includes the following forces:
///            * Earth gravity with higher-order zonal terms
///            * Sun, Moon gravity
///            * Radiation pressure
///            * Atmospheric drag: NRL-MISE 2000 density model, with option to include space weather effects (can be large)
///        * Stop time must be set by keyword argument, either explicitely or by duration
///        * Solid Earth tides are not (yet) included in the model
///
#[pyfunction(signature=(*args, **kwargs))]
pub fn propagate(
    args: &Bound<PyTuple>,
    mut kwargs: Option<&Bound<'_, PyDict>>,
) -> Result<PyObject> {
    let pypropsettings: Option<PyPropSettings> = kwargs_or_none(&mut kwargs, "propsettings")?;
    let propsettings = match pypropsettings {
        Some(p) => p.0,
        None => crate::orbitprop::PropSettings::default(),
    };

    let mut state0 = Vector6::zeros();
    let mut starttime: Instant = Instant::INVALID;
    let mut stoptime: Instant = Instant::INVALID;
    let mut output_phi: bool = false;
    let mut satproperties: Option<&dyn SatProperties> = None;
    let satproperties_static: SatPropertiesStatic;

    if args.len() > 0 {
        state0 = py_to_smatrix(&args.get_item(0)?)?;
    }
    if args.len() > 1 {
        starttime = args.get_item(1)?.extract::<PyInstant>()?.0;
    }
    if args.len() > 2 {
        stoptime = args.get_item(2)?.extract::<PyInstant>()?.0;
    }

    if let Some(kw) = kwargs {
        if let Some(kwp) = kw.get_item("pos")? {
            let pos = py_to_smatrix::<3, 1>(&kwp)?;
            state0[0] = pos[0];
            state0[1] = pos[1];
            state0[2] = pos[2];
            kw.del_item("pos")?;
        }
        if let Some(kwv) = kw.get_item("vel")? {
            let vel = py_to_smatrix::<3, 1>(&kwv)?;
            state0[3] = vel[0];
            state0[4] = vel[1];
            state0[5] = vel[2];
            kw.del_item("vel")?;
        }
        if let Some(kws) = kw.get_item("start")? {
            starttime = kws.extract::<PyInstant>()?.0;
            kw.del_item("start")?;
        }
        if let Some(kws) = kw.get_item("stop")? {
            stoptime = kws.extract::<PyInstant>()?.0;
            kw.del_item("stop")?;
        }
        if let Some(kwd) = kw.get_item("duration")? {
            stoptime = starttime + kwd.extract::<PyDuration>()?.0;
            kw.del_item("duration")?;
        }
        if let Some(kwd) = kw.get_item("duration_days")? {
            stoptime = starttime + Duration::from_days(kwd.extract::<f64>()?);
            kw.del_item("duration_days")?;
        }
        if let Some(kwd) = kw.get_item("duration_secs")? {
            stoptime = starttime + Duration::from_seconds(kwd.extract::<f64>()?);
            kw.del_item("duration_sec")?;
        }
        if let Some(kws) = kw.get_item("satproperties")? {
            satproperties_static = kws.extract::<PySatProperties>()?.0;
            satproperties = Some(&satproperties_static);
            kw.del_item("satproperties")?;
        }

        output_phi = kwargs_or_default(&mut kwargs, "output_phi", false)?;

        if !kw.is_empty() {
            let keystring: String = kw.iter().try_fold(String::from(""), |acc, (k, _v)| {
                let mut a2 = acc;
                a2.push_str(k.downcast::<PyString>()?.to_str()?);
                a2.push_str(", ");
                Ok::<_, PyErr>(a2)
            })?;
            bail!("Extraneous keyword arguments: {}", keystring);
        }
    }

    // Simple sate propagation
    if !output_phi {
        let res = crate::orbitprop::propagate(
            &state0,
            &starttime,
            &stoptime,
            &propsettings,
            satproperties,
        )?;
        pyo3::Python::with_gil(|py| -> Result<PyObject> {
            Ok(PyPropResult(PyPropResultType::R1(Box::new(res))).into_py_any(py)?)
        })
    }
    // Propagate with state transition matrix
    else {
        // Create the state to propagate
        let mut pv = na::SMatrix::<f64, 6, 7>::zeros();
        pv.fixed_view_mut::<6, 1>(0, 0).copy_from(&state0);
        pv.fixed_view_mut::<6, 6>(0, 1)
            .copy_from(&Matrix6::identity());

        let res =
            crate::orbitprop::propagate(&pv, &starttime, &stoptime, &propsettings, satproperties)?;
        pyo3::Python::with_gil(|py| -> Result<PyObject> {
            Ok(PyPropResult(PyPropResultType::R7(Box::new(res))).into_py_any(py)?)
        })
    }
}
