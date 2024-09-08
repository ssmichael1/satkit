use super::pyastrotime::PyAstroTime;
use super::pyduration::PyDuration;
use super::pypropresult::{PyPropResult, PyPropResultType};
use super::pypropsettings::PyPropSettings;
use super::pysatproperties::PySatProperties;
use super::pyutils::*;

use nalgebra as na;
use numpy as np;
use numpy::PyArrayMethods;

use crate::orbitprop;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// High-precision orbit propagator
///
/// Propagate statellite ephemeris (position, velocity in gcrs & time) to new time
/// and output new position and velocity
///
/// Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
///
/// Args:
///     pos (numpy.ndarray): Satellite Cartesian GCRF position in meters
///     vel (numpy.ndarray): Satellite Cartesian GCRF velocity in m/s
///        tm (satkit.time): Instant at which satellite is at "pos" & "vel"
///
/// Keyword Args:
///         stoptime (satkit.time, optional): astro.time object representing instant at
///                   which new position and velocity will be computed
///    duration_secs (float, optional): duration in seconds from "tm" for at which new
///                   position and velocity will be computed.  
///    duration_days (float, optional): duration in days from "tm" at which new position and
///                   velocity will be computed.  
///         duration (satkit.duration, optional): An astro.duration object setting duration
///                   from "tm" at which new position & velocity will be computed.
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
#[pyfunction(signature=(pos, vel, start, **kwargs))]
pub fn propagate(
    pos: &Bound<'_, np::PyArray1<f64>>,
    vel: &Bound<'_, np::PyArray1<f64>>,
    start: &PyAstroTime,
    mut kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if pos.len().unwrap() != 3 || vel.len().unwrap() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Position and velocity must be 1-d numpy arrays with length 3",
        ));
    }
    let pypropsettings: Option<PyPropSettings> = kwargs_or_none(&mut kwargs, "propsettings")?;

    let propsettings = match pypropsettings {
        Some(p) => p.inner,
        None => crate::orbitprop::PropSettings::default(),
    };

    let duration_secs: Option<f64> = kwargs_or_none(&mut kwargs, "duration_secs")?;
    let pyduration: Option<PyDuration> = kwargs_or_none(&mut kwargs, "duration")?;
    let duration_days: Option<f64> = kwargs_or_none(&mut kwargs, "duration_days")?;
    let pystoptime: Option<PyAstroTime> = kwargs_or_none(&mut kwargs, "stoptime")?;
    let output_phi: bool = kwargs_or_default(&mut kwargs, "output_phi", false)?;
    let output_dense: bool = kwargs_or_default(&mut kwargs, "output_dense", false)?;
    let pysatproperties: Option<PySatProperties> = kwargs_or_none(&mut kwargs, "satproperties")?;

    // Look for extraneous kwargs and return error
    if kwargs.is_some() {
        if !kwargs.unwrap().is_empty() {
            let keystring: String =
                kwargs
                    .unwrap()
                    .iter()
                    .fold(String::from(""), |acc, (k, _v)| {
                        let mut a2 = acc.clone();
                        a2.push_str(k.downcast::<PyString>().unwrap().to_str().unwrap());
                        a2.push_str(", ");
                        a2
                    });
            let s = format!("Invalid kwargs: {}", keystring);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(s));
        }
    }

    if duration_days == None && pystoptime == None && duration_secs == None && pyduration.is_none()
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Must set either duration or stop time",
        ));
    }

    // Multiple ways of setting stop time ; this is complicated
    let stoptime = match pystoptime {
        Some(p) => p.inner,
        None => {
            start.inner
                + match pyduration {
                    Some(v) => v.inner.days(),
                    None => match duration_days {
                        Some(v) => v,
                        None => duration_secs.unwrap() / 86400.0,
                    },
                }
        }
    };

    let satproperties: Option<&dyn orbitprop::SatProperties> = match &pysatproperties {
        None => None,
        Some(v) => Some(&v.inner),
    };

    // Simple sate propagation
    if output_phi == false {
        // Create the state to propagate
        let mut pv = na::SMatrix::<f64, 6, 1>::zeros();
        pv.fixed_view_mut::<3, 1>(0, 0)
            .copy_from_slice(unsafe { pos.as_slice().unwrap() });
        pv.fixed_view_mut::<3, 1>(3, 0)
            .copy_from_slice(unsafe { vel.as_slice().unwrap() });
        let res = crate::orbitprop::propagate(
            &pv,
            &start.inner,
            &stoptime,
            &propsettings,
            satproperties,
            output_dense,
        )
        .unwrap();
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            Ok(PyPropResult {
                inner: PyPropResultType::R1(res),
            }
            .into_py(py))
        })
    }
    // Propagate with state transition matrix
    else {
        // Create the state to propagate
        let mut pv = na::SMatrix::<f64, 6, 7>::zeros();
        pv.fixed_view_mut::<3, 1>(0, 0)
            .copy_from_slice(unsafe { pos.as_slice().unwrap() });
        pv.fixed_view_mut::<3, 1>(3, 0)
            .copy_from_slice(unsafe { vel.as_slice().unwrap() });
        pv.fixed_view_mut::<6, 6>(0, 1)
            .copy_from(&na::Matrix6::<f64>::identity());
        let res = crate::orbitprop::propagate(
            &pv,
            &start.inner,
            &stoptime,
            &propsettings,
            satproperties,
            output_dense,
        )
        .unwrap();
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            Ok(PyPropResult {
                inner: PyPropResultType::R7(res),
            }
            .into_py(py))
        })
    }
}
