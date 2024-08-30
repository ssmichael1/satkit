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

///
/// High-precision orbit propagator
///
/// Propagator uses advanced Runga-Kutta integrators and includes the following
/// forces:
///
///    1) Earth gravity, with zonal gravity up to order 16 (default is 4)
///    2) Gravitational force of moon
///    3) Gravitational force of sun
///    4) Solar radiation pressure (with user-specified satellite model)
///    5) Atmospheric drag, with correction for space wither
///       (with user-specified satellite model)
///
///
/// Propagate statellite ephemeris (position, velocity in gcrs & time) to new time
/// and output new position and velocity
///
/// Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
///
/// Inputs:
///   
///     pos:   3-element numpy array representing satellite GCRF position in meters
///     vel:   3-element numpy array representing satellite GCRF velocity in m/s
///      tm:   astro.time object representing instant at which satellite is at "pos" & "vel"
///
/// Optional keyword arguments:
///
///
/// 4 ways of setting propagation end:
/// (one of these must be used)
///   
///         stoptime: astro.time object representing instant at
///                   which new position and velocity will be computed
///    duration_secs: duration in seconds from "tm" for at which new
///                   position and velocity will be computed.  
///    duration_days: duration in days from "tm" at which new position and
///                   velocity will be computed.  
///         duration: An astro.duration object setting duration from "tm"
///                   at which new position & velocity will be computed.
///
///  Other keywords:
///
///
///       output_phi: boolean inticating Output 6x6 state transition matrix
///                   between "starttime" and "stoptime"
///                   default is False
///     propsettings: "propsettings" object with input settings for
///                   the propagation. if left out, default will be used.
///    satproperties: "SatPropertiesStatic" object with drag and
///                   radiation pressure succeptibility of satellite.
///                   If left out, drag and radiation pressure are neglected
///                   Dynamic drag & radiation pressure models are not
///                   yet implemented
///     output_dense: boolean indicacting output dense ODE solution that can
///                   be used for interpolation of state between
///                  "starttime" and "stoptime".  Default is False
///           
///
///
/// Output: Python satprop object holding resultant state, propagation statistics,
///         and possibly dense output for interpolation
///
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
