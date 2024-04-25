use super::pyastrotime::PyAstroTime;
use super::pyduration::PyDuration;
use super::pypropsettings::PyPropSettings;
use super::pysatproperties::PySatProperties;
use super::pyutils::*;

use nalgebra as na;
use numpy as np;
use numpy::PyArrayMethods;

use crate::orbitprop;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

fn lowlevel_propagate<const C: usize>(
    pos: &np::PyArray1<f64>,
    vel: &np::PyArray1<f64>,
    start: &crate::AstroTime,
    stop: &crate::AstroTime,
    dt: Option<f64>,
    pysatproperties: &Option<PySatProperties>,
    propsettings: &orbitprop::PropSettings,
) -> PyResult<Py<PyAny>> {
    // Create the state to propagate
    let mut pv = na::SMatrix::<f64, 6, C>::zeros();
    pv.fixed_view_mut::<3, 1>(0, 0)
        .copy_from_slice(unsafe { pos.as_slice().unwrap() });
    pv.fixed_view_mut::<3, 1>(3, 0)
        .copy_from_slice(unsafe { vel.as_slice().unwrap() });
    if C > 1 {
        pv.fixed_view_mut::<6, 6>(0, 1)
            .copy_from(&na::Matrix6::<f64>::identity());
    }

    let satproperties: Option<&dyn orbitprop::SatProperties> = match pysatproperties {
        None => None,
        Some(v) => Some(&v.inner),
    };

    // Finally, do the propagation
    let res =
        match crate::orbitprop::propagate(&pv, &start, &stop, dt, &propsettings, satproperties) {
            Ok(v) => v,
            Err(e) => {
                let estring = format!("Error propagating: {}", e.to_string());
                return Err(pyo3::exceptions::PyRuntimeError::new_err(estring));
            }
        };

    pyo3::Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        let r = PyDict::new_bound(py);
        let d = PyDict::new_bound(py);
        d.set_item("num_eval", res.num_eval)?;
        d.set_item("accepted_steps", res.accepted_steps)?;
        d.set_item("rejected_steps", res.rejected_steps)?;

        let tm: Vec<Py<PyAny>> = res
            .time
            .iter()
            .map(|x| PyAstroTime { inner: x.clone() }.into_py(py))
            .collect();

        let n = res.state.len();
        let pos = unsafe { np::PyArray2::<f64>::new_bound(py, [n, 3], false) };
        for idx in 0..n {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    res.state[idx].as_ptr(),
                    pos.as_raw_array_mut().as_mut_ptr().offset(idx as isize * 3),
                    3,
                );
            }
        }
        let vel = unsafe { np::PyArray2::<f64>::new_bound(py, [n, 3], false) };
        for idx in 0..n {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    res.state[idx].as_ptr().offset(3),
                    vel.as_raw_array_mut().as_mut_ptr().offset(idx as isize * 3),
                    3,
                );
            }
        }

        r.set_item("stats", d)?;
        r.set_item("time", tm)?;
        r.set_item("pos", pos)?;
        r.set_item("vel", vel)?;

        if C > 1 {
            let phi = unsafe { np::PyArray3::<f64>::new_bound(py, [n, 6, 6], false) };
            for idx in 0..n {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        res.state[idx].as_ptr().offset(6),
                        phi.as_raw_array_mut()
                            .as_mut_ptr()
                            .offset(idx as isize * 36),
                        36,
                    );
                }
            }
            r.set_item("Phi", phi)?;
        }
        Ok(r.to_object(py))
    })
}

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
///
///  3 ways of setting smaller interval over which to compute solution:
///  (defualt is none, i.e. solution only computed at propagation end)
///
///          dt_secs: Interval in seconds between "starttime" and "stoptime"
///                   at which solution will also be computed
///          dt_days: Interval in days between "starttime" and "stoptime" at which
///                   solution will also be computed
///               dt: astro.duration representing interval over which
///                   new position & velocity will be computed
///
///
///  Other keywords:
///
///
///       output_phi: Output 6x6 state transition matrix between "starttime" and
///                   "stoptime" (and at intervals, if specified)
///                   default is False
///     propsettings: "propsettings" object with input settings for
///                   the propagation. if left out, default will be used.
///    satproperties: "SatPropertiesStatic" object with drag and
///                   radiation pressure succeptibility of satellite.
///                   If left out, drag and radiation pressure are neglected
///                   Dynamic drag & radiation pressure models are not
///                   yet implemented
///
///
/// Output: Python dictionary with the following elements:
///  
///     "time": list of astro.time objects at which solution is computed
///      "pos": GCRF position in meters at "time".  Output is a Nx3 numpy
///             matrix, where N is the length of the output "time" list
///      "vel": GCRF velocity in meters / second at "time".  Output is a
///             Nx3 numpy matrix, where N is the length of the output
///             "time" list
///      "Phi": 6x6 State transition matrix corresponding to each time.
///             Output is Nx6x6 numpy matrix, where N is the lenght of
///             the output "time" list. Not included if output_phi
///             kwarg is set to false (the default)
///    "stats": Python dictionary with statistics for the propagation.  
///             This includes:
///                   "num_eval": Number of function evaluations of the force model
///                               required to get solution with desired accuracy
///             "accepted_steps": Accepted steps in the adpative Runga-Kutta solver
///             "rejected_steps": Rejected steps in the adaptive Runga-Kutta solver
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
    let mut dt_secs: Option<f64> = kwargs_or_none(&mut kwargs, "dt_secs")?;
    let dt_days: Option<f64> = kwargs_or_none(&mut kwargs, "dt_days")?;
    let dt_dur: Option<PyDuration> = kwargs_or_none(&mut kwargs, "dt")?;
    let duration_secs: Option<f64> = kwargs_or_none(&mut kwargs, "duration_secs")?;
    let pyduration: Option<PyDuration> = kwargs_or_none(&mut kwargs, "duration")?;
    let duration_days: Option<f64> = kwargs_or_none(&mut kwargs, "duration_days")?;
    let pystoptime: Option<PyAstroTime> = kwargs_or_none(&mut kwargs, "stoptime")?;
    let output_phi: bool = kwargs_or_default(&mut kwargs, "output_phi", false)?;
    let pysatproperties: Option<PySatProperties> = kwargs_or_none(&mut kwargs, "satproperties")?;

    // get duration over which to compute solution
    match dt_dur {
        Some(v) => dt_secs = Some(v.inner.seconds()),
        None => match dt_days {
            None => (),
            Some(v) => dt_secs = Some(v * 86400.0),
        },
    }

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

    // Simple sate propagation
    if output_phi == false {
        lowlevel_propagate::<1>(
            pos.as_gil_ref(),
            vel.as_gil_ref(),
            &start.inner,
            &stoptime,
            dt_secs,
            &pysatproperties,
            &propsettings,
        )
    }
    // Propagate with state transition matrix
    else {
        lowlevel_propagate::<7>(
            pos.as_gil_ref(),
            vel.as_gil_ref(),
            &start.inner,
            &stoptime,
            dt_secs,
            &pysatproperties,
            &propsettings,
        )
    }
}
