use crate::pyduration::PyDuration;
use crate::pyinstant::PyInstant;
use crate::pypropresult::{PyPropResult, PyPropResultType};
use crate::pypropsettings::PyPropSettings;
use crate::pysatproperties::{satproperties_arg, PySatProperties};
use crate::pyutils::*;
use pyo3::IntoPyObjectExt;

use satkit::mathtypes::*;
use satkit::orbitprop::SatProperties;
use satkit::orbitprop::SatPropertiesSimple;
use satkit::Duration;
use satkit::Instant;

use pyo3::prelude::*;

use anyhow::Result;

crate::arg_extractor!(begin_arg: Option<PyInstant>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid begin time: {e}"))
});
crate::arg_extractor!(end_arg: Option<PyInstant>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid end time: {e}"))
});
crate::arg_extractor!(duration_arg: Option<PyDuration>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid duration: {e}"))
});
crate::arg_extractor!(duration_secs_arg: Option<f64>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid duration_secs: {e}"))
});
crate::arg_extractor!(duration_days_arg: Option<f64>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid duration_days: {e}"))
});
crate::arg_extractor!(output_phi_arg: bool, |_| invalid_value("output_phi"));
crate::arg_extractor!(propsettings_arg: Option<PyPropSettings>, |_| invalid_value("propsettings"));

/// High-precision orbit propagator
///
/// Propagate statellite ephemeris (position, velocity in gcrs & time) to new time via adaptive Runge-Kutta 9/8 ordinary differential equation (ODE) integration
///
/// Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
///
/// Inputs:
///
///      state (npt.ArrayLike[float]): 6-element numpy array representing satellite GCRF position & velocity, in meters and meters/second
///      begin (satkit.time): Begin time of propagation, time of "state"
///        end (satkit.time, optional): End time of propagation
///
///
/// Optional keyword arguments:
///
///
/// 4 ways of setting propagation end:
/// (one of these must be used)
///
///              end: (satkit.time, optional): instant at which new position and
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
///              pos (npt.ArrayLike[float], optional): GCRF position, meters; replaces
///                   the first three elements of "state" (or stands in for it with "vel")
///              vel (npt.ArrayLike[float], optional): GCRF velocity, meters/second; replaces
///                   the last three elements of "state"
///       output_phi (bool): boolean inticating Output 6x6 state transition matrix
///                   between "begintime" and "endtime"
///                   default is False
///     propsettings (satkit.propsettings): Settings for
///                   the propagation. if left out, default will be used.
///    satproperties (satkit.satproperties): object with drag,
///                   radiation pressure, and thrust properties of satellite.
///                   If left out, drag, radiation pressure, and thrust are neglected
///
///
/// Returns:
///
///    satkit.propresult: object with new GCRF position (meters) and velocity (meters/second), and possibly
///                       state transition matrix between "begintime" and "endtime",
///                       and dense ODE solution that allow for interpolation, if requested
///
/// Raises:
///
///   RuntimeError: If "pos" or "vel" are not 3-element numpy arrays
///   RuntimeError: If neither "end", "duration", "duration_secs", or "duration_days" are set
///   TypeError: If an unknown keyword argument is passed, or "state" or "begin" is missing
///
///
///    Notes:
///        * Propagator uses advanced Runge-Kutta integrators and includes the following forces:
///            * Earth gravity with higher-order spherical-harmonic terms
///            * Sun, Moon gravity
///            * Solid Earth tides (IERS 2010 Step 1; configurable via propsettings.tide_model)
///            * General relativity (IERS 2010 Eq. 10.12: Schwarzschild, geodesic precession, Lense-Thirring)
///            * Radiation pressure
///            * Atmospheric drag: NRLMSISE-00 density model, with option to include space weather effects (can be large)
///        * End time must be set by keyword argument, either explicitly or by duration
///        * Dense-output interpolation is controlled by propsettings.enable_interp
///          (default True); use propresult.interp to query interpolated states
///
// `state` and `begin` take `None` defaults only so the undocumented
// `pos=` / `vel=` keywords can replace `state`; a missing `begin` is refused
// below.
#[pyfunction(signature=(
    state=None,
    begin=None,
    end=None,
    *,
    pos=None,
    vel=None,
    duration=None,
    duration_secs=None,
    duration_days=None,
    output_phi=false,
    propsettings=None,
    satproperties=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn propagate(
    py: Python,
    state: Option<&Bound<'_, PyAny>>,
    #[pyo3(from_py_with = begin_arg)] begin: Option<PyInstant>,
    #[pyo3(from_py_with = end_arg)] end: Option<PyInstant>,
    pos: Option<&Bound<'_, PyAny>>,
    vel: Option<&Bound<'_, PyAny>>,
    #[pyo3(from_py_with = duration_arg)] duration: Option<PyDuration>,
    #[pyo3(from_py_with = duration_secs_arg)] duration_secs: Option<f64>,
    #[pyo3(from_py_with = duration_days_arg)] duration_days: Option<f64>,
    #[pyo3(from_py_with = output_phi_arg)] output_phi: bool,
    #[pyo3(from_py_with = propsettings_arg)] propsettings: Option<PyPropSettings>,
    #[pyo3(from_py_with = satproperties_arg)] satproperties: Option<PySatProperties>,
) -> Result<Py<PyAny>> {
    let propsettings = propsettings.map_or_else(Default::default, |p| p.0);
    let satproperties: Option<SatPropertiesSimple> = satproperties.map(|p| p.0);

    if state.is_none() && pos.is_none() && vel.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "propagate() missing required argument 'state'",
        )
        .into());
    }
    let Some(PyInstant(begintime)) = begin else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "propagate() missing required argument 'begin'",
        )
        .into());
    };

    let mut state0 = match state {
        Some(s) => py_to_smatrix(s)?,
        None => Vector6::zeros(),
    };
    // `pos=` / `vel=` overwrite the corresponding half of `state`
    if let Some(p) = pos {
        let p = py_to_smatrix::<3, 1>(p)?;
        state0[0] = p[0];
        state0[1] = p[1];
        state0[2] = p[2];
    }
    if let Some(v) = vel {
        let v = py_to_smatrix::<3, 1>(v)?;
        state0[3] = v[0];
        state0[4] = v[1];
        state0[5] = v[2];
    }

    // The end time: `end`, overridden by `duration`, then `duration_days`,
    // then `duration_secs`
    let mut endtime = end.map_or(Instant::INVALID, |t| t.0);
    if let Some(d) = duration {
        endtime = begintime + d.0;
    }
    if let Some(d) = duration_days {
        endtime = begintime + Duration::from_days(d);
    }
    if let Some(d) = duration_secs {
        endtime = begintime + Duration::from_seconds(d);
    }

    // Release the GIL during the (potentially long-running) propagation
    // so other Python threads can make progress

    // Simple sate propagation
    if !output_phi {
        let res = py.detach(|| {
            satkit::orbitprop::propagate(
                &state0,
                &begintime,
                &endtime,
                &propsettings,
                satproperties.as_ref().map(|p| p as &dyn SatProperties),
            )
        })?;
        Ok(PyPropResult(PyPropResultType::R1(Box::new(res))).into_py_any(py)?)
    }
    // Propagate with state transition matrix
    else {
        // Create the state to propagate
        let mut pv = Matrix67::zeros();
        pv.set_block(0, 0, &state0);
        pv.set_block(0, 1, &Matrix6::eye());

        let res = py.detach(|| {
            satkit::orbitprop::propagate(
                &pv,
                &begintime,
                &endtime,
                &propsettings,
                satproperties.as_ref().map(|p| p as &dyn SatProperties),
            )
        })?;
        Ok(PyPropResult(PyPropResultType::R7(Box::new(res))).into_py_any(py)?)
    }
}
