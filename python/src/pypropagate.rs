use crate::pyduration::PyDuration;
use crate::pyinstant::TimeArg;
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

crate::arg_extractor!(begin_arg: Option<TimeArg>, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid begin time: {e}"))
});
crate::arg_extractor!(end_arg: Option<TimeArg>, |e| {
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
/// Propagate a satellite state (GCRF position and velocity) from ``begin`` to a
/// new time by numerical integration of the force model. The default integrator
/// is the adaptive Verner 9(8) Runge-Kutta pair; ``propsettings.integrator``
/// selects the others (lower-order Runge-Kutta pairs, the RODAS4 Rosenbrock
/// method, fixed-step Gauss-Jackson 8).
///
/// Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
///
/// Args:
///     state (array-like): 6-element GCRF position (meters) and velocity
///         (meters/second). Required unless ``pos`` and ``vel`` are given
///     begin (satkit.time | datetime.datetime | numpy.datetime64): time of ``state``. Required
///     end (satkit.time | datetime.datetime | numpy.datetime64, optional): time to propagate to
///
/// Keyword Args:
///     duration (satkit.duration, optional): duration from ``begin`` to propagate
///     duration_secs (float, optional): duration from ``begin``, seconds
///     duration_days (float, optional): duration from ``begin``, days
///     pos (array-like, optional): 3-element GCRF position, meters. Give it
///         together with ``vel``, in place of ``state``
///     vel (array-like, optional): 3-element GCRF velocity, meters/second. Give
///         it together with ``pos``
///     output_phi (bool): also compute the 6x6 state transition matrix between
///         ``begin`` and the end time. Default False
///     propsettings (satkit.propsettings, optional): settings for the
///         propagation. Default: ``propsettings()``
///     satproperties (satkit.satproperties, optional): drag, radiation pressure
///         and thrust properties of the satellite. Default: none of the three
///
/// Exactly one of ``end``, ``duration``, ``duration_secs`` and ``duration_days``
/// sets the end time. It may be before ``begin`` (backward propagation).
///
/// Returns:
///     satkit.propresult: the GCRF position (meters) and velocity (meters/second)
///     at the end time, the state transition matrix if requested, and dense
///     output for interpolation between the two times (``propresult.interp``)
///     unless ``propsettings.enable_interp`` is False
///
/// Raises:
///     TypeError: if ``state`` (or ``pos`` and ``vel``) or ``begin`` is missing,
///         ``pos`` is given without ``vel`` or the reverse, none or more than one
///         of ``end``, ``duration``, ``duration_secs``, ``duration_days`` is given,
///         an argument has the wrong type, or an unknown keyword is passed
///     ValueError: if ``state`` does not have 6 elements (``pos`` / ``vel`` 3),
///         the position is all zeros, or ``duration_secs`` / ``duration_days``
///         is NaN or infinite
///     RuntimeError: if the propagation fails: a required data file is missing,
///         the integrator does not converge or exceeds ``max_steps``,
///         ``output_phi`` is used with ``integrator.rodas4`` or
///         ``integrator.gauss_jackson8``, the span is shorter than 8 steps of
///         ``integrator.gauss_jackson8``, or ``require_eop_coverage`` is set and
///         the span is outside the Earth-orientation table
///
/// Notes:
///     Forces included:
///
///     * Earth gravity with higher-order spherical-harmonic terms
///     * Sun, Moon gravity
///     * Solid Earth tides (IERS 2010 Step 1; configurable via propsettings.tide_model)
///     * General relativity (IERS 2010 Eq. 10.12: Schwarzschild, geodesic precession, Lense-Thirring)
///     * Radiation pressure
///     * Atmospheric drag: NRLMSISE-00 density model, with option to include space weather effects (can be large)
///
// `state` and `begin` take `None` defaults only so the `pos=` / `vel=`
// keywords can replace `state`; a missing `begin` is refused below.
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
    #[pyo3(from_py_with = begin_arg)] begin: Option<TimeArg>,
    #[pyo3(from_py_with = end_arg)] end: Option<TimeArg>,
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
    // `pos` alone used to leave the velocity at zero (and `vel` alone the
    // position), or to overwrite half of `state`
    if pos.is_some() != vel.is_some() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "propagate() takes `pos` and `vel` together (or neither, with `state`)",
        )
        .into());
    }
    let Some(TimeArg(begintime)) = begin else {
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

    // Without any of these the end time used to be `Instant::INVALID`, which
    // surfaced as an obscure "Precomputed table would need N entries" error
    let given: Vec<&str> = [
        ("end", end.is_some()),
        ("duration", duration.is_some()),
        ("duration_secs", duration_secs.is_some()),
        ("duration_days", duration_days.is_some()),
    ]
    .iter()
    .filter_map(|(name, set)| set.then_some(*name))
    .collect();
    if given.is_empty() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "propagate() requires an end time: pass `end` or one of `duration`, \
             `duration_secs`, `duration_days`",
        )
        .into());
    }
    // Several used to be accepted, with a silent precedence
    // (duration_secs, then duration_days, duration, end)
    if given.len() > 1 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "propagate() takes one end time, but got {}: pass only one of `end`, \
             `duration`, `duration_secs`, `duration_days`",
            given.join(", ")
        ))
        .into());
    }

    let mut endtime = end.map_or(Instant::INVALID, |t| t.0);
    if let Some(d) = duration {
        endtime = begintime + d.0;
    }
    if let Some(d) = duration_days {
        crate::pyduration::check_duration_value(d, 86_400.0e6, "duration_days")?;
        endtime = begintime + Duration::from_days(d);
    }
    if let Some(d) = duration_secs {
        crate::pyduration::check_duration_value(d, 1.0e6, "duration_secs")?;
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
