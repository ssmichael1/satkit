use pyo3::prelude::*;

use crate::pygravity::GravModel;
use crate::{PyDuration, PyInstant};
use satkit::orbitprop::{Integrator, PropSettings, TideModel};

use pyo3::types::{PyBytes, PyDelta, PyDeltaAccess, PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

/// Choice of ODE integrator for orbit propagation
#[allow(non_camel_case_types)]
#[pyclass(name = "integrator", module = "satkit", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyIntegrator {
    /// Verner 9(8) with 8th-degree dense output, 21 stages (16 + 5 for the interpolant; default)
    rkv98 = 0,
    /// Verner 9(8) without interpolation, 16 stages
    rkv98_nointerp = 1,
    /// Verner 8(7) with 7th-degree dense output, 17 stages (13 + 4 for the interpolant)
    rkv87 = 2,
    /// Verner 6(5) with 6th-degree dense output, 10 stages
    rkv65 = 3,
    /// Tsitouras 5(4) with FSAL, 7 stages
    rkts54 = 4,
    /// RODAS4 — L-stable Rosenbrock 4(3), 6 stages. For stiff problems.
    rodas4 = 5,
    /// Gauss-Jackson 8 — 8th-order fixed-step multistep predictor-corrector
    /// for high-precision orbit propagation. Requires setting
    /// `gj_step_seconds` on the propsettings. Supports dense output
    /// (quintic Hermite interpolation); no STM support.
    gauss_jackson8 = 6,
}

crate::enum_pickle!(PyIntegrator, "integrator");

impl From<PyIntegrator> for Integrator {
    fn from(i: PyIntegrator) -> Self {
        match i {
            PyIntegrator::rkv98 => Integrator::RKV98,
            PyIntegrator::rkv98_nointerp => Integrator::RKV98NoInterp,
            PyIntegrator::rkv87 => Integrator::RKV87,
            PyIntegrator::rkv65 => Integrator::RKV65,
            PyIntegrator::rkts54 => Integrator::RKTS54,
            PyIntegrator::rodas4 => Integrator::RODAS4,
            PyIntegrator::gauss_jackson8 => Integrator::GaussJackson8,
        }
    }
}

impl From<Integrator> for PyIntegrator {
    fn from(i: Integrator) -> Self {
        match i {
            Integrator::RKV98 => PyIntegrator::rkv98,
            Integrator::RKV98NoInterp => PyIntegrator::rkv98_nointerp,
            Integrator::RKV87 => PyIntegrator::rkv87,
            Integrator::RKV65 => PyIntegrator::rkv65,
            Integrator::RKTS54 => PyIntegrator::rkts54,
            Integrator::RODAS4 => PyIntegrator::rodas4,
            Integrator::GaussJackson8 => PyIntegrator::gauss_jackson8,
        }
    }
}

/// Solid Earth tide model selector.
#[allow(non_camel_case_types)]
#[pyclass(name = "tidemodel", module = "satkit", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyTideModel {
    /// No solid Earth tide correction.
    none = 0,
    /// IERS 2010 §6.2.1 Step 1 — frequency-independent Love-number
    /// response. Accounts for ≈99% of the solid-tide signal. Default.
    solid_step1 = 1,
    /// IERS 2010 §6.2.1 Step 1 + §6.2.2 Step 2. Step 2 is not yet
    /// implemented; currently falls back to Step 1.
    solid_full = 2,
}

crate::enum_pickle!(PyTideModel, "tidemodel");

impl From<PyTideModel> for TideModel {
    fn from(t: PyTideModel) -> Self {
        match t {
            PyTideModel::none => TideModel::None,
            PyTideModel::solid_step1 => TideModel::SolidStep1,
            PyTideModel::solid_full => TideModel::SolidFull,
        }
    }
}

impl From<TideModel> for PyTideModel {
    fn from(t: TideModel) -> Self {
        match t {
            TideModel::None => PyTideModel::none,
            TideModel::SolidStep1 => PyTideModel::solid_step1,
            TideModel::SolidFull => PyTideModel::solid_full,
        }
    }
}

/// Settings for the high-precision orbit propagator
///
/// Keyword Args:
///     abs_error (float): Maximum absolute error of any element in the propagated state, in the
///         units of the state (meters for position elements, m/s for velocity elements). Default 1e-8
///     rel_error (float): Maximum relative error of any element in the propagated state, unitless. Default 1e-8
///     gravity_degree (int): Maximum degree of spherical harmonic gravity model (at most 70). Default 4
///     gravity_order (int): Maximum order of spherical harmonic gravity model. Default same as gravity_degree
///     gravity_model (satkit.gravmodel): Gravity model. Default gravmodel.egm2008
///     use_spaceweather (bool): Use space weather data for atmospheric density. Default True
///     use_sun_gravity (bool): Include sun third-body gravity. Default True
///     use_moon_gravity (bool): Include moon third-body gravity. Default True
///     tide_model (satkit.tidemodel): Solid Earth tide model. Default tidemodel.solid_step1
///     use_relativistic_correction (bool): Include general-relativistic acceleration. Default True
///     enable_interp (bool): Store dense output for interpolation. Default True. False also runs
///         integrator.rkv98 as its 16-stage no-interpolant tableau (24% fewer force evaluations per step)
///     integrator (satkit.integrator): ODE integrator. Default integrator.rkv98
///     gj_step_seconds (float): Fixed step size for integrator.gauss_jackson8, seconds. Default 60.0
///     max_steps (int): Maximum number of integrator steps. Default 1_000_000
///     require_eop_coverage (bool): Raise if the span extends past the EOP table. Default False
///     initial_step_secs (float | None): First step the adaptive integrators attempt, seconds.
///         Default None: derived from the initial state, tolerances and integrator order.
///         Set to a previous result's ``next_step_secs`` to warm-start a follow-on arc.
#[pyclass(name = "propsettings", module = "satkit", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPropSettings(pub PropSettings);

/// Reject degrees the evaluator cannot honour (it would otherwise silently
/// evaluate at the maximum). Mirrors `PropSettings::set_gravity`.
fn check_gravity_degree(val: u16) -> PyResult<u16> {
    use satkit::earthgravity::MAX_GRAVITY_DEGREE;
    if val > MAX_GRAVITY_DEGREE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gravity degree/order {val} exceeds the maximum supported value ({MAX_GRAVITY_DEGREE})"
        )));
    }
    Ok(val)
}

crate::arg_extractor!(gravity_model_arg: GravModel, |_| {
    pyo3::exceptions::PyValueError::new_err(
        "gravity_model must be a satkit.gravmodel enum value (e.g. satkit.gravmodel.egm96)",
    )
});
crate::arg_extractor!(integrator_arg: PyIntegrator, |_| {
    pyo3::exceptions::PyValueError::new_err(
        "integrator must be a satkit.integrator enum value (e.g. satkit.integrator.rkv98)",
    )
});
crate::arg_extractor!(tide_model_arg: PyTideModel, |_| {
    pyo3::exceptions::PyValueError::new_err(
        "tide_model must be a satkit.tidemodel enum value (e.g. satkit.tidemodel.solid_step1)",
    )
});

#[pymethods]
impl PyPropSettings {
    #[new]
    #[pyo3(signature=(
        *,
        abs_error=1e-8,
        rel_error=1e-8,
        gravity_degree=4,
        gravity_order=None,
        gravity_model=GravModel::egm2008,
        use_spaceweather=true,
        use_sun_gravity=true,
        use_moon_gravity=true,
        tide_model=PyTideModel::solid_step1,
        use_relativistic_correction=true,
        enable_interp=true,
        integrator=PyIntegrator::rkv98,
        gj_step_seconds=60.0,
        max_steps=1_000_000,
        require_eop_coverage=false,
        initial_step_secs=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        abs_error: f64,
        rel_error: f64,
        gravity_degree: u16,
        gravity_order: Option<u16>,
        #[pyo3(from_py_with = gravity_model_arg)] gravity_model: GravModel,
        use_spaceweather: bool,
        use_sun_gravity: bool,
        use_moon_gravity: bool,
        #[pyo3(from_py_with = tide_model_arg)] tide_model: PyTideModel,
        use_relativistic_correction: bool,
        enable_interp: bool,
        #[pyo3(from_py_with = integrator_arg)] integrator: PyIntegrator,
        gj_step_seconds: f64,
        max_steps: usize,
        require_eop_coverage: bool,
        initial_step_secs: Option<f64>,
    ) -> PyResult<Self> {
        let mut ps = PropSettings::default();
        ps.abs_error = abs_error;
        ps.rel_error = rel_error;
        ps.gravity_degree = check_gravity_degree(gravity_degree)?;
        // The order defaults to the degree and is clamped to it
        ps.gravity_order = gravity_order.map_or(ps.gravity_degree, |o| o.min(ps.gravity_degree));
        ps.gravity_model = gravity_model.into();
        ps.use_spaceweather = use_spaceweather;
        ps.use_sun_gravity = use_sun_gravity;
        ps.use_moon_gravity = use_moon_gravity;
        ps.tide_model = tide_model.into();
        ps.use_relativistic_correction = use_relativistic_correction;
        ps.enable_interp = enable_interp;
        ps.integrator = integrator.into();
        ps.gj_step_seconds = gj_step_seconds;
        ps.max_steps = max_steps;
        ps.require_eop_coverage = require_eop_coverage;
        ps.initial_step_secs = initial_step_secs;
        Ok(Self(ps))
    }

    /// Maximum absolute error of any element in the propagated state, in the units
    /// of the state (meters for position elements, m/s for velocity elements). Default 1e-8
    #[getter]
    fn get_abs_error(&self) -> f64 {
        self.0.abs_error
    }

    #[setter(abs_error)]
    fn set_abs_error(&mut self, val: f64) -> PyResult<()> {
        self.0.abs_error = val;
        Ok(())
    }

    /// Maximum relative error of any element in the propagated state, unitless. Default 1e-8
    #[getter]
    fn get_rel_error(&self) -> f64 {
        self.0.rel_error
    }

    #[setter(rel_error)]
    fn set_rel_error(&mut self, val: f64) -> PyResult<()> {
        self.0.rel_error = val;
        Ok(())
    }

    #[getter]
    fn get_gravity_degree(&self) -> u16 {
        self.0.gravity_degree
    }

    #[setter(gravity_degree)]
    fn set_gravity_degree(&mut self, val: u16) -> PyResult<()> {
        self.0.gravity_degree = check_gravity_degree(val)?;
        if self.0.gravity_order > val {
            self.0.gravity_order = val;
        }
        Ok(())
    }

    #[getter]
    fn get_gravity_order(&self) -> u16 {
        self.0.gravity_order
    }

    #[setter(gravity_order)]
    fn set_gravity_order(&mut self, val: u16) -> PyResult<()> {
        check_gravity_degree(val)?;
        if val > self.0.gravity_degree {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "gravity_order must be <= gravity_degree",
            ));
        }
        self.0.gravity_order = val;
        Ok(())
    }

    #[getter]
    fn get_use_sun_gravity(&self) -> bool {
        self.0.use_sun_gravity
    }

    #[setter(use_sun_gravity)]
    fn set_use_sun_gravity(&mut self, val: bool) -> PyResult<()> {
        self.0.use_sun_gravity = val;
        Ok(())
    }

    #[getter]
    fn get_use_moon_gravity(&self) -> bool {
        self.0.use_moon_gravity
    }

    #[setter(use_moon_gravity)]
    fn set_use_moon_gravity(&mut self, val: bool) -> PyResult<()> {
        self.0.use_moon_gravity = val;
        Ok(())
    }

    /// Store dense output so ``propresult.interp`` works between the begin and
    /// end times. Default True. When False, no dense output is stored and
    /// ``integrator.rkv98`` runs its 16-stage no-interpolant tableau (same
    /// order and error control, 24% fewer force evaluations per step).
    #[getter]
    fn get_enable_interp(&self) -> bool {
        self.0.enable_interp
    }

    #[setter(enable_interp)]
    fn set_enable_interp(&mut self, val: bool) -> PyResult<()> {
        self.0.enable_interp = val;
        Ok(())
    }

    #[getter]
    fn get_use_spaceweather(&self) -> bool {
        self.0.use_spaceweather
    }

    #[setter(use_spaceweather)]
    fn set_use_spacewather(&mut self, val: bool) -> PyResult<()> {
        self.0.use_spaceweather = val;
        Ok(())
    }

    #[getter]
    fn get_gravity_model(&self) -> GravModel {
        self.0.gravity_model.into()
    }

    #[setter(gravity_model)]
    fn set_gravity_model(&mut self, val: GravModel) -> PyResult<()> {
        self.0.gravity_model = val.into();
        Ok(())
    }

    #[getter]
    fn get_integrator(&self) -> PyIntegrator {
        self.0.integrator.into()
    }

    #[setter(integrator)]
    fn set_integrator(&mut self, val: PyIntegrator) -> PyResult<()> {
        self.0.integrator = val.into();
        Ok(())
    }

    /// Fixed step size used by integrator.gauss_jackson8, seconds. Ignored by
    /// adaptive integrators. Default 60.0
    #[getter]
    fn get_gj_step_seconds(&self) -> f64 {
        self.0.gj_step_seconds
    }

    #[setter(gj_step_seconds)]
    fn set_gj_step_seconds(&mut self, val: f64) -> PyResult<()> {
        self.0.gj_step_seconds = val;
        Ok(())
    }

    #[getter]
    fn get_max_steps(&self) -> usize {
        self.0.max_steps
    }

    #[setter(max_steps)]
    fn set_max_steps(&mut self, val: usize) -> PyResult<()> {
        self.0.max_steps = val;
        Ok(())
    }

    #[getter]
    fn get_tide_model(&self) -> PyTideModel {
        self.0.tide_model.into()
    }

    #[setter(tide_model)]
    fn set_tide_model(&mut self, val: PyTideModel) -> PyResult<()> {
        self.0.tide_model = val.into();
        Ok(())
    }

    #[getter]
    fn get_use_relativistic_correction(&self) -> bool {
        self.0.use_relativistic_correction
    }

    #[setter(use_relativistic_correction)]
    fn set_use_relativistic_correction(&mut self, val: bool) -> PyResult<()> {
        self.0.use_relativistic_correction = val;
        Ok(())
    }

    /// Fail a propagation whose span extends past the end of the loaded
    /// Earth-orientation-parameter (EOP) table instead of holding the last
    /// EOP row constant with a one-time warning. Default False. See
    /// ``satkit.frametransform.eop_coverage``.
    #[getter]
    fn get_require_eop_coverage(&self) -> bool {
        self.0.require_eop_coverage
    }

    #[setter(require_eop_coverage)]
    fn set_require_eop_coverage(&mut self, val: bool) -> PyResult<()> {
        self.0.require_eop_coverage = val;
        Ok(())
    }

    /// First step (seconds) the adaptive integrators attempt, or None for
    /// the default, derived from the initial state, the tolerances and the
    /// integrator order as ``1.5 * |r|/|v| * tol**(1/(p+1))`` with
    /// ``tol = rel_error + abs_error/|r|`` (about 170 s for rkv98 at 1e-9 in
    /// LEO; within a factor of ~2.5 of the settled stride, which the step
    /// controller closes within a step or two). Set it to a previous result's
    /// ``propresult.next_step_secs`` to warm-start a follow-on arc at full
    /// stride. A magnitude: backward propagation applies the sign, and a
    /// value longer than the arc is clamped to it. Ignored by
    /// ``integrator.gauss_jackson8``. Zero or non-finite raises from
    /// ``propagate``.
    #[getter]
    fn get_initial_step_secs(&self) -> Option<f64> {
        self.0.initial_step_secs
    }

    #[setter(initial_step_secs)]
    fn set_initial_step_secs(&mut self, val: Option<f64>) -> PyResult<()> {
        self.0.initial_step_secs = val;
        Ok(())
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        (PyTuple::empty(py), PyDict::new(py))
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // The `precomputed` cache is intentionally skipped (see PropSettings);
        // a restored settings object recomputes it lazily.
        let bytes = crate::pyutils::serde_pickle_to_vec(&self.0, "propsettings")?;
        PyBytes::new(py, &bytes).into_py_any(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        self.0 = crate::pyutils::serde_pickle_from_slice(state.as_bytes(py), "propsettings")?;
        Ok(())
    }

    /// Precompute sun/moon terms for fast propagation of many satellites over the same span
    ///
    /// Args:
    ///     begin (satkit.time): Begin time of propagation
    ///     end (satkit.time): End time of propagation
    ///     step (satkit.duration | float | datetime.timedelta, optional): Table step;
    ///         a float is interpreted as seconds. Default 60 seconds
    #[pyo3(signature=(begin, end, step=None))]
    fn precompute_terms(
        &mut self,
        begin: &PyInstant,
        end: &PyInstant,
        step: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let step_secs: Option<f64> = match step {
            None => None,
            Some(obj) => {
                if let Ok(d) = obj.extract::<PyDuration>() {
                    Some(d.0.as_seconds())
                } else if let Ok(secs) = obj.extract::<f64>() {
                    Some(secs)
                } else if let Ok(delta) = obj.cast::<PyDelta>() {
                    Some(
                        delta.get_days() as f64 * 86400.0
                            + delta.get_seconds() as f64
                            + delta.get_microseconds() as f64 * 1e-6,
                    )
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "step must be a satkit.duration, float (seconds), or datetime.timedelta",
                    ));
                }
            }
        };
        let result = match step_secs {
            Some(s) => self.0.precompute_terms_with_step(&begin.0, &end.0, s),
            None => self.0.precompute_terms(&begin.0, &end.0),
        };
        match result {
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            Ok(_) => Ok(()),
        }
    }
}

impl From<&PyPropSettings> for PropSettings {
    fn from(item: &PyPropSettings) -> Self {
        item.0.clone()
    }
}

impl From<&PropSettings> for PyPropSettings {
    fn from(item: &PropSettings) -> Self {
        Self(item.clone())
    }
}
