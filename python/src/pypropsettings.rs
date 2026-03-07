use pyo3::prelude::*;

use crate::pygravity::GravModel;
use crate::{PyDuration, PyInstant};
use satkit::orbitprop::{Integrator, PropSettings};

use pyo3::types::{PyDelta, PyDeltaAccess, PyDict, PyString};

/// Choice of ODE integrator for orbit propagation
#[allow(non_camel_case_types)]
#[pyclass(name = "integrator", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyIntegrator {
    /// Verner 9(8) with 9th-order dense output, 26 stages (default)
    rkv98 = 0,
    /// Verner 9(8) without interpolation, 16 stages
    rkv98_nointerp = 1,
    /// Verner 8(7) with 8th-order dense output, 21 stages
    rkv87 = 2,
    /// Verner 6(5), 10 stages
    rkv65 = 3,
    /// Tsitouras 5(4) with FSAL, 7 stages
    rkts54 = 4,
    /// RODAS4 — L-stable Rosenbrock 4(3), 6 stages. For stiff problems.
    rodas4 = 5,
}

impl From<PyIntegrator> for Integrator {
    fn from(i: PyIntegrator) -> Self {
        match i {
            PyIntegrator::rkv98 => Integrator::RKV98,
            PyIntegrator::rkv98_nointerp => Integrator::RKV98NoInterp,
            PyIntegrator::rkv87 => Integrator::RKV87,
            PyIntegrator::rkv65 => Integrator::RKV65,
            PyIntegrator::rkts54 => Integrator::RKTS54,
            PyIntegrator::rodas4 => Integrator::RODAS4,
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
        }
    }
}

#[pyclass(name = "propsettings", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPropSettings(pub PropSettings);

#[pymethods]
impl PyPropSettings {
    #[new]
    #[pyo3(signature=(**kwargs))]
    fn py_new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut ps = PropSettings::default();
        let mut order_explicitly_set = false;
        if let Some(kw) = kwargs {
            if let Some(abserr) = kw.get_item("abs_error")? {
                ps.abs_error = abserr.extract::<f64>()?;
                kw.del_item("abs_error")?;
            }
            if let Some(relerr) = kw.get_item("rel_error")? {
                ps.rel_error = relerr.extract::<f64>()?;
                kw.del_item("rel_error")?;
            }
            if let Some(gravdegree) = kw.get_item("gravity_degree")? {
                ps.gravity_degree = gravdegree.extract::<u16>()?;
                kw.del_item("gravity_degree")?;
            }
            if let Some(gravorder) = kw.get_item("gravity_order")? {
                ps.gravity_order = gravorder.extract::<u16>()?;
                order_explicitly_set = true;
                kw.del_item("gravity_order")?;
            }
            if let Some(interp) = kw.get_item("enable_iterp")? {
                ps.enable_interp = interp.extract::<bool>()?;
                kw.del_item("enable_interp")?;
            }
            if let Some(sw) = kw.get_item("use_spaceweather")? {
                ps.use_spaceweather = sw.extract::<bool>()?;
                kw.del_item("use_spaceweather")?;
            }
            if let Some(sun) = kw.get_item("use_sun_gravity")? {
                ps.use_sun_gravity = sun.extract::<bool>()?;
                kw.del_item("use_sun_gravity")?;
            }
            if let Some(moon) = kw.get_item("use_moon_gravity")? {
                ps.use_moon_gravity = moon.extract::<bool>()?;
                kw.del_item("use_moon_gravity")?;
            }
            if let Some(gm) = kw.get_item("gravity_model")? {
                let model: GravModel = gm.extract::<GravModel>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                        "gravity_model must be a satkit.gravmodel enum value (e.g. satkit.gravmodel.jgm3)"
                    ))?;
                ps.gravity_model = model.into();
                kw.del_item("gravity_model")?;
            }
            if let Some(integ) = kw.get_item("integrator")? {
                let integrator: PyIntegrator = integ.extract::<PyIntegrator>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                        "integrator must be a satkit.integrator enum value (e.g. satkit.integrator.rkv98)"
                    ))?;
                ps.integrator = integrator.into();
                kw.del_item("integrator")?;
            }
            if !kw.is_empty() {
                let keystring: String = kw.iter().fold(String::from(""), |acc, (k, _v)| {
                    let mut a2 = acc.clone();
                    a2.push_str(k.cast::<PyString>().unwrap().to_str().unwrap());
                    a2.push_str(", ");
                    a2
                });
                let s = format!("Invalid kwargs: {}", keystring);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(s));
            }
            if order_explicitly_set {
                // Clamp order to degree
                if ps.gravity_order > ps.gravity_degree {
                    ps.gravity_order = ps.gravity_degree;
                }
            } else {
                // Default order to degree when not explicitly provided
                ps.gravity_order = ps.gravity_degree;
            }
        }

        Ok(Self(ps))
    }

    #[getter]
    fn get_abs_error(&self) -> f64 {
        self.0.abs_error
    }

    #[setter(abs_error)]
    fn set_abs_error(&mut self, val: f64) -> PyResult<()> {
        self.0.abs_error = val;
        Ok(())
    }

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
        self.0.gravity_degree = val;
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

    fn __str__(&self) -> String {
        self.0.to_string()
    }

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
                } else if let Ok(delta) = {
                    #[allow(deprecated)]
                    obj.downcast::<PyDelta>()
                } {
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
