use satkit::mathtypes::Vector3;
use satkit::orbitprop::{EcomParams, SatProperties, SatPropertiesSimple, SimpleState};
use satkit::Instant;

use crate::pyecom::{ecom_from_block, encode_ecom_block, PyEcomParams, ECOM_BLOCK_LEN};
use crate::pyinstant::{PyInstant, TimeArg};
use crate::pythrust::{py_thrusts_to_profile, PyThrust};
use crate::pyutils::to_vector3;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyType};
use pyo3::IntoPyObjectExt;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Result};

// An argument of the wrong type is a `TypeError` (these were `ValueError`,
// or `RuntimeError` for `ecom`, when the keywords were parsed by hand)
crate::arg_extractor!(cdaoverm_arg: f64, |e| {
    pyo3::exceptions::PyTypeError::new_err(format!("cdaoverm must be a number: {e}"))
});
crate::arg_extractor!(craoverm_arg: f64, |e| {
    pyo3::exceptions::PyTypeError::new_err(format!("craoverm must be a number: {e}"))
});
crate::arg_extractor!(ecom_arg: Option<PyEcomParams>, |e| {
    pyo3::exceptions::PyTypeError::new_err(format!("ecom must be a satkit.ecomparams: {e}"))
});
/// `satproperties=` of `propagate` and `satstate.propagate`: `None`, a
/// `satkit.satproperties`, or an instance of a Python subclass of it
pub(crate) fn satproperties_arg(v: &Bound<'_, PyAny>) -> PyResult<Option<SatPropsArg>> {
    if v.is_none() {
        return Ok(None);
    }
    let Ok(props) = v.cast::<PySatProperties>() else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "satproperties must be a satkit.satproperties, got {}",
            v.get_type()
        )));
    };
    SatPropsArg::new(props).map(Some)
}

/// The methods a Python subclass of `satproperties` can override, in the
/// order of [`PySatPropertiesAdapter::overridden`]
const HOOKS: [&str; 4] = ["cd_a_over_m", "cr_a_over_m", "thrust_accel", "srp_ecom"];
const CD: usize = 0;
const CR: usize = 1;
const THRUST: usize = 2;
const ECOM: usize = 3;

/// The satellite properties of one propagation call
pub(crate) enum SatPropsArg {
    /// A plain `satproperties`, or a subclass that overrides none of
    /// [`HOOKS`]: evaluated in Rust, with no call into Python
    Simple(SatPropertiesSimple),
    /// A subclass that overrides at least one of [`HOOKS`]
    Custom(PySatPropertiesAdapter),
}

impl SatPropsArg {
    fn new(props: &Bound<'_, PySatProperties>) -> PyResult<Self> {
        let py = props.py();
        let inner = props.borrow().0.clone();
        let base = py.get_type::<PySatProperties>();
        let cls = props.get_type();
        if cls.is(&base) {
            return Ok(Self::Simple(inner));
        }
        // Looked up on the class, once: a method is overridden when the
        // subclass resolves the name to anything but the base class's own
        let mut overridden = [false; 4];
        for (flag, name) in overridden.iter_mut().zip(HOOKS) {
            *flag = !cls.getattr(name)?.is(&base.getattr(name)?);
        }
        if !overridden.iter().any(|o| *o) {
            return Ok(Self::Simple(inner));
        }
        Ok(Self::Custom(PySatPropertiesAdapter {
            obj: props.clone().into_any().unbind(),
            inner,
            overridden,
            failed: AtomicBool::new(false),
            err: Mutex::new(None),
        }))
    }

    pub(crate) fn as_dyn(&self) -> &dyn SatProperties {
        match self {
            Self::Simple(p) => p,
            Self::Custom(p) => p,
        }
    }

    /// Raise the first exception an overridden method raised during the
    /// propagation, if any. Call after every propagation, before using its
    /// result: the force model cannot return an error through the
    /// integrator, so the propagation ran on with the stored constants.
    pub(crate) fn check(&self) -> PyResult<()> {
        if let Self::Custom(p) = self {
            if let Some(e) = p.err.lock().unwrap_or_else(|e| e.into_inner()).take() {
                return Err(e);
            }
        }
        Ok(())
    }
}

/// [`SatProperties`] for an instance of a Python subclass of `satproperties`
///
/// Each overridden method is called in Python (attaching to the interpreter
/// for the call, since the propagation runs detached); the others are
/// answered from `inner`, a copy of the instance's stored values taken when
/// the propagation started, without touching the interpreter.
pub(crate) struct PySatPropertiesAdapter {
    obj: Py<PyAny>,
    inner: SatPropertiesSimple,
    overridden: [bool; 4],
    /// Set with `err`: read on every force evaluation, so kept lock-free
    failed: AtomicBool,
    /// The first exception raised by an override (see [`SatPropsArg::check`])
    err: Mutex<Option<PyErr>>,
}

impl PySatPropertiesAdapter {
    /// Call `f` with the interpreter attached. `None` if this hook is not
    /// overridden, an earlier call failed, or this one does (recording the
    /// exception): the caller then answers from `inner`.
    fn call<T>(&self, hook: usize, f: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> Option<T> {
        if !self.overridden[hook] || self.failed.load(Ordering::Relaxed) {
            return None;
        }
        Python::attach(|py| match f(self.obj.bind(py)) {
            Ok(v) => Some(v),
            Err(e) => {
                let mut slot = self.err.lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    *slot = Some(e);
                }
                self.failed.store(true, Ordering::Relaxed);
                None
            }
        })
    }

    /// `obj.<hook>(time, state)`
    fn call_state<'py>(
        obj: &Bound<'py, PyAny>,
        hook: usize,
        tm: &Instant,
        state: &SimpleState,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = obj.py();
        obj.call_method1(
            HOOKS[hook],
            (
                PyInstant(*tm),
                numpy::PyArray1::from_slice(py, state.as_slice()),
            ),
        )
    }

    /// A coefficient returned by `cd_a_over_m` / `cr_a_over_m`, held to the
    /// rule of the stored ones (see [`check_coefficient`])
    fn coefficient(hook: usize, r: &Bound<'_, PyAny>) -> PyResult<f64> {
        let v: f64 = r.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "satproperties.{} must return a number, got {}",
                HOOKS[hook],
                r.get_type()
            ))
        })?;
        check_coefficient(&format!("the value returned by {}", HOOKS[hook]), v)
    }
}

impl SatProperties for PySatPropertiesAdapter {
    fn cd_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64 {
        self.call(CD, |obj| {
            Self::coefficient(CD, &Self::call_state(obj, CD, tm, state)?)
        })
        .unwrap_or(self.inner.cdaoverm)
    }

    fn cr_a_over_m(&self, tm: &Instant, state: &SimpleState) -> f64 {
        self.call(CR, |obj| {
            Self::coefficient(CR, &Self::call_state(obj, CR, tm, state)?)
        })
        .unwrap_or(self.inner.craoverm)
    }

    fn thrust_accel(
        &self,
        tm: &Instant,
        pos_gcrf: &Vector3,
        vel_gcrf: &Vector3,
    ) -> Option<Vector3> {
        self.call(THRUST, |obj| {
            let py = obj.py();
            let r = obj.call_method1(
                HOOKS[THRUST],
                (
                    PyInstant(*tm),
                    numpy::PyArray1::from_slice(py, pos_gcrf.as_slice()),
                    numpy::PyArray1::from_slice(py, vel_gcrf.as_slice()),
                ),
            )?;
            if r.is_none() {
                return Ok(None);
            }
            let a = to_vector3(&r, "the acceleration returned by thrust_accel").map_err(|e| {
                if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "satproperties.thrust_accel must return a 3-element vector or None, got {}",
                        r.get_type()
                    ))
                } else {
                    e
                }
            })?;
            if !a.iter().all(|x| x.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "thrust_accel returned a non-finite acceleration: [{}, {}, {}]",
                    a[0], a[1], a[2]
                )));
            }
            Ok(Some(a))
        })
        .unwrap_or_else(|| self.inner.thrust_accel(tm, pos_gcrf, vel_gcrf))
    }

    fn srp_ecom(&self, tm: &Instant, state: &SimpleState) -> Option<EcomParams> {
        self.call(ECOM, |obj| {
            let r = Self::call_state(obj, ECOM, tm, state)?;
            if r.is_none() {
                return Ok(None);
            }
            match r.cast::<PyEcomParams>() {
                Ok(e) => Ok(Some(e.borrow().0)),
                Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "satproperties.srp_ecom must return a satkit.ecomparams or None, got {}",
                    r.get_type()
                ))),
            }
        })
        .unwrap_or(self.inner.ecom)
    }
}

/// Cd A / m scales the drag deceleration and Cr A / m the radiation
/// pressure; a negative one would reverse the force (accelerating the
/// satellite along its velocity, or pulling it toward the Sun)
fn check_coefficient(name: &str, v: f64) -> PyResult<f64> {
    if v.is_nan() || v < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must not be negative or NaN, got {v}"
        )));
    }
    Ok(v)
}

/// Satellite properties relevant for drag, radiation pressure, and thrust
///
/// This class lets the satellite radiation pressure, drag,
/// and thrust parameters be set for duration of propagation.
///
/// Attributes:
///     cdaoverm (float): Coefficient of drag times area over mass in m^2/kg; not negative
///     craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg; not negative
///     thrusts (list[thrust]): List of continuous thrust arcs (a copy; assign a list to change them)
///     ecom (ecomparams | None): ECOM empirical solar-radiation-pressure
///         coefficients, added to the cannonball term (use ``craoverm=0``
///         for a pure ECOM model). See :class:`ecomparams` for the
///         conventions and the "ECOM Solar Radiation Pressure" tutorial
///         for a fit against IGS GPS orbits.
///
/// Subclass it to make any of these a function of time and state: override
/// ``cd_a_over_m``, ``cr_a_over_m``, ``srp_ecom`` or ``thrust_accel`` (which
/// can return any acceleration, not only thrust). See those methods.
#[pyclass(name = "satproperties", module = "satkit", subclass, from_py_object)]
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
    ///         pressure; not negative. Default 0
    ///     thrusts (list[satkit.thrust] | None): continuous thrust arcs. Default None
    ///     ecom (satkit.ecomparams | None): ECOM empirical solar-radiation-pressure
    ///         coefficients, added to the cannonball term (use craoverm=0 for a
    ///         pure ECOM model). Default None
    ///
    /// Raises:
    ///     TypeError: if any argument is passed positionally (releases before
    ///         this check read positional arguments as (craoverm, cdaoverm),
    ///         the reverse of the documented order, so positional calls are
    ///         refused rather than reinterpreted), or an argument has the
    ///         wrong type
    ///     ValueError: if cdaoverm or craoverm is negative or NaN
    ///
    #[new]
    #[classmethod]
    // `*args` and `**kwargs` catch what the class itself does not take: for
    // `satproperties` and for a subclass without its own `__init__` they are
    // refused with the messages below; for a subclass with an `__init__`
    // they are that method's arguments, which Python also passes here.
    // `text_signature` hides them.
    #[pyo3(
        signature=(*args, cdaoverm=0.0, craoverm=0.0, thrusts=None, ecom=None, **kwargs),
        text_signature = "(*, cdaoverm=0.0, craoverm=0.0, thrusts=None, ecom=None)"
    )]
    fn new(
        cls: &Bound<'_, PyType>,
        args: &Bound<PyTuple>,
        #[pyo3(from_py_with = cdaoverm_arg)] cdaoverm: f64,
        #[pyo3(from_py_with = craoverm_arg)] craoverm: f64,
        thrusts: Option<Vec<PyThrust>>,
        #[pyo3(from_py_with = ecom_arg)] ecom: Option<PyEcomParams>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = cls.py();
        let own_init = !cls
            .getattr("__init__")?
            .is(&py.get_type::<pyo3::types::PyAny>().getattr("__init__")?);
        if !own_init {
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
            if let Some(name) = kwargs.and_then(|k| k.keys().iter().next()) {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "satproperties() got an unexpected keyword argument '{name}'"
                )));
            }
        }
        let mut props = SatPropertiesSimple::new(
            check_coefficient("cdaoverm", cdaoverm)?,
            check_coefficient("craoverm", craoverm)?,
        );
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
    ///     craoverm (float): Cr A / m (m^2/kg); ``ValueError`` if negative or NaN
    #[setter]
    fn set_craoverm(&mut self, craoverm: f64) -> PyResult<()> {
        self.0.craoverm = check_coefficient("craoverm", craoverm)?;
        Ok(())
    }

    /// Set the satellite's susceptibility to drag
    ///
    /// Args:
    ///     cdaoverm (float): Cd A / m (m^2/kg); ``ValueError`` if negative or NaN
    #[setter]
    fn set_cdaoverm(&mut self, cdaoverm: f64) -> PyResult<()> {
        self.0.cdaoverm = check_coefficient("cdaoverm", cdaoverm)?;
        Ok(())
    }

    /// The continuous thrust arcs
    ///
    /// A new list on every access: appending to it or removing from it does
    /// not change the properties. Assign a whole list to change the arcs
    /// (``props.thrusts = props.thrusts + [arc]``).
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
    /// A copy on every access: setting a coefficient on it does not change
    /// the properties; assign the modified ``ecomparams`` back.
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

    /// Drag susceptibility Cd A / m at a time and state
    ///
    /// The propagator calls this at every force evaluation below 1,000 km.
    /// The base class returns the stored ``cdaoverm``; override it in a
    /// subclass to make the drag coefficient a function of time and state.
    ///
    /// Args:
    ///     time (satkit.time): Time of the evaluation
    ///     state (numpy.ndarray): 6-element GCRF position [m] and velocity [m/s]
    ///
    /// Returns:
    ///     float: Cd A / m (m^2/kg); not negative
    #[pyo3(signature=(time, state))]
    fn cd_a_over_m(&self, time: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> f64 {
        let _ = (time, state);
        self.0.cdaoverm
    }

    /// Radiation-pressure susceptibility Cr A / m at a time and state
    ///
    /// The propagator calls this at every force evaluation. The base class
    /// returns the stored ``craoverm``; override it in a subclass to make
    /// the coefficient a function of time and state.
    ///
    /// Args:
    ///     time (satkit.time): Time of the evaluation
    ///     state (numpy.ndarray): 6-element GCRF position [m] and velocity [m/s]
    ///
    /// Returns:
    ///     float: Cr A / m (m^2/kg); not negative
    #[pyo3(signature=(time, state))]
    fn cr_a_over_m(&self, time: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> f64 {
        let _ = (time, state);
        self.0.craoverm
    }

    /// ECOM solar-radiation-pressure coefficients at a time and state
    ///
    /// The base class returns the stored ``ecom``; override it in a subclass
    /// to change the coefficients over the propagation (per-arc values, an
    /// attitude-mode switch).
    ///
    /// Args:
    ///     time (satkit.time): Time of the evaluation
    ///     state (numpy.ndarray): 6-element GCRF position [m] and velocity [m/s]
    ///
    /// Returns:
    ///     satkit.ecomparams | None: the coefficients, or None for no ECOM term
    #[pyo3(signature=(time, state))]
    fn srp_ecom(&self, time: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> Option<PyEcomParams> {
        let _ = (time, state);
        self.0.ecom.map(PyEcomParams)
    }

    /// Additional acceleration at a time and state, in GCRF
    ///
    /// The propagator adds the returned vector to the acceleration at every
    /// force evaluation. The base class returns the sum of the ``thrusts``
    /// arcs active at ``time``; override it in a subclass to supply any
    /// acceleration as a function of time and state: a guidance law, a
    /// solar sail, an empirical term.
    ///
    /// Args:
    ///     time (satkit.time): Time of the evaluation
    ///     pos (numpy.ndarray): 3-element GCRF position [m]
    ///     vel (numpy.ndarray): 3-element GCRF velocity [m/s]
    ///
    /// Returns:
    ///     numpy.ndarray | None: 3-element acceleration in GCRF [m/s^2], or
    ///     None for no acceleration
    #[pyo3(signature=(time, pos, vel))]
    fn thrust_accel(
        &self,
        py: Python<'_>,
        time: TimeArg,
        pos: &Bound<'_, PyAny>,
        vel: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let pos = to_vector3(pos, "pos")?;
        let vel = to_vector3(vel, "vel")?;
        SatProperties::thrust_accel(&self.0, &time.0, &pos, &vel)
            .map(|a| numpy::PyArray1::from_slice(py, a.as_slice()).into_py_any(py))
            .transpose()
    }

    // The state is the bytes below for a `satproperties`, and for an
    // instance of a subclass with attributes of its own a tuple of those
    // bytes and a copy of the instance `__dict__`.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> Result<()> {
        let (bytes, attrs) = match state.cast::<PyTuple>() {
            Ok(t) if t.len() == 2 => (t.get_item(0)?, Some(t.get_item(1)?)),
            Ok(_) => bail!("invalid satproperties pickle: expected (bytes, dict)"),
            Err(_) => (state.clone(), None),
        };
        let Ok(bytes) = bytes.cast::<PyBytes>() else {
            bail!("invalid satproperties pickle: expected bytes");
        };
        if let Some(attrs) = attrs {
            slf.getattr("__dict__")?.call_method1("update", (attrs,))?;
        }
        let state = bytes.as_bytes();
        let mut this = slf.borrow_mut();
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

        this.0.craoverm = read_f64(1);
        this.0.cdaoverm = read_f64(9);
        this.0.thrust = satkit::orbitprop::ThrustProfile::default();

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
            this.0.thrust.thrusts.push(
                satkit::orbitprop::ContinuousThrust::new(
                    satkit::mathtypes::Vector3::from_slice(&accel),
                    frame,
                    start,
                    end,
                )
                .map_err(|e| anyhow::anyhow!("invalid thrust in pickle: {e}"))?,
            );
        }

        this.0.ecom = None;
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
                        this.0.ecom = Some(ecom_from_block(&tail[1..])?);
                    }
                    other => bail!("invalid satproperties pickle: bad ECOM flag {other}"),
                }
            }
        }
        Ok(())
    }

    fn __getstate__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        // See `__setstate__` for the format. Values are written little-endian via
        // `to_le_bytes`, so there is no alignment assumption on the buffer.
        let mut raw: Vec<u8> =
            Vec::with_capacity(21 + this.0.thrust.thrusts.len() * 41 + 1 + ECOM_BLOCK_LEN);
        raw.push(3u8); // version
        raw.extend_from_slice(&this.0.craoverm.to_le_bytes());
        raw.extend_from_slice(&this.0.cdaoverm.to_le_bytes());
        raw.extend_from_slice(&(this.0.thrust.thrusts.len() as u32).to_le_bytes());
        for t in &this.0.thrust.thrusts {
            for v in t.accel.as_slice() {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            raw.push(crate::pyutils::maneuver_frame_to_u8(t.frame)?);
            raw.extend_from_slice(&t.start.raw.to_le_bytes());
            raw.extend_from_slice(&t.end.raw.to_le_bytes());
        }
        match &this.0.ecom {
            Some(e) => {
                raw.push(1u8);
                raw.extend_from_slice(&encode_ecom_block(e));
            }
            None => raw.push(0u8),
        }
        let bytes = PyBytes::new(py, &raw);
        // A subclass instance also carries its own attributes
        if let Ok(attrs) = slf.getattr("__dict__") {
            if attrs.len()? > 0 {
                return (bytes, attrs.call_method0("copy")?).into_py_any(py);
            }
        }
        bytes.into_py_any(py)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        let thrusts: Vec<String> = self
            .0
            .thrust
            .thrusts
            .iter()
            .map(|t| PyThrust(t.clone()).__repr__())
            .collect();
        format!(
            "satproperties(cdaoverm={:?}, craoverm={:?}, thrusts=[{}], ecom={})",
            self.0.cdaoverm,
            self.0.craoverm,
            thrusts.join(", "),
            self.0
                .ecom
                .map_or_else(|| "None".to_string(), |e| PyEcomParams(e).__repr__()),
        )
    }

    /// Equal when the drag and radiation-pressure coefficients, the thrust
    /// arcs (in order) and the ECOM coefficients are all equal
    fn __eq__(&self, other: &Self) -> bool {
        let (a, b) = (&self.0, &other.0);
        a.cdaoverm == b.cdaoverm
            && a.craoverm == b.craoverm
            && a.ecom == b.ecom
            && a.thrust.thrusts.len() == b.thrust.thrusts.len()
            && a.thrust
                .thrusts
                .iter()
                .zip(&b.thrust.thrusts)
                .all(|(x, y)| crate::pythrust::thrust_eq(x, y))
    }
}
