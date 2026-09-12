use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use satkit::kepler::{Anomaly, Kepler};

use crate::pyduration::PyDuration;
use crate::pyutils::py_to_smatrix;

///
/// Representation of Keplerian orbital elements
///
/// The anomaly can be given positionally as the true anomaly (6th argument)
/// or by exactly one of the keyword arguments ``true_anomaly``,
/// ``eccentric_anomaly`` or ``mean_anomaly``.
///
/// Args:
///     a: semi-major axis, meters (> 0)
///     eccen: Eccentricity (0 <= eccen < 1)
///     incl: Inclination, radians (0 <= incl <= pi)
///     raan: Right Ascension of the Ascending Node, radians
///     argp: Argument of Periapsis, radians (``w`` is accepted as an alias)
///     nu: True Anomaly, radians
///
/// Keyword Args:
///     true_anomaly: True Anomaly, radians
///     eccentric_anomaly: Eccentric Anomaly, radians
///     mean_anomaly: Mean Anomaly, radians
///     mu: Gravitational parameter of the central body, m^3/s^2
///         (default: Earth, ``satkit.consts.mu_earth``)
///
/// Raises:
///     ValueError: an element outside its domain (non-finite, ``a <= 0``,
///         ``eccen`` outside [0, 1), ``incl`` outside [0, pi], ``mu <= 0``),
///         or an ambiguous anomaly / ``argp``-``w`` specification. The
///         element setters apply the same checks.
///
/// Returns:
///     Kepler: Keplerian orbital elements
///
#[pyclass(name = "kepler", module = "satkit", from_py_object)]
#[derive(Clone)]
pub struct PyKepler(pub Kepler);

/// Map a core validation failure to the Python exception the stubs promise.
fn value_error(e: satkit::kepler::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

impl PyKepler {
    /// The same elements with the in-plane position replaced by `an`,
    /// converted to true anomaly by the core solver.
    fn with_anomaly(&self, an: Anomaly) -> Kepler {
        let k = &self.0;
        Kepler::new(k.a, k.eccen, k.incl, k.raan, k.argp, an).with_mu(k.mu)
    }

    /// Assign `field` on a copy, validate, and commit only if the result is
    /// a closed orbit; otherwise `ValueError` and the element set is unchanged.
    fn set_validated(&mut self, field: impl FnOnce(&mut Kepler)) -> PyResult<()> {
        let mut k = self.0;
        field(&mut k);
        k.validate().map_err(value_error)?;
        self.0 = k;
        Ok(())
    }

    /// `ValueError` for a non-finite anomaly, matching the message format of
    /// the core validation errors.
    fn check_anomaly(name: &str, val: f64) -> PyResult<()> {
        if val.is_finite() {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid Keplerian element {name} = {val}: must be finite"
            )))
        }
    }

    fn warn_w_deprecated(py: Python) -> PyResult<()> {
        let warning_type = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
        PyErr::warn(
            py,
            warning_type.as_any(),
            c"kepler.w is deprecated; use kepler.argp",
            2,
        )
    }
}

#[pymethods]
impl PyKepler {
    #[new]
    #[pyo3(signature = (a, eccen, incl, raan, argp=None, nu=None, *, w=None, true_anomaly=None, eccentric_anomaly=None, mean_anomaly=None, mu=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: f64,
        eccen: f64,
        incl: f64,
        raan: f64,
        argp: Option<f64>,
        nu: Option<f64>,
        w: Option<f64>,
        true_anomaly: Option<f64>,
        eccentric_anomaly: Option<f64>,
        mean_anomaly: Option<f64>,
        mu: Option<f64>,
    ) -> PyResult<Self> {
        let argp = match (argp, w) {
            (Some(v), None) | (None, Some(v)) => v,
            (None, None) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "missing required argument: 'argp' (argument of periapsis, radians)",
                ))
            }
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "specify the argument of periapsis as argp or w, not both",
                ))
            }
        };
        let an =
            match (nu, true_anomaly, eccentric_anomaly, mean_anomaly) {
                (Some(v), None, None, None) | (None, Some(v), None, None) => Anomaly::True(v),
                (None, None, Some(v), None) => Anomaly::Eccentric(v),
                (None, None, None, Some(v)) => Anomaly::Mean(v),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify exactly one of nu, true_anomaly, eccentric_anomaly, or mean_anomaly",
                )),
            };
        let mut k = Kepler::try_new(a, eccen, incl, raan, argp, an).map_err(value_error)?;
        if let Some(mu) = mu {
            k = k.with_mu(mu);
            k.validate().map_err(value_error)?;
        }
        Ok(Self(k))
    }

    #[getter]
    /// Semi-major axis, meters
    fn get_a(&self) -> f64 {
        self.0.a
    }

    #[setter(a)]
    fn set_a(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.a = val)
    }

    #[getter]
    /// Eccentricity, unitless
    fn get_eccen(&self) -> f64 {
        self.0.eccen
    }

    #[setter(eccen)]
    fn set_eccen(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.eccen = val)
    }

    #[getter]
    /// Inclination, radians
    fn get_inclination(&self) -> f64 {
        self.0.incl
    }

    #[setter(inclination)]
    fn set_inclination(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.incl = val)
    }

    #[getter]
    /// Right Ascension of the Ascending Node, radians
    fn get_raan(&self) -> f64 {
        self.0.raan
    }

    #[setter(raan)]
    fn set_raan(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.raan = val)
    }

    #[getter]
    /// Argument of Periapsis, radians
    fn get_argp(&self) -> f64 {
        self.0.argp
    }

    #[setter(argp)]
    fn set_argp(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.argp = val)
    }

    #[getter]
    /// Argument of Periapsis, radians
    ///
    /// Deprecated alias of ``argp`` (kept indefinitely); emits
    /// ``DeprecationWarning``.
    fn get_w(&self, py: Python) -> PyResult<f64> {
        Self::warn_w_deprecated(py)?;
        Ok(self.0.argp)
    }

    #[setter(w)]
    fn set_w(&mut self, py: Python, val: f64) -> PyResult<()> {
        Self::warn_w_deprecated(py)?;
        self.set_validated(|k| k.argp = val)
    }

    #[getter]
    /// True Anomaly, radians
    fn get_nu(&self) -> f64 {
        self.0.nu
    }

    #[setter(nu)]
    fn set_nu(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.nu = val)
    }

    #[getter]
    /// Gravitational parameter of the central body, m^3/s^2
    fn get_mu(&self) -> f64 {
        self.0.mu
    }

    #[setter(mu)]
    fn set_mu(&mut self, val: f64) -> PyResult<()> {
        self.set_validated(|k| k.mu = val)
    }

    /// Convert Keplerian elements to Cartesian
    /// position (meters) and velocity (meters/second)
    fn to_pv(&self) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (r, v) = self.0.to_pv();
        pyo3::Python::attach(|py| -> PyResult<(Py<PyAny>, Py<PyAny>)> {
            Ok((
                numpy::PyArray::from_slice(py, r.as_slice()).into_py_any(py)?,
                numpy::PyArray::from_slice(py, v.as_slice()).into_py_any(py)?,
            ))
        })
    }

    /// Convert Cartesian elements to kepler
    ///
    /// Args:
    ///     pos: 3-element position vector, meters
    ///     vel: 3-element velocity vector, meters/second
    ///
    /// Keyword Args:
    ///     mu: Gravitational parameter of the central body, m^3/s^2
    ///         (default: Earth); the returned elements carry it
    ///
    /// Raises:
    ///     ValueError: open (eccen >= 1) or rectilinear (zero angular
    ///         momentum) state, or ``mu`` not positive and finite
    ///     RuntimeError: inputs that are not 3-element vectors
    #[staticmethod]
    #[pyo3(signature = (pos, vel, *, mu=None))]
    fn from_pv(pos: &Bound<PyAny>, vel: &Bound<PyAny>, mu: Option<f64>) -> PyResult<Self> {
        let pos = py_to_smatrix(pos)?;
        let vel = py_to_smatrix(vel)?;
        let mu = mu.unwrap_or(satkit::consts::MU_EARTH);
        if !(mu.is_finite() && mu > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid Keplerian element mu = {mu}: gravitational parameter must be positive"
            )));
        }
        Kepler::from_pv_with_mu(pos, vel, mu)
            .map(Self)
            .map_err(value_error)
    }

    /// Propagate the elements forward (or backward) in time
    ///
    /// Args:
    ///     dt (duration | float | int): time to propagate; a number is
    ///         interpreted as seconds
    ///
    /// Returns:
    ///     kepler: new element set
    fn propagate(&self, dt: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(dt) = dt.extract::<PyDuration>() {
            Ok(Self(self.0.propagate(&dt.0)))
        } else {
            let secs = dt.extract::<f64>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "dt must be a satkit.duration or a number of seconds",
                )
            })?;
            Ok(Self(
                self.0.propagate(&satkit::Duration::from_seconds(secs)),
            ))
        }
    }

    /// Return the eccentric anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Eccentric Anomaly, radians
    #[getter]
    fn eccentric_anomaly(&self) -> f64 {
        self.0.eccentric_anomaly()
    }

    #[setter(eccentric_anomaly)]
    fn set_eccentric_anomaly(&mut self, val: f64) -> PyResult<()> {
        Self::check_anomaly("eccentric_anomaly", val)?;
        self.0 = self.with_anomaly(Anomaly::Eccentric(val));
        Ok(())
    }

    /// Return the mean motion of the satellite in radians/second
    ///
    /// Returns:
    ///    float: Mean motion, radians/second
    #[getter]
    fn mean_motion(&self) -> f64 {
        self.0.mean_motion()
    }

    /// Return the period of the satellite in seconds
    //
    /// Returns:
    ///   float: Period, seconds
    #[getter]
    fn period(&self) -> f64 {
        self.0.period()
    }

    /// Return the semiparameter (semi-latus rectum) of the orbit in meters
    ///
    /// Returns:
    ///     float: Semiparameter p = a (1 - e^2), meters
    #[getter]
    fn semiparameter(&self) -> f64 {
        self.0.semiparameter()
    }

    /// Radius of periapsis a (1 - e), meters
    #[getter]
    fn periapsis(&self) -> f64 {
        self.0.periapsis()
    }

    /// Radius of apoapsis a (1 + e), meters
    #[getter]
    fn apoapsis(&self) -> f64 {
        self.0.apoapsis()
    }

    /// Specific orbital energy -mu / 2a, J/kg
    #[getter]
    fn specific_energy(&self) -> f64 {
        self.0.specific_energy()
    }

    /// Magnitude of the specific angular momentum sqrt(mu p), m^2/s
    #[getter]
    fn angular_momentum(&self) -> f64 {
        self.0.angular_momentum()
    }

    /// Flight-path angle atan2(e sin nu, 1 + e cos nu), radians: the angle of
    /// the velocity above the local horizontal, zero at periapsis and
    /// apoapsis, positive while climbing
    #[getter]
    fn flight_path_angle(&self) -> f64 {
        self.0.flight_path_angle()
    }

    /// Argument of latitude u = argp + nu, radians in [0, 2 pi); well defined
    /// for circular orbits
    #[getter]
    fn argument_of_latitude(&self) -> f64 {
        self.0.argument_of_latitude()
    }

    /// True longitude raan + argp + nu, radians in [0, 2 pi); well defined
    /// for circular equatorial orbits
    #[getter]
    fn true_longitude(&self) -> f64 {
        self.0.true_longitude()
    }

    /// Return the mean anomaly of the satellite in radians
    ///
    /// Returns:
    ///     float: Mean Anomaly, radians
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly()
    }

    #[setter(mean_anomaly)]
    fn set_mean_anomaly(&mut self, val: f64) -> PyResult<()> {
        // A non-finite M is refused up front so the element set can never
        // hold a NaN anomaly; the core solver itself is iteration-capped and
        // cannot hang on any input regardless.
        Self::check_anomaly("mean_anomaly", val)?;
        self.0 = self.with_anomaly(Anomaly::Mean(val));
        Ok(())
    }

    /// Return the true anomaly of the satellite in radians
    ///
    /// Returns:
    ///   float: True Anomaly, radians
    #[getter]
    fn true_anomaly(&self) -> f64 {
        self.0.nu
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __repr__(&self) -> String {
        let k = &self.0;
        format!(
            "kepler(a={:.6e}, eccen={:.6e}, incl={:.6e}, raan={:.6e}, argp={:.6e}, nu={:.6e}, mu={:.6e})",
            k.a, k.eccen, k.incl, k.raan, k.argp, k.nu, k.mu
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        crate::pyutils::pack_f64s(
            py,
            &[
                self.0.a,
                self.0.eccen,
                self.0.incl,
                self.0.raan,
                self.0.argp,
                self.0.nu,
                self.0.mu,
            ],
        )
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> PyResult<()> {
        let [a, eccen, incl, raan, argp, nu, mu] = crate::pyutils::unpack_f64s(py, &state)?;
        self.0 = Kepler {
            a,
            eccen,
            incl,
            raan,
            argp,
            nu,
            mu,
        };
        Ok(())
    }

    fn __getnewargs_ex__<'a>(&self, py: Python<'a>) -> (Bound<'a, PyTuple>, Bound<'a, PyDict>) {
        let d = PyDict::new(py);
        let tp = PyTuple::new(py, vec![6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        (tp, d)
    }
}
