use pyo3::prelude::*;
use pyo3::types::{PyDateTime, PyDict, PyList, PyString};
use pyo3::IntoPyObjectExt;

use crate::pyinstant::ToTimeVec;
use crate::pyomm::omm_from_pydict;
use crate::pytle::PyTLE;
use numpy::PyArray1;
use numpy::PyArrayMethods;
use satkit::sgp4 as psgp4;

use anyhow::{bail, Result};

// Thin Python wrapper around SGP4 Error
#[allow(non_camel_case_types)]
/// Represent errors from SGP-4 propagation of two-line element sets (TLEs)
#[pyclass(name = "sgp4_error", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySGP4Error {
    success = psgp4::SGP4Error::SGP4Success as isize,
    eccen = psgp4::SGP4Error::SGP4ErrorEccen as isize,
    mean_motion = psgp4::SGP4Error::SGP4ErrorMeanMotion as isize,
    perturb_eccen = psgp4::SGP4Error::SGP4ErrorPerturbEccen as isize,
    semi_latus_rectum = psgp4::SGP4Error::SGP4ErrorSemiLatusRectum as isize,
    unused = psgp4::SGP4Error::SGP4ErrorUnused as isize,
    orbit_decay = psgp4::SGP4Error::SGP4ErrorOrbitDecay as isize,
    /// Only in the error array of ``sgp4(..., errflag=True)``: the element
    /// set cannot be propagated by classic SGP4 at all (an SGP4-XP set, or
    /// OMM metadata naming another theory, time system, frame or center)
    unsupported = 7,
}

crate::enum_pickle!(PySGP4Error, "sgp4_error");

#[allow(non_camel_case_types)]
/// Gravity constant to use for SGP4 propagation
#[pyclass(name = "sgp4_gravconst", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum GravConst {
    wgs72 = psgp4::GravConst::WGS72 as isize,
    wgs72old = psgp4::GravConst::WGS72OLD as isize,
    wgs84 = psgp4::GravConst::WGS84 as isize,
}

crate::enum_pickle!(GravConst, "sgp4_gravconst");

impl From<GravConst> for psgp4::GravConst {
    fn from(f: GravConst) -> Self {
        match f {
            GravConst::wgs72 => Self::WGS72,
            GravConst::wgs72old => Self::WGS72OLD,
            GravConst::wgs84 => Self::WGS84,
        }
    }
}

#[allow(non_camel_case_types)]
/// Ops Mode for SGP4 Propagation
#[pyclass(name = "sgp4_opsmode", eq, eq_int, from_py_object)]
#[derive(Clone, Eq, PartialEq)]
pub enum OpsMode {
    afspc = psgp4::OpsMode::AFSPC as isize,
    improved = psgp4::OpsMode::IMPROVED as isize,
}

crate::enum_pickle!(OpsMode, "sgp4_opsmode");

impl From<OpsMode> for psgp4::OpsMode {
    fn from(f: OpsMode) -> Self {
        match f {
            OpsMode::afspc => Self::AFSPC,
            OpsMode::improved => Self::IMPROVED,
        }
    }
}

impl From<psgp4::SGP4Error> for PySGP4Error {
    fn from(f: psgp4::SGP4Error) -> Self {
        match f {
            psgp4::SGP4Error::SGP4Success => Self::success,
            psgp4::SGP4Error::SGP4ErrorEccen => Self::eccen,
            psgp4::SGP4Error::SGP4ErrorMeanMotion => Self::mean_motion,
            psgp4::SGP4Error::SGP4ErrorPerturbEccen => Self::perturb_eccen,
            psgp4::SGP4Error::SGP4ErrorSemiLatusRectum => Self::semi_latus_rectum,
            psgp4::SGP4Error::SGP4ErrorUnused => Self::unused,
            psgp4::SGP4Error::SGP4ErrorOrbitDecay => Self::orbit_decay,
        }
    }
}

/// Convert a Python value to an Instant. can be string, datetime, numpy.datetime64, or PyInstant
pub(crate) fn epoch_from_val(val: &Bound<'_, PyAny>) -> Result<satkit::Instant> {
    if val.is_instance_of::<crate::pyinstant::PyInstant>() {
        let instant: crate::pyinstant::PyInstant = val.extract().unwrap();
        Ok(instant.0)
    } else if val.is_instance_of::<PyString>() {
        let s: String = val.extract()?;
        // ValueError, like the other time-string parsers
        satkit::Instant::from_rfc3339(&s).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid epoch string: {e}")).into()
        })
    } else if val.is_instance_of::<PyDateTime>() {
        // Exact, with the same naive-is-local convention as satkit.time
        let tm = val.cast::<PyDateTime>().map_err(PyErr::from)?;
        Ok(crate::pyinstant::datetime_to_instant(tm)?)
    } else if crate::pyinstant::is_time_scalar(val) {
        // numpy.datetime64 (a UTC label)
        Ok(val.extract::<crate::pyinstant::TimeArg>()?.0)
    } else {
        bail!("Invalid epoch type");
    }
}

/// The integer error code reported, at every time, for an element set that
/// failed to initialize: its SGP4 init code, or `unsupported` when the
/// source refused it (SGP4-XP, foreign OMM metadata).
fn init_error_code(e: &psgp4::Error) -> i32 {
    match e {
        psgp4::Error::SatRecInit(code) => *code as i32,
        psgp4::Error::Source(_) => PySGP4Error::unsupported as i32,
    }
}

/// One element set's positions and velocities (column-major 3×N, as in
/// `SGP4State`) and integer error codes at `ntimes` times. A set that failed
/// to initialize is NaN at every time, with its init error code.
type Flat = (Vec<f64>, Vec<f64>, Vec<i32>);

/// Takes the result by value so the position and velocity buffers are moved
/// out, not copied.
fn flatten(res: std::result::Result<psgp4::SGP4State, psgp4::Error>, ntimes: usize) -> Flat {
    match res {
        Ok(states) => (
            states.pos.into_vec(),
            states.vel.into_vec(),
            states.errcode.iter().map(|&x| x as i32).collect(),
        ),
        Err(e) => (
            vec![f64::NAN; 3 * ntimes],
            vec![f64::NAN; 3 * ntimes],
            vec![init_error_code(&e); ntimes],
        ),
    }
}

/// Pack positions, velocities and (optionally) error codes, flattened as in
/// [`flatten`] and concatenated over element sets, into the Python return
/// tuple, with `dims` the shape of the position and velocity arrays and
/// `edims` that of the error array.
fn pack_sgp4_result(
    py: Python,
    flat: Flat,
    errflag: bool,
    dims: Vec<usize>,
    edims: Vec<usize>,
) -> Result<Py<PyAny>> {
    let (pos, vel, codes) = flat;
    // ndarray is row-major while numeris/numpy are column-major, so the
    // column-major 3×N blocks read as (N, 3) rows.
    let pos = PyArray1::from_vec(py, pos).reshape(dims.clone())?;
    let vel = PyArray1::from_vec(py, vel).reshape(dims.clone())?;
    if !errflag {
        Ok((pos, vel).into_py_any(py)?)
    } else {
        Ok((pos, vel, PyArray1::from_vec(py, codes).reshape(edims)?).into_py_any(py)?)
    }
}

/// Copy of a Python TLE's contents, taken under a short shared borrow.
///
/// `try_borrow` rather than `borrow`: a TLE that another thread is modifying
/// at this instant raises `RuntimeError` instead of panicking (a panic
/// surfaces as `PanicException`, which `except Exception` does not catch).
fn snapshot_tle(pytle: &Bound<'_, PyTLE>) -> PyResult<satkit::TLE> {
    let tle = pytle.try_borrow().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "TLE is being modified by another thread; retry the call",
        )
    })?;
    Ok(tle.0.clone())
}

/// Store the propagated copy of a TLE back in its Python object.
///
/// The write-back only carries the SGP4 initialisation that SGP4 caches in
/// the TLE on first use, so later calls on the same object skip
/// re-initialising; the elements themselves are unchanged. It is therefore
/// best-effort: it is skipped when another thread holds a borrow of the TLE
/// right now (`try_borrow_mut` fails), or when the TLE's elements no longer
/// equal the `snapshot` taken before SGP4 ran (another thread modified it
/// meanwhile), so a concurrent edit is never overwritten with stale elements.
/// TLE equality ignores the cache, and the cache records the elements,
/// gravity model and ops mode it was built from, so writing back a cache
/// built for other settings than a concurrent call's is harmless: SGP4 just
/// re-initializes next time.
fn write_back_tle(pytle: &Bound<'_, PyTLE>, snapshot: &satkit::TLE, propagated: satkit::TLE) {
    if let Ok(mut cur) = pytle.try_borrow_mut() {
        if cur.0 == *snapshot {
            cur.0 = propagated;
        }
    }
}

/// Run SGP4 on one TLE or OMM at `time` with the GIL released and pack the
/// result. Shared by the TLE-object and OMM-dict branches of [`sgp4`].
///
/// An element set that fails to initialize raises, unless `errflag` is set:
/// then it is NaN at every time, with its init error code.
fn sgp4_one(
    py: Python,
    src: &mut (impl psgp4::SGP4Source + Send),
    time: &Bound<'_, PyAny>,
    gravconst: psgp4::GravConst,
    opsmode: psgp4::OpsMode,
    errflag: bool,
) -> Result<Py<PyAny>> {
    let crate::pyinstant::TimeInput {
        times: tmvec,
        scalar: time_scalar,
    } = time.to_time_input()?;
    let res = py.detach(|| psgp4::sgp4_full(src, tmvec.as_slice(), gravconst, opsmode));
    if !errflag {
        if let Err(e) = res {
            return Err(e.into());
        }
    }
    // (3,) for a single time; (N, 3) for a list / array of N times,
    // including N = 1. The error array is (N,) either way.
    let n = tmvec.len();
    let dims = if time_scalar { vec![3] } else { vec![n, 3] };
    pack_sgp4_result(py, flatten(res, n), errflag, dims, vec![n])
}

crate::arg_extractor!(pub(crate) gravconst_arg: GravConst, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid gravconst: {e}"))
});
crate::arg_extractor!(pub(crate) opsmode_arg: OpsMode, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid opsmode: {e}"))
});

/// SGP-4 propagator for TLE
///
/// Note:
///     Run Simplified General Perturbations (SGP)-4 propagator on Two-Line Element Set to
///     output satellite position and velocity at given time
///     in the "TEME" coordinate system
///
/// Note:
///     A detailed description is at:
///     https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf
///
/// Args:
///     tle (TLE | list[TLE] | dict): TLE or OMM dictionary (or list of TLEs) on which to operate
///     time (time | list[time] | npt.ArrayLike[time]): time(s) at which to compute position and velocity
///
/// Keyword Args:
///     gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is gravconst.wgs72
///     opsmode (satkit.sgp4_opsmode): opsmode.afspc (Air Force Space Command) or opsmode.improved.  Default is opsmode.afspc
///     errflag (bool): whether or not to output error conditions for each TLE and time output.  Default is False
///
/// Returns:
///     tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: position and velocity in
///     **meters** and **meters/second**, respectively, in the TEME frame at each of the
///     "Ntime" input times and each of the "Ntle" tles. Shape is (3,) for a single TLE and
///     single time, (Ntime, 3) for a single TLE and multiple times, (Ntle, 3) for a list of
///     TLEs and a single time, and (Ntle, Ntime, 3) for a list of TLEs and multiple times.
///     A list of TLEs keeps its TLE axis even with one element, and a list of times its
///     time axis ("list in, list out"); empty lists give empty arrays, e.g. (Ntle, 0, 3).
///     If errflag is True, a third element is returned: an int32 numpy array of error
///     codes for each TLE and time (0 = success). The codes are the integer values of
///     ``sgp4_error``, so ``err == satkit.sgp4_error.success`` compares elementwise.
///
/// Note:
///     Errors: a time at which propagation fails (e.g. the orbit has decayed) gives a
///     NaN row, with its code in the error array when errflag is True. An element set
///     that cannot be initialized at all (e.g. decayed or eccentricity out of range at
///     epoch, or an SGP4-XP set) raises ``RuntimeError`` when errflag is False; for a
///     list, the message gives its index. When errflag is True it does not raise: its
///     rows are NaN at every time and every time carries its init code
///     (``sgp4_error.unsupported`` for an SGP4-XP set or OMM metadata SGP4 cannot use),
///     so one bad element set does not fail a list.
///
/// Note:
///     Leap seconds: the time since epoch is the physical (SI) time elapsed. Across a
///     leap second this is one second more than the difference of the UTC labels that
///     Vallado's reference code and python-sgp4 use, so satkit differs from them by 1 s
///     of along-track motion (~7.6 km at LEO) per leap second between epoch and time.
///     This is deliberate: the satellite really flies 86,401 s over such a day, and the
///     SGP4 mean motion is per SI day.
///
/// Note:
///     TEME ("True Equator Mean Equinox") has the true equator and the mean equinox of
///     date, i.e. of each output time (not of the TLE epoch). Rotate to GCRF with
///     ``frametransform.rotation(frame.TEME, frame.GCRF, time)``.
///
/// Note:
///     Units: the canonical Vallado SGP4 implementation (and most other SGP4 libraries)
///     return position in kilometers and velocity in kilometers/second. satkit converts
///     these to meters and meters/second so that SGP4 output is consistent with every
///     other position and velocity in the library.
///
///
/// Example:
///
///
/// >>> import numpy as np
/// >>> import satkit
/// >>>
/// >>> lines = [
/// >>>     "0 INTELSAT 902",
/// >>>     "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
/// >>>     "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981",
/// >>> ]
/// >>>
/// >>> tle = satkit.TLE.from_lines(lines)[0]  # from_lines always returns a list
/// >>> tm = tle.epoch
/// >>>
/// >>> # Compute TEME position & velocity at epoch
/// >>> pteme, vteme = satkit.sgp4(tle, tm)
/// >>>
/// >>> # Rotate to ITRF frame; the velocity also loses the Earth-rotation term
/// >>> q = satkit.frametransform.qteme2itrf(tm)
/// >>> pitrf = q * pteme
/// >>> vitrf = q * vteme - np.cross(np.array([0, 0, satkit.consts.omega_earth]), pitrf)
/// >>>
/// >>> # convert to ITRF coordinate object
/// >>> coord = satkit.itrfcoord(pitrf)
/// >>>
/// >>> # Print ITRF coordinate object location
/// >>> print(coord)
/// ITRFCoord(lat:  -0.0362 deg, lon:  62.0172 deg, hae: 35799.52 km)
#[pyfunction]
#[pyo3(signature=(tle, time, *, gravconst=GravConst::wgs72, opsmode=OpsMode::afspc, errflag=false))]
pub fn sgp4(
    tle: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
    #[pyo3(from_py_with = gravconst_arg)] gravconst: GravConst,
    #[pyo3(from_py_with = opsmode_arg)] opsmode: OpsMode,
    errflag: bool,
) -> Result<Py<PyAny>> {
    let py = tle.py();
    let gravconst: psgp4::GravConst = gravconst.into();
    let opsmode: psgp4::OpsMode = opsmode.into();

    // Handle input as TLE
    if let Ok(pytle) = tle.cast::<PyTLE>() {
        // No borrow of the Python object is held while the GIL is released,
        // so other threads can read, modify or propagate the same TLE
        // meanwhile; SGP4 runs on a clone.
        let snapshot = snapshot_tle(pytle)?;
        let mut rtle = snapshot.clone();
        let out = sgp4_one(py, &mut rtle, time, gravconst, opsmode, errflag);
        write_back_tle(pytle, &snapshot, rtle);
        out
    }
    // Handle input as dict
    else if tle.is_instance_of::<PyDict>() {
        let dict: &Bound<'_, PyDict> = tle.cast().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE dictionary: {}", e))
        })?;
        let mut omm = omm_from_pydict(dict)?;
        sgp4_one(py, &mut omm, time, gravconst, opsmode, errflag)
    } else if tle.is_instance_of::<PyList>() {
        let plist = tle.cast::<PyList>().unwrap();
        let crate::pyinstant::TimeInput {
            times: tmarray,
            scalar: time_scalar,
        } = time.to_time_input()?;

        // Sources for the SGP4 computation, extracted with the GIL held;
        // the computation itself runs below with the GIL released.
        // TLE sources keep a handle to the originating Python object and
        // the snapshot the copy was made from, so the cached SGP4 init state
        // can be written back afterward (see `write_back_tle`).
        enum Sgp4Source {
            Tle(Py<PyTLE>, Box<satkit::TLE>, Box<satkit::TLE>),
            Omm(Box<satkit::omm::OMM>),
        }

        let mut sources: Vec<Sgp4Source> = plist
            .iter()
            .map(|item| -> Result<Sgp4Source> {
                if let Ok(pytle) = item.cast::<PyTLE>() {
                    let snapshot = Box::new(snapshot_tle(pytle)?);
                    let rtle = snapshot.clone();
                    Ok(Sgp4Source::Tle(pytle.clone().unbind(), rtle, snapshot))
                } else if item.is_instance_of::<PyDict>() {
                    let dict: &Bound<'_, PyDict> = item.cast().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid TLE dictionary: {}",
                            e
                        ))
                    })?;
                    Ok(Sgp4Source::Omm(Box::new(omm_from_pydict(dict)?)))
                } else {
                    bail!("Invalid TLE in list");
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Honor the gravconst / opsmode kwargs on the list path too (previously
        // this called the default-config `sgp4`, silently ignoring them).
        // Each element set's result is kept separately (see `sgp4_one` for
        // how one that fails to initialize is reported).
        let (gc, om) = (gravconst, opsmode);
        let results: Vec<std::result::Result<psgp4::SGP4State, psgp4::Error>> =
            tle.py().detach(|| {
                sources
                    .iter_mut()
                    .map(|src| match src {
                        Sgp4Source::Tle(_, rtle, _) => {
                            psgp4::sgp4_full(rtle.as_mut(), tmarray.as_slice(), gc, om)
                        }
                        Sgp4Source::Omm(omm) => {
                            psgp4::sgp4_full(omm.as_mut(), tmarray.as_slice(), gc, om)
                        }
                    })
                    .collect()
            });

        // Write the TLEs back to preserve their cached SGP4 init state
        for src in sources {
            if let Sgp4Source::Tle(pytle, rtle, snapshot) = src {
                write_back_tle(pytle.bind(py), &snapshot, *rtle);
            }
        }

        if !errflag {
            if let Some((i, e)) = results
                .iter()
                .enumerate()
                .find_map(|(i, r)| r.as_ref().err().map(|e| (i, e)))
            {
                bail!("element set {i} of the list: {e}");
            }
        }

        let ntimes = tmarray.len();
        // The TLEs actually propagated (the Python list could have been
        // changed by another thread while the GIL was released)
        let ntles = results.len();
        let mut flat: Flat = (
            Vec::with_capacity(ntles * ntimes * 3),
            Vec::with_capacity(ntles * ntimes * 3),
            Vec::with_capacity(ntles * ntimes),
        );
        for res in results {
            let (p, v, e) = flatten(res, ntimes);
            flat.0.extend(p);
            flat.1.extend(v);
            flat.2.extend(e);
        }

        // A list of TLEs always keeps its TLE axis, including a one-element
        // or empty list ("list in, list out", as for the time axis); only a
        // scalar time drops the time axis. Empty inputs give empty arrays.
        let (dims, edims) = if time_scalar {
            (vec![ntles, 3], vec![ntles])
        } else {
            (vec![ntles, ntimes, 3], vec![ntles, ntimes])
        };
        pack_sgp4_result(py, flat, errflag, dims, edims)
    } else {
        bail!("Invalid input type for argument 1");
    }
}
