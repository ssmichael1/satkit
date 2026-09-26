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
}

crate::enum_pickle!(PySGP4Error, "sgp4_error");

#[allow(non_camel_case_types)]
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

/// Convert a Python value to an Instant. can be string, datetime, or PyInstant
pub(crate) fn epoch_from_val(val: &Bound<'_, PyAny>) -> Result<satkit::Instant> {
    if val.is_instance_of::<crate::pyinstant::PyInstant>() {
        let instant: crate::pyinstant::PyInstant = val.extract().unwrap();
        Ok(instant.0)
    } else if val.is_instance_of::<PyString>() {
        let s: String = val.extract()?;
        satkit::Instant::from_rfc3339(&s)
            .map_err(|e| anyhow::anyhow!("Invalid epoch string: {}", e))
    } else if val.is_instance_of::<PyDateTime>() {
        // Exact, with the same naive-is-local convention as satkit.time
        let tm = val.cast::<PyDateTime>().map_err(PyErr::from)?;
        Ok(crate::pyinstant::datetime_to_instant(tm)?)
    } else {
        bail!("Invalid epoch type");
    }
}

/// Pack a single SGP4 propagation result (position/velocity, and optionally the
/// error codes) into the Python return tuple. Shared by the TLE-object and
/// OMM-dict branches of [`sgp4`].
fn pack_sgp4_result(
    py: Python,
    states: &psgp4::SGP4State,
    output_err: bool,
    time_scalar: bool,
) -> Result<Py<PyAny>> {
    // (3,) for a single time; (N, 3) for a list / array of N times,
    // including N = 1
    let dims = if time_scalar {
        vec![states.pos.as_slice().len()]
    } else {
        vec![states.pos.ncols(), states.pos.nrows()]
    };

    // ndarray is row-major while numeris/numpy are column-major, hence the
    // dimension switch above.
    let pos = PyArray1::from_slice(py, states.pos.as_slice()).reshape(dims.clone())?;
    let vel = PyArray1::from_slice(py, states.vel.as_slice()).reshape(dims)?;
    if !output_err {
        Ok((pos, vel).into_py_any(py)?)
    } else {
        let eint: Vec<i32> = states.errcode.iter().map(|x| *x as i32).collect();
        Ok((pos, vel, PyArray1::from_slice(py, eint.as_slice())).into_py_any(py)?)
    }
}

/// Run SGP4 on one TLE or OMM at `time` with the GIL released; also returns
/// whether `time` was a scalar. Shared by the TLE-object and OMM-dict
/// branches of [`sgp4`].
fn sgp4_one(
    py: Python,
    src: &mut (impl psgp4::SGP4Source + Send),
    time: &Bound<'_, PyAny>,
    gravconst: psgp4::GravConst,
    opsmode: psgp4::OpsMode,
) -> Result<(psgp4::SGP4State, bool)> {
    let crate::pyinstant::TimeInput {
        times: tmvec,
        scalar: time_scalar,
    } = time.to_time_input()?;
    let states = py.detach(|| psgp4::sgp4_full(src, tmvec.as_slice(), gravconst, opsmode))?;
    Ok((states, time_scalar))
}

crate::arg_extractor!(gravconst_arg: GravConst, |e| {
    pyo3::exceptions::PyValueError::new_err(format!("Invalid gravconst: {e}"))
});
crate::arg_extractor!(opsmode_arg: OpsMode, |e| {
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
///     If errflag is True, a third element is returned: an int32 numpy array of error
///     codes for each TLE and time (0 = success). The codes are the integer values of
///     ``sgp4_error``, so ``err == satkit.sgp4_error.success`` compares elementwise.
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
/// >>> tle = satkit.TLE.from_lines(lines)  # a single TLE, not a list
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
    if tle.is_instance_of::<PyTLE>() {
        let mut stle: PyRefMut<PyTLE> = tle
            .extract()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE: {}", e)))?;
        // Clone the TLE and run SGP4 on the clone, then write the TLE back
        // so the cached SGP4 init state is preserved
        let mut rtle = stle.0.clone();
        let (states, time_scalar) = sgp4_one(py, &mut rtle, time, gravconst, opsmode)?;
        stle.0 = rtle;
        pack_sgp4_result(py, &states, errflag, time_scalar)
    }
    // Handle input as dict
    else if tle.is_instance_of::<PyDict>() {
        let dict: &Bound<'_, PyDict> = tle.cast().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE dictionary: {}", e))
        })?;
        let mut omm = omm_from_pydict(dict)?;
        let (states, time_scalar) = sgp4_one(py, &mut omm, time, gravconst, opsmode)?;
        pack_sgp4_result(py, &states, errflag, time_scalar)
    } else if tle.is_instance_of::<PyList>() {
        let plist = tle.cast::<PyList>().unwrap();
        let crate::pyinstant::TimeInput {
            times: tmarray,
            scalar: time_scalar,
        } = time.to_time_input()?;
        // The output reshape below cannot represent zero-length inputs
        if plist.is_empty() {
            bail!("TLE list must not be empty");
        }
        if tmarray.is_empty() {
            bail!("Time array must not be empty");
        }

        // Sources for the SGP4 computation, extracted with the GIL held;
        // the computation itself runs below with the GIL released.
        // TLE sources keep a handle to the originating Python object so the
        // cached SGP4 init state can be written back afterward.
        enum Sgp4Source {
            Tle(Py<PyTLE>, Box<satkit::TLE>),
            Omm(Box<satkit::omm::OMM>),
        }

        let mut sources: Vec<Sgp4Source> = plist
            .iter()
            .map(|item| -> Result<Sgp4Source> {
                if item.is_instance_of::<PyTLE>() {
                    let pytle: Py<PyTLE> = item.extract().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE: {}", e))
                    })?;
                    let rtle = Box::new(pytle.borrow(item.py()).0.clone());
                    Ok(Sgp4Source::Tle(pytle, rtle))
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
        let (gc, om) = (gravconst, opsmode);
        let results: Vec<psgp4::SGP4State> = tle.py().detach(|| {
            sources
                .iter_mut()
                .map(|src| -> Result<psgp4::SGP4State> {
                    match src {
                        Sgp4Source::Tle(_, rtle) => {
                            Ok(psgp4::sgp4_full(rtle.as_mut(), tmarray.as_slice(), gc, om)?)
                        }
                        Sgp4Source::Omm(omm) => {
                            Ok(psgp4::sgp4_full(omm.as_mut(), tmarray.as_slice(), gc, om)?)
                        }
                    }
                })
                .collect::<Result<Vec<_>>>()
        })?;

        // Write the TLEs back to preserve their cached SGP4 init state
        for src in &sources {
            if let Sgp4Source::Tle(pytle, rtle) = src {
                pytle.borrow_mut(py).0 = rtle.as_ref().clone();
            }
        }

        let ntimes = tmarray.len();
        let mut pos: Vec<f64> = Vec::with_capacity(plist.len() * ntimes * 3);
        let mut vel: Vec<f64> = Vec::with_capacity(plist.len() * ntimes * 3);
        let mut eint: Vec<i32> = Vec::with_capacity(plist.len() * ntimes);
        for states in &results {
            pos.extend_from_slice(states.pos.as_slice());
            vel.extend_from_slice(states.vel.as_slice());
            if errflag {
                eint.extend(states.errcode.iter().map(|&x| x as i32));
            }
        }

        // Set dimensions of output to remove singleton dimensions
        let dims = match (plist.len() > 1, !time_scalar) {
            (true, true) => vec![plist.len(), ntimes, 3],
            (true, false) => vec![plist.len(), 3],
            (false, true) => vec![ntimes, 3],
            (false, false) => vec![3],
        };
        // Dims for error output
        let edims = match (plist.len() > 1, !time_scalar) {
            (true, true) => vec![plist.len(), ntimes],
            (true, false) => vec![plist.len()],
            (false, true) => vec![ntimes],
            (false, false) => vec![1],
        };

        let pos = PyArray1::from_vec(py, pos).reshape(dims.clone())?;
        let vel = PyArray1::from_vec(py, vel).reshape(dims)?;
        if !errflag {
            Ok((pos, vel).into_py_any(py)?)
        } else {
            Ok((pos, vel, PyArray1::from_vec(py, eint).reshape(edims)?).into_py_any(py)?)
        }
    } else {
        bail!("Invalid input type for argument 1");
    }
}
