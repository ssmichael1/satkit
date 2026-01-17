use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyDateTime};
use pyo3::IntoPyObjectExt;

use super::pyinstant::ToTimeVec;
use super::pytle::PyTLE;
use crate::sgp4 as psgp4;
use numpy::PyArray1;
use numpy::PyArrayMethods;

use anyhow::{bail, Result};

// Thin Python wrapper around SGP4 Error
#[allow(non_camel_case_types)]
#[pyclass(name = "sgp4_error", eq, eq_int)]
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

#[allow(non_camel_case_types)]
#[pyclass(name = "sgp4_gravconst", eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum GravConst {
    wgs72 = psgp4::GravConst::WGS72 as isize,
    wgs72old = psgp4::GravConst::WGS72OLD as isize,
    wgs84 = psgp4::GravConst::WGS84 as isize,
}

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
#[pyclass(name = "sgp4_opsmode", eq, eq_int)]
#[derive(Clone, Eq, PartialEq)]
pub enum OpsMode {
    afspc = psgp4::OpsMode::AFSPC as isize,
    improved = psgp4::OpsMode::IMPROVED as isize,
}

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
fn epoch_from_val(val: &Bound<'_, PyAny>) -> Result<crate::Instant> {
    if val.is_instance_of::<crate::pybindings::pyinstant::PyInstant>() {
        let instant: crate::pybindings::pyinstant::PyInstant = val.extract().unwrap();
        Ok(instant.0)
    }
    else if val.is_instance_of::<PyString>() {
        let s: String = val.extract()?;
        crate::Instant::from_rfc3339(&s).map_err(|e| {
            anyhow::anyhow!("Invalid epoch string: {}", e)
        })
    }
    else if val.is_instance_of::<PyDateTime>() {
        let tm: Py<PyDateTime> = val.extract().unwrap();
        pyo3::Python::attach(|py| {
            let ts: f64 = tm
                .call_method(py, "timestamp", (), None)?
                .extract::<f64>(py)?;
            Ok(crate::Instant::from_unixtime(ts))
        })
    }
    else {
        bail!("Invalid epoch type");
    }
}

/// OMM files can have floats as either strings or numbers
/// so handle both cases here
///
/// (very annoying!)
fn float_from_py(val: &Bound<'_, PyAny>) -> Result<f64> {
    if val.is_instance_of::<PyString>() {
        let s: String = val.extract()?;
        s.parse::<f64>().map_err(|e| {
            anyhow::anyhow!("Invalid float string: {}", e)
        })
    } else {
        val.extract::<f64>().map_err(|e| {
            anyhow::anyhow!("Invalid float value: {}", e)
        })
    }
}

fn omm_from_pydict(dict: &Bound<'_, PyDict>) -> Result<crate::OMM> {
    let mut omm = crate::OMM::default();

    omm.inclination = f64::NAN;
    omm.raan = f64::NAN;
    omm.eccentricity = f64::NAN;
    omm.arg_of_pericenter = f64::NAN;
    omm.mean_anomaly = f64::NAN;
    omm.mean_motion = f64::NAN;
    omm.epoch = String::new();

    if let Some(v) = dict.get_item("INCLINATION")? {
        omm.inclination = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("RA_OF_ASC_NODE")? {
        omm.raan = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("ECCENTRICITY")? {
        omm.eccentricity = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("ARG_OF_PERICENTER")? {
        omm.arg_of_pericenter = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("MEAN_ANOMALY")? {
        omm.mean_anomaly = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("MEAN_MOTION")? {
        omm.mean_motion = float_from_py(&v)?;
    }
    if let Some(v) = dict.get_item("EPOCH")? {
        omm.epoch = epoch_from_val(&v)?.as_rfc3339();
    }
    if let Some(v) = dict.get_item("BSTAR")? {
        omm.bstar = Some(float_from_py(&v)?);
    }
    if let Some(v) = dict.get_item("MEAN_MOTION_DOT")? {
        omm.mean_motion_dot = Some(float_from_py(&v)?);
    }
    if let Some(v) = dict.get_item("MEAN_MOTION_DDOT")? {
        omm.mean_motion_ddot = Some(float_from_py(&v)?);
    }
    if let Some(d) = dict.get_item("meanElements")? {
        let d = d.cast::<PyDict>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid meanElements dictionary: {}",
                e
            ))
        })?;
        if let Some(v) = d.get_item("EPOCH")? {
            omm.epoch = epoch_from_val(&v)?.as_rfc3339();
        }
        if let Some(v) = d.get_item("MEAN_MOTION")? {
            omm.mean_motion = float_from_py(&v)?;
        }
        if let Some(v) = d.get_item("ECCENTRICITY")? {
            omm.eccentricity = float_from_py(&v)?;
        }
        if let Some(v) = d.get_item("INCLINATION")? {
            omm.inclination = float_from_py(&v)?;
        }
        if let Some(v) = d.get_item("ARG_OF_PERICENTER")? {
            omm.arg_of_pericenter = float_from_py(&v)?;
        }
        if let Some(v) = d.get_item("RA_OF_ASC_NODE")? {
            omm.raan = float_from_py(&v)?;
        }
        if let Some(v) = d.get_item("MEAN_ANOMALY")? {
            omm.mean_anomaly = float_from_py(&v)?;
        }
    }
    if let Some(d) = dict.get_item("tleParameters")? {
        let d = d.cast::<PyDict>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid tleParameters dictionary: {}",
                e
            ))
        })?;
        if let Some(v) = d.get_item("BSTAR")? {
            omm.bstar = Some(float_from_py(&v)?);
        }
        if let Some(v) = d.get_item("MEAN_MOTION_DOT")? {
            omm.mean_motion_dot = Some(float_from_py(&v)?);
        }
        if let Some(v) = d.get_item("MEAN_MOTION_DDOT")? {
            omm.mean_motion_ddot = Some(float_from_py(&v)?);
        }
    }
    if omm.epoch.is_empty() {
        bail!("OMM epoch is required");
    }
    if omm.mean_motion.is_nan() {
        bail!("OMM mean motion is required");
    }
    if omm.eccentricity.is_nan() {
        bail!("OMM eccentricity is required");
    }
    if omm.inclination.is_nan() {
        bail!("OMM inclination is required");
    }
    if omm.arg_of_pericenter.is_nan() {
        bail!("OMM argument of pericenter is required");
    }
    if omm.raan.is_nan() {
        bail!("OMM RA of ascending node is required");
    }
    if omm.mean_anomaly.is_nan() {
        bail!("OMM mean anomaly is required");
    }

    Ok(omm)
}


/// """SGP-4 propagator for TLE
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
///     tle (TLE | list[TLE]): TLE (or list of TLES) on which to operate
///     tm (time | list[time] | npt.ArrayLike[time]): time(s) at which to compute position and velocity
///
/// Keyword Args:
///     gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is gravconst.wgs72
///     opsmode (satkit.sgp4_opsmode): opsmode.afspc (Air Force Space Command) or opsmode.improved.  Default is opsmode.afspc
///     errflag (bool): whether or not to output error conditions for each TLE and time output.  Default is False
///
/// Returns:
///     tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]: position and velocity in meters and meters/second, respectively, in the TEME frame at each of the "Ntime" input times and each of the "Ntle" tles
///
///
/// Example:
///
///
/// >>> lines = [
/// >>>        "0 INTELSAT 902",
/// >>>     "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
/// >>>     "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."
/// >>> ]
/// >>>
/// >>> tle = satkit.TLE.single_from_lines(lines)
/// >>>
/// >>> # Compute TEME position & velocity at epoch
/// >>> pteme, vteme = satkit.sgp4(tle, tle.epoch)
/// >>>
/// >>> # Rotate to ITRF frame
/// >>> q = satkit.frametransform.qteme2itrf(tm)
/// >>> pitrf = q * pteme
/// >>> vitrf = q * vteme - np.cross(np.array([0, 0, satkit.univ.omega_earth]), pitrf)
/// >>>
/// >>> # convert to ITRF coordinate object
/// >>> coord = satkit.itrfcoord.from_vector(pitrf)
/// >>>
/// >>> # Print ITRF coordinate object location
/// >>> print(coord)
/// ITRFCoord(lat:  -0.0363 deg, lon:  -2.2438 deg, hae: 35799.51 km)
#[pyfunction]
#[pyo3(signature=(tle, time, **kwds))]
pub fn sgp4(
    tle: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> Result<Py<PyAny>> {
    let mut output_err = false;
    let mut opsmode: OpsMode = OpsMode::afspc;
    let mut gravconst: GravConst = GravConst::wgs72;

    // Get keywords for the mode, gravconst, and errflag
    if let Some(kw) = kwds {
        if let Some(v) = kw.get_item("errflag")? {
            output_err = v.extract::<bool>()?;
        }
        if let Some(v) = kw.get_item("opsmode")? {
            opsmode = v.extract::<OpsMode>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid opsmode: {}", e))
            })?;
        }
        if let Some(v) = kw.get_item("gravconst")? {
            gravconst = v.extract::<GravConst>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid gravconst: {}", e))
            })?;
        }
    }

    // Handle input as TLE
    if tle.is_instance_of::<PyTLE>() {
        let mut stle: PyRefMut<PyTLE> = tle
            .extract()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE: {}", e)))?;
        let states = psgp4::sgp4_full(
            &mut stle.0,
            time.to_time_vec()?.as_slice(),
            gravconst.into(),
            opsmode.into(),
        )?;
        pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
            let dims = if states.pos.nrows() > 1 && states.pos.ncols() > 1 {
                vec![states.pos.ncols(), states.pos.nrows()]
            } else {
                vec![states.pos.len()]
            };

            // Note: this is a little confusing: ndarray uses
            // row major, nalgebra and numpy use column major,
            // hence the switch
            if !output_err {
                Ok((
                    PyArray1::from_slice(py, states.pos.data.as_slice())
                        .reshape(dims.clone())?
                        .into_py_any(py)?,
                    PyArray1::from_slice(py, states.vel.data.as_slice())
                        .reshape(dims)?
                        .into_py_any(py)?,
                )
                    .into_py_any(py)?)
            } else {
                let eint: Vec<i32> = states.errcode.iter().map(|x| *x as i32).collect();
                Ok((
                    PyArray1::from_slice(py, states.pos.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice(py, states.vel.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice(py, eint.as_slice()),
                )
                    .into_py_any(py)?)
            }
        })
    }
    // Handle input as dict
    else if tle.is_instance_of::<PyDict>() {
        let dict: &Bound<'_, PyDict> = tle.cast().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE dictionary: {}", e))
        })?;
        let mut omm = omm_from_pydict(dict)?;

        let states = psgp4::sgp4_full(
            &mut omm,
            time.to_time_vec()?.as_slice(),
            gravconst.into(),
            opsmode.into(),
        )?;
        pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
            let dims = if states.pos.nrows() > 1 && states.pos.ncols() > 1 {
                vec![states.pos.ncols(), states.pos.nrows()]
            } else {
                vec![states.pos.len()]
            };

            // Note: this is a little confusing: ndarray uses
            // row major, nalgebra and numpy use column major,
            // hence the switch
            if !output_err {
                Ok((
                    PyArray1::from_slice(py, states.pos.data.as_slice())
                        .reshape(dims.clone())?
                        .into_py_any(py)?,
                    PyArray1::from_slice(py, states.vel.data.as_slice())
                        .reshape(dims)?
                        .into_py_any(py)?,
                )
                    .into_py_any(py)?)
            } else {
                let eint: Vec<i32> = states.errcode.iter().map(|x| x.clone() as i32).collect();
                Ok((
                    PyArray1::from_slice(py, states.pos.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice(py, states.vel.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice(py, eint.as_slice()),
                )
                    .into_py_any(py)?)
            }
        })
    }
    else if tle.is_instance_of::<PyList>() {
        let plist = tle.cast::<PyList>().unwrap();
        let tmarray = time.to_time_vec()?;
        let results: Vec<psgp4::SGP4State> = plist.iter().map(|item| {
            if item.is_instance_of::<PyTLE>() {
                let mut stle: PyRefMut<PyTLE> = item
                    .extract()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE: {}", e)))?;
                psgp4::sgp4(&mut stle.0, tmarray.as_slice())
            }
            else if item.is_instance_of::<PyDict>() {
                let dict: &Bound<'_, PyDict> = item.cast().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid TLE dictionary: {}", e))
                })?;
                let mut omm = omm_from_pydict(dict)?;
                psgp4::sgp4(&mut omm, tmarray.as_slice())
            }
            else {
                bail!("Invalid TLE in list");
            }
        }).collect::<Result<Vec<_>>>()?;


        pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
            let n = plist.len() * tmarray.len() * 3;

            let parr = PyArray1::zeros(py, [n], false);
            let varr = PyArray1::zeros(py, [n], false);
            let ntimes = tmarray.len();

            // I'd prefer to create this uninitialized, which would probably be a bit faster,
            // but I can't figure out how...
            /*
            let mut earr = ndarray::Array::from_elem(
                (tles.len(), tmarray.len()),
                PySGP4Error::success.into_py(py),
            );
            */
            let mut eint = vec![0; ntimes * plist.len()];

            results.iter().enumerate().for_each(|(idx, states)| {
                unsafe {
                    let pdata: *mut f64 = parr.data();

                    std::ptr::copy_nonoverlapping(
                        states.pos.as_ptr(),
                        pdata.add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                    let vdata: *mut f64 = varr.data();
                    std::ptr::copy_nonoverlapping(
                        states.vel.as_ptr(),
                        vdata.add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                    if output_err {
                        let evals = states.errcode.iter().map(|&x| x as i32).collect::<Vec<i32>>();
                        std::ptr::copy_nonoverlapping(
                            evals.as_ptr(),
                            eint.as_mut_ptr().add(idx * ntimes),
                            ntimes,
                        )
                    }
                }

                //earr.slice_mut(ndarray::s![idx, ..]).assign(&e1);
            });

            // Set dimensions of output to remove singleton dimensions
            let dims = match (plist.len() > 1, ntimes > 1) {
                (true, true) => vec![plist.len(), ntimes, 3],
                (true, false) => vec![plist.len(), 3],
                (false, true) => vec![ntimes, 3],
                (false, false) => vec![3],
            };
            // Dims for error output

            let edims = match (plist.len() > 1, ntimes > 1) {
                (true, true) => vec![plist.len(), ntimes],
                (true, false) => vec![plist.len()],
                (false, true) => vec![ntimes],
                (false, false) => vec![1],
            };

            if !output_err {
                Ok((
                    parr.reshape(dims.clone()).unwrap(),
                    varr.reshape(dims).unwrap(),
                )
                    .into_py_any(py)?)
            } else {
                Ok((
                    parr.reshape(dims.clone()).unwrap(),
                    varr.reshape(dims).unwrap(),
                    PyArray1::from_slice(py, eint.as_slice()).reshape(edims)?,
                )
                    .into_py_any(py)?)
            }
        })
    } else {
        bail!("Invalid input type for argument 1");
    }
}
