use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::pyastrotime::ToTimeVec;
use super::pytle::PyTLE;
use crate::sgp4 as psgp4;
use numpy::PyArray1;
use numpy::PyArrayMethods;

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
    fn from(f: GravConst) -> psgp4::GravConst {
        match f {
            GravConst::wgs72 => psgp4::GravConst::WGS72,
            GravConst::wgs72old => psgp4::GravConst::WGS72OLD,
            GravConst::wgs84 => psgp4::GravConst::WGS84,
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
    fn from(f: OpsMode) -> psgp4::OpsMode {
        match f {
            OpsMode::afspc => psgp4::OpsMode::AFSPC,
            OpsMode::improved => psgp4::OpsMode::IMPROVED,
        }
    }
}

impl From<psgp4::SGP4Error> for PySGP4Error {
    fn from(f: psgp4::SGP4Error) -> PySGP4Error {
        match f {
            psgp4::SGP4Error::SGP4Success => PySGP4Error::success,
            psgp4::SGP4Error::SGP4ErrorEccen => PySGP4Error::eccen,
            psgp4::SGP4Error::SGP4ErrorMeanMotion => PySGP4Error::mean_motion,
            psgp4::SGP4Error::SGP4ErrorPerturbEccen => PySGP4Error::perturb_eccen,
            psgp4::SGP4Error::SGP4ErrorSemiLatusRectum => PySGP4Error::semi_latus_rectum,
            psgp4::SGP4Error::SGP4ErrorUnused => PySGP4Error::unused,
            psgp4::SGP4Error::SGP4ErrorOrbitDecay => PySGP4Error::orbit_decay,
        }
    }
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
) -> PyResult<PyObject> {
    let mut output_err = false;
    let mut opsmode: OpsMode = OpsMode::afspc;
    let mut gravconst: GravConst = GravConst::wgs72;
    if kwds.is_some() {
        let kw = kwds.unwrap();
        match kw.get_item("errflag").unwrap() {
            Some(v) => output_err = v.extract::<bool>()?,
            None => {}
        }
        match kw.get_item("opsmode").unwrap() {
            Some(v) => opsmode = v.extract::<OpsMode>()?,
            None => {}
        }
        match kw.get_item("gravconst").unwrap() {
            Some(v) => gravconst = v.extract::<GravConst>()?,
            None => {}
        }
    }
    if tle.is_instance_of::<PyTLE>() {
        let mut stle: PyRefMut<PyTLE> = tle.extract()?;
        let (r, v, e) = psgp4::sgp4_full(
            &mut stle.inner,
            time.to_time_vec()?.as_slice(),
            gravconst.into(),
            opsmode.into(),
        );
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let mut dims = vec![r.len()];
            if r.nrows() > 1 && r.ncols() > 1 {
                dims = vec![r.ncols(), r.nrows()];
            }

            // Note: this is a little confusing: ndarray uses
            // row major, nalgebra and numpy use column major,
            // hence the switch
            if output_err == false {
                Ok((
                    PyArray1::from_slice_bound(py, r.data.as_slice())
                        .reshape(dims.clone())?
                        .to_object(py),
                    PyArray1::from_slice_bound(py, v.data.as_slice())
                        .reshape(dims)?
                        .to_object(py),
                )
                    .to_object(py))
            } else {
                let eint: Vec<i32> = e.into_iter().map(|x| x as i32).collect();
                Ok((
                    PyArray1::from_slice_bound(py, r.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice_bound(py, v.data.as_slice()).reshape(dims.clone())?,
                    PyArray1::from_slice_bound(py, eint.as_slice()),
                )
                    .to_object(py))
            }
        })
    } else if tle.is_instance_of::<PyList>() {
        let mut tles = tle.extract::<Vec<PyRefMut<PyTLE>>>()?;
        let tmarray = time.to_time_vec()?;
        let results: Vec<psgp4::SGP4State> = tles
            .iter_mut()
            .map(|tle| psgp4::sgp4(&mut tle.inner, tmarray.as_slice()))
            .collect();

        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let n = tles.len() * tmarray.len() * 3;

            let parr = PyArray1::zeros_bound(py, [n], false);
            let varr = PyArray1::zeros_bound(py, [n], false);
            let ntimes = tmarray.len();

            // I'd prefer to create this uninitialized, which would probably be a bit faster,
            // but I can't figure out how...
            /*
            let mut earr = ndarray::Array::from_elem(
                (tles.len(), tmarray.len()),
                PySGP4Error::success.into_py(py),
            );
            */
            let mut eint = Vec::new();
            eint.resize(ntimes * tle.len()?, 0);
            results.iter().enumerate().for_each(|(idx, (p, v, e))| {
                unsafe {
                    let pdata: *mut f64 = parr.data();

                    std::ptr::copy_nonoverlapping(
                        p.as_ptr(),
                        pdata.add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                    let vdata: *mut f64 = varr.data();
                    std::ptr::copy_nonoverlapping(
                        v.as_ptr(),
                        vdata.add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                    eint[idx] = e[idx].clone() as i32;
                }

                //earr.slice_mut(ndarray::s![idx, ..]).assign(&e1);
            });

            // Set dimensions of output to remove singleton dimensions
            let dims = match (tles.len() > 1, ntimes > 1) {
                (true, true) => vec![tles.len(), ntimes, 3],
                (true, false) => vec![tles.len(), 3],
                (false, true) => vec![ntimes, 3],
                (false, false) => vec![3],
            };
            // Dims for error output

            let edims = match (tles.len() > 1, ntimes > 1) {
                (true, true) => vec![tles.len(), ntimes],
                (true, false) => vec![tles.len()],
                (false, true) => vec![ntimes],
                (false, false) => vec![1],
            };

            if output_err == false {
                Ok((
                    parr.reshape(dims.clone()).unwrap(),
                    varr.reshape(dims).unwrap(),
                )
                    .to_object(py))
            } else {
                Ok((
                    parr.reshape(dims.clone()).unwrap(),
                    varr.reshape(dims).unwrap(),
                    PyArray1::from_slice_bound(py, eint.as_slice()).reshape(edims)?,
                )
                    .to_object(py))
            }
        })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Invalid input type for argument 1",
        ))
    }
}
