use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::pyastrotime::ToTimeVec;
use super::pytle::PyTLE;
use crate::sgp4 as psgp4;
use numpy::PyArray1;

// Thin Python wrapper around SGP4 Error
#[allow(non_camel_case_types)]
#[pyclass(name = "sgp4error")]
#[derive(Clone)]
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
#[pyclass(name = "gravconst")]
#[derive(Clone)]
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
#[pyclass(name = "opsmode")]
#[derive(Clone)]
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

#[pyfunction]
#[pyo3(signature=(tle, time, **kwds))]
pub fn sgp4(tle: &PyAny, time: &PyAny, kwds: Option<&PyDict>) -> PyResult<PyObject> {
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
        match psgp4::sgp4_full(
            &mut stle.inner,
            time.to_time_vec()?.as_slice(),
            gravconst.into(),
            opsmode.into(),
        ) {
            Ok((r, v)) => pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                let mut dims = vec![r.len()];
                if r.nrows() > 1 && r.ncols() > 1 {
                    dims = vec![r.ncols(), r.nrows()];
                }

                // Note: this is a little confusing: ndarray uses
                // row major, nalgebra and numpy use column major,
                // hence the switch
                Ok((
                    PyArray1::from_slice(py, r.data.as_slice())
                        .reshape(dims.clone())
                        .unwrap()
                        .to_object(py),
                    PyArray1::from_slice(py, v.data.as_slice())
                        .reshape(dims)
                        .unwrap()
                        .to_object(py),
                )
                    .to_object(py))
            }),
            Err(e) => {
                if output_err == true {
                    let ep: PySGP4Error = e.0.into();
                    pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                        Ok((ep.into_py(py), e.1).to_object(py))
                    })
                } else {
                    let estr = format!("Error running sgp4: {}", e.1);
                    Err(pyo3::exceptions::PyRuntimeError::new_err(estr))
                }
            }
        }
    } else if tle.is_instance_of::<PyList>() {
        let mut tles = tle.extract::<Vec<PyRefMut<PyTLE>>>()?;
        let tmarray = time.to_time_vec()?;
        let results: Vec<psgp4::SGP4Result> = tles
            .iter_mut()
            .map(|tle| psgp4::sgp4(&mut tle.inner, tmarray.as_slice()))
            .collect();
        pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
            let n = tles.len() * tmarray.len() * 3;
            let parr: &PyArray1<f64> = PyArray1::zeros(py, [n], false);
            let varr: &PyArray1<f64> = PyArray1::zeros(py, [n], false);
            let ntimes = tmarray.len();

            results.iter().enumerate().for_each(|(idx, r)| match r {
                Ok((p, v)) => unsafe {
                    std::ptr::copy_nonoverlapping(
                        p.as_ptr(),
                        parr.data().add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                    std::ptr::copy_nonoverlapping(
                        v.as_ptr(),
                        varr.data().add(idx * ntimes * 3),
                        ntimes * 3,
                    );
                },

                Err(_e) => {}
            });
            let dims = vec![tles.len(), ntimes, 3];
            Ok((
                parr.reshape(dims.clone()).unwrap(),
                varr.reshape(dims).unwrap(),
            )
                .to_object(py))
        })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Invalid input type for argument 1",
        ))
    }
}
