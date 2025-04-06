use pyo3::prelude::*;

use crate::earthgravity::{accel, accel_and_partials, GravityModel};

use super::pyitrfcoord::PyITRFCoord;
use crate::itrfcoord::ITRFCoord;
use nalgebra as na;
use numpy as np;
use numpy::PyArrayMethods;

use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

///
/// Gravity model enumeration
///
/// For details of models, see:
/// http://icgem.gfz-potsdam.de/tom_longtime
///
#[allow(non_camel_case_types)]
#[pyclass(name = "gravmodel", eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum GravModel {
    jgm3 = GravityModel::JGM3 as isize,
    jgm2 = GravityModel::JGM2 as isize,
    egm96 = GravityModel::EGM96 as isize,
    itugrace16 = GravityModel::ITUGrace16 as isize,
}

impl From<GravModel> for GravityModel {
    fn from(g: GravModel) -> Self {
        match g {
            GravModel::jgm3 => Self::JGM3,
            GravModel::jgm2 => Self::JGM2,
            GravModel::egm96 => Self::EGM96,
            GravModel::itugrace16 => Self::ITUGrace16,
        }
    }
}

/// Acceleration vector due to Earth gravity
///
///
/// Args:
///     pos (numpy.ndarray|satkit.itrfcoord): position at which to compute acceleration.  itrfcoord or 3-element numpy array with Cartesian ITRF position in meters
///
/// Returns:
///     numpy.ndarray: 3-element numpy array representing acceleration due to Earth gravity at input position.  Units are m/s^2
///
/// Keyword Args:
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.jgm3
///     order (int): order of gravity model to use.  Default is 6, maximum is 16
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.
#[pyfunction]
#[pyo3(signature=(pos, **kwds))]
pub fn gravity(pos: &Bound<'_, PyAny>, kwds: Option<&Bound<'_, PyDict>>) -> Result<PyObject> {
    let mut order: usize = 6;
    let mut model: GravModel = GravModel::jgm3;
    if let Some(kw) = kwds {
        if let Some(v) = kw.get_item("model")? {
            model = v.extract::<GravModel>()?;
        }
        if let Some(v) = kw.get_item("order")? {
            order = v.extract::<usize>()?;
        }
    }

    if pos.is_instance_of::<PyITRFCoord>() {
        let pyitrf: PyRef<PyITRFCoord> = pos.extract()?;
        let itrf: ITRFCoord = pyitrf.0;
        let v = accel(&itrf.itrf, order, model.into());
        pyo3::Python::with_gil(|py| -> Result<PyObject> {
            let vpy = np::PyArray1::<f64>::from_slice(py, v.as_slice());
            Ok(vpy.into_py_any(py)?)
        })
    } else if pos.is_instance_of::<np::PyArray1<f64>>() {
        let vpy = pos.extract::<np::PyReadonlyArray1<f64>>().unwrap();
        if vpy.len().unwrap() != 3 {
            bail!("Input must have 3 elements");
        }
        let v: na::Vector3<f64> = na::Vector3::<f64>::from_row_slice(vpy.as_slice().unwrap());
        let a = accel(&v, order, model.into());
        pyo3::Python::with_gil(|py| -> Result<PyObject> {
            let vpy = np::PyArray1::<f64>::from_slice(py, a.as_slice());
            Ok(vpy.into_py_any(py)?)
        })
    } else {
        bail!("Input must be 3-element numpy or itrfcoord");
    }
}

/// Acceleration vector due to Earth gravity and partials with respect to position
///
///
/// Args:
///     pos (numpy.ndarray|satkit.itrfcoord): position at which to compute acceleration.  itrfcoord or 3-element numpy array with Cartesian ITRF position in meters
///
/// Returns:
///     (numpy.ndarray, numpy.ndarray): tuple of 3-element numpy array representing acceleration due to Earth gravity at input position and 3x3 numpy array of partials of acceleration with respect to position.  Units are m/s^2 for gravity and m/s^2/m for partials
///
/// Keyword Args:
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.jgm3
///     order (int): order of gravity model to use.  Default is 6, maximum is 16
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.
///
#[pyfunction]
#[pyo3(signature=(pos, **kwds))]
pub fn gravity_and_partials(
    pos: &Bound<'_, PyAny>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> Result<(PyObject, PyObject)> {
    let mut order: usize = 6;
    let mut model: GravModel = GravModel::jgm3;
    if let Some(kw) = kwds {
        if let Some(v) = kw.get_item("model")? {
            model = v.extract::<GravModel>()?;
        }
        if let Some(v) = kw.get_item("order")? {
            order = v.extract::<usize>()?;
        }
    }

    if pos.is_instance_of::<PyITRFCoord>() {
        let pyitrf: PyRef<PyITRFCoord> = pos.extract()?;
        let itrf: ITRFCoord = pyitrf.0;
        let (g, p) = accel_and_partials(&itrf.itrf, order, model.into());
        pyo3::Python::with_gil(|py| -> Result<(PyObject, PyObject)> {
            let gpy = np::PyArray1::<f64>::from_slice(py, g.as_slice());
            let ppy = unsafe { np::PyArray2::<f64>::new(py, [3, 3], false) };
            unsafe {
                std::ptr::copy_nonoverlapping(p.as_ptr(), ppy.as_raw_array_mut().as_mut_ptr(), 9);
            }
            Ok((gpy.into_py_any(py)?, ppy.into_py_any(py)?))
        })
    } else if pos.is_instance_of::<np::PyArray1<f64>>() {
        let vpy = pos.extract::<np::PyReadonlyArray1<f64>>().unwrap();
        if vpy.len().unwrap() != 3 {
            bail!("Input must have 3 elements");
        }
        let v: na::Vector3<f64> = na::Vector3::<f64>::from_row_slice(vpy.as_slice().unwrap());
        let (g, p) = accel_and_partials(&v, order, model.into());
        pyo3::Python::with_gil(|py| -> Result<(PyObject, PyObject)> {
            let gpy = np::PyArray1::<f64>::from_slice(py, g.as_slice());
            let ppy = unsafe { np::PyArray2::<f64>::new(py, [3, 3], false) };
            unsafe {
                std::ptr::copy_nonoverlapping(p.as_ptr(), ppy.as_raw_array_mut().as_mut_ptr(), 9);
            }
            Ok((gpy.into_py_any(py)?, ppy.into_py_any(py)?))
        })
    } else {
        bail!("Input must be 3-element numpy or itrfcoord");
    }
}
