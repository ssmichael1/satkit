use super::pyinstant::ToTimeVec;
use super::pyquaternion::PyQuaternion;

use crate::mathtypes::*;
use crate::Instant;

use nalgebra as na;
use numpy as np;
use numpy::ndarray;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObject;
use pyo3::IntoPyObjectExt;

use anyhow::{bail, Result};

pub fn kwargs_or_default<'py, T>(
    kwargs: &mut Option<&Bound<'py, PyDict>>,
    name: &str,
    default: T,
) -> PyResult<T>
where
    T: FromPyObjectOwned<'py>,
{
    if let Some(kw) = kwargs {
        match kw.get_item(name)? {
            None => Ok(default),
            Some(v) => {
                kw.del_item(name)?;
                let value = v.extract::<T>().map_err(|_e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid value for {}", name))
                })?;
                Ok(value)
            }
        }
    } else {
        Ok(default)
    }
}

pub fn kwargs_or_none<'py, T>(
    kwargs: &mut Option<&Bound<'py, PyDict>>,
    name: &str,
) -> PyResult<Option<T>>
where
    T: FromPyObjectOwned<'py>,
{
    if let Some(kw) = kwargs {
        match kw.get_item(name)? {
            None => Ok(None),
            Some(v) => {
                kw.del_item(name)?;
                Ok(Some(v.extract::<T>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid value for {}", name))
                })?))
            }
        }
    } else {
        Ok(None)
    }
}

pub fn py_vec3_of_time_arr(
    cfunc: &dyn Fn(&Instant) -> Vector3,
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    match tm.len() {
        1 => {
            let v: Vector3 = cfunc(&tm[0]);
            pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
                Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?)
            })
        }
        _ => {
            let n = tm.len();
            pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
                let out = np::PyArray2::<f64>::zeros(py, (n, 3), false);
                for (idx, time) in tm.iter().enumerate() {
                    let v: Vector3 = cfunc(time);
                    // I cannot figure out how to do this with a "safe" function,
                    // but... careful checking of dimensions above so this should
                    // never fail
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            v.as_ptr(),
                            out.as_raw_array_mut().as_mut_ptr().offset(idx as isize * 3),
                            3,
                        );
                    }
                }
                Ok(out.into_py_any(py)?)
            })
        }
    }
}

pub fn py_vec3_of_time_result_arr(
    cfunc: &dyn Fn(&Instant) -> Result<Vector3>,
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;

    match tm.len() {
        1 => match cfunc(&tm[0]) {
            Ok(v) => pyo3::Python::attach(|py| {
                Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?)
            }),
            Err(_) => bail!("Invalid time"),
        },
        _ => {
            let n = tm.len();
            pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
                let out = np::PyArray2::<f64>::zeros(py, (n, 3), false);
                for (idx, time) in tm.iter().enumerate() {
                    match cfunc(time) {
                        Ok(v) => {
                            // I cannot figure out how to do this with a "safe" function,
                            // but... careful checking of dimensions above so this should
                            // never fail
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    v.as_ptr(),
                                    out.as_raw_array_mut().as_mut_ptr().offset(idx as isize * 3),
                                    3,
                                );
                            }
                        }
                        Err(_) => {
                            bail!("Invalid time");
                        }
                    }
                }
                Ok(out.into_py_any(py)?)
            })
        }
    }
}

#[allow(dead_code)]
pub fn smatrix_to_py<const M: usize, const N: usize>(m: &Matrix<M, N>) -> Result<Py<PyAny>> {
    if N == 1 {
        pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
            Ok(PyArray1::from_slice(py, m.as_slice()).into_py_any(py)?)
        })
    } else {
        pyo3::Python::attach(|py| -> Result<Py<PyAny>> {
            Ok(PyArray1::from_slice(py, m.as_slice())
                .reshape([M, N])?
                .into_py_any(py)?)
        })
    }
}

/// Convert python object to fixed-size matrix
pub fn py_to_smatrix<const M: usize, const N: usize>(obj: &Bound<PyAny>) -> Result<Matrix<M, N>> {
    let mut m: Matrix<M, N> = Matrix::<M, N>::zeros();
    if obj.is_instance_of::<np::PyArray1<f64>>() {
        let arr = obj.extract::<np::PyReadonlyArray1<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid array shape: {}", e))
        })?;
        if arr.is_contiguous() {
            m.copy_from_slice(arr.as_slice()?);
        } else {
            let arr = arr.as_array();
            for row in 0..M {
                m[row] = arr[row];
            }
        }
    } else if obj.is_instance_of::<np::PyArray2<f64>>() {
        let arr = obj.extract::<np::PyReadonlyArray2<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid array shape: {}", e))
        })?;
        if arr.is_contiguous() {
            m.copy_from_slice(arr.as_slice()?);
        } else {
            let arr = arr.as_array();
            for row in 0..M {
                for col in 0..N {
                    m[(row, col)] = arr[(row, col)];
                }
            }
        }
    }
    Ok(m)
}

pub fn py_func_of_time_arr<'a, T: IntoPyObject<'a>>(
    cfunc: fn(&Instant) -> T,
    tmarr: &Bound<'a, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;

    match tm.len() {
        1 => Ok(cfunc(&tm[0]).into_py_any(tmarr.py())?),
        _ => {
            let tvec: Vec<T> = tm.iter().map(cfunc).collect();
            Ok(tvec.into_py_any(tmarr.py())?)
        }
    }
}

#[inline]
pub fn py_quat_from_time_arr(
    cfunc: fn(&Instant) -> Quaternion,
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    match tm.len() {
        1 => Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
            PyQuaternion(cfunc(&tm[0])).into_py_any(py)
        })?),
        _ => Ok(pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
            tm.iter()
                .map(|x| PyQuaternion(cfunc(x)))
                .collect::<Vec<PyQuaternion>>()
                .into_py_any(py)
        })?),
    }
}

#[inline]
pub fn vec2py<const T: usize>(py: Python, v: &Vector<T>) -> PyResult<Py<PyAny>> {
    PyArray1::from_slice(py, v.as_slice()).into_py_any(py)
}

pub fn slice2py1d(py: Python, s: &[f64]) -> PyResult<Py<PyAny>> {
    PyArray1::from_slice(py, s).into_py_any(py)
}

pub fn slice2py2d(py: Python, s: &[f64], rows: usize, cols: usize) -> PyResult<Py<PyAny>> {
    let arr = PyArray1::from_slice(py, s);
    match arr.reshape([rows, cols]) {
        Ok(a) => a.into_py_any(py),
        Err(e) => Err(e),
    }
}

#[allow(dead_code)]
pub fn mat2py<const M: usize, const N: usize>(py: Python, m: &Matrix<M, N>) -> Py<PyAny> {
    let p = unsafe { PyArray2::<f64>::new(py, [M, N], true) };
    unsafe {
        std::ptr::copy_nonoverlapping(m.as_ptr(), p.as_raw_array_mut().as_mut_ptr(), M * N);
    }
    p.into_py_any(py).unwrap()
}

#[inline]
pub fn tuple_func_of_time_arr<F>(cfunc: F, tmarr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>
where
    F: Fn(&Instant) -> Result<(na::Vector3<f64>, na::Vector3<f64>)>,
{
    let tm = tmarr.to_time_vec()?;
    match tm.len() {
        1 => match cfunc(&tm[0]) {
            Ok(r) => pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                (
                    PyArray1::from_slice(py, r.0.as_slice()),
                    PyArray1::from_slice(py, r.1.as_slice()),
                )
                    .into_py_any(py)
            }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        },
        _ => {
            let mut pout = ndarray::Array2::<f64>::zeros([tm.len(), 3]);
            let mut vout = ndarray::Array2::<f64>::zeros([tm.len(), 3]);

            for (i, tm) in tm.iter().enumerate() {
                match cfunc(tm) {
                    Ok(r) => {
                        pout.row_mut(i)
                            .assign(&ndarray::Array1::from_vec(vec![r.0[0], r.0[1], r.0[2]]));
                        vout.row_mut(i)
                            .assign(&ndarray::Array1::from_vec(vec![r.1[0], r.1[1], r.1[2]]));
                    }
                    Err(e) => return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
                }
            }
            pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                (
                    PyArray2::from_array(py, &pout),
                    PyArray2::from_array(py, &vout),
                )
                    .into_py_any(py)
            })
        }
    }
}
