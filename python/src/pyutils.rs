use crate::pyinstant::ToTimeVec;
use crate::pyquaternion::PyQuaternion;

use satkit::mathtypes::*;
use satkit::Instant;

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

/// Pack f64 values into little-endian bytes for `__getstate__` pickle support
pub fn pack_f64s(py: Python, vals: &[f64]) -> PyResult<Py<PyAny>> {
    let mut raw = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
}

/// Unpack little-endian `__setstate__` bytes into N f64 values
pub fn unpack_f64s<const N: usize>(
    py: Python,
    state: &Py<pyo3::types::PyBytes>,
) -> PyResult<[f64; N]> {
    let s = state.as_bytes(py);
    if s.len() != N * 8 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Invalid serialization length",
        ));
    }
    let mut out = [0.0; N];
    for (i, chunk) in s.chunks_exact(8).enumerate() {
        out[i] = f64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

/// Raise `ValueError` listing any keyword arguments that remain unconsumed
/// after all expected keywords have been extracted (and deleted) from `kw`
pub fn reject_unused_kwargs(kw: &Bound<'_, PyDict>) -> PyResult<()> {
    if kw.is_empty() {
        return Ok(());
    }
    let keys: Vec<String> = kw.iter().map(|(k, _v)| k.to_string()).collect();
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "Invalid keyword arguments: {}",
        keys.join(", ")
    )))
}

pub fn py_vec3_of_time_arr(
    cfunc: &(dyn Fn(&Instant) -> Vector3 + Sync),
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    let py = tmarr.py();
    match tm.len() {
        1 => {
            let v: Vector3 = cfunc(&tm[0]);
            Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?)
        }
        n => {
            // Release the GIL for the computation over the full time array
            let vals: Vec<f64> = py.detach(|| {
                let mut vals = Vec::with_capacity(n * 3);
                for time in tm.iter() {
                    vals.extend_from_slice(cfunc(time).as_slice());
                }
                vals
            });
            Ok(np::PyArray1::from_vec(py, vals)
                .reshape([n, 3])?
                .into_py_any(py)?)
        }
    }
}

pub fn py_vec3_of_time_result_arr(
    cfunc: &(dyn Fn(&Instant) -> Result<Vector3> + Sync),
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    let py = tmarr.py();
    match tm.len() {
        1 => match cfunc(&tm[0]) {
            Ok(v) => Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?),
            Err(_) => bail!("Invalid time"),
        },
        n => {
            // Release the GIL for the computation over the full time array
            let vals: Result<Vec<f64>> = py.detach(|| {
                let mut vals = Vec::with_capacity(n * 3);
                for time in tm.iter() {
                    match cfunc(time) {
                        Ok(v) => vals.extend_from_slice(v.as_slice()),
                        Err(_) => bail!("Invalid time"),
                    }
                }
                Ok(vals)
            });
            Ok(np::PyArray1::from_vec(py, vals?)
                .reshape([n, 3])?
                .into_py_any(py)?)
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
            m.as_mut_slice().copy_from_slice(arr.as_slice()?);
        } else {
            let arr = arr.as_array();
            for row in 0..M {
                m[(row, 0)] = arr[row];
            }
        }
    } else if obj.is_instance_of::<np::PyArray2<f64>>() {
        let arr = obj.extract::<np::PyReadonlyArray2<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid array shape: {}", e))
        })?;
        // Element-by-element to handle numpy row-major to numeris column-major
        let arr = arr.as_array();
        for row in 0..M {
            for col in 0..N {
                m[(row, col)] = arr[(row, col)];
            }
        }
    } else {
        // Fallback: try to extract as a flat sequence of floats (lists, tuples, etc.)
        let vals: Vec<f64> = obj.extract().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot convert to {M}x{N} matrix: {e}"
            ))
        })?;
        if vals.len() != M * N {
            anyhow::bail!(
                "Expected {} elements for {M}x{N} matrix, got {}",
                M * N,
                vals.len()
            );
        }
        m.as_mut_slice().copy_from_slice(&vals);
    }
    Ok(m)
}

pub fn py_func_of_time_arr<'a, T: IntoPyObject<'a> + Send>(
    cfunc: fn(&Instant) -> T,
    tmarr: &Bound<'a, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    let py = tmarr.py();

    match tm.len() {
        1 => Ok(cfunc(&tm[0]).into_py_any(py)?),
        _ => {
            // Release the GIL for the computation over the full time array
            let tvec: Vec<T> = py.detach(|| tm.iter().map(cfunc).collect());
            Ok(tvec.into_py_any(py)?)
        }
    }
}

#[inline]
pub fn py_quat_from_time_arr(
    cfunc: fn(&Instant) -> Quaternion,
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let tm = tmarr.to_time_vec()?;
    let py = tmarr.py();
    match tm.len() {
        1 => Ok(PyQuaternion(cfunc(&tm[0])).into_py_any(py)?),
        _ => {
            // Release the GIL for the computation over the full time array
            let quats: Vec<PyQuaternion> =
                py.detach(|| tm.iter().map(|x| PyQuaternion(cfunc(x))).collect());
            Ok(quats.into_py_any(py)?)
        }
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
        std::ptr::copy_nonoverlapping(
            m.as_slice().as_ptr(),
            p.as_raw_array_mut().as_mut_ptr(),
            M * N,
        );
    }
    p.into_py_any(py).unwrap()
}

#[inline]
pub fn tuple_func_of_time_arr<F>(cfunc: F, tmarr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>
where
    F: Fn(&Instant) -> Result<(Vector3, Vector3)> + Sync,
{
    let tm = tmarr.to_time_vec()?;
    let py = tmarr.py();
    match tm.len() {
        1 => match cfunc(&tm[0]) {
            Ok(r) => (
                PyArray1::from_slice(py, r.0.as_slice()),
                PyArray1::from_slice(py, r.1.as_slice()),
            )
                .into_py_any(py),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        },
        _ => {
            // Release the GIL for the computation over the full time array
            let arrs = py.detach(|| -> PyResult<_> {
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
                        Err(e) => {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
                        }
                    }
                }
                Ok((pout, vout))
            })?;
            (
                PyArray2::from_array(py, &arrs.0),
                PyArray2::from_array(py, &arrs.1),
            )
                .into_py_any(py)
        }
    }
}

#[allow(dead_code)]
/// Extract a single `satkit::Instant` from a Python object.
///
/// Accepts `satkit.time` (PyInstant) or `datetime.datetime` (interpreted as UTC).
/// Returns a `PyTypeError` if the object is neither.
pub fn instant_from_pyany(obj: &Bound<'_, PyAny>) -> PyResult<Instant> {
    let v = obj.to_time_vec()?;
    if v.len() != 1 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a single time value (satkit.time or datetime.datetime)",
        ));
    }
    Ok(v[0])
}
