use crate::pyinstant::{TimeInput, ToTimeVec};
use crate::pyquaternion::PyQuaternion;

use satkit::mathtypes::*;
use satkit::Instant;

use numpy as np;
use numpy::ndarray;

use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObject;
use pyo3::IntoPyObjectExt;

use anyhow::Result;

/// Emit a `DeprecationWarning` attributed to the Python line that called the
/// deprecated method.
///
/// A PyO3 method has no Python frame of its own, so from here stack level 1
/// is already the caller (level 2 would blame the caller's caller).
pub fn warn_deprecated(py: Python<'_>, msg: &std::ffi::CStr) -> PyResult<()> {
    let warning_type = py.get_type::<pyo3::exceptions::PyDeprecationWarning>();
    PyErr::warn(py, warning_type.as_any(), msg, 1)
}

/// Convert any real numeric array-like (a numpy array of any integer or
/// floating dtype, a numpy scalar, or a nested list / tuple of numbers) to a
/// float64 numpy array, as `numpy.asarray(obj, dtype=float)` would.
///
/// A float64 array is passed through without a copy. Anything that is not
/// real-numeric (complex, bool, string or object dtype) raises `TypeError`,
/// so, for example, the imaginary part of a complex array is never silently
/// dropped.
pub fn to_f64_ndarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, np::PyArrayDyn<f64>>> {
    if let Ok(a) = obj.cast::<np::PyArrayDyn<f64>>() {
        return Ok(a.clone());
    }
    let numpy = obj.py().import("numpy")?;
    let arr = numpy.call_method1("asarray", (obj,)).map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a real numeric array-like, got {}: {e}",
            obj.get_type()
        ))
    })?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if !matches!(kind.as_str(), "i" | "u" | "f") {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a real numeric array-like, got {} with dtype {}",
            obj.get_type(),
            arr.getattr("dtype")?
        )));
    }
    Ok(arr
        .call_method1("astype", (numpy.getattr("float64")?,))?
        .cast_into::<np::PyArrayDyn<f64>>()?)
}

/// A real 3-vector from any numeric array-like of length 3 (see
/// [`to_f64_ndarray`]); `what` names the argument in the error message.
pub fn to_vector3(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vector3> {
    let arr = to_f64_ndarray(obj)?;
    let ro = arr.readonly();
    let a = ro.as_array();
    if a.ndim() != 1 || a.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be a 3-element vector, got shape {:?}",
            a.shape()
        )));
    }
    Ok(numeris::vector![a[0], a[1], a[2]])
}

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
            // An explicit `name=None` means the same as leaving it out
            Some(v) if v.is_none() => {
                kw.del_item(name)?;
                Ok(None)
            }
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

/// Encode a maneuver / continuous-thrust coordinate frame as a single byte for
/// pickling. Only the frames that are valid for a maneuver or thrust are
/// representable; any other frame is a hard error rather than being silently
/// coerced to GCRF (the previous `_ => 0` behavior corrupted NTW/LVLH burns).
pub fn maneuver_frame_to_u8(frame: satkit::Frame) -> PyResult<u8> {
    use satkit::Frame;
    match frame {
        Frame::GCRF => Ok(0),
        Frame::RTN => Ok(1),
        Frame::NTW => Ok(2),
        Frame::LVLH => Ok(3),
        Frame::ITRF | Frame::TIRS | Frame::CIRS | Frame::TEME | Frame::EME2000 | Frame::ICRF => {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cannot serialize a maneuver/thrust in frame {frame}; \
             must be GCRF, RTN, NTW, or LVLH"
            )))
        }
    }
}

/// Decode a maneuver / continuous-thrust frame byte written by
/// [`maneuver_frame_to_u8`].
pub fn maneuver_frame_from_u8(tag: u8) -> PyResult<satkit::Frame> {
    use satkit::Frame;
    match tag {
        0 => Ok(Frame::GCRF),
        1 => Ok(Frame::RTN),
        2 => Ok(Frame::NTW),
        3 => Ok(Frame::LVLH),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid frame tag {other} in pickled state"
        ))),
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
    for (i, chunk) in s.as_chunks::<8>().0.iter().enumerate() {
        out[i] = f64::from_le_bytes(*chunk);
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

/// Pickle support shared by the enum classes (`frame`, `timescale`, ...):
/// `__reduce__` returns `(satkit.satkit._enum_member, (path, name))`, so
/// unpickling looks the member up by name. `path` is the class's attribute
/// path from the `satkit.satkit` extension module (`"frame"`,
/// `"moon.moonphase"`), which works for classes that live in a native
/// submodule too (those are not importable, so pickle could not find the
/// class itself by reference).
pub fn enum_reduce<'py>(
    slf: &Bound<'py, PyAny>,
    path: &'static str,
) -> PyResult<(Bound<'py, PyAny>, (&'static str, String))> {
    let cls = slf.get_type();
    for attr in cls.dir()?.iter() {
        let name: String = attr.extract()?;
        if name.starts_with('_') {
            continue;
        }
        let v = cls.getattr(name.as_str())?;
        if v.get_type().is(&cls) && v.eq(slf)? {
            let f = slf.py().import("satkit.satkit")?.getattr("_enum_member")?;
            return Ok((f, (path, name)));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "cannot pickle {}: member not found on its class",
        slf.repr()?
    )))
}

/// Unpickling half of [`enum_reduce`]: `satkit.satkit.<path>.<name>`
#[pyfunction]
#[pyo3(name = "_enum_member")]
pub fn enum_member<'py>(py: Python<'py>, path: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let mut obj = py.import("satkit.satkit")?.into_any();
    for part in path.split('.') {
        obj = obj.getattr(part)?;
    }
    obj.getattr(name)
}

/// `__reduce__` for an enum pyclass that has no other `#[pymethods]` block
/// (see [`enum_reduce`]).
#[macro_export]
macro_rules! enum_pickle {
    ($ty:ty, $path:literal) => {
        #[pyo3::pymethods]
        impl $ty {
            fn __reduce__<'py>(
                slf: &pyo3::Bound<'py, Self>,
            ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, (&'static str, String))> {
                $crate::pyutils::enum_reduce(slf.as_any(), $path)
            }
        }
    };
}

/// Raise `TypeError` (Python's own convention for a bad keyword) if `kw`
/// holds any key not in `allowed`, e.g. a misspelt `degre=`. Unlike
/// [`reject_unused_kwargs`] it does not need the keywords to be consumed.
pub fn reject_unknown_kwargs(
    fname: &str,
    kw: &Bound<'_, PyDict>,
    allowed: &[&str],
) -> PyResult<()> {
    let unknown: Vec<String> = kw
        .keys()
        .iter()
        .map(|k| k.to_string())
        .filter(|k| !allowed.contains(&k.as_str()))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{fname}() got unexpected keyword argument{} {} (accepted: {})",
        if unknown.len() == 1 { "" } else { "s" },
        unknown
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", "),
        allowed.join(", ")
    )))
}

pub fn py_vec3_of_time_arr(
    cfunc: &(dyn Fn(&Instant) -> Vector3 + Sync),
    tmarr: &Bound<'_, PyAny>,
) -> Result<Py<PyAny>> {
    let TimeInput { times: tm, scalar } = tmarr.to_time_input()?;
    let py = tmarr.py();
    match (scalar, tm.len()) {
        (true, _) => {
            let v: Vector3 = cfunc(&tm[0]);
            Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?)
        }
        (false, n) => {
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
    let TimeInput { times: tm, scalar } = tmarr.to_time_input()?;
    let py = tmarr.py();
    match (scalar, tm.len()) {
        (true, _) => {
            let v = cfunc(&tm[0])?;
            Ok(np::PyArray1::from_slice(py, v.as_slice()).into_py_any(py)?)
        }
        (false, n) => {
            // Release the GIL for the computation over the full time array
            let vals: Result<Vec<f64>> = py.detach(|| {
                let mut vals = Vec::with_capacity(n * 3);
                for time in tm.iter() {
                    vals.extend_from_slice(cfunc(time)?.as_slice());
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
        let arr = arr.as_array();
        if arr.len() != M * N {
            anyhow::bail!(
                "Expected {} elements for {M}x{N} matrix, got {}",
                M * N,
                arr.len()
            );
        }
        // Flat copy handles both contiguous and strided input
        for (dst, src) in m.as_mut_slice().iter_mut().zip(arr.iter()) {
            *dst = *src;
        }
    } else if obj.is_instance_of::<np::PyArray2<f64>>() {
        let arr = obj.extract::<np::PyReadonlyArray2<f64>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid array shape: {}", e))
        })?;
        let arr = arr.as_array();
        if arr.shape() != [M, N] {
            anyhow::bail!(
                "Expected {M}x{N} matrix, got {}x{}",
                arr.shape()[0],
                arr.shape()[1]
            );
        }
        // Element-by-element to handle numpy row-major to numeris column-major
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
    let TimeInput { times: tm, scalar } = tmarr.to_time_input()?;
    let py = tmarr.py();

    match scalar {
        true => Ok(cfunc(&tm[0]).into_py_any(py)?),
        false => {
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
    let TimeInput { times: tm, scalar } = tmarr.to_time_input()?;
    let py = tmarr.py();
    match scalar {
        true => Ok(PyQuaternion(cfunc(&tm[0])).into_py_any(py)?),
        false => {
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
    let TimeInput { times: tm, scalar } = tmarr.to_time_input()?;
    let py = tmarr.py();
    match scalar {
        true => match cfunc(&tm[0]) {
            Ok(r) => (
                PyArray1::from_slice(py, r.0.as_slice()),
                PyArray1::from_slice(py, r.1.as_slice()),
            )
                .into_py_any(py),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        },
        false => {
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
/// Accepts `satkit.time` (PyInstant) or `datetime.datetime` (a naive datetime
/// is local time, an aware one uses its own offset; see `datetime_to_instant`).
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
