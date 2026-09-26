use pyo3::prelude::*;

use satkit::earthgravity::{accel, accel_and_partials, GravityModel, MAX_GRAVITY_DEGREE};

use crate::pyitrfcoord::PyITRFCoord;
use numpy as np;
use satkit::mathtypes::*;

use crate::pyutils::{slice2py2d, vec2py};
use pyo3::types::PyDict;

use anyhow::{bail, Result};

/// Load the model's coefficients (a download, for itugrace16) before the
/// evaluator's infallible `get()` would panic on a missing file.
fn ensure_loaded(model: &GravModel) -> Result<()> {
    satkit::earthgravity::ensure_loaded(model.clone().into())
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Parse the arguments shared by `gravity` and `gravity_and_partials`: the
/// `model` / `degree` / `order` keywords (loading the model's coefficients)
/// and the ITRF position, an `itrfcoord` or a 3-element numpy array
fn gravity_args(
    fname: &str,
    pos: &Bound<'_, PyAny>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> Result<(Vector3, usize, usize, GravityModel)> {
    let mut degree: usize = 6;
    let mut order: Option<usize> = None;
    let mut model: GravModel = GravModel::egm2008;
    if let Some(kw) = kwds {
        crate::pyutils::reject_unknown_kwargs(fname, kw, &["model", "degree", "order"])?;
        if let Some(v) = kw.get_item("model")? {
            model = v
                .extract::<GravModel>()
                .map_err(|e| anyhow::anyhow!("Failed to extract gravity model: {}", e))?;
        }
        if let Some(v) = kw.get_item("degree")? {
            degree = v
                .extract::<usize>()
                .map_err(|e| anyhow::anyhow!("Failed to extract degree: {}", e))?;
        }
        if let Some(v) = kw.get_item("order")? {
            order = Some(
                v.extract::<usize>()
                    .map_err(|e| anyhow::anyhow!("Failed to extract order: {}", e))?,
            );
        }
    }
    let order = order.unwrap_or(degree);
    if degree > MAX_GRAVITY_DEGREE as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gravity degree {degree} exceeds the maximum supported value ({MAX_GRAVITY_DEGREE})"
        ))
        .into());
    }
    ensure_loaded(&model)?;

    let v: Vector3 = if pos.is_instance_of::<PyITRFCoord>() {
        let pyitrf: PyRef<PyITRFCoord> = pos
            .extract()
            .map_err(|e| anyhow::anyhow!("Failed to extract itrfcoord: {}", e))?;
        pyitrf.0.itrf
    } else if pos.is_instance_of::<np::PyArray1<f64>>() {
        let vpy = pos
            .extract::<np::PyReadonlyArray1<f64>>()
            .map_err(|e| anyhow::anyhow!("Failed to extract position array: {}", e))?;
        let varr = vpy.as_array();
        if varr.len() != 3 {
            bail!("Input must have 3 elements");
        }
        Vector3::from_slice(&[varr[0], varr[1], varr[2]])
    } else {
        bail!("Input must be 3-element numpy or itrfcoord");
    };
    Ok((v, degree, order, model.into()))
}

///
/// Gravity model enumeration
///
/// For details of models, see:
/// http://icgem.gfz-potsdam.de/tom_longtime
///
/// egm96, egm2008, jgm2 and jgm3 are compiled into satkit; itugrace16 is
/// downloaded on first use (CC BY 4.0). Each model's tide system
/// (tide-free / zero-tide) is read on load and the propagator's solid-tide
/// correction accounts for it.
///
#[allow(non_camel_case_types)]
#[pyclass(name = "gravmodel", eq, eq_int, from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum GravModel {
    jgm3 = GravityModel::JGM3 as isize,
    jgm2 = GravityModel::JGM2 as isize,
    egm96 = GravityModel::EGM96 as isize,
    itugrace16 = GravityModel::ITUGrace16 as isize,
    egm2008 = GravityModel::EGM2008 as isize,
}

crate::enum_pickle!(GravModel, "gravmodel");

impl From<GravModel> for GravityModel {
    fn from(g: GravModel) -> Self {
        match g {
            GravModel::jgm3 => Self::JGM3,
            GravModel::jgm2 => Self::JGM2,
            GravModel::egm96 => Self::EGM96,
            GravModel::itugrace16 => Self::ITUGrace16,
            GravModel::egm2008 => Self::EGM2008,
        }
    }
}

impl From<GravityModel> for GravModel {
    fn from(g: GravityModel) -> Self {
        match g {
            GravityModel::JGM3 => Self::jgm3,
            GravityModel::JGM2 => Self::jgm2,
            GravityModel::EGM96 => Self::egm96,
            GravityModel::ITUGrace16 => Self::itugrace16,
            GravityModel::EGM2008 => Self::egm2008,
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
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.egm2008
///     degree (int): maximum degree of gravity model to use.  Default is 6, maximum is 70
///     order (int): maximum order of gravity model to use.  Default is same as degree
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.
#[pyfunction]
// The keywords are parsed by hand from `kwds`; `text_signature` publishes
// them so `inspect.signature` and stubtest see the real parameters.
#[pyo3(
    signature=(pos, **kwds),
    text_signature = "(pos, *, model=..., degree=6, order=...)"
)]
pub fn gravity(
    py: Python,
    pos: &Bound<'_, PyAny>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> Result<Py<PyAny>> {
    let (v, degree, order, model) = gravity_args("gravity", pos, kwds)?;
    Ok(vec2py(py, &accel(&v, degree, order, model))?)
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
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.egm2008
///     degree (int): maximum degree of gravity model to use.  Default is 6, maximum is 70
///     order (int): maximum order of gravity model to use.  Default is same as degree
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.
///
#[pyfunction]
// The keywords are parsed by hand from `kwds`; `text_signature` publishes
// them so `inspect.signature` and stubtest see the real parameters.
#[pyo3(
    signature=(pos, **kwds),
    text_signature = "(pos, *, model=..., degree=6, order=...)"
)]
pub fn gravity_and_partials(
    py: Python,
    pos: &Bound<'_, PyAny>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let (v, degree, order, model) = gravity_args("gravity_and_partials", pos, kwds)?;
    let (g, p) = accel_and_partials(&v, degree, order, model);
    // The partials' column-major storage read as a row-major (C-order) array,
    // as before: the array is `p` transposed (the matrix is symmetric up to
    // rounding, so this only affects the last bits)
    Ok((vec2py(py, &g)?, slice2py2d(py, p.as_slice(), 3, 3)?))
}
