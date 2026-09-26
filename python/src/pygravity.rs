use pyo3::prelude::*;

use satkit::earthgravity::{accel, accel_and_partials, GravityModel, MAX_GRAVITY_DEGREE};

use crate::pyitrfcoord::PyITRFCoord;
use satkit::mathtypes::*;

use crate::pyutils::{slice2py2d, to_vector3, vec2py};

use anyhow::Result;

/// Load the model's coefficients (a download, for itugrace16) before the
/// evaluator's infallible `get()` would panic on a missing file.
fn ensure_loaded(model: &GravModel) -> Result<()> {
    satkit::earthgravity::ensure_loaded(model.clone().into())
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

crate::arg_extractor!(model_arg: GravModel, |e| {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract gravity model: {e}"))
});
crate::arg_extractor!(degree_arg: usize, |e| {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract degree: {e}"))
});
crate::arg_extractor!(order_arg: Option<usize>, |e| {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract order: {e}"))
});

/// Check the arguments shared by `gravity` and `gravity_and_partials`
/// (loading the model's coefficients) and read the ITRF position, an
/// `itrfcoord` or any real numeric 3-element array-like
fn gravity_args(
    pos: &Bound<'_, PyAny>,
    model: GravModel,
    degree: usize,
    order: Option<usize>,
) -> Result<(Vector3, usize, usize, GravityModel)> {
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
    } else {
        // Any real numeric 3-element array-like (list, tuple, integer
        // array), as the stub promises: `TypeError` for a non-numeric
        // input, `ValueError` for the wrong length
        to_vector3(pos, "pos")?
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
///     pos (satkit.itrfcoord|array-like): position at which to compute acceleration.  itrfcoord, or 3-element Cartesian ITRF position in meters (numpy array, list or tuple; integers are converted to float)
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
#[pyo3(signature=(pos, *, model=GravModel::egm2008, degree=6, order=None))]
pub fn gravity(
    py: Python,
    pos: &Bound<'_, PyAny>,
    #[pyo3(from_py_with = model_arg)] model: GravModel,
    #[pyo3(from_py_with = degree_arg)] degree: usize,
    #[pyo3(from_py_with = order_arg)] order: Option<usize>,
) -> Result<Py<PyAny>> {
    let (v, degree, order, model) = gravity_args(pos, model, degree, order)?;
    Ok(vec2py(py, &accel(&v, degree, order, model))?)
}

/// Acceleration vector due to Earth gravity and partials with respect to position
///
///
/// Args:
///     pos (satkit.itrfcoord|array-like): position at which to compute acceleration.  itrfcoord, or 3-element Cartesian ITRF position in meters (numpy array, list or tuple; integers are converted to float)
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
#[pyo3(signature=(pos, *, model=GravModel::egm2008, degree=6, order=None))]
pub fn gravity_and_partials(
    py: Python,
    pos: &Bound<'_, PyAny>,
    #[pyo3(from_py_with = model_arg)] model: GravModel,
    #[pyo3(from_py_with = degree_arg)] degree: usize,
    #[pyo3(from_py_with = order_arg)] order: Option<usize>,
) -> Result<(Py<PyAny>, Py<PyAny>)> {
    let (v, degree, order, model) = gravity_args(pos, model, degree, order)?;
    let (g, p) = accel_and_partials(&v, degree, order, model);
    // The partials' column-major storage read as a row-major (C-order) array,
    // as before: the array is `p` transposed (the matrix is symmetric up to
    // rounding, so this only affects the last bits)
    Ok((vec2py(py, &g)?, slice2py2d(py, p.as_slice(), 3, 3)?))
}
