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
    // The evaluator would silently truncate the order to the degree
    if order > degree {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gravity order {order} exceeds the degree ({degree})"
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
    // The field is singular at the origin (and NaN propagates): refuse both
    // rather than return an all-NaN acceleration
    if !v.iter().all(|x| x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pos must be finite (no NaN or infinity)",
        )
        .into());
    }
    if v.norm() == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pos is the origin, where the gravity field is singular",
        )
        .into());
    }
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
/// Members are hashable, compare equal to their integer value and to each
/// other by value, and cannot be constructed.
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
///     numpy.ndarray: 3-element numpy array (shape ``(3,)``) representing acceleration due to Earth gravity at input position, in the ITRF.  Units are m/s^2.  The centrifugal acceleration of the rotating frame is not included
///
/// Keyword Args:
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.egm2008
///     degree (int): maximum degree of gravity model to use, 0 to 70.  Default is 6.  0 and 1 both give the point-mass field
///     order (int): maximum order of gravity model to use, 0 to ``degree``.  Default is same as degree
///
/// Raises:
///     TypeError: ``pos`` is not an itrfcoord or a real numeric array-like; ``model`` is not a ``gravmodel``; ``degree`` or ``order`` is not an integer
///     ValueError: ``pos`` does not have 3 elements, holds a NaN or infinity, or is the origin; ``degree`` is negative or above 70; ``order`` is negative or above ``degree``
///     RuntimeError: the coefficient file of ``gravmodel.itugrace16`` is not in the data directory and cannot be downloaded
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and E. Gill, Springer, 2000.
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
///     (numpy.ndarray, numpy.ndarray): tuple of 3-element numpy array (shape ``(3,)``) representing acceleration due to Earth gravity at input position and 3x3 numpy array (shape ``(3, 3)``) of partials of acceleration with respect to ITRF position, ``partials[i, j] = d a_i / d r_j``.  Units are m/s^2 for gravity and m/s^2/m (1/s^2) for partials
///
/// Keyword Args:
///     model (satkit.gravmodel): gravity model to use.  Default is satkit.gravmodel.egm2008
///     degree (int): maximum degree of gravity model to use, 0 to 70.  Default is 6.  0 and 1 both give the point-mass field
///     order (int): maximum order of gravity model to use, 0 to ``degree``.  Default is same as degree
///
/// Raises:
///     TypeError: ``pos`` is not an itrfcoord or a real numeric array-like; ``model`` is not a ``gravmodel``; ``degree`` or ``order`` is not an integer
///     ValueError: ``pos`` does not have 3 elements, holds a NaN or infinity, or is the origin; ``degree`` is negative or above 70; ``order`` is negative or above ``degree``
///     RuntimeError: the coefficient file of ``gravmodel.itugrace16`` is not in the data directory and cannot be downloaded
///
/// Notes:
///     * For details of calculation, see Chapter 3.2 of "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and E. Gill, Springer, 2000.
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
