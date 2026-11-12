use pyo3::prelude::*;

use satkit::nrlmsise;
use satkit::Instant;

///
/// NRL-MSISE00 Atmospheric Model
///
/// Args:
///     alt_km (float): Altitude above the WGS-84 ellipsoid in kilometers
///         (-1 or above; the model runs from the ground to the exosphere)
///
/// Keyword Args:
///     latitude_deg (float): Geodetic latitude in degrees, in [-90, 90]. Default is 0
///     longitude_deg (float): Longitude in degrees. Default is 0
///     time (satkit.time|datetime.datetime|numpy.datetime64, optional): Time at
///         which to evaluate the model. Without a time the model is evaluated
///         for 00:00 UT on 1 January, on default indices (F10.7 = F10.7A = 150, Ap = 4)
///     use_spaceweather (bool): Look up F10.7 and Ap in the space weather
///         data at ``time``. Default is True. It has no effect without a
///         ``time``; with False, or for a time the data does not cover, the
///         default indices are used
///
/// Returns:
///     (float, float): Tuple of mass density (kg/m^3) and temperature (K)
///
/// Raises:
///     TypeError: ``time`` is not a single time value, or a number is expected
///     ValueError: ``alt_km`` is NaN, infinite or below -1; ``latitude_deg`` is
///         NaN or outside [-90, 90]; ``longitude_deg`` is NaN or infinite
///
#[pyfunction]
#[pyo3(signature=(alt_km, *, latitude_deg=0.0, longitude_deg=0.0, time=None, use_spaceweather=true))]
pub fn nrlmsise00(
    alt_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    time: Option<&Bound<'_, PyAny>>,
    use_spaceweather: bool,
) -> anyhow::Result<(f64, f64)> {
    let tm: Option<Instant> = time.map(crate::pyutils::instant_from_pyany).transpose()?;
    Ok(nrlmsise::nrlmsise(
        alt_km,
        Some(latitude_deg),
        Some(longitude_deg),
        tm.as_ref(),
        use_spaceweather,
    ))
}
