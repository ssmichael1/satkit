use pyo3::prelude::*;

use satkit::nrlmsise;
use satkit::Instant;

///
/// NRL-MSISE00 Atmospheric Model
///
/// Args:
///   alt_km (float): Altitude in kilometers
///
/// Keyword args:
///       latitude_deg (float):   Latitude in degrees
///      longitude_deg (float):   Longitude in degrees
///               time (satkit.time|datetime.datetime):  Time at which to evaluate the model
///   use_spaceweather (bool):   Use space weather database in calculation
///
/// Returns:
///  (float, float): Tuple of density (kg/m^3) and temperature (K)
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
