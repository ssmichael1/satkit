use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PyInstant;
use crate::nrlmsise;
use crate::Instant;

///
/// NRL-MSISE00 Atmospheric Model
///
/// Args:
///   alt_km (float): Altitude in kilometers
///
/// Keyword args:
///       latitude_deg (float):   Latitude in degrees
///      longitude_deg (float):   Longitude in degrees
///                 tm (satkit.time):   The time (Instant object)
///   use_spaceweather (bool):   Use space weather database in calculation
///
/// Returns:
///  (float, float): Tuple of density (kg/m^3) and temperature (K)
///
#[pyfunction]
#[pyo3(signature=(alt_km, **option_kwds))]
pub fn nrlmsise00(alt_km: f64, option_kwds: Option<&Bound<'_, PyDict>>) -> PyResult<(f64, f64)> {
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut tm: Option<Instant> = None;
    let mut use_spaceweather: bool = true;
    if option_kwds.is_some() {
        let kwds = option_kwds.unwrap();

        if let Some(kw) = kwds.get_item("latitude_deg")? {
            lat = Some(kw.extract::<f64>()?);
        }
        if let Some(v) = kwds.get_item("longitude_deg")? {
            lon = Some(v.extract::<f64>()?);
        }
        if let Some(v) = kwds.get_item("time")? {
            tm = Some(v.extract::<PyInstant>()?.0);
        }
        if let Some(v) = kwds.get_item("use_spaceweather")? {
            use_spaceweather = v.extract::<bool>()?;
        }
    }

    Ok(nrlmsise::nrlmsise(alt_km, lat, lon, tm, use_spaceweather))
}
