use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PyAstroTime;
use crate::nrlmsise;
use crate::AstroTime;

///
/// NRL-MSISE00 Atmospheric Model
///
/// Args:
///   alt_km (float): Altitude in kilometers
///
/// Keyword args:
///       latitude_deg (float):   Latitude in degrees
///      longitude_deg (float):   Longitude in degrees
///                 tm (satkit.time):   The time (astrotime object)
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
    let mut tm: Option<AstroTime> = None;
    let mut use_spaceweather: bool = true;
    if option_kwds.is_some() {
        let kwds = option_kwds.unwrap();
        match kwds.get_item("latitude_deg")? {
            Some(v) => lat = Some(v.extract::<f64>()?),
            None => (),
        }
        match kwds.get_item("longitude_deg")? {
            Some(v) => lon = Some(v.extract::<f64>()?),
            None => (),
        }
        match kwds.get_item("time")? {
            Some(v) => tm = Some(v.extract::<PyAstroTime>()?.inner),
            None => (),
        }
        match kwds.get_item("use_spaceweather")? {
            Some(v) => use_spaceweather = v.extract::<bool>()?,
            None => (),
        }
    }

    Ok(nrlmsise::nrlmsise(alt_km, lat, lon, tm, use_spaceweather))
}
