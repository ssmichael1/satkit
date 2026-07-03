use pyo3::prelude::*;
use pyo3::types::PyDict;

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
#[pyo3(signature=(alt_km, **option_kwds))]
pub fn nrlmsise00(
    alt_km: f64,
    option_kwds: Option<&Bound<'_, PyDict>>,
) -> anyhow::Result<(f64, f64)> {
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut tm: Option<Instant> = None;
    let mut use_spaceweather: bool = true;
    if let Some(kwds) = option_kwds {
        if let Some(kw) = kwds.get_item("latitude_deg")? {
            lat = Some(kw.extract::<f64>()?);
            kwds.del_item("latitude_deg")?;
        }
        if let Some(v) = kwds.get_item("longitude_deg")? {
            lon = Some(v.extract::<f64>()?);
            kwds.del_item("longitude_deg")?;
        }
        if let Some(v) = kwds.get_item("time")? {
            tm = Some(crate::pyutils::instant_from_pyany(&v)?);
            kwds.del_item("time")?;
        }
        if let Some(v) = kwds.get_item("use_spaceweather")? {
            use_spaceweather = v.extract::<bool>()?;
            kwds.del_item("use_spaceweather")?;
        }
        crate::pyutils::reject_unused_kwargs(kwds)?;
    }

    Ok(nrlmsise::nrlmsise(
        alt_km,
        lat,
        lon,
        tm.as_ref(),
        use_spaceweather,
    ))
}
