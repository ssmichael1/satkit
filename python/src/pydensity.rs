use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;

use satkit::nrlmsise;
use satkit::Instant;

use crate::pyinstant::is_time_scalar;
use crate::pyutils::instant_from_pyany;
use crate::PyITRFCoord;

/// The optional time argument: `None`, or a single `satkit.time` /
/// `datetime.datetime` / `numpy.datetime64`. Anything else is a `TypeError`
/// rather than being silently read as "no time" (which runs the model on its
/// default indices).
fn time_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<Instant>> {
    if obj.is_none() {
        Ok(None)
    } else if is_time_scalar(obj) {
        Ok(Some(instant_from_pyany(obj)?))
    } else {
        Err(PyTypeError::new_err(format!(
            "time must be satkit.time, datetime.datetime, numpy.datetime64 or None, not {}",
            obj.get_type().name()?
        )))
    }
}

///
/// NRL MSISE-00 Density Model
///
/// Called as ``nrlmsise(itrf, time=None)`` or
/// ``nrlmsise(altitude_meters, latitude_rad=0, longitude_rad=0, time=None)``
/// (positional arguments; the time may directly follow the altitude or the
/// latitude).
///
/// Args:
///     itrf (satkit.itrfcoord): position at which to compute density & temperature
///     altitude_meters (float): Altitude in meters
///     latitude_rad (float, optional): Latitude in radians. Default is 0.
///     longitude_rad (float, optional): Longitude in radians. Default is 0.
///     time (satkit.time|datetime.datetime, optional): Instant at which to compute
///         density & temperature, using the space weather at that time. Without a
///         time the model runs on its default indices (F10.7 = F10.7A = 150, Ap = 4).
///
/// Returns:
///     tuple: (rho, T) where rho is mass density in kg/m^3 and T is temperature in Kelvin
///
/// Raises:
///     TypeError: If an angle is not a real number, or ``time`` is not a
///         ``satkit.time``, ``datetime.datetime``, ``numpy.datetime64`` or ``None``
#[pyfunction(name = "nrlmsise")]
#[pyo3(signature=(*args))]
fn pynrlmsise(args: &Bound<'_, PyTuple>) -> PyResult<(f64, f64)> {
    let Ok(first) = args.get_item(0) else {
        return Err(PyTypeError::new_err("Invalid number of arguments"));
    };
    if first.is_instance_of::<PyITRFCoord>() {
        let itrf = first.extract::<PyITRFCoord>()?.0;
        let time = match args.len() {
            1 => None,
            2 => time_arg(&args.get_item(1)?)?,
            _ => {
                return Err(PyTypeError::new_err(
                    "nrlmsise(itrfcoord, time=None) takes at most 2 arguments",
                ))
            }
        };
        return Ok(nrlmsise::nrlmsise(
            itrf.hae() / 1.0e3,
            Some(itrf.latitude_deg()),
            Some(itrf.longitude_deg()),
            time.as_ref(),
            true,
        ));
    }
    let Ok(altitude) = first.extract::<f64>() else {
        return Err(PyTypeError::new_err(
            "Invalid arguments: expected an itrfcoord or an altitude in meters",
        ));
    };
    if args.len() > 4 {
        return Err(PyTypeError::new_err(
            "nrlmsise(altitude, latitude, longitude, time) takes at most 4 arguments",
        ));
    }
    // Up to two real numbers (latitude, longitude), then an optional time.
    // The stub documents the angles as radians (satkit's convention without a
    // `_deg` suffix); the model takes degrees, as the itrfcoord branch above
    // and `nrlmsise00(latitude_deg=...)` already supply.
    let mut angles: Vec<f64> = Vec::with_capacity(2);
    let mut time: Option<Instant> = None;
    for (i, item) in args.iter().enumerate().skip(1) {
        let is_time = item.is_none() || is_time_scalar(&item);
        if !is_time && angles.len() < 2 {
            let Ok(v) = item.extract::<f64>() else {
                return Err(PyTypeError::new_err(format!(
                    "latitude and longitude must be real numbers, not {}",
                    item.get_type().name()?
                )));
            };
            angles.push(v.to_degrees());
        } else if i == args.len() - 1 {
            time = time_arg(&item)?;
        } else {
            return Err(PyTypeError::new_err(
                "the time must be the last argument of nrlmsise(altitude, latitude, longitude, time)",
            ));
        }
    }
    Ok(nrlmsise::nrlmsise(
        altitude / 1.0e3,
        angles.first().copied(),
        angles.get(1).copied(),
        time.as_ref(),
        true,
    ))
}

#[pymodule]
pub fn density(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pynrlmsise, m)?)?;
    Ok(())
}
