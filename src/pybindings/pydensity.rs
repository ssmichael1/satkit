use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};
use pyo3::wrap_pyfunction;

use crate::nrlmsise;
use crate::Instant;

use super::PyITRFCoord;
use super::PyInstant;

///
/// NRL MSISE-00 Density Model
///
/// Args:
///     coord (itrfcoord): ITRF coordinate at which to compute density & temperature
///     time (satkit.time): Optional time object representing instant at which to compute
///
/// Returns:
///     tuple: (rho, T) where rho is atmosphere mass density in kg / m^3 and T is temperature in Kelvin
#[pyfunction(name = "nrlmsise")]
#[pyo3(signature=(*args))]
fn pynrlmsise(args: &Bound<'_, PyTuple>) -> PyResult<(f64, f64)> {
    //let args = args.as_gil_ref();

    if args.len() == 0 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Invalid number of arguments",
        ));
    }

    let time: Option<Instant> = {
        if args.get_item(args.len() - 1)?.is_instance_of::<PyInstant>() {
            Some(
                args.get_item(args.len() - 1)?
                    .extract::<PyInstant>()
                    .unwrap()
                    .0,
            )
        } else {
            None
        }
    };
    if args.get_item(0)?.is_instance_of::<PyITRFCoord>() {
        let itrf = args.get_item(0)?.extract::<PyITRFCoord>().unwrap().0;
        Ok(nrlmsise::nrlmsise(
            itrf.hae() / 1.0e3,
            Some(itrf.latitude_rad()),
            Some(itrf.longitude_rad()),
            time,
            true,
        ))
    } else if args.get_item(0)?.is_instance_of::<PyFloat>() {
        let altitude = args.get_item(0)?.extract::<f64>().unwrap();
        let latitude: Option<f64> = {
            if args.len() > 1 && args.get_item(1)?.is_instance_of::<PyFloat>() {
                Some(args.get_item(1)?.extract::<f64>().unwrap())
            } else {
                None
            }
        };
        let longitude: Option<f64> = {
            if args.len() > 2 && args.get_item(2)?.is_instance_of::<PyFloat>() {
                Some(args.get_item(2)?.extract::<f64>().unwrap())
            } else {
                None
            }
        };
        Ok(nrlmsise::nrlmsise(
            altitude / 1.0e3,
            latitude,
            longitude,
            time,
            true,
        ))
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("Invalid arguments"));
    }
}

#[pymodule]
pub fn density(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pynrlmsise, m)?).unwrap();
    Ok(())
}
