use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};

mod mod_utils;
mod pyconsts;
mod pydensity;
mod pyduration;
mod pyecom;
mod pyframes;
mod pyframetransform;
mod pygravity;
mod pyinstant;
mod pyitrfcoord;
mod pyjplephem;
mod pykepler;
mod pylpephem_moon;
mod pylpephem_planets;
mod pylpephem_sun;
mod pynrlmsise;
mod pypropresult;
mod pyquaternion;
mod pysatstate;
mod pysgp4;
mod pysolarsystem;
mod pyspaceweather;
mod pytle;
mod pytlefitstatus;

mod pylambert;
mod pypropagate;
mod pypropsettings;
mod pysatproperties;
mod pythrust;

mod pyomm;
mod pyutils;

use pyduration::PyDuration;
use pyframetransform as pyft;
use pyinstant::PyInstant;
use pyitrfcoord::{PyGeodet, PyITRFCoord};
use pykepler::PyKepler;
use pyquaternion::PyQuaternion;
use pysolarsystem::SolarSystem;

use pypropsettings::PyPropSettings;
use pysatstate::PySatState;

/// Space Weather Sub-Module
#[pymodule]
fn spaceweather(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyspaceweather::get, m)?)?;
    m.add_function(wrap_pyfunction!(pyspaceweather::coverage, m)?)?;
    m.add_function(wrap_pyfunction!(pyspaceweather::status, m)?)?;
    m.add_function(wrap_pyfunction!(
        pyspaceweather::disable_space_weather_time_warning,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pyspaceweather::init_from_path, m)?)?;
    m.add_function(wrap_pyfunction!(pyspaceweather::init_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(pyspaceweather::update, m)?)?;
    Ok(())
}

/// JPL Ephemeris Sub-Module
#[pymodule]
fn jplephem(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyjplephem::geocentric_pos, m)?)?;
    m.add_function(wrap_pyfunction!(pyjplephem::geocentric_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyjplephem::barycentric_pos, m)?)?;
    m.add_function(wrap_pyfunction!(pyjplephem::barycentric_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyjplephem::consts, m)?)?;

    Ok(())
}

/// Solar calculations
#[pymodule]
fn sun(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pylpephem_sun::pos_gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_sun::pos_mod, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_sun::rise_set, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_sun::shadowfunc, m)?)?;
    Ok(())
}

/// Lunar calculations
#[pymodule]
fn moon(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pylpephem_moon::pos_gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::phase, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::phase_name, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::illumination, m)?)?;
    m.add_class::<pylpephem_moon::MoonPhase>()?;
    Ok(())
}

/// Low-precision planetary ephemerides
#[pymodule]
fn planets(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pylpephem_planets::heliocentric_pos, m)?)?;
    Ok(())
}

/// Frame transform module: transform between varias coordinate frames
#[pymodule]
fn frametransform(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyft::earth_rotation_angle, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::gast, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::gmst, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::eqeq, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qitrf2tirs, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qtirs2cirs, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qitrf2gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qgcrf2itrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qitrf2gcrf_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qgcrf2itrf_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qteme2itrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qcirs2gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qteme2gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::pyeop, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::disable_eop_time_warning, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::eop_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::eop_source, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::eop_status, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::to_gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::from_gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::itrf_to_gcrf_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::gcrf_to_itrf_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::itrf_to_gcrf_state_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::gcrf_to_itrf_state_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qmod2gcrf, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::qtod2mod_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::rotation, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::rotation_with_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::rotation_approx, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::transform_state, m)?)?;
    m.add_function(wrap_pyfunction!(pyft::transform_state_approx, m)?)?;

    Ok(())
}

#[pymodule]
pub fn satkit(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(pyutils::enum_member, m)?)?;
    m.add_class::<PyInstant>()?;
    m.add_class::<PyDuration>()?;
    m.add_class::<pyinstant::PyTimeScale>()?;
    m.add_class::<pyinstant::PyWeekday>()?;
    m.add_class::<PyQuaternion>()?;
    m.add_class::<pyframes::PyFrame>()?;
    m.add_function(wrap_pyfunction!(pysgp4::sgp4, m)?)?;

    m.add_class::<pygravity::GravModel>()?;
    m.add_class::<pysgp4::GravConst>()?;
    m.add_class::<pysgp4::OpsMode>()?;
    m.add_class::<pysgp4::PySGP4Error>()?;

    m.add_function(wrap_pyfunction!(pygravity::gravity, m)?)?;
    m.add_function(wrap_pyfunction!(pygravity::gravity_and_partials, m)?)?;

    m.add_function(wrap_pyfunction!(pynrlmsise::nrlmsise00, m)?)?;

    m.add_class::<pyconsts::Consts>()?;
    m.add_class::<SolarSystem>()?;
    m.add_class::<pytle::PyTLE>()?;
    m.add_class::<pytlefitstatus::PyTleFitStatus>()?;

    m.add_class::<PyGeodet>()?;
    m.add_class::<PyITRFCoord>()?;

    m.add_class::<PyKepler>()?;
    m.add_class::<PySatState>()?;

    m.add_class::<PyPropSettings>()?;
    m.add_class::<pypropsettings::PyIntegrator>()?;
    m.add_class::<pypropsettings::PyTideModel>()?;
    m.add_class::<pysatproperties::PySatProperties>()?;
    m.add_class::<pyecom::PyEcomParams>()?;
    m.add_class::<pythrust::PyThrust>()?;
    m.add_class::<pypropresult::PyPropResult>()?;
    m.add_class::<pypropresult::PyPropStats>()?;
    m.add_function(wrap_pyfunction!(pypropagate::propagate, m)?)?;
    m.add_function(wrap_pyfunction!(pylambert::lambert, m)?)?;
    m.add_function(wrap_pyfunction!(pyomm::omm_from_url, m)?)?;
    m.add_function(wrap_pyfunction!(pyomm::omm_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(pyomm::omm_from_text, m)?)?;

    m.add_wrapped(wrap_pymodule!(frametransform))?;
    m.add_wrapped(wrap_pymodule!(jplephem))?;
    m.add_wrapped(wrap_pymodule!(sun))?;
    m.add_wrapped(wrap_pymodule!(moon))?;
    m.add_wrapped(wrap_pymodule!(planets))?;
    m.add_wrapped(wrap_pymodule!(spaceweather))?;

    m.add_wrapped(wrap_pymodule!(mod_utils::utils))?;
    m.add_wrapped(wrap_pymodule!(pydensity::density))?;

    Ok(())
}
