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
mod pylogging;
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
    m.add_function(wrap_pyfunction!(pylpephem_moon::pos_mod, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::phase, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::phase_name, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::illumination, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::rise_set, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::phase_times, m)?)?;
    m.add_function(wrap_pyfunction!(pylpephem_moon::next_phase, m)?)?;
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

/// Add a native submodule to the extension as the importable module
/// `satkit.<name>`.
///
/// PyO3 creates a nested `#[pymodule]` as a bare attribute: named `sun`
/// rather than `satkit.sun`, and absent from `sys.modules`, so
/// `import satkit.sun` and `from satkit.sun import pos_gcrf` failed although
/// the stubs (and `satkit.sun.pos_gcrf`) said they work, and its functions
/// could not be pickled (`pos_gcrf.__module__` was `"sun"`). This gives the
/// module and its functions their qualified name and registers the module.
/// Classes get theirs from `#[pyclass(module = ...)]`.
fn add_native_submodule(m: &Bound<'_, PyModule>, sub: Bound<'_, PyModule>) -> PyResult<()> {
    let short: String = sub.name()?.extract()?;
    let full = format!("satkit.{short}");
    sub.setattr("__name__", &full)?;
    for (_, obj) in sub.dict().iter() {
        if obj.cast::<pyo3::types::PyCFunction>().is_ok() {
            obj.setattr("__module__", &full)?;
        }
    }
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(&full, &sub)?;
    m.add(short.as_str(), sub)
}

#[pymodule]
pub fn satkit(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Forward satkit's warnings and notices (Rust `log` records) to
    // Python's `logging`, as loggers `satkit.<module>`.
    pylogging::init();
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
    m.add("TLEWarning", m.py().get_type::<pytle::TLEWarning>())?;
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

    // The eight native submodules, importable as `satkit.<name>`
    for sub in [
        wrap_pymodule!(frametransform)(py),
        wrap_pymodule!(jplephem)(py),
        wrap_pymodule!(sun)(py),
        wrap_pymodule!(moon)(py),
        wrap_pymodule!(planets)(py),
        wrap_pymodule!(spaceweather)(py),
        wrap_pymodule!(mod_utils::utils)(py),
        wrap_pymodule!(pydensity::density)(py),
    ] {
        add_native_submodule(m, sub.into_bound(py))?;
    }

    Ok(())
}
