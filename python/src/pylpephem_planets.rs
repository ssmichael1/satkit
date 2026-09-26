use crate::pysolarsystem::SolarSystem;
use crate::pyutils;
use pyo3::prelude::*;
use satkit::lpephem;

use anyhow::Result;

/// Approximate Heliocentric position of a planet
/// in ICRF frame but centered on sun not solar system barycenter
///
/// Notes:
///  * Keplerian-element approximation of Standish & Williams, see <https://ssd.jpl.nasa.gov/planets/approx_pos.html>
///  * Valid bodies are Mercury through Pluto and the Earth-Moon barycenter (EMB); the Sun and Moon raise RuntimeError
///  * Valid from 3000 BC to 3000 AD, with a more accurate element set for 1800 AD to 2050 AD
///  * JPL's approximate errors in heliocentric ecliptic longitude / latitude / range run from 15" / 1" / 1000 km (Mercury, 1800-2050) to 2000" / 30" / 8 million km (Uranus, 3000 BC-3000 AD)
///  * The Uranus, Neptune and Pluto elements follow the orbit about the solar-system barycenter, so from 1800 to 2050 their heliocentric errors are dominated by the Sun's unmodeled barycentric motion (up to ~2 arcmin and ~2.3 million km)
///
/// Args:
///    planet (satkit.solarsystem): Mercury through Pluto, or EMB
///    time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///   numpy.ndarray: 3-element numpy array for a single time, or Nx3 numpy
///   array for a list / array of N times, representing planet position in ICRF frame
///   at input time[s].  Units are meters
#[pyfunction]
pub fn heliocentric_pos(planet: &SolarSystem, time: &Bound<'_, PyAny>) -> Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(
        &|t| lpephem::heliocentric_pos(planet.into(), t).map_err(anyhow::Error::from),
        time,
    )
}
