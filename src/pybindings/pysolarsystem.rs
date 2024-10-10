use pyo3::prelude::*;

use crate::SolarSystem as SS;

/// Solar system bodies
///
/// Coordinates origin is the solar system barycenter
///
/// Notes:
///   * Positions for all bodies are natively relative to
///     solar system barycenter, with exception of moon,
///     which is computed in Geocentric system
///   * EMB (2) is the Earth-Moon barycenter
///   * The sun position is relative to the solar system barycenter
///     (it will be close to origin)
#[derive(PartialEq, Eq)]
#[pyclass(name = "solarsystem", eq, eq_int)]
pub enum SolarSystem {
    Mercury = SS::Mercury as isize,
    Venus = SS::Venus as isize,
    EMB = SS::EMB as isize,
    Mars = SS::Mars as isize,
    Jupiter = SS::Jupiter as isize,
    Saturn = SS::Saturn as isize,
    Uranus = SS::Uranus as isize,
    Neptune = SS::Neptune as isize,
    Pluto = SS::Pluto as isize,
    Moon = SS::Moon as isize,
    Sun = SS::Sun as isize,
}

impl From<&SolarSystem> for SS {
    fn from(s: &SolarSystem) -> SS {
        match s {
            &SolarSystem::Mercury => SS::Mercury,
            &SolarSystem::Venus => SS::Venus,
            &SolarSystem::EMB => SS::EMB,
            &SolarSystem::Mars => SS::Mars,
            &SolarSystem::Jupiter => SS::Jupiter,
            &SolarSystem::Saturn => SS::Saturn,
            &SolarSystem::Uranus => SS::Uranus,
            &SolarSystem::Neptune => SS::Neptune,
            &SolarSystem::Pluto => SS::Pluto,
            &SolarSystem::Moon => SS::Moon,
            &SolarSystem::Sun => SS::Sun,
        }
    }
}

impl IntoPy<PyObject> for SS {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ss: SolarSystem = match self {
            SS::Mercury => SolarSystem::Mercury,
            SS::Venus => SolarSystem::Venus,
            SS::EMB => SolarSystem::EMB,
            SS::Mars => SolarSystem::Mars,
            SS::Jupiter => SolarSystem::Jupiter,
            SS::Saturn => SolarSystem::Saturn,
            SS::Uranus => SolarSystem::Uranus,
            SS::Neptune => SolarSystem::Neptune,
            SS::Pluto => SolarSystem::Pluto,
            SS::Moon => SolarSystem::Moon,
            SS::Sun => SolarSystem::Sun,
        };
        ss.into_py(py)
    }
}
