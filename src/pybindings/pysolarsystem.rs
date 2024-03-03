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
#[pyclass(name = "solarsystem")]
pub enum SolarSystem {
    Mercury = SS::MERCURY as isize,
    Venus = SS::VENUS as isize,
    EMB = SS::EMB as isize,
    Mars = SS::MARS as isize,
    Jupiter = SS::JUPITER as isize,
    Saturn = SS::SATURN as isize,
    Uranus = SS::URANUS as isize,
    Neptune = SS::NEPTUNE as isize,
    Pluto = SS::PLUTO as isize,
    Moon = SS::MOON as isize,
    Sun = SS::SUN as isize,
}

impl From<&SolarSystem> for SS {
    fn from(s: &SolarSystem) -> SS {
        match s {
            &SolarSystem::Mercury => SS::MERCURY,
            &SolarSystem::Venus => SS::VENUS,
            &SolarSystem::EMB => SS::EMB,
            &SolarSystem::Mars => SS::MARS,
            &SolarSystem::Jupiter => SS::JUPITER,
            &SolarSystem::Saturn => SS::SATURN,
            &SolarSystem::Uranus => SS::URANUS,
            &SolarSystem::Neptune => SS::NEPTUNE,
            &SolarSystem::Pluto => SS::PLUTO,
            &SolarSystem::Moon => SS::MOON,
            &SolarSystem::Sun => SS::SUN,
        }
    }
}

impl IntoPy<PyObject> for SS {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ss: SolarSystem = match self {
            SS::MERCURY => SolarSystem::Mercury,
            SS::VENUS => SolarSystem::Venus,
            SS::EMB => SolarSystem::EMB,
            SS::MARS => SolarSystem::Mars,
            SS::JUPITER => SolarSystem::Jupiter,
            SS::SATURN => SolarSystem::Saturn,
            SS::URANUS => SolarSystem::Uranus,
            SS::NEPTUNE => SolarSystem::Neptune,
            SS::PLUTO => SolarSystem::Pluto,
            SS::MOON => SolarSystem::Moon,
            SS::SUN => SolarSystem::Sun,
        };
        ss.into_py(py)
    }
}
