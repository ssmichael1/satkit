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
    #[allow(clippy::upper_case_acronyms)]
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
    fn from(s: &SolarSystem) -> Self {
        match s {
            SolarSystem::Mercury => Self::Mercury,
            SolarSystem::Venus => Self::Venus,
            SolarSystem::EMB => Self::EMB,
            SolarSystem::Mars => Self::Mars,
            SolarSystem::Jupiter => Self::Jupiter,
            SolarSystem::Saturn => Self::Saturn,
            SolarSystem::Uranus => Self::Uranus,
            SolarSystem::Neptune => Self::Neptune,
            SolarSystem::Pluto => Self::Pluto,
            SolarSystem::Moon => Self::Moon,
            SolarSystem::Sun => Self::Sun,
        }
    }
}
