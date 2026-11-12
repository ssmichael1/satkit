use pyo3::prelude::*;

use satkit::SolarSystem as SS;

/// Solar system bodies for which the JPL ephemeris (``satkit.jplephem``)
/// and the low-precision planetary ephemeris (``satkit.planets``) can be
/// computed
///
/// Notes:
///   * ``EMB`` is the Earth-Moon barycenter; there is no ``Earth`` member
///     (see ``satkit.jplephem.barycentric_pos`` for the Earth's position)
///   * Members compare equal to their integer value (``int(solarsystem.Sun)``
///     is 10), the body's index in the JPL DE files
#[derive(PartialEq, Eq, Hash)]
#[pyclass(name = "solarsystem", module = "satkit", eq, eq_int, frozen)]
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

crate::enum_pickle!(SolarSystem, "solarsystem");

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
