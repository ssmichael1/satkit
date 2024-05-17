/// Solar system bodies
///
/// Coordinate origin is the solar system barycenter
///
/// # Notes:
///  * For native JPL Ephemerides function calls:
///    positions for all bodies are natively relative to
///    solar system barycenter, with exception of moon,
///    which is computed in Geocentric system
///  * EMB (2) is the Earth-Moon barycenter
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarSystem {
    /// Mercury
    Mercury = 0,
    /// Venus
    Venus = 1,
    /// Earth-Moon Barycenter
    EMB = 2,
    /// Mars
    Mars = 3,
    /// Jupiter
    Jupiter = 4,
    /// Saturn
    Saturn = 5,
    /// Uranus
    Uranus = 6,
    /// Neptune
    Neptune = 7,
    /// Pluto
    Pluto = 8,
    /// Moon (Geocentric)
    Moon = 9,
    /// Sun
    Sun = 10,
}
