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
    MERCURY = 0,
    /// Venus
    VENUS = 1,
    /// Earth-Moon Barycenter
    EMB = 2,
    /// Mars
    MARS = 3,
    /// Jupiter
    JUPITER = 4,
    /// Saturn
    SATURN = 5,
    /// Uranus
    URANUS = 6,
    /// Neptune
    NEPTUNE = 7,
    /// Pluto
    PLUTO = 8,
    /// Moon (Geocentric)
    MOON = 9,
    /// Sun
    SUN = 10,
}
