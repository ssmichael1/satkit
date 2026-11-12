use pyo3::prelude::*;

use satkit::consts as cconsts;

/// Some constants that are useful for satellite dynamics
///
/// The class is a namespace: it cannot be instantiated and its attributes
/// cannot be reassigned.
#[pyclass(name = "consts", module = "satkit", frozen, immutable_type)]
pub struct Consts {}

#[pymethods]
#[allow(non_upper_case_globals)]
impl Consts {
    #[classattr]
    ///  WGS-84 semi-major axis (equatorial radius), in meters
    const wgs84_a: f64 = cconsts::WGS84_A;

    #[classattr]
    ///  WGS-84 flattening, unitless
    const wgs84_f: f64 = cconsts::WGS84_F;

    #[classattr]
    /// WGS-84 equatorial Earth radius, meters (same value as ``wgs84_a``)
    const earth_radius: f64 = cconsts::EARTH_RADIUS;

    ///  Gravitational parameter of Earth, m^3/s^2 (WGS-84)
    #[classattr]
    const mu_earth: f64 = cconsts::MU_EARTH;

    /// Gravitational parameter of moon, m^3/s^2 (JPL DE440)
    #[classattr]
    const mu_moon: f64 = cconsts::MU_MOON;

    /// Gravitational parameter of sun, m^3/s^2 (JPL DE440)
    #[classattr]
    const mu_sun: f64 = cconsts::MU_SUN;

    /// Alternative notation for gravitational parameter of Earth, m^3/s^2
    /// (same value as ``mu_earth``)
    #[classattr]
    const GM: f64 = cconsts::GM;

    /// Rotation rate of Earth on own axis, rad/s
    #[classattr]
    const omega_earth: f64 = cconsts::OMEGA_EARTH;

    /// Speed of light, m/s
    #[classattr]
    const c: f64 = cconsts::C;

    /// Astronomical unit, meters: the exact defined constant of IAU 2012
    /// Resolution B1 (149 597 870 700 m), close to the mean Earth-Sun distance
    #[classattr]
    const au: f64 = cconsts::AU;

    /// Radius of sun, meters
    #[classattr]
    const sun_radius: f64 = cconsts::SUN_RADIUS;

    /// Solar radiation pressure at 1 AU (1367 W/m^2 / c), N/m^2
    #[classattr]
    const solar_pressure_1au: f64 = cconsts::SOLAR_PRESSURE_1AU;

    /// Radius of moon, meters
    #[classattr]
    const moon_radius: f64 = cconsts::MOON_RADIUS;

    /// Earth-moon mass ratio (Earth mass over Moon mass), unitless
    ///
    /// The ratio of the JPL DE440 gravitational parameters. It differs from
    /// ``mu_earth / mu_moon`` in the eighth digit, because ``mu_earth`` is
    /// the WGS-84 value.
    #[classattr]
    const earth_moon_mass_ratio: f64 = cconsts::EARTH_MOON_MASS_RATIO;

    /// Radius of a circular geosynchronous orbit (its semi-major axis),
    /// meters from the Earth's center: ``(mu_earth / omega_earth^2)^(1/3)``
    #[classattr]
    const geo_r: f64 = cconsts::GEO_R;

    /// Mu for Earth per JGM3 Gravity model, m^3/s^2
    #[classattr]
    const jgm3_mu: f64 = cconsts::JGM3_MU;

    /// Earth equatorial radius of the JGM3 gravity model, meters
    #[classattr]
    const jgm3_a: f64 = cconsts::JGM3_A;

    /// Earth J2 (oblateness) per JGM3 gravity model, unitless
    #[classattr]
    const jgm3_j2: f64 = cconsts::JGM3_J2;
}
