use pyo3::prelude::*;

use crate::consts as cconsts;

#[pyclass(name = "consts")]
pub struct Consts {}

#[pymethods]
#[allow(non_upper_case_globals)]
impl Consts {
    #[classattr]
    ///  WGS-84 semiparameter, in meters
    const wgs84_a: f64 = cconsts::WGS84_A;
    
    #[classattr]
    ///  WGS-84 flattening
    const wgs84_f: f64 = cconsts::WGS84_F;
    
    #[classattr]
    /// WGS-84 Earth radius, meters
    const earth_radius: f64 = cconsts::EARTH_RADIUS;

    ///  Gravitational parameter of Earth, m^3/s^2
    #[classattr]
    const mu_earth: f64 = cconsts::MU_EARTH;
    
    /// Gravitational parameter of moon, m^3/s^2
    #[classattr]
    const mu_moon: f64 = cconsts::MU_MOON;

    /// Gravitational parameter of sun, m^3/s^2
    #[classattr]
    const mu_sun: f64 = cconsts::MU_SUN;

    /// Alternative notation for gravitational parameter of Earth, m^3/s^2
    #[classattr]
    const GM: f64 = cconsts::GM;

    /// Rotation rate of Earth on own axis, rad/s
    #[classattr]
    const omega_earth: f64 = cconsts::OMEGA_EARTH;

    /// Speed of light, m/s
    #[classattr]
    const c: f64 = cconsts::C;

    /// Mean distance Earth to Sun, meters
    #[classattr]
    const au: f64 = cconsts::AU;

    /// Radius of sun, meters
    #[classattr]
    const sun_radius: f64 = cconsts::SUN_RADIUS;

    /// Radius of moon, meters
    #[classattr]
    const moon_radius: f64 = cconsts::MOON_RADIUS;

    /// Earth-moon mass ratio
    #[classattr]
    const earth_moon_mass_ratio: f64 = cconsts::EARTH_MOON_MASS_RATIO;

    /// Semiparameter for Geosynchronous orbits, meters
    #[classattr]
    const geo_r: f64 = cconsts::GEO_R;

    /// Mu for Earth per JGM3 Gravity model, m^3/s^2
    #[classattr]
    const jgm3_mu: f64 = cconsts::JGM3_MU;
    #[classattr]

    /// Earth semiparamter per JGM3 gravity model, m^3/s^2
    const jgm3_a: f64 = cconsts::JGM3_A;

    /// Earth J2 per JGM3 gravity model
    #[classattr]
    const jgm3_j2: f64 = cconsts::JGM3_J2;
}
