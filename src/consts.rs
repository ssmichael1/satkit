#[allow(non_upper_case_globals)]
///  WGS-84 semiparameter, in meters
pub const WGS84_A: f64 = 6378137.0;
///  WGS-84 flattening
pub const WGS84_F: f64 = 0.003352810664747;
/// WGS-84 Earth radius, meters
pub const EARTH_RADIUS: f64 = WGS84_A;
///  Gravitational parameter of Earth, m^3/s^2
pub const MU_EARTH: f64 = 3.986004418E14;
///  Gravitational parameter of Moon, m^3/s^2
pub const MU_MOON: f64 = 4.9048695E12;
///  Gravitational parameter of Sun, m^3/s^2
pub const MU_SUN: f64 = 1.32712440018E20;
///  Alternate name for gravitational parameter
pub const GM: f64 = MU_EARTH;
///  Earth rotation rate, radians / second
pub const OMEGA_EARTH: f64 = 7.292115090E-5;
///  speed of light, m/s
pub const C: f64 = 299792458.0;
///  Astronomical unit, meters
pub const AU: f64 = 1.495_978_707E11;
///  Sun radius in meters
pub const SUN_RADIUS: f64 = 695500000.0;
///  Moon radius, meters
pub const MOON_RADIUS: f64 = 1737400.0;
///  Ratio of earth mass to moon mass
pub const EARTH_MOON_MASS_RATIO: f64 = 81.300_568_221_497_22;
///  Geosynchronous Distance
pub const GEO_R: f64 = 42164172.58422766;
///  JGM3 Gravitational parameter of Earth, m^3/s^2
pub const JGM3_MU: f64 = 0.3986004415E+15;
///  JGM3 Semimajor axis of Earth, m
pub const JGM3_A: f64 = 0.6378136300E+07;
///  JGM3 J2 term
pub const JGM3_J2: f64 = -0.0010826360229829945;
//jgm3_J2:f64 = -sqrt(5.) * -0.484169548456e-03;
