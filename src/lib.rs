//! # "SatKit" : Satellite Tool SatKit
//!
//!
//! # Crate Features:
//! * Timescale transformations (UTC, GPS, UT1, TBD, TT, ...)
//! * High-precision coordinate transforms between:
//!   * International Terrestrial Reference Frame (ITRF)
//!   * Geocentric Celestial Reference Frame (GCRF) using IAU-2006 reduction
//!   * True-Equinox Mean Equator (TEME) frame used in SGP4 propagation of TLEs
//!   * Celestial Intermediate Reference Frame (CIRF)
//!   * Terrestrial Intermediate Reference Frame (TIRF)
//!   * Terrestrial Geodetic frame (latitude, longitude)
//! * Two-Line Element Set (TLE) processing, and propagation with SGP4
//! * Keplerian orbit propagation
//! * JPL planetary ephemerides
//! * High-order gravity models
//! * High-precision, high-speed numerical satellite orbit propagation with high-order (9/8) efficient Runga-Kutta solvers, ability to solve for state transition matrix for covariance propagation, and inclusion following forces:
//!   * High-order Earth gravity with multiple models
//!   * Solar gravity
//!   * Lunar gravity
//!   * Dra, with NRL MISE-00 density model and inclusion of space weather data
//!   * Radiation pressure
//!
//! # Language Bindings
//!
//! * Standalone Rust library available on on <https://crates.io>
//! * Python bindings availble on PyPi
//!
//!

// Type definitions
pub mod types;

/// Time and time bases (UTC, TAI, GPS, TT, etc...)
pub mod astrotime;
/// Universal constants
pub mod consts;
/// Earth orientation parameters (polar motion, delta-UT1, lenth of day)
pub mod earth_orientation_params;
/// Zonal gravity model for Earth gravity
pub mod earthgravity;
/// Conversion between coordinate frames
pub mod frametransform;
/// Internation Terrestrial Reference Frame coordinates &
/// transformations to Geodetic, East-North-Up, North-East-Down
pub mod itrfcoord;
/// Solar system body ephemerides, as published by the Jet Propulsion Laboratory (JPL)
pub mod jplephem;
/// Keplerian orbital elements
pub mod kepler;
/// Low-precision ephemeris for sun and moon
pub mod lpephem;
/// NRL-MISE00 Density model
pub mod nrlmsise;
/// High-Precision Orbit Propagation via Runga-Kutta 9(8) Integration
pub mod orbitprop;
/// SGP-4 Orbit Propagator
pub mod sgp4;
/// Solar system bodies
mod solarsystem;
/// Space Weather
pub mod spaceweather;
/// Two-line Element Set
pub mod tle;
/// Utility functions
pub mod utils;

// Filters
pub mod filters;

/// Coordinate frames
mod frames;

// Integrate ordinary differential equations
mod ode;

mod duration;

// Objects available at crate level
pub use astrotime::AstroTime;
pub use astrotime::Scale as TimeScale;
pub use duration::Duration;
pub use frames::Frame;
pub use itrfcoord::ITRFCoord;
pub use solarsystem::SolarSystem;
pub use tle::TLE;
pub use utils::SKErr;
pub use utils::SKResult;

#[cfg(feature = "pybindings")]
pub mod pybindings;
