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
//! # Getting started
//!
//! The library relies on the use of several external data sources for many of the calculations.  These include:
//!
//! * <a href="https://ssd.jpl.nasa.gov/ephem.html">JPL Planetary Ephemerides</a>
//! * <a href="http://icgem.gfz-potsdam.de/calculation">Earth Gravity Models</a>
//! * <a href="https://celestrak.org/SpaceData/">Space Weather Data and Earth orientation parameters</a>
//! * <a href="https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html">Coefficients for Earth-fixed to Inertial coordinate transforms</a>
//!
//! These data sources must be downloaded and placed in a directory that is accessible to the library.
//! The library provides a utility function to download these files from the internet.
//!
//! The data files need only be downloaded once.  However, the space weather data file (necessary for density calculations that impact satellite drag)
//! and the Earth Orientation Parameters (necessary for accurate inertial-to-earth frame transformations) are updated daily and should be
//! refreshed as necessary.
//!
//! ## Downloading the data files
//! ```no_run
//! // Print the directoyr where data will be stored
//! println!("Data directory: {:?}", satkit::utils::datadir());
//! // Update the data files (download those that are missing; refresh those that are out of date)
//! // This will always download the most-recent space weather data and Earth Orientation Parameters
//! // Other data files will be skipped if they are already present
//! satkit::utils::update_datafiles(None, false);
//! ```

#![warn(clippy::all, clippy::use_self, clippy::cargo)]

// Type definitions
pub mod types;

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

// Time and duration
mod time;
pub use time::{Duration, Instant, TimeScale, Weekday};

// Objects available at crate level
pub use frames::Frame;
pub use itrfcoord::ITRFCoord;
pub use solarsystem::SolarSystem;
pub use tle::TLE;
#[cfg(feature = "pybindings")]
pub mod pybindings;
