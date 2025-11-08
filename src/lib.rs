//! # SatKit: Satellite Toolkit
//!
//! A comprehensive, high-performance satellite astrodynamics library combining the speed of Rust
//! with the convenience of Python. SatKit provides industrial-grade satellite orbital mechanics
//! calculations with a clean, intuitive API. Built from the ground up in Rust for maximum performance
//! and memory safety, it offers complete Python bindings for all functionality, making advanced orbital
//! mechanics accessible to both systems programmers and data scientists.
//!
//! ## Core Features
//!
//! ### Time Systems
//! - Comprehensive timescale transformations (UTC, GPS, UT1, TDB, TT, TAI)
//! - Leap second handling
//! - High-precision time arithmetic and conversions
//!
//! ### Coordinate Frame Transformations
//! High-precision coordinate transforms between multiple reference frames:
//! - **International Terrestrial Reference Frame (ITRF)**: Earth-fixed frame
//! - **Geocentric Celestial Reference Frame (GCRF)**: Inertial frame using IAU-2006 reduction
//! - **True Equinox Mean Equator (TEME)**: Frame used in SGP4 propagation
//! - **Celestial Intermediate Reference Frame (CIRF)**: IAU-2006 intermediate frame
//! - **Terrestrial Intermediate Reference Frame (TIRF)**: Earth-rotation intermediate frame
//! - **Geodetic Coordinates**: Latitude, longitude, altitude conversions
//!
//! ### Orbit Propagation
//! Multiple propagation methods for various accuracy requirements:
//! - **SGP4**: Simplified General Perturbations for Two-Line Element (TLE) sets with fitting capability
//! - **Numerical Integration**: High-precision propagation using adaptive Runge-Kutta 9(8) methods
//! - **Keplerian**: Simplified two-body propagation
//! - **State Transition Matrix**: Support for covariance propagation
//!
//! ### Force Models
//! Comprehensive perturbation modeling:
//! - High-order Earth gravity (JGM2, JGM3, EGM96, ITU GRACE16)
//! - Solar and lunar gravity perturbations
//! - Atmospheric drag using NRLMSISE-00 density model with space weather data
//! - Solar radiation pressure
//!
//! ### Ephemerides
//! - **JPL Ephemerides**: High-precision planetary and lunar positions
//! - **Low-Precision Ephemerides**: Fast analytical models for sun and moon
//!
//! ### Additional Capabilities
//! - Keplerian orbital elements and conversions
//! - Geodesic distance calculations
//! - TLE parsing, generation, and orbit fitting
//! - Unscented Kalman Filter (UKF) implementation
//!
//! ## Language Bindings
//!
//! - **Rust**: Native library available on [crates.io](https://crates.io/crates/satkit)
//! - **Python**: Complete Python bindings via PyO3, available on [PyPI](https://pypi.org/project/satkit/)
//!   - Binary wheels for Windows, macOS (Intel & ARM), and Linux (x86_64 & ARM64)
//!   - Python versions 3.8 through 3.13
//!   - Documentation at <https://satellite-toolkit.readthedocs.io/>
//!
//! ## Getting Started
//!
//! ### Data Files
//!
//! The library requires external data files for many calculations:
//!
//! - [JPL Planetary Ephemerides](https://ssd.jpl.nasa.gov/ephem.html) - High-precision planetary positions
//! - [Earth Gravity Models](http://icgem.gfz-potsdam.de/) - Spherical harmonic coefficients
//! - [Space Weather Data](https://celestrak.org/SpaceData/) - Solar flux and geomagnetic indices
//! - [Earth Orientation Parameters](https://celestrak.org/SpaceData/) - Polar motion and UT1-UTC
//! - [IERS Conventions Tables](https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html) - Nutation coefficients
//!
//! Data files need to be downloaded once. Space weather and Earth orientation parameter files are
//! updated daily and should be refreshed periodically for optimal accuracy.
//!
//! ### Downloading Data Files
//!
//! ```no_run
//! // Print the directory where data will be stored
//! println!("Data directory: {:?}", satkit::utils::datadir());
//!
//! // Download required data files
//! // - Downloads missing files
//! // - Updates space weather and Earth orientation parameters
//! // - Skips files that already exist
//! satkit::utils::update_datafiles(None, false);
//! ```
//!
//! ## Example Usage
//!
//! ```no_run
//! use satkit::{Instant, Duration, TimeScale, ITRFCoord, SolarSystem};
//!
//! // Create a time instant
//! let time = Instant::from_datetime(2024, 1, 1, 12, 0, 0.0).unwrap();
//!
//! // Coordinate frame transformations
//! let itrf_pos = ITRFCoord::from_geodetic_deg(42.0, -71.0, 100.0);
//!
//! // Get planetary ephemeris
//! let (moon_pos, moon_vel) = satkit::jplephem::geocentric_state(
//!     SolarSystem::Moon,
//!     &time
//! ).unwrap();
//! ```
//!
//! ## References
//!
//! This implementation relies heaviliy on the following excellent references:
//!
//! - **"Fundamentals of Astrodynamics and Applications, Fourth Edition"**
//!   by D. Vallado, Microcosm Press and Springer, 2013
//! - **"Satellite Orbits: Models, Methods, Applications"**
//!   by O. Montenbruck and E. Gill, Springer, 2000
//!
//! ## License
//!
//! MIT License - See LICENSE file for details

#![warn(clippy::all, clippy::use_self, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

// Math type definitions, mostly nalgebra based
pub mod mathtypes;

/// Universal constants
pub mod consts;
/// Earth orientation parameters (polar motion, delta-UT1, length of day)
pub mod earth_orientation_params;
/// Zonal gravity model for Earth gravity
pub mod earthgravity;
/// Conversion between coordinate frames
pub mod frametransform;
/// International Terrestrial Reference Frame coordinates &
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
