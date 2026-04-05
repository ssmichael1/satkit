//! ODE integrators specific to orbit propagation.
//!
//! General-purpose integrators (Runge-Kutta, Rosenbrock) live in
//! [`numeris::ode`]. This module holds integrators that are
//! astrodynamics-specific enough that they don't belong in a general
//! numerical library — currently:
//!
//! - [`GaussJackson8`] — 8th-order fixed-step multistep predictor-corrector
//!   for 2nd-order ODEs, the dominant method for high-precision orbit
//!   propagation in space surveillance and astrodynamics codes.

mod gauss_jackson;

pub use gauss_jackson::{GaussJackson8, GJDenseOutput, GJSettings, GJSolution};
