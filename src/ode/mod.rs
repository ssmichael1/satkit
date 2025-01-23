//! Ordinary Differential Equation (ODE) solvers.
//!
//! This module provides a set of adaptive and non-adaptive ODE solvers.
//! All use Runga-Kutta methods to solve the ODEs.
//!
//! Solvers are adapted from Julia's DifferentialEquations.jl package.
//!
//! Solvers use coefficients from Jim Verner's website:
//! <https://www.sfu.ca/~jverner/>
//! which is awesome
//!
//! For certain solvers, interpolation between the starting and ending
//! points is enabled via separate interpolation functions.
//!

mod adaptive_solvers;
pub mod rk_adaptive;
pub mod rk_adaptive_settings;
pub mod rk_explicit;
mod types;

// NAlgebera bindings for ODE state
mod nalgebra;

pub use rk_adaptive::RKAdaptive;
pub use rk_adaptive_settings::RKAdaptiveSettings;

pub mod solvers {
    pub use super::adaptive_solvers::RKV98NoInterp;
    #[allow(unused)]
    pub use super::adaptive_solvers::RKF45;
    #[allow(unused)]
    pub use super::adaptive_solvers::RKTS54;
    #[allow(unused)]
    pub use super::adaptive_solvers::RKV65;
    #[allow(unused)]
    pub use super::adaptive_solvers::RKV87;
    #[allow(unused)]
    pub use super::adaptive_solvers::RKV98;
    #[allow(unused)]
    pub use super::rk_explicit::Midpoint;
    #[allow(unused)]
    pub use super::rk_explicit::RK4;
}

pub use types::*;

#[cfg(test)]
mod ode_tests;
