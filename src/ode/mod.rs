pub mod rk_adaptive;
pub mod rk_adaptive_settings;
pub mod rk_explicit;
//mod rkf45;
mod rkts54;
mod rkv65;
mod rkv65_table;
mod rkv87;
mod rkv87_table;
mod rkv98;
mod rkv98_nointerp;
mod rkv98_nointerp_table;
mod rkv98_table;
pub mod types;

// NAlgebera bindings for ODE state
mod nalgebra;

pub use rk_adaptive::RKAdaptive;
pub use rk_adaptive_settings::RKAdaptiveSettings;

pub mod solvers {
    #[allow(unused)]
    pub use super::rk_explicit::Midpoint;
    #[allow(unused)]
    pub use super::rk_explicit::RK4;
    #[allow(unused)]
    pub use super::rkts54::RKTS54;
    #[allow(unused)]
    pub use super::rkv65::RKV65;
    #[allow(unused)]
    pub use super::rkv87::RKV87;
    pub use super::rkv98::RKV98;
    pub use super::rkv98_nointerp::RKV98NoInterp;
}

pub use types::*;
