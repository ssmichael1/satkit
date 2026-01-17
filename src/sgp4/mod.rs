///
/// A pure-rust implementation of SGP4
///
/// manually and painstakingly converted from C++
/// in as straightforward a manner as possible
///
/// Note: generates correct results that match test vectors
/// provided by C++ implementation
///
/// Original C++ code by David Vallado, et al.
/// <https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf/>
///
///
pub use self::satrec::SatRec;

use crate::Instant;

#[derive(PartialEq, PartialOrd, Clone, Debug, Eq, Copy)]
pub enum GravConst {
    WGS72,
    WGS72OLD,
    WGS84,
}

#[derive(PartialEq, PartialOrd, Clone, Debug, Eq, Copy)]
pub enum OpsMode {
    AFSPC,
    IMPROVED,
}

mod dpper;
mod dscom;
mod dsinit;
mod dspace;
mod getgravconst;
mod initl;
pub mod satrec;
mod sgp4_impl;
mod sgp4_lowlevel;
mod sgp4init;

pub use sgp4_impl::sgp4;
pub use sgp4_impl::sgp4_full;
pub use sgp4_impl::SGP4Error;
pub use sgp4_impl::SGP4State;

/// Canonical inputs required to initialize an SGP4 `SatRec`.
///
/// Units match Vallado's `sgp4init` inputs:
/// - `jdsatepoch`: Julian date (UTC) of the element set epoch
/// - `no`: mean motion in radians / minute
/// - `ndot`: 1st derivative of mean motion in radians / minute^2
/// - `nddot`: 2nd derivative of mean motion in radians / minute^3
/// - angles are radians
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGP4InitArgs {
    pub jdsatepoch: f64,
    pub bstar: f64,
    pub ndot: f64,
    pub nddot: f64,
    pub ecco: f64,
    pub argpo: f64,
    pub inclo: f64,
    pub mo: f64,
    pub no: f64,
    pub nodeo: f64,
}

/// Source of SGP4 mean elements (e.g., TLE, CCSDS OMM) that can be propagated.
///
/// Implementations are responsible for any unit/time-system conversions needed
/// to produce `SGP4InitArgs`.
pub trait SGP4Source {
    /// The element-set epoch as a satkit `Instant`.
    fn epoch(&self) -> Instant;

    /// Mutable access to an optional cached `SatRec`.
    fn satrec_mut(&mut self) -> &mut Option<SatRec>;

    /// Produce canonical SGP4 initialization arguments.
    fn sgp4_init_args(&self) -> anyhow::Result<SGP4InitArgs>;
}
