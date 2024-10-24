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
pub use sgp4_impl::SGP4Result;
pub use sgp4_impl::SGP4State;
