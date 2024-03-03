///
/// A pure-rust implementation of SGP4
///
/// manually and painstakingly converted from C++
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

pub use sgp4::sgp4;
pub use sgp4::sgp4_full;
pub use sgp4::SGP4Error;
pub use sgp4::SGP4Result;
pub use sgp4::SGP4State;

mod dpper;
mod dscom;
mod dsinit;
mod dspace;
mod getgravconst;
mod initl;
pub mod satrec;
pub mod sgp4;
mod sgp4_lowlevel;
mod sgp4init;
