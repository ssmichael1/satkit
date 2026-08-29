///
/// A pure-rust implementation of SGP4
///
/// manually and painstakingly converted from C++
/// in as straightforward a manner as possible
///
/// Note: generates correct results that match test vectors
/// provided by C++ implementation
///
/// Original C++ code by David Vallado, et al. The algorithm, the TEME frame
/// and the test vectors this port is verified against are described in
/// Vallado, Crawford, Hujsak & Kelso, "Revisiting Spacetrack Report #3",
/// AIAA 2006-6753, <https://doi.org/10.2514/6.2006-6753>
/// (<https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>);
/// the original model is Hoots & Roehrich, Spacetrack Report No. 3 (1980).
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
mod error;
mod getgravconst;
mod initl;
pub mod satrec;
mod sgp4_impl;
mod sgp4_lowlevel;
mod sgp4init;

pub use error::{Error, Result, SGP4Error};
pub use sgp4_impl::sgp4;
pub use sgp4_impl::sgp4_full;
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

impl SGP4InitArgs {
    /// Build init args from mean elements in their natural catalog units:
    /// mean motion and its derivatives in rev/day (+ per day, per day²), and
    /// the four angles in degrees.
    ///
    /// This performs the rev/day → rad/min and degree → radian conversions
    /// shared by every SGP4 source (TLE, CCSDS OMM), so the conversion factors
    /// live in exactly one place.
    #[allow(clippy::too_many_arguments)]
    pub fn from_mean_elements(
        jdsatepoch: f64,
        bstar: f64,
        mean_motion: f64,
        mean_motion_dot: f64,
        mean_motion_ddot: f64,
        eccen: f64,
        inclination_deg: f64,
        raan_deg: f64,
        arg_of_perigee_deg: f64,
        mean_anomaly_deg: f64,
    ) -> Self {
        use std::f64::consts::PI;

        const TWOPI: f64 = PI * 2.0;

        Self {
            jdsatepoch,
            bstar,
            no: mean_motion / (1440.0 / TWOPI),
            ndot: mean_motion_dot / (1440.0 * 1440.0 / TWOPI),
            nddot: mean_motion_ddot / (1440.0 * 1440.0 * 1440.0 / TWOPI),
            ecco: eccen,
            inclo: inclination_deg.to_radians(),
            nodeo: raan_deg.to_radians(),
            argpo: arg_of_perigee_deg.to_radians(),
            mo: mean_anomaly_deg.to_radians(),
        }
    }
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
    fn sgp4_init_args(&self) -> Result<SGP4InitArgs>;
}
