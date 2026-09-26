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

use crate::{Instant, TimeScale};

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
/// - `epoch_days_1950`: element-set epoch in days since 1949-12-31 00:00 UTC
///   (Vallado's "jan 0, 1950"), i.e. UTC MJD − 33281. Carried relative to
///   1950 rather than as a full Julian date so the f64 keeps sub-microsecond
///   resolution (a full JD quantizes at ~40 µs).
/// - `no`: mean motion in radians / minute
/// - `ndot`: 1st derivative of mean motion in radians / minute^2
/// - `nddot`: 2nd derivative of mean motion in radians / minute^3
/// - angles are radians
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGP4InitArgs {
    pub epoch_days_1950: f64,
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
    /// the epoch (a UTC element-set epoch, as TLEs and OMMs define it), mean
    /// motion and its derivatives in rev/day (+ per day, per day²), and the
    /// four angles in degrees.
    ///
    /// This performs the epoch, rev/day → rad/min and degree → radian
    /// conversions shared by every SGP4 source (TLE, CCSDS OMM), so the
    /// conversion factors live in exactly one place.
    #[allow(clippy::too_many_arguments)]
    pub fn from_mean_elements(
        epoch: Instant,
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
        // MJD of Vallado's SGP4 epoch origin, 1949-12-31 00:00 UTC (JD 2433281.5)
        const MJD_1950_JAN0: f64 = 33281.0;

        Self {
            epoch_days_1950: epoch.as_mjd_with_scale(TimeScale::UTC) - MJD_1950_JAN0,
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

    /// Bit patterns of every field, for an exact (NaN-safe) cache key.
    const fn to_bits(self) -> [u64; 10] {
        [
            self.epoch_days_1950.to_bits(),
            self.bstar.to_bits(),
            self.ndot.to_bits(),
            self.nddot.to_bits(),
            self.ecco.to_bits(),
            self.argpo.to_bits(),
            self.inclo.to_bits(),
            self.mo.to_bits(),
            self.no.to_bits(),
            self.nodeo.to_bits(),
        ]
    }
}

/// Everything a cached [`SatRec`] was initialized from: the gravity model,
/// the ops mode and the init arguments. [`sgp4_full`] reuses a cached
/// `SatRec` only when its key matches the current call exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub(crate) struct SatRecKey {
    gravconst: GravConst,
    opsmode: OpsMode,
    args: [u64; 10],
}

impl SatRecKey {
    const fn new(gravconst: GravConst, opsmode: OpsMode, args: SGP4InitArgs) -> Self {
        Self {
            gravconst,
            opsmode,
            args: args.to_bits(),
        }
    }
}

/// Source of SGP4 mean elements (e.g., TLE, CCSDS OMM) that can be propagated.
///
/// Implementations are responsible for any unit/time-system conversions needed
/// to produce `SGP4InitArgs`. [`sgp4_full`] calls
/// [`sgp4_init_args`](Self::sgp4_init_args) on every propagation and
/// re-initializes the cached `SatRec` whenever the arguments, gravity model or
/// ops mode differ from those it was built with, so editing a source's
/// elements never propagates a stale initialization.
pub trait SGP4Source {
    /// The element-set epoch as a satkit `Instant`.
    fn epoch(&self) -> Instant;

    /// Mutable access to an optional cached `SatRec`. The cache stores its own
    /// key; implementations just hold the value.
    fn satrec_mut(&mut self) -> &mut Option<SatRec>;

    /// Produce canonical SGP4 initialization arguments.
    fn sgp4_init_args(&self) -> Result<SGP4InitArgs>;
}
