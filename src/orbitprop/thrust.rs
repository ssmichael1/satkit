use crate::frametransform;
use crate::mathtypes::*;
use crate::Frame;
use crate::Instant;

/// A constant thrust acceleration over a time window
#[derive(Debug, Clone)]
pub struct ContinuousThrust {
    /// Acceleration vector in the specified frame [m/s^2]
    pub accel: Vector3,
    /// Coordinate frame for the acceleration vector
    pub frame: Frame,
    /// Start time of the thrust arc
    pub start: Instant,
    /// End time of the thrust arc
    pub end: Instant,
}

impl ContinuousThrust {
    pub fn new(accel: Vector3, frame: Frame, start: Instant, end: Instant) -> Self {
        Self {
            accel,
            frame,
            start,
            end,
        }
    }

    /// Check if thrust is active at the given time
    pub fn is_active(&self, time: &Instant) -> bool {
        *time >= self.start && *time <= self.end
    }

    /// Compute thrust acceleration in GCRF at the given time and state.
    ///
    /// Supported frames: [`Frame::GCRF`], [`Frame::RTN`], [`Frame::NTW`],
    /// [`Frame::LVLH`]. Use NTW for thrust-along-velocity scenarios (most
    /// electric-propulsion mission profiles); use RTN for position-tied
    /// burn components; use LVLH if you're porting GN&C code written in
    /// the crewed-spaceflight / body-pointing convention.
    ///
    /// Returns `None` if thrust is not active at this time.
    pub fn accel_gcrf(
        &self,
        time: &Instant,
        pos_gcrf: &Vector3,
        vel_gcrf: &Vector3,
    ) -> Option<Vector3> {
        if !self.is_active(time) {
            return None;
        }
        Some(match self.frame {
            Frame::GCRF => self.accel,
            Frame::RTN => {
                let dcm = frametransform::rtn_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.accel
            }
            Frame::NTW => {
                let dcm = frametransform::ntw_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.accel
            }
            Frame::LVLH => {
                let dcm = frametransform::lvlh_to_gcrf(pos_gcrf, vel_gcrf);
                dcm * self.accel
            }
            Frame::ITRF
            | Frame::TIRS
            | Frame::CIRS
            | Frame::TEME
            | Frame::EME2000
            | Frame::ICRF => panic!(
                "Unsupported frame for thrust: {}. Must be GCRF, RTN, NTW, or LVLH",
                self.frame
            ),
        })
    }
}

/// A collection of thrust arcs
///
/// This is the primary thrust type used by the propagator.
/// It holds a list of `ContinuousThrust` entries and evaluates
/// the total thrust acceleration at any given time.
#[derive(Debug, Clone, Default)]
pub struct ThrustProfile {
    pub thrusts: Vec<ContinuousThrust>,
}

impl ThrustProfile {
    pub fn new(thrusts: Vec<ContinuousThrust>) -> Self {
        Self { thrusts }
    }

    /// Compute total thrust acceleration in GCRF at the given time and state
    ///
    /// Returns None if no thrust arcs are active at this time
    pub fn accel_gcrf(
        &self,
        time: &Instant,
        pos_gcrf: &Vector3,
        vel_gcrf: &Vector3,
    ) -> Option<Vector3> {
        let mut total = Vector3::zeros();
        let mut active = false;
        for t in &self.thrusts {
            if let Some(a) = t.accel_gcrf(time, pos_gcrf, vel_gcrf) {
                total += a;
                active = true;
            }
        }
        active.then_some(total)
    }

    pub fn is_empty(&self) -> bool {
        self.thrusts.is_empty()
    }
}
