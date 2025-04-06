use crate::frametransform::qgcrf2itrf_approx;
use crate::jplephem;
use crate::types::{Quaternion, Vector3};
use crate::Duration;
use crate::Instant;
use crate::SolarSystem;

pub type InterpType = (Quaternion, Vector3, Vector3);

use anyhow::Result;
#[derive(Debug, Clone)]
pub struct Precomputed {
    pub start: Instant,
    pub stop: Instant,
    pub step: f64,
    data: Vec<InterpType>,
}

impl Precomputed {
    pub fn new(start: &Instant, stop: &Instant) -> Result<Self> {
        let step: f64 = 60.0;

        let (pstart, pstop) = match stop > start {
            true => (
                start - Duration::from_seconds(240.0),
                stop + Duration::from_seconds(240.0),
            ),
            false => (
                stop - Duration::from_seconds(240.0),
                start + Duration::from_seconds(240.0),
            ),
        };

        Ok(Self {
            start: pstart,
            stop: pstop,
            step,
            data: {
                let nsteps: usize =
                    2 + ((pstop - pstart).as_seconds() / step.abs()).ceil() as usize;
                let mut data = Vec::with_capacity(nsteps);
                for idx in 0..nsteps {
                    let t = pstart + Duration::from_seconds((idx as f64) * step);
                    let q = qgcrf2itrf_approx(&t);
                    let psun = jplephem::geocentric_pos(SolarSystem::Sun, &t)?;
                    let pmoon = jplephem::geocentric_pos(SolarSystem::Moon, &t)?;
                    data.push((q, psun, pmoon));
                }
                data
            },
        })
    }

    pub fn interp(&self, t: &Instant) -> Result<InterpType> {
        if *t < self.start || *t > self.stop {
            anyhow::bail!(
                "Precomputed::interp: time {} is outside of precomputed range : {} to {}",
                *t,
                self.start,
                self.stop
            );
        }

        let idx = (t - self.start).as_seconds() / self.step;
        let delta = idx - idx.floor();
        let idx = idx.floor() as usize;

        let q = self.data[idx].0.slerp(&self.data[idx + 1].0, delta);
        let psun = self.data[idx].1 + (self.data[idx + 1].1 - self.data[idx].1) * delta;
        let pmoon = self.data[idx].2 + (self.data[idx + 1].2 - self.data[idx].2) * delta;
        Ok((q, psun, pmoon))
    }
}
