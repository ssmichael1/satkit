use crate::frametransform::qgcrf2itrf_approx;
use crate::jplephem;
use crate::mathtypes::{Quaternion, Vector3};
use crate::Duration;
use crate::Instant;
use crate::TimeLike;
use crate::SolarSystem;

pub type InterpType = (Quaternion, Vector3, Vector3);

use anyhow::Result;
#[derive(Debug, Clone)]
pub struct Precomputed {
    pub begin: Instant,
    pub end: Instant,
    pub step: f64,
    data: Vec<InterpType>,
}

impl Precomputed {
    pub fn new<T: TimeLike>(begin: &T, end: &T) -> Result<Self> {
        let begin = begin.as_instant();
        let end = end.as_instant();
        let step: f64 = 60.0;

        let (pbegin, pend) = match end > begin {
            true => (
                begin - Duration::from_seconds(240.0),
                end + Duration::from_seconds(240.0),
            ),
            false => (
                end - Duration::from_seconds(240.0),
                begin + Duration::from_seconds(240.0),
            ),
        };

        Ok(Self {
            begin: pbegin,
            end: pend,
            step,
            data: {
                let nsteps: usize =
                    2 + ((pend - pbegin).as_seconds() / step.abs()).ceil() as usize;
                let mut data = Vec::with_capacity(nsteps);
                for idx in 0..nsteps {
                    let t = pbegin + Duration::from_seconds((idx as f64) * step);
                    let q = qgcrf2itrf_approx(&t);
                    let psun = jplephem::geocentric_pos(SolarSystem::Sun, &t)?;
                    let pmoon = jplephem::geocentric_pos(SolarSystem::Moon, &t)?;
                    data.push((q, psun, pmoon));
                }
                data
            },
        })
    }

    pub fn interp<T: TimeLike>(&self, t: &T) -> Result<InterpType> {
        let t = t.as_instant();
        if t < self.begin || t > self.end {
            anyhow::bail!(
                "Precomputed::interp: time {} is outside of precomputed range : {} to {}",
                t,
                self.begin,
                self.end
            );
        }

        let idx = (t - self.begin).as_seconds() / self.step;
        let delta = idx - idx.floor();
        let idx = idx.floor() as usize;

        let q = self.data[idx].0.slerp(&self.data[idx + 1].0, delta);
        let psun = self.data[idx].1 + (self.data[idx + 1].1 - self.data[idx].1) * delta;
        let pmoon = self.data[idx].2 + (self.data[idx + 1].2 - self.data[idx].2) * delta;
        Ok((q, psun, pmoon))
    }
}
