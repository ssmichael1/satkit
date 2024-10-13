use crate::frametransform::qgcrf2itrf_approx;
use crate::jplephem;
use crate::types::*;
use crate::utils::SKResult;
use crate::AstroTime;
use crate::Duration;
use crate::SolarSystem;

pub type InterpType = (Quaternion, Vector3, Vector3);

#[derive(Debug, Clone)]
pub struct Precomputed {
    pub start: AstroTime,
    pub stop: AstroTime,
    pub step: f64,
    data: Vec<InterpType>,
}

impl Precomputed {
    pub fn new(start: &AstroTime, stop: &AstroTime) -> SKResult<Precomputed> {
        let step: f64 = 60.0;

        let (pstart, pstop) = match stop > start {
            true => (start, stop),
            false => (stop, start),
        };

        Ok(Precomputed {
            start: pstart.clone(),
            stop: pstop.clone(),
            step: step,
            data: {
                let nsteps: usize = 2 + ((pstop - pstart).seconds() / step.abs()).ceil() as usize;
                let mut data = Vec::new();
                data.reserve(nsteps);
                for idx in 0..nsteps {
                    let t = *pstart + Duration::Seconds((idx as f64) * step);
                    let q = qgcrf2itrf_approx(&t);
                    let psun = jplephem::geocentric_pos(SolarSystem::Sun, &t)?;
                    let pmoon = jplephem::geocentric_pos(SolarSystem::Moon, &t)?;
                    data.push((q, psun, pmoon));
                }
                data
            },
        })
    }

    pub fn interp(&self, t: &AstroTime) -> SKResult<InterpType> {
        if *t < self.start || *t > self.stop {
            return Err("Precomputed::interp: time is outside of precomputed range".into());
        }

        let idx = (t - self.start).seconds() / self.step;
        let delta = idx - idx.floor();
        let idx = idx.floor() as usize;

        let q = self.data[idx].0.slerp(&self.data[idx + 1].0, delta);
        let psun = self.data[idx].1 + (self.data[idx + 1].1 - self.data[idx].1) * delta;
        let pmoon = self.data[idx].2 + (self.data[idx + 1].2 - self.data[idx].2) * delta;
        Ok((q, psun, pmoon))
    }
}
