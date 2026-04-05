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

/// Default padding (seconds) applied to each end of the
/// [`Precomputed`] interp-table time range. Chosen to accommodate the
/// default [`GaussJackson8`](super::ode::GaussJackson8) startup, which
/// evaluates the force at `t0 ± 4·gj_step` around the propagation start
/// point. With the default `gj_step_seconds = 60.0`, 240 s of padding
/// exactly covers the startup span. For larger step sizes (or for safety
/// margin) use [`Precomputed::new_padded`] with a custom value.
pub const DEFAULT_PADDING_SECS: f64 = 240.0;

impl Precomputed {
    /// Create a precomputed interp table with default step (60 s) and
    /// default padding ([`DEFAULT_PADDING_SECS`]). Suitable for any
    /// satkit integrator *except* [`GaussJackson8`](super::ode::GaussJackson8)
    /// with `gj_step_seconds > 60`.
    pub fn new<T: TimeLike>(begin: &T, end: &T) -> Result<Self> {
        Self::new_padded(begin, end, 60.0, DEFAULT_PADDING_SECS)
    }

    /// Create a precomputed interp table with a custom interpolation step
    /// and the default padding.
    pub fn new_with_step<T: TimeLike>(begin: &T, end: &T, step_secs: f64) -> Result<Self> {
        Self::new_padded(begin, end, step_secs, DEFAULT_PADDING_SECS)
    }

    /// Create a precomputed interp table with both a custom interpolation
    /// step and custom bounds padding.
    ///
    /// The `padding_secs` parameter controls how far beyond the
    /// `[min(begin, end), max(begin, end)]` interval the interp table
    /// extends on each end. The padding must be large enough to cover
    /// any force-model evaluations the integrator makes outside the
    /// nominal propagation interval — in particular,
    /// [`GaussJackson8`](super::ode::GaussJackson8)'s startup procedure
    /// evaluates the force at `t0 ± 4·gj_step` around the starting epoch,
    /// so `padding_secs` must be at least `4·gj_step_seconds` (plus a
    /// small margin for floating-point safety).
    ///
    /// For convenience, [`PropSettings::required_precompute_padding`](super::PropSettings::required_precompute_padding)
    /// computes the correct value from a settings object.
    pub fn new_padded<T: TimeLike>(
        begin: &T,
        end: &T,
        step_secs: f64,
        padding_secs: f64,
    ) -> Result<Self> {
        let begin = begin.as_instant();
        let end = end.as_instant();
        let step: f64 = step_secs;
        let pad = Duration::from_seconds(padding_secs.max(0.0));

        let (pbegin, pend) = match end > begin {
            true => (begin - pad, end + pad),
            false => (end - pad, begin + pad),
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
