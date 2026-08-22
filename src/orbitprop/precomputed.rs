use crate::frametransform::qgcrf2itrf_approx;
use crate::jplephem;
use crate::mathtypes::{Quaternion, Vector3};
use crate::Duration;
use crate::Instant;
use crate::SolarSystem;
use crate::TimeLike;

pub type InterpType = (Quaternion, Vector3, Vector3);

use super::error::{Error, Result};
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
        if !step_secs.is_finite() || step_secs <= 0.0 {
            return Err(Error::InvalidPrecomputeStep { step: step_secs });
        }
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
                let nsteps: usize = 2 + ((pend - pbegin).as_seconds() / step).ceil() as usize;
                // Cap the up-front reservation: an absurd span would otherwise
                // attempt one giant allocation here; growing instead lets the
                // ephemeris range check in the loop below error out first.
                let mut data = Vec::with_capacity(nsteps.min(1 << 22));
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

    /// Interpolate if `t` falls within the table range; otherwise fall back
    /// to computing the values directly (exact, just slower than the table).
    ///
    /// This method cannot fail, which makes it safe to call from force-model
    /// evaluations where the integrator may probe slightly outside the
    /// padded propagation interval (e.g. an adaptive integrator's initial
    /// step-size heuristic). In the pathological case where the direct
    /// ephemeris lookup also fails (a time outside the JPL ephemeris span),
    /// the nearest table-edge sample is returned.
    pub fn interp_or_compute<T: TimeLike>(&self, t: &T) -> InterpType {
        let t = t.as_instant();
        if let Ok(v) = self.interp(&t) {
            return v;
        }
        let q = qgcrf2itrf_approx(&t);
        match (
            jplephem::geocentric_pos(SolarSystem::Sun, &t),
            jplephem::geocentric_pos(SolarSystem::Moon, &t),
        ) {
            (Ok(psun), Ok(pmoon)) => (q, psun, pmoon),
            _ => {
                let edge = if t < self.begin {
                    self.data.first()
                } else {
                    self.data.last()
                };
                edge.copied().unwrap_or((
                    Quaternion::identity(),
                    Vector3::zeros(),
                    Vector3::zeros(),
                ))
            }
        }
    }

    pub fn interp<T: TimeLike>(&self, t: &T) -> Result<InterpType> {
        let t = t.as_instant();
        if t < self.begin || t > self.end {
            return Err(Error::PrecomputedOutOfRange {
                time: t.to_string(),
                begin: self.begin.to_string(),
                end: self.end.to_string(),
            });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_step_errors() {
        let t0 = Instant::from_date(2015, 3, 20).unwrap();
        let t1 = t0 + Duration::from_seconds(3600.0);
        // Zero step would compute usize::MAX steps and abort on allocation
        assert!(matches!(
            Precomputed::new_with_step(&t0, &t1, 0.0),
            Err(Error::InvalidPrecomputeStep { .. })
        ));
        // Negative step silently built a table covering the wrong range
        assert!(Precomputed::new_with_step(&t0, &t1, -60.0).is_err());
        assert!(Precomputed::new_with_step(&t0, &t1, f64::NAN).is_err());
    }

    #[test]
    fn test_interp_or_compute_out_of_range() {
        let t0 = Instant::from_date(2015, 3, 20).unwrap();
        let t1 = t0 + Duration::from_seconds(3600.0);
        let pc = Precomputed::new(&t0, &t1).unwrap();

        // In range: matches the table interpolation
        let tin = t0 + Duration::from_seconds(100.0);
        let a = pc.interp(&tin).unwrap();
        let b = pc.interp_or_compute(&tin);
        assert_eq!(a.1.as_slice(), b.1.as_slice());

        // Out of range: interp errors, but interp_or_compute must return
        // finite values (regression: the force model unwrapped interp and
        // panicked when an adaptive integrator probed past the padding)
        let tout = t1 + Duration::from_seconds(86400.0);
        assert!(pc.interp(&tout).is_err());
        let (_q, psun, pmoon) = pc.interp_or_compute(&tout);
        assert!(psun.as_slice().iter().all(|v| v.is_finite()));
        assert!(pmoon.as_slice().iter().all(|v| v.is_finite()));
    }
}
