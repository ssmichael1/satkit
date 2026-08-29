use crate::frametransform::{qgcrf2itrf, qitrf2gcrf_slow_parts, qtirs2cirs};
use crate::jplephem;
use crate::mathtypes::{Quaternion, Vector3};
use crate::Duration;
use crate::Instant;
use crate::SolarSystem;
use crate::TimeLike;

/// `(q_gcrf2itrf, sun_pos_gcrf, moon_pos_gcrf, sun_vel_gcrf)` — SI units.
/// The Sun velocity feeds the geodesic-precession term of the relativistic
/// correction (see [`super::relativity`]).
/// One sample of the quantities the force model needs at a given time,
/// as stored in (and interpolated from) a [`Precomputed`] table.
///
/// Introduced as a named struct (replacing a tuple) so that adding a field
/// — as the Sun velocity was for the geodesic-precession term — is not a
/// breaking change for callers that destructure it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpSample {
    /// Rotation from GCRF to ITRF (full IAU 2006/2000A chain).
    pub qgcrf2itrf: Quaternion,
    /// Geocentric Sun position in GCRF, meters.
    pub sun_pos_gcrf: Vector3,
    /// Geocentric Moon position in GCRF, meters.
    pub moon_pos_gcrf: Vector3,
    /// Geocentric Sun velocity in GCRF, m/s (used by the geodesic term of
    /// the relativistic correction).
    pub sun_vel_gcrf: Vector3,
}

/// Alias kept for source compatibility with code that named the old tuple.
pub type InterpType = InterpSample;

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

/// Sampling interval (seconds) for the slowly varying precession-nutation
/// and polar-motion factors of the frame rotation stored in the table.
const SLOW_STEP_SECS: f64 = 3600.0;

/// Upper bound on the number of entries a [`Precomputed`] table may hold.
///
/// Each entry is ~104 bytes, so this is ≈1.7 GB — about 30 years at the
/// default 60 s step. A span/step combination needing more returns
/// [`Error::PrecomputeTooLarge`] *before* anything is allocated. (An
/// unbounded table could previously consume all memory for a tiny step or
/// a very long span, or silently wrap to a one-entry table for a
/// denormal step.)
pub const MAX_PRECOMPUTE_ENTRIES: usize = 16_777_216;

/// Number of table entries for a span at a given step, or an error if the
/// count is not representable or exceeds [`MAX_PRECOMPUTE_ENTRIES`].
fn table_len(span_secs: f64, step_secs: f64) -> Result<usize> {
    let n = (span_secs / step_secs).ceil();
    // `as` would saturate a non-finite or huge value and the `+ 2` below
    // could then wrap; check in f64 first.
    if !n.is_finite() || n < 0.0 || n > MAX_PRECOMPUTE_ENTRIES as f64 {
        return Err(Error::PrecomputeTooLarge {
            entries: if n.is_finite() {
                (n as u64).saturating_add(2)
            } else {
                u64::MAX
            },
            max: MAX_PRECOMPUTE_ENTRIES,
        });
    }
    let entries = n as usize + 2;
    if entries > MAX_PRECOMPUTE_ENTRIES {
        return Err(Error::PrecomputeTooLarge {
            entries: entries as u64,
            max: MAX_PRECOMPUTE_ENTRIES,
        });
    }
    Ok(entries)
}

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
        if !padding_secs.is_finite() {
            return Err(Error::InvalidPrecomputePadding {
                padding: padding_secs,
            });
        }
        let step: f64 = step_secs;
        let pad = Duration::from_seconds(padding_secs.max(0.0));

        let (pbegin, pend) = match end > begin {
            true => (begin - pad, end + pad),
            false => (end - pad, begin + pad),
        };

        // Size the tables before touching the ephemeris or allocating.
        let nsteps: usize = table_len((pend - pbegin).as_seconds(), step)?;
        // The last fine point may lie up to `step` beyond `pend`; size the
        // slow table from it so the slerp fraction stays within [0, 1].
        let nslow: usize = table_len(((nsteps - 1) as f64) * step, SLOW_STEP_SECS)?;

        Ok(Self {
            begin: pbegin,
            end: pend,
            step,
            data: {
                // Fail fast on a span the JPL ephemeris cannot cover before
                // doing any of the (comparatively expensive) frame work below.
                jplephem::geocentric_pos(SolarSystem::Sun, &pbegin)?;
                jplephem::geocentric_pos(SolarSystem::Sun, &pend)?;

                // The frame chain below needs the IERS nutation tables and an
                // EOP table. Load the former now so a missing data file is an
                // error here rather than a panic inside the force model, and
                // refuse to build a table with no EOP at all — zero polar
                // motion / UT1-UTC would silently bias the propagation by
                // metres. A span past the *end* of the EOP table is allowed
                // (the last row is held constant; `earth_orientation_params`
                // warns once), and `PropSettings::require_eop_coverage`
                // turns that into an error at `propagate()`.
                crate::frametransform::ierstable::preload()?;
                if crate::earth_orientation_params::coverage().is_none() {
                    return Err(Error::EopUnavailable);
                }

                // The GCRF→ITRF rotation is the full IAU 2006/2000A chain
                // (precession-nutation with EOP dX/dY, Earth rotation angle,
                // polar motion) — the same as `frametransform::qgcrf2itrf`.
                // Evaluating that directly at every table point costs
                // ~20 µs each (the nutation series), so the slowly varying
                // factors are sampled every `SLOW_STEP_SECS` and slerped,
                // while the fast Earth-rotation factor is computed exactly
                // at each point. Interpolation error is far below a
                // microarcsecond; see `test_table_matches_full_transform`.
                let slow: Vec<(Quaternion, Quaternion)> = (0..nslow)
                    .map(|i| {
                        let t = pbegin + Duration::from_seconds((i as f64) * SLOW_STEP_SECS);
                        qitrf2gcrf_slow_parts(&t)
                    })
                    .collect();
                let mut data = Vec::with_capacity(nsteps);
                for idx in 0..nsteps {
                    let dt = (idx as f64) * step;
                    let t = pbegin + Duration::from_seconds(dt);
                    let s = dt / SLOW_STEP_SECS;
                    let si = (s.floor() as usize).min(nslow - 2);
                    let frac = s - si as f64;
                    let q_cirs2gcrs = slow[si].0.slerp(&slow[si + 1].0, frac);
                    let q_itrf2tirs = slow[si].1.slerp(&slow[si + 1].1, frac);
                    let q = (q_cirs2gcrs * qtirs2cirs(&t) * q_itrf2tirs).conjugate();
                    let (psun, vsun) = jplephem::geocentric_state(SolarSystem::Sun, &t)?;
                    let pmoon = jplephem::geocentric_pos(SolarSystem::Moon, &t)?;
                    data.push(InterpSample {
                        qgcrf2itrf: q,
                        sun_pos_gcrf: psun,
                        moon_pos_gcrf: pmoon,
                        sun_vel_gcrf: vsun,
                    });
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
        let q = qgcrf2itrf(&t);
        match (
            jplephem::geocentric_state(SolarSystem::Sun, &t),
            jplephem::geocentric_pos(SolarSystem::Moon, &t),
        ) {
            (Ok((psun, vsun)), Ok(pmoon)) => InterpSample {
                qgcrf2itrf: q,
                sun_pos_gcrf: psun,
                moon_pos_gcrf: pmoon,
                sun_vel_gcrf: vsun,
            },
            _ => {
                let edge = if t < self.begin {
                    self.data.first()
                } else {
                    self.data.last()
                };
                edge.copied().unwrap_or(InterpSample {
                    qgcrf2itrf: Quaternion::identity(),
                    sun_pos_gcrf: Vector3::zeros(),
                    moon_pos_gcrf: Vector3::zeros(),
                    sun_vel_gcrf: Vector3::zeros(),
                })
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

        let (a, b) = (&self.data[idx], &self.data[idx + 1]);
        Ok(InterpSample {
            qgcrf2itrf: a.qgcrf2itrf.slerp(&b.qgcrf2itrf, delta),
            sun_pos_gcrf: a.sun_pos_gcrf + (b.sun_pos_gcrf - a.sun_pos_gcrf) * delta,
            moon_pos_gcrf: a.moon_pos_gcrf + (b.moon_pos_gcrf - a.moon_pos_gcrf) * delta,
            sun_vel_gcrf: a.sun_vel_gcrf + (b.sun_vel_gcrf - a.sun_vel_gcrf) * delta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tabulated GCRF→ITRF rotation must match the full
    /// `frametransform::qgcrf2itrf` chain to well under a milliarcsecond.
    /// (Regression: the table was once built from the ~1 arcsec IAU-76/FK5
    /// approximation, which tilted the gravity field enough to drift a LEO
    /// orbit by ~50 m over 7 days relative to GMAT.)
    #[test]
    fn test_table_matches_full_transform() {
        let t0 = Instant::from_datetime(2023, 5, 16, 20, 0, 0.0).unwrap();
        // (table, span in seconds, label): forward 7 days at the default
        // step; reversed span; a sub-hour span (single slow interval); a
        // coarse 600 s step.
        let week = 7.0 * 86400.0;
        let tables = [
            (
                Precomputed::new(&t0, &(t0 + Duration::from_seconds(week))).unwrap(),
                week,
                "forward",
            ),
            (
                Precomputed::new(&(t0 + Duration::from_seconds(week)), &t0).unwrap(),
                week,
                "reversed",
            ),
            (
                Precomputed::new(&t0, &(t0 + Duration::from_seconds(1800.0))).unwrap(),
                1800.0,
                "sub-hour",
            ),
            (
                Precomputed::new_with_step(&t0, &(t0 + Duration::from_seconds(week)), 600.0)
                    .unwrap(),
                week,
                "600 s step",
            ),
        ];
        for (pc, span, label) in &tables {
            let mut worst = 0.0f64;
            for k in 0..2000 {
                // Irrational stride so samples fall between table points.
                let dt = (k as f64 * 302.17 + 7.3) % span;
                let t = t0 + Duration::from_seconds(dt);
                let q_tab = pc.interp(&t).unwrap().qgcrf2itrf;
                let q_full = qgcrf2itrf(&t);
                let dq = q_tab.conjugate() * q_full;
                let a = dq.to_axis_angle().1.abs();
                worst = worst.max(a.min(std::f64::consts::TAU - a));
            }
            // Measured ~3e-9 rad at 60 s (the table slerp falls back to
            // normalized lerp for such small rotations, error ~θ³/48) and
            // ~3e-6·(600/60)³ ≈ 3e-6 rad would be nlerp at 600 s, but slerp
            // proper is used there. 1e-8 rad = 2 mas; the approximation this
            // replaced was ~4e-6 rad.
            assert!(
                worst < 1e-8,
                "{label}: table vs full transform: {worst:.3e} rad"
            );
        }
    }

    #[test]
    fn test_size_cap_and_bad_inputs() {
        let t0 = Instant::from_datetime(2023, 5, 16, 20, 0, 0.0).unwrap();
        let t1 = t0 + Duration::from_seconds(3600.0);

        // A tiny step would need ~3.8e9 entries: must error, and must do so
        // before allocating (previously this consumed >6 GB and was killed).
        let start = std::time::Instant::now();
        assert!(matches!(
            Precomputed::new_with_step(&t0, &t1, 1e-6),
            Err(Error::PrecomputeTooLarge { .. })
        ));
        assert!(start.elapsed().as_secs() < 2, "size check must be cheap");

        // A denormal step used to overflow the entry count and wrap to a
        // silent one-entry table.
        assert!(matches!(
            Precomputed::new_with_step(&t0, &t1, 1e-300),
            Err(Error::PrecomputeTooLarge { .. })
        ));

        // Non-finite padding is rejected rather than treated as zero / huge.
        assert!(matches!(
            Precomputed::new_padded(&t0, &t1, 60.0, f64::NAN),
            Err(Error::InvalidPrecomputePadding { .. })
        ));
        assert!(matches!(
            Precomputed::new_padded(&t0, &t1, 60.0, f64::INFINITY),
            Err(Error::InvalidPrecomputePadding { .. })
        ));

        // Ordinary tables are unaffected.
        let week = Precomputed::new(&t0, &(t0 + Duration::from_days(7.0))).unwrap();
        assert!(week.interp(&(t0 + Duration::from_days(3.5))).is_ok());
        assert_eq!(table_len(7.0 * 86400.0, 60.0).unwrap(), 2 + 7 * 1440);
    }

    /// A span past the end of the EOP table still builds (the last EOP row
    /// is held constant, with a one-time warning); the IERS tables preload.
    #[test]
    fn test_past_eop_coverage_builds() {
        crate::frametransform::ierstable::preload().expect("IERS tables present in tests");
        let cov = crate::earth_orientation_params::coverage().expect("EOP loaded in tests");
        let t0 = cov.last + Duration::from_days(30.0);
        let t1 = t0 + Duration::from_days(1.0);
        let pc = Precomputed::new(&t0, &t1).unwrap();
        let s = pc.interp(&(t0 + Duration::from_seconds(3600.0))).unwrap();
        assert!(s.qgcrf2itrf.to_axis_angle().1.is_finite());
    }

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
        assert_eq!(a.sun_pos_gcrf.as_slice(), b.sun_pos_gcrf.as_slice());

        // Out of range: interp errors, but interp_or_compute must return
        // finite values (regression: the force model unwrapped interp and
        // panicked when an adaptive integrator probed past the padding)
        let tout = t1 + Duration::from_seconds(86400.0);
        assert!(pc.interp(&tout).is_err());
        let s = pc.interp_or_compute(&tout);
        assert!(s.sun_pos_gcrf.as_slice().iter().all(|v| v.is_finite()));
        assert!(s.moon_pos_gcrf.as_slice().iter().all(|v| v.is_finite()));
        assert!(s.sun_vel_gcrf.as_slice().iter().all(|v| v.is_finite()));
    }
}
