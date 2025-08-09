use super::TLE;

use crate::Instant;
use anyhow::{bail, Context, Result};
use nalgebra as na;

use rmpfit::{MPFitter, MPResult};

struct Problem<'a> {
    pub states: &'a [[f64; 6]],
    pub times: &'a [Instant],
    pub epoch: Instant,
}

impl<'a> Problem<'a> {
    fn tle_from_params(&self, p: &[f64]) -> TLE {
        TLE {
            epoch: self.epoch,
            inclination: {
                let mut inc = p[0] % 360.0;
                if inc < 0.0 {
                    inc += 360.0;
                }
                inc
            },
            eccen: p[1] % 360.0,
            raan: {
                let mut raan = p[2] % 360.0;
                if raan < 0.0 {
                    raan += 360.0;
                }
                raan
            },
            arg_of_perigee: {
                let mut w = p[3] % 360.0;
                if w < 0.0 {
                    w += 360.0;
                }
                w
            },
            mean_motion: p[4],
            mean_anomaly: {
                let mut ma = p[5] % 360.0;
                if ma < 0.0 {
                    ma += 360.0;
                }
                ma
            },
            ..Default::default()
        }
    }
}

impl<'a> MPFitter for Problem<'a> {
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
        let mut tle = self.tle_from_params(params);
        let (pteme, _, _) = crate::sgp4::sgp4(&mut tle, self.times);

        for (i, state) in self.states.iter().enumerate() {
            for j in 0..3 {
                deviates[i * 3 + j] = pteme[(j, i)] - state[j];
            }
        }

        Ok(())
    }

    fn number_of_points(&self) -> usize {
        self.states.len() * 3
    }
}

impl TLE {
    pub fn fit_from_states(states: &[[f64; 6]], times: &[Instant], epoch: Instant) -> Result<Self> {
        // Make sure lengths are identical
        if states.len() != times.len() {
            bail!("States and times must have the same length");
        } else if states.is_empty() {
            bail!("States and times must not be empty");
        }

        // Get the minimum time
        let min_time = times.iter().min().unwrap();
        let max_time = times.iter().max().unwrap();
        if epoch < *min_time || epoch > *max_time {
            bail!(
                "Epoch is out of range. Must be between {} and {}",
                min_time,
                max_time
            );
        }

        // Find the point that is closest to the epoch in
        let mut closest_index = 0;
        let mut closest_time = *min_time;
        for (i, time) in times.iter().enumerate() {
            if *time == epoch {
                closest_index = i;
                closest_time = *time;
                break;
            } else if *time < epoch && *time > closest_time {
                closest_index = i;
                closest_time = *time;
            }
        }

        // Get the state
        let closest_state = states[closest_index];
        // Kepler representation
        let mut kepler = crate::kepler::Kepler::from_pv(
            na::Vector3::from_column_slice(&closest_state[..3]),
            na::Vector3::from_column_slice(&closest_state[3..]),
        )
        .context("Could not convert state to Keplerian elements")?;

        // Move Kepler state to epoch
        if (epoch - closest_time).as_microseconds().abs() > 10 {
            kepler = kepler.propagate(&(epoch - closest_time));
        }

        // Create initial guess of parameters from 2-body Kepler
        let mut init_params = [
            kepler.incl.to_degrees(),
            kepler.eccen,
            kepler.raan.to_degrees(),
            kepler.w.to_degrees(),
            kepler.mean_motion() * 60.0 * 60.0 * 24.0 / (2.0 * std::f64::consts::PI),
            kepler.mean_anomaly().to_degrees(),
        ];

        let mut p = Problem {
            states,
            times,
            epoch,
        };
        match p.mpfit(&mut init_params) {
            Ok(_) => Ok(p.tle_from_params(&init_params)),
            Err(e) => {
                bail!("Failed to fit TLE: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_from_states() -> Result<()> {
        let r0 = crate::consts::GEO_R;
        let v0 = (crate::consts::MU_EARTH / r0).sqrt();
        let inc: f64 = 15.0_f64.to_radians();
        let state0 = na::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2022, 5, 16, 12, 0, 0.0);

        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(300.0)),
            &settings,
            None,
        )?;

        let times = (0..30)
            .map(|i| time0 + crate::Duration::from_seconds(i as f64 * 10.0))
            .collect::<Vec<_>>();
        let states = times
            .iter()
            .map(|t| {
                let v = res.interp(t).unwrap();
                // Rotate into TEME frame
                let q = crate::frametransform::qteme2gcrf(t).conjugate();
                let p = q.transform_vector(&na::vector![v[0], v[1], v[2]]);
                let v = q.transform_vector(&na::vector![v[3], v[4], v[5]]);
                [p[0], p[1], p[2], v[0], v[1], v[2]]
            })
            .collect::<Vec<_>>();

        let tle = TLE::fit_from_states(&states, &times, time0)?;
        println!("Fitted TLE: {}", tle);
        Ok(())
    }
}
