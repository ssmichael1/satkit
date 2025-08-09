use super::TLE;

use crate::Instant;
use anyhow::{bail, Context, Result};
use nalgebra as na;

use rmpfit::{MPFitter, MPResult};

struct Problem {
    states: Vec<[f64; 6]>,
    times: Vec<Instant>,
    epoch: Instant,
    tle: TLE,
}

impl TLE {
    pub fn fit_from_states(
        states: &Vec<[f64; 6]>,
        times: &Vec<Instant>,
        epoch: Instant,
    ) -> Result<Self> {
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
        let kepler = crate::kepler::Kepler::from_pv(
            na::Vector3::from_column_slice(&closest_state[..3]),
            na::Vector3::from_column_slice(&closest_state[3..]),
        )
        .context("Could not convert state to Keplerian elements")?;

        // Create a TLE that matches this state
        let mut tle = Self::new();
        tle.epoch = closest_time;
        tle.mean_motion_dot = 0.0;
        tle.mean_motion_dot_dot = 0.0;
        tle.bstar = 0.0;
        tle.inclination = kepler.incl.to_degrees();
        tle.raan = kepler.raan.to_degrees();
        tle.arg_of_perigee = kepler.w.to_degrees();
        tle.mean_motion = kepler.mean_motion() * 60.0 * 60.0 * 24.0 / (2.0 * std::f64::consts::PI);
        tle.eccen = kepler.eccen;
        tle.mean_anomaly = kepler.mean_anomaly().to_degrees();

        // Implement fitting logic here
        Ok(tle)
    }
}
