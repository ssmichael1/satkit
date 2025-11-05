use super::TLE;

use crate::Instant;
use anyhow::{bail, Context, Result};

use crate::mathtypes::*;

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
            bstar: p[6],
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
    ///
    /// Fit a TLE from a set of states and times.
    ///
    /// This function uses a non-linear least squares fitting approach to find the best
    /// TLE parameters that match the provided states and times.
    ///
    /// # Arguments
    /// * `states_gcrf` - A slice of state vectors in GCRF coordinates.
    ///   State vectors are [f64; 6] with 1st three elements representing position
    ///   in meters and last three elements representing velocity in meters / second.
    /// * `times` - A slice of times corresponding to the state vectors.
    /// * `epoch` - The epoch time for the TLE.
    ///
    /// # Returns
    /// A tuple containing the fitted TLE and the status of the fitting process.
    ///
    /// # Notes:
    ///
    /// * This function makes use of the `rmpfit` crate for the non-linear least squares fitting.
    ///   This crate is a thin wrapper around the `cmpfit` C library, which implements
    ///   the Levenberg-Marquardt algorithm.
    ///   For details see the rmpfit page at <https://docs.rs/rmpfit/latest/rmpfit/>
    ///
    /// * The fitting process is performed in the TEME frame, with SGP4 used to generate
    ///   the states from the TLE.  The input GCRF states are rotated into the TEME frame
    ///   by this function.
    ///    
    /// * First and second derivatives of mean motion are ignored, as they are not
    ///   used by SGP4.
    ///
    /// * Parameters in the TLE that are fit:
    ///   - 0: Inclination (degrees)
    ///   - 1: Eccentricity
    ///   - 2: Right Ascension of Ascending Node (RAAN) (degrees)
    ///   - 3: Argument of Perigee (degrees)
    ///   - 4: Mean Motion (revolutions per day)
    ///   - 5: Mean Anomaly (degrees)
    ///   - 6: BSTAR drag term
    ///
    /// # Example:
    ///
    /// ```rust
    /// // Construct a GCRF state vector    
    /// let altitude = 400.0e3;
    /// let r0 = satkit::consts::EARTH_RADIUS + altitude;
    /// let v0 = (satkit::consts::MU_EARTH / r0).sqrt();
    /// let inc: f64 = 97.0_f64.to_radians();
    /// let state0 = nalgebra::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
    /// let time0 = satkit::Instant::from_datetime(2016, 5, 16, 12, 0, 0.0).unwrap();
    ///
    /// // High-fidelity orbit propagation settings
    /// let settings = satkit::orbitprop::PropSettings {
    ///     enable_interp: true,
    ///     ..Default::default()
    /// };
    ///
    /// // Satellite has finite drag
    /// let satprops = satkit::orbitprop::SatPropertiesStatic {
    ///     cdaoverm: 2.0 * 10.0 / 3500.0,
    ///     craoverm: 10.0 / 3500.0,
    /// };
    ///
    /// // Propagate over a day
    /// let res = satkit::orbitprop::propagate(
    ///     &state0,
    ///     &time0,
    ///     &(time0 + satkit::Duration::from_seconds(86400.0)),
    ///     &settings,
    ///     Some(&satprops),
    /// ).unwrap();
    ///
    /// // Get high-fidelity states every 10 seconds via interpolation
    /// let times = (0..860)
    ///     .map(|i| time0 + satkit::Duration::from_seconds(i as f64 * 10.0))
    ///     .collect::<Vec<_>>();
    /// let states = times
    ///     .iter()
    ///     .map(|t| {
    ///         let s = res.interp(t).unwrap();
    ///         [s[0], s[1], s[2], s[3], s[4], s[5]]
    ///     })
    ///     .collect::<Vec<_>>();
    ///
    /// // Fit a TLE from the states and times
    /// let (tle, status) = satkit::TLE::fit_from_states(&states, &times, time0).unwrap();
    ///
    /// // Print results
    /// println!("status = {:?}", status.success);
    /// println!("Fitted TLE: {}", tle);
    ///
    /// ```
    pub fn fit_from_states(
        states_gcrf: &[[f64; 6]],
        times: &[Instant],
        epoch: Instant,
    ) -> Result<(Self, rmpfit::MPStatus)> {
        // Make sure lengths are identical
        if states_gcrf.len() != times.len() {
            bail!("States and times must have the same length");
        } else if states_gcrf.is_empty() {
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

        // Rotate states to the TEME frame from GCRF
        // (TLEs represent state in TEME)
        let states_teme = times
            .iter()
            .enumerate()
            .map(|(i, time)| {
                let q = crate::frametransform::qteme2gcrf(time).conjugate();
                let p = q.transform_vector(&nalgebra::vector![
                    states_gcrf[i][0],
                    states_gcrf[i][1],
                    states_gcrf[i][2]
                ]);
                let v = q.transform_vector(&nalgebra::vector![
                    states_gcrf[i][3],
                    states_gcrf[i][4],
                    states_gcrf[i][5]
                ]);
                [p[0], p[1], p[2], v[0], v[1], v[2]]
            })
            .collect::<Vec<_>>();

        // Get the state
        let closest_state = states_teme[closest_index];
        // Kepler representation
        let mut kepler = crate::kepler::Kepler::from_pv(
            Vector3::from_column_slice(&closest_state[..3]),
            Vector3::from_column_slice(&closest_state[3..]),
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
            0.0,
        ];

        let mut p = Problem {
            states: &states_teme,
            times,
            epoch,
        };
        match p.mpfit(&mut init_params) {
            Ok(status) => Ok((p.tle_from_params(&init_params), status)),
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
        let state0 = nalgebra::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2022, 5, 16, 12, 0, 0.0)?;

        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(86400.0)),
            &settings,
            None,
        )?;

        let times = (0..8640)
            .map(|i| time0 + crate::Duration::from_seconds(i as f64 * 10.0))
            .collect::<Vec<_>>();
        let states = times
            .iter()
            .map(|t| {
                let s = res.interp(t).unwrap();
                [s[0], s[1], s[2], s[3], s[4], s[5]]
            })
            .collect::<Vec<_>>();

        let (_tle, _status) = TLE::fit_from_states(&states, &times, time0)?;
        Ok(())
    }

    #[test]
    fn test_fit_from_states_with_drag() -> Result<()> {
        let altitude = 400.0e3;
        let r0 = crate::consts::EARTH_RADIUS + altitude;
        let v0 = (crate::consts::MU_EARTH / r0).sqrt();
        let inc: f64 = 97.0_f64.to_radians();
        let state0 = nalgebra::vector![r0, 0.0, 0.0, 0.0, v0 * inc.cos(), v0 * inc.sin()];
        let time0: Instant = Instant::from_datetime(2016, 5, 16, 12, 0, 0.0)?;

        let settings = crate::orbitprop::PropSettings {
            enable_interp: true,
            ..Default::default()
        };

        let satprops = crate::orbitprop::SatPropertiesStatic {
            cdaoverm: 2.0 * 10.0 / 3500.0,
            craoverm: 10.0 / 3500.0,
        };

        let res = crate::orbitprop::propagate(
            &state0,
            &time0,
            &(time0 + crate::Duration::from_seconds(86400.0)),
            &settings,
            Some(&satprops),
        )?;

        let times = (0..8640)
            .map(|i| time0 + crate::Duration::from_seconds(i as f64 * 10.0))
            .collect::<Vec<_>>();
        let states = times
            .iter()
            .map(|t| {
                let s = res.interp(t).unwrap();
                [s[0], s[1], s[2], s[3], s[4], s[5]]
            })
            .collect::<Vec<_>>();

        let (tle, status) = TLE::fit_from_states(&states, &times, time0)?;
        println!("status = {:?}", status.success);
        println!("Fitted TLE: {}", tle);
        Ok(())
    }
}
