use super::sgp4_lowlevel::sgp4_lowlevel; // propagator
use super::sgp4init::sgp4init;

use crate::TimeLike;
use nalgebra::{Const, Dyn, OMatrix};

use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Eq, Copy )]
pub enum SGP4Error {
    #[error("Success")]
    SGP4Success = 0,
    #[error("Eccentricity > 1 or < 0")]
    SGP4ErrorEccen = 1,
    #[error("Mean motion < 0")]
    SGP4ErrorMeanMotion = 2,
    #[error("Perturbed Eccentricity > 1 or < 0")]
    SGP4ErrorPerturbEccen = 3,
    #[error("Semi-Latus Rectum < 0")]
    SGP4ErrorSemiLatusRectum = 4,
    #[error("Unused")]
    SGP4ErrorUnused = 5,
    #[error("Orbit Decayed")]
    SGP4ErrorOrbitDecay = 6,
}
impl From<i32> for SGP4Error {
    fn from(val: i32) -> Self {
        match val {
            0 => Self::SGP4Success,
            1 => Self::SGP4ErrorEccen,
            2 => Self::SGP4ErrorMeanMotion,
            3 => Self::SGP4ErrorPerturbEccen,
            4 => Self::SGP4ErrorSemiLatusRectum,
            6 => Self::SGP4ErrorOrbitDecay,
            _ => Self::SGP4ErrorUnused,
        }
    }
}

impl From<SGP4Error> for i32 {
    fn from(val: SGP4Error) -> Self {
        match val {
            SGP4Error::SGP4ErrorEccen => 1,
            SGP4Error::SGP4ErrorMeanMotion => 2,
            SGP4Error::SGP4ErrorOrbitDecay => 6,
            SGP4Error::SGP4ErrorPerturbEccen => 3,
            SGP4Error::SGP4ErrorSemiLatusRectum => 4,
            SGP4Error::SGP4ErrorUnused => -1,
            SGP4Error::SGP4Success => 0,
        }
    }
}

type StateArr = OMatrix<f64, Const<3>, Dyn>;

pub struct SGP4State {
    pub pos: StateArr,
    pub vel: StateArr,
    pub errcode: Vec<SGP4Error>,
}


use super::{GravConst, OpsMode, SGP4Source};

///
/// Run Simplified General Perturbations (SGP)-4 propagator on
/// Two-Line Element Set to
/// output satellite position and velocity at given time
/// in the "TEME" coordinate system
///
/// This is a shortcut to run sgp4_full with the WGS84 gravity model and IMPROVED ops mode
///
/// A detailed description is
/// [here](https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf)
///
///
/// # Arguments
///
/// * `sgp4source` - The source of SGP4 data, typically a TLE but could be a
///    orbital mean-elements message (OMM) or other source implementing the
///   SGP4Source trait.  Note: this is a mutable reference SGP4 states are cached
///   in the source object after first call to avoid re-initialization on subsequent calls
/// * `tm` -  The time at which to compute position and velocity
///   Input as a slice for convenience. `satkit::TimeLike` trait is used for time input,
///   can be `satkit::Instant` or if chrono feature is enabled, `chrono::DateTime<Utc>`
///
///
/// # Return
///
/// Result object containing either an OK value containing a SGP4State struct with
/// position (m) and velocity (m/s) Nx3 matrices (where N is the nuber of input
/// times in the slice) and err codes at each time, or an Err value containing
/// a description of the error
///
/// # Note:
///
/// This is a shortcut to run sgp4_full with the WGS84 gravity model and IMPROVED ops mode
///
/// # Example
///
/// ```
/// // Compute the Geodetic position of a satellite at
/// // the TLE epoch time
///
/// use satkit::TLE;
/// use satkit::sgp4::{sgp4, GravConst, OpsMode};
/// use satkit::frametransform::qteme2itrf;
/// use satkit::itrfcoord::ITRFCoord;
/// use nalgebra as na;
///
/// let line0: &str = "0 INTELSAT 902";
/// let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
/// let line2: &str = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
/// let mut tle = TLE::load_3line(&line0.to_string(),
///     &line1.to_string(),
///     &line2.to_string()
///     ).unwrap();
///
/// let tm = tle.epoch;
///
/// // SGP4 runs on a slice of times
/// let result = sgp4(&mut tle,
///     &[tm]
///     ).unwrap();
///
/// let pitrf = qteme2itrf(&tm).to_rotation_matrix() * result.pos;
/// let itrf = ITRFCoord::from_slice(pitrf.as_slice()).unwrap();
/// println!("Satellite position is: {}", itrf);
///
/// ```
///
#[inline]
pub fn sgp4<T: TimeLike>(sgp4source: &mut impl SGP4Source, tm: &[T]) -> anyhow::Result<SGP4State> {
    sgp4_full(sgp4source, tm, GravConst::WGS84, OpsMode::IMPROVED)
}

///
/// Run Simplified General Perturbations (SGP)-4 propagator on
/// Two-Line Element Set to
/// output satellite position and velocity at given time
/// in the "TEME" coordinate system
///
/// A detailed description is
/// [here](https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf)
///
///
/// # Arguments
///
/// * `sgp4source` - The source of SGP4 data, typically a TLE but could be a
///    orbital mean-elements message (OMM) or other source implementing the
///   SGP4Source trait.  Note: this is a mutable reference SGP4 states are cached
///   in the source object after first call to avoid re-initialization on subsequent calls.
/// * `tm` -  The time at which to compute position and velocity
///   Input as a slice for convenience. `satkit::TimeLike` trait is used for time input,
///   can be `satkit::Instant` or if chrono feature is enabled, `chrono::DateTime<Utc>`
///
/// * `gravconst` - The gravitational constant to use.
///
/// * `opsmode` - The operational mode to use.
///
///
/// # Return
///
/// Result object containing either an OK value containing a tuple with
/// position (m) and velocity (m/s) Nx3 matrices (where N is the number of input
/// times in the slice) or an Err value containing
/// a tuple with error code and error string
///
/// # Example
///
/// ```
/// // Compute the Geodetic position of a satellite at
/// // the TLE epoch time
///
/// use satkit::TLE;
/// use satkit::sgp4::{sgp4_full, GravConst, OpsMode};
/// use satkit::frametransform::qteme2itrf;
/// use satkit::itrfcoord::ITRFCoord;
/// use nalgebra as na;
///
/// let line0: &str = "0 INTELSAT 902";
/// let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
/// let line2: &str = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
/// let mut tle = TLE::load_3line(&line0.to_string(),
///     &line1.to_string(),
///     &line2.to_string()
///     ).unwrap();
///
/// let tm = tle.epoch;
///
/// // SGP4 runs on a slice of times
/// let result = sgp4_full(&mut tle,
///     &[tm],
///     GravConst::WGS84,
///     OpsMode::IMPROVED
///     ).unwrap();
///
/// let pitrf = qteme2itrf(&tm).to_rotation_matrix() * result.pos;
/// let itrf = ITRFCoord::from_slice(pitrf.as_slice()).unwrap();
/// println!("Satellite position is: {}", itrf);
///
/// ```
///
pub fn sgp4_full<T: TimeLike>(
    sgp4source: &mut impl SGP4Source,
    tm: &[T],
    gravconst: GravConst,
    opsmode: OpsMode,
) -> anyhow::Result<SGP4State> {
    if sgp4source.satrec_mut().is_none() {
        let args = sgp4source.sgp4_init_args()?;

        *sgp4source.satrec_mut() = Some(sgp4init(
            gravconst,
            opsmode,
            "satno",
            args.jdsatepoch - 2433281.5,
            args.bstar,
            args.ndot,
            args.nddot,
            args.ecco,
            args.argpo,
            args.inclo,
            args.mo,
            args.no,
            args.nodeo,
        ).map_err(|e| anyhow::anyhow!("SGP4 init error: {}", e))?);
    }

    let epoch = sgp4source.epoch();
    let s = sgp4source.satrec_mut().as_mut().expect("satrec initialized");

    let n = tm.len();
    let mut rarr = StateArr::zeros(n);
    let mut varr = StateArr::zeros(n);
    let mut earr = Vec::<SGP4Error>::with_capacity(n);

    for (pos, thetime) in tm.iter().enumerate() {
        let tsince = (thetime.as_instant() - epoch).as_days() * 1440.0;

        match sgp4_lowlevel(s, tsince) {
            Ok((r, v)) => {
                rarr.index_mut((.., pos)).copy_from_slice(&r);
                varr.index_mut((.., pos)).copy_from_slice(&v);
                earr.push(SGP4Error::SGP4Success)
            }
            Err(e) => earr.push(e.into()),
        }
    }
    Ok(SGP4State {
        pos: rarr * 1.0e3,
        vel: varr * 1.0e3,
        errcode: earr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tle::TLE;
    use crate::utils::test;
    use anyhow::{bail, Result};
    use std::io::BufRead;

    #[test]
    fn testsgp4() {
        let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
        let line2: &str =
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
        let line0: &str = "0 INTELSAT 902";

        let mut tle = TLE::load_3line(line0, line1, line2).unwrap();
        let tm = tle.epoch;

        let states = sgp4(&mut tle, &[tm]).unwrap();
        assert!(states.errcode[0] == SGP4Error::SGP4Success);
    }

    #[test]
    fn vallado_testvecs() -> Result<()> {
        let testdir = test::get_testvec_dir().unwrap().join("sgp4");
        if !testdir.is_dir() {
            bail!(
                "Required SGP4 test vectors directory: \"{}\" does not exist.
                    Clone test vectors from:
                    <https://storage.googleapis.com/satkit-testvecs/>
                    or using python script in satkit repo: `python/test/download_testvecs.py`
                    or set \"SATKIT_TESTVEC_ROOT\" to point to directory",
                testdir.to_string_lossy()
            );
        }
        let tlefile = testdir.join("SGP4-VER.TLE");
        let f = match std::fs::File::open(&tlefile) {
            Err(why) => bail!("Could not open {}: {}", tlefile.display(), why),
            Ok(file) => file,
        };
        let buf = std::io::BufReader::new(f);
        // Vallado test vectors include some extra information at the end of the line
        // So truncate all lines to 69 characters
        let lines: Vec<String> = buf
            .lines()
            .map(|l| {
                let line = l.unwrap();
                line.chars().take(69).collect()
            })
            .collect();

        let tles = TLE::from_lines(&lines).unwrap();

        assert!(tles.len() > 5);

        for mut tle in tles {
            let fname = format!("{:05}.e", tle.sat_num);

            let fh = testdir.join(fname);
            let ftle = match std::fs::File::open(&fh) {
                Err(why) => bail!("Could not open {}: {}", fh.display(), why),
                Ok(file) => file,
            };
            for line in std::io::BufReader::new(ftle).lines() {
                let maxposerr = 1.0e-5;
                let mut maxvelerr = 1.0e-5;

                let testvec: Vec<f64> = line
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap_or(-1.0))
                    .collect();
                if testvec.len() < 7 {
                    continue;
                }
                if testvec[0] < 0.0 {
                    continue;
                }
                let tm = tle.epoch + crate::Duration::from_seconds(testvec[0]);

                // Test vectors assume WGS72 gravity model and AFSPC ops mode
                let states = sgp4_full(&mut tle, &[tm], GravConst::WGS72, OpsMode::AFSPC);
                let states = match states {
                    Ok(s) => s,
                    Err(e) => {
                        // We know one of the test vectors is supposed to fail
                        if tle.sat_num == 33334 {
                            continue;
                        }
                        return Err(e);

                    }
                };
                if states.errcode[0] != SGP4Error::SGP4Success {
                    continue;
                }
                for idx in 0..3 {
                    // Account for truncation in truth data
                    if testvec[idx + 4].abs() < 1.0e-4 {
                        maxvelerr = 1.0e-4;
                    }
                    if testvec[idx + 4].abs() < 1.0e-6 {
                        maxvelerr = 1.0e-2;
                    }
                    let poserr =
                        (states.pos[idx].mul_add(1.0e-3, -testvec[idx + 1]) / testvec[idx + 1]).abs();
                    let velerr =
                        (states.vel[idx].mul_add(1.0e-3, -testvec[idx + 4]) / testvec[idx + 4]).abs();
                    assert!(poserr < maxposerr);
                    assert!(velerr < maxvelerr);
                }
            }
        }
        Ok(())
    }
}
