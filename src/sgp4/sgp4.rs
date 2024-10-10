use super::sgp4_lowlevel::sgp4_lowlevel; // propagator
use super::sgp4init::sgp4init;

use crate::astrotime::{AstroTime, Scale};
use crate::tle::TLE;
use nalgebra::{Const, Dyn, OMatrix};

use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq)]
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
    fn from(val: i32) -> SGP4Error {
        match val {
            0 => SGP4Error::SGP4Success,
            1 => SGP4Error::SGP4ErrorEccen,
            2 => SGP4Error::SGP4ErrorMeanMotion,
            3 => SGP4Error::SGP4ErrorPerturbEccen,
            4 => SGP4Error::SGP4ErrorSemiLatusRectum,
            6 => SGP4Error::SGP4ErrorOrbitDecay,
            _ => SGP4Error::SGP4ErrorUnused,
        }
    }
}

impl From<SGP4Error> for i32 {
    fn from(val: SGP4Error) -> i32 {
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
pub type SGP4State = (StateArr, StateArr, Vec<SGP4Error>);
pub type SGP4Result = Result<SGP4State, (SGP4Error, usize)>;

use std::f64::consts::PI;

use super::{GravConst, OpsMode};

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
/// * `tle` - The TLE on which top operate.  Note that a mutable reference
///           is passed, as SGP-4 metadata is stored after each propagation
/// * `tm` -  The time at which to compute position and velocity
///           Input as a slice for convenience.
///
///
/// # Return
///
/// Result object containing either an OK value containing a tuple with
/// position (m) and velocity (m/s) Nx3 matrices (where N is the nuber of input
/// times in the slice) or an Err value containing
/// a tuple with error code and error string
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
/// let (pteme, vteme, errs) = sgp4(&mut tle,
///     &[tm]
///     );
///
/// let pitrf = qteme2itrf(&tm).to_rotation_matrix() * pteme;
/// let itrf = ITRFCoord::from_slice(pitrf.as_slice()).unwrap();
/// println!("Satellite position is: {}", itrf);
///
/// ```
///
#[inline]
pub fn sgp4(tle: &mut TLE, tm: &[AstroTime]) -> SGP4State {
    sgp4_full(tle, tm, GravConst::WGS84, OpsMode::IMPROVED)
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
/// * `tle` - The TLE on which top operate.  Note that a mutable reference
///           is passed, as SGP-4 metadata is stored after each propagation
/// * `tm` -  The time at which to compute position and velocity
///           Input as a slice for convenience.
///
/// * `gravconst` - The gravitational constant to use.
///
/// * `opsmode` - The operational mode to use.
///
///
/// # Return
///
/// Result object containing either an OK value containing a tuple with
/// position (m) and velocity (m/s) Nx3 matrices (where N is the nuber of input
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
/// let (pteme, vteme, errs) = sgp4_full(&mut tle,
///     &[tm],
///     GravConst::WGS84,
///     OpsMode::IMPROVED
///     );
///
/// let pitrf = qteme2itrf(&tm).to_rotation_matrix() * pteme;
/// let itrf = ITRFCoord::from_slice(pitrf.as_slice()).unwrap();
/// println!("Satellite position is: {}", itrf);
///
/// ```
///
pub fn sgp4_full<'a>(
    tle: &mut TLE,
    tm: &[AstroTime],
    gravconst: GravConst,
    opsmode: OpsMode,
) -> SGP4State {
    const TWOPI: f64 = PI * 2.0;

    if tle.satrec.is_none() {
        let no = tle.mean_motion / (1440.0 / TWOPI);
        let bstar = tle.bstar;
        let ndot = tle.mean_motion_dot / (1440.0 * 1440.0 / TWOPI);
        let nddot = tle.mean_motion_dot_dot / (1440.0 * 1440.0 * 1440.0 / TWOPI);
        let inclo = tle.inclination * PI / 180.0;
        let nodeo = tle.raan * PI / 180.0;
        let argpo = tle.arg_of_perigee * PI / 180.0;
        let mo = tle.mean_anomaly * PI / 180.0;
        let ecco = tle.eccen;
        let jdsatepoch = tle.epoch.to_jd(Scale::UTC);

        match sgp4init(
            gravconst,
            opsmode,
            &"satno",
            jdsatepoch - 2433281.5,
            bstar,
            ndot,
            nddot,
            ecco,
            argpo,
            inclo,
            mo,
            no,
            nodeo,
        ) {
            Ok(sr) => tle.satrec = Some(sr),
            Err(e) => {
                let n = tm.len();

                let rarr = StateArr::zeros(n);
                let varr = StateArr::zeros(n);
                let earr = Vec::<SGP4Error>::from_iter((0..n).map(|_x| SGP4Error::from(e)));
                return (rarr, varr, earr);
            }
        }
    }

    let mut s = tle.satrec.as_mut().unwrap();

    let n = tm.len();
    let mut rarr = StateArr::zeros(n);
    let mut varr = StateArr::zeros(n);
    let mut earr = Vec::<SGP4Error>::new();
    earr.reserve(n);

    for (pos, thetime) in tm.iter().enumerate() {
        let tsince = (*thetime - tle.epoch).days() * 1440.0;

        match sgp4_lowlevel(&mut s, tsince) {
            Ok((r, v)) => {
                rarr.index_mut((.., pos)).copy_from_slice(&r);
                varr.index_mut((.., pos)).copy_from_slice(&v);
                earr.push(SGP4Error::SGP4Success)
            }
            Err(e) => earr.push(e.into()),
        }
    }
    (rarr * 1.0e3, varr * 1.0e3, earr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tle::TLE;
    use crate::utils::test;
    use std::io::BufRead;

    #[test]
    fn testsgp4() {
        let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
        let line2: &str =
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
        let line0: &str = "0 INTELSAT 902";

        let mut tle =
            TLE::load_3line(&line0.to_string(), &line1.to_string(), &line2.to_string()).unwrap();
        let tm = tle.epoch;

        let (_pos, _vel, err) = sgp4(&mut tle, &[tm]);
        assert!(err[0] == SGP4Error::SGP4Success);
    }

    #[test]
    fn vallado_testvecs() {
        let testdir = test::get_testvec_dir().unwrap().join("sgp4");
        if !testdir.is_dir() {
            panic!(
                "Required SGP4 test vectors directory: \"{}\" does not exist
                    clone test vectors repo at 
                    https://github.com/StevenSamirMichael/satkit-testvecs.git 
                    from root of repo or set \"SATKIT_TESTVEC_ROOT\" 
                    to point to directory",
                testdir.to_string_lossy()
            );
        }
        let tlefile = testdir.join("SGP4-VER.TLE");
        let f = match std::fs::File::open(&tlefile) {
            Err(why) => panic!("Could not open {}: {}", tlefile.display(), why),
            Ok(file) => file,
        };
        let buf = std::io::BufReader::new(f);
        let lines: Vec<String> = buf.lines().map(|l| l.unwrap()).collect();
        let tles = TLE::from_lines(&lines).unwrap();
        assert!(tles.len() > 5);

        for mut tle in tles {
            let fname = format!("{:05}.e", tle.sat_num);

            let fh = testdir.join(fname);
            let ftle = match std::fs::File::open(&fh) {
                Err(why) => panic!("Could not open {}: {}", fh.display(), why),
                Ok(file) => file,
            };
            for line in std::io::BufReader::new(ftle).lines() {
                let maxposerr = 1.0e-5;
                let mut maxvelerr = 1.0e-5;

                let testvec: Vec<f64> = line
                    .unwrap()
                    .trim()
                    .split_whitespace()
                    .map(|x| match x.parse() {
                        Ok(v) => v,
                        Err(_) => -1.0,
                    })
                    .collect();
                if testvec.len() < 7 {
                    continue;
                }
                if testvec[0] < 0.0 {
                    continue;
                }
                let tm = tle.epoch + testvec[0] / 86400.0;

                // Test vectors assume WGS72 gravity model and AFSPC ops mode
                let (pos, vel, err) = sgp4_full(&mut tle, &[tm], GravConst::WGS72, OpsMode::AFSPC);
                if err[0] != SGP4Error::SGP4Success {
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
                    let poserr = ((pos[idx] * 1.0e-3 - testvec[idx + 1]) / testvec[idx + 1]).abs();
                    let velerr = ((vel[idx] * 1.0e-3 - testvec[idx + 4]) / testvec[idx + 4]).abs();
                    assert!(poserr < maxposerr);
                    assert!(velerr < maxvelerr);
                }
            }
        }
    }
}
