use super::sgp4_lowlevel::sgp4_lowlevel; // propagator
use super::sgp4init::sgp4init;

use super::SGP4Error;
use crate::mathtypes::DMatrix;
use crate::TimeLike;

pub struct SGP4State {
    pub pos: DMatrix<f64>,
    pub vel: DMatrix<f64>,
    pub errcode: Vec<SGP4Error>,
}

use super::{GravConst, OpsMode, SGP4Source};

///
/// Run Simplified General Perturbations (SGP)-4 propagator on
/// Two-Line Element Set to
/// output satellite position and velocity at given time
/// in the "TEME" coordinate system
///
/// This is [`sgp4_full`] with the WGS72 gravity model and the AFSPC ops mode,
/// the same defaults as the Python `satkit.sgp4`. WGS72 is the gravity model
/// the element sets published by Space-Track and CelesTrak are fitted with.
///
/// A detailed description is in Vallado, Crawford, Hujsak & Kelso,
/// "Revisiting Spacetrack Report #3", AIAA 2006-6753
/// (<https://doi.org/10.2514/6.2006-6753>, PDF at
/// <https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>).
///
///
/// # Arguments
///
/// * `sgp4source` - The source of SGP4 data, typically a TLE but could be a
///   orbital mean-elements message (OMM) or other source implementing the
///   SGP4Source trait.  Note: this is a mutable reference; the SGP4
///   initialization is cached in the source object (see [`sgp4_full`])
/// * `tm` -  The time at which to compute position and velocity
///   Input as a slice for convenience. `satkit::TimeLike` trait is used for time input,
///   can be `satkit::Instant` or if chrono feature is enabled, `chrono::DateTime<Utc>`
///
///
/// # Return
///
/// Result object containing either an OK value containing a SGP4State struct with
/// position (m) and velocity (m/s) 3xN matrices (where N is the number of input
/// times in the slice) and err codes at each time, or an Err value containing
/// a description of the error
///
/// # Note:
///
/// The default gravity model was WGS84 (with the IMPROVED ops mode) before
/// satkit 0.24; call [`sgp4_full`] to choose it explicitly. Time since epoch
/// is physical elapsed time; see [`sgp4_full`] for how that treats leap
/// seconds.
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
/// // rotate position to ITRF and create ITRFCoord
/// let pos = numeris::vector![result.pos[(0,0)], result.pos[(1,0)], result.pos[(2,0)]];
/// let pitrf = qteme2itrf(&tm) * pos;
/// let itrf = ITRFCoord::from_slice(pitrf.as_slice()).unwrap();
/// println!("Satellite position is: {}", itrf);
///
/// ```
///
#[inline]
pub fn sgp4<T: TimeLike>(sgp4source: &mut impl SGP4Source, tm: &[T]) -> super::Result<SGP4State> {
    sgp4_full(sgp4source, tm, GravConst::WGS72, OpsMode::AFSPC)
}

///
/// Run Simplified General Perturbations (SGP)-4 propagator on
/// Two-Line Element Set to
/// output satellite position and velocity at given time
/// in the "TEME" coordinate system
///
/// A detailed description is in Vallado, Crawford, Hujsak & Kelso,
/// "Revisiting Spacetrack Report #3", AIAA 2006-6753
/// (<https://doi.org/10.2514/6.2006-6753>, PDF at
/// <https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>).
///
///
/// # Arguments
///
/// * `sgp4source` - The source of SGP4 data, typically a TLE but could be a
///   orbital mean-elements message (OMM) or other source implementing the
///   SGP4Source trait.  Note: this is a mutable reference; the SGP4
///   initialization is cached in the source object, together with the
///   gravity model, ops mode and element values it was built from. It is
///   reused only while all of those match, so editing the elements or
///   changing `gravconst` / `opsmode` re-initializes.
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
/// position (m) and velocity (m/s) 3xN matrices (where N is the number of input
/// times in the slice) or an Err value containing
/// a tuple with error code and error string
///
/// # Leap seconds
///
/// The time since epoch passed to SGP4 is the physical (SI) time elapsed
/// between the element-set epoch and `tm`. Across a leap second this is one
/// second more than the difference of the UTC labels, which is what Vallado's
/// reference code and python-sgp4 use (they count UTC minutes), so satkit
/// differs from them by 1 s of along-track motion (~7.6 km at LEO) for each
/// leap second between epoch and `tm`. This is deliberate: the satellite
/// really flies 86,401 s over a day with a leap second, and SGP4's mean motion
/// is per SI day.
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
/// // rotate position to ITRF and create ITRFCoord
/// let pos = numeris::vector![result.pos[(0,0)], result.pos[(1,0)], result.pos[(2,0)]];
/// let pitrf = qteme2itrf(&tm) * pos;
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
) -> super::Result<SGP4State> {
    // Always rebuild the init arguments (cheap): they are the cache key, and
    // this re-applies the source's validation (e.g. SGP4-XP rejection) after
    // its elements were edited.
    let args = sgp4source.sgp4_init_args()?;
    let key = super::SatRecKey::new(gravconst, opsmode, args);
    let cached = matches!(sgp4source.satrec_mut(), Some(s) if s.init_key == Some(key));
    if !cached {
        // Drop any stale record first, so a failed init leaves no cache
        *sgp4source.satrec_mut() = None;
        let mut satrec = sgp4init(
            gravconst,
            opsmode,
            args.epoch_days_1950,
            args.bstar,
            args.ndot,
            args.nddot,
            args.ecco,
            args.argpo,
            args.inclo,
            args.mo,
            args.no,
            args.nodeo,
        )
        .map_err(|code| super::Error::SatRecInit(code.into()))?;
        satrec.init_key = Some(key);
        *sgp4source.satrec_mut() = Some(satrec);
    }

    let epoch = sgp4source.epoch();
    let s = sgp4source
        .satrec_mut()
        .as_mut()
        .expect("satrec initialized");

    let n = tm.len();
    let mut rarr = DMatrix::<f64>::zeros(3, n);
    let mut varr = DMatrix::<f64>::zeros(3, n);
    let mut earr = Vec::<SGP4Error>::with_capacity(n);

    for (pos, thetime) in tm.iter().enumerate() {
        let tsince = (thetime.as_instant() - epoch).as_days() * 1440.0;

        match sgp4_lowlevel(s, tsince) {
            Ok((r, v)) => {
                for i in 0..3 {
                    rarr[(i, pos)] = r[i];
                    varr[(i, pos)] = v[i];
                }
                earr.push(SGP4Error::SGP4Success)
            }
            Err(e) => {
                // Leave the failed columns as NaN rather than zero: a zeroed
                // column reads as a valid position at Earth's center to any
                // caller that forgets to inspect `errcode`.
                for i in 0..3 {
                    rarr[(i, pos)] = f64::NAN;
                    varr[(i, pos)] = f64::NAN;
                }
                earr.push(e.into())
            }
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
                        return Err(e.into());
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
                    let poserr = (states.pos[(idx, 0)].mul_add(1.0e-3, -testvec[idx + 1])
                        / testvec[idx + 1])
                        .abs();
                    let velerr = (states.vel[(idx, 0)].mul_add(1.0e-3, -testvec[idx + 4])
                        / testvec[idx + 4])
                        .abs();
                    assert!(poserr < maxposerr);
                    assert!(velerr < maxvelerr);
                }
            }
        }
        Ok(())
    }

    const ISS1: &str = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9994";
    const ISS2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815350434159";

    fn pos(src: &mut TLE, t: crate::Instant, gc: GravConst, om: OpsMode) -> [f64; 3] {
        let s = sgp4_full(src, &[t], gc, om).unwrap();
        [s.pos[(0, 0)], s.pos[(1, 0)], s.pos[(2, 0)]]
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn default_is_wgs72_afspc() {
        let mut a = TLE::load_2line(ISS1, ISS2).unwrap();
        let t = a.epoch + crate::Duration::from_days(1.0);
        let s = sgp4(&mut a, &[t]).unwrap();
        let p = [s.pos[(0, 0)], s.pos[(1, 0)], s.pos[(2, 0)]];
        let mut b = TLE::load_2line(ISS1, ISS2).unwrap();
        assert_eq!(p, pos(&mut b, t, GravConst::WGS72, OpsMode::AFSPC));
        let mut c = TLE::load_2line(ISS1, ISS2).unwrap();
        assert!(dist(p, pos(&mut c, t, GravConst::WGS84, OpsMode::IMPROVED)) > 1.0);
    }

    #[test]
    fn cache_follows_gravconst_and_opsmode() {
        let mut a = TLE::load_2line(ISS1, ISS2).unwrap();
        let t = a.epoch + crate::Duration::from_days(3.0);
        let p72 = pos(&mut a, t, GravConst::WGS72, OpsMode::AFSPC);
        // A second call on the same TLE with another gravity model must not
        // reuse the WGS72 initialization (it did: ~120 m at 3 days)
        let p84 = pos(&mut a, t, GravConst::WGS84, OpsMode::AFSPC);
        let mut fresh = TLE::load_2line(ISS1, ISS2).unwrap();
        assert_eq!(p84, pos(&mut fresh, t, GravConst::WGS84, OpsMode::AFSPC));
        assert!(dist(p84, p72) > 10.0);
        // and back
        assert_eq!(p72, pos(&mut a, t, GravConst::WGS72, OpsMode::AFSPC));
        assert_eq!(
            pos(&mut a, t, GravConst::WGS72, OpsMode::IMPROVED),
            pos(
                &mut TLE::load_2line(ISS1, ISS2).unwrap(),
                t,
                GravConst::WGS72,
                OpsMode::IMPROVED
            )
        );
    }

    #[test]
    fn cache_follows_element_edits() {
        let mut a = TLE::load_2line(ISS1, ISS2).unwrap();
        let t = a.epoch + crate::Duration::from_minutes(90.0);
        let before = pos(&mut a, t, GravConst::WGS72, OpsMode::AFSPC);
        // Editing pub fields after a propagation takes effect (it used to
        // reuse the stale initialization: ~6000 km)
        a.inclination = 97.0;
        a.mean_motion = 14.2;
        let edited = pos(&mut a, t, GravConst::WGS72, OpsMode::AFSPC);
        let mut b = TLE::load_2line(ISS1, ISS2).unwrap();
        b.inclination = 97.0;
        b.mean_motion = 14.2;
        assert_eq!(edited, pos(&mut b, t, GravConst::WGS72, OpsMode::AFSPC));
        assert!(dist(edited, before) > 1.0e5);

        // So does the epoch
        let mut c = TLE::load_2line(ISS1, ISS2).unwrap();
        let _ = pos(&mut c, t, GravConst::WGS72, OpsMode::AFSPC);
        c.epoch += crate::Duration::from_days(1.0);
        let mut d = TLE::load_2line(ISS1, ISS2).unwrap();
        d.epoch += crate::Duration::from_days(1.0);
        assert_eq!(
            pos(&mut c, t, GravConst::WGS72, OpsMode::AFSPC),
            pos(&mut d, t, GravConst::WGS72, OpsMode::AFSPC)
        );

        // An element set edited into SGP4-XP is refused even though it was
        // propagated (and cached) before
        a.ephem_type = 4;
        let err = sgp4(&mut a, &[t]).err().expect("type 4 must be refused");
        assert!(err.to_string().contains("SGP4-XP"), "{err}");
        a.ephem_type = 0;
        assert_eq!(edited, pos(&mut a, t, GravConst::WGS72, OpsMode::AFSPC));
    }

    #[test]
    fn epoch_passed_at_full_precision() {
        // Days since 1950 from the integer-microsecond epoch, not from a
        // full Julian date in one f64 (which quantizes at ~40 µs and gave
        // 28002.295599940233 here)
        let l1 = "1 45608U 20031A   26243.29559994 -.00000360  00000+0  00000+0 0  9996";
        let l2 = "2 45608  63.1638  56.8572 7115327 266.6524  17.3178  2.00576418 45977";
        let tle = TLE::load_2line(l1, l2).unwrap();
        let args = tle.sgp4_init_args().unwrap();
        assert_eq!(args.epoch_days_1950, 28002.29559994);
    }

    #[test]
    fn tsince_is_physical_across_leap_second() {
        // Epoch 2016-12-31 12:00 UTC; 2017-01-01 12:00 UTC is 1440 UTC
        // minutes later, but 86,401 s later because of the leap second at
        // 2016-12-31T23:59:60. satkit propagates by the elapsed SI time,
        // deliberately 1 s (~7.6 km at LEO) ahead of Vallado / python-sgp4,
        // which use the UTC-minute difference.
        let l1 = "1 25544U 98067A   16366.50000000  .00016717  00000-0  10270-3 0  9994";
        let mut tle = TLE::load_2line(l1, ISS2).unwrap();
        let t = crate::Instant::from_datetime(2017, 1, 1, 12, 0, 0.0).unwrap();
        assert_eq!((t - tle.epoch).as_seconds(), 86401.0);
        let p = pos(&mut tle, t, GravConst::WGS72, OpsMode::AFSPC);

        let satrec = tle.satrec_mut().as_mut().unwrap();
        let (r_phys, _) = sgp4_lowlevel(satrec, 1440.0 + 1.0 / 60.0).unwrap();
        let (r_utc, _) = sgp4_lowlevel(satrec, 1440.0).unwrap();
        let km = |r: [f64; 3]| [r[0] * 1.0e3, r[1] * 1.0e3, r[2] * 1.0e3];
        assert!(dist(p, km([r_phys[0], r_phys[1], r_phys[2]])) < 1.0e-6);
        assert!(dist(p, km([r_utc[0], r_utc[1], r_utc[2]])) > 7.0e3);
    }
}
