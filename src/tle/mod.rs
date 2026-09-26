use crate::sgp4::SatRec;
use crate::Instant;

use crate::sgp4::{SGP4InitArgs, SGP4Source};

// TLE fitting from state vectors
mod fitting;

mod error;

pub use error::{Error, Result};
pub use fitting::{TleFitResult, TleFitStatus};

// 'I' and 'O' are not part of the allowed chars to avoid any confusion with 0 or 1
const ALPHA5_MATCHING: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ";

///
/// Structure representing a Two-Line Element Set (TLE), a satellite
/// ephemeris format from the 1970s that is still somehow in use
/// today and can be used to calculate satellite position and
/// velocity in the "TEME" frame (not-quite GCRF) using the
/// "Simplified General Perturbations-4" (SGP-4) mathematical
/// model that is also included in this package.
///
/// For details, see: <https://en.wikipedia.org/wiki/Two-line_element_set>
///
/// The TLE format is still commonly used to represent satellite
/// ephemerides, and satellite ephemerides catalogs in this format
/// are publicly available at www.space-track.org (registration
/// required)
///
/// TLEs sometimes have a "line 0" that includes the name of the satellite
///
/// # Example Usage:
///
///
/// ```
/// use satkit::TLE;
/// use satkit::Instant;
/// use satkit::sgp4::sgp4;
/// use satkit::frametransform;
/// use satkit::itrfcoord::ITRFCoord;
///
/// let lines = vec!["0 INTELSAT 902",
///     "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
///     "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."];
///
/// let mut tle = TLE::load_3line(lines[0], lines[1], lines[2]).unwrap();
/// let tm = Instant::from_datetime(2006, 5, 1, 11, 0, 0.0).unwrap();
///
/// // Use SGP4 to get position,
/// let states = sgp4(&mut tle, &[tm]).unwrap();
///
/// println!("pTEME = {}", states.pos);
/// // Rotate the position to the ITRF frame (Earth-fixed)
/// // Since pTEME is a 3xN array where N is the number of times
/// // (we are just using a single time)
/// // we need to convert to a fixed matrix to rotate
/// let pos = numeris::vector![states.pos[(0,0)], states.pos[(1,0)], states.pos[(2,0)]];
/// let pITRF = frametransform::qteme2itrf(&tm) * pos;
///
/// println!("pITRF = {:?}", pITRF);
///
/// // Convert to an "ITRF Coordinate" and print geodetic position
/// let itrf = ITRFCoord::from_slice(pITRF.as_slice()).unwrap();
///
/// println!("latitude = {} deg", itrf.latitude_deg());
/// println!("longitude = {} deg", itrf.longitude_deg());
/// println!("altitude = {} m", itrf.hae());
///
/// ```
///
///
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TLE {
    /// Name of satellite
    pub name: String,
    /// String describing launch
    pub intl_desig: String,
    /// Satellite NORAD number
    pub sat_num: i32,
    /// Launch year
    pub desig_year: i32,
    /// Numbered launch of year
    pub desig_launch: i32,
    /// Piece of launch
    pub desig_piece: String,
    /// TLE epoch
    pub epoch: Instant,
    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    pub mean_motion_dot: f64,
    /// One sixth of 2nd derivative of mean motion wrt tim, in revs/day^3
    pub mean_motion_dot_dot: f64,
    /// Starred ballistic coefficient, in units of inverse Earth radii
    pub bstar: f64,
    /// Ephemeris type (line 1, column 63). Usually 0. A value of 4 marks
    /// an SGP4-XP element set, which [`sgp4`](crate::sgp4::sgp4) rejects:
    /// its line 1 stores agom and a B term where a classic TLE stores
    /// nddot and B*, and satkit implements classic SGP4 only.
    pub ephem_type: u8,
    /// Bulliten number
    pub element_num: i32,
    /// Inclination, degrees
    pub inclination: f64,
    /// Right ascension of ascending node, degrees
    pub raan: f64,
    /// Eccentricity
    pub eccen: f64,
    /// Argument of perigee, degrees
    pub arg_of_perigee: f64,
    /// Mean anomaly, degrees
    pub mean_anomaly: f64,
    /// Mean motion, revs / day
    pub mean_motion: f64,
    /// Revolution number
    pub rev_num: i32,

    pub(crate) satrec: SatRecCache,
}

/// The SGP4 initialization cached in a [`TLE`]. It is derived data, not part
/// of the element set, so it never affects comparisons: a TLE equals its
/// pickle, clone or a freshly parsed copy whether or not it has been
/// propagated.
#[derive(Clone, Debug, Default)]
pub(crate) struct SatRecCache(pub(crate) Option<SatRec>);

impl PartialEq for SatRecCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl PartialOrd for SatRecCache {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        Some(std::cmp::Ordering::Equal)
    }
}

impl SGP4Source for TLE {
    fn epoch(&self) -> Instant {
        self.epoch
    }

    fn satrec_mut(&mut self) -> &mut Option<SatRec> {
        &mut self.satrec.0
    }

    fn sgp4_init_args(&self) -> crate::sgp4::Result<SGP4InitArgs> {
        // An SGP4-XP line 1 puts agom and a B term where a classic TLE has
        // nddot and B*; classic SGP4 would run on them and be silently wrong.
        if self.ephem_type == 4 {
            return Err(crate::sgp4::Error::source(Error::UnsupportedEphemerisType(
                self.ephem_type,
            )));
        }
        Ok(SGP4InitArgs::from_mean_elements(
            self.epoch,
            self.bstar,
            self.mean_motion,
            self.mean_motion_dot,
            self.mean_motion_dot_dot,
            self.eccen,
            self.inclination,
            self.raan,
            self.arg_of_perigee,
            self.mean_anomaly,
        ))
    }
}

impl TLE {
    /// Parse every TLE record in `lines`: [`Self::records`], collected.
    /// See [`Self::records`] for how lines are grouped into 2-line and
    /// 3-line records.
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::TLE;
    ///
    /// let lines = vec![
    ///     "2 PATHFINDER".to_string(),
    ///     "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995".to_string(),
    ///     "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085".to_string(),
    ///     "0 SHINSEI (MS-F2)".to_string(),
    ///     "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992".to_string(),
    ///     "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065".to_string(),
    /// ];
    ///
    /// let tles = TLE::from_lines(&lines).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Stops at the first record that fails to parse, returning its
    /// [`Error::Record`]. To keep the good records of a file with a few bad
    /// ones, iterate [`Self::records`] instead.
    pub fn from_lines(lines: &[String]) -> Result<Vec<Self>> {
        Self::records(lines).collect()
    }

    /// Parse TLE records one at a time from a sequence of lines.
    ///
    /// [`Self::from_lines`] (and `from_url`) collect this iterator, so what
    /// follows applies to them too. Trailing whitespace is trimmed, a
    /// line of at least 69 characters starting with `"1 "` or `"2 "` is a
    /// data line, and any other non-empty line before a line 2 is the
    /// satellite name. Each item is one record; a record that fails to
    /// parse yields an [`Error::Record`] naming the input line it starts on
    /// and its satellite, and the iterator carries on with the next record.
    ///
    /// Checksums are not verified unless requested with
    /// [`Records::check_checksums`].
    ///
    /// # Example
    ///
    /// Keep the good records of a file and skip the malformed ones:
    ///
    /// ```
    /// use satkit::TLE;
    ///
    /// let lines = [
    ///     "0 SHINSEI (MS-F2)",
    ///     "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992",
    ///     "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065",
    ///     "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995",
    ///     "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085",
    /// ];
    ///
    /// // Skip records that fail to parse
    /// let tles: Vec<TLE> = TLE::records(lines).filter_map(Result::ok).collect();
    /// assert_eq!(tles.len(), 2);
    ///
    /// // Or report them, with the input line each one starts on
    /// for rec in TLE::records(lines).check_checksums(true) {
    ///     if let Err(e) = rec {
    ///         eprintln!("skipping: {e}");
    ///     }
    /// }
    /// ```
    pub fn records<I>(lines: I) -> Records<I::IntoIter>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        Records {
            lines: lines.into_iter(),
            line_no: 0,
            name: None,
            line1: None,
            check_checksums: false,
        }
    }

    /// Load TLE(s) from a URL
    ///
    /// Fetches `url` and parses the plain-text response as
    /// [`Self::from_lines`] does; an [`Error::Record`]'s line number counts
    /// lines of the response.
    ///
    /// Requires the `download` Cargo feature. Returns [`Error::Offline`] without
    /// opening a connection when offline mode is on ([`crate::utils::set_offline`]
    /// or `SATKIT_OFFLINE`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use satkit::TLE;
    ///
    /// # #[cfg(feature = "download")]
    /// let tles = TLE::from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle").unwrap();
    /// ```
    #[cfg(feature = "download")]
    pub fn from_url(url: &str) -> Result<Vec<Self>> {
        // The same offline check (and reason) as the data downloads
        if let Some(reason) = crate::utils::download::offline_reason() {
            return Err(Error::Offline {
                url: url.to_string(),
                reason,
            });
        }
        let agent = crate::utils::download::http_agent();
        let mut resp =
            agent.get(url).call().map_err(
                |e| match crate::utils::download::celestrak_throttle_hint(url, &e) {
                    Some(msg) => Error::HttpThrottled(msg),
                    None => Error::Http(e),
                },
            )?;
        let body = resp.body_mut().read_to_string()?;
        Self::records(body.lines()).collect()
    }

    ///
    /// Return a default empty TLE.  Note that values are invalid.
    ///
    pub fn new() -> Self {
        Self {
            name: "none".to_string(),
            intl_desig: "".to_string(),
            sat_num: 0,
            desig_year: 0,
            desig_launch: 0,
            desig_piece: "A".to_string(),
            epoch: Instant::J2000,
            mean_motion_dot: 0.0,
            mean_motion_dot_dot: 0.0,
            bstar: 0.0,
            ephem_type: b'U',
            element_num: 0,
            inclination: 0.0,
            raan: 0.0,
            eccen: 0.0,
            arg_of_perigee: 0.0,
            mean_anomaly: 0.0,
            mean_motion: 0.0,
            rev_num: 0,
            satrec: SatRecCache(None),
        }
    }

    /// Discards the cached SGP4 initialization.
    ///
    /// Never required for correctness: SGP4 re-initializes on its own when
    /// the elements, gravity model or ops mode differ from those the cache
    /// was built with. This only frees the cached state.
    pub fn reset_cache(&mut self) {
        self.satrec = SatRecCache(None);
    }

    /// Parse one TLE from a name line and its two data lines
    ///
    /// # Arguments:
    ///
    ///   * `line0` - the "0"th line of the TLE, which sometimes contains
    ///     the satellite name
    ///   * `line1` - the 1st line of TLE
    ///   * `line2` - the 2nd line of the TLE
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::TLE;
    /// let line0: &str = "0 INTELSAT 902";
    /// let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
    /// let line2: &str = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
    /// let tle = TLE::load_3line(&line0.to_string(),
    ///     &line1.to_string(),
    ///     &line2.to_string()
    ///     ).unwrap();
    ///
    /// ```
    ///
    pub fn load_3line(line0: &str, line1: &str, line2: &str) -> Result<Self> {
        if line1.len() < 69 || line2.len() < 69 {
            return Err(Error::InvalidLineLengths {
                line1: line1.len(),
                line2: line2.len(),
            });
        }

        match Self::load_2line(line1, line2) {
            Ok(mut tle) => {
                // Strip the "0 " name-line prefix if present. Use `strip_prefix`
                // rather than byte slicing so a non-ASCII satellite name cannot
                // panic on a char boundary.
                tle.name = line0.strip_prefix("0 ").unwrap_or(line0).to_string();
                Ok(tle)
            }
            Err(e) => Err(e),
        }
    }

    /// Parse one TLE from its two data lines (its name is set to `"none"`)
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::TLE;
    /// let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
    /// let line2: &str = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
    /// let tle = TLE::load_2line(
    ///     &line1.to_string(),
    ///     &line2.to_string()
    ///     ).unwrap();
    ///
    /// ```
    ///
    pub fn load_2line(line1: &str, line2: &str) -> Result<Self> {
        // The field extraction below slices by byte range and indexes by
        // character position, both of which are only valid (and panic-free)
        // when the lines are pure ASCII. Reject non-ASCII up front.
        if !line1.is_ascii() {
            return Err(Error::NonAscii { line: 1 });
        }
        if !line2.is_ascii() {
            return Err(Error::NonAscii { line: 2 });
        }
        if line1.len() < 69 {
            return Err(Error::LineTooShort {
                line: 1,
                got: line1.len(),
            });
        }
        if line2.len() < 69 {
            return Err(Error::LineTooShort {
                line: 2,
                got: line2.len(),
            });
        }

        // Helper for converting parse errors into ParseField variants.
        fn parse_field<F: std::str::FromStr>(s: &str, field: &'static str) -> Result<F>
        where
            F::Err: std::fmt::Display,
        {
            s.parse::<F>().map_err(|e| Error::ParseField {
                field,
                message: e.to_string(),
            })
        }

        // The implied-exponent fields (nddot, bstar) can parse to infinity for
        // out-of-range exponents ("parse" returns Ok(inf), not Err), which
        // would overflow later when re-encoding the TLE. Reject them here.
        fn finite_field(v: f64, field: &'static str) -> Result<f64> {
            if v.is_finite() {
                Ok(v)
            } else {
                Err(Error::ParseField {
                    field,
                    message: format!("value {v} is not finite"),
                })
            }
        }

        let mut year: u32 = {
            let mut mstr: String = "1".to_owned();
            mstr.push_str(&line1[18..20]);
            let mut s: u32 = parse_field(&mstr, "year")?;
            s -= 100;
            s
        };
        // See: https://celestrak.org/columns/v04n03/
        // Years >= 1957 = 1900s
        // Years < 1957 = 2000s
        let century = if year >= 57 { 1900 } else { 2000 };
        year += century;
        let day_of_year: f64 = parse_field(&line1[20..32], "day of year")?;
        // An unbounded value would overflow the epoch arithmetic below.
        if !(1.0..367.0).contains(&day_of_year) {
            return Err(Error::ParseField {
                field: "day of year",
                message: format!("value {day_of_year} out of range [1, 367)"),
            });
        }

        // Note: day_of_year starts from 1, not zero. `add_utc_days` counts
        // 86400 s UTC days, so a 30 June leap second does not shift the
        // epoch.
        let epoch = Instant::from_date(year as i32, 1, 1)?.add_utc_days(day_of_year - 1.0);

        // A line 1 and a line 2 of different satellites (e.g. a line lost from
        // a file) would otherwise combine into a plausible hybrid element set.
        let (num1, num2) = (line1[2..7].trim(), line2[2..7].trim());
        if num1 != num2 && Self::alpha5_to_int(num1).ok() != Self::alpha5_to_int(num2).ok() {
            return Err(Error::SatNumMismatch {
                line1: num1.to_string(),
                line2: num2.to_string(),
            });
        }

        Ok(Self {
            name: "none".to_string(),
            sat_num: Self::alpha5_to_int(&line1[2..7]).map_err(|e| Error::ParseField {
                field: "satellite number",
                message: e.to_string(),
            })?,

            intl_desig: { line1[9..16].trim().to_string() },
            desig_year: { line1[9..11].trim().parse().unwrap_or(70) },
            desig_launch: { line1[11..14].trim().parse().unwrap_or_default() },
            desig_piece: parse_field(line1[14..18].trim(), "desig_piece")?,

            epoch,
            mean_motion_dot: {
                let mut mstr: String = "0".to_owned();
                mstr.push_str(&line1[34..43]);
                let mut m: f64 = parse_field(&mstr, "mean motion dot")?;
                if line1.chars().nth(33).unwrap() == '-' {
                    m *= -1.0;
                }
                m
            },
            mean_motion_dot_dot: {
                let mut mstr: String = "0.".to_owned();
                mstr.push_str(&line1[45..50]);
                mstr.push('E');
                mstr.push_str(&line1[50..53]);
                let mut m: f64 = finite_field(
                    parse_field(mstr.trim(), "mean motion dot dot")?,
                    "mean motion dot dot",
                )?;
                if line1.chars().nth(44).unwrap() == '-' {
                    m *= -1.0;
                }
                m
            },
            bstar: {
                let mut mstr: String = "0.".to_owned();
                mstr.push_str(&line1[54..59]);
                mstr.push('E');
                mstr.push_str(&line1[59..62]);
                let mut m: f64 =
                    finite_field(parse_field(mstr.trim(), "bstar (drag)")?, "bstar (drag)")?;
                if line1.chars().nth(53).unwrap() == '-' {
                    m *= -1.0;
                }
                m
            },
            ephem_type: { line1[62..63].trim().parse().unwrap_or_default() },
            element_num: parse_field(line1[64..68].trim(), "element number")?,

            inclination: parse_field(line2[8..16].trim(), "inclination")?,

            raan: parse_field(line2[17..25].trim(), "raan")?,

            eccen: {
                let mut mstr: String = "0.".to_owned();
                mstr.push_str(&line2[26..33]);
                parse_field(mstr.trim(), "eccentricity")?
            },
            arg_of_perigee: parse_field(line2[34..42].trim(), "arg of perigee")?,

            mean_anomaly: parse_field(line2[42..51].trim(), "mean anomaly")?,

            mean_motion: parse_field(line2[52..63].trim(), "mean motion")?,

            rev_num: parse_field(line2[63..68].trim(), "rev num")?,
            satrec: SatRecCache(None),
        })
    }

    /// Format this TLE back into the two canonical 69-char lines.
    ///
    /// The element set number (4 columns) and revolution number (5 columns)
    /// are written modulo 10,000 and 100,000, so a larger value wraps around
    /// the way catalog TLEs roll the revolution counter over; negative values
    /// are written as 0.
    ///
    /// # Returns:
    ///
    /// * `lines` - Result with OK value containing 2-element array of two strings representing the TLE lines
    ///
    /// # Example:
    ///
    /// ```rust
    /// let lines = [
    ///     "ISS (ZARYA)".to_string(),
    ///     "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9990".to_string(),
    ///     "2 B5544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487613".to_string(),
    /// ];
    /// // Construct the TLE from the lines
    /// let tle = satkit::TLE::from_lines(&lines).unwrap()[0].clone();
    ///
    /// // Show that we can re-create the same lines
    /// assert_eq!(tle.to_2line().unwrap()[0], lines[1]);
    /// assert_eq!(tle.to_2line().unwrap()[1], lines[2]);
    /// ```
    ///
    pub fn to_2line(&self) -> Result<[String; 2]> {
        // Epoch as (YY, DOY.fraction)
        let (yy, doy) = self.epoch_to_tle_ydoy()?;

        // Satellite number in alpha5
        let sat_alpha5 = Self::int_to_alpha5(self.sat_num)?;

        // Format ndot/2, nddot/6, bstar with correct implied fields
        let (ndot_sign, ndot_body) = tle_formatter::format_ndot(self.mean_motion_dot);
        let (nddot_sign, nddot_mant, nddot_exp2) =
            tle_formatter::format_implied(self.mean_motion_dot_dot);
        let (bstar_sign, bstar_mant, bstar_exp2) = tle_formatter::format_implied(self.bstar);

        // Ephemeris type as a single digit '0'..'9'
        let et = if (0..=9).contains(&self.ephem_type) {
            char::from(b'0' + self.ephem_type)
        } else {
            '0'
        };

        // ------- Build Line 1 -------
        let sat5 = format!("{:<5}", sat_alpha5); // cols 3-7

        // International designator triplet.
        // Last 2 digits of launch year, 3-digit launch number within year, 3-character piece identifier.
        // Never decoded, so no need to re-encode.
        let desig = format!("{:<8}", self.intl_desig); // cols 10-17

        let epoch = format!("{:0>2}{:012.8}", yy, doy); // cols 19-32
        let ndot = format!("{}{}", ndot_sign, ndot_body); // cols 34-43 (10 chars total)
        let nddot = format!("{}{}{}", nddot_sign, nddot_mant, nddot_exp2); // cols 45-52 (8 chars)
        let bstar = format!("{}{}{}", bstar_sign, bstar_mant, bstar_exp2); // cols 54-61 (8 chars)
                                                                           // Wrap to the column width (see the doc comment)
        let elem_no = format!("{:>4}", self.element_num.max(0) % 10_000); // cols 65-68

        let mut l1 = format!("1 {sat5}U {desig} {epoch} {ndot} {nddot} {bstar} {et} {elem_no}");

        let cksum1 = tle_formatter::tle_checksum(&l1);
        l1.push(char::from(b'0' + cksum1));

        // ------- Build Line 2 -------
        let incl = format!("{:8.4}", self.inclination);
        let raan = format!("{:8.4}", self.raan);
        if !(0.0..1.0).contains(&self.eccen) {
            return Err(Error::EccentricityOutOfRange(self.eccen));
        }
        let ecc7 = format!("{:0>7}", (self.eccen * 1.0e7 + 0.5).floor() as u64);
        let argp = format!("{:8.4}", self.arg_of_perigee);
        let mean_anom = format!("{:8.4}", self.mean_anomaly);
        let n = format!("{:11.8}", self.mean_motion);
        let rev = format!("{:>5}", self.rev_num.max(0) % 100_000);

        let mut l2 = format!("2 {sat_alpha5:<5} {incl} {raan} {ecc7} {argp} {mean_anom} {n}{rev}");

        // Ensure 68 chars before checksum (line-2 is stable with these widths)
        if l2.len() != 68 {
            if l2.len() < 68 {
                l2.push_str(&" ".repeat(68 - l2.len()));
            } else {
                l2.truncate(68);
            }
        }
        let cksum2 = tle_formatter::tle_checksum(&l2);
        l2.push(char::from(b'0' + cksum2));

        Ok([l1, l2])
    }

    /// Convenience: include "line 0" (name) above the two TLE lines.
    ///
    /// Format this TLE back into name line plus two canonical 69-char lines.
    ///
    /// # Returns:
    ///
    /// * `lines` - Result with OK value containing 3-element array of name line as string and
    ///   two strings representing the TLE lines
    ///
    /// # Example:
    ///
    /// ```rust
    /// let lines = [
    ///     "ISS (ZARYA)".to_string(),
    ///     "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9990".to_string(),
    ///     "2 B5544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487613".to_string(),
    /// ];
    /// // Construct the TLE from the lines
    /// let tle = satkit::TLE::from_lines(&lines).unwrap()[0].clone();
    ///
    /// // Show that we can re-create the same lines
    /// assert_eq!(tle.to_3line().unwrap()[0], lines[0]);
    /// assert_eq!(tle.to_3line().unwrap()[1], lines[1]);
    /// assert_eq!(tle.to_3line().unwrap()[2], lines[2]);
    /// ```
    //////
    pub fn to_3line(&self) -> Result<[String; 3]> {
        let [l1, l2] = self.to_2line()?;
        Ok([self.name.clone(), l1, l2])
    }

    /// Compute (two-digit year, fractional day-of-year) from epoch.
    fn epoch_to_tle_ydoy(&self) -> Result<(u8, f64)> {
        let (year, _, _, _, _, _) = self.epoch.as_datetime();

        if !(1957..=2056).contains(&year) {
            return Err(Error::YearOutOfRange(year));
        }

        // Day-of-year.
        let doy_int = self.epoch.day_of_year();

        // Fraction of day.
        // Note: inside a leap second the UTC MJD repeats 23:59:59.x (a TLE
        // day fraction cannot express 23:59:60).
        let frac = self.epoch.as_mjd_utc() % 1.0;
        let doy = (doy_int as f64) + frac;
        // Years >= 1957 = 1900s
        // Years < 1957 = 2000s
        // See: https://celestrak.org/columns/v04n03/
        let century = if year >= 1957 { 1900 } else { 2000 };
        let year = ((year - century) % 100) as u8;
        Ok((year, doy))
    }

    /// Convert an alpha5 formated Satellite Catalog Number, also known as NORAD ID, to a plain
    /// numerical ID.
    ///
    /// 5 digit NORAD IDs are getting exhausted while many formats, like TLE, rely on them being
    /// limited to 5 characters. Thus the introduction of the alpha 5 format.
    ///
    /// Up to number 99999 plain numerical id and alpha5 are identicial. Starting with 100000 the
    /// alpha5 string uses a character instead of the first digit to handle satellite numbers
    /// in the 100000 to 339999 range.
    /// 'I' and 'O' are not part of the allowed chars to avoid any confusion with 0 or 1
    ///
    /// # Arguments:
    ///  * `alpha5` - a reference to a str representing an alpha5 encoded satellite number.
    ///
    /// # Returns:
    ///  * An i32 of the plain numerical satellite number or string indicating error condition
    ///
    /// # Example
    /// ```
    /// use satkit::TLE;
    ///
    /// let sat_num = TLE::alpha5_to_int("S9994");
    /// // sat_num has the value 269994
    /// ```
    pub fn alpha5_to_int(alpha5: &str) -> Result<i32> {
        match alpha5.chars().nth(0) {
            // Alpha char is only possible at the first position, so if the first char is a
            // digit or a whitespace the standard `.parse()` can be used.
            Some(c) if c.is_ascii_digit() || c.is_whitespace() => match alpha5.trim().parse() {
                Ok(i) if i >= 0 => Ok(i),
                Ok(_) => Err(Error::InvalidSatNumValue),
                Err(e) => Err(Error::InvalidSatNum(format!("{e}"))),
            },
            Some(c) if c.is_alphabetic() => {
                match ALPHA5_MATCHING
                    .chars()
                    .position(|m| m == c.to_ascii_uppercase())
                {
                    Some(p) => match alpha5[1..].parse::<i32>() {
                        Ok(i) => Ok((p as i32 + 10) * 10000 + i),
                        Err(e) => Err(Error::InvalidSatNum(format!("{e}"))),
                    },
                    None => Err(Error::InvalidFirstDigit(c)),
                }
            }
            Some(c) => Err(Error::InvalidFirstDigit(c)),
            None => Err(Error::EmptySatNum),
        }
    }

    /// Convert a numerical Satellite Catalog Number, also known as NORAD ID, to an alpha5 String.
    ///
    /// 5 digit NORAD IDs are getting exhausted while many formats, like TLE, rely on them being
    /// limited to 5 characters. Thus the introduction of the alpha 5 format.
    ///
    /// Up to number 99999 plain numerical id and alpha5 are identicial. Starting with 100000 the
    /// alpha5 string uses a character instead of the first digit to handle satellite numbers
    /// in the 100000 to 339999 range.
    /// 'I' and 'O' are not part of the allowed chars to avoid any confusion with 0 or 1
    ///
    /// # Arguments:
    ///  * `sat_num` - An i32 of a plain numerical satellite number
    ///
    /// # Returns:
    ///   * A String representing an alpha5 encoded satellite number or string indicating error
    ///     condition
    ///
    /// # Example
    /// ```
    /// use satkit::TLE;
    ///
    /// let alpha5_sat_num = TLE::int_to_alpha5(269994);
    /// // alpha5_sat_num has the String value "S9994"
    /// ```
    pub fn int_to_alpha5(sat_num: i32) -> Result<String> {
        match sat_num {
            i @ 0..=99999 => Ok(format!("{:0>5}", i)),
            i @ 100000..=339999 => {
                let c = ALPHA5_MATCHING
                    .chars()
                    .nth(i as usize / 10000 - 10)
                    .unwrap();
                Ok(format!("{c}{:0>4}", i % 10000))
            }
            _i @ 340000.. => Err(Error::SatNumTooLargeForAlpha5),
            _ => Err(Error::InvalidSatNumValue),
        }
    }

    /// Return a string representation of the TLE
    /// in a human-readable format
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::TLE;
    /// let line0: &str = "0 INTELSAT 902";
    /// let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
    /// let line2: &str = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981";
    /// let tle = TLE::load_3line(&line0.to_string(),
    ///    &line1.to_string(),
    ///   &line2.to_string()
    /// ).unwrap();
    /// println!("{}", tle.to_pretty_string());
    /// ```
    ///
    pub fn to_pretty_string(&self) -> String {
        format!(
            r#"
            TLE: {}
                         NORAD ID: {},
                      Launch Year: {},
                            Epoch: {},
                  Mean Motion Dot: {} revs / day^2,
              Mean Motion Dot Dot: {} revs / day^3,
                             Drag: {},
                      Inclination: {} deg,
                             RAAN: {} deg,
                            eccen: {},
                   Arg of Perigee: {} deg,
                     Mean Anomaly: {} deg,
                      Mean Motion: {} revs / day
                            Rev #: {}
        "#,
            self.name,
            // Fall back to the raw number so Display never panics, even for a
            // sat_num that has no alpha5 representation.
            Self::int_to_alpha5(self.sat_num).unwrap_or_else(|_| self.sat_num.to_string()),
            match self.desig_year > 50 {
                true => self.desig_year + 1900,
                false => self.desig_year + 2000,
            },
            self.epoch,
            self.mean_motion_dot * 2.0,
            self.mean_motion_dot_dot * 6.0,
            self.bstar,
            self.inclination,
            self.raan,
            self.eccen,
            self.arg_of_perigee,
            self.mean_anomaly,
            self.mean_motion,
            self.rev_num,
        )
    }
}

/// Iterator over the TLE records in a sequence of lines, returned by
/// [`TLE::records`]. Each item is one parsed record, or an
/// [`Error::Record`] locating the one that failed.
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Records<I> {
    lines: I,
    /// 1-based number of the last line read
    line_no: usize,
    /// Pending name line and its line number
    name: Option<(usize, String)>,
    /// Pending line 1 and its line number
    line1: Option<(usize, String)>,
    check_checksums: bool,
}

impl<I> Records<I> {
    /// Also verify the checksum digit (column 69) of both data lines of
    /// every record; a mismatch yields [`Error::ChecksumMismatch`] (wrapped
    /// in [`Error::Record`]) naming the expected and actual digit. Off by
    /// default: element sets that were hand-edited or generated with a
    /// stale checksum are common and otherwise parse fine.
    ///
    /// ```
    /// use satkit::TLE;
    ///
    /// let lines = [
    ///     "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
    ///     "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617",
    /// ];
    /// // The checksum of line 2 is 5, not 7
    /// assert!(TLE::records(lines).all(|r| r.is_ok()));
    /// let err = TLE::records(lines).check_checksums(true).next().unwrap().unwrap_err();
    /// assert!(err.to_string().contains("checksum"));
    /// ```
    pub fn check_checksums(mut self, on: bool) -> Self {
        self.check_checksums = on;
        self
    }
}

impl<I> Iterator for Records<I>
where
    I: Iterator,
    I::Item: AsRef<str>,
{
    type Item = Result<TLE>;

    fn next(&mut self) -> Option<Result<TLE>> {
        while let Some(raw) = self.lines.next() {
            self.line_no += 1;
            // Trim trailing whitespace so CRLF-terminated files (trailing
            // `\r`) don't push line lengths off 69. A TLE data line is >= 69
            // chars (extra trailing content is ignored by `load_2line`) with
            // a `"1 "` / `"2 "` prefix; `starts_with` is byte-safe on
            // non-ASCII.
            let mut line = raw.as_ref().trim_end();
            if self.line_no == 1 {
                // A UTF-8 byte-order mark (written by some Windows editors)
                // is not part of the first line
                line = line.strip_prefix('\u{feff}').unwrap_or(line);
            }
            if line.len() >= 69 && line.starts_with("2 ") {
                let name = self.name.take();
                let line1 = self.line1.take();
                return Some(parse_record(
                    name,
                    line1,
                    (self.line_no, line),
                    self.check_checksums,
                ));
            }
            if line.is_empty() {
                continue;
            }
            // Its line 2 must come next: a pending line 1 followed by
            // anything else is a record of its own, and an error
            let orphan = self
                .line1
                .take()
                .map(|l1| orphan_line1(self.name.take(), l1, Some((self.line_no, line))));
            if line.len() >= 69 && line.starts_with("1 ") {
                self.line1 = Some((self.line_no, line.to_string()));
            } else {
                self.name = Some((self.line_no, line.to_string()));
            }
            if let Some(err) = orphan {
                return Some(Err(err));
            }
        }
        // The input ended after a line 1
        self.line1
            .take()
            .map(|l1| Err(orphan_line1(self.name.take(), l1, None)))
    }
}

/// Why a line read as a satellite name may be a TLE data line in disguise
/// (so its record will fail): e.g. "looks like a line 1 but is 68
/// characters; a TLE line is 69". `None` for an ordinary name.
fn data_line_lookalike(line: &str) -> Option<String> {
    let (body, why) = if let Some(b) = line.strip_prefix('\u{feff}') {
        (b, Some("starts with a UTF-8 byte-order mark".to_string()))
    } else if line.starts_with(char::is_whitespace) {
        (
            line.trim_start(),
            Some("starts with whitespace".to_string()),
        )
    } else {
        (line, None)
    };
    let which = if body.starts_with("1 ") {
        1
    } else if body.starts_with("2 ") {
        2
    } else {
        return None;
    };
    // A name can start with "1 "; a data line is ~69 characters
    if body.len() < 60 {
        return None;
    }
    let why = why.or_else(|| {
        (body.len() < 69).then(|| format!("is {} characters; a TLE line is 69", body.len()))
    })?;
    Some(format!("looks like a line {which} but {why}"))
}

/// Satellite description for [`Error::Record`]: the number (as written,
/// possibly alpha5) from the first data line that has one, and the name.
fn record_sat(data_lines: &[&str], name: Option<&(usize, String)>) -> Option<String> {
    let num = data_lines
        .iter()
        .filter_map(|l| l.get(2..7))
        .map(str::trim)
        .find(|s| !s.is_empty());
    let nm = name.map(|(_, n)| n.strip_prefix("0 ").unwrap_or(n).trim());
    match (num, nm) {
        (Some(n), Some(m)) => Some(format!("{n} \"{m}\"")),
        (Some(n), None) => Some(n.to_string()),
        (None, Some(m)) => Some(format!("\"{m}\"")),
        (None, None) => None,
    }
}

/// The [`Error::Record`] for a line 1 that no line 2 follows. `next` is the
/// line that ended the record (`None` at the end of the input).
fn orphan_line1(
    name: Option<(usize, String)>,
    line1: (usize, String),
    next: Option<(usize, &str)>,
) -> Error {
    let hint = next.and_then(|(n, l)| {
        data_line_lookalike(l).map(|why| format!("line {n}, which follows it, {why}"))
    });
    Error::Record {
        line: name.as_ref().map_or(line1.0, |(n, _)| *n),
        sat: record_sat(&[&line1.1], name.as_ref()),
        hint,
        error: Box::new(Error::MissingLine2),
    }
}

/// Parse one grouped record, wrapping any error in [`Error::Record`].
fn parse_record(
    name: Option<(usize, String)>,
    line1: Option<(usize, String)>,
    line2: (usize, &str),
    check_checksums: bool,
) -> Result<TLE> {
    let l1 = line1.as_ref().map_or("", |(_, s)| s.as_str());
    let l2 = line2.1;
    let parsed = if line1.is_none() {
        Err(Error::MissingLine1)
    } else {
        match &name {
            None => TLE::load_2line(l1, l2),
            Some((_, n)) => TLE::load_3line(n, l1, l2),
        }
    }
    .and_then(|tle| {
        if check_checksums {
            verify_checksum(l1, 1)?;
            verify_checksum(l2, 2)?;
        }
        Ok(tle)
    });

    parsed.map_err(|error| {
        let data_lines = [line1.as_ref().map(|(n, s)| (*n, s.as_str())), Some(line2)];
        let start = name
            .as_ref()
            .map(|(n, _)| *n)
            .into_iter()
            .chain(data_lines.iter().flatten().map(|(n, _)| *n))
            .min()
            .unwrap_or(line2.0);

        let mut hints: Vec<String> = Vec::new();
        // A data line that is not quite one (68 characters, a leading space
        // or byte-order mark) is read as the name, leaving its record short
        // of a line
        if let Some((n, why)) = name
            .as_ref()
            .and_then(|(n, s)| data_line_lookalike(s).map(|why| (n, why)))
        {
            hints.push(format!("line {n}, read as the satellite name, {why}"));
        }
        // A line longer than 69 characters is accepted (the extra is
        // ignored), but when a field then fails to parse, the likely cause
        // is a field one column too wide shifting the rest of the line.
        if matches!(
            error,
            Error::ParseField { .. } | Error::ChecksumMismatch { .. }
        ) {
            let long: Vec<String> = data_lines
                .iter()
                .flatten()
                .filter(|(_, s)| s.len() > 69)
                .map(|(n, s)| format!("line {n} is {} characters", s.len()))
                .collect();
            if !long.is_empty() {
                hints.push(format!(
                    "{}; a TLE line is 69, so its columns may be shifted",
                    long.join(", ")
                ));
            }
        }

        Error::Record {
            line: start,
            sat: record_sat(&[l1, l2], name.as_ref()),
            hint: (!hints.is_empty()).then(|| hints.join("; ")),
            error: Box::new(error),
        }
    })
}

/// Check the mod-10 checksum in column 69 of a (parsed, so ASCII and at
/// least 69 characters long) TLE data line.
fn verify_checksum(line: &str, which: u8) -> Result<()> {
    let expected = tle_formatter::tle_checksum(line);
    let found = line.as_bytes()[68] as char;
    if found.to_digit(10) == Some(u32::from(expected)) {
        Ok(())
    } else {
        Err(Error::ChecksumMismatch {
            line: which,
            expected,
            found,
        })
    }
}

impl Default for TLE {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TLE {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_pretty_string())
    }
}

mod tle_formatter {

    /// Format ndot/2 as sign + ".dddddddd" (10 cols split as [sign][9-body])
    pub fn format_ndot(v: f64) -> (char, String) {
        let sign = if v < 0.0 { '-' } else { ' ' };
        let mut body = format!("{:.8}", v.abs());
        if let Some(stripped) = body.strip_prefix('0') {
            body = stripped.to_string(); // turn "0.xxxxxxxx" into ".xxxxxxxx"
        }
        // ensure width 9 (".dddddddd")
        if body.len() < 9 {
            body = format!("{:>9}", body);
        } else if body.len() > 9 {
            body.truncate(9);
        }
        (sign, body)
    }

    /// Format value for implied-exponent fields (nddot/6 and bstar).
    /// Returns (sign, mantissa[5], exp[2 with sign]) per TLE ("MMMMM±E", where E is 0..9).
    pub fn format_implied(v: f64) -> (char, String, String) {
        if v == 0.0 {
            // Exact zero as " 00000-0"
            return (' ', "00000".to_string(), "-0".to_string());
        }
        let sign = if v < 0.0 { '-' } else { ' ' };
        let x = v.abs();

        // Represent v ≈ mant * 10^(e - 5) with mant in [0, 99999]
        let mut e10 = x.log10().floor() as i32; // base-10 exponent
        let mut mant = (x / 10f64.powi(e10) * 1.0e4).round() as i64;

        // Normalize if rounding pushed mant to 100000
        if mant == 100_000 {
            mant = 10_000;
            e10 += 1;
        }

        // TLE stores a single-digit exponent with sign: "±d"
        // e = e10 (we already accounted for mant being *1e5)
        let e = e10 + 1;
        let mant_s = format!("{:0>5}", mant.max(0));

        // Clamp to displayable range [-9, 9]; real TLEs fit this for these fields
        let e_clamped = e.clamp(-9, 9);
        let exp_s = format!("{:+}", e_clamped);

        (sign, mant_s, exp_s)
    }

    /// Compute the TLE checksum (mod 10) over the first 68 characters.
    pub fn tle_checksum(s: &str) -> u8 {
        let mut sum: u32 = 0;
        for (i, c) in s.chars().enumerate() {
            if i >= 68 {
                break;
            }
            sum += match c {
                '0'..='9' => c as u32 - '0' as u32,
                '-' => 1,
                _ => 0,
            };
        }
        (sum % 10) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{bail, Result};

    #[test]
    fn testload() -> Result<()> {
        let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
        let line2: &str =
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
        let line0: &str = "0 INTELSAT 902";
        match TLE::load_3line(line0, line1, line2) {
            Ok(_t) => {}

            Err(s) => {
                bail!("load_3line: Err = \"{}\"", s);
            }
        }
        match TLE::load_2line(line1, line2) {
            Ok(_t) => {}
            Err(s) => {
                bail!("load_2line: Err = \"{}\"", s);
            }
        }
        Ok(())
    }

    #[test]
    fn test_non_ascii_line_errors_not_panics() {
        // A non-ASCII line long enough to pass the length check must return a
        // clean error rather than panicking on a non-char-boundary byte slice.
        let line1 =
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  829é".to_string();
        let line2 = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."
            .to_string();
        assert!(matches!(
            TLE::load_2line(&line1, &line2),
            Err(Error::NonAscii { line: 1 })
        ));
        // A single multibyte-character line must not panic in from_lines
        // (regression: the old first-char indexing unwrapped past a 1-char
        // line). With no TLE data lines present, the result is empty.
        let lines = vec!["é".to_string()];
        assert!(TLE::from_lines(&lines).unwrap().is_empty());
    }

    #[test]
    fn test_malformed_numeric_fields_error_not_panic() {
        let line1 = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
        let line2 = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";

        // Unbounded day-of-year must error rather than overflow the epoch math
        let mut l1 = line1.to_string();
        l1.replace_range(20..32, "-99999999999");
        assert!(TLE::load_2line(&l1, line2).is_err());

        // Negative satellite number must be rejected at parse time; it has no
        // alpha5 representation and used to panic later in Display
        let mut l1 = line1.to_string();
        l1.replace_range(2..7, "  -99");
        assert!(TLE::load_2line(&l1, line2).is_err());

        // An out-of-range implied exponent parses to inf ("parse" returns
        // Ok(inf)); it must error here rather than overflow in to_2line
        let mut l1 = line1.to_string();
        l1.replace_range(54..62, "99999999");
        assert!(TLE::load_2line(&l1, line2).is_err());
    }

    #[test]
    fn test_display_never_panics_on_bad_satnum() {
        let mut tle = TLE::new();
        tle.sat_num = -99;
        // Display / to_pretty_string must fall back rather than panic
        assert!(tle.to_pretty_string().contains("-99"));
    }

    #[test]
    fn test_from_lines_crlf() -> Result<()> {
        // Lines with a trailing carriage return (CRLF files) must still parse
        // rather than being silently dropped by an exact length == 69 check.
        let line0 = "0 INTELSAT 902\r".to_string();
        let line1 =
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290\r".to_string();
        let line2 =
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.\r"
                .to_string();
        let tles = TLE::from_lines(&[line0, line1, line2])?;
        assert_eq!(tles.len(), 1);
        assert_eq!(tles[0].name, "INTELSAT 902");
        Ok(())
    }

    #[test]
    fn test_from_lines() -> Result<()> {
        let lines = vec![
            "2023-193D".to_string(),
            "1 58556U 23193D   25003.79555039  .00279397  31144-4  86159-3 0  9996".to_string(),
            "2 58556  97.2472  26.1173 0004235 271.4738  88.6051 15.91743157 60937".to_string(),
            "0 CPOD FLT2 (TYVAK-0033)".to_string(),
            "1 52780U 22057BB  23036.86744141  .00018086  00000-0  87869-3 0  9991".to_string(),
            "2 52780  97.5313 154.3283 0011660  53.1934 307.0368 15.18441019 16465".to_string(),
            "1998-067WV".to_string(),
            "1 60955U 98067WV  24295.33823779  .06453473  12009-4  26290-2 0  9998".to_string(),
            "2 60955  51.6166  43.0490 0010894 336.3668  23.6849 16.22453324  8315".to_string(),
            "2 PATHFINDER".to_string(),
            "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995".to_string(),
            "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085".to_string(),
            "0 SHINSEI (MS-F2)".to_string(),
            "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992".to_string(),
            "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065".to_string(),
            "OSCAR 7 (AO-7)".to_string(),
            "1 07530U 74089B   24323.87818483 -.00000039  00000+0  47934-4 0  9997".to_string(),
            "2 07530 101.9893 320.0351 0012269 147.9195 274.9996 12.53682684288423".to_string(),
            "1 52743U 22057M   23037.04954473  .00011781  00000-0  61944-3 0  9993".to_string(),
            "2 52743  97.5265 153.6940 0008594  82.9904  31.3082 15.15793680 38769".to_string(),
            "0 ISS (ZARYA)".to_string(),
            "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992".to_string(),
            "2 B5544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615".to_string(), // Note: Invalid checksum.
            "0 ISS (ZARYA)".to_string(),
            "1 Z9999U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992".to_string(),
            "2 Z9999  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615".to_string(), // Note: Invalid checksum.
        ];

        let tles = match TLE::from_lines(&lines) {
            Ok(t) => t,
            Err(s) => {
                bail!("load_lines: Err = \"{}\"", s);
            }
        };

        if tles.len() != 9 {
            bail!("load_lines: Err = \"Incorrect number of elements parsed\"");
        }

        if tles[0].name != "2023-193D" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[0].name
            );
        }

        if tles[1].name != "CPOD FLT2 (TYVAK-0033)" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[1].name
            );
        }

        if tles[2].name != "1998-067WV" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[2].name
            );
        }

        if tles[3].name != "2 PATHFINDER" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[3].name
            );
        }

        if tles[4].name != "SHINSEI (MS-F2)" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[4].name
            );
        }

        if tles[4].sat_num != 5485 {
            bail!(
                "load_lines: Err = \"Error parsing sat num {}\"",
                tles[4].sat_num
            );
        }

        if tles[5].name != "OSCAR 7 (AO-7)" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[5].name
            );
        }

        if tles[5].sat_num != 7530 {
            bail!(
                "load_lines: Err = \"Error parsing sat num {}\"",
                tles[5].sat_num
            );
        }

        if tles[6].name != "none" {
            bail!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[6].name
            );
        }

        Ok(())
    }

    /// An SGP4-XP element set (ephemeris type 4) parses, but SGP4 refuses
    /// to propagate it: its drag columns do not mean what classic SGP4
    /// expects. The lines are from the Astro Standards sample XP catalog,
    /// with checksums appended.
    #[test]
    fn test_sgp4xp_type4_rejected() -> Result<()> {
        let line1 = "1 00011U 59001A   23060.12028874 +.00002871  89876-2  73526-1 4 00010";
        let line2 = "2 00011  32.8652 309.4507 1466152  63.9843 312.2337 11.85947359392148";
        let mut tle = TLE::from_lines(&[line1.to_string(), line2.to_string()])?
            .pop()
            .unwrap();
        assert_eq!(tle.ephem_type, 4);
        let epoch = tle.epoch;
        let err = match crate::sgp4::sgp4(&mut tle, &[epoch]) {
            Ok(_) => panic!("type-4 element set must not propagate"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("SGP4-XP"), "unexpected message: {msg}");
        assert!(msg.contains("type 4"), "unexpected message: {msg}");
        Ok(())
    }

    #[test]
    fn test_from_invalid_from_lines() -> Result<()> {
        let res = TLE::from_lines(&[
            "0 INVALID TLE".to_string(),
            "1 12345U 67890A 12345.67890123  .00000123  00000-0  12345-6 0  9992".to_string(),
            "2 12345  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615".to_string(),
        ]);
        // The short line 1 is read as the name, leaving line 2 without a
        // line 1; the hint says what happened
        let msg = res.expect_err("short line 1 must fail").to_string();
        assert!(msg.contains("Line 2 without a line 1"), "{msg}");
        assert!(
            msg.contains("line 2, read as the satellite name, looks like a line 1 but is 67 characters; a TLE line is 69"),
            "{msg}"
        );

        Ok(())
    }

    // Two good records around one whose line 2 has an 8-digit eccentricity,
    // shifting every later column right by one (70 characters).
    fn lines_with_shifted_record() -> Vec<String> {
        [
            "0 SHINSEI (MS-F2)",
            "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992",
            "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065",
            "",
            "0 SHIFTED",
            "1 58556U 23193D   25003.79555039  .00279397  31144-4  86159-3 0  9996",
            "2 58556  97.2472  26.1173 00042351 271.4738  88.6051 15.91743157 60937",
            "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995",
            "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn test_records_locates_shifted_line() {
        let lines = lines_with_shifted_record();
        let recs: Vec<_> = TLE::records(&lines).collect();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].as_ref().unwrap().sat_num, 5485);
        assert_eq!(recs[2].as_ref().unwrap().sat_num, 45727);

        let err = recs[1].as_ref().unwrap_err();
        match err {
            Error::Record {
                line,
                sat,
                hint,
                error,
            } => {
                assert_eq!(*line, 5);
                assert_eq!(sat.as_deref(), Some("58556 \"SHIFTED\""));
                assert_eq!(
                    hint.as_deref(),
                    Some(
                        "line 7 is 70 characters; a TLE line is 69, so its columns may be shifted"
                    )
                );
                assert!(matches!(
                    **error,
                    Error::ParseField {
                        field: "mean anomaly",
                        ..
                    }
                ));
            }
            other => panic!("expected Error::Record, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.starts_with("TLE record starting at line 5 (sat 58556 \"SHIFTED\"): Could not parse mean anomaly"), "{msg}");
        assert!(msg.contains("line 7 is 70 characters"), "{msg}");

        // Skipping bad records keeps the good ones
        let good: Vec<TLE> = TLE::records(&lines).filter_map(|r| r.ok()).collect();
        assert_eq!(good.len(), 2);
        assert_eq!(good[0].name, "SHINSEI (MS-F2)");
        assert_eq!(good[1].name, "none");
    }

    #[test]
    fn test_from_lines_strict() {
        // from_lines stops at the first bad record, with the same located error
        let err = TLE::from_lines(&lines_with_shifted_record()).unwrap_err();
        assert!(matches!(err, Error::Record { line: 5, .. }), "{err:?}");
        assert!(err.to_string().contains("line 7 is 70 characters"));
    }

    #[test]
    fn test_record_error_without_name_or_long_line() {
        // A line 2 with no line 1: located on the line 2 itself, no hint
        let lines = [
            "",
            "",
            "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085",
        ];
        let err = TLE::records(lines).next().unwrap().unwrap_err();
        match &err {
            Error::Record {
                line, sat, hint, ..
            } => {
                assert_eq!(*line, 3);
                assert_eq!(sat.as_deref(), Some("45727"));
                assert!(hint.is_none());
            }
            other => panic!("expected Error::Record, got {other:?}"),
        }
        assert!(err.to_string().contains("Line 2 without a line 1"), "{err}");
    }

    #[test]
    fn test_checksum_validation() {
        // Valid checksums (including the 77-character INTELSAT line 2)
        let good = [
            "0 INTELSAT 902",
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.",
            "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995",
            "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085",
        ];
        assert_eq!(
            TLE::records(good)
                .check_checksums(true)
                .collect::<super::Result<Vec<_>>>()
                .unwrap()
                .len(),
            2
        );

        // Known-bad checksums: line 1 sums to 0 (not 2), line 2 to 3 (not 5)
        let bad = [
            "0 ISS (ZARYA)",
            "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
            "2 B5544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615",
        ];
        // Off by default, and with `false`
        assert!(TLE::records(bad).next().unwrap().is_ok());
        assert!(TLE::records(bad)
            .check_checksums(false)
            .next()
            .unwrap()
            .is_ok());
        assert!(TLE::from_lines(&bad.map(String::from)).is_ok());

        let err = TLE::records(bad)
            .check_checksums(true)
            .next()
            .unwrap()
            .unwrap_err();
        match &err {
            Error::Record { line, error, .. } => {
                assert_eq!(*line, 1);
                assert!(matches!(
                    **error,
                    Error::ChecksumMismatch {
                        line: 1,
                        expected: 0,
                        found: '2'
                    }
                ));
            }
            other => panic!("expected Error::Record, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("B5544 \"ISS (ZARYA)\""), "{msg}");
        assert!(
            msg.contains("column 69 is '2', but the line's checksum is 0"),
            "{msg}"
        );

        // Line 1 fixed: the line 2 mismatch is reported
        let mut bad2 = bad;
        bad2[1] = "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9990";
        let err = TLE::records(bad2)
            .check_checksums(true)
            .next()
            .unwrap()
            .unwrap_err();
        assert!(
            err.to_string().contains(
                "Line 2 checksum mismatch: column 69 is '5', but the line's checksum is 3"
            ),
            "{err}"
        );
    }

    #[test]
    fn test_from_invalid_tle2() -> Result<()> {
        let res = TLE::load_2line(
            "1 12345U 67890A 12345.67890123  .00000123  00000-0  12345-6 0 9992",
            "2 12345 51.6403 106.8969 0007877   6.1421 113.2479",
        );
        assert!(res.is_err(), "Expected error due to short line2, got OK");
        assert!(
            res.unwrap_err().to_string().contains("too short"),
            "Expected error about line being too short."
        );

        Ok(())
    }

    #[test]
    fn test_from_invalid_tle3() -> Result<()> {
        let res = TLE::load_3line(
            "0 INVALID TLE",
            "1 12345U 67890A 12345.67890123  .00000123  00000-0  12345-6 0 9992",
            "2 12345 51.6403 106.8969 0007877   6.1421 113.2479",
        );
        assert!(res.is_err(), "Expected error due to short line2, got OK");
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("Invalid TLE line lengths"),
            "Expected error about invalid line lengths."
        );
        Ok(())
    }

    #[test]
    fn test_alpha5_to_int() -> Result<()> {
        // 0-padded less-than-5-digits
        match TLE::alpha5_to_int("00091") {
            Ok(91) => {}
            Ok(i) => bail!("Error parsing '00091' as 91: got {}", i),
            Err(e) => bail!("Error parsing '00091' as 91: {}", e),
        }

        // Non-0-padded less-than-5-digits
        match TLE::alpha5_to_int("  982") {
            Ok(982) => {}
            Ok(i) => bail!("Error parsing '  982' as 982: got {}", i),
            Err(e) => bail!("Error parsing '  982' as 982: {}", e),
        }

        // Numerical 5 digit
        match TLE::alpha5_to_int("99993") {
            Ok(99993) => {}
            Ok(i) => bail!("Error parsing '99993' as 99993: got {}", i),
            Err(e) => bail!("Error parsing '99993' as 99993: {}", e),
        }

        // Alpha5
        match TLE::alpha5_to_int("S9994") {
            Ok(269994) => {}
            Ok(i) => bail!("Error parsing 'S9994' as 269994: got {}", i),
            Err(e) => bail!("Error parsing 'S9994' as 269994: {}", e),
        }

        Ok(())
    }

    #[test]
    fn test_int_to_alpha5() -> Result<()> {
        match TLE::int_to_alpha5(91) {
            Ok(ref s) if s == "00091" => {}
            Ok(ref s) => bail!("Error converting 91 to '00091': got {}", s),
            Err(e) => bail!("Error converting 91 to '00091': {}", e),
        }

        match TLE::int_to_alpha5(99993) {
            Ok(ref s) if s == "99993" => {}
            Ok(ref s) => bail!("Error converting 99993 to '99993': got {}", s),
            Err(e) => bail!("Error converting 99993 to '99993': {}", e),
        }

        // Alpha5
        match TLE::int_to_alpha5(269994) {
            Ok(ref s) if s == "S9994" => {}
            Ok(ref s) => bail!("Error converting 269994 to 'S9994': got {}", s),
            Err(e) => bail!("Error converting 269994 to 'S9994': {}", e),
        }

        Ok(())
    }

    #[test]
    fn test_3line_encoding() -> Result<()> {
        let line0 = "ISS (ZARYA)";
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let orig = TLE::load_3line(line0, line1, line2)?;

        // Format back to text
        let [l0, l1, l2] = orig.to_3line()?;

        // Check that it matches.
        assert_eq!(l1, line1, "Line 1 must match original");
        assert_eq!(l2, line2, "Line 2 must match original");
        assert_eq!(l0, line0, "Line 0 (name) must be preserved");

        Ok(())
    }

    #[test]
    fn test_2line_encoding() -> Result<()> {
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let orig = TLE::load_2line(line1, line2)?;

        // Format back to text
        let [l1, l2] = orig.to_2line()?;

        // Check that it matches.
        assert_eq!(l1, line1, "Line 1 must match original");
        assert_eq!(l2, line2, "Line 2 must match original");

        Ok(())
    }

    #[test]
    fn test_2line_encoding_many_times() -> Result<()> {
        let tle_examples = vec![
            [
                // "2023-193D"
                "1 58556U 23193D   25003.79555039  .00279397  31144-4  86159-3 0  9996".to_string(),
                "2 58556  97.2472  26.1173 0004235 271.4738  88.6051 15.91743157 60937".to_string(),
            ],
            [
                // "0 CPOD FLT2 (TYVAK-0033)"
                "1 52780U 22057BB  23036.86744141  .00018086  00000-0  87869-3 0  9991".to_string(),
                "2 52780  97.5313 154.3283 0011660  53.1934 307.0368 15.18441019 16465".to_string(),
            ],
            [
                // "1998-067WV"
                "1 60955U 98067WV  24295.33823779  .06453473  12009-4  26290-2 0  9998".to_string(),
                "2 60955  51.6166  43.0490 0010894 336.3668  23.6849 16.22453324  8315".to_string(),
            ],
            [
                // "2 PATHFINDER"
                "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995".to_string(),
                "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085".to_string(),
            ],
            // [
            //     // "0 SHINSEI (MS-F2)". Exclude because it does not use a 5-digit NORAD ID, and thus the encoding isn't as expected.
            //     "1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992".to_string(),
            //     "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065".to_string(),
            // ],
            [
                // "OSCAR 7 (AO-7)"
                "1 07530U 74089B   24323.87818483 -.00000039  00000+0  47934-4 0  9997".to_string(),
                "2 07530 101.9893 320.0351 0012269 147.9195 274.9996 12.53682684288423".to_string(),
            ],
            [
                "1 52743U 22057M   23037.04954473  .00011781  00000-0  61944-3 0  9993".to_string(),
                "2 52743  97.5265 153.6940 0008594  82.9904  31.3082 15.15793680 38769".to_string(),
            ],
            [
                // "0 ISS (ZARYA)"
                "1 B5544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992".to_string(),
                "2 B5544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487613".to_string(),
            ],
            [
                // "0 ISS (ZARYA)"
                "1 Z9999U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992".to_string(),
                "2 Z9999  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487611".to_string(),
            ],
        ];

        for tle in tle_examples {
            let tle_loaded = TLE::load_2line(&tle[0], &tle[1])?;
            let [l1, l2] = tle_loaded.to_2line()?;

            // Check that it matches.
            // Allow ignoring the sign of the exponent on zero.
            if tle[0].contains(" 00000+0 ") {
                let mut expected: String = tle[0].replace(" 00000+0 ", " 00000-0 ");

                // Increment the checksum digit at the end of the line.
                if let Some(last_char) = expected.chars().last() {
                    if let Some(digit) = last_char.to_digit(10) {
                        let new_digit = (digit + 1) % 10; // wrap around if needed
                        expected.pop(); // remove last char
                        expected.push(char::from_digit(new_digit, 10).unwrap());
                    }
                }

                assert_eq!(l1, expected, "Line 1 must match original");
            } else {
                assert_eq!(l2, tle[1], "Line 2 must match original");
            }
        }

        Ok(())
    }

    #[test]
    fn test_2line_encoding_with_invalid_past_date() -> Result<()> {
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let mut tle = TLE::load_2line(line1, line2)?;
        tle.epoch = Instant::from_date(1952, 6, 13)?;

        let result = tle.to_2line();

        // Check that it errors.
        assert!(
            result.is_err(),
            "Expected error due to epoch before 1957, got {:?}",
            result
        );

        Ok(())
    }

    #[test]
    fn test_2line_encoding_with_invalid_future_date() -> Result<()> {
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let mut tle = TLE::load_2line(line1, line2)?;
        tle.epoch = Instant::from_date(2057, 6, 13)?;

        let result = tle.to_2line();

        // Check that it errors.
        assert!(
            result.is_err(),
            "Expected error due to epoch after 2056, got {:?}",
            result
        );

        Ok(())
    }

    const ISS1: &str = "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992";
    const ISS2: &str = "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615";
    const PF1: &str = "1 45727U 20037E   24323.73967089  .00003818  00000+0  31595-3 0  9995";
    const PF2: &str = "2 45727  97.7798 139.6782 0011624 329.2427  30.8113 14.99451155239085";

    #[test]
    fn test_eq_ignores_sgp4_cache() {
        let fresh = TLE::load_2line(ISS1, ISS2).unwrap();
        let mut tle = fresh.clone();
        let t = tle.epoch;
        crate::sgp4::sgp4(&mut tle, &[t]).unwrap();
        assert!(tle.satrec.0.is_some());
        assert_eq!(tle, fresh);
        assert_eq!(tle.partial_cmp(&fresh), Some(std::cmp::Ordering::Equal));
        tle.reset_cache();
        assert!(tle.satrec.0.is_none());
        assert_eq!(tle, fresh);
    }

    #[test]
    fn test_satnum_mismatch_rejected() {
        // Line 1 of one satellite with line 2 of another: python-sgp4 raises,
        // and satkit used to build a hybrid element set
        let err = TLE::load_2line(ISS1, PF2).unwrap_err();
        assert!(
            matches!(&err, Error::SatNumMismatch { line1, line2 } if line1 == "25544" && line2 == "45727"),
            "{err:?}"
        );
        let err =
            TLE::from_lines(&["0 ISS".to_string(), ISS1.to_string(), PF2.to_string()]).unwrap_err();
        assert!(matches!(err, Error::Record { line: 1, .. }), "{err:?}");
        assert!(
            err.to_string().contains("Satellite number differs"),
            "{err}"
        );

        // The same number written with a leading zero is not a mismatch
        let l1 = "1 05485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992";
        let l2 = "2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065";
        assert_eq!(TLE::load_2line(l1, l2).unwrap().sat_num, 5485);
    }

    #[test]
    fn test_orphan_line1_is_an_error() {
        // A line 1 replaced by another line 1: its own record fails, and the
        // next record still parses
        let lines = ["0 LOST", ISS1, "0 PATHFINDER", PF1, PF2];
        let recs: Vec<_> = TLE::records(lines).collect();
        assert_eq!(recs.len(), 2);
        let err = recs[0].as_ref().unwrap_err();
        assert!(
            matches!(err, Error::Record { line: 1, sat: Some(s), error, .. }
                if s == "25544 \"LOST\"" && matches!(**error, Error::MissingLine2)),
            "{err:?}"
        );
        assert_eq!(recs[1].as_ref().unwrap().name, "PATHFINDER");

        let recs: Vec<_> = TLE::records([ISS1, PF1, PF2]).collect();
        assert_eq!(recs.len(), 2);
        assert!(matches!(recs[0], Err(Error::Record { line: 1, .. })));
        assert_eq!(recs[1].as_ref().unwrap().sat_num, 45727);

        // A trailing name + line 1 at the end of the input
        let lines = [PF1, PF2, "0 TRUNCATED", ISS1];
        let recs: Vec<_> = TLE::records(lines).collect();
        assert_eq!(recs.len(), 2);
        assert!(recs[0].is_ok());
        let err = recs[1].as_ref().unwrap_err();
        assert!(matches!(err, Error::Record { line: 3, .. }), "{err:?}");
        assert!(err.to_string().contains("Line 1 without a line 2"), "{err}");
        assert!(TLE::from_lines(&lines.map(String::from)).is_err());
    }

    #[test]
    fn test_hint_for_data_line_read_as_name() {
        // A 68-character line 1 is not a data line, so it becomes the name
        let short = &ISS1[..68];
        let err = TLE::from_lines(&[short.to_string(), ISS2.to_string()]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Line 2 without a line 1"), "{msg}");
        assert!(
            msg.contains("line 1, read as the satellite name, looks like a line 1 but is 68 characters; a TLE line is 69"),
            "{msg}"
        );

        // So does one with a leading space
        let err = TLE::from_lines(&[format!(" {ISS1}"), ISS2.to_string()]).unwrap_err();
        assert!(err.to_string().contains("starts with whitespace"), "{err}");

        // A 68-character line 2 ends its line 1's record
        let err = TLE::from_lines(&[ISS1.to_string(), ISS2[..68].to_string()]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Line 1 without a line 2"), "{msg}");
        assert!(
            msg.contains("line 2, which follows it, looks like a line 2 but is 68 characters"),
            "{msg}"
        );

        // A byte-order mark at the start of the input is dropped; one later
        // on (e.g. concatenated files) is named in the hint
        let tles = TLE::from_lines(&[format!("\u{feff}{ISS1}"), ISS2.to_string()]).unwrap();
        assert_eq!(tles[0].sat_num, 25544);
        let tles = TLE::from_lines(&[
            "\u{feff}0 ISS".to_string(),
            ISS1.to_string(),
            ISS2.to_string(),
        ])
        .unwrap();
        assert_eq!(tles[0].name, "ISS");
        let err = TLE::from_lines(&[
            PF1.to_string(),
            PF2.to_string(),
            format!("\u{feff}{ISS1}"),
            ISS2.to_string(),
        ])
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("starts with a UTF-8 byte-order mark"),
            "{err}"
        );
    }

    #[test]
    fn test_2line_wraps_counters() {
        let mut tle = TLE::load_2line(ISS1, ISS2).unwrap();
        tle.element_num = 12345;
        tle.rev_num = 123456;
        let [l1, l2] = tle.to_2line().unwrap();
        assert_eq!(l1.len(), 69, "{l1}");
        assert_eq!(l2.len(), 69, "{l2}");
        let back = TLE::records([l1.as_str(), l2.as_str()])
            .check_checksums(true)
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(back.element_num, 2345);
        assert_eq!(back.rev_num, 23456);
        assert_eq!(back.mean_motion, tle.mean_motion);
    }
}
