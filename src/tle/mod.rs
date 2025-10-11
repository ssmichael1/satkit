use crate::sgp4::SatRec;
use crate::Instant;

// TLE fitting from state vectors
mod fitting;

use anyhow::{bail, Context, Result};

// 'I' and 'O' are not part of the allowed chars to avoid any confusion with 0 or 1
const ALPHA5_MATCHING: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ";

///
/// Stucture representing a Two-Line Element Set (TLE), a satellite
/// ephemeris format from the 1970s that is still somehow in use
/// today and can be used to calculate satellite position and
/// velcocity in the "TEME" frame (not-quite GCRF) using the
/// "Simplified General Perturbations-4" (SGP-4) mathemematical
/// model that is also included in this package.
///
/// For details, see: <https://en.wikipedia.org/wiki/Two-line_element_set>
///
/// The TLE format is still commonly used to represent satellite
/// ephemerides, and satellite ephemerides catalogs in this format
/// are publicly availalble at www.space-track.org (registration
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
/// let (pTEME, vTEME, errs) = sgp4(&mut tle, &[tm]);
///
/// println!("pTEME = {}", pTEME);
/// // Rotate the position to the ITRF frame (Earth-fixed)
/// // Since pTEME is a 3xN array where N is the number of times
/// // (we are just using a single time)
/// // we need to convert to a fixed matrix to rotate
/// let pITRF = frametransform::qteme2itrf(&tm) * pTEME.fixed_view::<3,1>(0,0);
///
/// println!("pITRF = {}", pITRF);
///
/// // Convert to an "ITRF Coordinate" and print geodetic position
/// let itrf = ITRFCoord::from_slice(&pTEME.fixed_view::<3,1>(0,0).as_slice()).unwrap();
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
    /// Usually 0
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

    pub(crate) satrec: Option<SatRec>,
}

impl TLE {
    /// Load a vector of strings representing Two-Line Element Set (TLE) lines into a vector of
    /// TLE structures.
    /// This function will call [`Self::load_2line`] respectively [`Self::load_3line`] as required
    /// for each TLE entry it encounters in the input lines.
    ///
    /// Those TLEs can then be used to compute satellite position and
    /// velocity as a function of time.
    ///
    /// For details, see [here](https://en.wikipedia.org/wiki/Two-line_element_set)
    ///
    /// # Arguments:
    ///   * `lines` - a reference to a [`Vec`] of [`String`] representing TLE lines
    ///
    /// # Returns:
    ///  * A [`Vec`] of [`TLE`] objects or string indicating error condition
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
    ///
    /// ```
    pub fn from_lines(lines: &[String]) -> Result<Vec<Self>> {
        let mut tles: Vec<Self> = Vec::<Self>::new();
        let empty: &String = &String::new();
        let mut line0: &String = empty;
        let mut line1: &String = empty;
        let mut line2: &String;

        for line in lines {
            if line.len() < 2 {
                continue;
            }
            if line.chars().nth(0).unwrap() == '1'
                && line.chars().nth(1).unwrap() == ' '
                && line.len() == 69
            {
                line1 = line;
            } else if line.chars().nth(0).unwrap() == '2'
                && line.chars().nth(1).unwrap() == ' '
                && line.len() == 69
            {
                line2 = line;
                if line0.is_empty() {
                    tles.push(Self::load_2line(line1, line2)?);
                } else {
                    tles.push(Self::load_3line(line0, line1, line2)?);
                }
                line0 = empty;
                line1 = empty;
            } else {
                line0 = line;
            }
        }

        Ok(tles)
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
            satrec: None,
        }
    }

    /// Load 3 lines as string into a structure representing
    /// a Two-Line Element Set  (TLE)
    ///
    /// The TLE can then be used to compute satellite position and
    /// velocity as a function of time.
    ///
    /// For details, see [here](https://en.wikipedia.org/wiki/Two-line_element_set)
    ///
    /// # Arguments:
    ///
    ///   * `line0` - the "0"th line of the TLE, which sometimes contains
    ///     the satellite name
    ///   * `line1` - the 1st line of TLE
    ///   * `line2` - the 2nd line of the TLE
    ///
    /// # Returns:
    ///
    ///  * A TLE object or string indicating error condition
    ///
    /// # Example
    ///
    /// ```
    ///
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
            bail!(
                "Invalid TLE line lengths: line1 = {}, line2 = {}",
                line1.len(),
                line2.len()
            );
        }

        match Self::load_2line(line1, line2) {
            Ok(mut tle) => {
                tle.name = {
                    if line0.len() > 2 && line0.chars().nth(0).unwrap() == '0' {
                        line0[2..].to_string()
                    } else {
                        String::from(line0)
                    }
                };
                Ok(tle)
            }
            Err(e) => Err(e),
        }
    }

    /// Load 2 lines as strings into a structure representing
    /// a Two-Line Element Set  (TLE)
    ///
    /// The TLE can then be used to compute satellite position and
    /// velocity as a function of time.
    ///
    /// For details, see [here](https://en.wikipedia.org/wiki/Two-line_element_set)
    ///
    /// # Arguments:
    ///
    ///   * `line1` - the 1st line of TLE
    ///   * `line2` - the 2nd line of the TLE
    ///
    /// # Returns:
    ///
    ///  * A TLE object or string indicating error condition
    ///
    /// # Example
    ///
    /// ```
    ///
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
        if line1.len() < 69 {
            bail!(
                "Line 1 too short: expected 69 characters, got {}",
                line1.len()
            );
        }
        if line2.len() < 69 {
            bail!(
                "Line 2 too short: expected 69 characters, got {}",
                line2.len()
            );
        }

        let mut year: u32 = {
            let mut mstr: String = "1".to_owned();
            mstr.push_str(&line1[18..20]);
            let mut s = mstr.parse().context("Could not parse year")?;
            s -= 100;
            s
        };
        // See: https://celestrak.org/columns/v04n03/
        // Years >= 1957 = 1900s
        // Years < 1957 = 2000s
        let century = if year >= 57 { 1900 } else { 2000 };
        year += century;
        let day_of_year: f64 = line1[20..32]
            .parse()
            .context("Could not parse day of year")?;

        // Note: day_of_year starts from 1, not zero,
        // also, go from Jan 2 to avoid leap-second
        // issues, hence the "-2" at end
        let epoch = Instant::from_date(year as i32, 1, 2)
            .context("Invalid year, month, or day")?
            .add_utc_days(day_of_year - 2.0);

        Ok(Self {
            name: "none".to_string(),
            sat_num: Self::alpha5_to_int(&line1[2..7])
                .context("Could not parse satellite number")?,

            intl_desig: { line1[9..16].trim().to_string() },
            desig_year: { line1[9..11].trim().parse().unwrap_or(70) },
            desig_launch: { line1[11..14].trim().parse().unwrap_or_default() },
            desig_piece: line1[14..18]
                .trim()
                .parse()
                .context("Could not parse desig_piece")?,

            epoch,
            mean_motion_dot: {
                let mut mstr: String = "0".to_owned();
                mstr.push_str(&line1[34..43]);
                let mut m = mstr.parse().context("Could not parse mean motion dot")?;
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
                let mut m = mstr
                    .trim()
                    .parse()
                    .context("Coudl not parse mean motion dot dot")?;
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
                let mut m = mstr
                    .trim()
                    .parse()
                    .context("Could not parse bstar (drag)")?;
                if line1.chars().nth(53).unwrap() == '-' {
                    m *= -1.0;
                }
                m
            },
            ephem_type: { line1[62..63].trim().parse().unwrap_or_default() },
            element_num: line1[64..68]
                .trim()
                .parse()
                .context("Could not parse element number")?,

            inclination: line2[8..16]
                .trim()
                .parse()
                .context("Could not parse inclination")?,

            raan: line2[17..25]
                .trim()
                .parse()
                .context("Could not parse raan")?,

            eccen: {
                let mut mstr: String = "0.".to_owned();
                mstr.push_str(&line2[26..33]);
                mstr.trim()
                    .parse()
                    .context("Could not parse eccentricity")?
            },
            arg_of_perigee: line2[34..42]
                .trim()
                .parse()
                .context("Could not parse arg of perigee")?,

            mean_anomaly: line2[42..51]
                .trim()
                .parse()
                .context("Could not parse mean anomaly")?,

            mean_motion: line2[52..63]
                .trim()
                .parse()
                .context("Could not parse mean motion")?,

            rev_num: line2[63..68]
                .trim()
                .parse()
                .context("Could not parse rev num")?,
            satrec: None,
        })
    }

    /// Format this TLE back into the two canonical 69-char lines.
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
        let elem_no = format!("{:>4}", self.element_num.max(0)); // cols 65-68

        let mut l1 = format!("1 {sat5}U {desig} {epoch} {ndot} {nddot} {bstar} {et} {elem_no}");

        let cksum1 = tle_formatter::tle_checksum(&l1);
        l1.push(char::from(b'0' + cksum1));

        // ------- Build Line 2 -------
        let incl = format!("{:8.4}", self.inclination);
        let raan = format!("{:8.4}", self.raan);
        let ecc7 = format!("{:0>7}", (self.eccen.abs() * 1.0e7 + 0.5).floor() as u64);
        let argp = format!("{:8.4}", self.arg_of_perigee);
        let mean_anom = format!("{:8.4}", self.mean_anomaly);
        let n = format!("{:11.8}", self.mean_motion);
        let rev = format!("{:>5}", self.rev_num.max(0));

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
            bail!("Year out of range for TLE: {}", year);
        }

        // Day-of-year.
        let doy_int = self.epoch.day_of_year();

        // Fraction of day.
        // Note: This works with days that have leap seconds
        // (in which second of day is normalized to 86401 instead of 86400).
        let frac = self.epoch.as_mjd() % 1.0;
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
                Ok(i) => Ok(i),
                Err(e) => bail!("Invalid sat num: {}", e.to_string()),
            },
            Some(c) if c.is_alphabetic() => {
                match ALPHA5_MATCHING
                    .chars()
                    .position(|m| m == c.to_ascii_uppercase())
                {
                    Some(p) => match alpha5[1..].parse::<i32>() {
                        Ok(i) => Ok((p as i32 + 10) * 10000 + i),
                        Err(e) => bail!("Invalid sat num: {}", e.to_string()),
                    },
                    None => bail!("Invalid first digit in sat num: {}", c),
                }
            }
            Some(c) => bail!("Invalid first digit in sat num: {}", c),
            None => bail!("Parse error"),
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
            _i @ 340000.. => bail!("Sat num >= 340000 cannot be represented in alpha5 format"),
            _ => bail!("Invalid sat num value"),
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
            Self::int_to_alpha5(self.sat_num).unwrap(),
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

    #[test]
    fn test_from_invalid_from_lines() -> Result<()> {
        let res = TLE::from_lines(&[
            "0 INVALID TLE".to_string(),
            "1 12345U 67890A 12345.67890123  .00000123  00000-0  12345-6 0  9992".to_string(),
            "2 12345  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487615".to_string(),
        ]);
        assert!(res.is_err(), "Expected error due to short lines, got OK");
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("Invalid TLE line lengths"),
            "Expected error about invalid line lengths."
        );

        Ok(())
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
        assert!(result.is_err(), "Expected error due to epoch before 1957, got {:?}", result);

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
        assert!(result.is_err(), "Expected error due to epoch after 2056, got {:?}", result);

        Ok(())
    }
}
