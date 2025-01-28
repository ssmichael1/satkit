use crate::sgp4::SatRec;
use crate::skerror;
use crate::Instant;
use crate::SKResult;

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
/// let tm = Instant::from_datetime(2006, 5, 1, 11, 0, 0.0);
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
    pub fn from_lines(lines: &Vec<String>) -> SKResult<Vec<Self>> {
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
    pub fn load_3line(line0: &str, line1: &str, line2: &str) -> SKResult<Self> {
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
    pub fn load_2line(line1: &str, line2: &str) -> SKResult<Self> {
        let mut year: u32 = {
            let mut mstr: String = "1".to_owned();
            mstr.push_str(&line1[18..20]);
            let mut s: u32 = match mstr.parse() {
                Ok(y) => y,
                Err(_) => return crate::skerror!("Could not parse year"),
            };
            s -= 100;
            s
        };
        if year > 57 {
            year += 1900;
        } else {
            year += 2000;
        }
        let day_of_year: f64 = match line1[20..32].parse() {
            Ok(y) => y,
            Err(_) => return crate::skerror!("Could not parse day of year"),
        };

        // Note: day_of_year starts from 1, not zero,
        // also, go from Jan 2 to avoid leap-second
        // issues, hence the "-2" at end
        let epoch = Instant::from_date(year as i32, 1, 2).add_utc_days(day_of_year - 2.0);

        Ok(Self {
            name: "none".to_string(),
            sat_num: {
                match line1[2..7].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse sat number: {}", e.to_string()),
                }
            },
            intl_desig: { line1[9..16].trim().to_string() },
            desig_year: { line1[9..11].trim().parse().unwrap_or(70) },
            desig_launch: { line1[11..14].trim().parse().unwrap_or_default() },
            desig_piece: {
                match line1[14..18].trim().parse() {
                    Ok(l) => l,
                    Err(e) => return skerror!("Could not parse desig_piece: {}", e.to_string()),
                }
            },
            epoch,
            mean_motion_dot: {
                let mut mstr: String = "0".to_owned();
                mstr.push_str(&line1[34..43]);
                let mut m: f64 = match mstr.parse() {
                    Ok(y) => y,
                    Err(e) => {
                        return skerror!("Could not parse mean motion dot: {}", e.to_string())
                    }
                };
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
                let mut m: f64 = match mstr.trim().parse() {
                    Ok(y) => y,
                    Err(e) => {
                        return skerror!("Could not parse mean motion dot dot: {}", e.to_string())
                    }
                };
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
                let mut m: f64 = match mstr.trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse bstar (drag): {}", e.to_string()),
                };
                if line1.chars().nth(53).unwrap() == '-' {
                    m *= -1.0;
                }
                m
            },
            ephem_type: { line1[62..63].trim().parse().unwrap_or_default() },
            element_num: {
                match line1[64..68].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse element number: {}", e.to_string()),
                }
            },
            inclination: {
                match line2[8..16].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse inclination: {}", e.to_string()),
                }
            },
            raan: {
                match line2[17..25].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse raan: {}", e.to_string()),
                }
            },
            eccen: {
                let mut mstr: String = "0.".to_owned();
                mstr.push_str(&line2[26..33]);
                match mstr.trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse eccen: {}", e.to_string()),
                }
            },
            arg_of_perigee: {
                match line2[34..42].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse arg of perigee: {}", e.to_string()),
                }
            },
            mean_anomaly: {
                match line2[42..51].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse mean anomaly: {}", e.to_string()),
                }
            },
            mean_motion: {
                match line2[52..63].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse mean motion: {}", e.to_string()),
                }
            },
            rev_num: {
                match line2[63..68].trim().parse() {
                    Ok(y) => y,
                    Err(e) => return skerror!("Could not parse rev num: {}", e.to_string()),
                }
            },
            satrec: None,
        })
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
            self.sat_num,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{skerror, SKResult};

    #[test]
    fn testload() -> SKResult<()> {
        let line1: &str = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
        let line2: &str =
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300.";
        let line0: &str = "0 INTELSAT 902";
        match TLE::load_3line(line0, line1, line2) {
            Ok(_t) => {}

            Err(s) => {
                return skerror!("load_3line: Err = \"{}\"", s);
            }
        }
        match TLE::load_2line(line1, line2) {
            Ok(_t) => {}
            Err(s) => {
                return skerror!("load_2line: Err = \"{}\"", s);
            }
        }
        Ok(())
    }

    #[test]
    fn test_from_lines() -> SKResult<()> {
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
        ];

        let tles = match TLE::from_lines(&lines) {
            Ok(t) => t,
            Err(s) => {
                return skerror!("load_lines: Err = \"{}\"", s);
            }
        };

        if tles.len() != 7 {
            return skerror!("load_lines: Err = \"Incorrect number of elements parsed\"");
        }

        if tles[0].name != "2023-193D" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[0].name
            );
        }

        if tles[1].name != "CPOD FLT2 (TYVAK-0033)" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[1].name
            );
        }

        if tles[2].name != "1998-067WV" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[2].name
            );
        }

        if tles[3].name != "2 PATHFINDER" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[3].name
            );
        }

        if tles[4].name != "SHINSEI (MS-F2)" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[4].name
            );
        }

        if tles[4].sat_num != 5485 {
            return skerror!(
                "load_lines: Err = \"Error parsing sat num {}\"",
                tles[4].sat_num
            );
        }

        if tles[5].name != "OSCAR 7 (AO-7)" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[5].name
            );
        }

        if tles[5].sat_num != 7530 {
            return skerror!(
                "load_lines: Err = \"Error parsing sat num {}\"",
                tles[5].sat_num
            );
        }

        if tles[6].name != "none" {
            return skerror!(
                "load_lines: Err = \"Error parsing sat name {}\"",
                tles[6].name
            );
        }

        Ok(())
    }
}
