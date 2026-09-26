use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use satkit::tle::TLE;

use crate::pyinstant::ToTimeVec;
use anyhow::{bail, Result};

use crate::pytlefitstatus::PyTleFitStatus;

use std::fs::File;
use std::io;
use std::io::BufRead;

/// Two-Line Element Set (TLE) representing a satellite ephemeris
///
/// A Two-Line Element Set is a satellite ephemeris format from the 1970s
/// that is still in wide use. Its mean elements are propagated with the
/// "Simplified General Perturbations-4" (SGP4) model (``satkit.sgp4``),
/// which gives position and velocity in the "TEME" frame (not-quite GCRF).
///
/// For details, see: <https://en.wikipedia.org/wiki/Two-line_element_set>
///
/// Catalogs in this format are publicly available at
/// <https://www.space-track.org> (registration required) and
/// <https://celestrak.org> (no registration needed).
///
/// TLEs sometimes have a "line 0" that includes the name of the satellite.
///
/// Load TLEs with ``TLE.from_lines``, ``TLE.from_file`` or ``TLE.from_url``;
/// each returns a ``list[TLE]``, even for a single element set.
///
/// Example:
///     ```python
///     tle = satkit.TLE.from_lines([
///         "0 ISS (ZARYA)",
///         "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9991",
///         "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302878",
///     ])[0]
///     print(tle.name)
///     # ISS (ZARYA)
///     ```
#[pyclass(name = "TLE", module = "satkit")]
pub struct PyTLE(pub TLE);

/// The parsed TLEs as a list, or `ValueError` when there are none; `what`
/// names the source in the error
fn tle_list(tles: Vec<TLE>, what: &str) -> PyResult<Vec<PyTLE>> {
    if tles.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "No valid TLEs found in {what}"
        )));
    }
    Ok(tles.into_iter().map(PyTLE).collect())
}

/// Parse `lines` into TLEs, stopping at the first bad record
fn parse_lines(lines: &[String], check_checksum: bool) -> satkit::tle::Result<Vec<TLE>> {
    TLE::records(lines)
        .check_checksums(check_checksum)
        .collect()
}

#[pymethods]
impl PyTLE {
    /// Load TLEs from a text file, parsed as ``TLE.from_lines`` does
    ///
    /// Args:
    ///     filename (str): name of the text file holding the TLE lines
    ///     check_checksum (bool, optional): as in ``TLE.from_lines``
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set in the file
    ///
    /// Raises:
    ///     ValueError: if the file holds no TLEs
    ///     RuntimeError: if a record fails to parse; see ``TLE.from_lines``
    #[staticmethod]
    #[pyo3(signature = (filename, *, check_checksum=false))]
    fn from_file(filename: String, check_checksum: bool) -> Result<Vec<Self>> {
        let file = File::open(std::path::PathBuf::from(&filename))?;

        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .collect::<std::result::Result<_, _>>()?;

        Ok(tle_list(parse_lines(&lines, check_checksum)?, &filename)?)
    }

    #[new]
    fn new() -> Self {
        Self(TLE::new())
    }

    /// Load TLEs from a list of lines
    ///
    /// ``TLE.from_file`` and ``TLE.from_url`` parse their text the same way.
    ///
    /// Args:
    ///     lines (Sequence[str]): the TLE lines (2-line or 3-line format,
    ///         any number of element sets)
    ///     check_checksum (bool, optional): also verify the checksum digit
    ///         (column 69) of every data line. Default False.
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set, even if there is only one
    ///
    /// Raises:
    ///     ValueError: if the lines hold no TLEs
    ///     RuntimeError: if a record fails to parse (or, with
    ///         ``check_checksum``, has a wrong checksum); the message gives
    ///         the line the record starts on and its satellite
    #[staticmethod]
    #[pyo3(signature = (lines, *, check_checksum=false))]
    fn from_lines(lines: Vec<String>, check_checksum: bool) -> Result<Vec<Self>> {
        Ok(tle_list(parse_lines(&lines, check_checksum)?, "input")?)
    }

    /// Load TLEs from a URL, parsing the response as ``TLE.from_lines`` does
    ///
    /// Works with any URL that returns plain-text TLE data.
    ///
    /// Args:
    ///     url (str): URL to fetch TLE data from
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set in the response
    ///
    /// Raises:
    ///     ValueError: if the response holds no TLEs
    ///     RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
    ///         ``satkit.utils.set_offline(True)``; no connection is opened),
    ///         the request fails, or a record fails to parse
    ///
    /// Example:
    ///     ```python
    ///     tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
    ///     ```
    #[staticmethod]
    fn from_url(py: Python, url: String) -> Result<Vec<Self>> {
        let tles = py.detach(|| TLE::from_url(&url))?;
        Ok(tle_list(tles, "response")?)
    }

    /// Satellite NORAD Catalog Number
    #[getter(satnum)]
    const fn get_satnum(&self) -> i32 {
        self.0.sat_num
    }

    #[setter(satnum)]
    fn set_satnum(&mut self, value: i32) {
        self.0.sat_num = value;
        self.0.reset_cache();
    }

    /// International designator (e.g. "98067A": launch year, launch number, piece)
    #[getter(intl_desig)]
    fn get_intl_desig(&self) -> String {
        self.0.intl_desig.clone()
    }

    #[setter(intl_desig)]
    fn set_intl_desig(&mut self, value: String) {
        self.0.intl_desig = value;
        self.0.reset_cache();
    }

    /// Launch year from the international designator (2-digit, as in the TLE)
    #[getter(desig_year)]
    const fn get_desig_year(&self) -> i32 {
        self.0.desig_year
    }

    #[setter(desig_year)]
    fn set_desig_year(&mut self, value: i32) {
        self.0.desig_year = value;
        self.0.reset_cache();
    }

    /// Launch number of the year from the international designator
    #[getter(desig_launch)]
    const fn get_desig_launch(&self) -> i32 {
        self.0.desig_launch
    }

    #[setter(desig_launch)]
    fn set_desig_launch(&mut self, value: i32) {
        self.0.desig_launch = value;
        self.0.reset_cache();
    }

    /// Piece of the launch from the international designator (e.g. "A")
    #[getter(desig_piece)]
    fn get_desig_piece(&self) -> String {
        self.0.desig_piece.clone()
    }

    #[setter(desig_piece)]
    fn set_desig_piece(&mut self, value: String) {
        self.0.desig_piece = value;
        self.0.reset_cache();
    }

    /// Classification (line 1, column 8): "U", "C" or "S" in practice, but
    /// any single ASCII letter round-trips
    #[getter(classification)]
    fn get_classification(&self) -> String {
        self.0.classification.to_string()
    }

    #[setter(classification)]
    fn set_classification(&mut self, value: &str) -> PyResult<()> {
        let mut chars = value.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) if c.is_ascii_alphabetic() => {
                self.0.classification = c;
                // Does not affect SGP4, but reset for consistency with the
                // other setters.
                self.0.reset_cache();
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "classification must be a single ASCII letter, got {value:?}"
            ))),
        }
    }

    /// Ephemeris type (usually 0)
    #[getter(ephem_type)]
    const fn get_ephem_type(&self) -> u8 {
        self.0.ephem_type
    }

    #[setter(ephem_type)]
    fn set_ephem_type(&mut self, value: u8) {
        self.0.ephem_type = value;
        self.0.reset_cache();
    }

    /// Element set number
    #[getter(element_num)]
    const fn get_element_num(&self) -> i32 {
        self.0.element_num
    }

    #[setter(element_num)]
    fn set_element_num(&mut self, value: i32) {
        self.0.element_num = value;
        self.0.reset_cache();
    }

    /// Revolution number at epoch
    #[getter(rev_num)]
    const fn get_rev_num(&self) -> i32 {
        self.0.rev_num
    }

    #[setter(rev_num)]
    fn set_rev_num(&mut self, value: i32) {
        self.0.rev_num = value;
        self.0.reset_cache();
    }

    /// Orbit eccentricity, unitless, in range [0,1]
    #[getter(eccen)]
    const fn get_eccen(&self) -> f64 {
        self.0.eccen
    }

    #[setter(eccen)]
    fn set_eccen(&mut self, value: f64) {
        self.0.eccen = value;
        self.0.reset_cache();
    }

    /// Mean anomaly in degrees
    #[getter(mean_anomaly)]
    const fn get_mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly
    }
    #[setter(mean_anomaly)]
    fn set_mean_anomaly(&mut self, value: f64) {
        self.0.mean_anomaly = value;
        self.0.reset_cache();
    }

    /// Mean motion in revs / day
    #[getter(mean_motion)]
    const fn get_mean_motion(&self) -> f64 {
        self.0.mean_motion
    }
    #[setter(mean_motion)]
    fn set_mean_motion(&mut self, value: f64) {
        self.0.mean_motion = value;
        self.0.reset_cache();
    }

    /// inclination in degrees
    #[getter(inclination)]
    const fn get_inclination(&self) -> f64 {
        self.0.inclination
    }
    #[setter(inclination)]
    fn set_inclination(&mut self, value: f64) {
        self.0.inclination = value;
        self.0.reset_cache();
    }

    /// Epoch time of TLE
    #[getter(epoch)]
    fn get_epoch(&self) -> crate::pyinstant::PyInstant {
        crate::pyinstant::PyInstant(self.0.epoch)
    }
    // A single satkit.time or datetime.datetime; anything else, including a
    // list of times (which used to set the first element silently), raises
    // TypeError
    #[setter(epoch)]
    fn set_epoch(&mut self, value: crate::pyinstant::TimeArg) {
        self.0.epoch = value.0;
        self.0.reset_cache();
    }

    /// argument of perigee, degrees
    #[getter(arg_of_perigee)]
    const fn get_arg_of_perigee(&self) -> f64 {
        self.0.arg_of_perigee
    }
    #[setter(arg_of_perigee)]
    fn set_arg_of_perigee(&mut self, value: f64) {
        self.0.arg_of_perigee = value;
        self.0.reset_cache();
    }

    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    #[getter(mean_motion_dot)]
    const fn get_mean_motion_dot(&self) -> f64 {
        self.0.mean_motion_dot
    }
    #[setter(mean_motion_dot)]
    fn set_mean_motion_dot(&mut self, value: f64) {
        self.0.mean_motion_dot = value;
        self.0.reset_cache();
    }

    /// One sixth of 2nd derivative of mean motion wrt time, in revs/day^3
    #[getter(mean_motion_dot_dot)]
    const fn get_mean_motion_dot_dot(&self) -> f64 {
        self.0.mean_motion_dot_dot
    }
    #[setter(mean_motion_dot_dot)]
    fn set_mean_motion_dot_dot(&mut self, value: f64) {
        self.0.mean_motion_dot_dot = value;
        self.0.reset_cache();
    }

    /// Right Ascension of the Ascending Node, degrees
    #[getter(raan)]
    const fn get_raan(&self) -> f64 {
        self.0.raan
    }
    #[setter(raan)]
    fn set_raan(&mut self, value: f64) {
        self.0.raan = value;
        self.0.reset_cache();
    }

    /// Name of satellite
    #[getter(name)]
    fn name(&self) -> String {
        self.0.name.clone()
    }
    #[setter(name)]
    fn set_name(&mut self, value: String) {
        self.0.name = value;
        self.0.reset_cache();
    }

    /// Drag term (B*) of the satellite, in units of 1 / Earth radii
    #[getter(bstar)]
    const fn bstar(&self) -> f64 {
        self.0.bstar
    }
    #[setter(bstar)]
    fn set_bstar(&mut self, value: f64) {
        self.0.bstar = value;
        self.0.reset_cache();
    }

    fn __str__(&self) -> String {
        self.0.to_pretty_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    // Compares the element sets; the cached SGP4 initialization is ignored,
    // so a TLE equals its pickle or copy after it has been propagated
    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Build a TLE from an OMM (Orbital Mean-Element Message) dictionary
    ///
    /// The dictionary is the same shape ``satkit.sgp4`` accepts: the flat
    /// CCSDS keys of a Space-Track / CelesTrak JSON record (see
    /// ``satkit.OMMDict``), or the nested ``meanElements`` / ``tleParameters``
    /// groups of an XML-derived dict. Numbers may be strings.
    ///
    /// The six mean elements, epoch, ``BSTAR``, ``MEAN_MOTION_DOT``,
    /// ``MEAN_MOTION_DDOT``, ``NORAD_CAT_ID``, ``ELEMENT_SET_NO``,
    /// ``REV_AT_EPOCH`` and ``EPHEMERIS_TYPE`` carry over; absent optional
    /// values become zero. ``OBJECT_ID`` in ``YYYY-NNNP`` form becomes the
    /// international designator. Other metadata is dropped.
    ///
    /// Args:
    ///     omm (dict): OMM dictionary
    ///
    /// Returns:
    ///     TLE: the equivalent two-line element set
    #[staticmethod]
    fn from_omm(omm: &Bound<'_, pyo3::types::PyDict>) -> Result<Self> {
        let omm = crate::pyomm::omm_from_pydict(omm)?;
        Ok(Self(omm.to_tle()))
    }

    /// Render this TLE as an OMM (Orbital Mean-Element Message) dictionary
    ///
    /// The result uses the flat CCSDS keys (see ``satkit.OMMDict``) with
    /// ``EPOCH`` as an RFC 3339 string, angles in degrees and mean motion in
    /// revolutions per day, and can be passed back to ``satkit.sgp4`` or
    /// serialized with ``json.dumps``. ``OBJECT_ID`` is derived from the
    /// international designator (``98067A`` becomes ``1998-067A``), and
    /// ``CLASSIFICATION_TYPE`` is the TLE's classification letter.
    ///
    /// Returns:
    ///     dict: OMM dictionary
    fn to_omm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::pyomm::omm_to_pydict(py, &satkit::omm::OMM::from_tle(&self.0))
    }

    /// Output as 2 canonical TLE Lines
    ///
    /// The element set number (4 columns) and revolution number (5 columns) are
    /// written modulo 10,000 and 100,000, so a larger value wraps around the way
    /// catalog TLEs roll the revolution counter over.
    ///
    /// Raises:
    ///     ValueError: if ``sat_num`` is 340000 or above: Alpha-5 (used for
    ///         the 5-character satellite number field) ends at ``Z9999``
    ///         (339999), so the catalog number cannot be written to a TLE.
    ///         Use ``to_omm()`` instead.
    fn to_2line(&self) -> PyResult<[String; 2]> {
        self.0.to_2line().map_err(|e| match e {
            satkit::tle::Error::SatNumTooLargeForAlpha5 => {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            }
            e => anyhow::Error::from(e).into(),
        })
    }

    /// Output as 2 canonical TLE lines preceded by a name line (3-line element set)
    fn to_3line(&self) -> Result<[String; 3]> {
        Ok(self.0.to_3line()?)
    }

    /// Perform non-linear least squares fit of TLE parameters to a list of GCRF states
    ///
    /// Args:
    ///     states (list[numpy.ndarray]): List of GCRF states to fit to. Each state is a
    ///         6-element vector: the first 3 values are position in meters, the last 3
    ///         values are velocity in meters / second
    ///     times (list[satkit.time]): Times corresponding to the states
    ///     epoch (satkit.time): Epoch time for the TLE. Must be within range of times
    ///
    /// Keyword Args:
    ///     gravconst (satkit.sgp4_gravconst): gravity model SGP4 uses in the fit.  Default is gravconst.wgs72
    ///     opsmode (satkit.sgp4_opsmode): SGP4 ops mode used in the fit.  Default is opsmode.afspc
    ///
    /// Returns:
    ///     tuple[TLE, dict]: Fitted TLE and fitting results in a dictionary
    ///
    /// Note:
    ///     The fitted TLE reproduces the states when propagated with ``satkit.sgp4`` under
    ///     the same ``gravconst`` and ``opsmode``; the defaults match ``satkit.sgp4``'s.
    ///     satkit before 0.24 fitted with WGS84. The GCRF states are rotated to TEME with
    ///     the full IERS 2010 rotation.
    #[staticmethod]
    #[pyo3(signature=(states, times, epoch, *, gravconst=crate::pysgp4::GravConst::wgs72, opsmode=crate::pysgp4::OpsMode::afspc))]
    fn fit_from_states(
        py: Python,
        states: Vec<[f64; 6]>,
        times: &Bound<'_, PyAny>,
        epoch: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = crate::pysgp4::gravconst_arg)] gravconst: crate::pysgp4::GravConst,
        #[pyo3(from_py_with = crate::pysgp4::opsmode_arg)] opsmode: crate::pysgp4::OpsMode,
    ) -> Result<(Self, Py<PyAny>)> {
        let times = times.to_time_vec()?;
        let epoch = epoch.to_time_vec()?;
        if epoch.len() != 1 {
            bail!("epoch must be a single time value");
        }
        // Release the GIL during the (potentially long-running) fit
        let (gravconst, opsmode) = (gravconst.into(), opsmode.into());
        let (tle, result) =
            py.detach(|| TLE::fit_from_states_full(&states, &times, epoch[0], gravconst, opsmode))?;

        let stats = {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("status", PyTleFitStatus::from(result.status))?;
            dict.set_item("converged", {
                let s: PyTleFitStatus = result.status.into();
                s.converged()
            })?;
            dict.set_item("orig_norm", result.orig_norm)?;
            dict.set_item("best_norm", result.best_norm)?;
            dict.set_item("grad_norm", result.grad_norm)?;
            dict.set_item("n_iter", result.n_iter)?;
            dict.set_item("n_res_evals", result.n_res_evals)?;
            dict.into()
        };
        Ok((Self(tle), stats))
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        // Self-describing format v3 (see `__setstate__` for the layout):
        // a leading version byte, a 102-byte fixed field block (v2's
        // 101 bytes plus the classification letter), then three
        // length-prefixed UTF-8 strings (name, intl_desig, desig_piece).
        let mut raw: Vec<u8> = Vec::with_capacity(
            109 + self.0.name.len() + self.0.intl_desig.len() + self.0.desig_piece.len(),
        );
        raw.push(3u8); // version
        raw.extend_from_slice(&self.0.sat_num.to_le_bytes());
        raw.extend_from_slice(&self.0.desig_year.to_le_bytes());
        raw.extend_from_slice(&self.0.desig_launch.to_le_bytes());
        raw.extend_from_slice(&self.0.mean_motion_dot.to_le_bytes());
        raw.extend_from_slice(&self.0.mean_motion_dot_dot.to_le_bytes());
        raw.extend_from_slice(&self.0.bstar.to_le_bytes());
        raw.extend_from_slice(&self.0.inclination.to_le_bytes());
        raw.extend_from_slice(&self.0.raan.to_le_bytes());
        raw.extend_from_slice(&self.0.eccen.to_le_bytes());
        raw.extend_from_slice(&self.0.arg_of_perigee.to_le_bytes());
        raw.extend_from_slice(&self.0.mean_anomaly.to_le_bytes());
        raw.extend_from_slice(&self.0.mean_motion.to_le_bytes());
        // v2: the epoch as the Instant's raw i64 microseconds (exact)
        raw.extend_from_slice(&self.0.epoch.raw.to_le_bytes());
        raw.extend_from_slice(&self.0.rev_num.to_le_bytes());
        raw.extend_from_slice(&self.0.element_num.to_le_bytes());
        raw.push(self.0.ephem_type);
        raw.push(self.0.classification as u8);

        for s in [&self.0.name, &self.0.intl_desig, &self.0.desig_piece] {
            raw.extend_from_slice(&(s.len() as u16).to_le_bytes());
            raw.extend_from_slice(s.as_bytes());
        }

        pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let raw = state.extract::<Vec<u8>>(py)?;
        let bail = || {
            pyo3::exceptions::PyValueError::new_err("invalid TLE pickle: truncated or malformed")
        };

        // Version byte + at least the 101-byte v1/v2 fixed field block (see
        // __getstate__); v3 additionally has a classification byte at 102.
        if raw.len() < 102 {
            return Err(bail());
        }
        // Versions differ only in the epoch field at bytes 85..93, and v3
        // adds a classification byte at 102:
        //   v1 (satkit 0.20 – 0.23): TAI MJD as f64, which lost a
        //       microsecond ~1% of the time; still read, rounded to the
        //       nearest microsecond
        //   v2 (satkit 0.24-dev): the Instant's raw i64 microseconds, exact
        //   v3: v2, plus the classification letter; a v1/v2 pickle (no
        //       classification byte) loads with 'U'
        let version = raw[0];
        if version != 1 && version != 2 && version != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported TLE pickle version {} (expected 1, 2 or 3); pickles from \
                 satkit <= 0.19 must be regenerated",
                version
            )));
        }
        let rd_i32 = |at: usize| i32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
        let rd_f64 = |at: usize| f64::from_le_bytes(raw[at..at + 8].try_into().unwrap());

        self.0.sat_num = rd_i32(1);
        self.0.desig_year = rd_i32(5);
        self.0.desig_launch = rd_i32(9);
        self.0.mean_motion_dot = rd_f64(13);
        self.0.mean_motion_dot_dot = rd_f64(21);
        self.0.bstar = rd_f64(29);
        self.0.inclination = rd_f64(37);
        self.0.raan = rd_f64(45);
        self.0.eccen = rd_f64(53);
        self.0.arg_of_perigee = rd_f64(61);
        self.0.mean_anomaly = rd_f64(69);
        self.0.mean_motion = rd_f64(77);
        self.0.epoch = match version {
            1 => satkit::Instant::from_mjd_with_scale(rd_f64(85), satkit::TimeScale::TAI),
            _ => satkit::Instant::new(i64::from_le_bytes(raw[85..93].try_into().unwrap())),
        };
        self.0.rev_num = rd_i32(93);
        self.0.element_num = rd_i32(97);
        self.0.ephem_type = raw[101];

        // v3 has a classification byte right after the v1/v2 block; a
        // v1/v2 pickle carries no classification, so it loads as 'U'.
        let mut cnt = 102;
        self.0.classification = if version == 3 {
            if raw.len() < 103 {
                return Err(bail());
            }
            let c = raw[102];
            cnt = 103;
            if c.is_ascii_alphabetic() {
                c as char
            } else {
                return Err(bail());
            }
        } else {
            'U'
        };

        // Three length-prefixed UTF-8 strings: name, intl_desig, desig_piece.
        let read_str = |cnt: &mut usize| -> PyResult<String> {
            if *cnt + 2 > raw.len() {
                return Err(bail());
            }
            let len = u16::from_le_bytes(raw[*cnt..*cnt + 2].try_into().unwrap()) as usize;
            *cnt += 2;
            if *cnt + len > raw.len() {
                return Err(bail());
            }
            let s = String::from_utf8(raw[*cnt..*cnt + len].to_vec()).map_err(|_| bail())?;
            *cnt += len;
            Ok(s)
        };
        self.0.name = read_str(&mut cnt)?;
        self.0.intl_desig = read_str(&mut cnt)?;
        self.0.desig_piece = read_str(&mut cnt)?;
        self.0.reset_cache();

        Ok(())
    }
}

impl<'b> From<&'b mut PyTLE> for &'b mut TLE {
    fn from(s: &mut PyTLE) -> &mut TLE {
        &mut s.0
    }
}
