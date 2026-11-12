use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use satkit::tle::TLE;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};

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
///         "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
///         "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
///     ])[0]
///     print(tle.name)
///     # ISS (ZARYA)
///     ```
///
/// ``TLE()`` takes no arguments and gives an empty element set to fill in
/// through the properties: name ``"none"``, satellite number 0, a blank
/// international designator, classification ``"U"``, epoch J2000 and every
/// element zero.
///
/// Two TLEs are equal (``==``) when their names and every element are
/// equal; a TLE is mutable, so it is not hashable. ``repr(tle)`` is one
/// line naming the satellite and epoch; ``str(tle)`` lists every element.
#[pyclass(name = "TLE", module = "satkit")]
pub struct PyTLE(pub TLE);

pyo3::create_exception!(
    satkit,
    TLEWarning,
    pyo3::exceptions::PyUserWarning,
    "Warning for a TLE record skipped by a loader called with ``on_error=\"warn\"``\n\nThe message is the one ``on_error=\"raise\"`` raises as ``ValueError``: it gives\nthe input line the record starts on, its satellite and what failed to parse.\nWith ``on_error=\"warn\"``, input that holds no element sets at all also gets\none, saying so."
);

/// What a TLE loader does with a record that fails to parse (`on_error`)
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnError {
    Raise,
    Warn,
    Skip,
}

impl OnError {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "raise" => Ok(Self::Raise),
            "warn" => Ok(Self::Warn),
            "skip" => Ok(Self::Skip),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "on_error must be \"raise\", \"warn\" or \"skip\", not {s:?}"
            ))),
        }
    }
}

/// Parse `lines`, one result per record
fn parse_records(lines: &[String], check_checksum: bool) -> Vec<satkit::tle::Result<TLE>> {
    TLE::records(lines)
        .check_checksums(check_checksum)
        .collect()
}

/// Emit a `TLEWarning` attributed to the Python line that called the loader
/// (stack level 1).
fn tle_warning(py: Python<'_>, msg: &str) -> PyResult<()> {
    // An interior NUL cannot be in a C string; it never occurs in these
    // messages, but must not panic.
    let msg = std::ffi::CString::new(msg.replace('\0', " "))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    PyErr::warn(py, py.get_type::<TLEWarning>().as_any(), &msg, 1)
}

/// The TLEs of `records` as a list, treating input that is not a valid
/// element set as `on_error` says: a record that fails to parse, and input
/// with no records at all (`what` names the source in that message). Only
/// `on_error="raise"` raises for either; `"warn"` and `"skip"` return the
/// TLEs that parsed, which may be none.
fn tle_list(
    py: Python<'_>,
    records: Vec<satkit::tle::Result<TLE>>,
    on_error: OnError,
    what: &str,
) -> PyResult<Vec<PyTLE>> {
    if records.is_empty() {
        let msg = format!("No valid TLEs found in {what}");
        match on_error {
            OnError::Raise => return Err(PyValueError::new_err(msg)),
            OnError::Warn => tle_warning(py, &msg)?,
            OnError::Skip => {}
        }
    }
    let mut tles = Vec::with_capacity(records.len());
    for rec in records {
        match rec {
            Ok(tle) => tles.push(PyTLE(tle)),
            Err(e) if on_error == OnError::Raise => return Err(tle_err(e)),
            Err(e) if on_error == OnError::Warn => tle_warning(py, &e.to_string())?,
            Err(_) => {}
        }
    }
    Ok(tles)
}

/// `ValueError` unless `value` is in `0..=max`
fn check_desig_part(name: &str, value: i32, max: i32) -> PyResult<()> {
    if (0..=max).contains(&value) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be between 0 and {max}, got {value}"
        )))
    }
}

/// `ValueError` unless `value` fits the international designator field of a
/// TLE (line 1, columns 10-17)
fn check_intl_desig(value: &str) -> PyResult<()> {
    if value.len() <= 8 && value.bytes().all(|b| (0x20..0x7f).contains(&b)) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "intl_desig must be at most 8 printable ASCII characters, got {value:?}"
        )))
    }
}

/// The Python exception for a TLE error: `ValueError` for input that is not
/// a valid element set (text that does not parse, an element set that does
/// not fit the two-line format, fit inputs that do not match), the matching
/// `OSError` for an I/O failure, and `RuntimeError` for what fails at run
/// time (a download, offline mode, the fit itself).
pub(crate) fn tle_err(e: satkit::tle::Error) -> PyErr {
    use satkit::tle::Error as E;
    match e {
        E::Io(e) => e.into(),
        E::InvalidLineLengths { .. }
        | E::LineTooShort { .. }
        | E::NonAscii { .. }
        | E::ParseField { .. }
        | E::ChecksumMismatch { .. }
        | E::Record { .. }
        | E::SatNumMismatch { .. }
        | E::MissingLine2
        | E::MissingLine1
        | E::YearOutOfRange(_)
        | E::InvalidSatNum(_)
        | E::InvalidFirstDigit(_)
        | E::EmptySatNum
        | E::SatNumTooLargeForAlpha5
        | E::InvalidSatNumValue
        | E::InvalidEpoch(_)
        | E::StatesTimesLengthMismatch
        | E::EmptyStates
        | E::EpochOutOfRange { .. }
        | E::EccentricityOutOfRange(_)
        | E::InvalidIntlDesig(_) => PyValueError::new_err(e.to_string()),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

/// The Python exception for a failure to read the file `path`: the
/// `OSError` subclass of its errno (`FileNotFoundError`, `PermissionError`,
/// ...) with the path as its `filename`, or `ValueError` for content that
/// is not text (no errno).
pub(crate) fn io_err(e: std::io::Error, path: &std::path::Path) -> PyErr {
    let path = path.display().to_string();
    match e.raw_os_error() {
        Some(errno) => {
            // "No such file or directory (os error 2)" without the suffix
            let msg = e.to_string();
            let msg = msg.split(" (os error").next().unwrap_or(&msg).to_string();
            // OSError(errno, strerror, filename) picks the subclass itself
            PyOSError::new_err((errno, msg, path))
        }
        None => PyValueError::new_err(format!("{path}: {e}")),
    }
}

/// `value`, or `ValueError` if it is NaN or infinite
fn finite(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

#[pymethods]
impl PyTLE {
    /// Load TLEs from a text file, parsed as ``TLE.from_lines`` does
    ///
    /// Args:
    ///     filename (str | os.PathLike): path of the text file holding the TLE lines
    ///     check_checksum (bool, optional): as in ``TLE.from_lines``
    ///     on_error (str, optional): as in ``TLE.from_lines``
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set in the file
    ///
    /// Raises:
    ///     FileNotFoundError: if the file does not exist; the exception
    ///         names the path. Another I/O failure raises the matching
    ///         ``OSError`` (``PermissionError``, ``IsADirectoryError``, ...)
    ///     ValueError: with ``on_error="raise"``, if a record fails to parse
    ///         or the file holds no TLEs; see ``TLE.from_lines``
    #[staticmethod]
    #[pyo3(signature = (filename, *, check_checksum=false, on_error="raise"))]
    fn from_file(
        py: Python<'_>,
        filename: std::path::PathBuf,
        check_checksum: bool,
        on_error: &str,
    ) -> PyResult<Vec<Self>> {
        let on_error = OnError::parse(on_error)?;
        let file = File::open(&filename).map_err(|e| io_err(e, &filename))?;

        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| io_err(e, &filename))?;

        tle_list(
            py,
            parse_records(&lines, check_checksum),
            on_error,
            &filename.display().to_string(),
        )
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
    ///     on_error (str, optional): what to do with input that is not a
    ///         valid element set. ``"raise"`` (default) raises ``ValueError``
    ///         at the first record that fails to parse, and also if the
    ///         lines hold no element sets at all. ``"warn"`` leaves a bad
    ///         record out and emits a ``satkit.TLEWarning`` with the same
    ///         message (one more if the lines hold no element sets at all).
    ///         ``"skip"`` leaves bad records out silently. Only ``"raise"``
    ///         raises for what the lines contain: ``"warn"`` and ``"skip"``
    ///         always return a list
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set, even if there is only one.
    ///     With ``on_error="warn"`` or ``"skip"``, one per element set that
    ///     parsed: an empty list if none did, or if the lines (empty, or
    ///     text that is not TLE data) held none to begin with.
    ///
    /// Raises:
    ///     ValueError: with ``on_error="raise"``, if a record fails to parse
    ///         (or, with ``check_checksum``, has a wrong checksum), where the
    ///         message gives the line the record starts on and its
    ///         satellite, or if the lines hold no element sets; with any
    ///         ``on_error``, if it is not one of the three values
    ///     TypeError: if ``lines`` is not a sequence of strings
    ///
    /// Example:
    ///     ```python
    ///     import warnings
    ///
    ///     lines = [
    ///         "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
    ///         "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
    ///         # A letter O in the epoch day
    ///         "1 49140U 21082L   24O30.39663557  .00000076  00000-0  14180-4 0  9995",
    ///         "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
    ///     ]
    ///     with warnings.catch_warnings(record=True) as skipped:
    ///         warnings.simplefilter("always")
    ///         tles = satkit.TLE.from_lines(lines, on_error="warn")
    ///     print(len(tles), len(skipped))
    ///     # 1 1
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (lines, *, check_checksum=false, on_error="raise"))]
    fn from_lines(
        py: Python<'_>,
        lines: Vec<String>,
        check_checksum: bool,
        on_error: &str,
    ) -> PyResult<Vec<Self>> {
        let on_error = OnError::parse(on_error)?;
        tle_list(py, parse_records(&lines, check_checksum), on_error, "input")
    }

    /// Load TLEs from a URL, parsing the response as ``TLE.from_lines`` does
    ///
    /// Works with any URL that returns plain-text TLE data.
    ///
    /// Args:
    ///     url (str): URL to fetch TLE data from
    ///     on_error (str, optional): as in ``TLE.from_lines``
    ///
    /// Returns:
    ///     list[TLE]: one TLE per element set in the response
    ///
    /// Raises:
    ///     ValueError: with ``on_error="raise"``, if a record fails to parse
    ///         or the response holds no TLEs; see ``TLE.from_lines``
    ///     RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
    ///         ``satkit.utils.set_offline(True)``; no connection is opened)
    ///         or the request fails
    ///
    /// Example:
    ///     ```python
    ///     tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (url, *, on_error="raise"))]
    fn from_url(py: Python, url: String, on_error: &str) -> PyResult<Vec<Self>> {
        let on_error = OnError::parse(on_error)?;
        let records = py.detach(|| TLE::records_from_url(&url)).map_err(tle_err)?;
        tle_list(py, records, on_error, "response")
    }

    /// Satellite number, or equivalently the NORAD catalog number
    ///
    /// Setting it raises ``ValueError`` unless 0 <= value <= 999999999. A
    /// number above 339999 (the end of the Alpha-5 scheme) is allowed, as
    /// OMM carries up to nine digits, but cannot be written by ``to_2line``.
    #[getter(satnum)]
    const fn get_satnum(&self) -> i32 {
        self.0.sat_num
    }

    #[setter(satnum)]
    fn set_satnum(&mut self, value: i32) -> PyResult<()> {
        // OMM's NORAD_CAT_ID holds up to nine digits
        if !(0..=999_999_999).contains(&value) {
            return Err(PyValueError::new_err(format!(
                "satnum must be between 0 and 999999999, got {value}"
            )));
        }
        self.0.sat_num = value;
        self.0.reset_cache();
        Ok(())
    }

    /// International designator as written in the TLE (line 1, columns 10-17)
    ///
    /// For example ``"98067A"``: the last two digits of the launch year,
    /// the three-digit launch number of that year and the piece of the
    /// launch (one to three letters). An empty string if the element set
    /// has none, as for an analyst object or a new ``TLE()``.
    ///
    /// This string is the stored value: ``to_2line`` writes it and
    /// ``to_omm`` derives ``OBJECT_ID`` from it. ``desig_year``,
    /// ``desig_launch`` and ``desig_piece`` are read from it, and setting
    /// any of them rewrites it.
    ///
    /// Setting it raises ``ValueError`` if the value is longer than the 8
    /// columns of the TLE field or holds anything but printable ASCII.
    /// Surrounding spaces are removed.
    #[getter(intl_desig)]
    fn get_intl_desig(&self) -> String {
        self.0.intl_desig.clone()
    }

    #[setter(intl_desig)]
    fn set_intl_desig(&mut self, value: &str) -> PyResult<()> {
        let value = value.trim();
        check_intl_desig(value)?;
        self.0.intl_desig = value.to_string();
        self.0.reset_cache();
        Ok(())
    }

    /// Launch year from the international designator, two digits as in the TLE
    ///
    /// The first two characters of ``intl_desig`` (98 for ``"98067A"``):
    /// 57-99 are 1957-1999 and 0-56 are 2000-2056. ``None`` if the
    /// designator is blank or does not start with two digits.
    ///
    /// Setting it rewrites ``intl_desig`` and raises ``ValueError`` unless
    /// 0 <= value <= 99. A launch number the designator does not have yet
    /// is written as ``000``.
    #[getter(desig_year)]
    fn get_desig_year(&self) -> Option<i32> {
        self.0.desig_year()
    }

    #[setter(desig_year)]
    fn set_desig_year(&mut self, value: i32) -> PyResult<()> {
        check_desig_part("desig_year", value, 99)?;
        let launch = self.0.desig_launch().unwrap_or(0);
        self.set_desig_parts(value, launch, self.0.desig_piece().to_string());
        Ok(())
    }

    /// Launch number of the year from the international designator
    ///
    /// Characters 3-5 of ``intl_desig`` (67 for ``"98067A"``). ``None`` if
    /// the designator is blank or does not hold three digits there.
    ///
    /// Setting it rewrites ``intl_desig`` and raises ``ValueError`` unless
    /// 0 <= value <= 999. A launch year the designator does not have yet
    /// is written as ``00``.
    #[getter(desig_launch)]
    fn get_desig_launch(&self) -> Option<i32> {
        self.0.desig_launch()
    }

    #[setter(desig_launch)]
    fn set_desig_launch(&mut self, value: i32) -> PyResult<()> {
        check_desig_part("desig_launch", value, 999)?;
        let year = self.0.desig_year().unwrap_or(0);
        self.set_desig_parts(year, value, self.0.desig_piece().to_string());
        Ok(())
    }

    /// Piece of the launch from the international designator (e.g. "A")
    ///
    /// What follows the year and launch number in ``intl_desig``; an empty
    /// string if the designator is blank or has no piece.
    ///
    /// Setting it rewrites ``intl_desig`` and raises ``ValueError`` unless
    /// it is at most three ASCII letters. A launch year and number the
    /// designator does not have yet are written as zeros.
    #[getter(desig_piece)]
    fn get_desig_piece(&self) -> &str {
        self.0.desig_piece()
    }

    #[setter(desig_piece)]
    fn set_desig_piece(&mut self, value: &str) -> PyResult<()> {
        if value.len() > 3 || !value.bytes().all(|b| b.is_ascii_alphabetic()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "desig_piece must be at most 3 ASCII letters, got {value:?}"
            )));
        }
        let year = self.0.desig_year().unwrap_or(0);
        let launch = self.0.desig_launch().unwrap_or(0);
        self.set_desig_parts(year, launch, value.to_string());
        Ok(())
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

    /// Ephemeris type (usually 0). A value of 4 marks an SGP4-XP element
    /// set, which ``satkit.sgp4`` rejects: satkit implements classic SGP4 only.
    ///
    /// Must be in 0..255 (``OverflowError`` otherwise); ``to_2line`` writes
    /// a value above 9 as 0.
    #[getter(ephem_type)]
    const fn get_ephem_type(&self) -> u8 {
        self.0.ephem_type
    }

    #[setter(ephem_type)]
    fn set_ephem_type(&mut self, value: u8) {
        self.0.ephem_type = value;
        self.0.reset_cache();
    }

    /// Element set number (``to_2line`` writes it modulo 10,000, a negative value as 0)
    #[getter(element_num)]
    const fn get_element_num(&self) -> i32 {
        self.0.element_num
    }

    #[setter(element_num)]
    fn set_element_num(&mut self, value: i32) {
        self.0.element_num = value;
        self.0.reset_cache();
    }

    /// Revolution number at epoch (``to_2line`` writes it modulo 100,000, a negative value as 0)
    #[getter(rev_num)]
    const fn get_rev_num(&self) -> i32 {
        self.0.rev_num
    }

    #[setter(rev_num)]
    fn set_rev_num(&mut self, value: i32) {
        self.0.rev_num = value;
        self.0.reset_cache();
    }

    /// Orbit eccentricity, unitless, in range [0, 1)
    ///
    /// Setting it raises ``ValueError`` unless 0 <= value < 1.
    #[getter(eccen)]
    const fn get_eccen(&self) -> f64 {
        self.0.eccen
    }

    #[setter(eccen)]
    fn set_eccen(&mut self, value: f64) -> PyResult<()> {
        // Also refuses NaN
        if !(0.0..1.0).contains(&value) {
            return Err(PyValueError::new_err(format!(
                "eccen must be in [0, 1), got {value}"
            )));
        }
        self.0.eccen = value;
        self.0.reset_cache();
        Ok(())
    }

    /// Mean anomaly in degrees (setting a value that is not finite raises ``ValueError``)
    #[getter(mean_anomaly)]
    const fn get_mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly
    }
    #[setter(mean_anomaly)]
    fn set_mean_anomaly(&mut self, value: f64) -> PyResult<()> {
        self.0.mean_anomaly = finite("mean_anomaly", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// Mean motion in revs / day (setting a negative or non-finite value raises ``ValueError``)
    #[getter(mean_motion)]
    const fn get_mean_motion(&self) -> f64 {
        self.0.mean_motion
    }
    #[setter(mean_motion)]
    fn set_mean_motion(&mut self, value: f64) -> PyResult<()> {
        if finite("mean_motion", value)? < 0.0 {
            return Err(PyValueError::new_err(format!(
                "mean_motion must not be negative, got {value}"
            )));
        }
        self.0.mean_motion = value;
        self.0.reset_cache();
        Ok(())
    }

    /// Inclination in degrees (setting a value that is not finite raises ``ValueError``)
    #[getter(inclination)]
    const fn get_inclination(&self) -> f64 {
        self.0.inclination
    }
    #[setter(inclination)]
    fn set_inclination(&mut self, value: f64) -> PyResult<()> {
        self.0.inclination = finite("inclination", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// Epoch time of TLE
    ///
    /// Set it with a single ``satkit.time``, ``datetime.datetime`` or
    /// ``numpy.datetime64``; anything else, including a list of times,
    /// raises ``TypeError``. ``to_2line`` can only write an epoch in the
    /// years 1957-2056.
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

    /// Argument of perigee, degrees (setting a value that is not finite raises ``ValueError``)
    #[getter(arg_of_perigee)]
    const fn get_arg_of_perigee(&self) -> f64 {
        self.0.arg_of_perigee
    }
    #[setter(arg_of_perigee)]
    fn set_arg_of_perigee(&mut self, value: f64) -> PyResult<()> {
        self.0.arg_of_perigee = finite("arg_of_perigee", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    /// (the TLE stores the half; setting a value that is not finite raises ``ValueError``)
    #[getter(mean_motion_dot)]
    const fn get_mean_motion_dot(&self) -> f64 {
        self.0.mean_motion_dot
    }
    #[setter(mean_motion_dot)]
    fn set_mean_motion_dot(&mut self, value: f64) -> PyResult<()> {
        self.0.mean_motion_dot = finite("mean_motion_dot", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// One sixth of 2nd derivative of mean motion wrt time, in revs/day^3
    /// (the TLE stores the sixth; setting a value that is not finite raises ``ValueError``)
    #[getter(mean_motion_dot_dot)]
    const fn get_mean_motion_dot_dot(&self) -> f64 {
        self.0.mean_motion_dot_dot
    }
    #[setter(mean_motion_dot_dot)]
    fn set_mean_motion_dot_dot(&mut self, value: f64) -> PyResult<()> {
        self.0.mean_motion_dot_dot = finite("mean_motion_dot_dot", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// Right Ascension of the Ascending Node, degrees (setting a value that is not finite raises ``ValueError``)
    #[getter(raan)]
    const fn get_raan(&self) -> f64 {
        self.0.raan
    }
    #[setter(raan)]
    fn set_raan(&mut self, value: f64) -> PyResult<()> {
        self.0.raan = finite("raan", value)?;
        self.0.reset_cache();
        Ok(())
    }

    /// Name of the satellite: the name line of a 3-line element set
    ///
    /// ``"none"`` for an element set that has no name (one read from two
    /// lines, or a new ``TLE()``).
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
    /// (setting a value that is not finite raises ``ValueError``)
    #[getter(bstar)]
    const fn bstar(&self) -> f64 {
        self.0.bstar
    }
    #[setter(bstar)]
    fn set_bstar(&mut self, value: f64) -> PyResult<()> {
        self.0.bstar = finite("bstar", value)?;
        self.0.reset_cache();
        Ok(())
    }

    fn __str__(&self) -> String {
        self.0.to_pretty_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "TLE(name={:?}, satnum={}, intl_desig={:?}, epoch={})",
            self.0.name, self.0.sat_num, self.0.intl_desig, self.0.epoch
        )
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
    /// ``satkit.OMMDict``), or the nested groups of an ``xmltodict``
    /// rendering of the XML form. Numbers may be strings.
    ///
    /// The six mean elements, epoch, ``BSTAR``, ``MEAN_MOTION_DOT``,
    /// ``MEAN_MOTION_DDOT``, ``NORAD_CAT_ID``, ``ELEMENT_SET_NO``,
    /// ``REV_AT_EPOCH`` and ``EPHEMERIS_TYPE`` carry over; of these, the
    /// optional ones become zero when absent. ``OBJECT_NAME`` becomes the
    /// name, and the unnamed-TLE name ``"none"`` when it is absent, empty or
    /// ``UNKNOWN``. ``OBJECT_ID`` in ``YYYY-NNNP`` form becomes the
    /// international designator (``1998-067A`` gives ``98067A``); any other
    /// form leaves the designator empty. ``CLASSIFICATION_TYPE`` becomes the
    /// classification, ``"U"`` unless it is a single ASCII letter. Other
    /// metadata is dropped.
    ///
    /// This inverts ``to_omm``: ``TLE.from_omm(tle.to_omm()) == tle`` for a
    /// TLE read from two or three lines.
    ///
    /// Args:
    ///     omm (dict): OMM dictionary
    ///
    /// Returns:
    ///     TLE: the equivalent two-line element set
    ///
    /// Raises:
    ///     TypeError: if ``omm`` is not a dict
    ///     ValueError: if a required key (``EPOCH`` or one of the six mean
    ///         elements) is missing or a value cannot be read
    #[staticmethod]
    fn from_omm(omm: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let omm = crate::pyomm::omm_from_pydict(omm)?;
        Ok(Self(omm.to_tle()))
    }

    /// Render this TLE as an OMM (Orbital Mean-Element Message) dictionary
    ///
    /// The result uses the flat CCSDS keys (see ``satkit.OMMDict``) with
    /// ``EPOCH`` as an RFC 3339 string, angles in degrees and mean motion in
    /// revolutions per day, and can be passed back to ``satkit.sgp4`` or
    /// serialized with ``json.dumps``. ``OBJECT_ID`` is derived from the
    /// international designator (``98067A`` becomes ``1998-067A``, a blank
    /// one ``UNKNOWN``), ``OBJECT_NAME`` is the name (``UNKNOWN`` for an
    /// unnamed TLE), and ``CLASSIFICATION_TYPE`` is the TLE's classification
    /// letter. A negative ``satnum``, ``element_num`` or ``rev_num`` and an
    /// ``ephem_type`` above 9 have no OMM form and are left out.
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
    /// Returns:
    ///     list[str]: the 2 canonical TLE lines
    ///
    /// Raises:
    ///     ValueError: if the element set does not fit the format:
    ///         ``satnum`` is 340000 or above (Alpha-5, used for the
    ///         5-character satellite number field, ends at ``Z9999`` =
    ///         339999; use ``to_omm()`` instead) or negative, ``eccen`` is
    ///         outside [0, 1), or the epoch year is outside 1957-2056 (the
    ///         span of the two-digit year)
    fn to_2line(&self) -> PyResult<[String; 2]> {
        self.0.to_2line().map_err(tle_err)
    }

    /// Output as 2 canonical TLE lines preceded by a name line (3-line element set)
    ///
    /// Returns:
    ///     list[str]: the name (``"none"`` for an unnamed TLE), then the 2
    ///     canonical TLE lines of ``to_2line``
    ///
    /// Raises:
    ///     ValueError: as ``to_2line``
    fn to_3line(&self) -> PyResult<[String; 3]> {
        self.0.to_3line().map_err(tle_err)
    }

    /// Perform non-linear least squares fit of TLE parameters to a list of GCRF states
    ///
    /// Args:
    ///     states (list[numpy.ndarray]): List of GCRF states to fit to. Each state is a
    ///         6-element vector: the first 3 values are position in meters, the last 3
    ///         values are velocity in meters / second
    ///     times (list[satkit.time]): Times corresponding to the states, one per state
    ///         (a single time is accepted for a single state)
    ///     epoch (satkit.time): Epoch time for the TLE: a single time, within the range of times
    ///
    /// Keyword Args:
    ///     gravconst (satkit.sgp4_gravconst): gravity model SGP4 uses in the fit.  Default is sgp4_gravconst.wgs72
    ///     opsmode (satkit.sgp4_opsmode): SGP4 ops mode used in the fit.  Default is sgp4_opsmode.afspc
    ///
    /// Returns:
    ///     tuple[TLE, dict]: Fitted TLE and the fit results, a dictionary with the keys
    ///     ``status`` (a ``satkit.tlefitstatus``), ``converged`` (bool), ``orig_norm``,
    ///     ``best_norm``, ``grad_norm`` (floats), ``n_iter`` and ``n_res_evals`` (ints).
    ///     The fitted TLE has the name ``"none"``, satellite number 0 and a blank
    ///     international designator; set them afterwards if needed.
    ///
    /// Raises:
    ///     TypeError: if ``states`` is not a sequence of 6-element vectors, ``times`` or
    ///         ``epoch`` is not a time (``epoch`` must be a single time, not a list), or
    ///         ``gravconst`` / ``opsmode`` is not the enum
    ///     ValueError: if ``states`` is empty, ``states`` and ``times`` differ in length,
    ///         or ``epoch`` is outside the range of ``times``
    ///     RuntimeError: if the fit itself fails (SGP4 cannot evaluate the elements,
    ///         singular normal equations, or it ends on elements a TLE cannot hold)
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
        // A single time; a list of times (even of one) is a TypeError
        epoch: crate::pyinstant::TimeArg,
        #[pyo3(from_py_with = crate::pysgp4::gravconst_arg)] gravconst: crate::pysgp4::GravConst,
        #[pyo3(from_py_with = crate::pysgp4::opsmode_arg)] opsmode: crate::pysgp4::OpsMode,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let times = crate::pysgp4::time_input(times)?.times;
        // Release the GIL during the (potentially long-running) fit
        let (gravconst, opsmode) = (gravconst.into(), opsmode.into());
        let (tle, result) = py
            .detach(|| TLE::fit_from_states_full(&states, &times, epoch.0, gravconst, opsmode))
            .map_err(tle_err)?;

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
        // The designator year, launch number and piece are written for
        // older readers of the format only: they are parts of intl_desig,
        // and `__setstate__` reads that.
        let desig_piece = self.0.desig_piece();
        let mut raw: Vec<u8> = Vec::with_capacity(
            109 + self.0.name.len() + self.0.intl_desig.len() + desig_piece.len(),
        );
        raw.push(3u8); // version
        raw.extend_from_slice(&self.0.sat_num.to_le_bytes());
        raw.extend_from_slice(&self.0.desig_year().unwrap_or(0).to_le_bytes());
        raw.extend_from_slice(&self.0.desig_launch().unwrap_or(0).to_le_bytes());
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

        for s in [&self.0.name, &self.0.intl_desig, desig_piece] {
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
        // Bytes 5..13 hold the designator year and launch number, which are
        // parts of intl_desig (read below)
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
        // The piece is part of intl_desig. satkit <= 0.24.3 kept only the
        // first two letters of a three-letter piece in intl_desig, but all
        // of it here: restore the third.
        let piece = read_str(&mut cnt)?;
        if piece.len() == 3
            && piece.is_ascii()
            && self.0.intl_desig.len() == 7
            && self.0.intl_desig.get(5..) == piece.get(..2)
        {
            self.0.intl_desig.push_str(&piece[2..]);
        }
        self.0.reset_cache();

        Ok(())
    }
}

impl PyTLE {
    /// Rewrite `intl_desig` from its three parts (already validated)
    fn set_desig_parts(&mut self, year: i32, launch: i32, piece: String) {
        self.0.intl_desig = format!("{year:02}{launch:03}{piece}");
        self.0.reset_cache();
    }
}

impl<'b> From<&'b mut PyTLE> for &'b mut TLE {
    fn from(s: &mut PyTLE) -> &mut TLE {
        &mut s.0
    }
}
