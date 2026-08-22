use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use satkit::tle::TLE;

use crate::pyinstant::ToTimeVec;
use anyhow::{bail, Result};

use crate::pytlefitstatus::PyTleFitStatus;

use std::fs::File;
use std::io;
use std::io::BufRead;

#[pyclass(name = "TLE", module = "satkit")]
pub struct PyTLE(pub TLE);

/// Convert a satkit::TLE into a Python PyTLE object
pub fn tle_into_py(tle: TLE, py: Python<'_>) -> Py<PyAny> {
    PyTLE(tle).into_py_any(py).unwrap()
}

#[pymethods]
impl PyTLE {
    /// Return a list of TLES loaded from input text file.
    ///
    /// If the file contains lines only represent a single TLE, the TLE will
    /// be output, rather than a list with a single TLE element
    ///
    /// # Arguments:
    ///
    /// * `filename` - name of textfile lines for TLE(s) to load
    ///
    /// # Returns:
    ///
    /// * `tle` - a list of TLE objects or a single TLE if lines for
    ///           only 1 are passed in
    #[staticmethod]
    fn from_file(filename: String) -> Result<Py<PyAny>> {
        let file = File::open(std::path::PathBuf::from(filename))?;

        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .collect::<std::result::Result<_, _>>()?;

        Self::from_lines(lines)
    }

    #[new]
    fn new() -> Self {
        Self(TLE::new())
    }

    /// Return a list of TLES loaded from input list of lines
    ///
    /// If the file contains lines only represent a single TLE, the TLE will
    /// be output, rather than a list with a single TLE element
    ///
    /// # Arguments:
    ///
    /// * `lines` - list of strings lines for TLE(s) to load
    ///
    /// # Returns:
    ///
    /// * `tle` - a list of TLE objects or a single TLE if lines for
    ///           only 1 are passed in
    #[staticmethod]
    fn from_lines(lines: Vec<String>) -> Result<Py<PyAny>> {
        let v = TLE::from_lines(&lines)?;
        pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
            if v.len() > 1 {
                v.into_iter()
                    .map(|t| tle_into_py(t, py))
                    .collect::<Vec<_>>()
                    .into_py_any(py)
            } else {
                match v.into_iter().next() {
                    Some(t) => Ok(tle_into_py(t, py)),
                    None => Err(pyo3::exceptions::PyValueError::new_err(
                        "No valid TLEs found in input",
                    )),
                }
            }
        })
        .map_err(|e| e.into())
    }

    /// Load TLE(s) from a URL
    ///
    /// Fetches the content at the given URL and parses it as TLE lines.
    /// Works with any URL that returns plain-text TLE data (2-line or 3-line format).
    ///
    /// Args:
    ///     url (str): URL to fetch TLE data from
    ///
    /// Returns:
    ///     TLE or list[TLE]: Single TLE or list of TLEs parsed from the response
    ///
    /// Example:
    ///     ```python
    ///     tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
    ///     ```
    #[staticmethod]
    fn from_url(url: String) -> Result<Py<PyAny>> {
        let tles = TLE::from_url(&url)?;
        pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
            if tles.len() > 1 {
                tles.into_iter()
                    .map(|t| tle_into_py(t, py))
                    .collect::<Vec<_>>()
                    .into_py_any(py)
            } else {
                match tles.into_iter().next() {
                    Some(t) => Ok(tle_into_py(t, py)),
                    None => Err(pyo3::exceptions::PyValueError::new_err(
                        "No valid TLEs found in response",
                    )),
                }
            }
        })
        .map_err(|e| e.into())
    }

    /// Satellite NORAD Catalog Number
    #[getter(satnum)]
    const fn get_satnum(&self) -> i32 {
        self.0.sat_num
    }

    #[setter(satnum)]
    fn set_satnum(&mut self, value: i32) {
        self.0.sat_num = value;
    }

    /// International designator (e.g. "98067A": launch year, launch number, piece)
    #[getter(intl_desig)]
    fn get_intl_desig(&self) -> String {
        self.0.intl_desig.clone()
    }

    #[setter(intl_desig)]
    fn set_intl_desig(&mut self, value: String) {
        self.0.intl_desig = value;
    }

    /// Launch year from the international designator (2-digit, as in the TLE)
    #[getter(desig_year)]
    const fn get_desig_year(&self) -> i32 {
        self.0.desig_year
    }

    #[setter(desig_year)]
    fn set_desig_year(&mut self, value: i32) {
        self.0.desig_year = value;
    }

    /// Launch number of the year from the international designator
    #[getter(desig_launch)]
    const fn get_desig_launch(&self) -> i32 {
        self.0.desig_launch
    }

    #[setter(desig_launch)]
    fn set_desig_launch(&mut self, value: i32) {
        self.0.desig_launch = value;
    }

    /// Piece of the launch from the international designator (e.g. "A")
    #[getter(desig_piece)]
    fn get_desig_piece(&self) -> String {
        self.0.desig_piece.clone()
    }

    #[setter(desig_piece)]
    fn set_desig_piece(&mut self, value: String) {
        self.0.desig_piece = value;
    }

    /// Ephemeris type (usually 0)
    #[getter(ephem_type)]
    const fn get_ephem_type(&self) -> u8 {
        self.0.ephem_type
    }

    #[setter(ephem_type)]
    fn set_ephem_type(&mut self, value: u8) {
        self.0.ephem_type = value;
    }

    /// Element set number
    #[getter(element_num)]
    const fn get_element_num(&self) -> i32 {
        self.0.element_num
    }

    #[setter(element_num)]
    fn set_element_num(&mut self, value: i32) {
        self.0.element_num = value;
    }

    /// Revolution number at epoch
    #[getter(rev_num)]
    const fn get_rev_num(&self) -> i32 {
        self.0.rev_num
    }

    #[setter(rev_num)]
    fn set_rev_num(&mut self, value: i32) {
        self.0.rev_num = value;
    }

    /// Orbit eccentricity
    #[getter(eccen)]
    const fn get_eccen(&self) -> f64 {
        self.0.eccen
    }

    #[setter(eccen)]
    fn set_eccen(&mut self, value: f64) {
        self.0.eccen = value;
    }

    /// Mean anomaly in degrees
    #[getter(mean_anomaly)]
    const fn get_mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly
    }
    #[setter(mean_anomaly)]
    fn set_mean_anomaly(&mut self, value: f64) {
        self.0.mean_anomaly = value;
    }

    /// Mean motion in revs / day
    #[getter(mean_motion)]
    const fn get_mean_motion(&self) -> f64 {
        self.0.mean_motion
    }
    #[setter(mean_motion)]
    fn set_mean_motion(&mut self, value: f64) {
        self.0.mean_motion = value;
    }

    /// inclination in degrees
    #[getter(inclination)]
    const fn get_inclination(&self) -> f64 {
        self.0.inclination
    }
    #[setter(inclination)]
    fn set_inclination(&mut self, value: f64) {
        self.0.inclination = value;
    }

    /// Epoch time of TLE
    #[getter(epoch)]
    fn get_epoch(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(crate::pyinstant::instant_into_py(self.0.epoch, py))
    }
    #[setter(epoch)]
    fn set_epoch(&mut self, value: &Bound<'_, PyAny>) -> Result<()> {
        let epoch = value.to_time_vec()?;
        if epoch.is_empty() {
            bail!("epoch must be a single time value");
        }
        self.0.epoch = epoch[0];
        Ok(())
    }

    /// argument of perigee, degrees
    #[getter(arg_of_perigee)]
    const fn get_arg_of_perigee(&self) -> f64 {
        self.0.arg_of_perigee
    }
    #[setter(arg_of_perigee)]
    fn set_arg_of_perigee(&mut self, value: f64) {
        self.0.arg_of_perigee = value;
    }

    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    #[getter(mean_motion_dot)]
    const fn get_mean_motion_dot(&self) -> f64 {
        self.0.mean_motion_dot
    }
    #[setter(mean_motion_dot)]
    fn set_mean_motion_dot(&mut self, value: f64) {
        self.0.mean_motion_dot = value;
    }

    /// One sixth of 2nd derivative of mean motion wrt time, in revs/day^3
    #[getter(mean_motion_dot_dot)]
    const fn get_mean_motion_dot_dot(&self) -> f64 {
        self.0.mean_motion_dot_dot
    }
    #[setter(mean_motion_dot_dot)]
    fn set_mean_motion_dot_dot(&mut self, value: f64) {
        self.0.mean_motion_dot_dot = value;
    }

    /// Right Ascension of the Ascending Node, degrees
    #[getter(raan)]
    const fn get_raan(&self) -> f64 {
        self.0.raan
    }
    #[setter(raan)]
    fn set_raan(&mut self, value: f64) {
        self.0.raan = value;
    }

    /// Name of satellite
    #[getter(name)]
    fn name(&self) -> String {
        self.0.name.clone()
    }
    #[setter(name)]
    fn set_name(&mut self, value: String) {
        self.0.name = value;
    }

    // Drag
    #[getter(bstar)]
    const fn bstar(&self) -> f64 {
        self.0.bstar
    }
    #[setter(bstar)]
    fn set_bstar(&mut self, value: f64) {
        self.0.bstar = value;
    }

    fn __str__(&self) -> String {
        self.0.to_pretty_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __ne__(&self, other: &Self) -> bool {
        self.0 != other.0
    }

    /// Output as 2 canonical TLE Lines
    fn to_2line(&self) -> Result<[String; 2]> {
        Ok(self.0.to_2line()?)
    }

    // Output as 2 canonical TLE lines preceded by a name line (3-line element set)
    fn to_3line(&self) -> Result<[String; 3]> {
        Ok(self.0.to_3line()?)
    }

    // Fit a TLE from GCRF states and times
    #[staticmethod]
    fn fit_from_states(
        py: Python,
        states: Vec<[f64; 6]>,
        times: &Bound<'_, PyAny>,
        epoch: &Bound<'_, PyAny>,
    ) -> Result<(Self, Py<PyAny>)> {
        let times = times.to_time_vec()?;
        let epoch = epoch.to_time_vec()?;
        if epoch.len() != 1 {
            bail!("epoch must be a single time value");
        }
        // Release the GIL during the (potentially long-running) fit
        let (tle, result) = py.detach(|| TLE::fit_from_states(&states, &times, epoch[0]))?;

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
        // Self-describing format v1 (see `__setstate__` for the layout):
        // a leading version byte, a 101-byte fixed field block, then three
        // length-prefixed UTF-8 strings (name, intl_desig, desig_piece).
        let mut raw: Vec<u8> = Vec::with_capacity(
            108 + self.0.name.len() + self.0.intl_desig.len() + self.0.desig_piece.len(),
        );
        raw.push(1u8); // version
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
        raw.extend_from_slice(
            &self
                .0
                .epoch
                .as_mjd_with_scale(satkit::TimeScale::TAI)
                .to_le_bytes(),
        );
        raw.extend_from_slice(&self.0.rev_num.to_le_bytes());
        raw.extend_from_slice(&self.0.element_num.to_le_bytes());
        raw.push(self.0.ephem_type);

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

        // Version byte + 101-byte fixed field block (see __getstate__).
        if raw.len() < 102 {
            return Err(bail());
        }
        if raw[0] != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported TLE pickle version {} (expected 1); pickles from \
                 satkit <= 0.19 must be regenerated",
                raw[0]
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
        self.0.epoch = satkit::Instant::from_mjd_with_scale(rd_f64(85), satkit::TimeScale::TAI);
        self.0.rev_num = rd_i32(93);
        self.0.element_num = rd_i32(97);
        self.0.ephem_type = raw[101];

        // Three length-prefixed UTF-8 strings: name, intl_desig, desig_piece.
        let mut cnt = 102;
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

        Ok(())
    }
}

impl<'b> From<&'b mut PyTLE> for &'b mut TLE {
    fn from(s: &mut PyTLE) -> &mut TLE {
        &mut s.0
    }
}
