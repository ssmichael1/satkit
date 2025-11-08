use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::tle::TLE;

use crate::pybindings::pyinstant::ToTimeVec;
use anyhow::{bail, Result};

// Import PyMPSuccess from its module (adjust the path if needed)
use crate::pybindings::pympsuccess::PyMPSuccess;

use std::fs::File;
use std::io;
use std::io::BufRead;

#[pyclass(name = "TLE", module = "satkit")]
pub struct PyTLE(pub TLE);
impl<'py> IntoPyObject<'py> for TLE {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyTLE(self).into_bound_py_any(py).unwrap())
    }
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
            .map(|v| -> String { v.unwrap() })
            .collect();

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
        TLE::from_lines(&lines).and_then(|v| {
            pyo3::Python::attach(|py| {
                if v.len() > 1 {
                    v.into_py_any(py)
                } else {
                    v[0].clone().into_py_any(py)
                }
            })
            .map_err(|e| e.into())
        })
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
        self.0.epoch.into_py_any(py)
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

    /// Output as 2 canonical TLE Lines
    fn to_2line(&self) -> Result<[String; 2]> {
        self.0.to_2line()
    }

    // Output as 2 canonical TLE lines preceded by a name line (3-line element set)
    fn to_3line(&self) -> Result<[String; 3]> {
        self.0.to_3line()
    }

    // Fit a TLE from GCRF states and times
    #[staticmethod]
    fn fit_from_states(
        states: Vec<[f64; 6]>,
        times: &Bound<'_, PyAny>,
        epoch: &Bound<'_, PyAny>,
    ) -> Result<(Self, Py<PyAny>)> {
        let times = times.to_time_vec()?;
        let epoch = epoch.to_time_vec()?;
        if epoch.len() != 1 {
            bail!("epoch must be a single time value");
        }
        let (tle, status) = TLE::fit_from_states(&states, &times, epoch[0])?;

        Ok((
            Self(tle),
            pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("success", PyMPSuccess::from(status.success))?;
                dict.set_item("best_norm", status.best_norm)?;
                dict.set_item("orig_norm", status.orig_norm)?;
                dict.set_item("n_iter", status.n_iter)?;
                dict.set_item("n_fev", status.n_fev)?;
                dict.set_item("n_par", status.n_par)?;
                dict.set_item("n_free", status.n_free)?;
                dict.set_item("n_pegged", status.n_pegged)?;
                dict.set_item("n_func", status.n_func)?;
                dict.set_item("resid", status.resid)?;
                dict.set_item("xerror", status.xerror)?;
                dict.set_item("covar", status.covar)?;

                Ok(dict.into())
            })?,
        ))
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        let nbytes: usize =
            102 + self.0.name.len() + self.0.intl_desig.len() + self.0.desig_piece.len();
        let mut raw = vec![0u8; nbytes];
        raw[0..4].clone_from_slice(&self.0.sat_num.to_le_bytes());
        raw[4..8].clone_from_slice(&self.0.desig_year.to_le_bytes());
        raw[8..12].clone_from_slice(&self.0.desig_launch.to_le_bytes());
        raw[12..20].clone_from_slice(&self.0.mean_motion_dot.to_le_bytes());
        raw[20..28].clone_from_slice(&self.0.mean_motion_dot_dot.to_le_bytes());
        raw[28..36].clone_from_slice(&self.0.bstar.to_le_bytes());
        raw[36..44].clone_from_slice(&self.0.inclination.to_le_bytes());
        raw[44..52].clone_from_slice(&self.0.raan.to_le_bytes());
        raw[52..60].clone_from_slice(&self.0.eccen.to_le_bytes());
        raw[60..68].clone_from_slice(&self.0.arg_of_perigee.to_le_bytes());
        raw[68..76].clone_from_slice(&self.0.mean_anomaly.to_le_bytes());
        raw[76..84].clone_from_slice(&self.0.mean_motion.to_le_bytes());
        raw[84..92].clone_from_slice(
            &self
                .0
                .epoch
                .as_mjd_with_scale(crate::TimeScale::TAI)
                .to_le_bytes(),
        );
        raw[92..96].clone_from_slice(&self.0.rev_num.to_le_bytes());

        let mut cnt = 96;

        let namelen = self.0.name.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&namelen.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.0.name.len()].clone_from_slice(self.0.name.as_bytes());
        cnt += self.0.name.len();

        let intl_len = self.0.intl_desig.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&intl_len.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.0.intl_desig.len()].clone_from_slice(self.0.intl_desig.as_bytes());
        cnt += self.0.intl_desig.len();

        let piece_len = self.0.desig_piece.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&piece_len.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.0.desig_piece.len()].clone_from_slice(self.0.desig_piece.as_bytes());

        pyo3::types::PyBytes::new(py, &raw).into_py_any(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let raw = state.extract::<Vec<u8>>(py)?;

        self.0.sat_num = i32::from_le_bytes(raw[0..4].try_into().unwrap());
        self.0.desig_year = i32::from_le_bytes(raw[4..8].try_into().unwrap());
        self.0.desig_launch = i32::from_le_bytes(raw[8..12].try_into().unwrap());
        self.0.mean_motion_dot = f64::from_le_bytes(raw[12..20].try_into().unwrap());
        self.0.mean_motion_dot_dot = f64::from_le_bytes(raw[20..28].try_into().unwrap());
        self.0.bstar = f64::from_le_bytes(raw[28..36].try_into().unwrap());
        self.0.inclination = f64::from_le_bytes(raw[36..44].try_into().unwrap());
        self.0.raan = f64::from_le_bytes(raw[44..52].try_into().unwrap());
        self.0.eccen = f64::from_le_bytes(raw[52..60].try_into().unwrap());
        self.0.arg_of_perigee = f64::from_le_bytes(raw[60..68].try_into().unwrap());
        self.0.mean_anomaly = f64::from_le_bytes(raw[68..76].try_into().unwrap());
        self.0.mean_motion = f64::from_le_bytes(raw[76..84].try_into().unwrap());
        self.0.epoch = crate::Instant::from_mjd_with_scale(
            f64::from_le_bytes(raw[84..92].try_into().unwrap()),
            crate::TimeScale::TAI,
        );
        self.0.rev_num = i32::from_le_bytes(raw[92..96].try_into().unwrap());

        let mut cnt = 96;

        let namelen = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.0.name = String::from_utf8(raw[cnt..cnt + namelen as usize].to_vec()).unwrap();
        cnt += namelen as usize;

        let intl_len = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.0.intl_desig = String::from_utf8(raw[cnt..cnt + intl_len as usize].to_vec()).unwrap();
        cnt += intl_len as usize;

        let piece_len = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.0.desig_piece =
            String::from_utf8(raw[cnt..cnt + piece_len as usize].to_vec()).unwrap();

        Ok(())
    }
}

impl<'b> From<&'b mut PyTLE> for &'b mut TLE {
    fn from(s: &mut PyTLE) -> &mut TLE {
        &mut s.0
    }
}
