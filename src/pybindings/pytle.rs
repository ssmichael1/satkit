use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::tle::TLE;
use std::fs::File;
use std::io::{self, BufRead};

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
    fn from_file(filename: String) -> PyResult<PyObject> {
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
    fn from_lines(lines: Vec<String>) -> PyResult<PyObject> {
        match TLE::from_lines(&lines) {
            Ok(v) => pyo3::Python::with_gil(|py| -> PyResult<PyObject> {
                if v.len() > 1 {
                    v.into_py_any(py)
                } else {
                    v[0].clone().into_py_any(py)
                }
            }),
            Err(e) => {
                let serr = format!("Error loading TLEs: {}", e);
                Err(pyo3::exceptions::PyImportError::new_err(serr))
            }
        }
    }

    /// Satellite NORAD Catalog Number
    #[getter]
    const fn get_satnum(&self) -> PyResult<i32> {
        Ok(self.0.sat_num)
    }

    /// Orbit eccentricity
    #[getter]
    const fn get_eccen(&self) -> PyResult<f64> {
        Ok(self.0.eccen)
    }

    /// Mean anomaly in degrees
    #[getter]
    const fn get_mean_anomaly(&self) -> PyResult<f64> {
        Ok(self.0.mean_anomaly)
    }

    /// Mean motion in revs / day
    #[getter]
    const fn get_mean_motion(&self) -> PyResult<f64> {
        Ok(self.0.mean_motion)
    }

    /// inclination in degrees
    #[getter]
    const fn get_inclination(&self) -> PyResult<f64> {
        Ok(self.0.inclination)
    }

    /// Epoch time of TLE
    #[getter]
    fn get_epoch(&self, py: Python) -> PyResult<PyObject> {
        self.0.epoch.into_py_any(py)
    }

    /// argument of perigee, degrees
    #[getter]
    const fn get_arg_of_perigee(&self) -> PyResult<f64> {
        Ok(self.0.arg_of_perigee)
    }

    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    #[getter]
    const fn get_mean_motion_dot(&self) -> PyResult<f64> {
        Ok(self.0.mean_motion_dot)
    }

    /// One sixth of 2nd derivative of mean motion wrt time, in revs/day^3
    #[getter]
    const fn get_mean_motion_dot_dot(&self) -> PyResult<f64> {
        Ok(self.0.mean_motion_dot_dot)
    }

    /// Name of satellite
    fn name(&self) -> PyResult<String> {
        Ok(self.0.name.clone())
    }

    // Drag
    const fn bstar(&self) -> PyResult<f64> {
        Ok(self.0.bstar)
    }

    fn __str__(&self) -> String {
        self.0.to_pretty_string()
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
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

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
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
