use pyo3::prelude::*;

use crate::tle::TLE;
use std::fs::File;
use std::io::{self, BufRead};

#[pyclass(name = "TLE", module = "satkit")]
pub struct PyTLE {
    pub inner: TLE,
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
        let file = File::open(&std::path::PathBuf::from(filename))?;

        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .into_iter()
            .map(|v| -> String { v.unwrap() })
            .collect();

        PyTLE::from_lines(lines)
    }

    #[new]
    fn new() -> PyTLE {
        PyTLE { inner: TLE::new() }
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
                    Ok(v.into_py(py))
                } else {
                    Ok(v[0].clone().into_py(py))
                }
            }),
            Err(e) => {
                let serr = format!("Error loading TLEs: {}", e.to_string());
                Err(pyo3::exceptions::PyImportError::new_err(serr))
            }
        }
    }

    /// Satellite NORAD Catalog Number
    #[getter]
    fn get_satnum(&self) -> PyResult<i32> {
        Ok(self.inner.sat_num)
    }

    /// Orbit eccentricity
    #[getter]
    fn get_eccen(&self) -> PyResult<f64> {
        Ok(self.inner.eccen)
    }

    /// Mean anomaly in degrees
    #[getter]
    fn get_mean_anomaly(&self) -> PyResult<f64> {
        Ok(self.inner.mean_anomaly)
    }

    /// Mean motion in revs / day
    #[getter]
    fn get_mean_motion(&self) -> PyResult<f64> {
        Ok(self.inner.mean_motion)
    }

    /// inclination in degrees
    #[getter]
    fn get_inclination(&self) -> PyResult<f64> {
        Ok(self.inner.inclination)
    }

    /// Epoch time of TLE
    #[getter]
    fn get_epoch(&self) -> PyResult<crate::astrotime::AstroTime> {
        Ok(self.inner.epoch)
    }

    /// argument of perigee, degrees
    #[getter]
    fn get_arg_of_perigee(&self) -> PyResult<f64> {
        Ok(self.inner.arg_of_perigee)
    }

    /// One half of 1st derivative of mean motion wrt time, in revs/day^2
    #[getter]
    fn get_mean_motion_dot(&self) -> PyResult<f64> {
        Ok(self.inner.mean_motion_dot)
    }

    /// One sixth of 2nd derivative of mean motion wrt time, in revs/day^3
    #[getter]
    fn get_mean_motion_dot_dot(&self) -> PyResult<f64> {
        Ok(self.inner.mean_motion_dot_dot)
    }

    /// Name of satellite
    fn name(&self) -> PyResult<String> {
        Ok(self.inner.name.clone())
    }

    // Drag
    fn bstar(&self) -> PyResult<f64> {
        Ok(self.inner.bstar)
    }

    fn __str__(&self) -> String {
        self.inner.to_pretty_string()
    }

    fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        let nbytes: usize = 102
            + self.inner.name.len()
            + self.inner.intl_desig.len()
            + self.inner.desig_piece.len();
        let mut raw = vec![0u8; nbytes];
        raw[0..4].clone_from_slice(&self.inner.sat_num.to_le_bytes());
        raw[4..8].clone_from_slice(&self.inner.desig_year.to_le_bytes());
        raw[8..12].clone_from_slice(&self.inner.desig_launch.to_le_bytes());
        raw[12..20].clone_from_slice(&self.inner.mean_motion_dot.to_le_bytes());
        raw[20..28].clone_from_slice(&self.inner.mean_motion_dot_dot.to_le_bytes());
        raw[28..36].clone_from_slice(&self.inner.bstar.to_le_bytes());
        raw[36..44].clone_from_slice(&self.inner.inclination.to_le_bytes());
        raw[44..52].clone_from_slice(&self.inner.raan.to_le_bytes());
        raw[52..60].clone_from_slice(&self.inner.eccen.to_le_bytes());
        raw[60..68].clone_from_slice(&self.inner.arg_of_perigee.to_le_bytes());
        raw[68..76].clone_from_slice(&self.inner.mean_anomaly.to_le_bytes());
        raw[76..84].clone_from_slice(&self.inner.mean_motion.to_le_bytes());
        raw[84..92].clone_from_slice(&self.inner.epoch.to_mjd(crate::TimeScale::TAI).to_le_bytes());
        raw[92..96].clone_from_slice(&self.inner.rev_num.to_le_bytes());

        let mut cnt = 96;

        let namelen = self.inner.name.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&namelen.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.inner.name.len()].clone_from_slice(&self.inner.name.as_bytes());
        cnt += self.inner.name.len();

        let intl_len = self.inner.intl_desig.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&intl_len.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.inner.intl_desig.len()]
            .clone_from_slice(&self.inner.intl_desig.as_bytes());
        cnt += self.inner.intl_desig.len();

        let piece_len = self.inner.desig_piece.len() as u16;
        raw[cnt..cnt + 2].clone_from_slice(&piece_len.to_le_bytes());
        cnt += 2;
        raw[cnt..cnt + self.inner.desig_piece.len()]
            .clone_from_slice(&self.inner.desig_piece.as_bytes());

        Ok(pyo3::types::PyBytes::new_bound(py, &raw).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let raw = state.extract::<Vec<u8>>(py)?;

        self.inner.sat_num = i32::from_le_bytes(raw[0..4].try_into().unwrap());
        self.inner.desig_year = i32::from_le_bytes(raw[4..8].try_into().unwrap());
        self.inner.desig_launch = i32::from_le_bytes(raw[8..12].try_into().unwrap());
        self.inner.mean_motion_dot = f64::from_le_bytes(raw[12..20].try_into().unwrap());
        self.inner.mean_motion_dot_dot = f64::from_le_bytes(raw[20..28].try_into().unwrap());
        self.inner.bstar = f64::from_le_bytes(raw[28..36].try_into().unwrap());
        self.inner.inclination = f64::from_le_bytes(raw[36..44].try_into().unwrap());
        self.inner.raan = f64::from_le_bytes(raw[44..52].try_into().unwrap());
        self.inner.eccen = f64::from_le_bytes(raw[52..60].try_into().unwrap());
        self.inner.arg_of_perigee = f64::from_le_bytes(raw[60..68].try_into().unwrap());
        self.inner.mean_anomaly = f64::from_le_bytes(raw[68..76].try_into().unwrap());
        self.inner.mean_motion = f64::from_le_bytes(raw[76..84].try_into().unwrap());
        self.inner.epoch = crate::AstroTime::from_mjd(
            f64::from_le_bytes(raw[84..92].try_into().unwrap()),
            crate::TimeScale::TAI,
        );
        self.inner.rev_num = i32::from_le_bytes(raw[92..96].try_into().unwrap());

        let mut cnt = 96;

        let namelen = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.inner.name = String::from_utf8(raw[cnt..cnt + namelen as usize].to_vec()).unwrap();
        cnt += namelen as usize;

        let intl_len = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.inner.intl_desig =
            String::from_utf8(raw[cnt..cnt + intl_len as usize].to_vec()).unwrap();
        cnt += intl_len as usize;

        let piece_len = u16::from_le_bytes(raw[cnt..cnt + 2].try_into().unwrap());
        cnt += 2;
        self.inner.desig_piece =
            String::from_utf8(raw[cnt..cnt + piece_len as usize].to_vec()).unwrap();

        Ok(())
    }
}

impl IntoPy<PyObject> for TLE {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let tle: PyTLE = PyTLE { inner: self };
        tle.into_py(py)
    }
}

impl<'b> From<&'b mut PyTLE> for &'b mut TLE {
    fn from<'a>(s: &'a mut PyTLE) -> &'a mut TLE {
        &mut s.inner
    }
}
