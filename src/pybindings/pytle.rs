use pyo3::prelude::*;

use crate::tle::TLE;
use std::fs::File;
use std::io::{self, BufRead};

#[pyclass(name = "TLE")]
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
