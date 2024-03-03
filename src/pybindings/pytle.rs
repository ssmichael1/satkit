use pyo3::prelude::*;

use crate::tle::TLE;

#[pyclass(name = "TLE")]
pub struct PyTLE {
    pub inner: TLE,
}

#[pymethods]
impl PyTLE {
    /// Return list of TLEs from input lines
    /// lines is a list of strings
    #[staticmethod]
    fn from_lines(lines: Vec<String>) -> PyResult<Vec<TLE>> {
        match TLE::from_lines(&lines) {
            Ok(v) => Ok(v),
            Err(e) => {
                let serr = format!("Error loading TLEs: {}", e.to_string());
                Err(pyo3::exceptions::PyImportError::new_err(serr))
            }
        }
    }

    /// Return a single TLE from input lines
    /// lines is list of strings
    ///
    /// If additional lines represent more than a single TLE,
    /// they will be ignored
    #[staticmethod]
    fn single_from_lines(lines: Vec<String>) -> PyResult<Self> {
        if lines.len() == 3 {
            match TLE::load_3line(&lines[0], &lines[1], &lines[2]) {
                Ok(v) => return Ok(PyTLE { inner: v }),
                Err(e) => {
                    let serr = format!("Error loading TLE: {}", e);
                    return Err(pyo3::exceptions::PyImportError::new_err(serr));
                }
            }
        } else if lines.len() == 2 {
            match TLE::load_2line(&lines[0], &lines[1]) {
                Ok(v) => return Ok(PyTLE { inner: v }),
                Err(e) => {
                    let serr = format!("Error loading TLE: {}", e);
                    return Err(pyo3::exceptions::PyImportError::new_err(serr));
                }
            }
        } else {
            Err(pyo3::exceptions::PyImportError::new_err(
                "Invalid number of lines",
            ))
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
