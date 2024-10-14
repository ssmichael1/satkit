use pyo3::prelude::*;

use crate::filters::ukf::UKF;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyUntypedArrayMethods;

use pyo3::exceptions::PyValueError;

use crate::pybindings::pyutils::*;
use crate::types::*;

enum UKFType {
    None,
    UKF1(UKF<1>),
    UKF2(UKF<2>),
    UKF3(UKF<3>),
    UKF4(UKF<4>),
    UKF5(UKF<5>),
    UKF6(UKF<6>),
    UKF7(UKF<7>),
    UKF8(UKF<8>),
    UKF9(UKF<9>),
    UKF10(UKF<10>),
}

#[pyclass(name = "ukf", module = "satkit")]
pub struct PyUKF {
    ukf: UKFType,
}

fn pfunc<const N: usize>(x: Matrix<N, 1>, f: &PyObject) -> PyResult<Matrix<N, 1>> {
    pyo3::Python::with_gil(|py| {
        let x = smatrix_to_py::<N, 1>(&x)?;
        let x = f.call1(py, (x,))?;
        let x = x.extract::<PyReadonlyArray1<f64>>(py)?;
        let x = py_to_smatrix::<N, 1>(&x)?;
        Ok(x)
    })
}

#[pymethods]
impl PyUKF {
    #[new]
    fn new_default() -> PyUKF {
        PyUKF { ukf: UKFType::None }
    }

    fn predict(&mut self, f: PyObject) -> PyResult<()> {
        match self.ukf {
            UKFType::UKF1(ref mut ukf) => {
                ukf.predict(|x| pfunc(x, &f).unwrap());
            }
            _ => {}
        }
        Ok(())
    }

    #[getter]
    fn get_cov(&self) -> PyResult<PyObject> {
        match self.ukf {
            UKFType::UKF1(ref ukf) => Ok(smatrix_to_py::<1, 1>(&ukf.p)?),
            UKFType::UKF2(ref ukf) => Ok(smatrix_to_py::<2, 2>(&ukf.p)?),
            UKFType::UKF3(ref ukf) => Ok(smatrix_to_py::<3, 3>(&ukf.p)?),
            UKFType::UKF4(ref ukf) => Ok(smatrix_to_py::<4, 4>(&ukf.p)?),
            UKFType::UKF5(ref ukf) => Ok(smatrix_to_py::<5, 5>(&ukf.p)?),
            UKFType::UKF6(ref ukf) => Ok(smatrix_to_py::<6, 6>(&ukf.p)?),
            UKFType::UKF7(ref ukf) => Ok(smatrix_to_py::<7, 7>(&ukf.p)?),
            UKFType::UKF8(ref ukf) => Ok(smatrix_to_py::<8, 8>(&ukf.p)?),
            UKFType::UKF9(ref ukf) => Ok(smatrix_to_py::<9, 9>(&ukf.p)?),
            UKFType::UKF10(ref ukf) => Ok(smatrix_to_py::<10, 10>(&ukf.p)?),
            _ => Err(PyValueError::new_err(
                "Covariance matrix must be less than 10x10 elements",
            )),
        }
    }

    #[getter]
    fn get_state(&self) -> PyResult<PyObject> {
        match self.ukf {
            UKFType::UKF1(ref ukf) => Ok(smatrix_to_py::<1, 1>(&ukf.x)?),
            UKFType::UKF2(ref ukf) => Ok(smatrix_to_py::<2, 1>(&ukf.x)?),
            UKFType::UKF3(ref ukf) => Ok(smatrix_to_py::<3, 1>(&ukf.x)?),
            UKFType::UKF4(ref ukf) => Ok(smatrix_to_py::<4, 1>(&ukf.x)?),
            UKFType::UKF5(ref ukf) => Ok(smatrix_to_py::<5, 1>(&ukf.x)?),
            UKFType::UKF6(ref ukf) => Ok(smatrix_to_py::<6, 1>(&ukf.x)?),
            UKFType::UKF7(ref ukf) => Ok(smatrix_to_py::<7, 1>(&ukf.x)?),
            UKFType::UKF8(ref ukf) => Ok(smatrix_to_py::<8, 1>(&ukf.x)?),
            UKFType::UKF9(ref ukf) => Ok(smatrix_to_py::<9, 1>(&ukf.x)?),
            UKFType::UKF10(ref ukf) => Ok(smatrix_to_py::<10, 1>(&ukf.x)?),
            _ => Err(PyValueError::new_err(
                "State vector must be less than 10 elements",
            )),
        }
    }

    #[setter(state)]
    fn set_state(&mut self, val: PyReadonlyArray1<f64>) -> PyResult<()> {
        let rval = val.len();
        if rval > 10 {
            return Err(PyValueError::new_err(
                "State vector must be less than 10 elements",
            ));
        }

        match self.ukf {
            UKFType::None => {
                self.ukf = match rval {
                    1 => UKFType::UKF1(UKF::new_default()),
                    2 => UKFType::UKF2(UKF::new_default()),
                    3 => UKFType::UKF3(UKF::new_default()),
                    4 => UKFType::UKF4(UKF::new_default()),
                    5 => UKFType::UKF5(UKF::new_default()),
                    6 => UKFType::UKF6(UKF::new_default()),
                    7 => UKFType::UKF7(UKF::new_default()),
                    8 => UKFType::UKF8(UKF::new_default()),
                    9 => UKFType::UKF9(UKF::new_default()),
                    10 => UKFType::UKF10(UKF::new_default()),
                    _ => UKFType::None,
                };
            }
            _ => {}
        }

        match self.ukf {
            UKFType::None => {
                return Err(PyValueError::new_err(
                    "State vector must be less than 10 elements",
                ));
            }
            UKFType::UKF1(ref mut ukf) => {
                ukf.x = py_to_smatrix::<1, 1>(&val)?;
            }
            UKFType::UKF2(ref mut ukf) => {
                ukf.x = py_to_smatrix::<2, 1>(&val)?;
            }
            UKFType::UKF3(ref mut ukf) => {
                ukf.x = py_to_smatrix::<3, 1>(&val)?;
            }
            UKFType::UKF4(ref mut ukf) => {
                ukf.x = py_to_smatrix::<4, 1>(&val)?;
            }
            UKFType::UKF5(ref mut ukf) => {
                ukf.x = py_to_smatrix::<5, 1>(&val)?;
            }
            UKFType::UKF6(ref mut ukf) => {
                ukf.x = py_to_smatrix::<6, 1>(&val)?;
            }
            UKFType::UKF7(ref mut ukf) => {
                ukf.x = py_to_smatrix::<7, 1>(&val)?;
            }
            UKFType::UKF8(ref mut ukf) => {
                ukf.x = py_to_smatrix::<8, 1>(&val)?;
            }
            UKFType::UKF9(ref mut ukf) => {
                ukf.x = py_to_smatrix::<9, 1>(&val)?;
            }
            UKFType::UKF10(ref mut ukf) => {
                ukf.x = py_to_smatrix::<10, 1>(&val)?;
            }
            _ => {}
        }

        Ok(())
    }
}
