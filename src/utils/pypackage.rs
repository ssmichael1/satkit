use pyo3::prelude::*;
use pyo3::types::PyModule;
use anyhow::Result;
use std::path::PathBuf;

/// Used with python bindings, returns directory path of the satkit_data package
pub fn get_datadir_package_path() -> Result<Option<PathBuf>> {
    Python::with_gil(|py| {
        let importlib = PyModule::import(py, "importlib.util")?;
        let spec = importlib.call_method1("find_spec", ("satkit_data",))?;
        if spec.is_none() {
            return Ok(None);
        }
        let origin = spec.getattr("origin")?.extract::<String>()?;
        let path = std::path::Path::new(&origin)
            .parent()
            .map(|p| p.to_path_buf());
        Ok(path)
    })
}