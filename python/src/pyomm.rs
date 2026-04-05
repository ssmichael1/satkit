use pyo3::prelude::*;
use pyo3::types::PyDict;

use anyhow::Result;

/// Load OMM(s) from a URL as a list of dictionaries
///
/// Fetches the content at the given URL and auto-detects JSON vs XML format.
/// Returns a list of dictionaries that can be passed directly to ``satkit.sgp4()``.
///
/// Args:
///     url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)
///
/// Returns:
///     list[dict]: List of OMM dictionaries
///
/// Example:
///     ```python
///     omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
///     pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
///     ```
#[pyfunction]
pub fn omm_from_url(url: String) -> Result<Py<PyAny>> {
    let omms = satkit::omm::OMM::from_url(&url)?;

    pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
        let list: Vec<Bound<'_, PyDict>> = omms
            .iter()
            .map(|omm| omm_to_pydict(py, omm))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(list.into_pyobject(py)?.into())
    })
    .map_err(|e| e.into())
}

fn omm_to_pydict<'py>(py: Python<'py>, omm: &satkit::omm::OMM) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);

    d.set_item("OBJECT_NAME", &omm.object_name)?;
    d.set_item("OBJECT_ID", &omm.object_id)?;
    d.set_item("EPOCH", &omm.epoch)?;
    d.set_item("MEAN_MOTION", omm.mean_motion)?;
    d.set_item("ECCENTRICITY", omm.eccentricity)?;
    d.set_item("INCLINATION", omm.inclination)?;
    d.set_item("RA_OF_ASC_NODE", omm.raan)?;
    d.set_item("ARG_OF_PERICENTER", omm.arg_of_pericenter)?;
    d.set_item("MEAN_ANOMALY", omm.mean_anomaly)?;

    if let Some(v) = omm.bstar {
        d.set_item("BSTAR", v)?;
    }
    if let Some(v) = omm.mean_motion_dot {
        d.set_item("MEAN_MOTION_DOT", v)?;
    }
    if let Some(v) = omm.mean_motion_ddot {
        d.set_item("MEAN_MOTION_DDOT", v)?;
    }
    if let Some(v) = omm.norad_cat_id {
        d.set_item("NORAD_CAT_ID", v)?;
    }
    if let Some(v) = omm.element_set_no {
        d.set_item("ELEMENT_SET_NO", v)?;
    }
    if let Some(v) = omm.rev_at_epoch {
        d.set_item("REV_AT_EPOCH", v)?;
    }
    if let Some(v) = omm.ephemeris_type {
        d.set_item("EPHEMERIS_TYPE", v)?;
    }
    if let Some(ref v) = omm.classification_type {
        d.set_item("CLASSIFICATION_TYPE", v)?;
    }

    Ok(d)
}
