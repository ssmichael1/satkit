//! Python view of CCSDS Orbital Mean-Element Messages.
//!
//! Python has no OMM class: an OMM is a plain `dict` keyed by the CCSDS
//! field names, which is what `json.load` on a Space-Track or CelesTrak
//! response produces. The loaders here return such dicts, `satkit.sgp4`
//! accepts them, and `TLE.from_omm` / `TLE.to_omm` convert them.
//!
//! Every dict goes through the same serde parser as the Rust `OMM` type, so
//! the tolerance rules (quoted numbers, `null`/empty optionals, extra keys
//! kept) and the validation rules are identical in both languages.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDateTime, PyDict, PyFloat, PyInt, PyList, PyString};

use anyhow::{anyhow, Result};
use satkit::omm::OMM;
use serde_json::{Map, Value};

/// Keys of xmltodict-style nested dicts whose children are hoisted to the
/// top level. `body`/`segment`/`data` are the CCSDS XML structure; the rest
/// are the field groups.
const GROUP_KEYS: &[&str] = &[
    "body",
    "segment",
    "header",
    "metadata",
    "data",
    "meanElements",
    "spacecraftParameters",
    "tleParameters",
];

/// Python scalar/container → JSON value. `EPOCH` additionally accepts a
/// `satkit.time` or `datetime.datetime`.
fn py_to_json(key: &str, obj: &Bound<'_, PyAny>) -> Result<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract::<bool>()?));
    }
    if key == "EPOCH"
        && (obj.is_instance_of::<crate::pyinstant::PyInstant>()
            || obj.is_instance_of::<PyDateTime>())
    {
        return Ok(Value::String(
            crate::pysgp4::epoch_from_val(obj)?.as_rfc3339(),
        ));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Value::String(obj.extract::<String>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Value::from(obj.extract::<f64>()?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let items = list
            .iter()
            .map(|v| py_to_json(key, &v))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Value::Array(items));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict.iter() {
            let k: String = k.extract()?;
            let v = py_to_json(&k, &v)?;
            map.insert(k, v);
        }
        return Ok(Value::Object(map));
    }
    // numpy scalars and anything else with a numeric protocol
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::from(f));
    }
    Ok(Value::String(obj.str()?.to_string()))
}

/// xmltodict renders `<USER_DEFINED parameter="X">v</USER_DEFINED>` as
/// `{"@parameter": "X", "#text": "v"}` (a list of them, or a single dict).
fn merge_user_defined(map: &mut Map<String, Value>, obj: &Bound<'_, PyAny>) -> Result<()> {
    let Ok(group) = obj.cast::<PyDict>() else {
        return Ok(());
    };
    let Some(entries) = group.get_item("USER_DEFINED")? else {
        return Ok(());
    };
    let entries: Vec<Bound<'_, PyAny>> = match entries.cast::<PyList>() {
        Ok(list) => list.iter().collect(),
        Err(_) => vec![entries],
    };
    for entry in entries {
        let Ok(entry) = entry.cast::<PyDict>() else {
            continue;
        };
        let Some(name) = entry.get_item("@parameter")? else {
            continue;
        };
        let name: String = name.extract()?;
        let value = match entry.get_item("#text")? {
            Some(v) => py_to_json(&name, &v)?,
            None => Value::Null,
        };
        map.insert(name, value);
    }
    Ok(())
}

/// Flattens a Python OMM dict into one JSON object with CCSDS keys.
///
/// Accepts the flat Space-Track/CelesTrak JSON shape, and any level of the
/// xmltodict rendering of the XML form (`omm`, `body`, `segment`, or `data`
/// node): group children are hoisted, `@version` becomes `CCSDS_OMM_VERS`,
/// other `@attr`/`#text` keys are dropped, and `userDefinedParameters` are
/// keyed by their `parameter` attribute. A key that appears at more than one
/// level keeps the last one seen.
fn flatten(map: &mut Map<String, Value>, dict: &Bound<'_, PyDict>) -> Result<()> {
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        if key == "@version" {
            map.insert("CCSDS_OMM_VERS".to_string(), py_to_json(&key, &v)?);
        } else if key.starts_with('@') || key.starts_with('#') || key == "covarianceMatrix" {
            continue;
        } else if GROUP_KEYS.contains(&key.as_str()) {
            if let Ok(sub) = v.cast::<PyDict>() {
                flatten(map, sub)?;
            }
        } else if key == "userDefinedParameters" {
            merge_user_defined(map, &v)?;
        } else {
            map.insert(key.clone(), py_to_json(&key, &v)?);
        }
    }
    Ok(())
}

/// Builds a Rust `OMM` from a Python dict.
///
/// Used by `satkit.sgp4` for dict inputs and by `TLE.from_omm`.
pub fn omm_from_pydict(dict: &Bound<'_, PyDict>) -> Result<OMM> {
    let mut map = Map::new();
    flatten(&mut map, dict)?;
    serde_json::from_value::<OMM>(Value::Object(map)).map_err(|e| {
        anyhow!(
            "Invalid OMM dictionary: {e}. Expected the CCSDS keys EPOCH, MEAN_MOTION, \
             ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER and MEAN_ANOMALY \
             (as in Space-Track/CelesTrak JSON), or the nested meanElements/tleParameters \
             groups of an XML-derived dict"
        )
    })
}

fn json_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_bound_py_any(py)?,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_bound_py_any(py)?
            } else if let Some(u) = n.as_u64() {
                u.into_bound_py_any(py)?
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_bound_py_any(py)?
            }
        }
        Value::String(s) => s.into_bound_py_any(py)?,
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, v) in map {
                d.set_item(k, json_to_py(py, v)?)?;
            }
            d.into_any()
        }
    })
}

/// Renders a Rust `OMM` as a flat Python dict with every populated CCSDS
/// field plus the extra fields, `EPOCH` as an RFC 3339 string.
pub fn omm_to_pydict<'py>(py: Python<'py>, omm: &OMM) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(omm)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let obj = json_to_py(py, &value)?;
    Ok(obj.cast_into::<PyDict>()?)
}

fn omms_to_pylist(py: Python<'_>, omms: &[OMM]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for omm in omms {
        list.append(omm_to_pydict(py, omm)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Load OMM(s) from a URL as a list of dictionaries
///
/// Fetches the content at the given URL and auto-detects JSON vs XML format.
/// Returns a list of dictionaries that can be passed directly to ``satkit.sgp4()``.
///
/// Args:
///     url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)
///
/// Returns:
///     list[dict]: One dict per message, see ``satkit.OMMDict``
///
/// Example:
///     ```python
///     omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
///     pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
///     ```
#[pyfunction]
pub fn omm_from_url(py: Python<'_>, url: String) -> Result<Py<PyAny>> {
    let omms = py.detach(|| OMM::from_url(&url))?;
    Ok(omms_to_pylist(py, &omms)?)
}

/// Load OMM(s) from a JSON or XML file as a list of dictionaries
///
/// The format is detected from the content, not the file extension.
///
/// Args:
///     filename (str): Path to a file holding one OMM or an array of them
///         (JSON), or a CCSDS NDM/XML document
///
/// Returns:
///     list[dict]: One dict per message, see ``satkit.OMMDict``
#[pyfunction]
pub fn omm_from_file(py: Python<'_>, filename: String) -> Result<Py<PyAny>> {
    let omms = py.detach(|| OMM::from_file(&filename))?;
    Ok(omms_to_pylist(py, &omms)?)
}

/// Parse OMM(s) from JSON or XML text as a list of dictionaries
///
/// Text starting with ``[`` or ``{`` is parsed as JSON, text starting with
/// ``<`` as XML.
///
/// Args:
///     text (str): One OMM or an array of them (JSON), or a CCSDS NDM/XML document
///
/// Returns:
///     list[dict]: One dict per message, see ``satkit.OMMDict``
#[pyfunction]
pub fn omm_from_text(py: Python<'_>, text: String) -> Result<Py<PyAny>> {
    let omms = py.detach(|| OMM::from_text(&text))?;
    Ok(omms_to_pylist(py, &omms)?)
}
