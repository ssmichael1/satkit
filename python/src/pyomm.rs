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
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use anyhow::Result;
use satkit::omm::OMM;
use serde_json::{Map, Value};

/// Keys of xmltodict-style nested dicts whose children are hoisted to the
/// top level. `ndm`/`omm` are the root elements of a CCSDS XML document
/// (`xmltodict.parse` returns `{"omm": {...}}` or `{"ndm": {"omm": ...}}`),
/// `body`/`segment`/`data` are its structure; the rest are the field groups.
const GROUP_KEYS: &[&str] = &[
    "ndm",
    "omm",
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
/// `satkit.time`, `datetime.datetime` or `numpy.datetime64`.
fn py_to_json(key: &str, obj: &Bound<'_, PyAny>) -> Result<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract::<bool>()?));
    }
    if key == "EPOCH" && crate::pyinstant::is_time_scalar(obj) {
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

/// xmltodict renders an element that has attributes, such as
/// `<MEAN_MOTION units="rev/day">15.5</MEAN_MOTION>`, as
/// `{"@units": "rev/day", "#text": "15.5"}`: the `#text` of such a dict.
fn element_text<'py>(obj: &Bound<'py, PyAny>) -> Result<Option<Bound<'py, PyAny>>> {
    match obj.cast::<PyDict>() {
        Ok(d) => Ok(d.get_item("#text")?),
        Err(_) => Ok(None),
    }
}

/// Flattens a Python OMM dict into one JSON object with CCSDS keys.
///
/// Accepts the flat Space-Track/CelesTrak JSON shape, and any level of the
/// xmltodict rendering of the XML form (the whole document with its `omm`
/// or `ndm` root key, or the `omm`, `body`, `segment`, or `data` node):
/// group children are hoisted, `@version` becomes `CCSDS_OMM_VERS`, other
/// `@attr`/`#text` keys are dropped, an element with attributes is its
/// text, and `userDefinedParameters` are keyed by their `parameter`
/// attribute. A key that appears at more than one level keeps the last one
/// seen.
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
            } else if let Ok(list) = v.cast::<PyList>() {
                // xmltodict renders repeated elements (the `omm` messages of
                // an `ndm`) as a list. One element set is one message.
                match (list.len(), list.get_item(0)) {
                    (1, Ok(only)) => {
                        if let Ok(sub) = only.cast::<PyDict>() {
                            flatten(map, sub)?;
                        }
                    }
                    (n, _) => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid OMM dictionary: \"{key}\" holds {n} messages, but one \
                             dictionary is one element set; pass them one at a time, e.g. \
                             [{{\"omm\": m}} for m in d[\"ndm\"][\"omm\"]]"
                        ))
                        .into());
                    }
                }
            }
        } else if key == "userDefinedParameters" {
            merge_user_defined(map, &v)?;
        } else if let Some(text) = element_text(&v)? {
            // An element with attributes, e.g. `<MEAN_MOTION units="rev/day">`
            map.insert(key.clone(), py_to_json(&key, &text)?);
        } else {
            map.insert(key.clone(), py_to_json(&key, &v)?);
        }
    }
    Ok(())
}

/// Builds a Rust `OMM` from a Python dict.
///
/// Used by `satkit.sgp4` for dict inputs and by `TLE.from_omm`. A dict that
/// is not an OMM (a required key missing, a value that cannot be read) is a
/// `ValueError`.
pub fn omm_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<OMM> {
    let mut map = Map::new();
    flatten(&mut map, dict)?;
    serde_json::from_value::<OMM>(Value::Object(map)).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid OMM dictionary: {e}. Expected the CCSDS keys EPOCH, MEAN_MOTION, \
             ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER and MEAN_ANOMALY \
             (as in Space-Track/CelesTrak JSON), or the nested meanElements/tleParameters \
             groups of an XML-derived dict"
        ))
    })
}

/// The Python exception for an OMM loader error: `ValueError` for content
/// that is not an OMM document, the matching `OSError` for an I/O failure,
/// and `RuntimeError` for what fails at run time (a download, offline mode).
fn omm_err(e: satkit::omm::Error) -> PyErr {
    use satkit::omm::Error as E;
    match e {
        E::Io(e) => e.into(),
        E::Http(_) | E::HttpThrottled(_) | E::Offline { .. } | E::XmlFeatureDisabled => {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        }
        e => pyo3::exceptions::PyValueError::new_err(e.to_string()),
    }
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
/// Fetches the content at the given URL and auto-detects the format (JSON,
/// XML or CSV), as ``omm_from_text`` does. A CelesTrak GP query with no
/// ``FORMAT`` answers in CSV.
/// Returns a list of dictionaries that can be passed directly to ``satkit.sgp4()``.
///
/// Args:
///     url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)
///
/// Returns:
///     list[dict]: one dict per message, with every CCSDS field the source
///     provided (the keys of ``satkit.OMMDict``) plus the source's extra keys
///
/// Raises:
///     RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
///         ``satkit.utils.set_offline(True)``; no connection is opened) or
///         the request fails
///     ValueError: if the response is not JSON, XML or CSV, or is malformed
///
/// Example:
///     ```python
///     omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
///     pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
///     ```
#[pyfunction]
pub fn omm_from_url(py: Python<'_>, url: String) -> PyResult<Py<PyAny>> {
    let omms = py.detach(|| OMM::from_url(&url)).map_err(omm_err)?;
    omms_to_pylist(py, &omms)
}

/// Load OMM(s) from a JSON, XML or CSV file as a list of dictionaries
///
/// The format is detected from the content, not the file extension, as
/// ``omm_from_text`` does.
///
/// Args:
///     filename (str | os.PathLike): path to a file holding one OMM or an
///         array of them (JSON), a CCSDS NDM/XML document, or a CSV table
///
/// Returns:
///     list[dict]: one dict per message, as ``omm_from_text`` returns
///
/// Raises:
///     FileNotFoundError: if the file does not exist; the exception names
///         the path. Another I/O failure raises the matching ``OSError``
///     ValueError: if the content is none of the three formats or is malformed
#[pyfunction]
pub fn omm_from_file(py: Python<'_>, filename: std::path::PathBuf) -> PyResult<Py<PyAny>> {
    let omms = py.detach(|| {
        // Read here, not with `OMM::from_file`, to name the path in the error
        let text =
            std::fs::read_to_string(&filename).map_err(|e| crate::pytle::io_err(e, &filename))?;
        OMM::from_text(&text).map_err(omm_err)
    })?;
    omms_to_pylist(py, &omms)
}

/// Parse OMM(s) from JSON, XML or CSV text as a list of dictionaries
///
/// Text starting with ``[`` or ``{`` is parsed as JSON, text starting with
/// ``<`` as XML, and text whose first line holds a comma as CSV: a header
/// row of OMM keywords and one row per message, as CelesTrak and Space-Track
/// serve it. KVN is not supported.
///
/// Args:
///     text (str): One OMM or an array of them (JSON), a CCSDS NDM/XML
///         document, or a CSV table
///
/// Returns:
///     list[dict]: one dict per message. The CCSDS fields have the keys and
///     value types of ``satkit.OMMDict``, with ``EPOCH`` an RFC 3339 string;
///     keys that are not CCSDS fields (Space-Track's ``OBJECT_TYPE``,
///     ``TLE_LINE1``, ...; XML ``USER_DEFINED`` parameters; extra CSV
///     columns) are kept as the source gave them.
///
/// Raises:
///     ValueError: if the text is none of the three formats or is
///         malformed; for CSV the message gives the line of the bad row, and
///         a table with no ``EPOCH`` or ``MEAN_MOTION`` column is refused
#[pyfunction]
pub fn omm_from_text(py: Python<'_>, text: String) -> PyResult<Py<PyAny>> {
    let omms = py.detach(|| OMM::from_text(&text)).map_err(omm_err)?;
    omms_to_pylist(py, &omms)
}
