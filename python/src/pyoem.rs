//! Python view of CCSDS Orbit Ephemeris Messages.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use satkit::oem::{Error, Interpolation, Segment, OEM};
use satkit::Instant;

use crate::pyframes::PyFrame;
use crate::pyinstant::{PyInstant, PyTimeScale, ToTimeVec};
use crate::pyutils::{slice2py1d, slice2py2d};

/// `ValueError` for content that is not an OEM or a time that cannot be
/// interpolated, the matching `OSError` for an I/O failure.
fn oem_err(e: Error) -> PyErr {
    match e {
        Error::Io(e) => e.into(),
        Error::XmlFeatureDisabled => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        e => PyValueError::new_err(e.to_string()),
    }
}

fn parse_method(method: Option<&str>) -> PyResult<Option<Interpolation>> {
    match method.map(|m| m.trim().to_ascii_lowercase()).as_deref() {
        None => Ok(None),
        Some("lagrange") => Ok(Some(Interpolation::Lagrange)),
        Some("hermite") => Ok(Some(Interpolation::Hermite)),
        Some(_) => Err(PyValueError::new_err(format!(
            "method must be \"lagrange\" or \"hermite\", not {:?}",
            method.unwrap_or_default()
        ))),
    }
}

/// State of `seg` at `time` with the file's interpolation, or with the
/// method and/or degree given in its place.
fn interp_segment(
    seg: &Segment,
    time: &Instant,
    method: Option<Interpolation>,
    degree: Option<u32>,
) -> Result<[f64; 6], Error> {
    let (pos, vel) = if method.is_none() && degree.is_none() {
        seg.interp(time)?
    } else {
        // Whatever is not given comes from the file. A method the file
        // names but satkit lacks only matters if it is the one asked for
        let (m, d) = match (method, degree) {
            (Some(m), Some(d)) => (m, d),
            (Some(m), None) => (
                m,
                seg.metadata
                    .interpolation_degree
                    .unwrap_or(satkit::oem::DEFAULT_INTERPOLATION_DEGREE),
            ),
            (None, d) => {
                let (m, file_degree) = seg.interpolation()?;
                (m, d.unwrap_or(file_degree))
            }
        };
        seg.interp_with(time, m, d)?
    };
    Ok([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
}

/// Shared body of `OEM.interp` and `OEMSegment.interp`.
fn interp_py(
    py: Python,
    time: &Bound<'_, PyAny>,
    method: Option<&str>,
    degree: Option<u32>,
    one: impl Fn(&Instant, Option<Interpolation>, Option<u32>) -> Result<[f64; 6], Error>,
) -> PyResult<Py<PyAny>> {
    let method = parse_method(method)?;
    let input = time.to_time_input()?;
    if input.scalar {
        let state = one(&input.times[0], method, degree).map_err(oem_err)?;
        return slice2py1d(py, &state);
    }
    let mut flat = Vec::with_capacity(input.times.len() * 6);
    for t in &input.times {
        flat.extend(one(t, method, degree).map_err(oem_err)?);
    }
    slice2py2d(py, &flat, input.times.len(), 6)
}

fn opt_time(t: Option<Instant>) -> Option<PyInstant> {
    t.map(PyInstant)
}

/// CCSDS Orbit Ephemeris Message (OEM): a tabulated trajectory
///
/// An OEM is a table of position and velocity vectors of one object at a
/// sequence of times, with the reference frame, central body and time
/// system they are expressed in (CCSDS 502.0-B). It is the usual format
/// for exchanging ephemerides between operators and agencies.
///
/// Load one with ``OEM.from_file`` or ``OEM.from_text``; both read the KVN
/// (text) and XML encodings. The message is a list of ``segments``, each
/// with its own metadata and states; ``interp`` interpolates the state at
/// any time a segment covers.
///
/// The file gives kilometers and km/s; satkit returns **meters and m/s**,
/// as everywhere else. States stay in the frame the file names
/// (``OEMSegment.ref_frame``); nothing is rotated.
///
/// Example:
///     ```python
///     oem = satkit.OEM.from_text(
///         "CCSDS_OEM_VERS = 2.0\n"
///         "CREATION_DATE = 2024-01-01T00:00:00\n"
///         "ORIGINATOR = EXAMPLE\n"
///         "META_START\n"
///         "OBJECT_NAME = DEMOSAT\n"
///         "OBJECT_ID = 2024-001A\n"
///         "CENTER_NAME = EARTH\n"
///         "REF_FRAME = EME2000\n"
///         "TIME_SYSTEM = UTC\n"
///         "START_TIME = 2024-01-01T00:00:00\n"
///         "STOP_TIME = 2024-01-01T00:02:00\n"
///         "META_STOP\n"
///         "2024-01-01T00:00:00 7000.000    0.000 0.0  0.000000 7.546053 0.0\n"
///         "2024-01-01T00:01:00 6985.363  452.448 0.0 -0.487742 7.530274 0.0\n"
///         "2024-01-01T00:02:00 6941.512  903.003 0.0 -0.973444 7.483003 0.0\n"
///     )
///     seg = oem.segments[0]
///     print(seg.object_name, seg.ref_frame, len(seg))
///     # DEMOSAT EME2000 3
///     state = oem.interp(satkit.time(2024, 1, 1, 0, 0, 30))
///     print(state[:3] / 1e3)  # position, km
///     # [6996.34...  226.34...    0.  ]
///     ```
#[pyclass(name = "OEM", module = "satkit", frozen)]
pub struct PyOEM(pub Arc<OEM>);

#[pymethods]
impl PyOEM {
    /// Load an OEM from a file
    ///
    /// The encoding (KVN or XML) is detected from the content, not the
    /// file extension.
    ///
    /// Args:
    ///     filename (str | os.PathLike): path of the file
    ///
    /// Returns:
    ///     OEM: the message
    ///
    /// Raises:
    ///     FileNotFoundError: if the file does not exist; the exception names
    ///         the path. Another I/O failure raises the matching ``OSError``
    ///     ValueError: if the content is not a valid OEM; for KVN the message
    ///         gives the line
    #[staticmethod]
    fn from_file(py: Python<'_>, filename: std::path::PathBuf) -> PyResult<Self> {
        py.detach(|| {
            let text = std::fs::read_to_string(&filename)
                .map_err(|e| crate::pytle::io_err(e, &filename))?;
            OEM::from_text(&text).map_err(oem_err)
        })
        .map(|oem| Self(Arc::new(oem)))
    }

    /// Parse an OEM from text
    ///
    /// Text starting with ``CCSDS_OEM_VERS`` is read as KVN, text starting
    /// with ``<`` as XML (a bare ``<oem>`` or one inside an ``<ndm>``).
    /// Message versions 1.0 to 3.0 are accepted.
    ///
    /// Args:
    ///     text (str): the content of an OEM file
    ///
    /// Returns:
    ///     OEM: the message
    ///
    /// Raises:
    ///     ValueError: if the text is neither encoding, a mandatory keyword is
    ///         missing, a value cannot be read, the epochs of a segment are not
    ///         increasing, or ``TIME_SYSTEM`` is not one of UTC, TAI, TT, GPS,
    ///         TDB or UT1. For KVN the message gives the line
    #[staticmethod]
    fn from_text(py: Python<'_>, text: String) -> PyResult<Self> {
        py.detach(|| OEM::from_text(&text))
            .map(|oem| Self(Arc::new(oem)))
            .map_err(oem_err)
    }

    /// ``CCSDS_OEM_VERS``: version of the message format, e.g. ``"2.0"``
    #[getter]
    fn version(&self) -> &str {
        &self.0.version
    }

    /// ``COMMENT`` lines of the header
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.0.comments.clone()
    }

    /// ``CLASSIFICATION`` of the header (version 3), or None
    #[getter]
    fn classification(&self) -> Option<&str> {
        self.0.classification.as_deref()
    }

    /// ``CREATION_DATE``: when the message was written (UTC)
    #[getter]
    fn creation_date(&self) -> PyInstant {
        PyInstant(self.0.creation_date)
    }

    /// ``ORIGINATOR``: who produced the message
    #[getter]
    fn originator(&self) -> &str {
        &self.0.originator
    }

    /// ``MESSAGE_ID`` of the header (version 3), or None
    #[getter]
    fn message_id(&self) -> Option<&str> {
        self.0.message_id.as_deref()
    }

    /// The segments of the message, in file order; at least one
    #[getter]
    fn segments(&self) -> Vec<PyOEMSegment> {
        (0..self.0.segments.len())
            .map(|idx| PyOEMSegment {
                oem: Arc::clone(&self.0),
                idx,
            })
            .collect()
    }

    /// Interpolate the state at one or more times
    ///
    /// Each time is interpolated in the first segment whose usable span
    /// holds it, and the state is in that segment's frame. Segments are
    /// never interpolated across.
    ///
    /// With no ``method`` or ``degree`` the interpolation is the one the
    /// file recommends (``OEMSegment.interpolation`` and
    /// ``interpolation_degree``); a file that names no method gets Hermite,
    /// and a missing degree is 7.
    ///
    /// Args:
    ///     time (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): time(s) at which to interpolate
    ///     method (str, optional): ``"lagrange"`` (a polynomial through
    ///         ``degree + 1`` points, position and velocity separately) or
    ///         ``"hermite"`` (a polynomial matching position and velocity at
    ///         ``(degree + 1) / 2`` points). Default: the file's
    ///     degree (int, optional): degree of the polynomial, at least 1.
    ///         Default: the file's
    ///
    /// Returns:
    ///     numpy.ndarray: 6-element state [x, y, z, vx, vy, vz] in meters and
    ///     m/s for a single time; for a list / tuple / array of N times, one
    ///     (N, 6) array
    ///
    /// Raises:
    ///     ValueError: if a time is outside every segment's usable span, the
    ///         file names an interpolation method satkit does not have and
    ///         ``method`` is not given, ``method`` is not one of the two
    ///         names, or ``degree`` is 0
    ///     TypeError: if time is not a time or a list, tuple or 1-D array of
    ///         times
    #[pyo3(signature=(time, *, method=None, degree=None))]
    fn interp(
        &self,
        py: Python,
        time: &Bound<'_, PyAny>,
        method: Option<&str>,
        degree: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        let oem = &self.0;
        interp_py(py, time, method, degree, |t, m, d| {
            match oem.segment_at(t) {
                Some(seg) => interp_segment(seg, t, m, d),
                // Reports the span of the whole message
                None => oem.interp(t).map(|_| [0.0; 6]),
            }
        })
    }

    fn __len__(&self) -> usize {
        self.0.segments.len()
    }

    fn __repr__(&self) -> String {
        let first = &self.0.segments[0].metadata;
        format!(
            "OEM(version={:?}, originator={:?}, object_name={:?}, segments={})",
            self.0.version,
            self.0.originator,
            first.object_name,
            self.0.segments.len()
        )
    }
}

/// One segment of an OEM: a metadata block and the states under it
///
/// Obtained from ``OEM.segments``. A producer starts a new segment where
/// the states must not be interpolated across: a maneuver, or a change of
/// frame or central body.
///
/// Positions are in meters, velocities in m/s and accelerations in m/s²,
/// in the frame ``ref_frame`` centered on ``center_name``. ``len(segment)``
/// is the number of states.
#[pyclass(name = "OEMSegment", module = "satkit", frozen)]
pub struct PyOEMSegment {
    oem: Arc<OEM>,
    idx: usize,
}

impl PyOEMSegment {
    fn seg(&self) -> &Segment {
        &self.oem.segments[self.idx]
    }

    fn column(
        &self,
        py: Python,
        f: impl Fn(&satkit::oem::State) -> satkit::mathtypes::Vector3,
    ) -> PyResult<Py<PyAny>> {
        let states = &self.seg().states;
        let flat: Vec<f64> = states
            .iter()
            .flat_map(|s| {
                let v = f(s);
                [v[0], v[1], v[2]]
            })
            .collect();
        slice2py2d(py, &flat, states.len(), 3)
    }
}

#[pymethods]
impl PyOEMSegment {
    /// ``OBJECT_NAME``
    #[getter]
    fn object_name(&self) -> &str {
        &self.seg().metadata.object_name
    }

    /// ``OBJECT_ID``, normally the international designator (``1998-067A``)
    #[getter]
    fn object_id(&self) -> &str {
        &self.seg().metadata.object_id
    }

    /// ``CENTER_NAME``: the origin of the frame, e.g. ``"EARTH"``
    #[getter]
    fn center_name(&self) -> &str {
        &self.seg().metadata.center_name
    }

    /// ``REF_FRAME`` as written, e.g. ``"EME2000"``, ``"ITRF2000"``
    #[getter]
    fn ref_frame(&self) -> &str {
        &self.seg().metadata.ref_frame
    }

    /// The ``satkit.frame`` that ``ref_frame`` names, or None
    ///
    /// ``EME2000``, ``GCRF``, ``ICRF`` and ``TEME`` map to the frame of the
    /// same name and every ITRF realization (``ITRF2000``, ``ITRF-93``, ...)
    /// to ``satkit.frame.ITRF``; other frames give None. The name alone does
    /// not make the states Earth-centered: check ``center_name`` too.
    #[getter]
    fn frame(&self) -> Option<PyFrame> {
        self.seg().metadata.frame().map(PyFrame::from)
    }

    /// ``REF_FRAME_EPOCH``: epoch of the frame, or None
    #[getter]
    fn ref_frame_epoch(&self) -> Option<PyInstant> {
        opt_time(self.seg().metadata.ref_frame_epoch)
    }

    /// ``TIME_SYSTEM`` as written, e.g. ``"UTC"``
    #[getter]
    fn time_system(&self) -> &str {
        &self.seg().metadata.time_system
    }

    /// The ``satkit.timescale`` that ``time_system`` names; the epochs of the
    /// segment were read in it
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        use satkit::TimeScale as T;
        match self.seg().metadata.time_scale {
            T::UTC => PyTimeScale::UTC,
            T::TT => PyTimeScale::TT,
            T::UT1 => PyTimeScale::UT1,
            T::TAI => PyTimeScale::TAI,
            T::GPS => PyTimeScale::GPS,
            T::TDB => PyTimeScale::TDB,
            T::Invalid => PyTimeScale::Invalid,
        }
    }

    /// ``START_TIME``: epoch of the first state
    #[getter]
    fn start_time(&self) -> PyInstant {
        PyInstant(self.seg().metadata.start_time)
    }

    /// ``STOP_TIME``: epoch of the last state
    #[getter]
    fn stop_time(&self) -> PyInstant {
        PyInstant(self.seg().metadata.stop_time)
    }

    /// ``USEABLE_START_TIME``: start of the span the producer says is fit for
    /// interpolation, or None
    #[getter]
    fn useable_start_time(&self) -> Option<PyInstant> {
        opt_time(self.seg().metadata.useable_start_time)
    }

    /// ``USEABLE_STOP_TIME``: end of that span, or None
    #[getter]
    fn useable_stop_time(&self) -> Option<PyInstant> {
        opt_time(self.seg().metadata.useable_stop_time)
    }

    /// The span ``interp`` accepts, as a (start, stop) tuple of ``satkit.time``
    ///
    /// The ``USEABLE_START_TIME`` to ``USEABLE_STOP_TIME`` span where the
    /// file gives one, and never past the first and last state.
    #[getter]
    fn usable_span(&self) -> (PyInstant, PyInstant) {
        let (a, b) = self.seg().usable_span();
        (PyInstant(a), PyInstant(b))
    }

    /// ``INTERPOLATION`` as written: the method the producer recommends
    /// (``"LAGRANGE"``, ``"HERMITE"``, ...), or None
    #[getter]
    fn interpolation(&self) -> Option<&str> {
        self.seg().metadata.interpolation.as_deref()
    }

    /// ``INTERPOLATION_DEGREE``, or None
    #[getter]
    fn interpolation_degree(&self) -> Option<u32> {
        self.seg().metadata.interpolation_degree
    }

    /// ``COMMENT`` lines of the metadata block
    #[getter]
    fn metadata_comments(&self) -> Vec<String> {
        self.seg().metadata.comments.clone()
    }

    /// ``COMMENT`` lines at the start of the ephemeris data
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.seg().comments.clone()
    }

    /// Epochs of the states, a list of ``satkit.time`` in increasing order
    #[getter]
    fn time(&self) -> Vec<PyInstant> {
        self.seg()
            .states
            .iter()
            .map(|s| PyInstant(s.time))
            .collect()
    }

    /// Positions, an (N, 3) array in meters
    #[getter]
    fn pos(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.column(py, |s| s.pos)
    }

    /// Velocities, an (N, 3) array in m/s
    #[getter]
    fn vel(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.column(py, |s| s.vel)
    }

    /// Accelerations, an (N, 3) array in m/s², or None unless every state
    /// of the segment gives one
    #[getter]
    fn acc(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.seg().states.iter().any(|s| s.acc.is_none()) {
            return Ok(None);
        }
        self.column(py, |s| s.acc.unwrap_or_default()).map(Some)
    }

    /// Covariance matrices of the segment, a list of dictionaries (empty if
    /// the segment has none)
    ///
    /// Each has the keys ``time`` (satkit.time), ``ref_frame`` (the
    /// ``COV_REF_FRAME`` string such as ``"RTN"``, or None for the segment's
    /// frame), ``cov`` (the symmetric (6, 6) covariance of
    /// [x, y, z, vx, vy, vz] in m², m²/s and m²/s²) and ``comments``.
    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for c in &self.seg().covariances {
            let d = PyDict::new(py);
            d.set_item("time", PyInstant(c.time))?;
            d.set_item("ref_frame", c.ref_frame.as_deref())?;
            // Symmetric, so row- and column-major order agree
            d.set_item("cov", slice2py2d(py, c.matrix.as_slice(), 6, 6)?)?;
            d.set_item("comments", c.comments.clone())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Interpolate the state at one or more times within the segment
    ///
    /// As ``OEM.interp``, restricted to this segment.
    ///
    /// Args:
    ///     time (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): time(s) at which to interpolate
    ///     method (str, optional): ``"lagrange"`` or ``"hermite"``. Default:
    ///         the file's, or Hermite if it names none
    ///     degree (int, optional): degree of the polynomial, at least 1.
    ///         Default: the file's, or 7
    ///
    /// Returns:
    ///     numpy.ndarray: 6-element state [x, y, z, vx, vy, vz] in meters and
    ///     m/s for a single time; for a list / tuple / array of N times, one
    ///     (N, 6) array
    ///
    /// Raises:
    ///     ValueError: if a time is outside ``usable_span``, the file names an
    ///         interpolation method satkit does not have and ``method`` is not
    ///         given, ``method`` is not one of the two names, or ``degree`` is 0
    ///     TypeError: if time is not a time or a list, tuple or 1-D array of
    ///         times
    #[pyo3(signature=(time, *, method=None, degree=None))]
    fn interp(
        &self,
        py: Python,
        time: &Bound<'_, PyAny>,
        method: Option<&str>,
        degree: Option<u32>,
    ) -> PyResult<Py<PyAny>> {
        let seg = self.seg();
        interp_py(py, time, method, degree, |t, m, d| {
            interp_segment(seg, t, m, d)
        })
    }

    fn __len__(&self) -> usize {
        self.seg().states.len()
    }

    fn __repr__(&self) -> String {
        let m = &self.seg().metadata;
        format!(
            "OEMSegment(object_name={:?}, ref_frame={:?}, center_name={:?}, start_time={}, stop_time={}, states={})",
            m.object_name,
            m.ref_frame,
            m.center_name,
            m.start_time,
            m.stop_time,
            self.seg().states.len()
        )
    }
}
