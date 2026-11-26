//! CCSDS Orbit Ephemeris Messages (OEM).
//!
//! An OEM is a table of position and velocity vectors of one object at a
//! sequence of epochs, with the reference frame, central body and time
//! system they are expressed in (CCSDS 502.0-B, *Orbit Data Messages*). It
//! is the usual way operators and agencies exchange ephemerides; NASA
//! publishes the ISS trajectory as one, for example.
//!
//! [`OEM::from_text`] and [`OEM::from_file`] read the KVN ("keyword = value")
//! encoding and, with the `omm-xml` feature, the XML encoding, of message
//! versions 1.0 to 3.0. A message is a list of [`Segment`]s, each with its
//! own [`Metadata`], state vectors and optional covariance matrices.
//! [`OEM::interp`] and [`Segment::interp`] interpolate the states.
//!
//! # Units and time
//!
//! The file gives kilometers and km/s; this module converts to the SI units
//! used in the rest of satkit: **meters, m/s and m/s²**, and covariances in
//! m², m²/s and m²/s². Epochs are read in the segment's `TIME_SYSTEM` and
//! held as [`Instant`]s, so they compare correctly across segments and with
//! any other satkit time.
//!
//! # Frames
//!
//! States are kept in the frame the file gives (`REF_FRAME`); nothing is
//! rotated on reading. [`Metadata::frame`] maps the name to a satkit
//! [`Frame`] where there is one, for use with
//! [`frametransform`](crate::frametransform).
//!
//! # Example
//!
//! ```
//! use satkit::oem::OEM;
//! use satkit::Instant;
//!
//! let text = "\
//! CCSDS_OEM_VERS = 2.0
//! CREATION_DATE = 2024-01-01T00:00:00
//! ORIGINATOR = EXAMPLE
//!
//! META_START
//! OBJECT_NAME = DEMOSAT
//! OBJECT_ID = 2024-001A
//! CENTER_NAME = EARTH
//! REF_FRAME = EME2000
//! TIME_SYSTEM = UTC
//! START_TIME = 2024-01-01T00:00:00
//! STOP_TIME = 2024-01-01T00:02:00
//! META_STOP
//!
//! 2024-01-01T00:00:00 7000.000    0.000 0.000  0.000000 7.546053 0.000000
//! 2024-01-01T00:01:00 6985.363  452.448 0.000 -0.487742 7.530274 0.000000
//! 2024-01-01T00:02:00 6941.512  903.003 0.000 -0.973444 7.483003 0.000000
//! ";
//! let oem = OEM::from_text(text)?;
//! let seg = &oem.segments[0];
//! assert_eq!(seg.metadata.object_name, "DEMOSAT");
//! assert_eq!(seg.states.len(), 3);
//! // meters, not kilometers
//! assert_eq!(seg.states[0].pos[0], 7.0e6);
//!
//! let t = Instant::from_datetime(2024, 1, 1, 0, 0, 30.0)?;
//! let (pos, vel) = oem.interp(&t)?;
//! # let _ = (pos, vel);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod interp;
mod kvn;
#[cfg(feature = "omm-xml")]
mod xml;

pub use error::{Error, Result};

use crate::mathtypes::{Matrix6, Vector3};
use crate::{Frame, Instant, TimeScale};

/// Interpolation degree used when the file names none.
pub const DEFAULT_INTERPOLATION_DEGREE: u32 = 7;

/// Kilometers (the OEM unit) to meters.
const KM: f64 = 1.0e3;

/// How the state vectors of a segment are interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Interpolation {
    /// A polynomial through `degree + 1` neighbouring points, applied to
    /// position and velocity independently.
    Lagrange,
    /// A polynomial matching position and velocity at `(degree + 1) / 2`
    /// neighbouring points; the velocity returned is its derivative.
    Hermite,
}

/// One line of ephemeris: the state at an epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Epoch of the state
    pub time: Instant,
    /// Position, meters, in the segment's reference frame
    pub pos: Vector3,
    /// Velocity, m/s
    pub vel: Vector3,
    /// Acceleration, m/s², if the line gives one
    pub acc: Option<Vector3>,
}

/// A 6x6 position/velocity covariance at an epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    /// Epoch of the covariance
    pub time: Instant,
    /// `COV_REF_FRAME`: the frame of the matrix (for example `RTN`), if
    /// given; the segment's `REF_FRAME` otherwise
    pub ref_frame: Option<String>,
    /// Symmetric covariance of (x, y, z, vx, vy, vz): m² in the
    /// position block, m²/s in the cross blocks, m²/s² in the velocity block
    pub matrix: Matrix6,
    /// `COMMENT` lines of the covariance
    pub comments: Vec<String>,
}

/// The metadata block of a segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    /// `COMMENT` lines of the metadata block
    pub comments: Vec<String>,
    /// `OBJECT_NAME`
    pub object_name: String,
    /// `OBJECT_ID`, normally the international designator (`1998-067A`)
    pub object_id: String,
    /// `CENTER_NAME`: the origin of the frame, e.g. `EARTH`
    pub center_name: String,
    /// `REF_FRAME` as written, e.g. `EME2000`, `ITRF2000`, `TEME`
    pub ref_frame: String,
    /// `REF_FRAME_EPOCH`: epoch of the frame, for frames that need one
    pub ref_frame_epoch: Option<Instant>,
    /// `TIME_SYSTEM` as written, e.g. `UTC`
    pub time_system: String,
    /// The time scale `time_system` names; every epoch of the segment was
    /// read in it
    pub time_scale: TimeScale,
    /// `START_TIME`: epoch of the first state
    pub start_time: Instant,
    /// `STOP_TIME`: epoch of the last state
    pub stop_time: Instant,
    /// `USEABLE_START_TIME`: start of the span the producer says is fit
    /// for interpolation
    pub useable_start_time: Option<Instant>,
    /// `USEABLE_STOP_TIME`: end of that span
    pub useable_stop_time: Option<Instant>,
    /// `INTERPOLATION` as written: the method the producer recommends
    pub interpolation: Option<String>,
    /// `INTERPOLATION_DEGREE`
    pub interpolation_degree: Option<u32>,
}

impl Metadata {
    /// The satkit frame `ref_frame` names, if there is one.
    ///
    /// `EME2000`, `GCRF`, `ICRF` and `TEME` map to the frame of the same
    /// name, and any ITRF realization (`ITRF2000`, `ITRF-93`, `ITRF2014`,
    /// ...) to [`Frame::ITRF`], satkit making no distinction between them.
    /// Other frames (`TOD`, `MCI`, ...) give `None`.
    ///
    /// The name alone does not make the states Earth-centered: check
    /// [`center_name`](Self::center_name) too.
    pub fn frame(&self) -> Option<Frame> {
        let name = self.ref_frame.trim().to_ascii_uppercase();
        match name.as_str() {
            "EME2000" => Some(Frame::EME2000),
            "GCRF" => Some(Frame::GCRF),
            "ICRF" => Some(Frame::ICRF),
            "TEME" => Some(Frame::TEME),
            s if s.starts_with("ITRF") => Some(Frame::ITRF),
            _ => None,
        }
    }
}

/// One segment of an OEM: a metadata block and the ephemeris under it.
///
/// A producer starts a new segment where the states must not be
/// interpolated across (a maneuver, a change of frame or of central body).
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// The metadata block
    pub metadata: Metadata,
    /// `COMMENT` lines at the start of the ephemeris data
    pub comments: Vec<String>,
    /// State vectors, in strictly increasing time order; never empty
    pub states: Vec<State>,
    /// Covariance matrices, if the segment has any
    pub covariances: Vec<Covariance>,
}

/// A CCSDS Orbit Ephemeris Message.
#[derive(Debug, Clone, PartialEq)]
pub struct OEM {
    /// `CCSDS_OEM_VERS`: version of the message format, e.g. `2.0`
    pub version: String,
    /// `COMMENT` lines of the header
    pub comments: Vec<String>,
    /// `CLASSIFICATION` (version 3)
    pub classification: Option<String>,
    /// `CREATION_DATE` (UTC)
    pub creation_date: Instant,
    /// `ORIGINATOR`
    pub originator: String,
    /// `MESSAGE_ID` (version 3)
    pub message_id: Option<String>,
    /// The segments, in file order; never empty
    pub segments: Vec<Segment>,
}

/// The time scale an OEM `TIME_SYSTEM` names.
fn time_scale(time_system: &str) -> Result<TimeScale> {
    match time_system.trim().to_ascii_uppercase().as_str() {
        "UTC" => Ok(TimeScale::UTC),
        "TAI" => Ok(TimeScale::TAI),
        "TT" => Ok(TimeScale::TT),
        "GPS" => Ok(TimeScale::GPS),
        "TDB" => Ok(TimeScale::TDB),
        "UT1" => Ok(TimeScale::UT1),
        _ => Err(Error::UnsupportedTimeSystem(time_system.trim().to_string())),
    }
}

/// Parse a CCSDS epoch (`YYYY-MM-DDThh:mm:ss[.f][Z]` or the day-of-year
/// form `YYYY-DDDThh:mm:ss[.f][Z]`) whose label is in `scale`.
fn parse_epoch(field: &'static str, text: &str, scale: TimeScale) -> Result<Instant> {
    let invalid = |e: crate::InstantError| Error::InvalidField {
        field,
        message: format!("cannot parse {:?}: {e}", text.trim()),
    };
    // Read the label as UTC (which checks the syntax and handles the
    // day-of-year form), then reinterpret its fields in `scale`
    let utc = crate::omm::parse_epoch(text).map_err(invalid)?;
    if scale == TimeScale::UTC {
        return Ok(utc);
    }
    let (y, mo, d, h, mi, s) = utc.as_datetime();
    Instant::from_datetime_with_scale(y, mo, d, h, mi, s, scale).map_err(invalid)
}

#[cfg(feature = "omm-xml")]
fn parse_f64(field: &'static str, text: &str) -> Result<f64> {
    let t = text.trim();
    match t.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(Error::InvalidField {
            field,
            message: format!("cannot parse {t:?} as a number"),
        }),
    }
}

/// Metadata from a keyword lookup; shared by the KVN and XML readers.
fn build_metadata<'a>(
    get: impl Fn(&'static str) -> Option<&'a str>,
    comments: Vec<String>,
) -> Result<Metadata> {
    let get = |key: &'static str| get(key).map(str::trim).filter(|s| !s.is_empty());
    let req = |key: &'static str| get(key).ok_or(Error::MissingField(key));

    let time_system = req("TIME_SYSTEM")?.to_string();
    let scale = time_scale(&time_system)?;
    let opt_epoch = |key: &'static str| get(key).map(|s| parse_epoch(key, s, scale)).transpose();

    let interpolation_degree = get("INTERPOLATION_DEGREE")
        .map(|s| {
            s.parse::<u32>().map_err(|e| Error::InvalidField {
                field: "INTERPOLATION_DEGREE",
                message: format!("cannot parse {s:?}: {e}"),
            })
        })
        .transpose()?;

    Ok(Metadata {
        comments,
        object_name: req("OBJECT_NAME")?.to_string(),
        object_id: req("OBJECT_ID")?.to_string(),
        center_name: req("CENTER_NAME")?.to_string(),
        ref_frame: req("REF_FRAME")?.to_string(),
        ref_frame_epoch: opt_epoch("REF_FRAME_EPOCH")?,
        time_scale: scale,
        start_time: parse_epoch("START_TIME", req("START_TIME")?, scale)?,
        stop_time: parse_epoch("STOP_TIME", req("STOP_TIME")?, scale)?,
        useable_start_time: opt_epoch("USEABLE_START_TIME")?,
        useable_stop_time: opt_epoch("USEABLE_STOP_TIME")?,
        interpolation: get("INTERPOLATION").map(str::to_string),
        interpolation_degree,
        time_system,
    })
}

/// A state from its epoch text and 6 (no acceleration) or 9 values in km,
/// km/s and km/s².
fn build_state(epoch: &str, v: &[f64], scale: TimeScale) -> Result<State> {
    debug_assert!(v.len() == 6 || v.len() == 9);
    Ok(State {
        time: parse_epoch("EPOCH", epoch, scale)?,
        pos: numeris::vector![v[0] * KM, v[1] * KM, v[2] * KM],
        vel: numeris::vector![v[3] * KM, v[4] * KM, v[5] * KM],
        acc: (v.len() == 9).then(|| numeris::vector![v[6] * KM, v[7] * KM, v[8] * KM]),
    })
}

/// A covariance from the 21 values of its lower triangle, row by row, in
/// km², km²/s and km²/s².
fn build_covariance(
    epoch: &str,
    ref_frame: Option<&str>,
    lower: &[f64],
    comments: Vec<String>,
    scale: TimeScale,
) -> Result<Covariance> {
    debug_assert_eq!(lower.len(), 21);
    let mut matrix = Matrix6::zeros();
    let mut k = 0;
    for i in 0..6 {
        for j in 0..=i {
            let v = lower[k] * KM * KM;
            matrix[(i, j)] = v;
            matrix[(j, i)] = v;
            k += 1;
        }
    }
    Ok(Covariance {
        time: parse_epoch("EPOCH", epoch, scale)?,
        ref_frame: ref_frame
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        matrix,
        comments,
    })
}

/// Checks shared by both readers once a segment is assembled.
fn check_segment(segment: &Segment) -> Result<()> {
    if segment.states.is_empty() {
        return Err(Error::Empty("state vectors in a segment"));
    }
    if let Some(w) = segment.states.windows(2).find(|w| w[1].time <= w[0].time) {
        return Err(Error::EpochOrder(w[1].time));
    }
    Ok(())
}

impl Segment {
    /// The span the segment can be interpolated over.
    ///
    /// This is the `USEABLE_START_TIME` to `USEABLE_STOP_TIME` span where
    /// the file gives one, and never extends past the first and last state.
    pub fn usable_span(&self) -> (Instant, Instant) {
        let first = self.states[0].time;
        let last = self.states[self.states.len() - 1].time;
        let start =
            self.metadata
                .useable_start_time
                .map_or(first, |t| if t > first { t } else { first });
        let stop = self
            .metadata
            .useable_stop_time
            .map_or(last, |t| if t < last { t } else { last });
        (start, stop)
    }

    /// Whether `time` is inside [`usable_span`](Self::usable_span).
    pub fn contains(&self, time: &Instant) -> bool {
        let (start, stop) = self.usable_span();
        *time >= start && *time <= stop
    }

    /// The interpolation the file recommends for the segment.
    ///
    /// `INTERPOLATION = LAGRANGE` or `HERMITE` with `INTERPOLATION_DEGREE`,
    /// `LINEAR` being Lagrange of degree 1. With no `INTERPOLATION`
    /// keyword the method is Hermite, which uses the velocities and so
    /// holds its accuracy on sparse ephemerides. A missing degree is
    /// [`DEFAULT_INTERPOLATION_DEGREE`].
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedInterpolation`] if the file names another method.
    pub fn interpolation(&self) -> Result<(Interpolation, u32)> {
        let degree = self
            .metadata
            .interpolation_degree
            .unwrap_or(DEFAULT_INTERPOLATION_DEGREE);
        match self.metadata.interpolation.as_deref() {
            None => Ok((Interpolation::Hermite, degree)),
            Some(name) => match name.trim().to_ascii_uppercase().as_str() {
                "LAGRANGE" => Ok((Interpolation::Lagrange, degree)),
                "HERMITE" => Ok((Interpolation::Hermite, degree)),
                "LINEAR" => Ok((Interpolation::Lagrange, 1)),
                _ => Err(Error::UnsupportedInterpolation(name.trim().to_string())),
            },
        }
    }

    /// Position (meters) and velocity (m/s) at `time`, in the segment's
    /// frame, with the interpolation the file recommends
    /// ([`interpolation`](Self::interpolation)).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if `time` is outside
    /// [`usable_span`](Self::usable_span), and
    /// [`Error::UnsupportedInterpolation`] if the file names a method this
    /// module does not implement (use [`interp_with`](Self::interp_with)).
    pub fn interp(&self, time: &Instant) -> Result<(Vector3, Vector3)> {
        let (method, degree) = self.interpolation()?;
        self.interp_with(time, method, degree)
    }

    /// Position (meters) and velocity (m/s) at `time` with a chosen
    /// interpolation method and polynomial degree.
    ///
    /// The points used are the ones nearest `time`; a segment with fewer
    /// points than the degree asks for uses all of them.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if `time` is outside
    /// [`usable_span`](Self::usable_span), [`Error::InvalidDegree`] if
    /// `degree` is 0.
    pub fn interp_with(
        &self,
        time: &Instant,
        method: Interpolation,
        degree: u32,
    ) -> Result<(Vector3, Vector3)> {
        if degree == 0 {
            return Err(Error::InvalidDegree);
        }
        let (start, stop) = self.usable_span();
        if *time < start || *time > stop {
            return Err(Error::OutOfRange {
                time: *time,
                start,
                stop,
            });
        }
        Ok(interp::interp(&self.states, time, method, degree as usize))
    }
}

impl OEM {
    /// Parse an OEM from text, detecting the encoding.
    ///
    /// Text starting with `<` is read as XML (requires the `omm-xml`
    /// feature); text starting with `CCSDS_OEM_VERS` as KVN.
    ///
    /// # Errors
    ///
    /// [`Error::UnrecognizedFormat`] for anything else, and the parser's
    /// error for a malformed message.
    pub fn from_text(s: &str) -> Result<Self> {
        let trimmed = s.trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n']);
        if trimmed.starts_with('<') {
            #[cfg(feature = "omm-xml")]
            {
                Self::from_xml_string(trimmed)
            }
            #[cfg(not(feature = "omm-xml"))]
            {
                Err(Error::XmlFeatureDisabled)
            }
        } else if trimmed.starts_with("CCSDS_OEM_VERS") {
            Self::from_kvn_string(s)
        } else {
            Err(Error::UnrecognizedFormat)
        }
    }

    /// Read an OEM from a file, detecting the encoding from the content
    /// (not the extension) as [`from_text`](Self::from_text) does.
    ///
    /// # Errors
    ///
    /// An error if the file cannot be read or is not a valid OEM.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::from_text(&std::fs::read_to_string(path)?)
    }

    /// The first segment whose usable span holds `time`.
    pub fn segment_at(&self, time: &Instant) -> Option<&Segment> {
        self.segments.iter().find(|s| s.contains(time))
    }

    /// Position (meters) and velocity (m/s) at `time`, interpolated in the
    /// segment that holds it ([`Segment::interp`]) and expressed in that
    /// segment's frame.
    ///
    /// Segments are never interpolated across: a time in a gap between
    /// two of them is out of range.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if no segment holds `time` (the span reported
    /// runs from the start of the first segment to the end of the last),
    /// and the errors of [`Segment::interp`].
    pub fn interp(&self, time: &Instant) -> Result<(Vector3, Vector3)> {
        self.segment_for(time)?.interp(time)
    }

    /// As [`interp`](Self::interp), with a chosen method and degree
    /// ([`Segment::interp_with`]).
    pub fn interp_with(
        &self,
        time: &Instant,
        method: Interpolation,
        degree: u32,
    ) -> Result<(Vector3, Vector3)> {
        self.segment_for(time)?.interp_with(time, method, degree)
    }

    fn segment_for(&self, time: &Instant) -> Result<&Segment> {
        self.segment_at(time).ok_or_else(|| Error::OutOfRange {
            time: *time,
            start: self.segments[0].usable_span().0,
            stop: self.segments[self.segments.len() - 1].usable_span().1,
        })
    }
}

#[cfg(test)]
mod tests;
