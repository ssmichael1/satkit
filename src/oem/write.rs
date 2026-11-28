//! Building OEMs and writing the KVN encoding.

use std::fmt::Write as _;

use super::{
    check_segment, time_scale, Covariance, Error, Metadata, Result, Segment, State,
    DEFAULT_INTERPOLATION_DEGREE, KM, OEM,
};
use crate::orbitprop::PropagationResult;
use crate::{Duration, Instant, TimeScale};

/// `t` as a CCSDS calendar epoch (`YYYY-MM-DDThh:mm:ss.ffffff`) in `scale`.
fn format_epoch(t: &Instant, scale: TimeScale) -> String {
    let (y, mo, d, h, mi, s) = t.as_datetime_with_scale(scale);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:09.6}")
}

/// A value that must be present and fit on one KVN line.
fn required<'a>(field: &'static str, value: &'a str) -> Result<&'a str> {
    let v = value.trim();
    if v.is_empty() {
        return Err(Error::MissingField(field));
    }
    one_line(field, v)
}

fn one_line<'a>(field: &'static str, value: &'a str) -> Result<&'a str> {
    if value.contains(['\n', '\r']) {
        return Err(Error::InvalidField {
            field,
            message: "contains a line break".to_string(),
        });
    }
    Ok(value)
}

fn write_comments(out: &mut String, comments: &[String]) -> Result<()> {
    for c in comments {
        let _ = writeln!(out, "COMMENT {}", one_line("COMMENT", c)?);
    }
    Ok(())
}

fn write_metadata(out: &mut String, m: &Metadata) -> Result<()> {
    // Epochs are written in `time_scale`, so the label must name it
    let time_system = required("TIME_SYSTEM", &m.time_system)?;
    let scale = m.time_scale;
    if time_scale(time_system)? != scale {
        return Err(Error::InvalidField {
            field: "TIME_SYSTEM",
            message: format!("{time_system:?} does not name the segment's time scale {scale:?}"),
        });
    }
    let epoch = |t: &Instant| format_epoch(t, scale);

    out.push_str("META_START\n");
    write_comments(out, &m.comments)?;
    let _ = writeln!(
        out,
        "OBJECT_NAME = {}",
        required("OBJECT_NAME", &m.object_name)?
    );
    let _ = writeln!(out, "OBJECT_ID = {}", required("OBJECT_ID", &m.object_id)?);
    let _ = writeln!(
        out,
        "CENTER_NAME = {}",
        required("CENTER_NAME", &m.center_name)?
    );
    let _ = writeln!(out, "REF_FRAME = {}", required("REF_FRAME", &m.ref_frame)?);
    if let Some(t) = &m.ref_frame_epoch {
        let _ = writeln!(out, "REF_FRAME_EPOCH = {}", epoch(t));
    }
    let _ = writeln!(out, "TIME_SYSTEM = {time_system}");
    let _ = writeln!(out, "START_TIME = {}", epoch(&m.start_time));
    if let Some(t) = &m.useable_start_time {
        let _ = writeln!(out, "USEABLE_START_TIME = {}", epoch(t));
    }
    if let Some(t) = &m.useable_stop_time {
        let _ = writeln!(out, "USEABLE_STOP_TIME = {}", epoch(t));
    }
    let _ = writeln!(out, "STOP_TIME = {}", epoch(&m.stop_time));
    if let Some(name) = &m.interpolation {
        let _ = writeln!(out, "INTERPOLATION = {}", one_line("INTERPOLATION", name)?);
    }
    if let Some(degree) = m.interpolation_degree {
        let _ = writeln!(out, "INTERPOLATION_DEGREE = {degree}");
    }
    out.push_str("META_STOP\n\n");
    Ok(())
}

fn write_state(out: &mut String, s: &State, scale: TimeScale) -> Result<()> {
    let finite = |v: &crate::mathtypes::Vector3| (0..3).all(|k| v[k].is_finite());
    if !finite(&s.pos) || !finite(&s.vel) || s.acc.as_ref().is_some_and(|a| !finite(a)) {
        return Err(Error::InvalidField {
            field: "EPOCH",
            message: format!("the state at {} is not finite", s.time),
        });
    }
    out.push_str(&format_epoch(&s.time, scale));
    // Micrometers and nanometers per second
    for k in 0..3 {
        let _ = write!(out, " {:.9}", s.pos[k] / KM);
    }
    for k in 0..3 {
        let _ = write!(out, " {:.12}", s.vel[k] / KM);
    }
    if let Some(acc) = &s.acc {
        for k in 0..3 {
            let _ = write!(out, " {:.15}", acc[k] / KM);
        }
    }
    out.push('\n');
    Ok(())
}

fn write_covariance(out: &mut String, c: &Covariance, scale: TimeScale) -> Result<()> {
    write_comments(out, &c.comments)?;
    let _ = writeln!(out, "EPOCH = {}", format_epoch(&c.time, scale));
    if let Some(frame) = &c.ref_frame {
        let _ = writeln!(out, "COV_REF_FRAME = {}", required("COV_REF_FRAME", frame)?);
    }
    // Lower triangle, row by row
    for i in 0..6 {
        for j in 0..=i {
            let v = c.matrix[(i, j)];
            if !v.is_finite() {
                return Err(Error::InvalidField {
                    field: "COVARIANCE",
                    message: format!("the covariance at {} is not finite", c.time),
                });
            }
            let sep = if j == 0 { "" } else { " " };
            let _ = write!(out, "{sep}{:.14e}", v / (KM * KM));
        }
        out.push('\n');
    }
    Ok(())
}

impl Segment {
    /// A segment of Earth-centered states with UTC epochs.
    ///
    /// `ref_frame` is the CCSDS name of the frame the states are in (`GCRF`,
    /// `EME2000`, `ITRF2014`, `TEME`, ...). `START_TIME` and `STOP_TIME` are
    /// the first and last epochs. The other metadata (interpolation hint,
    /// usable span, comments, another center or time system) can be set on
    /// the fields of the segment returned.
    ///
    /// # Errors
    ///
    /// [`Error::Empty`] if `states` is empty and [`Error::EpochOrder`] if
    /// its epochs are not strictly increasing.
    pub fn from_states(
        object_name: &str,
        object_id: &str,
        ref_frame: &str,
        states: Vec<State>,
    ) -> Result<Self> {
        let (Some(first), Some(last)) = (states.first(), states.last()) else {
            return Err(Error::Empty("state vectors in a segment"));
        };
        let segment = Self {
            metadata: Metadata {
                comments: Vec::new(),
                object_name: object_name.to_string(),
                object_id: object_id.to_string(),
                center_name: "EARTH".to_string(),
                ref_frame: ref_frame.to_string(),
                ref_frame_epoch: None,
                time_system: "UTC".to_string(),
                time_scale: TimeScale::UTC,
                start_time: first.time,
                stop_time: last.time,
                useable_start_time: None,
                useable_stop_time: None,
                interpolation: None,
                interpolation_degree: None,
            },
            comments: Vec::new(),
            states,
            covariances: Vec::new(),
        };
        check_segment(&segment)?;
        Ok(segment)
    }
}

impl OEM {
    /// A version 2.0 message holding `segments`, created now.
    ///
    /// # Errors
    ///
    /// [`Error::Empty`] if `segments` is empty.
    pub fn new(originator: &str, segments: Vec<Segment>) -> Result<Self> {
        if segments.is_empty() {
            return Err(Error::Empty("segments"));
        }
        Ok(Self {
            version: "2.0".to_string(),
            comments: Vec::new(),
            classification: None,
            creation_date: Instant::now(),
            originator: originator.to_string(),
            message_id: None,
            segments,
        })
    }

    /// An OEM of the trajectory of a high-precision propagation, sampled
    /// every `step`.
    ///
    /// The message has one segment, in the GCRF with UTC epochs, holding the
    /// state at the earlier end of the propagation, every `step` after it,
    /// and the later end. Its metadata recommends Hermite interpolation of
    /// degree [`DEFAULT_INTERPOLATION_DEGREE`]. State transition matrices in
    /// the result are not written.
    ///
    /// The propagation needs dense output
    /// ([`PropagationResult::can_interp`]), the default for
    /// [`propagate`](crate::orbitprop::propagate).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStep`] if `step` is not positive, and
    /// [`Error::Propagation`] if the result cannot be interpolated.
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::oem::OEM;
    /// use satkit::orbitprop::{propagate, PropSettings, SimpleState};
    /// use satkit::{Duration, Instant};
    ///
    /// let start = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0)?;
    /// let mut state = SimpleState::zeros();
    /// state[0] = 7.0e6;
    /// state[4] = (satkit::consts::MU_EARTH / 7.0e6).sqrt();
    /// let mut settings = PropSettings::default();
    /// settings.gravity_degree = 4;
    /// let result = propagate(
    ///     &state,
    ///     &start,
    ///     &(start + Duration::from_minutes(30.0)),
    ///     &settings,
    ///     None,
    /// )?;
    ///
    /// let oem = OEM::from_propagation(
    ///     &result,
    ///     Duration::from_seconds(60.0),
    ///     "DEMOSAT",
    ///     "2024-001A",
    ///     "EXAMPLE",
    /// )?;
    /// assert_eq!(oem.segments[0].states.len(), 31);
    /// let text = oem.to_kvn_string()?;
    /// assert_eq!(OEM::from_text(&text)?.segments[0].states.len(), 31);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_propagation<const T: usize>(
        result: &PropagationResult<T>,
        step: Duration,
        object_name: &str,
        object_id: &str,
        originator: &str,
    ) -> Result<Self> {
        let step_secs = step.as_seconds();
        if step_secs.is_nan() || step_secs <= 0.0 {
            return Err(Error::InvalidStep);
        }
        // A backward propagation is written forward in time
        let (begin, end) = if result.time_end < result.time_begin {
            (result.time_end, result.time_begin)
        } else {
            (result.time_begin, result.time_end)
        };
        let (state_begin, state_end) = if result.time_end < result.time_begin {
            (&result.state_end, &result.state_begin)
        } else {
            (&result.state_begin, &result.state_end)
        };
        let state = |time: Instant, x: &crate::mathtypes::Matrix<6, T>| State {
            time,
            pos: numeris::vector![x[(0, 0)], x[(1, 0)], x[(2, 0)]],
            vel: numeris::vector![x[(3, 0)], x[(4, 0)], x[(5, 0)]],
            acc: None,
        };

        // The samples strictly between the two ends. Instants count whole
        // microseconds, so a sample can round onto its neighbour or the end
        let span = (end - begin).as_seconds();
        let mut inner: Vec<Instant> = (1..)
            .map(|i| i as f64 * step_secs)
            .take_while(|dt| *dt < span)
            .map(|dt| begin + Duration::from_seconds(dt))
            .filter(|t| *t > begin && *t < end)
            .collect();
        inner.dedup();

        // The ends are the propagation's own states, not interpolated
        let mut states = vec![state(begin, state_begin)];
        if !inner.is_empty() {
            let x = result.interp_batch(&inner)?;
            states.extend(x.iter().zip(&inner).map(|(x, t)| state(*t, x)));
        } else if end > begin && !result.can_interp() {
            // Refuse a table of the two ends alone when the samples asked
            // for cannot be had, even if the step is longer than the span
            result.interp(&end)?;
        }
        if end > begin {
            states.push(state(end, state_end));
        }

        let mut segment = Segment::from_states(object_name, object_id, "GCRF", states)?;
        segment.metadata.interpolation = Some("HERMITE".to_string());
        segment.metadata.interpolation_degree = Some(DEFAULT_INTERPOLATION_DEGREE);
        Self::new(originator, vec![segment])
    }

    /// The message in the KVN ("keyword = value") encoding.
    ///
    /// Positions, velocities and covariances are converted back to the
    /// kilometers of the format, and written to the micrometer and the
    /// nanometer per second. Epochs are written to the microsecond in each
    /// segment's [`time_scale`](Metadata::time_scale). `version` is written
    /// as it is: accelerations and covariances need 2.0 or later,
    /// `classification` and `message_id` 3.0.
    ///
    /// # Errors
    ///
    /// [`Error::MissingField`] for an empty mandatory field,
    /// [`Error::InvalidField`] for a value holding a line break, a
    /// non-finite number, or a `time_system` that does not name the
    /// segment's `time_scale`, and [`Error::Empty`] or
    /// [`Error::EpochOrder`] for a message the reader would reject.
    pub fn to_kvn_string(&self) -> Result<String> {
        if self.segments.is_empty() {
            return Err(Error::Empty("segments"));
        }
        let mut out = String::new();
        let _ = writeln!(
            out,
            "CCSDS_OEM_VERS = {}",
            required("CCSDS_OEM_VERS", &self.version)?
        );
        write_comments(&mut out, &self.comments)?;
        if let Some(c) = &self.classification {
            let _ = writeln!(out, "CLASSIFICATION = {}", required("CLASSIFICATION", c)?);
        }
        let _ = writeln!(
            out,
            "CREATION_DATE = {}",
            format_epoch(&self.creation_date, TimeScale::UTC)
        );
        let _ = writeln!(
            out,
            "ORIGINATOR = {}",
            required("ORIGINATOR", &self.originator)?
        );
        if let Some(id) = &self.message_id {
            let _ = writeln!(out, "MESSAGE_ID = {}", required("MESSAGE_ID", id)?);
        }
        out.push('\n');

        for segment in &self.segments {
            check_segment(segment)?;
            let scale = segment.metadata.time_scale;
            write_metadata(&mut out, &segment.metadata)?;
            write_comments(&mut out, &segment.comments)?;
            for s in &segment.states {
                write_state(&mut out, s, scale)?;
            }
            out.push('\n');
            if !segment.covariances.is_empty() {
                out.push_str("COVARIANCE_START\n");
                for c in &segment.covariances {
                    write_covariance(&mut out, c, scale)?;
                }
                out.push_str("COVARIANCE_STOP\n\n");
            }
        }
        Ok(out)
    }

    /// Write the message to a file in the KVN encoding
    /// ([`to_kvn_string`](Self::to_kvn_string)).
    ///
    /// # Errors
    ///
    /// The errors of [`to_kvn_string`](Self::to_kvn_string), and
    /// [`Error::Io`] if the file cannot be written.
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        Ok(std::fs::write(path, self.to_kvn_string()?)?)
    }
}
