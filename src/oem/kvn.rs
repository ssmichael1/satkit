//! KVN ("keyword = value") encoding of the OEM.

use std::collections::HashMap;

use super::{
    build_covariance, build_metadata, build_state, check_segment, parse_epoch, Covariance, Error,
    Metadata, Result, Segment, OEM,
};
use crate::TimeScale;

const HEADER_KEYS: &[&str] = &[
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
];

const METADATA_KEYS: &[&str] = &[
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
];

/// Where the reader is in the message.
enum Section {
    Header,
    Metadata,
    Data,
    Covariance,
}

/// A covariance whose values are still being read.
struct PendingCovariance {
    line: usize,
    epoch: String,
    ref_frame: Option<String>,
    values: Vec<f64>,
    comments: Vec<String>,
}

fn kvn_err(line: usize, message: impl Into<String>) -> Error {
    Error::Kvn {
        line,
        message: message.into(),
    }
}

/// Puts the line number on an error from the shared builders.
fn at_line<T>(line: usize, r: Result<T>) -> Result<T> {
    r.map_err(|e| match e {
        Error::InvalidField { .. }
        | Error::MissingField(_)
        | Error::UnsupportedTimeSystem(_)
        | Error::EpochOrder(_)
        | Error::Empty(_) => kvn_err(line, e.to_string()),
        e => e,
    })
}

/// `KEY = value` → (KEY, value); `None` for a line with no `=`.
fn key_value(line: &str) -> Option<(&str, &str)> {
    let (k, v) = line.split_once('=')?;
    let k = k.trim();
    let is_keyword = !k.is_empty()
        && k.bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
    is_keyword.then_some((k, v.trim()))
}

/// The text of a `COMMENT` line, if it is one.
fn comment(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("COMMENT")?;
    (rest.is_empty() || rest.starts_with([' ', '\t'])).then(|| rest.trim())
}

fn numbers(line: usize, tokens: &[&str]) -> Result<Vec<f64>> {
    tokens
        .iter()
        .map(|t| match t.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(v),
            _ => Err(kvn_err(line, format!("cannot parse {t:?} as a number"))),
        })
        .collect()
}

fn finish_covariance(p: PendingCovariance, scale: TimeScale) -> Result<Covariance> {
    if p.values.len() != 21 {
        return Err(kvn_err(
            p.line,
            format!(
                "covariance has {} values; the lower triangle of a 6x6 matrix is 21",
                p.values.len()
            ),
        ));
    }
    at_line(
        p.line,
        build_covariance(
            &p.epoch,
            p.ref_frame.as_deref(),
            &p.values,
            p.comments,
            scale,
        ),
    )
}

/// The segment being read: its metadata is complete, its data is not.
struct OpenSegment {
    line: usize,
    segment: Segment,
}

fn close_segment(open: Option<OpenSegment>, segments: &mut Vec<Segment>) -> Result<()> {
    if let Some(open) = open {
        at_line(open.line, check_segment(&open.segment))?;
        segments.push(open.segment);
    }
    Ok(())
}

impl OEM {
    /// Parse an OEM from KVN text.
    ///
    /// The reader follows CCSDS 502.0-B for versions 1.0 to 3.0: a header,
    /// then one or more segments of a `META_START` … `META_STOP` block,
    /// ephemeris lines of an epoch and 6 values (9 with accelerations) and
    /// optional `COVARIANCE_START` … `COVARIANCE_STOP` blocks. `COMMENT`
    /// lines are kept with the block they are in; blank lines are skipped.
    ///
    /// # Errors
    ///
    /// [`Error::Kvn`], giving the line, for a keyword that does not belong
    /// where it is, a missing mandatory keyword, a value that cannot be
    /// read, a `TIME_SYSTEM` that is not supported, or epochs out of order.
    pub fn from_kvn_string(s: &str) -> Result<Self> {
        let mut section = Section::Header;
        let mut version: Option<String> = None;
        let mut header: HashMap<&'static str, String> = HashMap::new();
        let mut header_comments = Vec::new();

        let mut meta: HashMap<&'static str, String> = HashMap::new();
        let mut meta_comments = Vec::new();
        let mut meta_line = 0;

        let mut segments = Vec::new();
        let mut open: Option<OpenSegment> = None;
        let mut cov: Option<PendingCovariance> = None;
        let mut cov_comments = Vec::new();

        for (idx, raw) in s.lines().enumerate() {
            let n = idx + 1;
            let line = raw.trim_matches(['\u{feff}', ' ', '\t', '\r']);
            if line.is_empty() {
                continue;
            }
            if version.is_none() {
                match key_value(line) {
                    Some(("CCSDS_OEM_VERS", v)) if !v.is_empty() => {
                        version = Some(v.to_string());
                        continue;
                    }
                    _ => return Err(kvn_err(n, "expected CCSDS_OEM_VERS = <version>")),
                }
            }

            match section {
                Section::Header => {
                    if let Some(c) = comment(line) {
                        header_comments.push(c.to_string());
                    } else if line == "META_START" {
                        section = Section::Metadata;
                        meta_line = n;
                    } else if let Some((k, v)) = key_value(line) {
                        let Some(key) = HEADER_KEYS.iter().find(|h| **h == k) else {
                            return Err(kvn_err(n, format!("unexpected header keyword {k}")));
                        };
                        header.insert(key, v.to_string());
                    } else {
                        return Err(kvn_err(n, "expected a header keyword or META_START"));
                    }
                }
                Section::Metadata => {
                    if let Some(c) = comment(line) {
                        meta_comments.push(c.to_string());
                    } else if line == "META_STOP" {
                        let metadata: Metadata = at_line(
                            meta_line,
                            build_metadata(
                                |k| meta.get(k).map(String::as_str),
                                std::mem::take(&mut meta_comments),
                            ),
                        )?;
                        meta.clear();
                        open = Some(OpenSegment {
                            line: meta_line,
                            segment: Segment {
                                metadata,
                                comments: Vec::new(),
                                states: Vec::new(),
                                covariances: Vec::new(),
                            },
                        });
                        section = Section::Data;
                    } else if let Some((k, v)) = key_value(line) {
                        let Some(key) = METADATA_KEYS.iter().find(|m| **m == k) else {
                            return Err(kvn_err(n, format!("unexpected metadata keyword {k}")));
                        };
                        meta.insert(key, v.to_string());
                    } else {
                        return Err(kvn_err(n, "expected a metadata keyword or META_STOP"));
                    }
                }
                Section::Data => {
                    if line == "META_START" {
                        close_segment(open.take(), &mut segments)?;
                        section = Section::Metadata;
                        meta_line = n;
                        continue;
                    }
                    // `open` is set on every path into this section
                    let Some(cur) = open.as_mut() else {
                        return Err(kvn_err(n, "ephemeris data before any metadata"));
                    };
                    let seg = &mut cur.segment;
                    if let Some(c) = comment(line) {
                        seg.comments.push(c.to_string());
                    } else if line == "COVARIANCE_START" {
                        section = Section::Covariance;
                    } else {
                        let tokens: Vec<&str> = line.split_whitespace().collect();
                        if tokens.len() != 7 && tokens.len() != 10 {
                            return Err(kvn_err(
                                n,
                                format!(
                                    "expected an ephemeris line of an epoch and 6 or 9 values, \
                                     found {} fields",
                                    tokens.len()
                                ),
                            ));
                        }
                        let values = numbers(n, &tokens[1..])?;
                        let state =
                            at_line(n, build_state(tokens[0], &values, seg.metadata.time_scale))?;
                        if seg.states.last().is_some_and(|p| state.time <= p.time) {
                            return Err(kvn_err(n, Error::EpochOrder(state.time).to_string()));
                        }
                        seg.states.push(state);
                    }
                }
                Section::Covariance => {
                    let Some(cur) = open.as_mut() else {
                        return Err(kvn_err(n, "covariance data before any metadata"));
                    };
                    let seg = &mut cur.segment;
                    let scale = seg.metadata.time_scale;
                    if let Some(c) = comment(line) {
                        cov_comments.push(c.to_string());
                    } else if line == "COVARIANCE_STOP" {
                        if let Some(p) = cov.take() {
                            seg.covariances.push(finish_covariance(p, scale)?);
                        }
                        cov_comments.clear();
                        section = Section::Data;
                    } else if let Some((k, v)) = key_value(line) {
                        match k {
                            "EPOCH" => {
                                if let Some(p) = cov.take() {
                                    seg.covariances.push(finish_covariance(p, scale)?);
                                }
                                // Check the epoch here so the error names this line
                                at_line(n, parse_epoch("EPOCH", v, scale))?;
                                cov = Some(PendingCovariance {
                                    line: n,
                                    epoch: v.to_string(),
                                    ref_frame: None,
                                    values: Vec::with_capacity(21),
                                    comments: std::mem::take(&mut cov_comments),
                                });
                            }
                            "COV_REF_FRAME" => match cov.as_mut() {
                                Some(p) if p.values.is_empty() => {
                                    p.ref_frame = Some(v.to_string());
                                }
                                _ => {
                                    return Err(kvn_err(
                                        n,
                                        "COV_REF_FRAME must follow the EPOCH of its covariance",
                                    ))
                                }
                            },
                            _ => {
                                return Err(kvn_err(
                                    n,
                                    format!("unexpected covariance keyword {k}"),
                                ))
                            }
                        }
                    } else {
                        let Some(p) = cov.as_mut() else {
                            return Err(kvn_err(n, "covariance values before an EPOCH"));
                        };
                        let tokens: Vec<&str> = line.split_whitespace().collect();
                        p.values.extend(numbers(n, &tokens)?);
                    }
                }
            }
        }

        let last = s.lines().count();
        match section {
            Section::Header => {}
            Section::Metadata => return Err(kvn_err(last, "META_START with no META_STOP")),
            Section::Covariance => {
                return Err(kvn_err(last, "COVARIANCE_START with no COVARIANCE_STOP"))
            }
            Section::Data => close_segment(open.take(), &mut segments)?,
        }
        let Some(version) = version else {
            return Err(kvn_err(1, "expected CCSDS_OEM_VERS = <version>"));
        };
        if segments.is_empty() {
            return Err(Error::Empty("segments"));
        }

        let field = |key: &'static str| header.get(key).filter(|s| !s.is_empty());
        let creation_date = parse_epoch(
            "CREATION_DATE",
            field("CREATION_DATE").ok_or(Error::MissingField("CREATION_DATE"))?,
            TimeScale::UTC,
        )?;
        Ok(Self {
            version,
            comments: header_comments,
            classification: field("CLASSIFICATION").cloned(),
            creation_date,
            originator: field("ORIGINATOR")
                .ok_or(Error::MissingField("ORIGINATOR"))?
                .clone(),
            message_id: field("MESSAGE_ID").cloned(),
            segments,
        })
    }
}
