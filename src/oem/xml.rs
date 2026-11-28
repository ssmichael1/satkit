//! CCSDS NDM/XML encoding of the OEM (feature `omm-xml`).

use serde::Deserialize;

use super::{
    build_covariance, build_metadata, build_state, check_segment, parse_epoch, parse_f64, Error,
    Result, Segment, OEM,
};
use crate::TimeScale;

#[derive(Debug, Deserialize)]
struct XmlRoot {
    #[serde(rename = "oem", default)]
    oems: Vec<XmlMessage>,
}

#[derive(Debug, Deserialize)]
struct XmlMessage {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XmlHeader,
    body: XmlBody,
}

#[derive(Debug, Deserialize)]
struct XmlHeader {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "CLASSIFICATION")]
    classification: Option<String>,
    #[serde(rename = "CREATION_DATE")]
    creation_date: Option<String>,
    #[serde(rename = "ORIGINATOR")]
    originator: Option<String>,
    #[serde(rename = "MESSAGE_ID")]
    message_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct XmlBody {
    #[serde(rename = "segment", default)]
    segments: Vec<XmlSegment>,
}

#[derive(Debug, Deserialize)]
struct XmlSegment {
    metadata: XmlMetadata,
    data: XmlData,
}

#[derive(Debug, Deserialize)]
struct XmlMetadata {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "OBJECT_NAME")]
    object_name: Option<String>,
    #[serde(rename = "OBJECT_ID")]
    object_id: Option<String>,
    #[serde(rename = "CENTER_NAME")]
    center_name: Option<String>,
    #[serde(rename = "REF_FRAME")]
    ref_frame: Option<String>,
    #[serde(rename = "REF_FRAME_EPOCH")]
    ref_frame_epoch: Option<String>,
    #[serde(rename = "TIME_SYSTEM")]
    time_system: Option<String>,
    #[serde(rename = "START_TIME")]
    start_time: Option<String>,
    #[serde(rename = "USEABLE_START_TIME")]
    useable_start_time: Option<String>,
    #[serde(rename = "USEABLE_STOP_TIME")]
    useable_stop_time: Option<String>,
    #[serde(rename = "STOP_TIME")]
    stop_time: Option<String>,
    #[serde(rename = "INTERPOLATION")]
    interpolation: Option<String>,
    #[serde(rename = "INTERPOLATION_DEGREE")]
    interpolation_degree: Option<String>,
}

#[derive(Debug, Deserialize)]
struct XmlData {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "stateVector", default)]
    states: Vec<XmlState>,
    #[serde(rename = "covarianceMatrix", default)]
    covariances: Vec<XmlCovariance>,
}

/// The text of an element, whatever attributes (`units="km"`) it carries.
#[derive(Debug, Deserialize, Default)]
struct Text {
    #[serde(rename = "$text", default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct XmlState {
    #[serde(rename = "EPOCH", default)]
    epoch: Text,
    #[serde(rename = "X", default)]
    x: Text,
    #[serde(rename = "Y", default)]
    y: Text,
    #[serde(rename = "Z", default)]
    z: Text,
    #[serde(rename = "X_DOT", default)]
    x_dot: Text,
    #[serde(rename = "Y_DOT", default)]
    y_dot: Text,
    #[serde(rename = "Z_DOT", default)]
    z_dot: Text,
    #[serde(rename = "X_DDOT", default)]
    x_ddot: Text,
    #[serde(rename = "Y_DDOT", default)]
    y_ddot: Text,
    #[serde(rename = "Z_DDOT", default)]
    z_ddot: Text,
}

/// Element names of the lower triangle, row by row.
const COV_KEYS: [&str; 21] = [
    "CX_X",
    "CY_X",
    "CY_Y",
    "CZ_X",
    "CZ_Y",
    "CZ_Z",
    "CX_DOT_X",
    "CX_DOT_Y",
    "CX_DOT_Z",
    "CX_DOT_X_DOT",
    "CY_DOT_X",
    "CY_DOT_Y",
    "CY_DOT_Z",
    "CY_DOT_X_DOT",
    "CY_DOT_Y_DOT",
    "CZ_DOT_X",
    "CZ_DOT_Y",
    "CZ_DOT_Z",
    "CZ_DOT_X_DOT",
    "CZ_DOT_Y_DOT",
    "CZ_DOT_Z_DOT",
];

#[derive(Debug, Deserialize)]
struct XmlCovariance {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "EPOCH")]
    epoch: Option<String>,
    #[serde(rename = "COV_REF_FRAME")]
    ref_frame: Option<String>,
    /// The 21 `C*` elements, by name
    #[serde(flatten)]
    elements: std::collections::HashMap<String, Text>,
}

fn required<'a>(field: &'static str, text: &'a Option<String>) -> Result<&'a str> {
    text.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or(Error::MissingField(field))
}

fn required_text<'a>(field: &'static str, text: &'a Text) -> Result<&'a str> {
    required(field, &text.text)
}

fn segment(xml: XmlSegment) -> Result<Segment> {
    let m = xml.metadata;
    let metadata = build_metadata(
        |key| {
            match key {
                "OBJECT_NAME" => &m.object_name,
                "OBJECT_ID" => &m.object_id,
                "CENTER_NAME" => &m.center_name,
                "REF_FRAME" => &m.ref_frame,
                "REF_FRAME_EPOCH" => &m.ref_frame_epoch,
                "TIME_SYSTEM" => &m.time_system,
                "START_TIME" => &m.start_time,
                "USEABLE_START_TIME" => &m.useable_start_time,
                "USEABLE_STOP_TIME" => &m.useable_stop_time,
                "STOP_TIME" => &m.stop_time,
                "INTERPOLATION" => &m.interpolation,
                "INTERPOLATION_DEGREE" => &m.interpolation_degree,
                _ => &None,
            }
            .as_deref()
        },
        m.comments.clone(),
    )?;
    let scale = metadata.time_scale;

    let mut states = Vec::with_capacity(xml.data.states.len());
    for s in &xml.data.states {
        let mut v = vec![
            parse_f64("X", required_text("X", &s.x)?)?,
            parse_f64("Y", required_text("Y", &s.y)?)?,
            parse_f64("Z", required_text("Z", &s.z)?)?,
            parse_f64("X_DOT", required_text("X_DOT", &s.x_dot)?)?,
            parse_f64("Y_DOT", required_text("Y_DOT", &s.y_dot)?)?,
            parse_f64("Z_DOT", required_text("Z_DOT", &s.z_dot)?)?,
        ];
        if s.x_ddot.text.is_some() || s.y_ddot.text.is_some() || s.z_ddot.text.is_some() {
            v.push(parse_f64("X_DDOT", required_text("X_DDOT", &s.x_ddot)?)?);
            v.push(parse_f64("Y_DDOT", required_text("Y_DDOT", &s.y_ddot)?)?);
            v.push(parse_f64("Z_DDOT", required_text("Z_DDOT", &s.z_ddot)?)?);
        }
        states.push(build_state(required_text("EPOCH", &s.epoch)?, &v, scale)?);
    }

    let mut covariances = Vec::with_capacity(xml.data.covariances.len());
    for c in xml.data.covariances {
        let mut lower = Vec::with_capacity(21);
        for key in COV_KEYS {
            let text = c.elements.get(key).ok_or(Error::MissingField(key))?;
            lower.push(parse_f64(key, required_text(key, text)?)?);
        }
        covariances.push(build_covariance(
            required("EPOCH", &c.epoch)?,
            c.ref_frame.as_deref(),
            &lower,
            c.comments,
            scale,
        )?);
    }

    let segment = Segment {
        metadata,
        comments: xml.data.comments,
        states,
        covariances,
    };
    check_segment(&segment)?;
    Ok(segment)
}

impl TryFrom<XmlMessage> for OEM {
    type Error = Error;

    fn try_from(xml: XmlMessage) -> Result<Self> {
        let segments = xml
            .body
            .segments
            .into_iter()
            .map(segment)
            .collect::<Result<Vec<_>>>()?;
        if segments.is_empty() {
            return Err(Error::Empty("segments"));
        }
        let h = xml.header;
        let text = |s: Option<String>| s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        Ok(Self {
            version: text(xml.version).ok_or(Error::MissingField("CCSDS_OEM_VERS"))?,
            comments: h.comments,
            creation_date: parse_epoch(
                "CREATION_DATE",
                required("CREATION_DATE", &h.creation_date)?,
                TimeScale::UTC,
            )?,
            originator: required("ORIGINATOR", &h.originator)?.to_string(),
            classification: text(h.classification),
            message_id: text(h.message_id),
            segments,
        })
    }
}

impl OEM {
    /// Parse an OEM from XML text: a bare `<oem>` message, or the one
    /// `<oem>` inside an `<ndm>` wrapper.
    ///
    /// Available only when the `omm-xml` feature is enabled.
    ///
    /// # Errors
    ///
    /// An error if the XML is malformed, a mandatory element is missing, a
    /// value cannot be read, or an `<ndm>` holds no `<oem>` or several.
    pub fn from_xml_string(s: &str) -> Result<Self> {
        if s.contains("<ndm") {
            let root: XmlRoot = quick_xml::de::from_str(s)?;
            let mut oems = root.oems;
            match oems.len() {
                1 => Self::try_from(oems.remove(0)),
                0 => Err(Error::Empty("<oem> message in the <ndm>")),
                n => Err(Error::InvalidField {
                    field: "ndm",
                    message: format!("holds {n} <oem> messages; expected one"),
                }),
            }
        } else {
            let msg: XmlMessage = quick_xml::de::from_str(s)?;
            Self::try_from(msg)
        }
    }
}
