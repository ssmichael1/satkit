//! Orbital Mean-Element Messages (OMM)
//!
//! An OMM carries one SGP4 mean-element set plus catalog metadata, as
//! specified in CCSDS 502.0-B-3 (Orbit Data Messages), Tables 4-1 to 4-3:
//! <https://ccsds.org/Pubs/502x0b3e1.pdf>. Space-Track and CelesTrak publish
//! OMMs in JSON and XML: <https://www.space-track.org/documentation#/omm>.
//!
//! Real-world producers deviate from the standard: numeric fields may arrive
//! as quoted strings, mandatory metadata (`CCSDS_OMM_VERS`, `CENTER_NAME`,
//! `REF_FRAME`, `TIME_SYSTEM`, `MEAN_ELEMENT_THEORY`) is often omitted, and
//! catalog-specific extras (`OBJECT_TYPE`, `RCS_SIZE`, `TLE_LINE1`, ...) are
//! appended. The parser accepts all of that: numbers are read from either
//! JSON numbers or strings, the metadata fields are optional, and unknown
//! keys are kept verbatim in [`OMM::extra_fields`].
//!
//! [`OMM`] implements [`SGP4Source`], so an OMM can be handed directly to
//! [`sgp4`](crate::sgp4::sgp4). It converts to and from [`TLE`] with
//! [`OMM::from_tle`] and [`OMM::to_tle`].

use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

mod error;

pub use error::{Error, Result};

#[cfg(feature = "omm-xml")]
mod xml;

use crate::sgp4::{SGP4InitArgs, SGP4Source, SatRec};
use crate::tle::TLE;
use crate::{Instant, TimeScale};

// ---------------------------------------------------------------------------
// Field parsing shared by the JSON (serde) and XML paths
// ---------------------------------------------------------------------------

/// Parse an optional textual field. `None`, empty, and whitespace-only text
/// all read as "absent".
pub(crate) fn parse_opt_field<T>(field: &'static str, text: Option<&str>) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: Display,
{
    match text.map(str::trim) {
        None | Some("") => Ok(None),
        Some(s) => s.parse::<T>().map(Some).map_err(|e| Error::InvalidField {
            field,
            message: format!("cannot parse {s:?}: {e}"),
        }),
    }
}

/// Parse a mandatory textual field; absent or empty text is an error.
pub(crate) fn parse_req_field<T>(field: &'static str, text: Option<&str>) -> Result<T>
where
    T: FromStr,
    T::Err: Display,
{
    parse_opt_field(field, text)?.ok_or(Error::MissingField(field))
}

/// Text of a JSON scalar: numbers become their decimal text, strings are
/// trimmed, `null` and empty strings read as absent.
fn value_text(v: &Value) -> std::result::Result<Option<String>, String> {
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::String(s) => {
            let s = s.trim();
            Ok((!s.is_empty()).then(|| s.to_string()))
        }
        Value::Bool(_) => Err("expected a number or string, found a boolean".into()),
        Value::Array(_) => Err("expected a number or string, found an array".into()),
        Value::Object(_) => Err("expected a number or string, found an object".into()),
    }
}

/// serde adapter: optional numeric field given as a JSON number, a quoted
/// number, `null`, or an empty string.
fn de_opt<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: Display,
{
    use serde::de::Error as _;
    let v = Value::deserialize(deserializer)?;
    match value_text(&v).map_err(D::Error::custom)? {
        None => Ok(None),
        Some(s) => s
            .parse::<T>()
            .map(Some)
            .map_err(|e| D::Error::custom(format!("cannot parse {s:?}: {e}"))),
    }
}

/// serde adapter: mandatory numeric field, same tolerance as [`de_opt`].
fn de_req<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: Display,
{
    use serde::de::Error as _;
    de_opt(deserializer)?.ok_or_else(|| D::Error::custom("value is null or empty"))
}

fn de_epoch<'de, D>(deserializer: D) -> std::result::Result<Instant, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as _;
    let s = String::deserialize(deserializer)?;
    Instant::from_rfc3339(s.trim()).map_err(|e| D::Error::custom(format!("EPOCH {s:?}: {e}")))
}

fn unknown() -> String {
    "UNKNOWN".to_string()
}

fn ser_epoch<S>(epoch: &Instant, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&epoch.as_rfc3339())
}

// ---------------------------------------------------------------------------
// The message
// ---------------------------------------------------------------------------

/// A CCSDS Orbital Mean-Element Message.
///
/// Field names follow CCSDS 502.0-B-3 Tables 4-1 to 4-3; the serde renames
/// give the exact JSON keys. Angles are in degrees, mean motion in rev/day,
/// its derivatives in rev/day² and rev/day³ (the TLE convention, i.e. the
/// values are ṅ/2 and n̈/6), and `bstar` in inverse Earth radii — the
/// same units as [`TLE`].
///
/// Only the six mean elements and `EPOCH` are required to parse; the
/// standard's other mandatory fields are optional here because Space-Track
/// and CelesTrak omit some of them and trimmed-down records omit more. Keys that are
/// not part of the structure are kept in [`extra_fields`](Self::extra_fields)
/// and written back out by [`Serialize`].
///
/// # Propagation
///
/// `OMM` implements [`SGP4Source`]. The SGP4 initialization is cached in the
/// struct after the first propagation; if you change any element afterwards,
/// call [`reset_cache`](Self::reset_cache) so the next propagation reinitializes.
/// Propagation refuses a message whose `MEAN_ELEMENT_THEORY` is not SGP4,
/// whose `TIME_SYSTEM` is not UTC, or whose `EPHEMERIS_TYPE` is 4 (SGP4-XP).
///
/// # Example
///
/// ```
/// use satkit::prelude::*;
///
/// // A CelesTrak-style JSON record (Space-Track quotes its numbers; that works too)
/// let json = r#"{
///     "OBJECT_NAME": "ISS (ZARYA)",
///     "OBJECT_ID": "1998-067A",
///     "EPOCH": "2026-02-14T05:08:48.534432",
///     "MEAN_MOTION": 15.4859353,
///     "ECCENTRICITY": 0.00110623,
///     "INCLINATION": 51.6315,
///     "RA_OF_ASC_NODE": 188.3997,
///     "ARG_OF_PERICENTER": 96.9141,
///     "MEAN_ANOMALY": 263.3106,
///     "NORAD_CAT_ID": 25544,
///     "BSTAR": 0.00016303535,
///     "MEAN_MOTION_DOT": 8.429e-5,
///     "MEAN_MOTION_DDOT": 0
/// }"#;
///
/// let mut omm = OMM::from_json_string(json).unwrap().remove(0);
/// let times = [omm.epoch, omm.epoch + Duration::from_minutes(10.0)];
/// let states = sgp4(&mut omm, &times).unwrap();
/// assert_eq!(states.pos.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OMM {
    /// `CCSDS_OMM_VERS`. Mandatory in the standard, usually absent in practice.
    #[serde(rename = "CCSDS_OMM_VERS", skip_serializing_if = "Option::is_none")]
    pub omm_version: Option<String>,
    /// `COMMENT`. Multiple XML comment lines are joined with newlines.
    #[serde(rename = "COMMENT", skip_serializing_if = "Option::is_none")]
    pub comments: Option<String>,
    /// `ORIGINATOR`
    #[serde(rename = "ORIGINATOR", skip_serializing_if = "Option::is_none")]
    pub originator: Option<String>,
    /// `CLASSIFICATION` (message classification, not the TLE `CLASSIFICATION_TYPE`)
    #[serde(rename = "CLASSIFICATION", skip_serializing_if = "Option::is_none")]
    pub classification: Option<String>,
    /// `MESSAGE_ID`
    #[serde(rename = "MESSAGE_ID", skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// `OBJECT_NAME`. Mandatory in the standard; `UNKNOWN` when absent.
    #[serde(rename = "OBJECT_NAME", default = "unknown")]
    pub object_name: String,
    /// `OBJECT_ID`, the international designator as `YYYY-NNNP`. Mandatory
    /// in the standard; `UNKNOWN` when absent.
    #[serde(rename = "OBJECT_ID", default = "unknown")]
    pub object_id: String,
    /// `CENTER_NAME`, expected `EARTH`
    #[serde(rename = "CENTER_NAME", skip_serializing_if = "Option::is_none")]
    pub center_name: Option<String>,
    /// `REF_FRAME`, expected `TEME` for SGP4 elements
    #[serde(rename = "REF_FRAME", skip_serializing_if = "Option::is_none")]
    pub reference_frame: Option<String>,
    /// `REF_FRAME_EPOCH`
    #[serde(rename = "REF_FRAME_EPOCH", skip_serializing_if = "Option::is_none")]
    pub reference_frame_epoch: Option<String>,
    /// `TIME_SYSTEM`; propagation requires `UTC` when present
    #[serde(rename = "TIME_SYSTEM", skip_serializing_if = "Option::is_none")]
    pub time_system: Option<String>,
    /// `MEAN_ELEMENT_THEORY`; propagation requires `SGP4` when present
    #[serde(
        rename = "MEAN_ELEMENT_THEORY",
        skip_serializing_if = "Option::is_none"
    )]
    pub mean_element_theory: Option<String>,
    /// `EPOCH` (mandatory), UTC. Read from and written as an RFC 3339 string.
    #[serde(
        rename = "EPOCH",
        deserialize_with = "de_epoch",
        serialize_with = "ser_epoch"
    )]
    pub epoch: Instant,
    /// `MEAN_MOTION` (mandatory), rev/day
    #[serde(rename = "MEAN_MOTION", deserialize_with = "de_req")]
    pub mean_motion: f64,
    /// `ECCENTRICITY` (mandatory)
    #[serde(rename = "ECCENTRICITY", deserialize_with = "de_req")]
    pub eccentricity: f64,
    /// `INCLINATION` (mandatory), degrees
    #[serde(rename = "INCLINATION", deserialize_with = "de_req")]
    pub inclination: f64,
    /// `RA_OF_ASC_NODE` (mandatory), degrees
    #[serde(rename = "RA_OF_ASC_NODE", deserialize_with = "de_req")]
    pub raan: f64,
    /// `ARG_OF_PERICENTER` (mandatory), degrees
    #[serde(rename = "ARG_OF_PERICENTER", deserialize_with = "de_req")]
    pub arg_of_pericenter: f64,
    /// `MEAN_ANOMALY` (mandatory), degrees
    #[serde(rename = "MEAN_ANOMALY", deserialize_with = "de_req")]
    pub mean_anomaly: f64,
    /// `GM`, km³/s²
    #[serde(
        rename = "GM",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub gm: Option<f64>,
    /// `MASS`, kg
    #[serde(
        rename = "MASS",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub mass: Option<f64>,
    /// `SOLAR_RAD_AREA`, m²
    #[serde(
        rename = "SOLAR_RAD_AREA",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub solar_rad_area: Option<f64>,
    /// `DRAG_AREA`, m²
    #[serde(
        rename = "DRAG_AREA",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub drag_area: Option<f64>,
    /// `SOLAR_RAD_COEFF`
    #[serde(
        rename = "SOLAR_RAD_COEFF",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub solar_rad_coeff: Option<f64>,
    /// `DRAG_COEFF`
    #[serde(
        rename = "DRAG_COEFF",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub drag_coeff: Option<f64>,
    /// `EPHEMERIS_TYPE`: 0 for SGP4, 4 for SGP4-XP (which satkit cannot propagate)
    #[serde(
        rename = "EPHEMERIS_TYPE",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub ephemeris_type: Option<u8>,
    /// `CLASSIFICATION_TYPE`: `U`, `C`, or `S`
    #[serde(
        rename = "CLASSIFICATION_TYPE",
        skip_serializing_if = "Option::is_none"
    )]
    pub classification_type: Option<String>,
    /// `NORAD_CAT_ID`
    #[serde(
        rename = "NORAD_CAT_ID",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub norad_cat_id: Option<u32>,
    /// `ELEMENT_SET_NO`
    #[serde(
        rename = "ELEMENT_SET_NO",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub element_set_no: Option<u32>,
    /// `REV_AT_EPOCH`
    #[serde(
        rename = "REV_AT_EPOCH",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub rev_at_epoch: Option<u32>,
    /// `BSTAR`, inverse Earth radii. Absent reads as zero for propagation.
    #[serde(
        rename = "BSTAR",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub bstar: Option<f64>,
    /// `BTERM`, m²/kg (SGP4-XP ballistic coefficient)
    #[serde(
        rename = "BTERM",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub bterm: Option<f64>,
    /// `MEAN_MOTION_DOT`, rev/day² (TLE convention: ṅ/2). Absent reads as zero.
    #[serde(
        rename = "MEAN_MOTION_DOT",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub mean_motion_dot: Option<f64>,
    /// `MEAN_MOTION_DDOT`, rev/day³ (TLE convention: n̈/6). Absent reads as zero.
    #[serde(
        rename = "MEAN_MOTION_DDOT",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub mean_motion_ddot: Option<f64>,
    /// `AGOM`, m²/kg (SGP4-XP solar radiation pressure coefficient)
    #[serde(
        rename = "AGOM",
        default,
        deserialize_with = "de_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub agom: Option<f64>,

    /// Cached SGP4 record, initialized lazily on first propagation.
    #[serde(skip)]
    pub(crate) satrec: Option<SatRec>,

    /// Every key that is not one of the fields above, verbatim. Space-Track
    /// puts `OBJECT_TYPE`, `RCS_SIZE`, `LAUNCH_DATE`, `TLE_LINE1`, ... here;
    /// XML `USER_DEFINED` parameters land here as strings.
    #[serde(flatten)]
    pub extra_fields: HashMap<String, Value>,
}

impl OMM {
    /// Builds an SGP4 message from the six mean elements.
    ///
    /// Metadata is filled in for an Earth-centered SGP4 set (`CENTER_NAME`
    /// `EARTH`, `REF_FRAME` `TEME`, `TIME_SYSTEM` `UTC`, `MEAN_ELEMENT_THEORY`
    /// `SGP4`); `OBJECT_NAME` and `OBJECT_ID` are set to `UNKNOWN`, and every
    /// other optional field is `None`. Set `bstar`, `mean_motion_dot`, and
    /// the catalog identifiers afterwards as needed.
    ///
    /// # Arguments
    /// - `epoch`: element-set epoch (UTC)
    /// - `mean_motion`: rev/day
    /// - `eccentricity`: unitless
    /// - `inclination`, `raan`, `arg_of_pericenter`, `mean_anomaly`: degrees
    pub fn from_mean_elements(
        epoch: Instant,
        mean_motion: f64,
        eccentricity: f64,
        inclination: f64,
        raan: f64,
        arg_of_pericenter: f64,
        mean_anomaly: f64,
    ) -> Self {
        Self {
            omm_version: None,
            comments: None,
            originator: None,
            classification: None,
            message_id: None,
            object_name: "UNKNOWN".to_string(),
            object_id: "UNKNOWN".to_string(),
            center_name: Some("EARTH".to_string()),
            reference_frame: Some("TEME".to_string()),
            reference_frame_epoch: None,
            time_system: Some("UTC".to_string()),
            mean_element_theory: Some("SGP4".to_string()),
            epoch,
            mean_motion,
            eccentricity,
            inclination,
            raan,
            arg_of_pericenter,
            mean_anomaly,
            gm: None,
            mass: None,
            solar_rad_area: None,
            drag_area: None,
            solar_rad_coeff: None,
            drag_coeff: None,
            ephemeris_type: None,
            classification_type: None,
            norad_cat_id: None,
            element_set_no: None,
            rev_at_epoch: None,
            bstar: None,
            bterm: None,
            mean_motion_dot: None,
            mean_motion_ddot: None,
            agom: None,
            satrec: None,
            extra_fields: HashMap::new(),
        }
    }

    /// Builds an OMM from a [`TLE`].
    ///
    /// The mean elements, epoch, drag terms, and catalog numbers copy across
    /// directly (both types use the same units). The TLE's two-digit
    /// international designator (`98067A`) becomes the CCSDS `OBJECT_ID`
    /// (`1998-067A`); an empty designator becomes `UNKNOWN`, as does an empty
    /// or placeholder (`none`) name. TLEs do not carry the classification
    /// letter, so `classification_type` is `None`.
    pub fn from_tle(tle: &TLE) -> Self {
        let mut omm = Self::from_mean_elements(
            tle.epoch,
            tle.mean_motion,
            tle.eccen,
            tle.inclination,
            tle.raan,
            tle.arg_of_perigee,
            tle.mean_anomaly,
        );
        // TLE::new() and 2-line parsing leave the name as "none"
        omm.object_name = match tle.name.trim() {
            "" | "none" => unknown(),
            name => name.to_string(),
        };
        omm.object_id = object_id_from_intl_desig(&tle.intl_desig);
        omm.bstar = Some(tle.bstar);
        omm.mean_motion_dot = Some(tle.mean_motion_dot);
        omm.mean_motion_ddot = Some(tle.mean_motion_dot_dot);
        // TLE::new() leaves the field as b'U'; only a real digit is meaningful.
        omm.ephemeris_type = (tle.ephem_type <= 9).then_some(tle.ephem_type);
        omm.norad_cat_id = u32::try_from(tle.sat_num).ok();
        omm.element_set_no = u32::try_from(tle.element_num).ok();
        omm.rev_at_epoch = u32::try_from(tle.rev_num).ok();
        omm
    }

    /// Converts this message to a [`TLE`].
    ///
    /// Inverse of [`from_tle`](Self::from_tle). `OBJECT_ID` in `YYYY-NNNP`
    /// form becomes the TLE international designator; any other form leaves
    /// the designator empty. Absent optional fields become zero. Metadata
    /// that has no TLE column (`ORIGINATOR`, `COMMENT`, `extra_fields`, ...)
    /// is dropped. The result is a plain element set: it does not check
    /// `EPHEMERIS_TYPE` or `MEAN_ELEMENT_THEORY`, so an SGP4-XP message
    /// converts, and the resulting TLE keeps its ephemeris type 4.
    pub fn to_tle(&self) -> TLE {
        let mut tle = TLE::new();
        tle.name = self.object_name.clone();
        if let Some((desig, year, launch, piece)) = intl_desig_from_object_id(&self.object_id) {
            tle.intl_desig = desig;
            tle.desig_year = year;
            tle.desig_launch = launch;
            tle.desig_piece = piece;
        } else {
            tle.intl_desig = String::new();
        }
        tle.sat_num = self
            .norad_cat_id
            .and_then(|n| i32::try_from(n).ok())
            .unwrap_or(0);
        tle.epoch = self.epoch;
        tle.mean_motion_dot = self.mean_motion_dot.unwrap_or(0.0);
        tle.mean_motion_dot_dot = self.mean_motion_ddot.unwrap_or(0.0);
        tle.bstar = self.bstar.unwrap_or(0.0);
        tle.ephem_type = self.ephemeris_type.unwrap_or(0);
        tle.element_num = self
            .element_set_no
            .and_then(|n| i32::try_from(n).ok())
            .unwrap_or(0);
        tle.inclination = self.inclination;
        tle.raan = self.raan;
        tle.eccen = self.eccentricity;
        tle.arg_of_perigee = self.arg_of_pericenter;
        tle.mean_anomaly = self.mean_anomaly;
        tle.mean_motion = self.mean_motion;
        tle.rev_num = self
            .rev_at_epoch
            .and_then(|n| i32::try_from(n).ok())
            .unwrap_or(0);
        tle
    }

    /// Discards the cached SGP4 initialization.
    ///
    /// Call this after editing `epoch` or any mean element of a message that
    /// has already been propagated; otherwise the next propagation reuses the
    /// initialization computed from the old values.
    pub fn reset_cache(&mut self) {
        self.satrec = None;
    }

    /// The element-set epoch.
    ///
    /// Kept for compatibility with versions where `epoch` was a string; it
    /// cannot fail any more.
    #[deprecated(since = "0.22.0", note = "read the `epoch` field directly")]
    pub fn epoch_instant(&self) -> Result<Instant> {
        Ok(self.epoch)
    }

    /// Deserializes one or more OMM records from an already-parsed JSON value.
    ///
    /// Accepts a single OMM object or an array of them.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is neither an object nor an array, or if
    /// a record is missing mandatory fields or has unparsable values.
    pub fn from_json_value(value: Value) -> Result<Vec<Self>> {
        match value {
            Value::Array(items) => items
                .into_iter()
                .map(|v| serde_json::from_value(v).map_err(Error::from))
                .collect(),
            Value::Object(_) => Ok(vec![serde_json::from_value(value)?]),
            Value::Null => Err(Error::UnexpectedJsonShape("null")),
            Value::Bool(_) => Err(Error::UnexpectedJsonShape("a boolean")),
            Value::Number(_) => Err(Error::UnexpectedJsonShape("a number")),
            Value::String(_) => Err(Error::UnexpectedJsonShape("a string")),
        }
    }

    /// Deserializes one or more OMM records from a JSON string.
    ///
    /// Accepts a single OMM object or an array of them, as returned by the
    /// Space-Track and CelesTrak JSON endpoints.
    ///
    /// # Examples
    ///
    /// ```
    /// use satkit::prelude::OMM;
    ///
    /// let json = r#"[
    ///   {
    ///     "OBJECT_NAME": "ISS (ZARYA)",
    ///     "OBJECT_ID": "1998-067A",
    ///     "EPOCH": "2026-02-14T05:08:48.534432",
    ///     "MEAN_MOTION": 15.4859353,
    ///     "ECCENTRICITY": 0.00110623,
    ///     "INCLINATION": 51.6315,
    ///     "RA_OF_ASC_NODE": 188.3997,
    ///     "ARG_OF_PERICENTER": 96.9141,
    ///     "MEAN_ANOMALY": 263.3106
    ///   }
    /// ]"#;
    ///
    /// let omms = OMM::from_json_string(json)?;
    /// assert_eq!(omms.len(), 1);
    /// assert_eq!(omms[0].object_id, "1998-067A");
    /// # Ok::<(), satkit::omm::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed or required OMM fields are missing/invalid.
    pub fn from_json_string(s: &str) -> Result<Vec<Self>> {
        Self::from_json_value(serde_json::from_str(s)?)
    }

    /// Deserializes one or more OMM records from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON payload is invalid.
    pub fn from_json_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Self::from_json_value(serde_json::from_reader(reader)?)
    }

    /// Deserializes OMM records from text, detecting the format.
    ///
    /// Text starting with `[` or `{` is parsed as JSON, text starting with
    /// `<` as XML (requires the `omm-xml` feature). KVN is not supported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnrecognizedFormat`] for anything else, and the
    /// format parser's error for malformed input.
    pub fn from_text(s: &str) -> Result<Vec<Self>> {
        let trimmed = s.trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n']);
        if trimmed.starts_with('[') || trimmed.starts_with('{') {
            Self::from_json_string(trimmed)
        } else if trimmed.starts_with('<') {
            #[cfg(feature = "omm-xml")]
            {
                Self::from_xml_string(trimmed)
            }
            #[cfg(not(feature = "omm-xml"))]
            {
                Err(Error::XmlFeatureDisabled)
            }
        } else {
            Err(Error::UnrecognizedFormat)
        }
    }

    /// Deserializes OMM records from a JSON or XML file, detecting the format
    /// from the content (not the extension) as [`from_text`](Self::from_text).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or its content is not a
    /// valid OMM document.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>> {
        Self::from_text(&std::fs::read_to_string(path)?)
    }

    /// Load OMM(s) from a URL, detecting JSON vs XML from the response body.
    ///
    /// Works with CelesTrak and Space-Track API endpoints.
    ///
    /// Requires the `download` Cargo feature.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use satkit::omm::OMM;
    ///
    /// # #[cfg(feature = "download")]
    /// let omms = OMM::from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json").unwrap();
    /// ```
    #[cfg(feature = "download")]
    pub fn from_url(url: &str) -> Result<Vec<Self>> {
        let agent = crate::utils::download::http_agent();
        let mut resp =
            agent.get(url).call().map_err(
                |e| match crate::utils::download::celestrak_throttle_hint(url, &e) {
                    Some(msg) => Error::HttpThrottled(msg),
                    None => Error::Http(e),
                },
            )?;
        let body = resp.body_mut().read_to_string()?;
        Self::from_text(&body)
    }
}

impl From<&TLE> for OMM {
    fn from(tle: &TLE) -> Self {
        Self::from_tle(tle)
    }
}

impl From<&OMM> for TLE {
    fn from(omm: &OMM) -> Self {
        omm.to_tle()
    }
}

/// `98067A` → `1998-067A`. Empty input → `UNKNOWN`; anything that is not
/// `YYNNNP` is returned unchanged.
fn object_id_from_intl_desig(desig: &str) -> String {
    let desig = desig.trim();
    if desig.is_empty() {
        return "UNKNOWN".to_string();
    }
    if desig.len() >= 6 && desig.is_char_boundary(2) && desig.is_char_boundary(5) {
        if let (Ok(yy), Ok(launch)) = (desig[..2].parse::<u32>(), desig[2..5].parse::<u32>()) {
            // Same century rule as TLE epochs: 57-99 → 1900s, 00-56 → 2000s.
            let year = if yy >= 57 { 1900 + yy } else { 2000 + yy };
            return format!("{year}-{launch:03}{}", &desig[5..]);
        }
    }
    desig.to_string()
}

/// `1998-067A` → (`98067A`, 98, 67, `A`). `None` if the id is not `YYYY-NNNP`.
fn intl_desig_from_object_id(id: &str) -> Option<(String, i32, i32, String)> {
    let id = id.trim();
    let (year, rest) = id.split_once('-')?;
    if year.len() != 4 || rest.len() < 4 || !rest.is_char_boundary(3) {
        return None;
    }
    let year: i32 = year.parse().ok()?;
    let launch: i32 = rest[..3].parse().ok()?;
    let piece = rest[3..].trim();
    if piece.is_empty() || !piece.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let yy = year.rem_euclid(100);
    Some((
        format!("{yy:02}{launch:03}{piece}"),
        yy,
        launch,
        piece.to_string(),
    ))
}

impl SGP4Source for OMM {
    fn epoch(&self) -> Instant {
        self.epoch
    }

    fn satrec_mut(&mut self) -> &mut Option<SatRec> {
        &mut self.satrec
    }

    fn sgp4_init_args(&self) -> crate::sgp4::Result<SGP4InitArgs> {
        if let Some(theory) = &self.mean_element_theory {
            if !theory.trim().eq_ignore_ascii_case("SGP4") {
                return Err(crate::sgp4::Error::source(
                    Error::UnsupportedMeanElementTheory(theory.clone()),
                ));
            }
        }

        if let Some(ts) = &self.time_system {
            if !ts.trim().eq_ignore_ascii_case("UTC") {
                return Err(crate::sgp4::Error::source(Error::UnsupportedTimeSystem(
                    ts.clone(),
                )));
            }
        }

        // Space-Track distributes SGP4-XP element sets with EPHEMERIS_TYPE 4
        // and MEAN_ELEMENT_THEORY still "SGP4"; classic SGP4 produces garbage
        // from them, so refuse rather than propagate silently.
        if let Some(4) = self.ephemeris_type {
            return Err(crate::sgp4::Error::source(Error::UnsupportedEphemerisType(
                4,
            )));
        }

        Ok(SGP4InitArgs::from_mean_elements(
            self.epoch.as_jd_with_scale(TimeScale::UTC),
            self.bstar.unwrap_or(0.0),
            self.mean_motion,
            self.mean_motion_dot.unwrap_or(0.0),
            self.mean_motion_ddot.unwrap_or(0.0),
            self.eccentricity,
            self.inclination,
            self.raan,
            self.arg_of_pericenter,
            self.mean_anomaly,
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::sgp4::{sgp4_full, GravConst, OpsMode, SGP4Error};
    use crate::time::Duration;
    use crate::utils::test::*;

    fn iss_json() -> &'static str {
        r#"{
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "EPOCH": "2026-02-14T05:08:48.534432",
            "MEAN_MOTION": 15.4859353,
            "ECCENTRICITY": 0.00110623,
            "INCLINATION": 51.6315,
            "RA_OF_ASC_NODE": 188.3997,
            "ARG_OF_PERICENTER": 96.9141,
            "MEAN_ANOMALY": 263.3106,
            "EPHEMERIS_TYPE": 0,
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 55269,
            "BSTAR": 0.00016303535,
            "MEAN_MOTION_DOT": 8.429e-5,
            "MEAN_MOTION_DDOT": 0,
            "OBJECT_TYPE": "PAYLOAD",
            "RCS_SIZE": null
        }"#
    }

    fn col3(m: &crate::mathtypes::DMatrix<f64>, c: usize) -> [f64; 3] {
        [m[(0, c)], m[(1, c)], m[(2, c)]]
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn propagate(omm: &mut OMM, times: &[Instant]) -> crate::sgp4::SGP4State {
        let states = sgp4_full(omm, times, GravConst::WGS72, OpsMode::IMPROVED).unwrap();
        for e in &states.errcode {
            assert_eq!(*e, SGP4Error::SGP4Success);
        }
        states
    }

    #[test]
    fn test_parse_omm_spacetrack_json() {
        let filename = get_testvec_dir().unwrap().join("omm/spacetrack_omm.json");
        let msg = OMM::from_json_file(filename).unwrap();
        assert!(msg.len() > 100);

        // Space-Track quotes every number; strings must have parsed to numbers
        let first = &msg[0];
        assert!(first.mean_motion > 0.0);
        assert_eq!(first.ephemeris_type, Some(0));
        assert_eq!(first.mean_element_theory.as_deref(), Some("SGP4"));
        // Space-Track extras are kept verbatim
        assert!(first.extra_fields.contains_key("OBJECT_TYPE"));
        assert!(first.extra_fields["TLE_LINE1"].is_string());

        let mut omm = first.clone();
        let times = vec![omm.epoch, omm.epoch + Duration::from_minutes(10.0)];
        propagate(&mut omm, &times);
    }

    /// Space-Track ships the TLE lines alongside each OMM: the same element
    /// set through both parsers must give the same SGP4 state. The TLE epoch
    /// has 0.864 ms resolution (8 decimals of a day) versus the OMM's 1 µs,
    /// so allow a few meters of along-track difference.
    #[test]
    fn test_omm_matches_tle_lines() {
        let filename = get_testvec_dir().unwrap().join("omm/spacetrack_omm.json");
        let msg = OMM::from_json_file(filename).unwrap();
        let mut checked = 0;
        for omm in msg.iter().take(200) {
            let l1 = omm.extra_fields["TLE_LINE1"].as_str().unwrap();
            let l2 = omm.extra_fields["TLE_LINE2"].as_str().unwrap();
            let mut tle = TLE::load_2line(l1, l2).unwrap();
            assert!((tle.epoch - omm.epoch).as_seconds().abs() < 1e-3);
            assert_eq!(tle.sat_num as u32, omm.norad_cat_id.unwrap());

            let times = vec![omm.epoch, omm.epoch + Duration::from_hours(3.0)];
            let mut omm = omm.clone();
            let so = sgp4_full(&mut omm, &times, GravConst::WGS72, OpsMode::AFSPC).unwrap();
            let st = sgp4_full(&mut tle, &times, GravConst::WGS72, OpsMode::AFSPC).unwrap();
            if so.errcode.iter().any(|e| *e != SGP4Error::SGP4Success) {
                continue; // decayed or otherwise unpropagatable in both
            }
            for c in 0..times.len() {
                let dp = dist(col3(&so.pos, c), col3(&st.pos, c));
                assert!(dp < 10.0, "{}: pos differs by {dp} m", omm.object_name);
            }
            checked += 1;
        }
        assert!(checked > 100);
    }

    #[test]
    fn test_parse_omm_celestrak_json() {
        let filename = get_testvec_dir().unwrap().join("omm/celestrak_omm.json");
        let msg = OMM::from_json_file(filename).unwrap();
        assert!(msg.len() > 100);
        assert!(msg[0].extra_fields.is_empty());
        assert_eq!(msg[0].norad_cat_id, Some(694));
    }

    #[test]
    fn test_bare_object_and_array_parse_the_same() {
        let one = OMM::from_json_string(iss_json()).unwrap();
        let many = OMM::from_json_string(&format!("[{}]", iss_json())).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(many.len(), 1);
        assert_eq!(one[0].mean_motion, many[0].mean_motion);
        assert_eq!(one[0].extra_fields["OBJECT_TYPE"], Value::from("PAYLOAD"));
        assert_eq!(one[0].extra_fields["RCS_SIZE"], Value::Null);

        assert!(matches!(
            OMM::from_json_string("42"),
            Err(Error::UnexpectedJsonShape(_))
        ));
        assert!(matches!(
            OMM::from_text("MEAN_MOTION = 1"),
            Err(Error::UnrecognizedFormat)
        ));
    }

    #[test]
    fn test_from_text_detects_json() {
        let omms = OMM::from_text(&format!("\n  {}", iss_json())).unwrap();
        assert_eq!(omms[0].norad_cat_id, Some(25544));
    }

    #[test]
    fn test_tolerant_scalars() {
        // quoted numbers, empty strings and nulls for optionals, whitespace
        let json = r#"{
            "OBJECT_NAME": "X", "OBJECT_ID": "UNKNOWN",
            "EPOCH": " 2026-02-14T05:08:48Z ",
            "MEAN_MOTION": "15.4859353", "ECCENTRICITY": " 0.001 ",
            "INCLINATION": 51.6315, "RA_OF_ASC_NODE": "188.3997",
            "ARG_OF_PERICENTER": 96.9141, "MEAN_ANOMALY": "263.3106",
            "BSTAR": "", "MEAN_MOTION_DOT": null, "NORAD_CAT_ID": "25544",
            "EPHEMERIS_TYPE": ""
        }"#;
        let omm = OMM::from_json_string(json).unwrap().remove(0);
        assert_eq!(omm.mean_motion, 15.4859353);
        assert_eq!(omm.bstar, None);
        assert_eq!(omm.mean_motion_dot, None);
        assert_eq!(omm.ephemeris_type, None);
        assert_eq!(omm.norad_cat_id, Some(25544));

        // a required element that is empty is an error, not zero
        let bad = json.replace("\"15.4859353\"", "\"\"");
        assert!(OMM::from_json_string(&bad).is_err());
        // and so is a non-numeric one
        let bad = json.replace("\"15.4859353\"", "\"fast\"");
        let err = OMM::from_json_string(&bad).unwrap_err().to_string();
        assert!(err.contains("fast"), "{err}");
    }

    #[test]
    fn test_rejects_sgp4_xp_and_foreign_metadata() {
        let mut omm = OMM::from_json_string(iss_json()).unwrap().remove(0);
        let times = vec![omm.epoch];

        omm.ephemeris_type = Some(4);
        let err = sgp4_full(&mut omm, &times, GravConst::WGS72, OpsMode::AFSPC)
            .err()
            .unwrap();
        assert!(err.to_string().contains("EPHEMERIS_TYPE 4"), "{err}");
        omm.ephemeris_type = Some(0);

        omm.mean_element_theory = Some("DSST".into());
        assert!(sgp4_full(&mut omm, &times, GravConst::WGS72, OpsMode::AFSPC).is_err());
        omm.mean_element_theory = Some(" sgp4 ".into()); // trimmed, case-insensitive
        propagate(&mut omm, &times);

        // Validation happens at initialization, which is now cached
        omm.time_system = Some("TAI".into());
        propagate(&mut omm, &times);
        omm.reset_cache();
        assert!(sgp4_full(&mut omm, &times, GravConst::WGS72, OpsMode::AFSPC).is_err());
    }

    #[test]
    fn test_serialize_round_trip() {
        let omm = OMM::from_json_string(iss_json()).unwrap().remove(0);
        let text = serde_json::to_string(&omm).unwrap();
        let back = OMM::from_json_string(&text).unwrap().remove(0);
        assert_eq!(back.epoch, omm.epoch);
        assert_eq!(back.mean_motion, omm.mean_motion);
        assert_eq!(back.bstar, omm.bstar);
        assert_eq!(back.norad_cat_id, omm.norad_cat_id);
        assert_eq!(back.extra_fields, omm.extra_fields);

        // absent optionals are not written out; the epoch is an RFC 3339 string
        let v: Value = serde_json::from_str(&text).unwrap();
        assert!(v.get("GM").is_none());
        assert_eq!(v["EPOCH"], Value::from("2026-02-14T05:08:48.534432Z"));
        assert_eq!(v["OBJECT_TYPE"], Value::from("PAYLOAD"));
    }

    #[test]
    fn test_tle_conversion_round_trip() {
        let l1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003";
        let l2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357";
        let tle = TLE::load_2line(l1, l2).unwrap();

        let omm = OMM::from_tle(&tle);
        assert_eq!(omm.object_id, "1998-067A");
        assert_eq!(omm.norad_cat_id, Some(25544));
        assert_eq!(omm.rev_at_epoch, Some(29935));
        assert_eq!(omm.element_set_no, Some(900));
        assert_eq!(omm.ephemeris_type, Some(0));
        assert_eq!(omm.bstar, Some(tle.bstar));
        assert_eq!(omm.mean_motion_dot, Some(tle.mean_motion_dot));

        let back = omm.to_tle();
        assert_eq!(back.intl_desig, "98067A");
        assert_eq!(back.desig_year, 98);
        assert_eq!(back.desig_launch, 67);
        assert_eq!(back.desig_piece, "A");
        assert_eq!(back.sat_num, 25544);
        assert_eq!(back.epoch, tle.epoch);
        assert_eq!(back.mean_motion, tle.mean_motion);
        assert_eq!(back.eccen, tle.eccen);
        assert_eq!(back.bstar, tle.bstar);
        assert_eq!(back.rev_num, tle.rev_num);
        assert_eq!(back.to_2line().unwrap(), tle.to_2line().unwrap());

        // Both propagate identically
        let times = vec![tle.epoch + Duration::from_hours(1.0)];
        let mut omm = omm;
        let mut tle = tle;
        let so = propagate(&mut omm, &times);
        let st = sgp4_full(&mut tle, &times, GravConst::WGS72, OpsMode::IMPROVED).unwrap();
        assert!(dist(col3(&so.pos, 0), col3(&st.pos, 0)) < 1e-6);

        // Designators that are not YYYY-NNNP survive as-is / empty
        assert_eq!(object_id_from_intl_desig(""), "UNKNOWN");
        assert_eq!(object_id_from_intl_desig("57001B"), "1957-001B");
        assert_eq!(object_id_from_intl_desig("T00001"), "T00001");
        assert!(intl_desig_from_object_id("UNKNOWN").is_none());
        assert!(intl_desig_from_object_id("1998-067").is_none());
        assert_eq!(
            intl_desig_from_object_id("2023-146X"),
            Some(("23146X".to_string(), 23, 146, "X".to_string()))
        );
    }

    #[test]
    fn test_from_mean_elements_propagates() {
        let mut omm = OMM::from_mean_elements(
            Instant::from_rfc3339("2026-02-14T05:08:48.534432Z").unwrap(),
            15.4859353,
            0.00110623,
            51.6315,
            188.3997,
            96.9141,
            263.3106,
        );
        omm.bstar = Some(0.00016303535);
        let times = vec![omm.epoch + Duration::from_minutes(30.0)];
        let a = col3(&propagate(&mut omm, &times).pos, 0);

        // Editing an element after a propagation takes effect once the cache is reset
        omm.mean_anomaly += 10.0;
        let stale = col3(&propagate(&mut omm, &times).pos, 0);
        assert_eq!(stale, a);
        omm.reset_cache();
        let fresh = col3(&propagate(&mut omm, &times).pos, 0);
        assert!(dist(fresh, a) > 1e3);
    }
}
