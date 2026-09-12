//! CCSDS NDM/XML representation of OMMs (feature `omm-xml`).

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::{parse_opt_field, parse_req_field, Error, Result, OMM};

#[derive(Debug, Deserialize)]
struct OmmXmlRoot {
    #[serde(rename = "omm", default)]
    omms: Vec<OmmXmlMessage>,
}

#[derive(Debug, Deserialize)]
struct OmmXmlMessage {
    #[serde(rename = "@version")]
    version: Option<String>,
    #[serde(rename = "header")]
    header: Option<OmmXmlHeader>,
    #[serde(rename = "body")]
    body: OmmXmlBody,
}

#[derive(Debug, Deserialize)]
struct OmmXmlHeader {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "CREATION_DATE")]
    creation_date: Option<String>,
    #[serde(rename = "ORIGINATOR")]
    originator: Option<String>,
    #[serde(rename = "CLASSIFICATION")]
    classification: Option<String>,
    #[serde(rename = "MESSAGE_ID")]
    message_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OmmXmlBody {
    #[serde(rename = "segment")]
    segment: OmmXmlSegment,
}

#[derive(Debug, Deserialize)]
struct OmmXmlSegment {
    #[serde(rename = "metadata")]
    metadata: OmmXmlMetadata,
    #[serde(rename = "data")]
    data: OmmXmlData,
}

#[derive(Debug, Deserialize)]
struct OmmXmlMetadata {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "OBJECT_NAME")]
    object_name: Option<String>,
    #[serde(rename = "OBJECT_ID")]
    object_id: Option<String>,
    #[serde(rename = "CENTER_NAME")]
    center_name: Option<String>,
    #[serde(rename = "REF_FRAME")]
    reference_frame: Option<String>,
    #[serde(rename = "REF_FRAME_EPOCH")]
    reference_frame_epoch: Option<String>,
    #[serde(rename = "TIME_SYSTEM")]
    time_system: Option<String>,
    #[serde(rename = "MEAN_ELEMENT_THEORY")]
    mean_element_theory: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OmmXmlData {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "meanElements")]
    mean_elements: OmmXmlMeanElements,
    #[serde(rename = "spacecraftParameters")]
    spacecraft_parameters: Option<OmmXmlSpacecraftParameters>,
    #[serde(rename = "tleParameters")]
    tle_parameters: Option<OmmXmlTleParameters>,
    #[serde(rename = "userDefinedParameters")]
    user_defined_parameters: Option<OmmXmlUserDefinedParameters>,
}

#[derive(Debug, Deserialize)]
struct OmmXmlMeanElements {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "MEAN_MOTION")]
    mean_motion: Option<String>,
    #[serde(rename = "ECCENTRICITY")]
    eccentricity: Option<String>,
    #[serde(rename = "INCLINATION")]
    inclination: Option<String>,
    #[serde(rename = "RA_OF_ASC_NODE")]
    raan: Option<String>,
    #[serde(rename = "ARG_OF_PERICENTER")]
    arg_of_pericenter: Option<String>,
    #[serde(rename = "MEAN_ANOMALY")]
    mean_anomaly: Option<String>,
    #[serde(rename = "GM")]
    gm: Option<String>,
}

/// CCSDS puts mass and the drag/SRP areas and coefficients in their own
/// block; older producers put them under `tleParameters`. Both are read.
#[derive(Debug, Deserialize, Default)]
struct OmmXmlSpacecraftParameters {
    #[serde(rename = "MASS")]
    mass: Option<String>,
    #[serde(rename = "SOLAR_RAD_AREA")]
    solar_rad_area: Option<String>,
    #[serde(rename = "SOLAR_RAD_COEFF")]
    solar_rad_coeff: Option<String>,
    #[serde(rename = "DRAG_AREA")]
    drag_area: Option<String>,
    #[serde(rename = "DRAG_COEFF")]
    drag_coeff: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OmmXmlTleParameters {
    #[serde(rename = "EPHEMERIS_TYPE")]
    ephemeris_type: Option<String>,
    #[serde(rename = "CLASSIFICATION_TYPE")]
    classification_type: Option<String>,
    #[serde(rename = "NORAD_CAT_ID")]
    norad_cat_id: Option<String>,
    #[serde(rename = "ELEMENT_SET_NO")]
    element_set_no: Option<String>,
    #[serde(rename = "REV_AT_EPOCH")]
    rev_at_epoch: Option<String>,
    #[serde(rename = "BSTAR")]
    bstar: Option<String>,
    #[serde(rename = "BTERM")]
    bterm: Option<String>,
    #[serde(rename = "MEAN_MOTION_DOT")]
    mean_motion_dot: Option<String>,
    #[serde(rename = "MEAN_MOTION_DDOT")]
    mean_motion_ddot: Option<String>,
    #[serde(rename = "AGOM")]
    agom: Option<String>,
    #[serde(rename = "MASS")]
    mass: Option<String>,
    #[serde(rename = "SOLAR_RAD_AREA")]
    solar_rad_area: Option<String>,
    #[serde(rename = "DRAG_AREA")]
    drag_area: Option<String>,
    #[serde(rename = "SOLAR_RAD_COEFF")]
    solar_rad_coeff: Option<String>,
    #[serde(rename = "DRAG_COEFF")]
    drag_coeff: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OmmXmlUserDefinedParameters {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "USER_DEFINED", default)]
    params: Vec<OmmXmlUserDefined>,
}

/// `<USER_DEFINED parameter="NAME">value</USER_DEFINED>`; an empty element
/// (`<USER_DEFINED parameter="RCS_SIZE"/>`) is a null, as in Space-Track JSON.
#[derive(Debug, Deserialize)]
struct OmmXmlUserDefined {
    #[serde(rename = "@parameter")]
    name: String,
    #[serde(rename = "$text", default)]
    value: Option<String>,
}

fn join_comments(groups: &[&[String]]) -> Option<String> {
    let lines: Vec<&str> = groups
        .iter()
        .flat_map(|g| g.iter().map(|s| s.trim()))
        .filter(|s| !s.is_empty())
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

impl TryFrom<OmmXmlMessage> for OMM {
    type Error = Error;

    fn try_from(xml: OmmXmlMessage) -> Result<Self> {
        let header = xml.header;
        let metadata = xml.body.segment.metadata;
        let data = xml.body.segment.data;
        let mean = data.mean_elements;
        let tle = data.tle_parameters.unwrap_or_default();
        let sc = data.spacecraft_parameters.unwrap_or_default();
        let user = data.user_defined_parameters.unwrap_or_default();

        let comments = join_comments(&[
            header.as_ref().map_or(&[][..], |h| &h.comments),
            &metadata.comments,
            &data.comments,
            &user.comments,
        ]);

        let mut extra_fields: HashMap<String, Value> = user
            .params
            .into_iter()
            .map(|p| {
                let v = match p.value.as_deref().map(str::trim) {
                    None | Some("") => Value::Null,
                    Some(s) => Value::String(s.to_string()),
                };
                (p.name, v)
            })
            .collect();
        // CREATION_DATE has no struct field; JSON keeps it as an extra, so
        // the XML form does the same.
        if let Some(created) = header
            .as_ref()
            .and_then(|h| h.creation_date.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            extra_fields.insert(
                "CREATION_DATE".to_string(),
                Value::String(created.to_string()),
            );
        }

        // Prefer the CCSDS spacecraftParameters block, fall back to the
        // legacy placement under tleParameters.
        let first = |a: Option<String>, b: Option<String>| a.or(b);

        Ok(Self {
            omm_version: xml.version,
            comments,
            originator: header.as_ref().and_then(|h| h.originator.clone()),
            classification: header.as_ref().and_then(|h| h.classification.clone()),
            message_id: header.as_ref().and_then(|h| h.message_id.clone()),
            object_name: metadata.object_name.unwrap_or_else(super::unknown),
            object_id: metadata.object_id.unwrap_or_else(super::unknown),
            center_name: metadata.center_name,
            reference_frame: metadata.reference_frame,
            reference_frame_epoch: metadata.reference_frame_epoch,
            time_system: metadata.time_system,
            mean_element_theory: metadata.mean_element_theory,
            epoch: crate::Instant::from_rfc3339(mean.epoch.trim())?,
            mean_motion: parse_req_field("MEAN_MOTION", mean.mean_motion.as_deref())?,
            eccentricity: parse_req_field("ECCENTRICITY", mean.eccentricity.as_deref())?,
            inclination: parse_req_field("INCLINATION", mean.inclination.as_deref())?,
            raan: parse_req_field("RA_OF_ASC_NODE", mean.raan.as_deref())?,
            arg_of_pericenter: parse_req_field(
                "ARG_OF_PERICENTER",
                mean.arg_of_pericenter.as_deref(),
            )?,
            mean_anomaly: parse_req_field("MEAN_ANOMALY", mean.mean_anomaly.as_deref())?,
            gm: parse_opt_field("GM", mean.gm.as_deref())?,
            mass: parse_opt_field("MASS", first(sc.mass, tle.mass).as_deref())?,
            solar_rad_area: parse_opt_field(
                "SOLAR_RAD_AREA",
                first(sc.solar_rad_area, tle.solar_rad_area).as_deref(),
            )?,
            drag_area: parse_opt_field("DRAG_AREA", first(sc.drag_area, tle.drag_area).as_deref())?,
            solar_rad_coeff: parse_opt_field(
                "SOLAR_RAD_COEFF",
                first(sc.solar_rad_coeff, tle.solar_rad_coeff).as_deref(),
            )?,
            drag_coeff: parse_opt_field(
                "DRAG_COEFF",
                first(sc.drag_coeff, tle.drag_coeff).as_deref(),
            )?,
            ephemeris_type: parse_opt_field("EPHEMERIS_TYPE", tle.ephemeris_type.as_deref())?,
            classification_type: tle
                .classification_type
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            norad_cat_id: parse_opt_field("NORAD_CAT_ID", tle.norad_cat_id.as_deref())?,
            element_set_no: parse_opt_field("ELEMENT_SET_NO", tle.element_set_no.as_deref())?,
            rev_at_epoch: parse_opt_field("REV_AT_EPOCH", tle.rev_at_epoch.as_deref())?,
            bstar: parse_opt_field("BSTAR", tle.bstar.as_deref())?,
            bterm: parse_opt_field("BTERM", tle.bterm.as_deref())?,
            mean_motion_dot: parse_opt_field("MEAN_MOTION_DOT", tle.mean_motion_dot.as_deref())?,
            mean_motion_ddot: parse_opt_field("MEAN_MOTION_DDOT", tle.mean_motion_ddot.as_deref())?,
            agom: parse_opt_field("AGOM", tle.agom.as_deref())?,
            satrec: None,
            extra_fields,
        })
    }
}

impl OMM {
    /// Deserializes OMM records from an XML string.
    ///
    /// Supports CelesTrak/CCSDS NDM wrappers (`<ndm><omm>...`) and single
    /// message payloads (`<omm>...`). Header, metadata, and data `COMMENT`
    /// lines are joined into [`comments`](Self::comments);
    /// `userDefinedParameters` land in [`extra_fields`](Self::extra_fields)
    /// keyed by their `parameter` attribute.
    ///
    /// Available only when the `omm-xml` feature is enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "omm-xml")]
    /// # {
    /// use satkit::prelude::OMM;
    ///
    /// let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
    /// <ndm>
    ///   <omm version="2.0">
    ///     <header><CREATION_DATE/><ORIGINATOR/></header>
    ///     <body>
    ///       <segment>
    ///         <metadata>
    ///           <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
    ///           <OBJECT_ID>1998-067A</OBJECT_ID>
    ///           <CENTER_NAME>EARTH</CENTER_NAME>
    ///           <REF_FRAME>TEME</REF_FRAME>
    ///           <TIME_SYSTEM>UTC</TIME_SYSTEM>
    ///           <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
    ///         </metadata>
    ///         <data>
    ///           <meanElements>
    ///             <EPOCH>2026-02-14T05:08:48.534432</EPOCH>
    ///             <MEAN_MOTION>15.48593530</MEAN_MOTION>
    ///             <ECCENTRICITY>.00110623</ECCENTRICITY>
    ///             <INCLINATION>51.6315</INCLINATION>
    ///             <RA_OF_ASC_NODE>188.3997</RA_OF_ASC_NODE>
    ///             <ARG_OF_PERICENTER>96.9141</ARG_OF_PERICENTER>
    ///             <MEAN_ANOMALY>263.3106</MEAN_ANOMALY>
    ///           </meanElements>
    ///         </data>
    ///       </segment>
    ///     </body>
    ///   </omm>
    /// </ndm>
    /// "#;
    ///
    /// let omms = OMM::from_xml_string(xml)?;
    /// assert_eq!(omms.len(), 1);
    /// assert_eq!(omms[0].object_id, "1998-067A");
    /// # }
    /// # Ok::<(), satkit::omm::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if XML parsing fails or if required OMM fields are missing/invalid.
    pub fn from_xml_string(s: &str) -> Result<Vec<Self>> {
        if s.contains("<ndm") {
            let root: OmmXmlRoot = quick_xml::de::from_str(s)?;
            root.omms
                .into_iter()
                .map(Self::try_from)
                .collect::<Result<Vec<_>>>()
        } else {
            let msg: OmmXmlMessage = quick_xml::de::from_str(s)?;
            Ok(vec![Self::try_from(msg)?])
        }
    }

    /// Deserializes OMM records from an XML file.
    ///
    /// Available only when the `omm-xml` feature is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, XML parsing fails, or required
    /// OMM fields are missing/invalid.
    pub fn from_xml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>> {
        let s = std::fs::read_to_string(path)?;
        Self::from_xml_string(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::OMM;
    use crate::utils::test::get_testvec_dir;
    use serde_json::Value;

    #[test]
    fn test_parse_omm_celestrak_xml() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ndm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd">
    <omm id="CCSDS_OMM_VERS" version="2.0">
        <header><CREATION_DATE/><ORIGINATOR/></header>
        <body>
            <segment>
                <metadata>
                    <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
                    <OBJECT_ID>1998-067A</OBJECT_ID>
                    <CENTER_NAME>EARTH</CENTER_NAME>
                    <REF_FRAME>TEME</REF_FRAME>
                    <TIME_SYSTEM>UTC</TIME_SYSTEM>
                    <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
                </metadata>
                <data>
                    <meanElements>
                        <EPOCH>2026-02-14T05:08:48.534432</EPOCH>
                        <MEAN_MOTION>15.48593530</MEAN_MOTION>
                        <ECCENTRICITY>.00110623</ECCENTRICITY>
                        <INCLINATION>51.6315</INCLINATION>
                        <RA_OF_ASC_NODE>188.3997</RA_OF_ASC_NODE>
                        <ARG_OF_PERICENTER>96.9141</ARG_OF_PERICENTER>
                        <MEAN_ANOMALY>263.3106</MEAN_ANOMALY>
                    </meanElements>
                    <tleParameters>
                        <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
                        <CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
                        <NORAD_CAT_ID>25544</NORAD_CAT_ID>
                        <ELEMENT_SET_NO>999</ELEMENT_SET_NO>
                        <REV_AT_EPOCH>55269</REV_AT_EPOCH>
                        <BSTAR>.16303535E-3</BSTAR>
                        <MEAN_MOTION_DOT>.8429E-4</MEAN_MOTION_DOT>
                        <MEAN_MOTION_DDOT>0</MEAN_MOTION_DDOT>
                    </tleParameters>
                </data>
            </segment>
        </body>
    </omm>
</ndm>
"#;

        let msg = OMM::from_xml_string(xml).unwrap();
        assert_eq!(msg.len(), 1);
        assert_eq!(msg[0].object_id, "1998-067A");
        assert_eq!(msg[0].omm_version.as_deref(), Some("2.0"));
        assert_eq!(msg[0].norad_cat_id, Some(25544));
        assert_eq!(msg[0].comments, None);
        assert!(msg[0].extra_fields.is_empty());
        assert_eq!(msg[0].epoch.as_rfc3339(), "2026-02-14T05:08:48.534432Z");

        // The same message via the auto-detecting entry point
        let auto = OMM::from_text(xml).unwrap();
        assert_eq!(auto[0].mean_motion, msg[0].mean_motion);
    }

    /// Space-Track XML carries a header COMMENT and a userDefinedParameters
    /// block; both must survive, matching what the JSON endpoint gives.
    #[test]
    fn test_spacetrack_xml_matches_json() {
        let dir = get_testvec_dir().unwrap();
        let xml = OMM::from_xml_file(dir.join("omm/spacetrack_omm.xml")).unwrap();
        let json = OMM::from_json_file(dir.join("omm/spacetrack_omm.json")).unwrap();
        assert_eq!(xml.len(), json.len());

        let (x, j) = (&xml[0], &json[0]);
        assert_eq!(
            x.comments.as_deref(),
            Some("GENERATED VIA SPACE-TRACK.ORG API")
        );
        assert_eq!(x.originator, j.originator);
        assert_eq!(x.omm_version, j.omm_version);
        assert_eq!(x.epoch, j.epoch);
        assert_eq!(x.mean_motion, j.mean_motion);
        assert_eq!(x.bstar, j.bstar);
        assert_eq!(x.norad_cat_id, j.norad_cat_id);
        assert_eq!(x.ephemeris_type, j.ephemeris_type);
        assert_eq!(x.extra_fields["OBJECT_TYPE"], j.extra_fields["OBJECT_TYPE"]);
        assert_eq!(
            x.extra_fields["SEMIMAJOR_AXIS"],
            j.extra_fields["SEMIMAJOR_AXIS"]
        );
        assert_eq!(x.extra_fields["RCS_SIZE"], Value::Null);
        assert_eq!(j.extra_fields["RCS_SIZE"], Value::Null);
        assert_eq!(
            x.extra_fields["CREATION_DATE"],
            j.extra_fields["CREATION_DATE"]
        );
    }
}
