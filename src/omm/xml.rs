use anyhow::Result;
use serde::Deserialize;

use super::OMM;

fn parse_required_f64(field: &str, value: Option<String>) -> anyhow::Result<f64> {
    let value = value.ok_or_else(|| anyhow::anyhow!("Missing required field {field}"))?;
    value
        .trim()
        .parse::<f64>()
        .map_err(|e| anyhow::anyhow!("Invalid float for {field}: {e}"))
}

fn parse_optional_f64(value: Option<String>) -> anyhow::Result<Option<f64>> {
    match value {
        None => Ok(None),
        Some(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse::<f64>()
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("Invalid float value: {e}"))
            }
        }
    }
}

fn parse_optional_u8(value: Option<String>) -> anyhow::Result<Option<u8>> {
    match value {
        None => Ok(None),
        Some(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse::<u8>()
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("Invalid u8 value: {e}"))
            }
        }
    }
}

fn parse_optional_u32(value: Option<String>) -> anyhow::Result<Option<u32>> {
    match value {
        None => Ok(None),
        Some(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse::<u32>()
                    .map(Some)
                    .map_err(|e| anyhow::anyhow!("Invalid u32 value: {e}"))
            }
        }
    }
}

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
    #[serde(rename = "ORIGINATOR")]
    originator: Option<String>,
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
    #[serde(rename = "OBJECT_NAME")]
    object_name: String,
    #[serde(rename = "OBJECT_ID")]
    object_id: String,
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
    #[serde(rename = "CLASSIFICATION")]
    classification: Option<String>,
    #[serde(rename = "MESSAGE_ID")]
    message_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OmmXmlData {
    #[serde(rename = "meanElements")]
    mean_elements: OmmXmlMeanElements,
    #[serde(rename = "tleParameters")]
    tle_parameters: Option<OmmXmlTleParameters>,
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

#[derive(Debug, Deserialize)]
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

impl TryFrom<OmmXmlMessage> for OMM {
    type Error = anyhow::Error;

    fn try_from(xml: OmmXmlMessage) -> std::result::Result<Self, Self::Error> {
        let metadata = xml.body.segment.metadata;
        let mean = xml.body.segment.data.mean_elements;
        let tle = xml.body.segment.data.tle_parameters;

        let (ephemeris_type, classification_type, norad_cat_id, element_set_no, rev_at_epoch) =
            if let Some(ref tle) = tle {
                (
                    parse_optional_u8(tle.ephemeris_type.clone())?,
                    tle.classification_type.clone(),
                    parse_optional_u32(tle.norad_cat_id.clone())?,
                    parse_optional_u32(tle.element_set_no.clone())?,
                    parse_optional_u32(tle.rev_at_epoch.clone())?,
                )
            } else {
                (None, None, None, None, None)
            };

        Ok(OMM {
            omm_version: xml.version,
            comments: None,
            originator: xml.header.and_then(|h| h.originator),
            classification: metadata.classification,
            message_id: metadata.message_id,
            object_name: metadata.object_name,
            object_id: metadata.object_id,
            center_name: metadata.center_name,
            reference_frame: metadata.reference_frame,
            reference_frame_epoch: metadata.reference_frame_epoch,
            time_system: metadata.time_system,
            mean_element_theory: metadata.mean_element_theory,
            epoch: mean.epoch,
            mean_motion: parse_required_f64("MEAN_MOTION", mean.mean_motion)?,
            eccentricity: parse_required_f64("ECCENTRICITY", mean.eccentricity)?,
            inclination: parse_required_f64("INCLINATION", mean.inclination)?,
            raan: parse_required_f64("RA_OF_ASC_NODE", mean.raan)?,
            arg_of_pericenter: parse_required_f64("ARG_OF_PERICENTER", mean.arg_of_pericenter)?,
            mean_anomaly: parse_required_f64("MEAN_ANOMALY", mean.mean_anomaly)?,
            gm: parse_optional_f64(mean.gm)?,
            mass: parse_optional_f64(tle.as_ref().and_then(|x| x.mass.clone()))?,
            solar_rad_area: parse_optional_f64(
                tle.as_ref().and_then(|x| x.solar_rad_area.clone()),
            )?,
            drag_area: parse_optional_f64(tle.as_ref().and_then(|x| x.drag_area.clone()))?,
            solar_rad_coeff: parse_optional_f64(
                tle.as_ref().and_then(|x| x.solar_rad_coeff.clone()),
            )?,
            drag_coeff: parse_optional_f64(tle.as_ref().and_then(|x| x.drag_coeff.clone()))?,
            ephemeris_type,
            classification_type,
            norad_cat_id,
            element_set_no,
            rev_at_epoch,
            bstar: parse_optional_f64(tle.as_ref().and_then(|x| x.bstar.clone()))?,
            bterm: parse_optional_f64(tle.as_ref().and_then(|x| x.bterm.clone()))?,
            mean_motion_dot: parse_optional_f64(
                tle.as_ref().and_then(|x| x.mean_motion_dot.clone()),
            )?,
            mean_motion_ddot: parse_optional_f64(
                tle.as_ref().and_then(|x| x.mean_motion_ddot.clone()),
            )?,
            agom: parse_optional_f64(tle.as_ref().and_then(|x| x.agom.clone()))?,
            satrec: None,
            extra_fields: std::collections::HashMap::new(),
        })
    }
}

impl OMM {
    /// Deserializes OMM records from an XML string.
    ///
    /// Supports CelesTrak/CCSDS NDM wrappers (`<ndm><omm>...`) and single
    /// message payloads (`<omm>...`).
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
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if XML parsing fails or if required OMM fields are missing/invalid.
    pub fn from_xml_string(s: &str) -> Result<Vec<OMM>> {
        if s.contains("<ndm") {
            let root: OmmXmlRoot = quick_xml::de::from_str(s).map_err(|e| anyhow::anyhow!(e))?;
            root.omms
                .into_iter()
                .map(OMM::try_from)
                .collect::<Result<Vec<_>>>()
        } else {
            let msg: OmmXmlMessage = quick_xml::de::from_str(s).map_err(|e| anyhow::anyhow!(e))?;
            Ok(vec![OMM::try_from(msg)?])
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
    pub fn from_xml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<OMM>> {
        let s = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!(e))?;
        Self::from_xml_string(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::OMM;

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
    }
}
