//! Orbital Mean-Element Messages (OMM)
//!
//! See: https://ccsds.org/Pubs/502x0b3e1.pdf
//! Also: https://www.space-track.org/documentation#/omm
//!
//!
//! Author notes:
//!
//! - This is a confusing standard that does not appear to be rigidly adhered to.
//!
//!

use serde::{Deserialize, Deserializer};

use anyhow::Result;

use crate::sgp4::{SatRec, SGP4InitArgs, SGP4Source};
use crate::{Instant, TimeScale};

fn de_f64_from_number_or_string<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let v = Value::deserialize(deserializer)?;
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| Error::custom("invalid number")),
        Value::String(s) => s
            .parse::<f64>()
            .map_err(|e| Error::custom(format!("invalid float string: {e}"))),
        _ => Err(Error::custom("expected number or string")),
    }
}


fn de_opt_f64_from_number_or_string<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let v = Value::deserialize(deserializer)?;
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| Error::custom("invalid number"))
            .map(Some),
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None) // remove this branch if you want "" to be an error
            } else {
                s.parse::<f64>()
                    .map(Some)
                    .map_err(|e| Error::custom(format!("invalid float string: {e}")))
            }
        }
        _ => Err(Error::custom("expected number, string, or null")),
    }
}

fn de_opt_u32_from_number_or_string<'de, D>(
    deserializer: D,
) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let v = Value::deserialize(deserializer)?;
    match v {
        Value::Null => Ok(None),

        Value::Number(n) => {
            let v = n
                .as_u64()
                .ok_or_else(|| Error::custom("invalid number for u32"))?;
            u32::try_from(v)
                .map(Some)
                .map_err(|_| Error::custom("u32 out of range"))
        }

        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None) // remove if "" should be an error
            } else {
                s.parse::<u32>()
                    .map(Some)
                    .map_err(|e| Error::custom(format!("invalid u32 string: {e}")))
            }
        }

        _ => Err(Error::custom("expected number, string, or null")),
    }
}

fn de_opt_u8_from_number_or_string<'de, D>(
    deserializer: D,
) -> Result<Option<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let v = Value::deserialize(deserializer)?;
    match v {
        Value::Null => Ok(None),

        Value::Number(n) => {
            let v = n
                .as_u64()
                .ok_or_else(|| Error::custom("invalid number for u8"))?;
            u8::try_from(v)
                .map(Some)
                .map_err(|_| Error::custom("u8 out of range"))
        }

        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                Ok(None) // remove if "" should be an error
            } else {
                s.parse::<u8>()
                    .map(Some)
                    .map_err(|e| Error::custom(format!("invalid u8 string: {e}")))
            }
        }

        _ => Err(Error::custom("expected number, string, or null")),
    }
}

/// OMM Structure
///
/// See Table 4-1, Table 4-2, Table 4-3 of CCSDS 502.0-B-3
///
#[derive(Debug, Deserialize, Clone, Default)]
pub struct OMM {
    #[serde(rename = "CCSDS_OMM_VERS")] // CCSDS says this is required, but it often is missing
    pub omm_version: Option<String>,
    #[serde(rename = "COMMENT")] // optional
    pub comments: Option<String>,
    #[serde(rename = "ORIGINATOR")] // Optional
    pub originator: Option<String>,
    #[serde(rename = "CLASSIFICATION")]
    pub classification: Option<String>,
    #[serde(rename = "MESSAGE_ID")]
    pub message_id: Option<String>,
    #[serde(rename = "OBJECT_NAME")] // Mandatory
    pub object_name: String,
    #[serde(rename = "OBJECT_ID")] // Mandatory
    pub object_id: String,
    #[serde(rename = "CENTER_NAME")] // Mandatory but often ignored
    pub center_name: Option<String>,
    #[serde(rename = "REF_FRAME")] // Mandatory but often ignored
    pub reference_frame: Option<String>,
    #[serde(rename = "REF_FRAME_EPOCH")] // Optional
    pub reference_frame_epoch: Option<String>,
    #[serde(rename = "TIME_SYSTEM")] // Mandatory but often ignored
    pub time_system: Option<String>,
    #[serde(rename = "MEAN_ELEMENT_THEORY")] // Mandatory but often ignored
    pub mean_element_theory: Option<String>,
    #[serde(rename = "EPOCH")] // Mandatory
    pub epoch: String,
    #[serde(rename = "MEAN_MOTION")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub mean_motion: f64,
    #[serde(rename = "ECCENTRICITY")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub eccentricity: f64,
    #[serde(rename = "INCLINATION")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub inclination: f64,
    #[serde(rename = "RA_OF_ASC_NODE")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub raan: f64,
    #[serde(rename = "ARG_OF_PERICENTER")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub arg_of_pericenter: f64,
    #[serde(rename = "MEAN_ANOMALY")] // Mandatory
    #[serde(deserialize_with = "de_f64_from_number_or_string")]
    pub mean_anomaly: f64,
    #[serde(rename = "GM")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub gm: Option<f64>,
    #[serde(rename = "MASS")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub mass: Option<f64>,
    #[serde(rename = "SOLAR_RAD_AREA")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub solar_rad_area: Option<f64>,
    #[serde(rename = "DRAG_AREA")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub drag_area: Option<f64>,
    #[serde(rename = "SOLAR_RAD_COEFF")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub solar_rad_coeff: Option<f64>,
    #[serde(rename = "DRAG_COEFF")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub drag_coeff: Option<f64>,
    #[serde(rename="EPHEMERIS_TYPE")] // Optional
    #[serde(default, deserialize_with = "de_opt_u8_from_number_or_string")]
    pub ephemeris_type: Option<u8>,
    #[serde(rename="CLASSIFICATION_TYPE")] // Optional
    pub classification_type: Option<String>,
    #[serde(rename="NORAD_CAT_ID")] // Optional
    #[serde(default, deserialize_with = "de_opt_u32_from_number_or_string")]
    pub norad_cat_id: Option<u32>,
    #[serde(rename="ELEMENT_SET_NO")] // Optional
    #[serde(default, deserialize_with = "de_opt_u32_from_number_or_string")]
    pub element_set_no: Option<u32>,
    #[serde(rename="REV_AT_EPOCH")] // Optional
    #[serde(default, deserialize_with = "de_opt_u32_from_number_or_string")]
    pub rev_at_epoch: Option<u32>,
    #[serde(rename="BSTAR")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub bstar: Option<f64>,
    #[serde(rename="BTERM")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub bterm: Option<f64>,
    #[serde(rename="MEAN_MOTION_DOT")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub mean_motion_dot: Option<f64>,
    #[serde(rename="MEAN_MOTION_DDOT")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub mean_motion_ddot: Option<f64>,
    #[serde(rename="AGOM")] // Optional
    #[serde(default, deserialize_with = "de_opt_f64_from_number_or_string")]
    pub agom: Option<f64>,

    /// Cached SGP4 record, initialized lazily on first propagation.
    #[serde(skip, default)]
    pub(crate) satrec: Option<SatRec>,

    #[serde(flatten)]
    pub extra_fields: std::collections::HashMap<String, serde_json::Value>,
}

impl OMM {
    fn epoch_instant(&self) -> anyhow::Result<Instant> {
        Ok(Instant::from_rfc3339(&self.epoch).map_err(|e| anyhow::anyhow!(e))?)
    }

    pub fn from_json_string(s: &str) -> Result<Vec<OMM>> {
        serde_json::from_str(s).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn from_json_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<OMM>> {
        let file = std::fs::File::open(path).map_err(|e| anyhow::anyhow!(e))?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| anyhow::anyhow!(e))
    }
}

impl SGP4Source for OMM {
    fn epoch(&self) -> Instant {
        // `sgp4_full` only calls `epoch()` after `sgp4_init_args()` succeeds.
        self.epoch_instant().unwrap_or(Instant::INVALID)
    }

    fn satrec_mut(&mut self) -> &mut Option<SatRec> {
        &mut self.satrec
    }

    fn sgp4_init_args(&self) -> anyhow::Result<SGP4InitArgs> {
        use std::f64::consts::PI;

        const TWOPI: f64 = PI * 2.0;

        if let Some(theory) = &self.mean_element_theory {
            if theory.trim().to_ascii_uppercase() != "SGP4" {
                anyhow::bail!("Unsupported MEAN_ELEMENT_THEORY: {theory}");
            }
        }

        if let Some(ts) = &self.time_system {
            if ts.trim().to_ascii_uppercase() != "UTC" {
                anyhow::bail!("Unsupported TIME_SYSTEM for SGP4: {ts}");
            }
        }

        let epoch = self.epoch_instant()?;

        Ok(SGP4InitArgs {
            jdsatepoch: epoch.as_jd_with_scale(TimeScale::UTC),
            bstar: self.bstar.unwrap_or(0.0),
            // Convert rev/day(+derivatives) to rad/min(+derivatives), matching TLE.
            no: self.mean_motion / (1440.0 / TWOPI),
            ndot: self.mean_motion_dot.unwrap_or(0.0) / (1440.0 * 1440.0 / TWOPI),
            nddot: self.mean_motion_ddot.unwrap_or(0.0) / (1440.0 * 1440.0 * 1440.0 / TWOPI),
            ecco: self.eccentricity,
            inclo: self.inclination.to_radians(),
            nodeo: self.raan.to_radians(),
            argpo: self.arg_of_pericenter.to_radians(),
            mo: self.mean_anomaly.to_radians(),
        })
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::utils::test::*;

    #[test]
    fn test_parse_omm_spacetrack_json() {
        let filename = get_testvec_dir().unwrap().join("omm/spacetrack_omm.json");
        let file = std::fs::File::open(filename).unwrap();
        let reader = std::io::BufReader::new(file);

        let msg: Vec<OMM> = serde_json::from_reader(reader).unwrap();
        println!("number of OMMs: {}", msg.len());
        println!("first OMM: {:#?}", msg[0]);

        // Test SGP4 initialization from first OMM
        let mut omm = msg[0].clone();


        // Actually run SGP4 propagation for 10 minutes past epoch
        let epoch = omm.epoch_instant().unwrap();
        println!("OMM epoch: {}", epoch);
        let times = vec![epoch, epoch + crate::time::Duration::from_minutes(10.0)];
        let states = crate::sgp4::sgp4_full(&mut omm, &times, crate::sgp4::GravConst::WGS72, crate::sgp4::OpsMode::IMPROVED).unwrap();
        for (i, _t) in times.iter().enumerate() {
            assert!(states.errcode[i] == crate::sgp4::SGP4Error::SGP4Success);
        }

    }

    #[test]
    fn test_parse_omm_celestrak_json() {
        let filename = get_testvec_dir().unwrap().join("omm/celestrak_omm.json");
        let file = std::fs::File::open(filename).unwrap();
        let reader = std::io::BufReader::new(file);

        let msg: Vec<OMM> = serde_json::from_reader(reader).unwrap();
        println!("number of OMMs: {}", msg.len());
        println!("first OMM: {:#?}", msg[0]);
    }


}
