//! OMM records as CSV: a header row of OMM keywords and one row per message,
//! as CelesTrak (its default GP format) and Space-Track serve them.

use serde_json::{Map, Value};

use super::{Error, Result, OMM};

/// One CSV record: the 1-based line it starts on and its fields.
type Record = (usize, Vec<String>);

/// Split CSV text into records (RFC 4180): comma-separated fields, a field
/// optionally enclosed in double quotes, inside which `""` is a literal quote
/// and commas and line breaks are data. Records end at LF or CRLF; blank
/// lines are skipped.
fn records(text: &str) -> Result<Vec<Record>> {
    let mut out = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    // Whether the current field was quoted: `""` is a field, not a blank line
    let mut quoted = false;
    let mut line = 1;
    let mut start = 1;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if field.is_empty() && !quoted => {
                quoted = true;
                let mut closed = false;
                while let Some(q) = chars.next() {
                    match q {
                        '"' if chars.peek() == Some(&'"') => {
                            chars.next();
                            field.push('"');
                        }
                        '"' => {
                            closed = true;
                            break;
                        }
                        _ => {
                            if q == '\n' {
                                line += 1;
                            }
                            field.push(q);
                        }
                    }
                }
                if !closed {
                    return Err(Error::Csv {
                        line: start,
                        message: "a quoted field is not closed".to_string(),
                    });
                }
                if !matches!(chars.peek(), None | Some(',' | '\r' | '\n')) {
                    return Err(Error::Csv {
                        line,
                        message: "text follows the closing quote of a field".to_string(),
                    });
                }
            }
            ',' => {
                fields.push(std::mem::take(&mut field));
                quoted = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                if !(fields.is_empty() && field.is_empty() && !quoted) {
                    fields.push(std::mem::take(&mut field));
                    out.push((start, std::mem::take(&mut fields)));
                }
                quoted = false;
                line += 1;
                start = line;
            }
            _ => field.push(c),
        }
    }
    if !(fields.is_empty() && field.is_empty() && !quoted) {
        fields.push(field);
        out.push((start, fields));
    }
    Ok(out)
}

impl OMM {
    /// Deserializes OMM records from CSV text: a header row of OMM keywords
    /// (`OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,...`) and one row per
    /// message.
    ///
    /// This is the layout CelesTrak serves (its default for GP queries, and
    /// `FORMAT=csv`) and Space-Track's `format/csv`. It is not a CCSDS
    /// encoding (the standard defines KVN and XML), but it is the most
    /// compact one and the columns are the CCSDS keywords.
    ///
    /// Fields may be quoted (Space-Track quotes every one); columns may come
    /// in any order. An empty cell reads as an absent field, except
    /// `OBJECT_NAME` and `OBJECT_ID`, which are kept as empty strings
    /// (analyst objects have no international designator). Columns that are
    /// not OMM fields (`RMS` and `DATA_SOURCE` in CelesTrak's supplemental
    /// data, Space-Track's `OBJECT_TYPE`, `TLE_LINE1`, ...) are kept in
    /// [`extra_fields`](Self::extra_fields) as strings, an empty cell as
    /// `null`.
    ///
    /// # Example
    ///
    /// ```
    /// use satkit::prelude::OMM;
    ///
    /// let csv = "\
    /// OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT
    /// ISS (ZARYA),1998-067A,2026-02-14T05:08:48.534432,15.4859353,.00110623,51.6315,188.3997,96.9141,263.3106,0,U,25544,999,55269,.16303535E-3,.8429E-4,0
    /// ";
    ///
    /// let omms = OMM::from_csv_string(csv)?;
    /// assert_eq!(omms.len(), 1);
    /// assert_eq!(omms[0].norad_cat_id, Some(25544));
    /// assert_eq!(omms[0].bstar, Some(0.00016303535));
    /// # Ok::<(), satkit::omm::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Csv`], with the line the record starts on, if the header has
    /// no `EPOCH` or `MEAN_MOTION` column (the table is not an OMM table), a
    /// row has a different number of fields than the header (which is how a
    /// truncated download shows), a quoted field is malformed, or a row is
    /// missing a mandatory field or holds an unparsable value. No records
    /// are returned in that case.
    pub fn from_csv_string(s: &str) -> Result<Vec<Self>> {
        let mut rows = records(s.trim_start_matches('\u{feff}'))?.into_iter();
        let Some((header_line, header)) = rows.next() else {
            return Err(Error::Csv {
                line: 1,
                message: "no header row".to_string(),
            });
        };
        let header: Vec<String> = header.iter().map(|h| h.trim().to_string()).collect();
        for required in ["EPOCH", "MEAN_MOTION"] {
            if !header.iter().any(|h| h == required) {
                return Err(Error::Csv {
                    line: header_line,
                    message: format!(
                        "the header row has no {required} column, so this is not a table of \
                         OMMs (its columns: {})",
                        header.join(", ")
                    ),
                });
            }
        }
        rows.map(|(line, fields)| {
            if fields.len() != header.len() {
                return Err(Error::Csv {
                    line,
                    message: format!(
                        "{} fields where the header row has {} (a truncated file ends this way)",
                        fields.len(),
                        header.len()
                    ),
                });
            }
            let map: Map<String, Value> = header
                .iter()
                .zip(fields)
                .map(|(key, text)| {
                    let keep_empty = matches!(key.as_str(), "OBJECT_NAME" | "OBJECT_ID");
                    let value = if text.trim().is_empty() && !keep_empty {
                        Value::Null
                    } else {
                        Value::String(text)
                    };
                    (key.clone(), value)
                })
                .collect();
            serde_json::from_value(Value::Object(map)).map_err(|e| Error::Csv {
                line,
                message: e.to_string(),
            })
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT";
    const ISS: &str = "ISS (ZARYA),1998-067A,2026-11-11T04:58:11.303328,15.49310717,.00073162,51.6322,292.5559,12.7435,347.3737,0,U,25544,999,59454,.17362007E-3,.9235E-4,0";

    #[test]
    fn celestrak_layout() {
        let omms = OMM::from_csv_string(&format!("{HEADER}\r\n{ISS}\r\n")).unwrap();
        assert_eq!(omms.len(), 1);
        let o = &omms[0];
        assert_eq!(o.object_name, "ISS (ZARYA)");
        assert_eq!(o.object_id, "1998-067A");
        assert_eq!(o.epoch.as_rfc3339(), "2026-11-11T04:58:11.303328Z");
        assert_eq!(o.mean_motion, 15.49310717);
        assert_eq!(o.eccentricity, 0.00073162);
        assert_eq!(o.norad_cat_id, Some(25544));
        assert_eq!(o.rev_at_epoch, Some(59454));
        assert_eq!(o.bstar, Some(0.17362007e-3));
        assert_eq!(o.mean_motion_dot, Some(0.9235e-4));
        assert_eq!(o.mean_motion_ddot, Some(0.0));
        assert_eq!(o.classification_type.as_deref(), Some("U"));
        // CelesTrak omits the constant metadata
        assert!(o.center_name.is_none() && o.mean_element_theory.is_none());
        assert!(o.extra_fields.is_empty());
    }

    #[test]
    fn same_record_as_json() {
        let json = r#"{"OBJECT_NAME":"ISS (ZARYA)","OBJECT_ID":"1998-067A","EPOCH":"2026-11-11T04:58:11.303328","MEAN_MOTION":15.49310717,"ECCENTRICITY":0.00073162,"INCLINATION":51.6322,"RA_OF_ASC_NODE":292.5559,"ARG_OF_PERICENTER":12.7435,"MEAN_ANOMALY":347.3737,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":25544,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":59454,"BSTAR":0.00017362007,"MEAN_MOTION_DOT":9.235e-5,"MEAN_MOTION_DDOT":0}"#;
        let j = OMM::from_json_string(json).unwrap().remove(0);
        let c = OMM::from_csv_string(&format!("{HEADER}\n{ISS}"))
            .unwrap()
            .remove(0);
        assert_eq!(
            serde_json::to_value(&c).unwrap(),
            serde_json::to_value(&j).unwrap()
        );
    }

    #[test]
    fn quoted_fields_extra_columns_and_empty_cells() {
        // Space-Track's layout: every field quoted, its own columns, another
        // column order; a quoted name holding a comma, a quote and a newline
        let csv = concat!(
            "\u{feff}\"NORAD_CAT_ID\",\"OBJECT_NAME\",\"OBJECT_ID\",\"EPOCH\",\"MEAN_MOTION\",\"ECCENTRICITY\",",
            "\"INCLINATION\",\"RA_OF_ASC_NODE\",\"ARG_OF_PERICENTER\",\"MEAN_ANOMALY\",\"BSTAR\",\"OBJECT_TYPE\",\"DECAY_DATE\"\n",
            "\"270449\",\"\",\"\",\"2026-11-10T12:00:00.000000\",\"1.00270000\",\"0.00012340\",",
            "\"0.0500\",\"10.0000\",\"20.0000\",\"30.0000\",\"\",\"UNKNOWN\",\"\"\n",
            "\n",
            "\"25544\",\"A \"\"B\"\", C\nD\",\"1998-067A\",\"2026-315T04:58:11\",\"15.5\",\"0.001\",",
            "\"51.6\",\"1\",\"2\",\"3\",\"0.0001\",\"PAYLOAD\",\"\"",
        );
        let omms = OMM::from_csv_string(csv).unwrap();
        assert_eq!(omms.len(), 2);
        // An analyst object: empty name and designator are kept as they are
        assert_eq!(omms[0].norad_cat_id, Some(270449));
        assert_eq!(omms[0].object_name, "");
        assert_eq!(omms[0].object_id, "");
        assert_eq!(omms[0].bstar, None);
        assert_eq!(omms[0].extra_fields["OBJECT_TYPE"], "UNKNOWN");
        assert_eq!(omms[0].extra_fields["DECAY_DATE"], Value::Null);
        assert_eq!(omms[1].object_name, "A \"B\", C\nD");
        // Day-of-year epoch
        assert_eq!(omms[1].epoch.as_rfc3339(), "2026-11-11T04:58:11.000000Z");
    }

    #[test]
    fn not_an_omm_table() {
        let err = OMM::from_csv_string(
            "NORAD_CAT_ID,OBJECT_NUMBER,CURRENT_NAME\n\"85\",\"85\",\"SCOUT X-1 DEB\"\n",
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, Error::Csv { line: 1, .. }), "{err:?}");
        assert!(msg.contains("no EPOCH column"), "{msg}");
        assert!(msg.contains("CURRENT_NAME"), "{msg}");
    }

    #[test]
    fn bad_rows_are_located() {
        // A row cut short, as a truncated download ends
        let cut = &ISS[..ISS.len() - 20];
        let err = OMM::from_csv_string(&format!("{HEADER}\n{ISS}\n{cut}")).unwrap_err();
        assert!(matches!(err, Error::Csv { line: 3, .. }), "{err:?}");
        assert!(err.to_string().contains("truncated"), "{err}");

        // An unparsable value
        let bad = ISS.replace("15.49310717", "15.4931O717");
        let err = OMM::from_csv_string(&format!("{HEADER}\n{ISS}\n\n{bad}\n")).unwrap_err();
        assert!(matches!(err, Error::Csv { line: 4, .. }), "{err:?}");
        assert!(err.to_string().contains("15.4931O717"), "{err}");

        // A mandatory field left empty
        let empty = ISS.replace(",15.49310717,", ",,");
        let err = OMM::from_csv_string(&format!("{HEADER}\n{empty}\n")).unwrap_err();
        assert!(matches!(err, Error::Csv { line: 2, .. }), "{err:?}");

        // Malformed quoting
        let err = OMM::from_csv_string(&format!("{HEADER}\n\"ISS")).unwrap_err();
        assert!(err.to_string().contains("not closed"), "{err}");
        let err = OMM::from_csv_string(&format!("{HEADER}\n\"ISS\"x,1")).unwrap_err();
        assert!(err.to_string().contains("closing quote"), "{err}");

        assert!(OMM::from_csv_string("").is_err());
        // A header alone is an empty table
        assert!(OMM::from_csv_string(HEADER).unwrap().is_empty());
    }

    #[test]
    fn from_text_detects_csv() {
        let omms = OMM::from_text(&format!("\n{HEADER}\n{ISS}\n")).unwrap();
        assert_eq!(omms[0].norad_cat_id, Some(25544));
        // KVN and other text is still refused
        assert!(matches!(
            OMM::from_text("CCSDS_OMM_VERS = 2.0\nCREATION_DATE = 2026-01-01T00:00:00"),
            Err(Error::UnrecognizedFormat)
        ));
        assert!(matches!(
            OMM::from_text("No GP data found"),
            Err(Error::UnrecognizedFormat)
        ));
    }
}
