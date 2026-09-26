//! CelesTrak / CSSI `SW-All.csv` reader.
//!
//! The legacy format: one CSV table carrying the observed record, the
//! NOAA/SWPC 45-day daily forecast and monthly predicted rows, merged by
//! CelesTrak from GFZ Potsdam (geomagnetic), DRAO / Natural Resources Canada
//! (F10.7) and SILSO (sunspot number).
//!
//! satkit no longer downloads this file, but reads it on request so that a
//! cached copy, an offline bundle, or the file GMAT and Orekit consume can be
//! supplied through [`init_from_path`](super::init_from_path).

use super::{Error, Result, SpaceWeatherDataType, SpaceWeatherRecord};
use crate::Instant;

fn str2num<T: core::str::FromStr>(
    s: &str,
    sidx: usize,
    eidx: usize,
    field: &'static str,
) -> Result<T> {
    s.chars()
        .skip(sidx)
        .take(eidx - sidx)
        .collect::<String>()
        .trim()
        .parse()
        .map_err(|_| Error::InvalidNumber(field))
}

/// Parse a `SW-All.csv` text buffer into space-weather records.
pub fn parse_csv(text: &str) -> Result<Vec<SpaceWeatherRecord>> {
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| -> Result<SpaceWeatherRecord> {
            let lvals: Vec<&str> = line.split(",").collect();
            // The record reads fixed column indices up to 30; bail on a
            // truncated line rather than panicking on out-of-bounds indexing.
            if lvals.len() < 31 {
                return Err(Error::InvalidEntry);
            }

            let year: u32 = str2num(lvals[0], 0, 4, "year")?;
            let mon: u32 = str2num(lvals[0], 5, 7, "month")?;
            let day: u32 = str2num(lvals[0], 8, 10, "day of month")?;

            Ok(SpaceWeatherRecord {
                date: (Instant::from_date(year as i32, mon as i32, day as i32)?),
                bsrn: lvals[1].parse().unwrap_or(-1),
                nd: lvals[2].parse().unwrap_or(-1),
                kp: {
                    let mut kparr: [i32; 8] = [-1, -1, -1, -1, -1, -1, -1, -1];
                    for idx in 0..8 {
                        kparr[idx] = lvals[idx + 3].parse().unwrap_or(-1);
                    }
                    kparr
                },
                kp_sum: lvals[11].parse().unwrap_or(-1),
                ap: {
                    let mut aparr: [i32; 8] = [-1, -1, -1, -1, -1, -1, -1, -1];
                    for idx in 0..8 {
                        aparr[idx] = lvals[12 + idx].parse().unwrap_or(-1)
                    }
                    aparr
                },
                ap_avg: lvals[20].parse().unwrap_or(-1),
                cp: lvals[21].parse().unwrap_or(-1.0),
                c9: lvals[22].parse().unwrap_or(-1),
                isn: lvals[23].parse().unwrap_or(-1),
                f10p7_obs: lvals[24].parse().unwrap_or(-1.0),
                f10p7_adj: lvals[25].parse().unwrap_or(-1.0),
                data_type: SpaceWeatherDataType::parse(lvals[26]),
                f10p7_obs_c81: lvals[27].parse().unwrap_or(-1.0),
                f10p7_obs_l81: lvals[28].parse().unwrap_or(-1.0),
                f10p7_adj_c81: lvals[29].parse().unwrap_or(-1.0),
                f10p7_adj_l81: lvals[30].parse().unwrap_or(-1.0),
            })
        })
        .collect()
}
