use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::utils::{datadir, download_file, download_if_not_exist};
use crate::Instant;
use crate::TimeLike;
use thiserror::Error;

use std::sync::RwLock;

use std::sync::OnceLock;

/// Errors produced by the [`spaceweather`](crate::spaceweather) module.
#[derive(Debug, Error)]
pub enum Error {
    /// A field in the CSV space-weather record could not be parsed as the
    /// expected numeric type.
    #[error("Invalid number in file: {0}")]
    InvalidNumber(&'static str),

    /// No space-weather record exists for the requested time.
    #[error("No space weather record found for date")]
    NoRecordForDate,

    /// The configured data directory is read-only and cannot receive an
    /// updated space-weather file.
    #[error(
        "Data directory is read-only. Try setting the environment variable SATKIT_DATA \
         to a writeable directory and re-starting or explicitly set data directory to \
         a writeable directory"
    )]
    DataDirReadOnly,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    InvalidEpoch(#[from] crate::time::InstantError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the `spaceweather` module.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct SpaceWeatherRecord {
    /// Date of record
    pub date: Instant,
    /// Bartels Solar Radiation Number.
    /// A sequence of 27-day intervals counted continuously from 1832 February 8
    pub bsrn: i32,
    /// Number of day within the bsrn
    pub nd: i32,
    /// Kp
    pub kp: [i32; 8],
    pub kp_sum: i32,
    pub ap: [i32; 8],
    pub ap_avg: i32,
    /// Planetary daily character figure
    pub cp: f64,
    /// Scale cp to \[0, 9\]
    pub c9: i32,
    /// International Sunspot Number
    pub isn: i32,
    pub f10p7_obs: f64,
    pub f10p7_adj: f64,
    pub f10p7_obs_c81: f64,
    pub f10p7_obs_l81: f64,
    pub f10p7_adj_c81: f64,
    pub f10p7_adj_l81: f64,
}

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

impl PartialEq for SpaceWeatherRecord {
    fn eq(&self, other: &Self) -> bool {
        self.date == other.date
    }
}

impl PartialOrd for SpaceWeatherRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.date.partial_cmp(&other.date)
    }
}

impl PartialEq<Instant> for SpaceWeatherRecord {
    fn eq(&self, other: &Instant) -> bool {
        self.date == *other
    }
}

impl PartialOrd<Instant> for SpaceWeatherRecord {
    fn partial_cmp(&self, other: &Instant) -> Option<Ordering> {
        self.date.partial_cmp(other)
    }
}

fn load_space_weather_csv() -> Result<Vec<SpaceWeatherRecord>> {
    let path = datadir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("SW-All.csv");
    download_if_not_exist(&path, Some("http://celestrak.org/SpaceData/"))?;

    let file = File::open(&path)?;
    io::BufReader::new(file)
        .lines()
        .skip(1)
        .map(|rline| -> Result<SpaceWeatherRecord> {
            let line = rline.unwrap();
            let lvals: Vec<&str> = line.split(",").collect();

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
                f10p7_obs_c81: lvals[27].parse().unwrap_or(-1.0),
                f10p7_obs_l81: lvals[28].parse().unwrap_or(-1.0),
                f10p7_adj_c81: lvals[29].parse().unwrap_or(-1.0),
                f10p7_adj_l81: lvals[30].parse().unwrap_or(-1.0),
            })
        })
        .collect()
}

fn space_weather_singleton() -> &'static RwLock<Result<Vec<SpaceWeatherRecord>>> {
    static INSTANCE: OnceLock<RwLock<Result<Vec<SpaceWeatherRecord>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| RwLock::new(load_space_weather_csv()))
}

///
/// Return full Space Weather record from Space Weather file,
/// as a function of requested instant in time,
/// linearly interpolated between time records in the file
///
/// # Arguments
///
/// * `tm` - time instant at which to retrieve space weather record
///
/// # Returns
///
/// * Full space weather record
///
/// # Notes:
///
/// * Space weather is updated daily in a file: sw19571001.txt
pub fn get<T: TimeLike>(tm: &T) -> Result<SpaceWeatherRecord> {
    let tm = tm.as_instant();
    let sw_lock = space_weather_singleton().read().unwrap();
    let sw = sw_lock.as_ref().unwrap();

    // First, try simple indexing
    let idx = (tm - sw[0].date).as_days().floor() as usize;
    if idx < sw.len() && (tm - sw[idx].date).as_days().abs() < 1.0 {
        return Ok(sw[idx].clone());
    }

    sw.iter()
        .rev()
        .find(|x| x.date <= tm)
        .cloned()
        .ok_or(Error::NoRecordForDate)
}

/// Download new Space Weather file, and load it.
pub fn update() -> Result<()> {
    // Get data directory
    let d = datadir()?;
    if d.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    // Download most-recent EOP
    let url = "https://celestrak.org/SpaceData/sw19571001.txt";
    download_file(url, &d, true)?;

    *space_weather_singleton().write().unwrap() = load_space_weather_csv();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let tm: Instant = Instant::from_datetime(2023, 11, 14, 0, 0, 0.0).unwrap();
        let r = get(&tm);
        println!("r = {:?}", r);
        println!("rdate = {}", r.unwrap().date);
    }
}
