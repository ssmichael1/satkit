use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::astrotime::AstroTime;
use crate::utils::{datadir, download_file, download_if_not_exist, skerror, SKResult};

use std::sync::RwLock;

use once_cell::sync::OnceCell;

#[derive(Debug, Clone)]
pub struct SpaceWeatherRecord {
    /// Date of record
    pub date: AstroTime,
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

fn str2num<T: core::str::FromStr>(s: &str, sidx: usize, eidx: usize) -> Option<T> {
    match s
        .chars()
        .skip(sidx)
        .take(eidx - sidx)
        .collect::<String>()
        .trim()
        .parse()
    {
        Ok(v) => Some(v),
        Err(_) => None,
    }
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

impl PartialEq<AstroTime> for SpaceWeatherRecord {
    fn eq(&self, other: &AstroTime) -> bool {
        self.date == *other
    }
}

impl PartialOrd<AstroTime> for SpaceWeatherRecord {
    fn partial_cmp(&self, other: &AstroTime) -> Option<Ordering> {
        self.date.partial_cmp(other)
    }
}

fn load_space_weather_csv() -> SKResult<Vec<SpaceWeatherRecord>> {
    let path = datadir().unwrap_or(PathBuf::from(".")).join("SW-All.csv");
    download_if_not_exist(&path, Some("http://celestrak.org/SpaceData/"))?;

    let file = File::open(&path)?;
    io::BufReader::new(file)
        .lines()
        .skip(1)
        .map(|rline| {
            let line = rline.unwrap();
            let lvals: Vec<&str> = line.split(",").collect();

            let year: u32 = match str2num(lvals[0], 0, 4) {
                Some(v) => v,
                None => return skerror!("invalid year in file"),
            };
            let mon: u32 = match str2num(lvals[0], 5, 7) {
                Some(v) => v,
                None => return skerror!("Invalid month in file"),
            };
            let day: u32 = match str2num(lvals[0], 8, 10) {
                Some(v) => v,
                None => return skerror!("invalid day in file"),
            };
            Ok(SpaceWeatherRecord {
                date: (AstroTime::from_date(year as i32, mon, day)),
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

fn space_weather_singleton() -> &'static RwLock<SKResult<Vec<SpaceWeatherRecord>>> {
    static INSTANCE: OnceCell<RwLock<SKResult<Vec<SpaceWeatherRecord>>>> = OnceCell::new();
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
pub fn get(tm: AstroTime) -> SKResult<SpaceWeatherRecord> {
    let sw_lock = space_weather_singleton().read().unwrap();
    let sw = sw_lock.as_ref().unwrap();

    // First, try simple indexing
    let idx = (tm - sw[0].date).days().floor() as usize;
    if idx < sw.len() && (tm - sw[idx].date).days().abs() < 1.0 {
        return Ok(sw[idx].clone());
    }

    // More-complex lookup (is it in the future?)
    // Increase efficiency by looking backward
    let rec = sw.iter().rev().find(|x| x.date <= tm);
    if let Some(r) = rec {
        Ok(r.clone())
    } else {
        skerror!("Invalid date")
    }
}

/// Download new Space Weather file, and load it.
pub fn update() -> SKResult<()> {
    // Get data directory
    let d = datadir()?;
    if d.metadata()?.permissions().readonly() {
        return skerror!(
            r#"Data directory is read-only. 
             Try setting the environment variable SATKIT_DATA
             to a writeable directory and re-starting or explicitly set
             data directory to writeable directory"#
        );
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
        let tm: AstroTime = AstroTime::from_datetime(2023, 11, 14, 0, 0, 0.0);
        let r = get(tm);
        println!("r = {:?}", r);
        println!("rdate = {}", r.unwrap().date);
    }
}
