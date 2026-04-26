use super::download::{self, download_file_async, download_to_string};
use crate::utils::datadir;
use serde_json::Value;
use std::path::PathBuf;
use std::thread::JoinHandle;
use thiserror::Error;

/// Errors produced by [`update_datafiles`] and the underlying download
/// orchestration helpers.
#[derive(Debug, Error)]
pub enum Error {
    /// The downloaded JSON file does not parse as the expected array of
    /// file URLs.
    #[error("Expected JSON array of file URLs")]
    NotJsonArray,

    /// An entry inside the JSON array is not a string URL.
    #[error("Expected string URL")]
    NotJsonString,

    /// Encountered an unexpected JSON node while traversing the
    /// recursive directory manifest.
    #[error("Invalid JSON manifest entry")]
    InvalidManifestEntry,

    /// One or more entries in the JSON manifest could not be parsed.
    #[error("Could not parse manifest entries")]
    ManifestParseFailed,

    /// The configured data directory is read-only and cannot receive
    /// new or refreshed files.
    #[error(
        "Data directory is read-only. Try setting SATKIT_DATA environment variable \
         to a writeable directory and re-starting"
    )]
    DataDirReadOnly,

    /// A worker thread launched by [`download_file_async`] panicked.
    #[error("Background download thread panicked")]
    ThreadPanic,

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] download::Error),
}

/// Convenient type alias used throughout the `update_data` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Download a list of files from a JSON file
fn download_from_url_json(json_url: String, basedir: &std::path::Path) -> Result<()> {
    let json_base: Value = serde_json::from_str(download_to_string(json_url.as_str())?.as_str())?;
    let arr = json_base.as_array().ok_or(Error::NotJsonArray)?;
    let vresult: Vec<JoinHandle<download::Result<bool>>> = arr
        .iter()
        .map(|url| -> Result<JoinHandle<download::Result<bool>>> {
            let url_str = url.as_str().ok_or(Error::NotJsonString)?;
            Ok(download_file_async(url_str.to_string(), basedir, true))
        })
        .collect::<Result<Vec<_>>>()?;
    // Wait for all the threads to finish
    for jh in vresult {
        jh.join().map_err(|_| Error::ThreadPanic)??;
    }

    Ok(())
}

/// Download a list of files from a JSON file
fn download_from_json(
    v: &Value,
    basedir: std::path::PathBuf,
    baseurl: String,
    overwrite: &bool,
    thandles: &mut Vec<JoinHandle<download::Result<bool>>>,
) -> Result<()> {
    if let Some(obj) = v.as_object() {
        let r1: Vec<Result<()>> = obj
            .iter()
            .map(|(key, val)| -> Result<()> {
                let pbnew = basedir.join(key);
                if !pbnew.is_dir() {
                    std::fs::create_dir_all(pbnew.clone())?;
                }
                let mut newurl = baseurl.clone();
                newurl.push_str(format!("/{key}").as_str());
                download_from_json(val, pbnew, newurl, overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| res.is_err())
            .collect();
        if !r1.is_empty() {
            return Err(Error::ManifestParseFailed);
        }
    } else if let Some(arr) = v.as_array() {
        let r2: Vec<Result<()>> = arr
            .iter()
            .map(|val| -> Result<()> {
                download_from_json(val, basedir.clone(), baseurl.clone(), overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| res.is_err())
            .collect();
        if !r2.is_empty() {
            return Err(Error::ManifestParseFailed);
        }
    } else if let Some(s) = v.as_str() {
        let mut newurl = baseurl;
        newurl.push_str(format!("/{s}").as_str());
        thandles.push(download_file_async(newurl, &basedir, *overwrite));
    } else {
        return Err(Error::InvalidManifestEntry);
    }

    Ok(())
}

fn download_datadir(basedir: PathBuf, baseurl: String, overwrite: &bool) -> Result<()> {
    if !basedir.is_dir() {
        std::fs::create_dir_all(basedir.clone())?;
    }

    let mut fileurl = baseurl.clone();
    fileurl.push_str("/files.json");

    let json_base: Value = serde_json::from_str(download_to_string(fileurl.as_str())?.as_str())?;
    let mut thandles: Vec<JoinHandle<download::Result<bool>>> = Vec::new();
    download_from_json(&json_base, basedir, baseurl, overwrite, &mut thandles)?;
    // Wait for all the threads to finish
    for jh in thandles {
        jh.join().map_err(|_| Error::ThreadPanic)??;
    }
    Ok(())
}

///
/// Download and update any necessary data files for "satkit" calculations
///
/// # Arguments
/// dir: The directory to download to, optional.  If not provided, the default data directory is used.
/// overwrite_if_exists: If true, overwrite any existing files.  If false, skip files that already exist.
///
/// # Returns
/// Result<()>
///
/// # Notes
///
/// This function downloads the data files necessary for "satkit" calculations.  These files include
/// data necessary for calculating the JPL ephemerides, inertial-to-Earth-fixed rotations, high-order
/// gravity field coefficients, and other data necessary for satellite calculations.
///
/// The data files also include space weather and Earth orientation parameters.  These files are always
/// downloaded, as they are updated at least daily.
///
pub fn update_datafiles(dir: Option<PathBuf>, overwrite_if_exists: bool) -> Result<()> {
    let downloaddir = match dir {
        Some(d) => d,
        None => datadir()?,
    };
    if downloaddir.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    println!(
        "Downloading data files to {}",
        downloaddir.to_str().unwrap()
    );
    // Download old files
    download_datadir(
        downloaddir.clone(),
        String::from("https://storage.googleapis.com/astrokit-astro-data"),
        &overwrite_if_exists,
    )?;

    println!("Now downloading files that are regularly updated:");
    println!("  Space Weather & Earth Orientation Parameters");
    // Get a list of files that are updated with new data, and download them
    download_from_url_json(
        String::from("https://storage.googleapis.com/astrokit-astro-data/files_refresh.json"),
        &downloaddir,
    )?;

    println!("  Solar Cycle Forecast");
    if let Err(e) = crate::solar_cycle_forecast::update() {
        eprintln!("Warning: could not download solar cycle forecast: {e}");
    }

    Ok(())
}
