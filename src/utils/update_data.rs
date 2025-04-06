use super::download_file_async;
use super::download_to_string;
use crate::utils::datadir;
use json::JsonValue;
use std::path::PathBuf;
use std::thread::JoinHandle;

use anyhow::{bail, Result};

/// Download a list of files from a JSON file
fn download_from_url_json(json_url: String, basedir: &std::path::Path) -> Result<()> {
    let json_base: JsonValue = json::parse(download_to_string(json_url.as_str())?.as_str())?;
    let vresult: Vec<std::thread::JoinHandle<Result<bool>>> = json_base
        .members()
        .map(|url| -> JoinHandle<Result<bool>> {
            download_file_async(url.to_string(), basedir, true)
        })
        .collect();
    // Wait for all the threads to funish
    for jh in vresult {
        jh.join().unwrap()?;
    }

    Ok(())
}

/// Download a list of files from a JSON file
fn download_from_json(
    v: &JsonValue,
    basedir: std::path::PathBuf,
    baseurl: String,
    overwrite: &bool,
    thandles: &mut Vec<JoinHandle<Result<bool>>>,
) -> Result<()> {
    if v.is_object() {
        let r1: Vec<Result<()>> = v
            .entries()
            .map(|entry: (&str, &JsonValue)| -> Result<()> {
                let pbnew = basedir.join(entry.0);
                if !pbnew.is_dir() {
                    std::fs::create_dir_all(pbnew.clone())?;
                }
                let mut newurl = baseurl.clone();
                newurl.push_str(format!("/{}", entry.0).as_str());
                download_from_json(entry.1, pbnew, newurl, overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| res.is_err())
            .collect();
        if !r1.is_empty() {
            bail!("Could not parse entries");
        }
    } else if v.is_array() {
        let r2: Vec<Result<()>> = v
            .members()
            .map(|val| -> Result<()> {
                download_from_json(val, basedir.clone(), baseurl.clone(), overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| res.is_err())
            .collect();
        if !r2.is_empty() {
            bail!("could not parse array entries");
        }
    } else if v.is_string() {
        let mut newurl = baseurl;
        newurl.push_str(format!("/{}", v).as_str());
        thandles.push(download_file_async(newurl, &basedir, *overwrite));
    } else {
        bail!("invalid json for downloading files??!!");
    }

    Ok(())
}

fn download_datadir(basedir: PathBuf, baseurl: String, overwrite: &bool) -> Result<()> {
    if !basedir.is_dir() {
        std::fs::create_dir_all(basedir.clone())?;
    }

    let mut fileurl = baseurl.clone();
    fileurl.push_str("/files.json");

    let json_base: JsonValue = json::parse(download_to_string(fileurl.as_str())?.as_str())?;
    let mut thandles: Vec<JoinHandle<Result<bool>>> = Vec::new();
    download_from_json(&json_base, basedir, baseurl, overwrite, &mut thandles)?;
    // Wait for all the threads to funish
    for jh in thandles {
        jh.join().unwrap()?;
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
        bail!(
            r#"
            Data directory is read-only.
            Try setting SATKIT_DATA environment
            variable to a writeable directory and re-starting
            "#
        );
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
    Ok(())
}
