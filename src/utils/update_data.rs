use super::download_file_async;
use super::download_to_string;
use crate::skerror;
use crate::utils::datadir;
use crate::SKResult;
use json::JsonValue;
use std::path::PathBuf;
use std::thread::JoinHandle;

fn download_from_url_json(json_url: String, basedir: &std::path::PathBuf) -> SKResult<()> {
    let json_base: JsonValue = json::parse(download_to_string(json_url.as_str())?.as_str())?;
    let vresult: Vec<std::thread::JoinHandle<SKResult<bool>>> = json_base
        .members()
        .map(|url| -> JoinHandle<SKResult<bool>> {
            download_file_async(url.to_string(), basedir, true)
        })
        .collect();
    // Wait for all the threads to funish
    for jh in vresult {
        jh.join().unwrap()?;
    }

    Ok(())
}

fn download_from_json(
    v: &JsonValue,
    basedir: std::path::PathBuf,
    baseurl: String,
    overwrite: &bool,
    thandles: &mut Vec<JoinHandle<SKResult<bool>>>,
) -> SKResult<()> {
    if v.is_object() {
        let r1: Vec<SKResult<()>> = v
            .entries()
            .map(|entry: (&str, &JsonValue)| -> SKResult<()> {
                let pbnew = basedir.join(entry.0);
                if !pbnew.is_dir() {
                    std::fs::create_dir_all(pbnew.clone())?;
                }
                let mut newurl = baseurl.clone();
                newurl.push_str(format!("/{}", entry.0).as_str());
                download_from_json(entry.1, pbnew.clone(), newurl, overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| match res {
                Ok(_) => false,
                Err(_) => true,
            })
            .collect();
        if r1.len() > 0 {
            return skerror!("Could not parse entries");
        }
    } else if v.is_array() {
        let r2: Vec<SKResult<()>> = v
            .members()
            .map(|val| -> SKResult<()> {
                download_from_json(val, basedir.clone(), baseurl.clone(), overwrite, thandles)?;
                Ok(())
            })
            .filter(|res| match res {
                Ok(_) => false,
                Err(_) => true,
            })
            .collect();
        if r2.len() > 0 {
            return skerror!("could not parse array entries");
        }
    } else if v.is_string() {
        let mut newurl = baseurl.clone();
        newurl.push_str(format!("/{}", v).as_str());
        thandles.push(download_file_async(newurl, &basedir, overwrite.clone()));
    } else {
        return skerror!("invalid json for downloading files??!!");
    }

    Ok(())
}

fn download_datadir(basedir: PathBuf, baseurl: String, overwrite: &bool) -> SKResult<()> {
    if !basedir.is_dir() {
        std::fs::create_dir_all(basedir.clone())?;
    }

    let mut fileurl = baseurl.clone();
    fileurl.push_str("/files.json");

    let json_base: JsonValue = json::parse(download_to_string(fileurl.as_str())?.as_str())?;
    let mut thandles: Vec<JoinHandle<SKResult<bool>>> = Vec::new();
    download_from_json(&json_base, basedir, baseurl, overwrite, &mut thandles)?;
    // Wait for all the threads to funish
    for jh in thandles {
        jh.join().unwrap()?;
    }
    Ok(())
}

pub fn update_datafiles(dir: Option<PathBuf>, overwrite_if_exists: bool) -> SKResult<()> {
    let downloaddir = match dir {
        Some(d) => d,
        None => datadir()?,
    };
    if downloaddir.metadata()?.permissions().readonly() {
        return skerror!(
            r#"
            Data directory is read-only.
            Try setting SATKIT_DATA environment
            variable to a writeable directory and re-starting
            "#
        );
    }

    println!(
        "Downloading data files to {}",
        downloaddir.clone().to_str().unwrap()
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

/*
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn update_data() {
        match update_datafiles(None, false) {
            Ok(()) => (),
            Err(e) => {
                println!("Error: {}", e.to_string());
                assert!(1 == 0);
            }
        }
    }
}
*/
