use std::path::Path;
use thiserror::Error;

/// Errors produced by the [`utils::download`](crate::utils::download) helpers.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned by all download helpers when satkit was built without the
    /// `download` Cargo feature.
    #[error("satkit was built without the `download` feature")]
    FeatureDisabled,

    /// Returned by [`download_if_not_exist`] when the requested file is
    /// missing on disk and satkit was built without the `download` feature
    /// to fetch it.
    #[error("File {path} not found and satkit was built without the `download` feature")]
    FileNotFoundNoDownload { path: String },

    /// Returned when a download path or URL has no valid UTF-8 file-name
    /// component (e.g. ends in `/` or `..`).
    #[error("Path or URL has no valid file name: {path}")]
    InvalidFileName { path: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),
}

/// Convenient type alias used throughout the `download` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Stream `reader` to `final_path` atomically: write to a sibling `.part` file
/// and rename it into place on success. An interrupted transfer (network drop,
/// Ctrl-C) then leaves only the discardable `.part` file rather than a truncated
/// final file that later runs would trust as complete.
#[cfg(feature = "download")]
fn write_atomic(reader: &mut impl std::io::Read, final_path: &Path) -> Result<()> {
    let part_path = {
        let mut p = final_path.as_os_str().to_owned();
        p.push(".part");
        std::path::PathBuf::from(p)
    };
    let mut write = || -> Result<()> {
        let mut dest = std::fs::File::create(&part_path)?;
        std::io::copy(reader, &mut dest)?;
        dest.sync_all()?;
        Ok(())
    };
    match write() {
        Ok(()) => {
            std::fs::rename(&part_path, final_path)?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part_path);
            Err(e)
        }
    }
}

#[cfg(feature = "download")]
pub fn download_if_not_exist(fname: &Path, seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        return Ok(());
    }
    let baseurl = seturl.unwrap_or("https://storage.googleapis.com/astrokit-astro-data/");
    let basename =
        fname
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| Error::InvalidFileName {
                path: fname.display().to_string(),
            })?;
    let url = format!("{}{}", baseurl, basename);
    // Try to set proxy, if any, from environment variables
    let agent = ureq::Agent::new_with_defaults();

    let mut resp = agent.get(url.as_str()).call()?;

    write_atomic(&mut resp.body_mut().as_reader(), fname)?;
    Ok(())
}

#[cfg(not(feature = "download"))]
pub fn download_if_not_exist(fname: &Path, _seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        Ok(())
    } else {
        Err(Error::FileNotFoundNoDownload {
            path: fname.display().to_string(),
        })
    }
}

#[cfg(feature = "download")]
pub fn download_file(url: &str, downloaddir: &Path, overwrite_if_exists: bool) -> Result<bool> {
    let fname = std::path::Path::new(url)
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| Error::InvalidFileName {
            path: url.to_string(),
        })?;
    let fullpath = downloaddir.join(fname);
    if fullpath.exists() && !overwrite_if_exists {
        println!("File {} exists; skipping download", fname);
        return Ok(false);
    }

    let agent = ureq::Agent::new_with_defaults();
    let mut resp = agent.get(url).call()?;

    println!("Downloading {}", fname);
    write_atomic(&mut resp.body_mut().as_reader(), &fullpath)?;
    Ok(true)
}

#[cfg(not(feature = "download"))]
pub fn download_file(_url: &str, _downloaddir: &Path, _overwrite_if_exists: bool) -> Result<bool> {
    Err(Error::FeatureDisabled)
}

#[cfg(feature = "download")]
pub fn download_file_async(
    url: String,
    downloaddir: &Path,
    overwrite_if_exists: bool,
) -> std::thread::JoinHandle<Result<bool>> {
    let dclone = downloaddir.to_path_buf();
    let urlclone = url;
    let overwriteclone = overwrite_if_exists;
    std::thread::spawn(move || download_file(urlclone.as_str(), &dclone, overwriteclone))
}

#[cfg(not(feature = "download"))]
pub fn download_file_async(
    _url: String,
    _downloaddir: &Path,
    _overwrite_if_exists: bool,
) -> std::thread::JoinHandle<Result<bool>> {
    std::thread::spawn(|| Err(Error::FeatureDisabled))
}

#[cfg(feature = "download")]
pub fn download_to_string(url: &str) -> Result<String> {
    let agent = ureq::Agent::new_with_defaults();
    let mut resp = agent.get(url).call()?;
    let thestring = std::io::read_to_string(resp.body_mut().as_reader())?;
    Ok(thestring)
}

#[cfg(not(feature = "download"))]
pub fn download_to_string(_url: &str) -> Result<String> {
    Err(Error::FeatureDisabled)
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn write_atomic_renames_and_leaves_no_part_file() {
        let dir = std::env::temp_dir();
        let final_path = dir.join("satkit_write_atomic_test.bin");
        let part_path = dir.join("satkit_write_atomic_test.bin.part");
        let _ = std::fs::remove_file(&final_path);
        let _ = std::fs::remove_file(&part_path);

        let data = b"hello satkit";
        write_atomic(&mut Cursor::new(&data[..]), &final_path).unwrap();

        assert!(
            final_path.is_file(),
            "final file should exist after success"
        );
        assert!(!part_path.exists(), "the .part file must be renamed away");
        assert_eq!(std::fs::read(&final_path).unwrap(), data);
        let _ = std::fs::remove_file(&final_path);
    }
}
