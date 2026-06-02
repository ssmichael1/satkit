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

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),
}

/// Convenient type alias used throughout the `download` module.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "download")]
pub fn download_if_not_exist(fname: &Path, seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        return Ok(());
    }
    let baseurl = seturl.unwrap_or("https://storage.googleapis.com/astrokit-astro-data/");
    let url = format!(
        "{}{}",
        baseurl,
        fname.file_name().unwrap().to_str().unwrap()
    );
    // Try to set proxy, if any, from environment variables
    let agent = ureq::Agent::new_with_defaults();

    let mut resp = agent.get(url.as_str()).call()?;

    let mut dest = std::fs::File::create(fname)?;
    std::io::copy(&mut resp.body_mut().as_reader(), &mut dest)?;
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
    let fname = std::path::Path::new(url).file_name().unwrap();
    let fullpath = downloaddir.join(fname);
    if fullpath.exists() && !overwrite_if_exists {
        println!("File {} exists; skipping download", fname.to_str().unwrap());
        return Ok(false);
    }

    let agent = ureq::Agent::new_with_defaults();
    let mut resp = agent.get(url).call()?;

    println!("Downloading {}", fname.to_str().unwrap());
    let mut dest = std::fs::File::create(fullpath)?;
    std::io::copy(&mut resp.body_mut().as_reader(), &mut dest)?;
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
