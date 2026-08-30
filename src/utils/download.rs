use std::path::Path;
use thiserror::Error;

/// Errors produced by the [`utils::download`](crate::utils::download) and
/// [`utils::manifest`](crate::utils::manifest) helpers.
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

    /// Returned when a download path, URL or manifest name has no valid
    /// single-component file name (e.g. ends in `/`, is absolute, or contains
    /// `..`).
    #[error("Path or URL has no valid file name: {path}")]
    InvalidFileName { path: String },

    /// The embedded (or a supplied) data manifest failed validation.
    #[error("Data manifest is invalid: {reason}")]
    ManifestInvalid { reason: String },

    /// A downloaded file's size or SHA-256 did not match the manifest.
    #[error(
        "{name} from {url}: {what} mismatch (expected {expected}, got {actual}); \
         the partial download was discarded"
    )]
    HashMismatch {
        name: String,
        url: String,
        what: &'static str,
        expected: String,
        actual: String,
    },

    /// Every candidate URL for a manifest file failed; `attempts` holds one
    /// `"<url>: <error>"` line per URL tried.
    #[error("Could not download {name} from any source:\n  {}", attempts.join("\n  "))]
    AllSourcesFailed { name: String, attempts: Vec<String> },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),
}

/// Convenient type alias used throughout the `download` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Path of the sibling `.part` file used for atomic writes.
fn part_path(final_path: &Path) -> std::path::PathBuf {
    let mut p = final_path.as_os_str().to_owned();
    p.push(".part");
    std::path::PathBuf::from(p)
}

/// Stream `reader` to `final_path` atomically: write to a sibling `.part` file
/// and rename it into place on success. An interrupted transfer (network drop,
/// Ctrl-C) then leaves only the discardable `.part` file rather than a truncated
/// final file that later runs would trust as complete.
#[cfg(feature = "download")]
fn write_atomic(reader: &mut impl std::io::Read, final_path: &Path) -> Result<()> {
    let part = part_path(final_path);
    let mut write = || -> Result<()> {
        let mut dest = std::fs::File::create(&part)?;
        std::io::copy(reader, &mut dest)?;
        dest.sync_all()?;
        Ok(())
    };
    match write() {
        Ok(()) => {
            std::fs::rename(&part, final_path)?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

/// Like [`write_atomic`], but the stream is hashed as it is written and the
/// `.part` file is only renamed into place if its size and SHA-256 match
/// `entry`. On mismatch the partial file is deleted and
/// [`Error::HashMismatch`] is returned, so a corrupt or substituted download
/// never becomes a trusted data file. Used by
/// [`manifest::fetch_static_file`](crate::utils::manifest::fetch_static_file).
pub(crate) fn write_atomic_verified(
    reader: &mut impl std::io::Read,
    final_path: &Path,
    entry: &crate::utils::manifest::ManifestEntry,
    url: &str,
) -> Result<()> {
    use sha2::{Digest, Sha256};
    let part = part_path(final_path);
    let mut write = || -> Result<(u64, String)> {
        let mut dest = std::fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut total: u64 = 0;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            std::io::Write::write_all(&mut dest, &buf[..n])?;
            total += n as u64;
        }
        dest.sync_all()?;
        let digest = hasher.finalize();
        Ok((total, digest.iter().map(|b| format!("{b:02x}")).collect()))
    };
    let outcome = write();
    let (size, sha) = match outcome {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            return Err(e);
        }
    };
    let mismatch = |what: &'static str, expected: String, actual: String| {
        let _ = std::fs::remove_file(&part);
        Error::HashMismatch {
            name: entry.name.clone(),
            url: url.to_string(),
            what,
            expected,
            actual,
        }
    };
    if size != entry.size {
        return Err(mismatch("size", entry.size.to_string(), size.to_string()));
    }
    if sha != entry.sha256 {
        return Err(mismatch("sha256", entry.sha256.clone(), sha));
    }
    std::fs::rename(&part, final_path)?;
    Ok(())
}

/// Ensure `fname` exists, downloading it if necessary.
///
/// * With `seturl == None` the file's base name is looked up in the embedded
///   [data manifest](crate::utils::manifest): a known static file is fetched
///   through [`fetch_static_file`](crate::utils::manifest::fetch_static_file)
///   (release asset → origin → legacy bucket, SHA-256 verified). A name that
///   is not in the manifest falls back to an *unverified* fetch from the
///   legacy bucket, so user-supplied alternative files keep working.
/// * With `seturl == Some(base)` (the celestrak refresh files) the file is
///   fetched unverified from `base + name`, as before.
#[cfg(feature = "download")]
pub fn download_if_not_exist(fname: &Path, seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        return Ok(());
    }
    let basename =
        fname
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| Error::InvalidFileName {
                path: fname.display().to_string(),
            })?;
    if seturl.is_none() {
        if let Some(entry) = crate::utils::manifest::embedded().entry(basename) {
            let dir = fname.parent().unwrap_or_else(|| Path::new("."));
            crate::utils::manifest::fetch_static_file(entry, dir, false)?;
            return Ok(());
        }
        eprintln!(
            "Warning: {basename} is not in satkit's data manifest; downloading it unverified"
        );
    }
    let baseurl = seturl.unwrap_or("https://storage.googleapis.com/astrokit-astro-data/");
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

/// Download `url` into `downloaddir` (unverified; used for the regularly
/// refreshed EOP / space-weather files). Returns `Ok(false)` if the file
/// already exists and `overwrite_if_exists` is false.
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

    #[test]
    fn write_atomic_verified_rejects_bad_bytes() {
        use crate::utils::manifest::{sha256_hex, ManifestEntry};
        let dir = std::env::temp_dir().join(format!("satkit_wav_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let final_path = dir.join("f.bin");
        let entry = ManifestEntry {
            name: "f.bin".into(),
            size: 5,
            sha256: sha256_hex(b"hello"),
            urls: vec!["https://example.invalid/f.bin".into()],
            source: String::new(),
            license: String::new(),
            tier: String::new(),
            default: true,
        };
        // Wrong bytes, right length -> sha mismatch, nothing left behind.
        let err =
            write_atomic_verified(&mut Cursor::new(&b"hellp"[..]), &final_path, &entry, "test")
                .unwrap_err();
        assert!(
            matches!(err, Error::HashMismatch { what: "sha256", .. }),
            "{err}"
        );
        assert!(!final_path.exists() && !part_path(&final_path).exists());
        // Wrong length -> size mismatch.
        let err = write_atomic_verified(
            &mut Cursor::new(&b"hello!"[..]),
            &final_path,
            &entry,
            "test",
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::HashMismatch { what: "size", .. }),
            "{err}"
        );
        // Right bytes -> file in place.
        write_atomic_verified(&mut Cursor::new(&b"hello"[..]), &final_path, &entry, "test")
            .unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"hello");
        assert!(!part_path(&final_path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
