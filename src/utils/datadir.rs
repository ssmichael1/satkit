#[cfg(feature = "pybindings")]
use process_path::get_dylib_path;
use std::cell::OnceCell;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{bail, Result};

// Pointer to the one and only data directory
static DATADIR_SINGLETON: Mutex<OnceCell<Option<PathBuf>>> = Mutex::new(OnceCell::new());

pub fn testdirs() -> Vec<PathBuf> {
    let mut testdirs: Vec<PathBuf> = Vec::new();

    // Look for paths in environment variable
    if let Ok(val) = std::env::var("SATKIT_DATA") {
        testdirs.push(Path::new(&val).to_path_buf())
    }

    // Look for paths in current library directory
    #[cfg(feature = "pybindings")]
    if let Some(v) = get_dylib_path() {
        testdirs.push(Path::new(&v).parent().unwrap().join("satkit-data"));
    }

    // Look for paths under home directory
    match std::env::var("HOME") {
        Ok(val) => {
            let vstr = &val;

            #[cfg(target_os = "macos")]
            testdirs.push(
                Path::new(vstr)
                    .join("Library")
                    .join("Application Support")
                    .join("satkit-data"),
            );
            testdirs.push(Path::new(vstr).join(".satkit-data"));
        }
        Err(_e) => (),
    }

    testdirs.push(Path::new(&"/usr/share/satkit-data").to_path_buf());

    // On mac, look in root library directory
    #[cfg(target_os = "macos")]
    testdirs.push(Path::new(&"/Library/Application Support/satkit-data").to_path_buf());

    testdirs
}

/// Explicitly set data directory where data files will be stored
/// Generally this should not be needed
pub fn set_datadir(d: &Path) -> Result<()> {
    if !d.is_dir() {
        bail!("Data directory does not exist");
    }

    let mut dd = DATADIR_SINGLETON.lock().unwrap();
    dd.take();
    match dd.set(Some(d.to_path_buf())) {
        Ok(_) => Ok(()),
        Err(_) => bail!("Could not set data directory"),
    }
}

/// Get directory where astronomy data is stored
///
/// Tries the following paths in order, and stops when the
/// files are found
///
/// *  "SATKIT_DATA" environment variable
/// *  ${HOME}/Library/Application Support/satkit-data (on MacOS only)
/// *  ${HOME}/.satkit-data
/// *  /usr/share/satkit-data
/// *  /Library/Application Support/satkit-data (on MacOS only)
///
/// Returns:
///
///  * anyhow::Result<<std::path::PathBuf>> representing directory
///    where files are stored
pub fn datadir() -> Result<PathBuf> {
    let dd = DATADIR_SINGLETON.lock().unwrap();
    let res = dd.get_or_init(|| {
        let td: Vec<PathBuf> = testdirs();

        // Check for already-populated directory
        for ref dir in td.clone() {
            let p = PathBuf::from(&dir).join("tab5.2a.txt");
            if p.is_file() {
                return Some(dir.to_path_buf());
            }
        }

        // Check for writeable directory that already exists
        for ref dir in td.clone() {
            if dir.is_dir() && !dir.metadata().unwrap().permissions().readonly() {
                return Some(dir.to_path_buf());
            }
        }

        // Check for directory that we can create that is writable
        for ref dir in td {
            if matches!(std::fs::create_dir_all(dir), Ok(())) {
                return Some(dir.to_path_buf());
            }
        }

        None
    });

    match res.as_ref() {
        Some(v) => Ok(v.clone()),
        None => bail!("Could not find valid writeable data directory"),
    }
}

/// Return true if data files exist in data directory
pub fn data_found() -> bool {
    match datadir() {
        Ok(d) => {
            let p = PathBuf::from(&d).join("tab5.2a.txt");
            p.is_file()
        }
        // If can't find or create data directory, the data has not been found
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn datadir() {
        use crate::utils::datadir;
        let d = datadir::datadir();
        println!("d = {:?}", d.as_ref().unwrap());
        assert!(d.is_ok());
    }
}
