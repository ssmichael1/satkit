use crate::skerror;
use crate::SKResult;
use once_cell::sync::OnceCell;
#[cfg(feature = "pybindings")]
use process_path::get_dylib_path;
use std::path::Path;
use std::path::PathBuf;

pub fn testdirs() -> Vec<PathBuf> {
    let mut testdirs: Vec<PathBuf> = Vec::new();

    // Look for paths in environment variable
    match std::env::var(&"ASTRO_DATA") {
        Ok(val) => testdirs.push(Path::new(&val).to_path_buf()),
        Err(_) => (),
    }

    // Look for paths in current library directory
    #[cfg(feature = "pybindings")]
    match get_dylib_path() {
        Some(v) => {
            testdirs.push(Path::new(&v).parent().unwrap().join("astro-data"));
            testdirs.push(
                Path::new(&v)
                    .parent()
                    .unwrap()
                    .join("share")
                    .join("astro-data"),
            );
            testdirs.push(v);
        }
        None => (),
    }

    // Look for paths under home directory
    match std::env::var(&"HOME") {
        Ok(val) => {
            let vstr = &String::from(val);

            #[cfg(target_os = "macos")]
            testdirs.push(
                Path::new(vstr)
                    .join("Library")
                    .join("Application Support")
                    .join("astro-data"),
            );
            testdirs.push(Path::new(vstr).join("astro-data"));
            testdirs.push(Path::new(vstr).to_path_buf());
        }
        Err(_e) => (),
    }

    testdirs.push(Path::new(&"/usr/share/astrodata").to_path_buf());

    // On mac, look in root library directory
    #[cfg(target_os = "macos")]
    testdirs.push(Path::new(&"/Library/Application Support/astro-data").to_path_buf());

    testdirs
}

/// Get directory where astronomy data is stored
///
/// Tries the following paths in order, and stops when the
/// files are found
///
/// *  "ASTRO_DATA" environment variable
/// *  ${HOME}/astro-data
/// *  ${HOME}
/// *  /usr/share/astro-data
/// *  On Mac Only:
///    * /Library/Application Support/astro-data
///    * ${Home}/Library/Application Support/astro-data
///
/// Returns:
///
///  * SKResult<<std::path::PathBuf>> representing directory
///    where files are stored
///
pub fn datadir() -> SKResult<PathBuf> {
    static INSTANCE: OnceCell<Option<PathBuf>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        // Check for already-populated directory
        for ref dir in testdirs() {
            let p = PathBuf::from(&dir).join("tab5.2a.txt");
            if p.is_file() {
                return Some(dir.to_path_buf().clone());
            }
        }
        // Check for directory that we can create that is writable
        for ref dir in testdirs() {
            match std::fs::create_dir_all(dir) {
                Ok(()) => return Some(dir.to_path_buf().clone()),
                Err(_) => {}
            }
        }

        // Check for writeable directory
        for ref dir in testdirs() {
            if dir.is_dir() {
                if !dir.metadata().unwrap().permissions().readonly() {
                    return Some(dir.to_path_buf().clone());
                }
            }
        }

        None
    });
    match res.as_ref() {
        Some(v) => Ok(v.clone()),
        None => skerror!("Could not find valid writeable data directory"),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn datadir() {
        use crate::utils::datadir;
        let d = datadir::datadir();
        println!("d = {:?}", d.as_ref().unwrap());
        assert_eq!(d.is_err(), false);
    }
}
