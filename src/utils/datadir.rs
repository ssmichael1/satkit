//! Where satkit's data files live.
//!
//! Two questions are answered separately:
//!
//! * **Where to look** for a file — [`search_dirs`] / [`find_file`]: an
//!   ordered list of read candidates. Any of them may be read-only (a
//!   system-wide `/usr/share/satkit-data`, or the optional `satkit-data`
//!   Python package inside `site-packages`).
//! * **Where to write** downloads — [`datadir`]: exactly one directory, the
//!   platform user-data directory unless `SATKIT_DATA` (or [`set_datadir`])
//!   overrides it. satkit never creates a directory next to its own shared
//!   library or inside `site-packages`: such a directory would be wiped on
//!   reinstall and is often not writable.
//!
//! The core data (IERS nutation tables, gravity models) is compiled into the
//! library, so a data directory is only needed for the JPL ephemeris and the
//! regularly refreshed Earth-orientation / space-weather files.
//!
//! # Search order
//!
//! 1. `SATKIT_DATA` (environment) — also the write location when set
//! 2. a directory given to [`set_datadir`] — also the write location
//! 3. directories added with [`add_search_dir`] (e.g. by the Python package
//!    when the optional `satkit_data` bundle is importable)
//! 4. `<dir of the satkit shared library>/satkit-data`
//! 5. `<site-packages>/satkit_data/data` (the `satkit-data` pip package,
//!    found relative to the shared library)
//! 6. the platform user-data directory (the default write location):
//!    macOS `~/Library/Application Support/satkit-data`,
//!    Linux/other `$XDG_DATA_HOME/satkit-data` (default
//!    `~/.local/share/satkit-data`), Windows `%LOCALAPPDATA%\satkit-data`
//! 7. `~/.satkit-data` (legacy location, read only)
//! 8. `/usr/share/satkit-data`
//! 9. macOS: `/Library/Application Support/satkit-data`
//!
//! The resolution is a pure function of the environment ([`resolve`]) so it
//! can be tested for every platform without touching the file system.

use process_path::get_dylib_path;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use thiserror::Error;

/// Errors that can occur while resolving or setting the satkit data directory.
#[derive(Debug, Error)]
pub enum Error {
    /// The path passed to [`set_datadir`] is not an existing directory.
    #[error("Data directory does not exist")]
    DirectoryDoesNotExist,

    /// The data-directory singleton has already been initialized and cannot
    /// be overwritten.
    #[error("Could not set data directory")]
    SetFailed,

    /// No write location could be determined or created (no `SATKIT_DATA`,
    /// no home / `LOCALAPPDATA` directory, and the fallback could not be
    /// created).
    #[error("Could not find or create a writeable data directory (set SATKIT_DATA)")]
    NoWriteableDirectory,
}

/// Convenient type alias used throughout the `datadir` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Directory name used under every platform location.
const DIR_NAME: &str = "satkit-data";

/// Explicit override from [`set_datadir`] (search-first and write location).
static EXPLICIT: Mutex<Option<PathBuf>> = Mutex::new(None);
/// Extra read-only search directories from [`add_search_dir`].
static EXTRA_SEARCH: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
/// Cached write location (resolved once; cleared by [`set_datadir`]).
static WRITE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// The environment inputs the resolver depends on, captured so the
/// resolution can be unit-tested per platform.
#[derive(Debug, Clone, Default)]
pub struct Env {
    /// `SATKIT_DATA`
    pub satkit_data: Option<PathBuf>,
    /// Directory containing the satkit shared library, if known.
    pub dylib_dir: Option<PathBuf>,
    /// `HOME` (Unix) / user profile.
    pub home: Option<PathBuf>,
    /// `XDG_DATA_HOME` (Linux).
    pub xdg_data_home: Option<PathBuf>,
    /// `LOCALAPPDATA` (Windows).
    pub local_app_data: Option<PathBuf>,
    /// Target OS: `"macos"`, `"windows"`, or anything else (treated as
    /// Linux/Unix).
    pub os: &'static str,
}

impl Env {
    /// Capture the live process environment.
    pub fn current() -> Self {
        let var = |k: &str| {
            std::env::var_os(k)
                .map(PathBuf::from)
                .filter(|p| !p.as_os_str().is_empty())
        };
        Self {
            satkit_data: var("SATKIT_DATA"),
            dylib_dir: get_dylib_path().and_then(|p| Path::new(&p).parent().map(Path::to_path_buf)),
            home: var("HOME").or_else(|| var("USERPROFILE")),
            xdg_data_home: var("XDG_DATA_HOME"),
            local_app_data: var("LOCALAPPDATA"),
            os: std::env::consts::OS,
        }
    }
}

/// Result of [`resolve`]: the read candidates in order and the write location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// Directories to look in, in order (may include the write dir).
    pub search: Vec<PathBuf>,
    /// The single directory downloads go to, if one could be determined.
    pub write: Option<PathBuf>,
}

/// The platform user-data directory for satkit, if the environment gives one.
fn user_data_dir(env: &Env) -> Option<PathBuf> {
    match env.os {
        "macos" => env
            .home
            .as_ref()
            .map(|h| h.join("Library").join("Application Support").join(DIR_NAME)),
        "windows" => env
            .local_app_data
            .as_ref()
            .map(|d| d.join(DIR_NAME))
            .or_else(|| {
                env.home
                    .as_ref()
                    .map(|h| h.join("AppData").join("Local").join(DIR_NAME))
            }),
        _ => env
            .xdg_data_home
            .as_ref()
            .map(|d| d.join(DIR_NAME))
            .or_else(|| {
                env.home
                    .as_ref()
                    .map(|h| h.join(".local").join("share").join(DIR_NAME))
            }),
    }
}

/// Pure resolution of search and write locations from `env` plus the
/// explicit override and extra search directories (see the
/// [module docs](self) for the order).
pub fn resolve(env: &Env, explicit: Option<&Path>, extra: &[PathBuf]) -> Resolved {
    let mut search: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !search.contains(&p) {
            search.push(p);
        }
    };
    if let Some(d) = &env.satkit_data {
        push(d.clone());
    }
    if let Some(d) = explicit {
        push(d.to_path_buf());
    }
    for d in extra {
        push(d.clone());
    }
    if let Some(dylib) = &env.dylib_dir {
        push(dylib.join(DIR_NAME));
        if let Some(site_packages) = dylib.parent() {
            push(site_packages.join("satkit_data").join("data"));
        }
    }
    let user_dir = user_data_dir(env);
    if let Some(d) = &user_dir {
        push(d.clone());
    }
    if let Some(h) = &env.home {
        push(h.join(".satkit-data"));
    }
    if env.os != "windows" {
        push(PathBuf::from("/usr/share").join(DIR_NAME));
    }
    if env.os == "macos" {
        push(PathBuf::from("/Library/Application Support").join(DIR_NAME));
    }
    let write = env
        .satkit_data
        .clone()
        .or_else(|| explicit.map(Path::to_path_buf))
        .or(user_dir);
    Resolved { search, write }
}

fn resolved_now() -> Resolved {
    let explicit = EXPLICIT.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let extra = EXTRA_SEARCH
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    resolve(&Env::current(), explicit.as_deref(), &extra)
}

/// The directories searched for data files, in order (see module docs).
pub fn search_dirs() -> Vec<PathBuf> {
    resolved_now().search
}

/// Find `name` in the first search directory that contains it.
pub fn find_file(name: &str) -> Option<PathBuf> {
    search_dirs()
        .into_iter()
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// The path to use for `name`: where it was found, or where it would be
/// written (the write location, or `.` if none can be determined).
pub fn path_for(name: &str) -> PathBuf {
    find_file(name).unwrap_or_else(|| datadir().unwrap_or_else(|_| PathBuf::from(".")).join(name))
}

/// Add a read-only search directory (tried after `SATKIT_DATA` and
/// [`set_datadir`], before the platform locations). Used by the Python
/// package to register the optional `satkit_data` bundle wherever it is
/// installed.
pub fn add_search_dir(d: &Path) {
    let mut v = EXTRA_SEARCH.lock().unwrap_or_else(|e| e.into_inner());
    if !v.iter().any(|x| x == d) {
        v.push(d.to_path_buf());
    }
}

/// Explicitly set the data directory: it becomes the first search location
/// (after `SATKIT_DATA`) and the write location for downloads.
pub fn set_datadir(d: &Path) -> Result<()> {
    if !d.is_dir() {
        return Err(Error::DirectoryDoesNotExist);
    }
    *EXPLICIT.lock().unwrap_or_else(|e| e.into_inner()) = Some(d.to_path_buf());
    *WRITE_DIR.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(())
}

/// The directory downloads are written to (the platform user-data
/// directory, or `SATKIT_DATA` / [`set_datadir`] when given), created on
/// first use if necessary. Files are *looked up* across all of
/// [`search_dirs`]; this is only the write location.
pub fn datadir() -> Result<PathBuf> {
    let mut cache = WRITE_DIR.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(d) = cache.as_ref() {
        return Ok(d.clone());
    }
    let Some(dir) = resolved_now().write else {
        return Err(Error::NoWriteableDirectory);
    };
    if !dir.is_dir() {
        std::fs::create_dir_all(&dir).map_err(|_| Error::NoWriteableDirectory)?;
    }
    *cache = Some(dir.clone());
    Ok(dir)
}

/// `true` if `dir` holds a JPL ephemeris file (`linux_p*.4XX` /
/// `lnxp*.4XX`) — the one data file that is neither compiled in nor
/// refreshed daily, and therefore the marker of a provisioned data location.
pub fn has_ephemeris(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    rd.flatten()
        .any(|e| is_ephemeris_name(&e.file_name().to_string_lossy()))
}

/// `true` for JPL Linux-binary ephemeris file names (`linux_p1550p2650.440`,
/// `lnxp1900p2053.421`, ...).
pub fn is_ephemeris_name(name: &str) -> bool {
    (name.starts_with("linux_p") || name.starts_with("lnxp"))
        && name.rsplit('.').next().is_some_and(|ext| {
            ext.len() == 3 && ext.starts_with('4') && ext.chars().all(|c| c.is_ascii_digit())
        })
}

/// Return true if a JPL ephemeris file is present in any search directory.
pub fn data_found() -> bool {
    search_dirs().iter().any(|d| has_ephemeris(d))
}

/// Legacy name for [`search_dirs`].
#[deprecated(note = "use search_dirs()")]
pub fn testdirs() -> Vec<PathBuf> {
    search_dirs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(os: &'static str) -> Env {
        Env {
            satkit_data: None,
            dylib_dir: Some(PathBuf::from("/venv/lib/python3.13/site-packages/satkit")),
            home: Some(PathBuf::from("/home/u")),
            xdg_data_home: None,
            local_app_data: None,
            os,
        }
    }

    #[test]
    fn offline_resolve_macos_order_and_write_dir() {
        let r = resolve(&env("macos"), None, &[]);
        assert_eq!(
            r.write.as_deref(),
            Some(Path::new("/home/u/Library/Application Support/satkit-data"))
        );
        assert_eq!(
            r.search,
            vec![
                PathBuf::from("/venv/lib/python3.13/site-packages/satkit/satkit-data"),
                PathBuf::from("/venv/lib/python3.13/site-packages/satkit_data/data"),
                PathBuf::from("/home/u/Library/Application Support/satkit-data"),
                PathBuf::from("/home/u/.satkit-data"),
                PathBuf::from("/usr/share/satkit-data"),
                PathBuf::from("/Library/Application Support/satkit-data"),
            ]
        );
    }

    #[test]
    fn offline_resolve_linux_xdg() {
        let mut e = env("linux");
        let r = resolve(&e, None, &[]);
        assert_eq!(
            r.write.as_deref(),
            Some(Path::new("/home/u/.local/share/satkit-data"))
        );
        assert!(!r.search.iter().any(|p| p.starts_with("/Library")));
        e.xdg_data_home = Some(PathBuf::from("/xdg"));
        let r = resolve(&e, None, &[]);
        assert_eq!(r.write.as_deref(), Some(Path::new("/xdg/satkit-data")));
        assert_eq!(r.search[2], PathBuf::from("/xdg/satkit-data"));
        assert!(
            r.search.contains(&PathBuf::from("/home/u/.satkit-data")),
            "legacy read candidate"
        );
    }

    #[test]
    fn offline_resolve_windows_localappdata() {
        let mut e = env("windows");
        e.home = None;
        e.local_app_data = Some(PathBuf::from(r"C:\Users\u\AppData\Local"));
        let r = resolve(&e, None, &[]);
        assert_eq!(
            r.write.as_deref(),
            Some(
                Path::new(r"C:\Users\u\AppData\Local")
                    .join(DIR_NAME)
                    .as_path()
            )
        );
        assert!(!r.search.iter().any(|p| p.starts_with("/usr/share")));
        // With neither LOCALAPPDATA nor a home dir, there is no write location
        // — and never one inside site-packages or next to the dylib.
        e.local_app_data = None;
        let r = resolve(&e, None, &[]);
        assert_eq!(r.write, None);
        assert!(r
            .search
            .iter()
            .all(|p| !p.ends_with("satkit-data") || p.starts_with("/venv")));
    }

    #[test]
    fn offline_resolve_precedence_of_overrides() {
        let mut e = env("linux");
        e.satkit_data = Some(PathBuf::from("/data"));
        let r = resolve(&e, Some(Path::new("/explicit")), &[PathBuf::from("/extra")]);
        assert_eq!(
            r.write.as_deref(),
            Some(Path::new("/data")),
            "SATKIT_DATA wins for writes"
        );
        assert_eq!(
            &r.search[..3],
            &[
                PathBuf::from("/data"),
                PathBuf::from("/explicit"),
                PathBuf::from("/extra")
            ]
        );
        e.satkit_data = None;
        let r = resolve(&e, Some(Path::new("/explicit")), &[]);
        assert_eq!(
            r.write.as_deref(),
            Some(Path::new("/explicit")),
            "set_datadir is the write dir when no env"
        );
        let r = resolve(&e, None, &[PathBuf::from("/extra")]);
        assert_eq!(
            r.write.as_deref(),
            Some(Path::new("/home/u/.local/share/satkit-data")),
            "extra search dirs never become the write dir"
        );
    }

    #[test]
    fn offline_ephemeris_names() {
        assert!(is_ephemeris_name("linux_p1550p2650.440"));
        assert!(is_ephemeris_name("lnxp1900p2053.421"));
        assert!(!is_ephemeris_name("linux_p1550p2650.440.part"));
        assert!(!is_ephemeris_name("EGM96.gfc"));
    }

    #[test]
    fn datadir_resolves() {
        let d = datadir();
        println!("d = {:?}", d.as_ref().unwrap());
        assert!(d.is_ok());
    }
}
