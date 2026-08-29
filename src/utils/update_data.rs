//! Download / refresh the data files satkit needs.
//!
//! Static files (ephemeris, IERS tables, gravity coefficients, leap-second
//! list) come from the embedded [data manifest](crate::utils::manifest)
//! and are SHA-256 verified; the regularly updated files (EOP, space weather)
//! are listed in the manifest's `refresh` section and fetched unverified from
//! celestrak on every run. See `data/README.md` for the design.

use super::download::{self, download_file_async};
use super::manifest::{self, FetchOutcome};
use crate::utils::datadir;
use std::path::PathBuf;
use std::thread::JoinHandle;
use thiserror::Error;

/// Errors produced by [`update_datafiles`].
#[derive(Debug, Error)]
pub enum Error {
    /// A refresh-manifest URL did not use `https://`.
    #[error("Manifest URL {url:?} must use https://")]
    InsecureManifestUrl { url: String },

    /// A manifest file name was not a single plain path component
    /// (absolute, contained `..`, or contained a path separator). Such a
    /// name would be joined onto the data directory and could escape it.
    #[error("Manifest file name {name:?} is not a plain path component")]
    InvalidManifestPath { name: String },

    /// The configured data directory is read-only and cannot receive
    /// new or refreshed files.
    #[error(
        "Data directory is read-only. Try setting SATKIT_DATA environment variable \
         to a writeable directory and re-starting"
    )]
    DataDirReadOnly,

    /// A worker thread launched by [`download_file_async`] or the static
    /// fetch panicked.
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

/// Fetch every default static file of the embedded manifest into `dir`,
/// in parallel, verifying each against its pinned size and SHA-256.
///
/// Returns one `(name, outcome)` per file. `force` re-downloads even when a
/// matching file is already present.
pub fn download_static_files(
    dir: &std::path::Path,
    force: bool,
) -> Result<Vec<(String, FetchOutcome)>> {
    let m = manifest::embedded();
    let handles: Vec<(String, JoinHandle<download::Result<FetchOutcome>>)> = m
        .default_files()
        .map(|entry| {
            let entry = entry.clone();
            let dir = dir.to_path_buf();
            let name = entry.name.clone();
            (
                name,
                std::thread::spawn(move || manifest::fetch_static_file(&entry, &dir, force)),
            )
        })
        .collect();
    let mut out = Vec::with_capacity(handles.len());
    for (name, jh) in handles {
        let outcome = jh.join().map_err(|_| Error::ThreadPanic)??;
        out.push((name, outcome));
    }
    Ok(out)
}

/// Download the regularly refreshed files (EOP, space weather) listed in the
/// manifest's `refresh` section, always overwriting.
fn download_refresh_files(dir: &std::path::Path) -> Result<()> {
    let m = manifest::embedded();
    let handles: Vec<JoinHandle<download::Result<bool>>> = m
        .refresh
        .iter()
        .map(|url| -> Result<_> {
            if !url.starts_with("https://") {
                return Err(Error::InsecureManifestUrl { url: url.clone() });
            }
            Ok(download_file_async(url.clone(), dir, true))
        })
        .collect::<Result<Vec<_>>>()?;
    for jh in handles {
        jh.join().map_err(|_| Error::ThreadPanic)??;
    }
    Ok(())
}

///
/// Download and update any necessary data files for "satkit" calculations
///
/// # Arguments
/// dir: The directory to download to, optional.  If not provided, the default data directory is used.
/// overwrite_if_exists: If true, re-download static files even when a verified copy is present.
///   If false, a static file whose size and SHA-256 already match the manifest is left alone.
///
/// # Returns
/// Result<()>
///
/// # Notes
///
/// Static files (JPL ephemeris, IERS nutation tables, gravity coefficients,
/// leap-second list) are described by the embedded
/// [data manifest](crate::utils::manifest): each is fetched from the first
/// working source (`SATKIT_DATA_URL` mirror if set, then the GitHub release
/// asset, the origin server, and the legacy bucket) and is only accepted
/// when its size and SHA-256 match the manifest.
///
/// The space weather and Earth orientation files are refreshed from
/// celestrak on every call, and the NOAA solar-cycle forecast is fetched;
/// these change daily and are not pinned.
///
pub fn update_datafiles(dir: Option<PathBuf>, overwrite_if_exists: bool) -> Result<()> {
    let downloaddir = match dir {
        Some(d) => d,
        None => datadir()?,
    };
    if !downloaddir.is_dir() {
        std::fs::create_dir_all(&downloaddir)?;
    }
    if downloaddir.metadata()?.permissions().readonly() {
        return Err(Error::DataDirReadOnly);
    }

    let m = manifest::embedded();
    println!(
        "Downloading data files ({}) to {}",
        m.data_version,
        downloaddir.to_string_lossy()
    );
    if let Some(mirror) = manifest::mirror_base() {
        println!("  {} = {mirror} (tried first)", manifest::MIRROR_ENV);
    }
    for (name, outcome) in download_static_files(&downloaddir, overwrite_if_exists)? {
        match outcome {
            FetchOutcome::AlreadyPresent => println!("  {name}: present and verified"),
            FetchOutcome::Downloaded { url } => println!("  {name}: downloaded from {url}"),
        }
    }

    println!("Now downloading files that are regularly updated:");
    println!("  Space Weather & Earth Orientation Parameters");
    download_refresh_files(&downloaddir)?;

    println!("  Solar Cycle Forecast");
    if let Err(e) = crate::solar_cycle_forecast::update() {
        eprintln!("Warning: could not download solar cycle forecast: {e}");
    }

    // Refresh the in-memory space-weather / EOP singletons from the files just
    // downloaded, so a process whose lazy first load failed (e.g. it started
    // before the data directory was populated) recovers without a restart.
    let sw_path = downloaddir.join("SW-All.csv");
    if sw_path.is_file() {
        if let Err(e) = crate::spaceweather::init_from_path(&sw_path) {
            eprintln!("Warning: could not load downloaded space-weather file: {e}");
        }
    }
    let eop_path = downloaddir.join("EOP-All.csv");
    if eop_path.is_file() {
        if let Err(e) = crate::earth_orientation_params::init_from_path(&eop_path) {
            eprintln!("Warning: could not load downloaded EOP file: {e}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::manifest::{sha256_hex, ManifestEntry};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // All fetch tests hold `ENV_LOCK`: `candidate_urls()` reads SATKIT_DATA_URL,
    // and the mirror test sets it, so they must not run concurrently.

    /// A minimal in-process HTTP/1.1 server: `GET /<path>` returns the bytes
    /// registered for that path or 404. Counts requests so tests can assert
    /// what was (not) downloaded. Stops when `stop` is set.
    struct TestServer {
        base: String,
        hits: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn start(files: HashMap<String, Vec<u8>>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let port = listener.local_addr().unwrap().port();
            let hits = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let files = Arc::new(Mutex::new(files));
            let (h2, s2, f2) = (hits.clone(), stop.clone(), files.clone());
            let thread = std::thread::spawn(move || {
                while !s2.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut sock, _)) => {
                            h2.fetch_add(1, Ordering::Relaxed);
                            sock.set_nonblocking(false).unwrap();
                            let mut buf = vec![0u8; 4096];
                            let n = sock.read(&mut buf).unwrap_or(0);
                            let req = String::from_utf8_lossy(&buf[..n]).to_string();
                            let path = req
                                .lines()
                                .next()
                                .and_then(|l| l.split_whitespace().nth(1))
                                .unwrap_or("/")
                                .trim_start_matches('/')
                                .to_string();
                            let body = f2.lock().unwrap().get(&path).cloned();
                            let resp = match body {
                                Some(b) => {
                                    let mut r = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                        b.len()
                                    )
                                    .into_bytes();
                                    r.extend_from_slice(&b);
                                    r
                                }
                                None => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                            };
                            let _ = sock.write_all(&resp);
                            let _ = sock.flush();
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                base: format!("http://127.0.0.1:{port}"),
                hits,
                stop,
                thread: Some(thread),
            }
        }
        fn url(&self, path: &str) -> String {
            format!("{}/{path}", self.base)
        }
        fn hits(&self) -> usize {
            self.hits.load(Ordering::Relaxed)
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn entry(name: &str, bytes: &[u8], urls: Vec<String>) -> ManifestEntry {
        ManifestEntry {
            name: name.into(),
            size: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            urls,
            source: "test".into(),
            license: String::new(),
            tier: "core".into(),
            default: true,
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("satkit_fetch_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn fetch_success_is_verified_and_cached() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let data = b"the quick brown fox".to_vec();
        let srv = TestServer::start(HashMap::from([("good.bin".to_string(), data.clone())]));
        let dir = tmpdir("ok");
        let e = entry("good.bin", &data, vec![srv.url("good.bin")]);

        let out = manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert_eq!(
            out,
            FetchOutcome::Downloaded {
                url: srv.url("good.bin")
            }
        );
        assert_eq!(std::fs::read(dir.join("good.bin")).unwrap(), data);
        assert!(!dir.join("good.bin.part").exists());
        assert_eq!(srv.hits(), 1);

        // Second call: present + hash matches -> no request at all.
        let out = manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert_eq!(out, FetchOutcome::AlreadyPresent);
        assert_eq!(srv.hits(), 1, "verified file must not be re-downloaded");

        // force = true re-downloads.
        let out = manifest::fetch_static_file(&e, &dir, true).unwrap();
        assert!(matches!(out, FetchOutcome::Downloaded { .. }));
        assert_eq!(srv.hits(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fetch_falls_through_404_to_next_url() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let data = b"payload".to_vec();
        let first = TestServer::start(HashMap::new()); // serves nothing -> 404
        let second = TestServer::start(HashMap::from([("f.bin".to_string(), data.clone())]));
        let dir = tmpdir("fallthrough");
        let e = entry(
            "f.bin",
            &data,
            vec![first.url("f.bin"), second.url("f.bin")],
        );
        let out = manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert_eq!(
            out,
            FetchOutcome::Downloaded {
                url: second.url("f.bin")
            }
        );
        assert_eq!(first.hits(), 1);
        assert_eq!(second.hits(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fetch_rejects_hash_mismatch_and_tries_next_url() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let good = b"correct bytes".to_vec();
        let bad = b"corrupt bytes".to_vec(); // same length: exercises the sha check, not the size check
        let first = TestServer::start(HashMap::from([("f.bin".to_string(), bad)]));
        let second = TestServer::start(HashMap::from([("f.bin".to_string(), good.clone())]));
        let dir = tmpdir("mismatch");
        let e = entry(
            "f.bin",
            &good,
            vec![first.url("f.bin"), second.url("f.bin")],
        );
        let out = manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert_eq!(
            out,
            FetchOutcome::Downloaded {
                url: second.url("f.bin")
            }
        );
        assert_eq!(std::fs::read(dir.join("f.bin")).unwrap(), good);
        assert!(
            !dir.join("f.bin.part").exists(),
            "corrupt partial must be removed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fetch_reports_every_failed_source() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let a = TestServer::start(HashMap::new());
        let b = TestServer::start(HashMap::from([("f.bin".to_string(), b"wrong".to_vec())]));
        let dir = tmpdir("allfail");
        let e = entry("f.bin", b"right", vec![a.url("f.bin"), b.url("f.bin")]);
        let err = manifest::fetch_static_file(&e, &dir, false).unwrap_err();
        match &err {
            download::Error::AllSourcesFailed { name, attempts } => {
                assert_eq!(name, "f.bin");
                assert_eq!(attempts.len(), 2);
                assert!(attempts[0].starts_with(&a.url("f.bin")), "{}", attempts[0]);
                assert!(attempts[1].starts_with(&b.url("f.bin")), "{}", attempts[1]);
                assert!(attempts[1].contains("mismatch"), "{}", attempts[1]);
            }
            other => panic!("unexpected error {other}"),
        }
        assert!(!dir.join("f.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mirror_override_is_tried_before_manifest_urls() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let data = b"mirror payload".to_vec();
        let mirror = TestServer::start(HashMap::from([("f.bin".to_string(), data.clone())]));
        let official = TestServer::start(HashMap::from([("f.bin".to_string(), data.clone())]));
        let dir = tmpdir("mirror");
        let e = entry("f.bin", &data, vec![official.url("f.bin")]);
        std::env::set_var(manifest::MIRROR_ENV, &mirror.base);
        let out = manifest::fetch_static_file(&e, &dir, false);
        std::env::remove_var(manifest::MIRROR_ENV);
        assert_eq!(
            out.unwrap(),
            FetchOutcome::Downloaded {
                url: mirror.url("f.bin")
            }
        );
        assert_eq!(mirror.hits(), 1);
        assert_eq!(
            official.hits(),
            0,
            "official URL must not be contacted when the mirror works"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_corrupt_file_is_replaced() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let data = b"fresh".to_vec();
        let srv = TestServer::start(HashMap::from([("f.bin".to_string(), data.clone())]));
        let dir = tmpdir("corrupt");
        std::fs::write(dir.join("f.bin"), b"stale").unwrap(); // same size, wrong hash
        let e = entry("f.bin", &data, vec![srv.url("f.bin")]);
        let out = manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert!(matches!(out, FetchOutcome::Downloaded { .. }));
        assert_eq!(std::fs::read(dir.join("f.bin")).unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real network, full run: `update_datafiles` into a temp dir; prints the
    /// URL each file came from. `cargo test --lib real_network_update -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires network access; downloads ~110 MB"]
    fn real_network_update_datafiles_into_tmp() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmpdir("full");
        let t0 = std::time::Instant::now();
        update_datafiles(Some(dir.clone()), false).unwrap();
        println!("update_datafiles took {:.1} s", t0.elapsed().as_secs_f64());
        for e in manifest::embedded().default_files() {
            assert!(
                e.verify(&dir.join(&e.name)).unwrap(),
                "{} not verified",
                e.name
            );
        }
        assert!(dir.join("EOP-All.csv").is_file() && dir.join("SW-All.csv").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real network: exercises the GitHub-asset → origin/GCS fallthrough for
    /// the smallest manifest file. `cargo test -- --ignored real_network`.
    #[test]
    #[ignore = "requires network access"]
    fn real_network_fetch_smallest_file() {
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = manifest::embedded();
        let e = m.entry("tab5.2d.txt").unwrap();
        let dir = tmpdir("net");
        let out = manifest::fetch_static_file(e, &dir, false).unwrap();
        println!("{out:?}");
        assert!(e.verify(&dir.join("tab5.2d.txt")).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
