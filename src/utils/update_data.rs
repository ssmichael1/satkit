//! Download / refresh the data files satkit needs.
//!
//! Static files (ephemeris, IERS tables, gravity coefficients, leap-second
//! list) come from the embedded [data manifest](crate::utils::manifest)
//! and are SHA-256 verified; the regularly updated files (EOP, space weather)
//! are listed in the manifest's `refresh` and `eop` sections and fetched
//! unverified from their sources, rate-limited to their publication cadence
//! and with a conditional GET outside it. See `data/README.md` for the
//! design.

use super::download::{self, refresh_file_async, RefreshOutcome};
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

    /// A worker thread launched by [`refresh_file_async`] or the static
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

/// Refresh the regularly updated files: the plain URLs of the manifest's
/// `refresh` section (space weather) in parallel with the Earth orientation
/// refresh, which tries the manifest's `eop` sources in order (IERS
/// `finals2000A.all` mirrors, then CelesTrak's `EOP-All.csv`).
///
/// Each one goes through [`refresh_file`](download::refresh_file), which
/// skips the request entirely while the local copy is inside its publication
/// cadence and otherwise sends a conditional GET. `force` re-fetches
/// unconditionally. Returns one `(name, url, outcome)` per file.
fn download_refresh_files(
    dir: &std::path::Path,
    force: bool,
) -> Result<Vec<(String, String, RefreshOutcome)>> {
    let m = manifest::embedded();
    let handles: Vec<(String, JoinHandle<download::Result<RefreshOutcome>>)> = m
        .refresh
        .iter()
        .map(|url| -> Result<_> {
            if !url.starts_with("https://") {
                return Err(Error::InsecureManifestUrl { url: url.clone() });
            }
            let name = url.rsplit('/').next().unwrap_or(url).to_string();
            Ok((name, refresh_file_async(url.clone(), dir, force)))
        })
        .collect::<Result<Vec<_>>>()?;
    let eop_dir = dir.to_path_buf();
    let eop =
        std::thread::spawn(move || crate::earth_orientation_params::refresh_into(&eop_dir, force));
    let msafe_dir = dir.to_path_buf();
    let msafe =
        std::thread::spawn(move || crate::spaceweather::msafe::refresh_into(&msafe_dir, force));
    let mut out = Vec::with_capacity(handles.len() + 1);
    for (name, jh) in handles {
        let url = m.refresh.iter().find(|u| u.ends_with(&name)).cloned();
        out.push((
            name,
            url.unwrap_or_default(),
            jh.join().map_err(|_| Error::ThreadPanic)??,
        ));
    }
    let eop = eop.join().map_err(|_| Error::ThreadPanic)??;
    out.push((eop.source.file_name().to_string(), eop.url, eop.fetch));
    // MSAFE is best-effort: NASA's hosting is the least dependable of the
    // three, and an observed-only table is still usable.
    match msafe.join().map_err(|_| Error::ThreadPanic)? {
        Ok(fetch) => out.push((
            crate::spaceweather::MSAFE_FILE.to_string(),
            String::new(),
            fetch,
        )),
        Err(e) => eprintln!("Warning: MSAFE forecast not refreshed: {e}"),
    }
    Ok(out)
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
/// The only static file downloaded is the JPL ephemeris, described by the
/// embedded [data manifest](crate::utils::manifest): it is fetched from the
/// first working source (`SATKIT_DATA_URL` mirror if set, then the GitHub
/// release asset, the origin server, and the legacy bucket) and is only
/// accepted when its size and SHA-256 match the manifest. The IERS nutation
/// tables and the EGM96 / EGM2008 / JGM2 / JGM3 gravity coefficients are
/// compiled into the library and not downloaded (their manifest entries are
/// `default: false`, still fetchable by name); ITU_GRACE16 is fetched only
/// when that model is first used. A copy placed in a search directory takes
/// precedence.
///
/// The space weather file is refreshed from celestrak, the Earth orientation
/// file from the IERS `finals2000A.all` mirrors (CelesTrak's `EOP-All.csv`
/// when both are unreachable — see
/// [`earth_orientation_params::refresh_into`](crate::earth_orientation_params::refresh_into)),
///; these change daily and are
/// not pinned. The refresh respects each file's publication cadence: a copy
/// newer than that (3 h for space weather, 24 h for EOP) is left alone
/// without contacting the server, and otherwise the request is conditional so
/// an unchanged file costs a `304`. `overwrite_if_exists` forces a full
/// re-fetch of these too.
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

    println!("Regularly updated files (Space Weather, Earth Orientation Parameters):");
    for (name, url, outcome) in download_refresh_files(&downloaddir, overwrite_if_exists)? {
        match outcome {
            RefreshOutcome::Fresh { age_secs } => println!(
                "  {name}: current ({:.1} h old); no request made",
                age_secs as f64 / 3600.0
            ),
            RefreshOutcome::NotModified => println!("  {name}: unchanged on the server (304)"),
            RefreshOutcome::Downloaded => println!("  {name}: downloaded from {url}"),
        }
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
    if let Err(e) = crate::earth_orientation_params::load_from_dir(&downloaddir) {
        eprintln!("Warning: could not load downloaded EOP file: {e}");
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
        conditional_hits: Arc<AtomicUsize>,
        files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn start(files: HashMap<String, Vec<u8>>) -> Self {
            Self::start_with_last_modified(files, None)
        }

        /// As [`start`], but every 200 carries `Last-Modified: <lm>` and a
        /// request whose `If-Modified-Since` equals it is answered `304`.
        fn start_with_last_modified(files: HashMap<String, Vec<u8>>, lm: Option<&str>) -> Self {
            let last_modified = lm.map(str::to_string);
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let port = listener.local_addr().unwrap().port();
            let hits = Arc::new(AtomicUsize::new(0));
            let conditional_hits = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let files = Arc::new(Mutex::new(files));
            let (h2, s2, f2) = (hits.clone(), stop.clone(), files.clone());
            let c2 = conditional_hits.clone();
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
                            // Header names are matched case-insensitively:
                            // HTTP/1.1 does not fix their case and clients
                            // differ.
                            let ims = req.lines().find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                k.trim()
                                    .eq_ignore_ascii_case("if-modified-since")
                                    .then(|| v.trim().to_string())
                            });
                            if ims.is_some() {
                                c2.fetch_add(1, Ordering::Relaxed);
                            }
                            let body = f2.lock().unwrap().get(&path).cloned();
                            let lm_header = last_modified
                                .as_deref()
                                .map(|lm| format!("Last-Modified: {lm}\r\n"))
                                .unwrap_or_default();
                            let not_modified = matches!(
                                (last_modified.as_deref(), ims.as_deref()),
                                (Some(lm), Some(ims)) if lm == ims
                            );
                            let resp = match body {
                                Some(_) if not_modified => format!(
                                    "HTTP/1.1 304 Not Modified\r\n{lm_header}Connection: close\r\n\r\n"
                                )
                                .into_bytes(),
                                Some(b) => {
                                    let mut r = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{lm_header}Connection: close\r\n\r\n",
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
                conditional_hits,
                files,
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
        /// Requests that carried an `If-Modified-Since` header.
        fn conditional_hits(&self) -> usize {
            self.conditional_hits.load(Ordering::Relaxed)
        }
        /// Replace the bytes served for `path` (simulates a feed update).
        fn set_body(&self, path: &str, body: Vec<u8>) {
            self.files.lock().unwrap().insert(path.to_string(), body);
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

    /// Under offline mode a lazy fetch is a typed error and **no HTTP
    /// request is made**: the in-process server sees zero hits.
    #[test]
    fn offline_mode_blocks_fetch_without_network_io() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bytes = b"offline test bytes".to_vec();
        let server = TestServer::start(HashMap::from([("f.txt".to_string(), bytes.clone())]));
        let e = entry("f.txt", &bytes, vec![server.url("f.txt")]);
        let dir = tmpdir("offline");
        download::set_offline(true);
        let err = manifest::fetch_static_file(&e, &dir, false).unwrap_err();
        download::set_offline(false);
        // Leave the process in its environment-driven state afterwards.
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                download::clear_offline_override();
            }
        }
        let _restore = Restore;
        assert!(
            matches!(&err, download::Error::Offline { name, urls, .. } if name == "f.txt" && urls.len() == 1),
            "{err}"
        );
        assert!(err.to_string().contains(&server.url("f.txt")));
        assert_eq!(server.hits(), 0, "offline mode must not open a connection");
        assert!(!dir.join("f.txt").exists());
        // With offline mode lifted the same fetch succeeds.
        manifest::fetch_static_file(&e, &dir, false).unwrap();
        assert_eq!(server.hits(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `set_offline` overrides `SATKIT_OFFLINE` in both directions; with no
    /// setter call the environment decides.
    #[test]
    fn offline_setter_overrides_environment() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prior_env = std::env::var_os(download::OFFLINE_ENV);
        // Env says offline, setter says online -> online.
        std::env::set_var(download::OFFLINE_ENV, "1");
        download::set_offline(false);
        assert!(!download::is_offline());
        // Env says online, setter says offline -> offline.
        std::env::remove_var(download::OFFLINE_ENV);
        download::set_offline(true);
        assert!(download::is_offline());
        download::set_offline(false);
        assert!(!download::is_offline());
        // Back to environment-driven: with the var unset that is "online".
        download::clear_offline_override();
        assert!(!download::is_offline());
        match prior_env {
            Some(v) => std::env::set_var(download::OFFLINE_ENV, v),
            None => std::env::remove_var(download::OFFLINE_ENV),
        }
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
            download::Error::AllSourcesFailed {
                name,
                attempts,
                hint,
            } => {
                assert_eq!(name, "f.bin");
                assert_eq!(attempts.len(), 2);
                // Plain HTTP failures speak for themselves; no hint appended.
                assert!(hint.is_none(), "{hint:?}");
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
    /// Many concurrent fetches of the same file: each writes its own
    /// `.part.<pid>.<seq>`, exactly one verified final file results, no
    /// temporary file is left behind and every caller succeeds.
    #[test]
    fn concurrent_fetches_of_one_file_yield_one_verified_copy() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let server = TestServer::start(HashMap::from([("big.bin".to_string(), bytes.clone())]));
        let e = std::sync::Arc::new(entry("big.bin", &bytes, vec![server.url("big.bin")]));
        let dir = std::sync::Arc::new(tmpdir("concurrent"));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (e, dir) = (e.clone(), dir.clone());
                std::thread::spawn(move || {
                    crate::utils::manifest::fetch_static_file(&e, &dir, false)
                })
            })
            .collect();
        for h in handles {
            let outcome = h.join().unwrap().expect("every concurrent fetch succeeds");
            assert!(matches!(
                outcome,
                FetchOutcome::Downloaded { .. } | FetchOutcome::AlreadyPresent
            ));
        }
        assert!(
            e.verify(&dir.join("big.bin")).unwrap(),
            "final file verified"
        );
        let leftovers: Vec<String> = std::fs::read_dir(&*dir)
            .unwrap()
            .flatten()
            .map(|d| d.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".part"))
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
        assert!(server.hits() >= 1 && server.hits() <= 8);
        let _ = std::fs::remove_dir_all(&*dir);
    }

    /// An on-disk manifest-pinned file is hashed once and then trusted via
    /// the sidecar marker until it changes; a wrong copy is `CorruptFile`.
    #[test]
    fn on_disk_file_is_verified_once_via_sidecar_marker() {
        use crate::utils::download::Error;
        use crate::utils::manifest::Verified;
        let bytes = b"correct contents of a pinned file".to_vec();
        let e = entry(
            "pinned.bin",
            &bytes,
            vec!["https://example.invalid/p".into()],
        );
        let dir = tmpdir("sidecar");
        let path = dir.join("pinned.bin");
        let marker = ManifestEntry::verified_marker_path(&path);

        // Wrong bytes, right size -> corrupt, no marker written.
        std::fs::write(&path, b"wrong!! contents of a pinned file").unwrap();
        let err = e.ensure_verified(&path).unwrap_err();
        assert!(
            matches!(err, Error::CorruptFile { what: "sha256", .. }),
            "{err}"
        );
        assert!(!marker.exists());
        // Wrong size -> corrupt without hashing.
        std::fs::write(&path, b"short").unwrap();
        assert!(matches!(
            e.ensure_verified(&path).unwrap_err(),
            Error::CorruptFile { what: "size", .. }
        ));

        // Right bytes -> hashed once, marker created, then cached.
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(e.ensure_verified(&path).unwrap(), Verified::Hashed);
        assert!(marker.exists());
        assert_eq!(e.ensure_verified(&path).unwrap(), Verified::Cached);

        // Same size, different bytes, newer mtime -> the marker no longer
        // matches, the file is re-hashed and the corruption is caught.
        std::fs::write(&path, b"wrong!! contents of a pinned file").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert!(matches!(
            e.ensure_verified(&path).unwrap_err(),
            Error::CorruptFile { what: "sha256", .. }
        ));

        // Restored bytes with yet another mtime -> hashed again, then cached.
        // The mtime is set explicitly: Windows file times advance in ~1-15 ms
        // ticks, so a plain rewrite can land on the marker's original mtime
        // and be reported as `Cached`.
        std::fs::write(&path, &bytes).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later + std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(e.ensure_verified(&path).unwrap(), Verified::Hashed);
        assert_eq!(e.ensure_verified(&path).unwrap(), Verified::Cached);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ureq's default agent reads `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`
    /// (and honours `NO_PROXY`); constructing it with a proxy set must not
    /// fail or touch the network.
    #[test]
    fn proxy_env_is_accepted_by_the_agent() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::set_var("HTTPS_PROXY", "http://proxy.invalid:3128");
        let agent = crate::utils::download::http_agent();
        let has_proxy = agent.config().proxy().is_some();
        std::env::remove_var("HTTPS_PROXY");
        assert!(
            has_proxy,
            "ureq should pick the proxy up from the environment"
        );
    }

    /// URL each file came from. `cargo test --lib real_network_update -- --ignored --nocapture`.
    /// A few real `finals2000A.all` lines (Bulletin A columns) and a
    /// CelesTrak table, for the EOP source-order tests.
    const FINALS: &str = "\
26 917 61300.00 I  0.190054 0.000090  0.329163 0.000090  I-0.0086337 0.0000267                 P     0.084    0.128     0.235    0.160
26 918 61301.00 P  0.189180 0.000600  0.329137 0.000401  P-0.0091919 0.0001080                 P     0.094    0.128     0.236    0.160
";
    const EOP_CSV: &str = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n\
        2026-09-18,61301,0.187672,0.328316,-0.0073956,0.0000148,-0.124590,-0.011105,0.000295,-0.000027,37,P\n";

    fn eop_sources(server: &TestServer) -> Vec<crate::utils::manifest::RefreshSource> {
        vec![
            crate::utils::manifest::RefreshSource {
                name: "finals2000A.all".into(),
                urls: vec![
                    server.url("usno/finals2000A.all"),
                    server.url("iers/finals2000A.all"),
                ],
            },
            crate::utils::manifest::RefreshSource {
                name: "EOP-All.csv".into(),
                urls: vec![server.url("celestrak/EOP-All.csv")],
            },
        ]
    }

    /// The EOP refresh walks the manifest's sources in order: the second
    /// IERS mirror when the first is down, CelesTrak's file when both are,
    /// and a typed error naming every URL when nothing answers. A mirror
    /// that answers with an HTML page instead of the file is skipped too.
    /// Between forced fetches, a copy inside its 24 h cadence is reported
    /// current without any request — even when every mirror is down.
    #[test]
    fn eop_refresh_falls_through_mirrors_then_celestrak() {
        use crate::earth_orientation_params::{self as eop, EopSource};
        let _guard = manifest::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::utils::download::clear_offline_override();
        let dir = tmpdir("eop_order");

        // Both IERS mirrors up: the first one is used and nothing else is asked.
        let server = TestServer::start(HashMap::from([
            (
                "usno/finals2000A.all".to_string(),
                FINALS.as_bytes().to_vec(),
            ),
            (
                "iers/finals2000A.all".to_string(),
                FINALS.as_bytes().to_vec(),
            ),
            (
                "celestrak/EOP-All.csv".to_string(),
                EOP_CSV.as_bytes().to_vec(),
            ),
        ]));
        let out = eop::refresh_into_with_sources(&dir, &eop_sources(&server), false).unwrap();
        assert_eq!(out.source, EopSource::IersFinals2000A);
        assert_eq!(out.url, server.url("usno/finals2000A.all"));
        assert_eq!(out.fetch, RefreshOutcome::Downloaded);
        assert_eq!(server.hits(), 1);
        assert!(dir.join("finals2000A.all").is_file());
        assert!(!dir.join("EOP-All.csv").exists());

        // Inside the cadence: current, no request made.
        let out = eop::refresh_into_with_sources(&dir, &eop_sources(&server), false).unwrap();
        assert_eq!(out.source, EopSource::IersFinals2000A);
        assert!(matches!(out.fetch, RefreshOutcome::Fresh { .. }), "{out:?}");
        assert_eq!(server.hits(), 1);
        drop(server);

        // First mirror answers with a notice page: rejected, second mirror used.
        let page = b"<!DOCTYPE html><html><body>maintenance</body></html>".to_vec();
        let server = TestServer::start(HashMap::from([
            ("usno/finals2000A.all".to_string(), page.clone()),
            (
                "iers/finals2000A.all".to_string(),
                FINALS.as_bytes().to_vec(),
            ),
        ]));
        let out = eop::refresh_into_with_sources(&dir, &eop_sources(&server), true).unwrap();
        assert_eq!(out.url, server.url("iers/finals2000A.all"));
        assert_eq!(out.fetch, RefreshOutcome::Downloaded);
        assert_eq!(server.hits(), 2);
        drop(server);

        // Both mirrors 404: CelesTrak's file is fetched instead.
        let server = TestServer::start(HashMap::from([(
            "celestrak/EOP-All.csv".to_string(),
            EOP_CSV.as_bytes().to_vec(),
        )]));
        let out = eop::refresh_into_with_sources(&dir, &eop_sources(&server), true).unwrap();
        assert_eq!(out.source, EopSource::CelesTrak);
        assert_eq!(server.hits(), 3);
        assert!(dir.join("EOP-All.csv").is_file());
        drop(server);

        // Nothing answers: every URL is named, and the files already on disk
        // are untouched.
        let server = TestServer::start(HashMap::new());
        let err = eop::refresh_into_with_sources(&dir, &eop_sources(&server), true).unwrap_err();
        match &err {
            download::Error::AllSourcesFailed { attempts, .. } => assert_eq!(attempts.len(), 3),
            other => panic!("expected AllSourcesFailed, got {other:?}"),
        }
        assert!(err.to_string().contains("celestrak/EOP-All.csv"));
        assert_eq!(
            std::fs::read_to_string(dir.join("finals2000A.all")).unwrap(),
            FINALS
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("EOP-All.csv")).unwrap(),
            EOP_CSV
        );

        // ... but an unforced refresh still finds the primary copy inside its
        // cadence and asks nobody.
        let hits_before = server.hits();
        let out = eop::refresh_into_with_sources(&dir, &eop_sources(&server), false).unwrap();
        assert_eq!(out.source, EopSource::IersFinals2000A);
        assert!(matches!(out.fetch, RefreshOutcome::Fresh { .. }), "{out:?}");
        assert_eq!(server.hits(), hits_before);
        drop(server);

        // (Which file the loader then picks is covered in
        // earth_orientation_params::tests::load_from_paths_picks_the_fresher_file.)

        // Offline: no request is made at all, and no cadence shortcut either.
        crate::utils::download::set_offline(true);
        let server = TestServer::start(HashMap::from([(
            "usno/finals2000A.all".to_string(),
            FINALS.as_bytes().to_vec(),
        )]));
        let err = eop::refresh_into_with_sources(&dir, &eop_sources(&server), false).unwrap_err();
        assert!(matches!(err, download::Error::Offline { .. }), "{err}");
        assert_eq!(server.hits(), 0);
        crate::utils::download::clear_offline_override();

        let _ = std::fs::remove_dir_all(&dir);
    }

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
        assert!(dir.join("finals2000A.all").is_file() || dir.join("EOP-All.csv").is_file());
        assert!(dir.join("SW-All.csv").is_file());
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

    const LAST_MODIFIED: &str = "Wed, 17 Sep 2026 12:00:00 GMT";

    /// Write the freshness sidecar by hand, with the size and mtime of the
    /// file as it is now, so only the fields under test differ.
    fn write_marker(path: &std::path::Path, checked_at: u64, last_modified: &str) {
        let md = std::fs::metadata(path).unwrap();
        let mtime = md
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        std::fs::write(
            download::refresh_marker_path(path),
            format!("{checked_at} {} {mtime}\n{last_modified}\n", md.len()),
        )
        .unwrap();
    }

    /// Make the file look `age_secs` old, keeping the recorded
    /// `Last-Modified`. This is how a test reaches the "cadence has elapsed"
    /// branch without sleeping.
    fn backdate_marker(path: &std::path::Path, age_secs: u64) {
        let (checked_at, lm) = download::read_refresh_marker(path).expect("marker written");
        write_marker(path, checked_at - age_secs, &lm.unwrap_or_default());
    }

    /// Inside the publication cadence, a refresh makes **no request at all**:
    /// the whole point of the gate is that re-running a script, or a CI job
    /// that restored a cached data directory, does not touch CelesTrak.
    #[test]
    fn refresh_inside_cadence_makes_no_request() {
        // Shares `ENV_LOCK` with the offline tests: they flip the global
        // offline flag, which would turn these fetches into `Offline` errors.
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let body = b"feed,body\n1,2\n".to_vec();
        let server = TestServer::start(HashMap::from([("Feed.csv".to_string(), body.clone())]));
        let dir = tmpdir("refresh_fresh");
        let url = server.url("Feed.csv");

        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(server.hits(), 1);
        assert_eq!(std::fs::read(dir.join("Feed.csv")).unwrap(), body);

        // Second call, immediately: served from disk, server untouched.
        match download::refresh_file(&url, &dir, false).unwrap() {
            RefreshOutcome::Fresh { age_secs } => assert!(age_secs < 60),
            other => panic!("expected Fresh, got {other:?}"),
        }
        assert_eq!(server.hits(), 1);

        // `force` is the escape hatch and always transfers.
        assert_eq!(
            download::refresh_file(&url, &dir, true).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(server.hits(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Once the cadence has elapsed the request goes out, but conditionally:
    /// an unchanged file comes back as a bodyless `304` and the copy on disk
    /// is kept.
    #[test]
    fn refresh_past_cadence_is_conditional() {
        // Shares `ENV_LOCK` with the offline tests: they flip the global
        // offline flag, which would turn these fetches into `Offline` errors.
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let body = b"feed,body\n1,2\n".to_vec();
        let server = TestServer::start_with_last_modified(
            HashMap::from([("Feed.csv".to_string(), body.clone())]),
            Some(LAST_MODIFIED),
        );
        let dir = tmpdir("refresh_304");
        let url = server.url("Feed.csv");
        let path = dir.join("Feed.csv");

        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Downloaded
        );
        // The first fetch has nothing to compare against, so it is not
        // conditional.
        assert_eq!(server.conditional_hits(), 0);

        backdate_marker(&path, 4 * 3600);
        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::NotModified
        );
        assert_eq!(server.hits(), 2);
        assert_eq!(server.conditional_hits(), 1);
        assert_eq!(std::fs::read(&path).unwrap(), body);

        // The 304 re-stamps the marker, so the gate closes again.
        assert!(matches!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Fresh { .. }
        ));
        assert_eq!(server.hits(), 2);

        // `force` must not send the conditional header: it is the way to get
        // the bytes back when a local copy is suspect.
        assert_eq!(
            download::refresh_file(&url, &dir, true).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(server.conditional_hits(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// New bytes on the server replace the local copy once the cadence has
    /// elapsed — the gate must not pin a stale file indefinitely.
    #[test]
    fn refresh_past_cadence_installs_changed_bytes() {
        // Shares `ENV_LOCK` with the offline tests: they flip the global
        // offline flag, which would turn these fetches into `Offline` errors.
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let old = b"feed,body\n1,2\n".to_vec();
        let new = b"feed,body\n1,2\n3,4\n".to_vec();
        let server = TestServer::start_with_last_modified(
            HashMap::from([("Feed.csv".to_string(), old.clone())]),
            // A `Last-Modified` the client will never echo back, so the
            // server always answers 200: the "file changed" case.
            Some("Thu, 18 Sep 2026 12:00:00 GMT"),
        );
        let dir = tmpdir("refresh_changed");
        let url = server.url("Feed.csv");
        let path = dir.join("Feed.csv");

        download::refresh_file(&url, &dir, false).unwrap();
        // Cadence elapsed, and the recorded `Last-Modified` no longer matches
        // the server's, so the conditional GET returns 200 with a body.
        write_marker(&path, 0, "Mon, 01 Jan 1990 00:00:00 GMT");
        server.set_body("Feed.csv", new.clone());

        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(std::fs::read(&path).unwrap(), new);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A marker left behind by a deleted file must not suppress the fetch.
    #[test]
    fn refresh_refetches_when_file_is_gone() {
        // Shares `ENV_LOCK` with the offline tests: they flip the global
        // offline flag, which would turn these fetches into `Offline` errors.
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let body = b"feed,body\n1,2\n".to_vec();
        let server = TestServer::start(HashMap::from([("Feed.csv".to_string(), body.clone())]));
        let dir = tmpdir("refresh_gone");
        let url = server.url("Feed.csv");

        download::refresh_file(&url, &dir, false).unwrap();
        std::fs::remove_file(dir.join("Feed.csv")).unwrap();
        assert!(download::refresh_marker_path(&dir.join("Feed.csv")).is_file());

        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(server.hits(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The gate lengths are CelesTrak's publication cadences.
    #[test]
    fn refresh_cadence_matches_celestrak_publication() {
        assert_eq!(download::refresh_min_age_secs("SW-All.csv"), 3 * 3600);
        assert_eq!(download::refresh_min_age_secs("EOP-All.csv"), 24 * 3600);
    }

    /// A file replaced underneath the sidecar must be re-fetched, not
    /// reported as current. Without the size/mtime fields the stale marker
    /// would both hold the age gate shut and echo the previous file's
    /// `Last-Modified`, so the swapped-in bytes would be trusted forever.
    #[test]
    fn refresh_marker_is_ignored_when_the_file_changed_underneath() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let body = b"feed,body\n1,2\n".to_vec();
        let server = TestServer::start_with_last_modified(
            HashMap::from([("Feed.csv".to_string(), body.clone())]),
            Some(LAST_MODIFIED),
        );
        let dir = tmpdir("refresh_swapped");
        let url = server.url("Feed.csv");
        let path = dir.join("Feed.csv");

        download::refresh_file(&url, &dir, false).unwrap();
        assert_eq!(server.hits(), 1);
        assert!(download::read_refresh_marker(&path).is_some());

        // Someone drops a different copy in, well inside the cadence.
        std::fs::write(&path, b"feed,body\n9,9\n9,9\n").unwrap();
        assert!(
            download::read_refresh_marker(&path).is_none(),
            "the marker must stop describing a file it no longer matches"
        );

        // Not `Fresh`, and not a conditional request that would come back 304.
        assert_eq!(
            download::refresh_file(&url, &dir, false).unwrap(),
            RefreshOutcome::Downloaded
        );
        assert_eq!(server.conditional_hits(), 0);
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An empty 200 body is rejected before it can replace a good file. No
    /// file satkit downloads is legitimately empty, and for a feed without a
    /// parser this is the only thing between a broken server and a truncated
    /// table.
    #[test]
    fn refresh_rejects_an_empty_body() {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let body = b"feed,body\n1,2\n".to_vec();
        let server = TestServer::start(HashMap::from([("Feed.csv".to_string(), body.clone())]));
        let dir = tmpdir("refresh_empty");
        let url = server.url("Feed.csv");
        let path = dir.join("Feed.csv");

        download::refresh_file(&url, &dir, false).unwrap();
        server.set_body("Feed.csv", Vec::new());

        let err = download::refresh_file(&url, &dir, true).unwrap_err();
        assert!(
            matches!(&err, download::Error::ContentRejected { reason, .. } if reason.contains("empty")),
            "expected ContentRejected, got {err:?}"
        );
        // The good file is still there, and no `.part` was left behind.
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".part"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
