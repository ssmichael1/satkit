//! Embedded manifest of satkit's static data files, with verified download.
//!
//! # Why a manifest
//!
//! satkit needs a handful of third-party data files at runtime (JPL
//! ephemeris, IERS nutation tables, gravity coefficients). Historically these were mirrored on a Google Cloud
//! Storage bucket and fetched by name with no integrity check, so a
//! satkit release did not determine which bytes a user actually got, and a
//! file could change under a fixed URL without anyone noticing.
//!
//! `data/manifest.json` (compiled in via [`MANIFEST_JSON`]) pins every
//! static file by size and SHA-256 and lists, in order of preference, the
//! URLs it may be fetched from. A given satkit build therefore always
//! resolves to the same data bytes, downloads are verified before they are
//! trusted, and the same manifest drives the CI cache key, the Python
//! bootstrap script and (by hand, for now) the conda recipe.
//!
//! # URL order
//!
//! 1. `SATKIT_DATA_URL` (environment) — an optional base URL tried first for
//!    every file: a corporate mirror, an air-gapped file share exposed over
//!    HTTP, or a local test server. `http://` is permitted here because such
//!    mirrors are frequently plain HTTP on a private network; every download
//!    is still hash-verified.
//! 2. The GitHub release asset (`release_base/<name>`): CDN-backed, stable
//!    per-tag URL, no bandwidth charge to the maintainer.
//! 3. The origin server, where one exists *and serves byte-identical data*
//!    (JPL for the DE files, IERS for the `tab5.2*` tables) — no mirror is
//!    needed for these at all.
//! 4. The legacy GCS bucket, kept as a transitional fallback until the
//!    release assets have been published for a release or two.
//!
//! Every URL in the manifest itself must be `https://`; the mirror override
//! is the only place plain HTTP is accepted.
//!
//! # Verification
//!
//! [`fetch_static_file`] streams each candidate into `<name>.part`, hashing
//! as it goes, and only renames it into place when both the size and the
//! SHA-256 match the manifest. On mismatch the partial file is deleted and
//! the next URL is tried; if every URL fails the error lists each attempt.
//! An existing file whose hash matches is never re-downloaded.
//!
//! The regularly refreshed files (`EOP-All.csv`, `SW-All.csv`) are *not*
//! pinned — they change daily — and are listed under `refresh` as plain
//! URLs (celestrak).

use serde::Deserialize;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

use super::download::{self, Error, Result};

/// The manifest as compiled into the library (`data/manifest.json`).
pub const MANIFEST_JSON: &str = include_str!("../../data/manifest.json");

/// Environment variable naming a base URL to try before every manifest URL.
pub const MIRROR_ENV: &str = "SATKIT_DATA_URL";

/// Tests that set [`MIRROR_ENV`] take this lock so they cannot race each
/// other (the test harness runs tests in parallel threads).
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Manifest of the static data files (see the [module docs](self)).
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// Manifest schema version; this module understands `2`.
    pub manifest_version: u32,
    /// Data-bundle version; also the GitHub release tag the assets live under.
    pub data_version: String,
    /// Base URL of the GitHub release that hosts the assets.
    pub release_base: String,
    /// The pinned static files.
    pub files: Vec<ManifestEntry>,
    /// Regularly refreshed files (EOP, space weather): plain URLs, never pinned.
    #[serde(default)]
    pub refresh: Vec<String>,
}

/// One pinned static data file.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestEntry {
    /// File name (a single plain path component) under the data directory.
    pub name: String,
    /// Exact size in bytes.
    pub size: u64,
    /// Lower-case hex SHA-256 of the file contents.
    pub sha256: String,
    /// Download URLs in order of preference (all `https://`).
    pub urls: Vec<String>,
    /// Originating organisation (JPL, IERS, ICGEM-GFZ, NRL, ...).
    #[serde(default)]
    pub source: String,
    /// Licence / attribution note.
    #[serde(default)]
    pub license: String,
    /// `core` (small, required for frames/gravity), `ephemeris`, or
    /// `reference` (downloaded for reference only, not read at runtime).
    #[serde(default)]
    pub tier: String,
    /// Whether [`crate::utils::update_datafiles`] fetches this file. Optional
    /// alternatives (e.g. the smaller DE421 ephemeris) are listed with
    /// `default: false` so they are still pinned and fetchable by name.
    #[serde(default = "default_true")]
    pub default: bool,
}

fn default_true() -> bool {
    true
}

/// How [`ManifestEntry::ensure_verified`] established that a file on disk
/// matches the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verified {
    /// The sidecar marker matched the file's size and modification time, so
    /// the (expensive) hash was skipped.
    Cached,
    /// The file was hashed and matched; the marker was (re)written.
    Hashed,
}

/// Outcome of [`fetch_static_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// The file was already present and its hash matched; nothing downloaded.
    AlreadyPresent,
    /// The file was downloaded (and verified) from this URL.
    Downloaded { url: String },
}

impl Manifest {
    /// Parse a manifest from JSON and validate it.
    pub fn parse(json: &str) -> Result<Self> {
        let m: Self = serde_json::from_str(json)?;
        m.validate()?;
        Ok(m)
    }

    /// Structural validation: known schema version, plain file names, unique
    /// names, positive sizes, 64-hex SHA-256, at least one `https://` URL per
    /// file, `https://` refresh URLs.
    pub fn validate(&self) -> Result<()> {
        let invalid = |msg: String| Error::ManifestInvalid { reason: msg };
        if self.manifest_version != 2 {
            return Err(invalid(format!(
                "unsupported manifest_version {}",
                self.manifest_version
            )));
        }
        if !self.release_base.starts_with("https://") {
            return Err(invalid("release_base must be https://".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for e in &self.files {
            validate_file_name(&e.name)?;
            if !seen.insert(e.name.as_str()) {
                return Err(invalid(format!("duplicate file name {:?}", e.name)));
            }
            if e.size == 0 {
                return Err(invalid(format!("{}: size must be positive", e.name)));
            }
            if e.sha256.len() != 64 || !e.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(invalid(format!("{}: sha256 must be 64 hex chars", e.name)));
            }
            if e.sha256.chars().any(|c| c.is_ascii_uppercase()) {
                return Err(invalid(format!("{}: sha256 must be lower-case", e.name)));
            }
            if e.urls.is_empty() {
                return Err(invalid(format!("{}: needs at least one URL", e.name)));
            }
            for u in &e.urls {
                if !u.starts_with("https://") {
                    return Err(invalid(format!("{}: URL {u:?} must be https://", e.name)));
                }
            }
        }
        for u in &self.refresh {
            if !u.starts_with("https://") {
                return Err(invalid(format!("refresh URL {u:?} must be https://")));
            }
        }
        Ok(())
    }

    /// Look up a file by name.
    pub fn entry(&self, name: &str) -> Option<&ManifestEntry> {
        self.files.iter().find(|e| e.name == name)
    }

    /// The files [`crate::utils::update_datafiles`] downloads by default.
    pub fn default_files(&self) -> impl Iterator<Item = &ManifestEntry> {
        self.files.iter().filter(|e| e.default)
    }
}

/// A manifest file name is joined onto the data directory, so it must be a
/// single plain path component: an absolute name would *replace* the base
/// directory in `Path::join`, and `..` or embedded separators would escape it.
pub fn validate_file_name(name: &str) -> Result<()> {
    let bad = name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
        || Path::new(name).is_absolute();
    if bad {
        return Err(Error::InvalidFileName {
            path: name.to_string(),
        });
    }
    Ok(())
}

/// The compiled-in manifest, parsed once. The embedded JSON is validated by
/// a unit test, so a failure here means a build with a corrupt
/// `data/manifest.json` — not a runtime condition worth propagating.
pub fn embedded() -> &'static Manifest {
    static M: OnceLock<Manifest> = OnceLock::new();
    M.get_or_init(|| {
        Manifest::parse(MANIFEST_JSON).expect("embedded data/manifest.json is invalid")
    })
}

/// Lower-case hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

/// Lower-case hex SHA-256 of a file, streamed.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Modification time of a file as `(secs, nanos)` since the Unix epoch
/// (`(0, 0)` if the platform does not report one).
fn mtime_parts(md: &std::fs::Metadata) -> (u64, u32) {
    md.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The mirror base URL from [`MIRROR_ENV`], if set and non-empty, without a
/// trailing slash.
pub fn mirror_base() -> Option<String> {
    std::env::var(MIRROR_ENV)
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
}

impl ManifestEntry {
    /// URLs to try, in order: the [`MIRROR_ENV`] mirror (if set), then the
    /// manifest's own list.
    pub fn candidate_urls(&self) -> Vec<String> {
        let mut v = Vec::with_capacity(self.urls.len() + 1);
        if let Some(base) = mirror_base() {
            v.push(format!("{base}/{}", self.name));
        }
        v.extend(self.urls.iter().cloned());
        v
    }

    /// Path of the sidecar marker recording that `path` was verified:
    /// `<path>.sha256-verified`, containing `<sha256> <size> <mtime-secs> <mtime-nanos>`.
    pub fn verified_marker_path(path: &Path) -> std::path::PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".sha256-verified");
        std::path::PathBuf::from(p)
    }

    /// Write the sidecar marker for `path` (best effort; a read-only
    /// directory simply means the next load hashes the file again).
    pub fn write_verified_marker(&self, path: &Path) -> std::io::Result<()> {
        let md = std::fs::metadata(path)?;
        let (secs, nanos) = mtime_parts(&md);
        std::fs::write(
            Self::verified_marker_path(path),
            format!("{} {} {secs} {nanos}\n", self.sha256, md.len()),
        )
    }

    /// Verify that an existing on-disk copy of this file matches the
    /// manifest, hashing it at most once per change.
    ///
    /// The first load of a manifest-pinned file hashes it (≈0.3 s for the
    /// 102 MB DE440) and records a sidecar marker with the hash, size and
    /// modification time; later loads compare size and mtime against the
    /// marker and skip the hash ([`Verified::Cached`]). A file whose size or
    /// hash does not match is reported as [`Error::CorruptFile`] — it is never
    /// silently trusted. Files that are not in the manifest are not checked
    /// (a user-supplied ephemeris is trusted as before).
    pub fn ensure_verified(&self, path: &Path) -> Result<Verified> {
        let md = std::fs::metadata(path)?;
        let corrupt = |what: &'static str, expected: String, actual: String| Error::CorruptFile {
            name: self.name.clone(),
            path: path.display().to_string(),
            what,
            values: Box::new((expected, actual)),
        };
        if !md.is_file() || md.len() != self.size {
            return Err(corrupt(
                "size",
                format!("{} bytes", self.size),
                format!("{} bytes", md.len()),
            ));
        }
        let (secs, nanos) = mtime_parts(&md);
        if let Ok(marker) = std::fs::read_to_string(Self::verified_marker_path(path)) {
            let f: Vec<&str> = marker.split_whitespace().collect();
            if f.len() == 4
                && f[0] == self.sha256
                && f[1] == md.len().to_string()
                && f[2] == secs.to_string()
                && f[3] == nanos.to_string()
            {
                return Ok(Verified::Cached);
            }
        }
        let actual = sha256_file(path)?;
        if actual != self.sha256 {
            return Err(corrupt("sha256", self.sha256.clone(), actual));
        }
        let _ = self.write_verified_marker(path);
        Ok(Verified::Hashed)
    }

    /// `true` if `path` exists with exactly the pinned size and SHA-256.
    /// The size is compared first as a cheap pre-check; the hash of a 100 MB
    /// ephemeris takes ~0.3 s and is only computed when the size matches.
    pub fn verify(&self, path: &Path) -> std::io::Result<bool> {
        let md = match std::fs::metadata(path) {
            Ok(md) => md,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        if !md.is_file() || md.len() != self.size {
            return Ok(false);
        }
        Ok(sha256_file(path)? == self.sha256)
    }
}

/// Fetch one static file into `dest_dir`, verifying it against the manifest.
///
/// * If `<dest_dir>/<name>` already exists with the pinned size and hash and
///   `force` is `false`, nothing is downloaded ([`FetchOutcome::AlreadyPresent`]).
/// * Otherwise each candidate URL ([`ManifestEntry::candidate_urls`]) is tried
///   in order. The response is streamed to `<name>.part` while being hashed;
///   the file is renamed into place only if size and SHA-256 match, otherwise
///   the partial file is removed and the next URL is tried.
/// * If every URL fails, [`Error::AllSourcesFailed`] lists each URL and why.
#[cfg(feature = "download")]
pub fn fetch_static_file(
    entry: &ManifestEntry,
    dest_dir: &Path,
    force: bool,
) -> Result<FetchOutcome> {
    validate_file_name(&entry.name)?;
    let dest = dest_dir.join(&entry.name);
    if !force && dest.is_file() {
        match entry.ensure_verified(&dest) {
            Ok(_) => return Ok(FetchOutcome::AlreadyPresent),
            Err(Error::CorruptFile { .. }) => eprintln!(
                "Warning: {} exists but does not match the manifest (size/sha256); re-downloading",
                dest.display()
            ),
            Err(e) => return Err(e),
        }
    }
    if download::is_offline() {
        return Err(Error::Offline {
            name: entry.name.clone(),
            reason: "SATKIT_OFFLINE is set",
            urls: entry.candidate_urls(),
        });
    }
    if !dest_dir.is_dir() {
        std::fs::create_dir_all(dest_dir)?;
    }

    let mut attempts: Vec<String> = Vec::new();
    let mut hint: Option<String> = None;
    for url in entry.candidate_urls() {
        match download_verified(&url, entry, &dest) {
            Ok(()) => return Ok(FetchOutcome::Downloaded { url }),
            Err(e) => {
                // Every source fails the same way behind an intercepting
                // proxy; carry the explanation out once instead of leaving
                // four bare `invalid peer certificate` lines.
                if hint.is_none() {
                    if let Error::Http(ref http) = e {
                        hint = download::tls_trust_hint(http).map(|h| format!("\n{h}"));
                    }
                }
                attempts.push(format!("{url}: {e}"));
            }
        }
    }
    Err(Error::AllSourcesFailed {
        name: entry.name.clone(),
        attempts,
        hint,
    })
}

/// Without the `download` feature no network I/O is possible: a file that is
/// already present and verified is reported as such, anything else is a
/// typed [`Error::Offline`] naming the manifest URLs.
#[cfg(not(feature = "download"))]
pub fn fetch_static_file(
    entry: &ManifestEntry,
    dest_dir: &Path,
    force: bool,
) -> Result<FetchOutcome> {
    validate_file_name(&entry.name)?;
    let dest = dest_dir.join(&entry.name);
    if !force && dest.is_file() {
        match entry.ensure_verified(&dest) {
            Ok(_) => return Ok(FetchOutcome::AlreadyPresent),
            Err(e @ Error::CorruptFile { .. }) => return Err(e),
            Err(e) => return Err(e),
        }
    }
    Err(download::offline_error(
        &entry.name,
        "satkit was built without the `download` feature",
    ))
}

/// GET `url` into `dest` atomically, verifying size and SHA-256 before the
/// final rename.
#[cfg(feature = "download")]
fn download_verified(url: &str, entry: &ManifestEntry, dest: &Path) -> Result<()> {
    let agent = crate::utils::download::http_agent();
    let mut resp = agent.get(url).call()?;
    let mut reader = resp.body_mut().as_reader();
    download::write_atomic_verified(&mut reader, dest, entry, url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_valid() {
        let m = Manifest::parse(MANIFEST_JSON).expect("embedded manifest parses and validates");
        assert_eq!(m.manifest_version, 2);
        assert!(m.entry("linux_p1550p2650.440").is_some());
        assert!(m.entry("tab5.2a.txt").is_some());
        assert!(m.entry("EGM96.gfc").is_some());
        assert!(
            m.entry("msis21.parm").is_none(),
            "NRLMSIS 2.1 licence forbids redistribution"
        );
        assert!(
            m.entry("leap-seconds.list").is_none(),
            "nothing reads it; the runtime leap-second table is compiled in"
        );
        // Refresh files must not be pinned.
        assert!(m.entry("EOP-All.csv").is_none());
        assert!(m.entry("SW-All.csv").is_none());
        assert!(m.refresh.iter().any(|u| u.ends_with("EOP-All.csv")));
        // Every entry: GitHub release asset first.
        for e in &m.files {
            assert!(
                e.urls[0].starts_with(&m.release_base),
                "{}: first URL should be the release asset",
                e.name
            );
        }
        // The only default download is the DE440 ephemeris: everything else
        // pinned here is either embedded in the binary (IERS tables,
        // gravity to degree 70) or an alternative (DE421), kept fetchable
        // by name but not worth downloading on every install.
        let defaults: Vec<&str> = m.default_files().map(|e| e.name.as_str()).collect();
        assert_eq!(defaults, ["linux_p1550p2650.440"]);
        assert_eq!(embedded().data_version, m.data_version);
    }

    #[test]
    fn validation_rejects_bad_entries() {
        let base = r#"{"manifest_version":2,"data_version":"x","release_base":"https://h/r","files":[FILES],"refresh":[]}"#;
        let ok = r#"{"name":"a.bin","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","urls":["https://h/r/a.bin"]}"#;
        assert!(Manifest::parse(&base.replace("FILES", ok)).is_ok());
        for (bad, why) in [
            (ok.replace("\"size\":1", "\"size\":0"), "zero size"),
            (ok.replace("aaaa", "AAAA"), "upper-case hash"),
            (ok.replace("aaaaaaaa", "aaaaaaa"), "short hash"),
            (ok.replace("https://", "http://"), "http url"),
            (ok.replace("\"a.bin\"", "\"../a.bin\""), "path escape"),
            (ok.replace("\"a.bin\"", "\"/a.bin\""), "absolute"),
            (ok.replace("[\"https://h/r/a.bin\"]", "[]"), "no urls"),
            (format!("{ok},{ok}"), "duplicate"),
        ] {
            assert!(
                Manifest::parse(&base.replace("FILES", &bad)).is_err(),
                "{why}"
            );
        }
        assert!(Manifest::parse(
            &base
                .replace("FILES", ok)
                .replace("\"manifest_version\":2", "\"manifest_version\":1")
        )
        .is_err());
    }

    #[test]
    fn file_name_validation() {
        for ok in ["EGM96.gfc", "sw-data_v2", "a.b.c"] {
            assert!(validate_file_name(ok).is_ok(), "{ok:?}");
        }
        for bad in [
            "",
            ".",
            "..",
            "../x",
            "/etc/passwd",
            "a/b",
            "a\\b",
            "x/../y",
            "C:\\x",
            "a\0b",
        ] {
            assert!(
                matches!(validate_file_name(bad), Err(Error::InvalidFileName { .. })),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let dir = std::env::temp_dir().join(format!("satkit_sha_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc.txt");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), sha256_hex(b"abc"));
        let entry = ManifestEntry {
            name: "abc.txt".into(),
            size: 3,
            sha256: sha256_hex(b"abc"),
            urls: vec!["https://example.invalid/abc.txt".into()],
            source: String::new(),
            license: String::new(),
            tier: String::new(),
            default: true,
        };
        assert!(entry.verify(&p).unwrap());
        std::fs::write(&p, b"abd").unwrap();
        assert!(!entry.verify(&p).unwrap(), "same size, different bytes");
        std::fs::write(&p, b"abcd").unwrap();
        assert!(!entry.verify(&p).unwrap(), "different size");
        assert!(!entry.verify(&dir.join("missing")).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cost of the one-time hash of the real DE440 file (the largest pinned
    /// file), for the docs: `cargo test --lib time_de440_hash -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the real DE440 file in a search directory"]
    fn time_de440_hash() {
        let Some(path) = crate::utils::find_data_file("linux_p1550p2650.440") else {
            eprintln!("DE440 not present; skipping");
            return;
        };
        let entry = embedded().entry("linux_p1550p2650.440").unwrap();
        let marker = ManifestEntry::verified_marker_path(&path);
        let _ = std::fs::remove_file(&marker);
        let t = std::time::Instant::now();
        let first = entry.ensure_verified(&path).unwrap();
        let hashed = t.elapsed();
        let t = std::time::Instant::now();
        let second = entry.ensure_verified(&path).unwrap();
        let cached = t.elapsed();
        println!("DE440 verify: first {first:?} in {hashed:?}; second {second:?} in {cached:?}");
        assert_eq!(second, Verified::Cached);
    }

    #[test]
    fn mirror_env_is_tried_first() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let entry = embedded().entry("tab5.2d.txt").unwrap().clone();
        std::env::set_var(MIRROR_ENV, "http://mirror.local/data/");
        let urls = entry.candidate_urls();
        std::env::remove_var(MIRROR_ENV);
        assert_eq!(urls[0], "http://mirror.local/data/tab5.2d.txt");
        assert_eq!(&urls[1..], &entry.urls[..]);
        assert_eq!(entry.candidate_urls(), entry.urls);
    }
}
