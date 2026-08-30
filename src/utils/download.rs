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

    /// A download was needed but is forbidden: either `SATKIT_OFFLINE=1` is
    /// set or satkit was built without the `download` feature. No network I/O
    /// is attempted. `urls` lists where the file could be obtained manually
    /// (from the data manifest, when the file is a pinned one).
    #[error(
        "{name} is not present and cannot be downloaded ({reason}). \
         Provide it in the data directory (SATKIT_DATA) or install the `satkit-data` bundle; \
         sources: {}",
        if urls.is_empty() { "(none listed)".to_string() } else { urls.join(", ") }
    )]
    Offline {
        name: String,
        reason: &'static str,
        urls: Vec<String>,
    },

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
    /// `"<url>: <error>"` line per URL tried and `hint`, when the failures
    /// have a common actionable cause (a TLS trust failure behind an
    /// intercepting proxy), says what to do about it — once, rather than
    /// repeated under every URL.
    #[error(
        "Could not download {name} from any source:\n  {}{}",
        attempts.join("\n  "),
        hint.as_deref().unwrap_or("")
    )]
    AllSourcesFailed {
        name: String,
        attempts: Vec<String>,
        hint: Option<String>,
    },

    /// A download completed, but its content is not what the file is meant
    /// to hold — a captive portal or proxy interstitial answering `200 OK`
    /// with an HTML page, or a truncated feed. The partial download is
    /// discarded and any existing copy left in place, so a good data file is
    /// never replaced by a bad one.
    #[error(
        "{name} downloaded from {url} was rejected: {reason}. \
         The partial download was discarded and any existing {name} left in place \
         (a proxy that answers with an HTML notice instead of the file looks like this)"
    )]
    ContentRejected {
        name: String,
        url: String,
        reason: String,
    },

    /// Replacing an existing data file failed even after retries. On
    /// Windows a file another process has open cannot be renamed over; on
    /// any platform this also covers a directory that became read-only.
    #[error("could not replace {path} (is another process using it?): {source}")]
    ReplaceFailed {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// A manifest-pinned file present on disk does not match its pinned size
    /// or SHA-256 — corrupt, truncated, or a different file under the same
    /// name. When downloads are allowed the file is re-fetched; under offline
    /// mode (or without the `download` feature) this error is returned.
    #[error(
        "{name} at {path} is corrupt: {what} {} does not match the manifest ({}). \
         Delete or replace the file, or allow downloads so it can be re-fetched",
        .values.1, .values.0
    )]
    CorruptFile {
        name: String,
        path: String,
        what: &'static str,
        /// `(expected, actual)`; boxed to keep the error type small.
        values: Box<(String, String)>,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "download")]
    #[error(transparent)]
    Http(#[from] ureq::Error),

    /// An HTTP request failed. Unlike [`Error::Http`] this names the URL that
    /// was being fetched and, when the failure is one whose underlying message
    /// is too terse to act on (a TLS trust failure behind an intercepting
    /// proxy), appends guidance on how to fix it.
    #[cfg(feature = "download")]
    #[error("could not fetch {url}: {source}{}", hint.as_deref().unwrap_or(""))]
    Request {
        url: String,
        /// Boxed to keep this enum — and every error type that carries it,
        /// up to `orbitprop::Error` — small; `ureq::Error` is 64 bytes on its
        /// own, which is most of the budget `clippy::result_large_err` allows.
        #[source]
        source: Box<ureq::Error>,
        /// Guidance appended to the message, already formatted with a leading
        /// newline; `None` when the underlying error speaks for itself.
        hint: Option<String>,
    },
}

/// Convenient type alias used throughout the `download` module.
pub type Result<T> = std::result::Result<T, Error>;

/// Environment variable that forbids all downloads: when set to any value
/// other than `0`/`false`/empty, every download helper returns
/// [`Error::Offline`] immediately instead of opening a connection. Use it in
/// sandboxes, CI, and air-gapped deployments to make a missing file a loud,
/// typed error rather than a hang or a surprise download. [`set_offline`]
/// overrides it programmatically.
pub const OFFLINE_ENV: &str = "SATKIT_OFFLINE";

/// Offline-mode state: 0 = not set programmatically (use the environment),
/// 1 = downloads allowed, 2 = downloads forbidden.
static OFFLINE_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force offline mode on or off for this process, overriding
/// [`OFFLINE_ENV`]. Precedence: the last call to `set_offline` wins; if it
/// was never called, the environment variable is consulted.
///
/// Offline mode blocks **downloads only** — the explicit
/// [`update_datafiles`](crate::utils::update_datafiles) and every lazy
/// first-use fetch (ephemeris, EOP / space-weather refresh, non-embedded
/// files). It does not change where files are searched, and the compiled-in
/// core data is unaffected. The error returned is the same typed
/// [`Error::Offline`] a build without the `download` feature returns.
pub fn set_offline(offline: bool) {
    OFFLINE_OVERRIDE.store(
        if offline { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// `true` if downloads are currently forbidden (see [`set_offline`] for the
/// precedence between the setter and [`OFFLINE_ENV`]). One atomic load when
/// the setter has been used; an environment lookup otherwise.
pub fn is_offline() -> bool {
    match OFFLINE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => std::env::var(OFFLINE_ENV)
            .map(|v| !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(false),
    }
}

/// Alias of [`is_offline`].
pub fn offline_requested() -> bool {
    is_offline()
}

/// Forget any [`set_offline`] call so the environment decides again
/// (test helper: keeps one test's override from leaking into the next).
#[cfg(test)]
pub(crate) fn clear_offline_override() {
    OFFLINE_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// The manifest URLs for `name`, for error messages (empty if not pinned).
fn manifest_urls(name: &str) -> Vec<String> {
    crate::utils::manifest::embedded()
        .entry(name)
        .map(|e| e.urls.clone())
        .unwrap_or_default()
}

/// Build the [`Error::Offline`] for `name`, naming why no download is possible.
pub(crate) fn offline_error(name: &str, reason: &'static str) -> Error {
    Error::Offline {
        name: name.to_string(),
        reason,
        urls: manifest_urls(name),
    }
}

/// Return [`Error::Offline`] if [`OFFLINE_ENV`] is set (checked before any
/// network I/O by every download helper).
/// User-Agent sent with every HTTP request satkit makes.
///
/// CelesTrak's usage policy asks clients to identify themselves and not to
/// re-request the same GP data more than about once every two hours; an
/// anonymous default agent string (`ureq/3.x`) is both unidentifiable and
/// shared with every other ureq user, which makes throttling decisions land
/// on satkit's requests. A descriptive, versioned agent also lets data
/// providers reach the project if a client misbehaves.
pub(crate) const USER_AGENT: &str = concat!(
    "satkit/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/ssmichael1/satkit)"
);

/// Environment variable that chooses the root certificates satkit verifies
/// download servers against. Three accepted values:
///
/// * a path to a PEM file — trust exactly the certificates in it. It may
///   hold any number of concatenated certificates and it *replaces* the
///   trust store, so it must also cover satkit's other download hosts
///   (GitHub, JPL, CelesTrak): the usual recipe is a public bundle
///   (`python -m certifi`) with the private CA appended.
/// * `platform` — the operating system's own trust store: the macOS keychain,
///   the Windows certificate store, `/etc/ssl` and friends on Unix. This is
///   the default, and it is where a TLS-inspecting proxy's private CA is
///   installed.
/// * `webpki` — the Mozilla root list compiled into the binary, ignoring the
///   machine entirely. For a minimal container with no system trust store,
///   and for anyone who would rather not trust whatever an administrator has
///   added to that store.
///
/// Set it to a PEM path where TLS is intercepted by a proxy whose CA is not
/// installed system-wide.
///
/// Deliberately not `SSL_CERT_FILE`: Python tooling routinely points that at
/// a stock public bundle, which is exactly the trust store that fails on an
/// intercepting network — honouring it would break the case this variable
/// exists to fix.
pub const CA_BUNDLE_ENV: &str = "SATKIT_CA_BUNDLE";

/// The root certificates satkit verifies servers against: those in
/// [`CA_BUNDLE_ENV`] when it names a readable PEM file, otherwise the
/// platform's own trust store (macOS keychain, Windows certificate store,
/// `/etc/ssl` and friends on Unix).
///
/// The platform store rather than ureq's default of the Mozilla root list
/// compiled into the binary: an organisation whose gateway inspects TLS
/// re-signs every certificate with a private CA, which is installed in the
/// platform store and can never be in the Mozilla list. Against the
/// compiled-in list every download on such a network fails with
/// `invalid peer certificate: UnknownIssuer`. The trade-off is deliberate —
/// trusting the platform store means trusting whatever an administrator has
/// added to it.
#[cfg(feature = "download")]
fn root_certs() -> ureq::tls::RootCerts {
    let setting = match std::env::var_os(CA_BUNDLE_ENV) {
        Some(v) if !v.is_empty() => v,
        _ => return ureq::tls::RootCerts::PlatformVerifier,
    };
    if setting.eq_ignore_ascii_case("platform") {
        return ureq::tls::RootCerts::PlatformVerifier;
    }
    if setting.eq_ignore_ascii_case("webpki") {
        return ureq::tls::RootCerts::WebPki;
    }
    let path = std::path::PathBuf::from(setting);
    match load_ca_bundle(&path) {
        Ok(roots) => roots,
        Err(reason) => {
            eprintln!(
                "Warning: ignoring {CA_BUNDLE_ENV}={}: {reason}; \
                 verifying against the platform trust store instead",
                path.display()
            );
            ureq::tls::RootCerts::PlatformVerifier
        }
    }
}

/// Every certificate in the PEM file at `path`, as a root-certificate set.
/// `Err` carries the reason for the warning [`root_certs`] prints: the file
/// is unreadable, is not valid PEM, or holds no certificate.
#[cfg(feature = "download")]
fn load_ca_bundle(path: &Path) -> std::result::Result<ureq::tls::RootCerts, String> {
    let pem = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut certs = Vec::new();
    for item in ureq::tls::parse_pem(&pem) {
        match item {
            Ok(ureq::tls::PemItem::Certificate(cert)) => certs.push(cert),
            // A bundle that also carries a private key is odd but harmless.
            Ok(_) => {}
            Err(e) => return Err(format!("not a valid PEM file ({e})")),
        }
    }
    if certs.is_empty() {
        return Err("no certificate found in the file".to_string());
    }
    Ok(ureq::tls::RootCerts::from(certs))
}

/// Build the HTTP agent used for all satkit downloads: ureq's defaults
/// (proxy settings from `HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY`, redirects,
/// timeouts) plus the [`USER_AGENT`] string and the [`root_certs`] trust
/// store.
#[cfg(feature = "download")]
pub(crate) fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .user_agent(USER_AGENT)
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .root_certs(root_certs())
                .build(),
        )
        .build()
        .into()
}

/// For a TLS trust failure — what an intercepting proxy produces when its CA
/// is not in the trust store satkit is using — an actionable message.
/// `None` for any other error.
#[cfg(feature = "download")]
pub(crate) fn tls_trust_hint(err: &ureq::Error) -> Option<String> {
    let trust_failure = match err {
        ureq::Error::Io(e) => {
            let msg = e.to_string();
            msg.contains("invalid peer certificate") || msg.contains("UnknownIssuer")
        }
        ureq::Error::Tls(_) => true,
        _ => false,
    };
    if !trust_failure {
        return None;
    }
    Some(format!(
        "The server's certificate is not trusted by this machine's certificate store. \
         This is what a TLS-inspecting proxy looks like: it re-signs traffic with a private \
         CA that must be installed in the system trust store. Install that CA system-wide, \
         or set {CA_BUNDLE_ENV} to a PEM file holding it together with the public roots. \
         If no such proxy is expected on this network, the certificate really is untrusted \
         and the download should not be forced through."
    ))
}

/// Wrap a failed request as [`Error::Request`]: the URL that was being
/// fetched, plus a hint for the failures whose own message does not say what
/// to do about them.
#[cfg(feature = "download")]
pub(crate) fn request_error(url: &str, source: ureq::Error) -> Error {
    let hint = celestrak_throttle_hint(url, &source)
        .or_else(|| tls_trust_hint(&source))
        .map(|h| format!("\n{h}"));
    Error::Request {
        url: url.to_string(),
        source: Box::new(source),
        hint,
    }
}

/// For an HTTP error from a `celestrak.org` request, an actionable message
/// explaining CelesTrak's throttling of repeated identical GP queries
/// (HTTP 503, sometimes 403). `None` for any other host or error.
#[cfg(feature = "download")]
pub(crate) fn celestrak_throttle_hint(url: &str, err: &ureq::Error) -> Option<String> {
    let status = match err {
        ureq::Error::StatusCode(code @ (403 | 503)) => *code,
        _ => return None,
    };
    let host_ok = url
        .split("//")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .map(|h| h.ends_with("celestrak.org") || h.ends_with("celestrak.com"))
        .unwrap_or(false);
    if !host_ok {
        return None;
    }
    let proxy_note = if status == 403 {
        " A 403 can also come from a filtering proxy between you and CelesTrak rather than from \
         CelesTrak itself — check whether the body is an HTML block page."
    } else {
        ""
    };
    Some(format!(
        "CelesTrak returned HTTP {status} for {url}. CelesTrak throttles repeated identical \
         GP queries: it asks for at most one request per object every ~2 hours (satkit already \
         sends a descriptive User-Agent). Do not retry in a loop — cache the response (save the \
         TLE/OMM text and parse it with `TLE::from_lines` / the OMM parsers) and re-fetch only \
         when a newer element set is needed.{proxy_note}"
    ))
}

#[cfg(feature = "download")]
pub(crate) fn check_online(name: &str) -> Result<()> {
    if offline_requested() {
        return Err(offline_error(name, "SATKIT_OFFLINE is set"));
    }
    Ok(())
}

/// Sequence number so every in-flight download in this process gets its own
/// temporary file (see [`part_path`]).
static PART_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Path of the temporary file a download is streamed into before the atomic
/// rename: `<final>.part.<pid>.<seq>`. The process id and a per-process
/// counter make the name unique, so two processes (or two threads) fetching
/// the same file at the same time never write into each other's partial
/// file — each completes independently and the rename is what serialises
/// them. A leftover `*.part.*` file from a killed process is harmless and
/// can simply be deleted.
#[cfg(feature = "download")]
fn part_path(final_path: &Path) -> std::path::PathBuf {
    let seq = PART_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut p = final_path.as_os_str().to_owned();
    p.push(format!(".part.{}.{seq}", std::process::id()));
    std::path::PathBuf::from(p)
}

/// Run `op` up to `attempts` times, sleeping `delay` between failures.
/// `NotFound` is returned immediately (the source of a rename is gone; no
/// retry can help). Everything else — notably the Windows sharing-violation
/// (`PermissionDenied`) raised when renaming over a file another process has
/// open, or a transient EBUSY — is retried.
pub(crate) fn retry_io<F: FnMut() -> std::io::Result<()>>(
    attempts: u32,
    delay: std::time::Duration,
    mut op: F,
) -> std::io::Result<()> {
    let mut last = None;
    for i in 0..attempts.max(1) {
        match op() {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(e),
            Err(e) => {
                last = Some(e);
                if i + 1 < attempts {
                    std::thread::sleep(delay);
                }
            }
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("retry_io: no attempts")))
}

/// Atomically move the completed temporary file onto `final_path`, replacing
/// any existing file. Retries a few times so that, on Windows, a reader that
/// briefly holds the old file open does not make the download fail; if it
/// still cannot be replaced the temporary file is removed and
/// [`Error::ReplaceFailed`] names the path.
#[cfg(feature = "download")]
fn rename_into_place(part: &Path, final_path: &Path) -> Result<()> {
    let r = retry_io(6, std::time::Duration::from_millis(50), || {
        std::fs::rename(part, final_path)
    });
    if let Err(source) = r {
        let _ = std::fs::remove_file(part);
        return Err(Error::ReplaceFailed {
            path: final_path.display().to_string(),
            source,
        });
    }
    Ok(())
}

/// `Err` if the file at `path` opens with an HTML document rather than the
/// data that was asked for. A filtering proxy or captive portal that answers
/// `200 OK` with a notice page is otherwise indistinguishable from a
/// successful download for the files satkit fetches unverified.
#[cfg(feature = "download")]
fn reject_html(path: &Path) -> std::result::Result<(), String> {
    use std::io::Read;
    let mut head = [0u8; 256];
    let n = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|e| e.to_string())?;
    let start = String::from_utf8_lossy(&head[..n])
        .trim_start_matches(['\u{feff}', ' ', '\n', '\r', '\t'])
        .to_ascii_lowercase();
    if start.starts_with("<!doctype html") || start.starts_with("<html") {
        return Err("the response is an HTML page, not the data file".to_string());
    }
    Ok(())
}

/// Check a freshly downloaded, *unverified* file before it replaces the copy
/// on disk. The manifest-pinned files are covered by their SHA-256; these are
/// the ones with nothing to compare against — so the daily CelesTrak feeds
/// are parsed with the same parser that will later read them, and anything
/// else is at least checked for not being an HTML notice page.
///
/// `name` is the file's base name; `path` is the completed `.part` file.
#[cfg(feature = "download")]
fn check_content(name: &str, path: &Path) -> std::result::Result<(), String> {
    let is_html_file = std::path::Path::new(name)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm"));
    if !is_html_file {
        reject_html(path)?;
    }
    match name {
        "EOP-All.csv" => crate::earth_orientation_params::validate_file(path),
        "SW-All.csv" => crate::spaceweather::validate_file(path),
        _ => Ok(()),
    }
}

/// Stream `reader` to `final_path` atomically: write to a sibling `.part` file
/// and rename it into place on success. An interrupted transfer (network drop,
/// Ctrl-C) then leaves only the discardable `.part` file rather than a truncated
/// final file that later runs would trust as complete.
///
/// The completed `.part` file is passed through [`check_content`] before the
/// rename: these downloads carry no hash to verify, and replacing a good
/// `EOP-All.csv` with a proxy's notice page would turn a failed download into
/// a silently wrong table days later. `url` only names the source in
/// [`Error::ContentRejected`].
#[cfg(feature = "download")]
fn write_atomic(reader: &mut impl std::io::Read, final_path: &Path, url: &str) -> Result<()> {
    let part = part_path(final_path);
    let mut write = || -> Result<()> {
        let mut dest = std::fs::File::create(&part)?;
        std::io::copy(reader, &mut dest)?;
        dest.sync_all()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    let name = final_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    if let Err(reason) = check_content(name, &part) {
        let _ = std::fs::remove_file(&part);
        return Err(Error::ContentRejected {
            name: name.to_string(),
            url: url.to_string(),
            reason,
        });
    }
    rename_into_place(&part, final_path)
}

/// Like [`write_atomic`], but the stream is hashed as it is written and the
/// `.part` file is only renamed into place if its size and SHA-256 match
/// `entry`. On mismatch the partial file is deleted and
/// [`Error::HashMismatch`] is returned, so a corrupt or substituted download
/// never becomes a trusted data file. Used by
/// [`manifest::fetch_static_file`](crate::utils::manifest::fetch_static_file).
#[cfg(feature = "download")]
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
    // Another process may have completed the same download while this one
    // was in flight (each writes its own `.part.<pid>.<seq>`). If a verified
    // final file is already there, ours is redundant: discard it rather than
    // replacing a file a third process may just have started reading.
    if final_path.is_file() && entry.verify(final_path).unwrap_or(false) {
        let _ = std::fs::remove_file(&part);
        let _ = entry.write_verified_marker(final_path);
        return Ok(());
    }
    rename_into_place(&part, final_path)?;
    let _ = entry.write_verified_marker(final_path);
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
    check_online(basename)?;
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
    let agent = http_agent();

    let mut resp = agent
        .get(url.as_str())
        .call()
        .map_err(|e| request_error(&url, e))?;

    write_atomic(&mut resp.body_mut().as_reader(), fname, &url)?;
    Ok(())
}

#[cfg(not(feature = "download"))]
pub fn download_if_not_exist(fname: &Path, _seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        return Ok(());
    }
    let name = fname
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("<unnamed>");
    Err(offline_error(
        name,
        "satkit was built without the `download` feature",
    ))
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
    check_online(fname)?;

    let agent = http_agent();
    let mut resp = agent.get(url).call().map_err(|e| request_error(url, e))?;

    println!("Downloading {}", fname);
    write_atomic(&mut resp.body_mut().as_reader(), &fullpath, url)?;
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
    check_online(url)?;
    let agent = http_agent();
    let mut resp = agent.get(url).call().map_err(|e| request_error(url, e))?;
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
        write_atomic(&mut Cursor::new(&data[..]), &final_path, "test").unwrap();

        assert!(
            final_path.is_file(),
            "final file should exist after success"
        );
        assert!(!part_path.exists(), "the .part file must be renamed away");
        assert!(
            !leftover_parts(&final_path),
            "no .part.<pid>.<seq> file may remain"
        );
        assert_eq!(std::fs::read(&final_path).unwrap(), data);
        let _ = std::fs::remove_file(&final_path);
    }

    /// Any `<final>.part.*` sibling still on disk.
    fn leftover_parts(final_path: &Path) -> bool {
        let dir = final_path.parent().unwrap();
        let prefix = format!(
            "{}.part.",
            final_path.file_name().unwrap().to_string_lossy()
        );
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    }

    #[test]
    fn part_paths_are_unique_per_call() {
        let f = Path::new("/tmp/x.bin");
        let a = part_path(f);
        let b = part_path(f);
        assert_ne!(a, b);
        assert!(a
            .to_string_lossy()
            .contains(&format!(".part.{}.", std::process::id())));
    }

    /// The rename helper retries transient failures (the Windows
    /// sharing-violation case) but gives up after the configured attempts,
    /// and never retries a missing source.
    #[test]
    fn retry_io_retries_transient_failures_then_gives_up() {
        let mut calls = 0;
        let r = retry_io(5, std::time::Duration::from_millis(1), || {
            calls += 1;
            if calls < 3 {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "busy",
                ))
            } else {
                Ok(())
            }
        });
        assert!(r.is_ok());
        assert_eq!(calls, 3, "succeeds on the third attempt");

        let mut calls = 0;
        let r = retry_io(4, std::time::Duration::from_millis(1), || {
            calls += 1;
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "busy",
            ))
        });
        assert_eq!(r.unwrap_err().kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(calls, 4, "exhausts every attempt");

        let mut calls = 0;
        let r = retry_io(4, std::time::Duration::from_millis(1), || {
            calls += 1;
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"))
        });
        assert_eq!(r.unwrap_err().kind(), std::io::ErrorKind::NotFound);
        assert_eq!(calls, 1, "a missing source is not retried");
    }

    /// `rename_into_place` surfaces a persistent failure as a typed error
    /// naming the destination and removes the temporary file.
    #[test]
    fn rename_into_place_reports_typed_error() {
        let dir = std::env::temp_dir().join(format!("satkit_rename_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("f.bin.part.1.1");
        std::fs::write(&part, b"x").unwrap();
        // Renaming onto a path whose parent does not exist cannot succeed.
        let dest = dir.join("missing-subdir").join("f.bin");
        let err = rename_into_place(&part, &dest).unwrap_err();
        assert!(matches!(err, Error::ReplaceFailed { .. }), "{err}");
        assert!(err.to_string().contains("f.bin"));
        assert!(!part.exists(), "temporary file is cleaned up");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One header line and one row, in the columns `parse_csv` reads.
    const EOP_CSV: &str = "DATE,MJD,X,Y,UT1-UTC,LOD,dPSI,dEPS,DX,DY,DAT,DATA_TYPE\n\
                           2026-08-30,61282,0.10,0.20,0.30,0.0004,0,0,0.0001,0.0002,37,O\n";

    #[test]
    fn a_proxy_notice_page_never_replaces_a_good_data_file() {
        let dir = std::env::temp_dir().join(format!("satkit_notice_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("EOP-All.csv");

        // A good table on disk, as a machine that has been online would have.
        write_atomic(&mut Cursor::new(EOP_CSV.as_bytes()), &path, "test").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), EOP_CSV);

        // The daily refresh comes back as a proxy interstitial with HTTP 200.
        let page = b"<!DOCTYPE html>\n<html><body>Review Usage Policy</body></html>";
        let err = write_atomic(
            &mut Cursor::new(&page[..]),
            &path,
            "https://celestrak.org/SpaceData/EOP-All.csv",
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::ContentRejected { ref name, .. } if name == "EOP-All.csv"),
            "{err}"
        );
        assert!(err.to_string().contains("HTML page"), "{err}");
        // The table that was there is still there, and nothing is left over.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), EOP_CSV);
        assert!(!part_path(&path).exists());

        // A body that is not HTML but does not parse is rejected just as well.
        let err = write_atomic(&mut Cursor::new(&b"1,2,3\n"[..]), &path, "test").unwrap_err();
        assert!(matches!(err, Error::ContentRejected { .. }), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), EOP_CSV);

        // A file with no content check of its own only has to not be a page.
        let other = dir.join("notes.txt");
        write_atomic(&mut Cursor::new(&b"1,2,3\n"[..]), &other, "test").unwrap();
        assert!(other.is_file());
        let err = write_atomic(&mut Cursor::new(&page[..]), &other, "test").unwrap_err();
        assert!(matches!(err, Error::ContentRejected { .. }), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
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

#[cfg(all(test, feature = "download"))]
mod agent_tests {
    use super::*;

    #[test]
    fn agent_sends_descriptive_user_agent() {
        let agent = http_agent();
        match agent.config().user_agent() {
            ureq::config::AutoHeaderValue::Provided(v) => assert_eq!(v.as_str(), USER_AGENT),
            other => panic!("expected a provided User-Agent, got {other:?}"),
        }
        assert!(USER_AGENT.starts_with("satkit/"));
        assert!(USER_AGENT.contains("github.com/ssmichael1/satkit"));
    }

    /// A self-signed certificate, generated for these tests only, used to
    /// check that a PEM bundle is read and every certificate in it kept.
    const TEST_CERT_PEM: &str = "\
-----BEGIN CERTIFICATE-----\n\
MIICtjCCAZ4CCQD+TdvyxtLNoDANBgkqhkiG9w0BAQsFADAdMRswGQYDVQQDDBJz\n\
YXRraXQgdGVzdCByb290IDEwHhcNMjYwODMwMTkxNjMyWhcNMzYwODI3MTkxNjMy\n\
WjAdMRswGQYDVQQDDBJzYXRraXQgdGVzdCByb290IDEwggEiMA0GCSqGSIb3DQEB\n\
AQUAA4IBDwAwggEKAoIBAQC8KtMusQe6lwq6iV46RqCYg7fXePr+hiPs0oV5z4xo\n\
Mae9oZ3Sz/C9Vu2hwk+WhACelNYMA9FKkeRJzN4DOH3PKvsqELFW2mZRzV9iwYn/\n\
68p//+SVLgKXjjf+dFUt1QJin27OCfnSREgbclf50+V/1ZtntVSW5laCBkrrvIR0\n\
ro1xDqjopt8vSODmkYyO/bGnmTYP2w+n/7imCSJ0SsjNHHTSG1r9SQtm7jqfF/lb\n\
EeY+8J6j2zFEJ+XC+WSYbVrCrsthE/FEoEg3XBexX2gDbty4xABFdSQJ4vqOaoA5\n\
Re3pAsYCEa6ZT19JXtPX77DZ+6qcaHjAKna+LhCrpfV1AgMBAAEwDQYJKoZIhvcN\n\
AQELBQADggEBADW+76NDtE6/dyYoBaxQXtRo4pBMBLKm6hY6EVCF6n+X+0egAG2q\n\
igGwSNYhf/4bkreX2WJSMWz2aTQwKRIcERGoHy22ftBQbtkNWmdsUazf4Wt8h2cW\n\
+G3iY4j7trhc5wf5vQxSGzKJpUArWmslQzezYsOJXcaVIBi0ib3c8bWxU51fy7TQ\n\
EJ2v6j5DChUCY06hOu2Nc+uc1rt8JoPMspPKyWBup18RQAuJ2vHoS0Do++nhKN/0\n\
nqylLCIo7Z6QSP2wB/zARZQB9OLch0Fp5N3QsmtQpj+MQ3z9QYhySjE/ABNz8XHG\n\
19lkXVD84ByEe7n7YsO0klDklCd5NPsFtj4=\n\
-----END CERTIFICATE-----";

    /// Serialised access to [`CA_BUNDLE_ENV`] plus a scratch directory: the
    /// harness runs tests in parallel threads and the variable is process-wide.
    fn with_ca_bundle_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = crate::utils::manifest::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match value {
            Some(v) => std::env::set_var(CA_BUNDLE_ENV, v),
            None => std::env::remove_var(CA_BUNDLE_ENV),
        }
        let out = f();
        std::env::remove_var(CA_BUNDLE_ENV);
        out
    }

    #[test]
    fn root_certs_default_to_the_platform_trust_store() {
        // The point of the platform store: a TLS-inspecting proxy's private CA
        // is installed there and can never be in a compiled-in Mozilla list.
        with_ca_bundle_env(None, || {
            assert!(matches!(
                root_certs(),
                ureq::tls::RootCerts::PlatformVerifier
            ));
        });
        with_ca_bundle_env(Some("PLATFORM"), || {
            assert!(matches!(
                root_certs(),
                ureq::tls::RootCerts::PlatformVerifier
            ));
        });
    }

    #[test]
    fn ca_bundle_env_can_select_the_compiled_in_roots() {
        with_ca_bundle_env(Some("webpki"), || {
            assert!(matches!(root_certs(), ureq::tls::RootCerts::WebPki));
        });
    }

    #[test]
    fn ca_bundle_env_loads_every_certificate_in_the_file() {
        let dir = std::env::temp_dir().join(format!("satkit_ca_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bundle.pem");
        std::fs::write(&path, format!("{TEST_CERT_PEM}\n{TEST_CERT_PEM}\n")).unwrap();

        let roots = with_ca_bundle_env(Some(path.to_str().unwrap()), root_certs);
        match roots {
            ureq::tls::RootCerts::Specific(certs) => assert_eq!(certs.len(), 2),
            other => panic!("expected the file's certificates, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unusable_ca_bundle_falls_back_to_the_platform_store() {
        let dir = std::env::temp_dir().join(format!("satkit_ca_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty.pem");
        std::fs::write(&empty, b"no certificates here\n").unwrap();

        assert!(load_ca_bundle(Path::new("/satkit/no/such/bundle.pem")).is_err());
        assert!(load_ca_bundle(&empty)
            .unwrap_err()
            .contains("no certificate"));
        // A misconfigured variable must not turn into a failed download: warn
        // and verify against the platform store, which is what would have been
        // used had the variable never been set.
        for bad in [empty.to_str().unwrap(), "/satkit/no/such/bundle.pem"] {
            assert!(matches!(
                with_ca_bundle_env(Some(bad), root_certs),
                ureq::tls::RootCerts::PlatformVerifier
            ));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tls_trust_hint_only_for_certificate_failures() {
        let cert_err = ureq::Error::Io(std::io::Error::other(
            "invalid peer certificate: UnknownIssuer",
        ));
        let hint = tls_trust_hint(&cert_err).unwrap();
        assert!(hint.contains(CA_BUNDLE_ENV) && hint.contains("proxy"));
        assert!(
            tls_trust_hint(&ureq::Error::Io(std::io::Error::other("connection reset"))).is_none()
        );
        assert!(tls_trust_hint(&ureq::Error::StatusCode(500)).is_none());

        // The wrapper names the URL and carries the hint into the message.
        let err = request_error("https://example.org/f.bin", cert_err);
        let msg = err.to_string();
        assert!(matches!(err, Error::Request { .. }), "{msg}");
        assert!(msg.contains("https://example.org/f.bin") && msg.contains(CA_BUNDLE_ENV));
        // An error that speaks for itself gets no hint, only the URL.
        let plain = request_error(
            "https://example.org/f.bin",
            ureq::Error::Io(std::io::Error::other("connection reset")),
        );
        assert!(matches!(plain, Error::Request { hint: None, .. }));
        assert!(plain.to_string().contains("connection reset"));
    }

    #[test]
    fn celestrak_throttle_hint_only_for_celestrak_503_403() {
        let url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE";
        let hint = celestrak_throttle_hint(url, &ureq::Error::StatusCode(503)).unwrap();
        assert!(
            hint.contains("HTTP 503") && hint.contains("2 hours") && hint.contains("from_lines")
        );
        assert!(celestrak_throttle_hint(url, &ureq::Error::StatusCode(403)).is_some());
        assert!(celestrak_throttle_hint(url, &ureq::Error::StatusCode(404)).is_none());
        assert!(
            celestrak_throttle_hint("https://example.org/x", &ureq::Error::StatusCode(503))
                .is_none()
        );
    }
}
