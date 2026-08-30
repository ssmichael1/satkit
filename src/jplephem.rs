//!
//! JPL Solar System Ephemerides
//!
//! # Introduction
//!
//! This module provides high-precision ephemerides
//! for bodies in the solar system, calculated from the
//! Jet Propulsion Laboratory (JPL) ephemeris data files
//!
//! # Links
//!
//! Ephemerides files can be found at:
//! <https://ssd.jpl.nasa.gov/ftp/eph/planets//>
//!
//!
//! # Notes
//!
//! For little-endian systems, download from the "Linux" subdirectory
//! For big-endian systems, download from the "SunOS" subdirectory
//!

use crate::solarsystem::SolarSystem;

use crate::utils::{datadir, download_if_not_exist};

use std::array::TryFromSliceError;
use std::num::{ParseFloatError, ParseIntError};
use std::str::Utf8Error;
use std::string::FromUtf8Error;
use std::sync::OnceLock;

use crate::mathtypes::*;
use crate::{Instant, TimeLike, TimeScale};

use thiserror::Error;

/// Errors produced by the [`jplephem`](crate::jplephem) module.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned when [`Instant`]'s Julian date falls outside the
    /// `[jd_start, jd_stop]` window of the loaded ephemerides file.
    #[error("Invalid Julian date: {0}")]
    InvalidJulianDate(f64),

    /// The Chebyshev dispatcher hit a coefficient count outside the table
    /// of supported sizes — typically because the requested body is not
    /// represented in the loaded ephemeris file.
    #[error("Invalid body")]
    InvalidBody,

    /// The opened ephemerides file is shorter than required to hold the
    /// declared Chebyshev coefficient block.
    #[error("Invalid record size for cheby data")]
    InvalidRecordSize,

    /// The opened ephemerides file is too short (or too corrupt) to contain
    /// the fixed DE header and pointer table.
    #[error("Ephemeris file is truncated or malformed (header too short)")]
    InvalidHeader,

    /// The file exists but is not a JPL Linux-format binary ephemeris (its
    /// header does not start with `JPL Planetary Ephemeris DE`).
    #[error("not a JPL binary ephemeris (header starts with {0:?}); expected a file such as linux_p1550p2650.440 (DE440) or lnxp1900p2053.421 (DE421)")]
    NotJplEphemeris(String),

    /// The lazily-loaded JPL ephemeris singleton failed to load (e.g. the data
    /// file is missing). Carries the original error's message (the underlying
    /// error is not `Clone`, so it cannot be re-surfaced directly).
    #[error("Failed to load JPL ephemeris: {0}")]
    LoadFailed(String),

    /// Returned by [`init_from_bytes`] / [`init_from_path`] when the
    /// JPL ephemeris singleton has already been initialized (either by an
    /// earlier `init_*` call or by a lazy load triggered by a position
    /// query). Initialization must happen before any other call into this
    /// module.
    #[error("JPL ephemeris singleton is already initialized")]
    AlreadyInitialized,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] Utf8Error),

    #[error(transparent)]
    FromUtf8(#[from] FromUtf8Error),

    #[error(transparent)]
    TryFromSlice(#[from] TryFromSliceError),

    #[error(transparent)]
    ParseInt(#[from] ParseIntError),

    #[error(transparent)]
    ParseFloat(#[from] ParseFloatError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the `jplephem` module.
pub type Result<T> = std::result::Result<T, Error>;

impl TryFrom<i32> for SolarSystem {
    type Error = ();
    fn try_from(v: i32) -> std::result::Result<Self, Self::Error> {
        match v {
            x if x == Self::Mercury as i32 => Ok(Self::Mercury),
            x if x == Self::Venus as i32 => Ok(Self::Venus),
            x if x == Self::EMB as i32 => Ok(Self::EMB),
            x if x == Self::Mars as i32 => Ok(Self::Mars),
            x if x == Self::Jupiter as i32 => Ok(Self::Jupiter),
            x if x == Self::Saturn as i32 => Ok(Self::Saturn),
            x if x == Self::Uranus as i32 => Ok(Self::Uranus),
            x if x == Self::Neptune as i32 => Ok(Self::Neptune),
            x if x == Self::Pluto as i32 => Ok(Self::Pluto),
            x if x == Self::Moon as i32 => Ok(Self::Moon),
            x if x == Self::Sun as i32 => Ok(Self::Sun),
            _ => Err(()),
        }
    }
}

/// JPL Ephemeris Structure
///
/// Included ephemerides and solar system constants loaded from the
/// JPL ephemerides file.
///
/// Also includes functions to compute heliocentric and geocentric
/// positions and velocities of solar system bodies as a function
/// of time
#[derive(Debug)]
struct JPLEphem {
    /// Version of ephemeris code
    _de_version: i32,
    /// Julian date of start of ephemerides database
    jd_start: f64,
    /// Julian date of end of ephemerides database
    jd_stop: f64,
    /// Step size in Julian date
    jd_step: f64,
    /// Length of 1 astronomical unit, km
    _au: f64,
    /// Earth/Moon Ratio
    emrat: f64,

    // Offset lookup table
    ipt: [[usize; 3]; 15],
    consts: std::collections::HashMap<String, f64>,
    cheby: DMatrix<f64>,
}

/// Pre-computed parameters for Chebyshev polynomial evaluation
struct ChebySetup {
    t_seg: f64,
    offset0: usize,
    int_num: usize,
    nsubint: usize,
    ncoeff: usize,
}

macro_rules! dispatch_ncoeff {
    ($self:expr, $method:ident, $setup:expr) => {
        match $setup.ncoeff {
            6 => $self.$method::<6>($setup),
            7 => $self.$method::<7>($setup),
            8 => $self.$method::<8>($setup),
            10 => $self.$method::<10>($setup),
            11 => $self.$method::<11>($setup),
            12 => $self.$method::<12>($setup),
            13 => $self.$method::<13>($setup),
            14 => $self.$method::<14>($setup),
            _ => return Err(Error::InvalidBody),
        }
    };
}

/// Filename to fall back to when neither `SATKIT_JPLEPHEM_FILE` is set nor
/// any other `linux_p*.4XX` file is present in [`datadir`]. This is the only
/// JPL ephemeris file that we know lives on the GCS bundle and is therefore
/// the only one we'll attempt to auto-download.
const LEGACY_DEFAULT_FILENAME: &str = "linux_p1550p2650.440";

/// Resolve the path to the JPL ephemeris file the singleton should load.
///
/// Resolution order:
/// 1. `SATKIT_JPLEPHEM_FILE` env var. If the value contains a path separator
///    or is absolute it's used directly; otherwise it's resolved against
///    [`datadir`].
/// 2. Autodetect: scan [`datadir`] for files matching `linux_p*.4XX` or
///    `lnxp*.4XX` and pick the highest DE-version suffix (e.g. `.440` >
///    `.430` > `.421`). The two prefixes cover the entire JPL DE4XX
///    family — `lnxp*` for DE421 and earlier, `linux_p*` for DE430 and
///    later.
/// 3. Fall back to [`LEGACY_DEFAULT_FILENAME`] under [`datadir`], which will
///    be auto-downloaded on first use.
fn resolve_default_path() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Ok(v) = std::env::var("SATKIT_JPLEPHEM_FILE") {
        let p = PathBuf::from(&v);
        if p.is_absolute() || v.contains(std::path::MAIN_SEPARATOR) {
            return p;
        }
        // A bare name: wherever it is found in the search directories, or
        // the write location (so a manifest-pinned name can be downloaded).
        return crate::utils::datadir::path_for(&v);
    }

    // Autodetect across every search directory (an installed `satkit-data`
    // package, a system-wide dir, the user data dir, ...): the highest DE
    // version wins; on a tie the earlier search directory wins.
    let mut best: Option<(u32, PathBuf)> = None;
    for dd in crate::utils::data_search_dirs() {
        let Ok(rd) = std::fs::read_dir(&dd) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // JPL's Linux little-endian binaries use two prefix conventions:
            //   `linux_p<start>p<stop>.4XX` — DE430 and later (DE430/440/441)
            //   `lnxp<start>p<stop>.4XX`    — DE421 and earlier
            if !crate::utils::datadir::is_ephemeris_name(name) {
                continue;
            }
            let Ok(de_version) = name.rsplit('.').next().unwrap_or("").parse::<u32>() else {
                continue;
            };
            if best.as_ref().is_none_or(|(cur, _)| de_version > *cur) {
                best = Some((de_version, path));
            }
        }
    }
    if let Some((_, path)) = best {
        return path;
    }
    datadir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(LEGACY_DEFAULT_FILENAME)
}

/// Module-scope singleton so [`init_from_bytes`] and [`init_from_path`] can
/// populate it before any lazy load triggered by a position query.
static JPL_INSTANCE: OnceLock<Result<JPLEphem>> = OnceLock::new();

fn jplephem_singleton() -> &'static Result<JPLEphem> {
    JPL_INSTANCE.get_or_init(|| {
        let path = resolve_default_path();
        JPLEphem::from_path(&path).map_err(|e| {
            let via = match std::env::var("SATKIT_JPLEPHEM_FILE") {
                Ok(v) => format!(" (selected by SATKIT_JPLEPHEM_FILE={v:?})"),
                Err(_) => String::new(),
            };
            Error::LoadFailed(format!("{}{via}: {e}", path.display()))
        })
    })
}

/// Borrow the loaded ephemeris, mapping a cached load failure to a returned
/// error instead of panicking. Used by the public query functions so a missing
/// ephemeris file surfaces as an `Err` rather than aborting the process.
fn jpl() -> Result<&'static JPLEphem> {
    jplephem_singleton().as_ref().map_err(|e| match e {
        Error::LoadFailed(msg) => Error::LoadFailed(msg.clone()),
        other => Error::LoadFailed(other.to_string()),
    })
}

/// Initialize the JPL ephemeris singleton from an in-memory byte buffer.
///
/// This is the entry point for embedded or sandboxed contexts where the
/// ephemeris file lives in a database, application bundle, or other
/// non-filesystem source rather than on disk. The `bytes` slice must be a
/// JPL native binary DE file (little-endian "linux_p*.4XX" layout).
///
/// Must be called *before* any position / state query, otherwise the lazy
/// default-resolver init has already won and this returns
/// [`Error::AlreadyInitialized`].
pub fn init_from_bytes(bytes: &[u8]) -> Result<()> {
    JPL_INSTANCE
        .set(JPLEphem::parse(bytes))
        .map_err(|_| Error::AlreadyInitialized)
}

/// Initialize the JPL ephemeris singleton from a file at `path`.
///
/// Same semantics as [`init_from_bytes`] but reads the file from disk. Use
/// this when you want to point satkit at a specific ephemeris file at
/// runtime without going through the env-var or autodetect resolution.
///
/// Must be called *before* any position / state query, otherwise returns
/// [`Error::AlreadyInitialized`].
pub fn init_from_path(path: &std::path::Path) -> Result<()> {
    JPL_INSTANCE
        .set(JPLEphem::from_path(path))
        .map_err(|_| Error::AlreadyInitialized)
}

impl JPLEphem {
    fn consts(&self, s: &str) -> Option<&f64> {
        self.consts.get(s)
    }

    /// Compute Chebyshev setup parameters for a given body and time
    fn cheby_setup(&self, body: SolarSystem, tm: &Instant) -> Result<ChebySetup> {
        let tt = tm.as_jd_with_scale(TimeScale::TT);
        if self.jd_start > tt || self.jd_stop < tt {
            return Err(Error::InvalidJulianDate(tt));
        }

        let t_int = (tt - self.jd_start) / self.jd_step;
        // tt == jd_stop passes the range check above but floors to an index
        // one past the last record (and likewise for the sub-interval), so
        // clamp both; the endpoint then evaluates at t_seg == 1.0 exactly.
        let int_num = (t_int.floor() as usize).min(self.cheby.ncols() - 1);
        let bidx = body as usize;

        let ncoeff = self.ipt[bidx][1];
        let nsubint = self.ipt[bidx][2];

        // An unpopulated body (older/corrupt files zero its ipt entries)
        // would underflow the 1-based offset below.
        if self.ipt[bidx][0] == 0 || ncoeff == 0 || nsubint == 0 {
            return Err(Error::InvalidBody);
        }

        let t_int_2 = (t_int - int_num as f64) * nsubint as f64;
        let sub_int_num = (t_int_2.floor() as usize).min(nsubint - 1);
        let t_seg = 2.0f64.mul_add(t_int_2 - sub_int_num as f64, -1.0);

        let offset0 = self.ipt[bidx][0] - 1 + sub_int_num * ncoeff * 3;

        Ok(ChebySetup {
            t_seg,
            offset0,
            int_num,
            nsubint,
            ncoeff,
        })
    }

    /// Construct a JPL Ephemerides object from the provided binary data file.
    ///
    /// The parser is fully header-driven: `de_version`, `n_con`, the
    /// interpolation pointer table `ipt[15][3]`, kernel size, and the JD
    /// span are all read from the file header, so DE405 / DE421 / DE430 /
    /// DE440 / DE441 (including the short-span `de440s` variant) all load
    /// through the same code path.
    ///
    /// Auto-download is attempted only when the file name is one pinned by
    /// the data manifest (the default [`LEGACY_DEFAULT_FILENAME`], DE440, or
    /// the smaller DE421 `lnxp1900p2053.421`), through the SHA-256-verified
    /// fetch. For any other path the file is expected to already exist.
    /// `SATKIT_OFFLINE=1` makes a missing pinned file a typed error instead.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// use satkit::jplephem;
    /// use satkit::SolarSystem;
    /// use satkit::Instant;
    ///
    /// // Construct time: March 1 2021 12:00pm UTC
    /// let t = Instant::from_datetime(2021, 3, 1, 12, 0, 0.0).unwrap();
    ///
    /// // Find geocentric moon position at this time in the GCRF frame
    /// let p = jplephem::geocentric_pos(SolarSystem::Moon, &t).unwrap();
    /// println!("p = {}", p);
    /// ```
    ///
    fn from_path(path: &std::path::Path) -> Result<Self> {
        // The ephemeris is the one data file that is neither compiled in nor
        // refreshed daily, so it is fetched lazily on first use — but only
        // for names the data manifest pins (DE440 by default, DE421 as the
        // small alternative), through the SHA-256-verified fetch. Any other
        // path is expected to exist already. `SATKIT_OFFLINE=1` (or a build
        // without the `download` feature) turns the download into a typed
        // error naming the manifest URLs instead of touching the network.
        if !path.is_file() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if let Some(entry) = crate::utils::manifest::embedded().entry(name) {
                eprintln!(
                    "satkit: downloading the JPL ephemeris {name} ({:.0} MB) to {} \
                     (SHA-256 verified). Set SATKIT_JPLEPHEM_FILE to use another file, \
                     SATKIT_DATA_URL to fetch from a mirror, or SATKIT_OFFLINE=1 to forbid downloads.",
                    entry.size as f64 / 1e6,
                    path.parent().unwrap_or(std::path::Path::new(".")).display()
                );
                download_if_not_exist(path, None)?;
            }
        }
        Self::parse(&std::fs::read(path)?)
    }

    /// Parse a JPL ephemeris from an in-memory byte buffer.
    ///
    /// The parser is fully header-driven: `de_version`, `n_con`, the
    /// interpolation pointer table `ipt[15][3]`, the kernel size, and the JD
    /// span are all read from the file header, so DE405 / DE421 / DE430 /
    /// DE440 / DE441 all load through the same code path.
    fn parse(raw: &[u8]) -> Result<Self> {
        use std::collections::HashMap;

        // Dimensions of ephemeris for given index
        const fn dimension(idx: usize) -> usize {
            if idx == 11 {
                2
            } else if idx == 14 {
                1
            } else {
                3
            }
        }

        // The fixed header runs from byte 0 through the 15x3 pointer table that
        // ends at 2696 + 15*3*4 = 2876. Guard once here so the many fixed-offset
        // slices below can't panic on a truncated/corrupt file.
        if raw.len() < 2876 {
            return Err(Error::InvalidHeader);
        }

        let title: &str = std::str::from_utf8(&raw[0..84]).map_err(|_| {
            Error::NotJplEphemeris(String::from_utf8_lossy(&raw[0..24]).into_owned())
        })?;
        if !title.starts_with("JPL Planetary Ephemeris") {
            return Err(Error::NotJplEphemeris(title.chars().take(24).collect()));
        }

        // Get version
        let de_version: i32 = title
            .get(26..29)
            .ok_or(Error::InvalidHeader)?
            .parse()
            .map_err(|_| Error::NotJplEphemeris(title.chars().take(30).collect()))?;

        let jd_start = f64::from_le_bytes(raw[2652..2660].try_into()?);
        let jd_stop: f64 = f64::from_le_bytes(raw[2660..2668].try_into()?);
        let jd_step: f64 = f64::from_le_bytes(raw[2668..2676].try_into()?);
        let n_con: i32 = i32::from_le_bytes(raw[2676..2680].try_into()?);
        let au: f64 = f64::from_le_bytes(raw[2680..2688].try_into()?);
        let emrat: f64 = f64::from_le_bytes(raw[2688..2696].try_into()?);

        // A corrupt/crafted header would otherwise detonate downstream: a
        // non-positive or non-finite step makes the record count meaningless
        // (0.0 yields usize::MAX records), and a negative constant count is
        // nonsense.
        if !jd_start.is_finite()
            || !jd_stop.is_finite()
            || !jd_step.is_finite()
            || jd_step <= 0.0
            || jd_stop < jd_start
            || n_con < 0
        {
            return Err(Error::InvalidHeader);
        }

        // Get table
        let ipt: [[usize; 3]; 15] = {
            let mut ipt: [[usize; 3]; 15] = [[0, 0, 0]; 15];
            let mut idx = 2696;
            #[allow(clippy::needless_range_loop)]
            for ix in 0..15 {
                for iy in 0..3 {
                    ipt[ix][iy] = u32::from_le_bytes(raw[idx..(idx + 4)].try_into()?) as usize;
                    idx += 4;
                }
            }

            ipt[12][0] = ipt[12][1];
            ipt[12][1] = ipt[12][2];
            ipt[12][2] = ipt[13][0];

            if de_version > 430 && n_con != 400 {
                if n_con > 400 {
                    let idx = (n_con as usize - 400) * 6;
                    if idx + 4 > raw.len() {
                        return Err(Error::InvalidHeader);
                    }
                    ipt[13][0] = u32::from_le_bytes(raw[idx..(idx + 4)].try_into()?) as usize;
                } else {
                    ipt[13][0] = 1_usize;
                }
            }

            // Check for garbage data not populated in earlier files
            if ipt[13][0] != (ipt[12][0] + ipt[12][1] * ipt[12][2] * 3)
                || ipt[14][0] != (ipt[13][0] + ipt[13][1] * ipt[13][2] * 3)
            {
                ipt.iter_mut().skip(13).for_each(|x| {
                    x[0] = 0;
                    x[1] = 0;
                    x[2] = 0;
                });
            }
            ipt
        };

        // Kernel size. The counts come straight from the file, so use checked
        // arithmetic: a crafted file with huge ipt entries must error rather
        // than overflow.
        let kernel_size: usize = {
            let mut ks: usize = 4;
            for (ix, entry) in ipt.iter().enumerate() {
                let term = 2usize
                    .checked_mul(entry[1])
                    .and_then(|v| v.checked_mul(entry[2]))
                    .and_then(|v| v.checked_mul(dimension(ix)))
                    .ok_or(Error::InvalidRecordSize)?;
                ks = ks.checked_add(term).ok_or(Error::InvalidRecordSize)?;
            }
            ks
        };

        Ok(Self {
            _de_version: de_version,
            jd_start,
            jd_stop,
            jd_step,
            _au: au,
            emrat,
            ipt,
            consts: {
                let mut hm = HashMap::new();

                // Read in constants defined in file
                for ix in 0..n_con {
                    let ix = ix as usize;
                    let sidx: usize = kernel_size * 4 + ix * 8;
                    let eidx: usize = sidx + 8;
                    let stridx: usize = if ix >= 400 {
                        84 * 3 + 400 * 6 + 5 * 8 + 41 * 4 + ix * 6
                    } else {
                        84 * 3 + ix * 6
                    };
                    // Both offsets are derived from file-declared sizes; bounds
                    // check before slicing so a corrupt file errors cleanly.
                    if eidx > raw.len() || stridx + 6 > raw.len() {
                        return Err(Error::InvalidRecordSize);
                    }
                    let val: f64 = f64::from_le_bytes(raw[sidx..eidx].try_into()?);
                    let s = String::from_utf8(raw[stridx..(stridx + 6)].to_vec())?;

                    hm.insert(String::from(s.trim()), val);
                }
                hm
            },
            cheby: {
                let ncoeff: usize = kernel_size / 2;
                let nrecords = ((jd_stop - jd_start) / jd_step) as usize;
                let record_size = kernel_size.checked_mul(4).ok_or(Error::InvalidRecordSize)?;

                // Validate the declared sizes against the actual buffer with
                // checked arithmetic BEFORE allocating: a crafted header must
                // not trigger an overflow or a multi-GB allocation.
                let data_bytes = ncoeff
                    .checked_mul(nrecords)
                    .and_then(|v| v.checked_mul(8))
                    .ok_or(Error::InvalidRecordSize)?;
                let required = record_size
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(data_bytes))
                    .ok_or(Error::InvalidRecordSize)?;
                if raw.len() < required {
                    return Err(Error::InvalidRecordSize);
                }
                let mut v: DMatrix<f64> = DMatrix::zeros(ncoeff, nrecords);

                // Bulk-copy the coefficient block (native-endian f64s). `raw` is
                // a `Vec<u8>` (alignment 1), so we must NOT reinterpret its
                // pointer as `*const f64` — that is undefined behavior. Copy
                // byte-wise into the (8-aligned) matrix storage instead; f64 has
                // no invalid bit patterns, so this is sound and just as fast.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw.as_ptr().add(record_size * 2),
                        v.as_mut_slice().as_mut_ptr() as *mut u8,
                        ncoeff * nrecords * 8,
                    );
                }
                v
            },
        })
    }

    fn body_pos_optimized<const N: usize>(&self, setup: &ChebySetup) -> Result<Vector3> {
        let mut t = Vector::<N>::zeros();
        t[0] = 1.0;
        t[1] = setup.t_seg;
        for j in 2..N {
            t[j] = (2.0 * setup.t_seg).mul_add(t[j - 1], -t[j - 2]);
        }

        let mut pos = Vector3::zeros();
        for ix in 0..3 {
            let mut sum = 0.0;
            for k in 0..N {
                sum += self.cheby[(setup.offset0 + N * ix + k, setup.int_num)] * t[k];
            }
            pos[ix] = sum;
        }

        Ok(pos * 1.0e3)
    }

    /// Return the position of the given body in the Barycentric
    /// coordinate system (origin is solarsystem barycenter)
    ///
    /// # Inputs
    ///
    ///  * `body` - the solar system body for which to return position
    ///  * `tm` - The time at which to return position
    ///
    /// # Return
    ///
    ///  * 3-vector of cartesian Heliocentric position in meters
    ///
    ///
    /// # Notes:
    ///  * Positions for all bodies are natively relative to solar system barycenter,
    ///    with exception of moon, which is computed in Geocentric system
    ///  * EMB (2) is the Earth-Moon barycenter
    ///  * The sun position is relative to the solar system barycenter
    ///    (it will be close to origin)
    fn barycentric_pos(&self, body: SolarSystem, tm: &Instant) -> Result<Vector3> {
        let setup = self.cheby_setup(body, tm)?;
        dispatch_ncoeff!(self, body_pos_optimized, &setup)
    }
    /// Return the position & velocity the given body in the barycentric coordinate system
    /// (origin is solar system barycenter)
    ///
    /// # Arguments
    ///  * `body` - the solar system body for which to return position
    ///  * `tm` - The time at which to return position
    ///
    /// # Return
    ///  * 2-element tuple with following values:
    ///    * 3-vector of cartesian Heliocentric position in meters
    ///    * 3-vector of cartesian Heliocentric velocity in meters / second
    ///
    ///
    /// # Notes:
    ///  * Positions for all bodies are natively relative to solar system barycenter,
    ///    with exception of moon, which is computed in Geocentric system
    ///  * EMB (2) is the Earth-Moon barycenter
    ///  * The sun position is relative to the solar system barycenter
    ///    (it will be close to origin)
    fn barycentric_state(&self, body: SolarSystem, tm: &Instant) -> Result<(Vector3, Vector3)> {
        let setup = self.cheby_setup(body, tm)?;
        dispatch_ncoeff!(self, body_state_optimized, &setup)
    }

    fn body_state_optimized<const N: usize>(
        &self,
        setup: &ChebySetup,
    ) -> Result<(Vector3, Vector3)> {
        let mut t = Vector::<N>::zeros();
        let mut v = Vector::<N>::zeros();
        t[0] = 1.0;
        t[1] = setup.t_seg;
        v[0] = 0.0;
        v[1] = 1.0;
        for j in 2..N {
            t[j] = (2.0 * setup.t_seg).mul_add(t[j - 1], -t[j - 2]);
            v[j] = 2.0f64.mul_add(t[j - 1], (2.0 * setup.t_seg).mul_add(v[j - 1], -v[j - 2]));
        }

        let mut pos = Vector3::zeros();
        let mut vel = Vector3::zeros();
        for ix in 0..3 {
            let mut psum = 0.0;
            let mut vsum = 0.0;
            for k in 0..N {
                let coeff = self.cheby[(setup.offset0 + N * ix + k, setup.int_num)];
                psum += coeff * t[k];
                vsum += coeff * v[k];
            }
            pos[ix] = psum;
            vel[ix] = vsum;
        }

        Ok((
            pos * 1.0e3,
            vel * 2.0e3 * setup.nsubint as f64 / self.jd_step / 86400.0,
        ))
    }

    /// Return the position of the given body in
    /// Geocentric coordinate system
    ///
    /// # Arguments
    ///  * body - the solar system body for which to return position
    ///  * tm - The time at which to return position
    ///
    /// # Return
    ///    3-vector of cartesian Geocentric position in meters
    ///
    fn geocentric_pos(&self, body: SolarSystem, tm: &Instant) -> Result<Vector3> {
        if body == SolarSystem::Moon {
            self.barycentric_pos(body, tm)
        } else {
            let emb: Vector3 = self.barycentric_pos(SolarSystem::EMB, tm)?;
            let moon: Vector3 = self.barycentric_pos(SolarSystem::Moon, tm)?;
            let b: Vector3 = self.barycentric_pos(body, tm)?;

            // Compute the position of the body relative to the Earth-moon
            // barycenter, then "correct" to Earth-center by accounting
            // for moon position and Earth/moon mass ratio
            Ok(b - emb + moon / (1.0 + self.emrat))
        }
    }

    /// Return the position and velocity of the given body in
    ///  Geocentric coordinate system
    ///
    /// # Arguments
    ///
    ///  * `body` - the solar system body for which to return position
    ///  * `tm` - The time at which to return position
    ///
    /// # Return
    ///   * 2-element tuple with following elements:
    ///     * 3-vector of cartesian Geocentric position in meters
    ///     * 3-vector of cartesian Geocentric velocity in meters / second
    ///       Note: velocity is relative to Earth
    ///
    fn geocentric_state(&self, body: SolarSystem, tm: &Instant) -> Result<(Vector3, Vector3)> {
        if body == SolarSystem::Moon {
            self.barycentric_state(body, tm)
        } else {
            let emb: (Vector3, Vector3) = self.barycentric_state(SolarSystem::EMB, tm)?;
            let moon: (Vector3, Vector3) = self.barycentric_state(SolarSystem::Moon, tm)?;
            let b: (Vector3, Vector3) = self.barycentric_state(body, tm)?;

            // Compute the position of the body relative to the Earth-moon
            // barycenter, then "correct" to Earth-center by accounting
            // for moon position and Earth/moon mass ratio
            Ok((
                b.0 - emb.0 + moon.0 / (1.0 + self.emrat),
                b.1 - emb.1 + moon.1 / (1.0 + self.emrat),
            ))
        }
    }
}

pub fn consts(s: &str) -> Option<&f64> {
    jplephem_singleton().as_ref().ok()?.consts(s)
}

/// Return the position of the given body in the Barycentric
/// coordinate system (origin is solarsystem barycenter)
///
/// # Arguments
///  * `body` - the solar system body for which to return position
///  * `tm` - The time at which to return position
///
/// # Returns
///    3-vector of cartesian Heliocentric position in meters
///
///
/// # Notes:
///  * Positions for all bodies are natively relative to solar system barycenter,
///    with exception of moon, which is computed in Geocentric system
///  * EMB (2) is the Earth-Moon barycenter
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
pub fn barycentric_pos<T: TimeLike>(body: SolarSystem, tm: &T) -> Result<Vector3> {
    let tm = tm.as_instant();
    jpl()?.barycentric_pos(body, &tm)
}

/// Return the position and velocity of the given body in
///  Geocentric coordinate system
///
/// # Arguments
///  * `body` - the solar system body for which to return position
///  * `tm` - The time at which to return position
///
/// # Returns
///   * two-element tuple with following elements:
///     * 3-vector of cartesian Geocentric position in meters
///     * 3-vector of cartesian Geocentric velocity in meters / second
///       Note: velocity is relative to Earth
///
pub fn geocentric_state<T: TimeLike>(body: SolarSystem, tm: &T) -> Result<(Vector3, Vector3)> {
    let tm = tm.as_instant();
    jpl()?.geocentric_state(body, &tm)
}

/// Return the position of the given body in
/// Geocentric coordinate system
///
/// # Arguments
///  * `body` - the solar system body for which to return position
///  * `tm` - The time at which to return position
///
/// # Returns
///    3-vector of Cartesian Geocentric position in meters
///
pub fn geocentric_pos<T: TimeLike>(body: SolarSystem, tm: &T) -> Result<Vector3> {
    let tm = tm.as_instant();
    jpl()?.geocentric_pos(body, &tm)
}

/// Return the position & velocity the given body in the barycentric coordinate system
/// (origin is solar system barycenter)
///
/// # Arguments
///  * `body` - the solar system body for which to return position
///  * `tm` - The time at which to return position
///
/// # Returns
///  * two-element tuple with following values:
///    * 3-vector of cartesian Heliocentric position in meters
///    * 3-vector of cartesian Heliocentric velocity in meters / second
///
///
/// # Notes:
///  * Positions for all bodies are natively relative to solar system barycenter,
///    with exception of moon, which is computed in Geocentric system
///  * EMB (2) is the Earth-Moon barycenter
///  * The sun position is relative to the solar system barycenter
///    (it will be close to origin)
pub fn barycentric_state<T: TimeLike>(body: SolarSystem, tm: &T) -> Result<(Vector3, Vector3)> {
    let tm = tm.as_instant();
    jpl()?.barycentric_state(body, &tm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test;
    use std::io::{self, BufRead};

    #[test]
    fn test_parse_rejects_corrupt_files() {
        // Truncated
        assert!(JPLEphem::parse(&[0u8; 100]).is_err());
        // Valid version digits but zeroed jd_start/jd_stop/jd_step: a zero
        // step used to yield usize::MAX records downstream
        let mut raw = vec![0u8; 4096];
        raw[0..84].fill(b' ');
        raw[0..29].copy_from_slice(b"JPL Planetary Ephemeris DE440");
        assert!(matches!(JPLEphem::parse(&raw), Err(Error::InvalidHeader)));
        // A header that is not a JPL ephemeris at all is reported as such.
        assert!(matches!(
            JPLEphem::parse(&vec![b'#'; 4096]),
            Err(Error::NotJplEphemeris(_))
        ));
    }

    #[test]
    fn test_query_at_exact_end_epoch() {
        // Regression: a query exactly at jd_stop passed the range check but
        // floored to a record index one past the last Chebyshev record,
        // panicking on the matrix index instead of returning a result.
        let jpl = jplephem_singleton().as_ref().unwrap();
        let tm = Instant::from_jd_with_scale(jpl.jd_stop, TimeScale::TT);
        // The Instant roundtrip may land a hair above or below jd_stop;
        // either way the call must not panic, and in-range must be Ok.
        if tm.as_jd_with_scale(TimeScale::TT) <= jpl.jd_stop {
            assert!(jpl.geocentric_state(SolarSystem::Moon, &tm).is_ok());
        } else {
            assert!(jpl.geocentric_state(SolarSystem::Moon, &tm).is_err());
        }
        // Just inside the boundary must always succeed
        let tm_in = tm - crate::Duration::from_seconds(1.0);
        assert!(jpl.geocentric_state(SolarSystem::Moon, &tm_in).is_ok());
    }

    /// A file that is not a JPL ephemeris must fail with a clear error, not
    /// a header-parse artefact ("invalid digit found in string").
    #[test]
    fn rejects_non_ephemeris_bytes() {
        let junk = vec![b'#'; 4000];
        match JPLEphem::parse(&junk) {
            Err(Error::NotJplEphemeris(_)) => {}
            other => panic!("expected NotJplEphemeris, got {other:?}"),
        }
        assert!(matches!(
            JPLEphem::parse(b"short"),
            Err(Error::InvalidHeader)
        ));
    }

    #[test]
    fn load_test() {
        //let tm = &Instant::from_date(2010, 3, 1);
        let jpl = jplephem_singleton().as_ref().unwrap();

        let tm = Instant::from_jd_with_scale(2451545.0, TimeScale::TT);
        //let tm = &Instant::from_jd(2451545.0, Scale::UTC);
        let (_, _): (Vector3, Vector3) = jpl.geocentric_state(SolarSystem::Moon, &tm).unwrap();
        println!("au = {:.20}", jpl._au);
    }

    /// Load the test vectors that come with the JPL ephemeris files
    /// and compare calculated positions to test vectors.
    #[test]
    fn testvecs() {
        // Read in test vectors from the NASA JPL web site

        let jpl = jplephem_singleton().as_ref().unwrap();

        let testvecfile = test::get_testvec_dir()
            .unwrap()
            .join("jplephem")
            .join("testpo.440");

        if !testvecfile.is_file() {
            println!(
                "Required JPL ephemeris test vectors file: \"{}\" does not exist
                clone test vectors repo at
                https://github.com/StevenSamirMichael/satkit-testvecs.git
                from root of repo or set \"SATKIT_TESTVEC_ROOT\"
                to point to directory",
                testvecfile.to_string_lossy()
            );
            return;
        }

        let file = std::fs::File::open(testvecfile).unwrap();
        let b = io::BufReader::new(file);

        for rline in b.lines().skip(14) {
            let line = match rline {
                Ok(v) => v,
                Err(_) => continue,
            };
            let s: Vec<&str> = line.split_whitespace().collect();
            assert!(s.len() >= 7);
            let jd: f64 = s[2].parse().unwrap();
            let tar: i32 = s[3].parse().unwrap();
            let src: i32 = s[4].parse().unwrap();
            let coord: usize = s[5].parse().unwrap();
            let truth: f64 = s[6].parse().unwrap();
            let tm = Instant::from_jd_with_scale(jd, TimeScale::TT);
            if tar <= 10 && src <= 10 && coord <= 6 {
                let (mut tpos, mut tvel) = jpl
                    .geocentric_state(SolarSystem::try_from(tar - 1).unwrap(), &tm)
                    .unwrap();
                let (mut spos, mut svel) = jpl
                    .geocentric_state(SolarSystem::try_from(src - 1).unwrap(), &tm)
                    .unwrap();

                // in test vectors, index 3 is not EMB, but rather Earth
                // (this took me a long time to figure out...)
                if tar == 3 {
                    tpos = Vector3::zeros();
                    let (_mpos, mvel): (Vector3, Vector3) = jplephem_singleton()
                        .as_ref()
                        .unwrap()
                        .geocentric_state(SolarSystem::Moon, &tm)
                        .unwrap();
                    // Earth velocity from EMB velocity minus scaled
                    // moon velocity
                    tvel -= mvel / (1.0 + jplephem_singleton().as_ref().unwrap().emrat);
                }
                if src == 3 {
                    spos = Vector3::zeros();
                    let (_mpos, mvel): (Vector3, Vector3) = jplephem_singleton()
                        .as_ref()
                        .unwrap()
                        .geocentric_state(SolarSystem::Moon, &tm)
                        .unwrap();
                    // Earth velocity from EMB velocity minus scaled
                    // moon velocity
                    svel -= mvel / (1.0 + jplephem_singleton().as_ref().unwrap().emrat);
                }

                // Comparing positions
                if coord <= 3 {
                    let calc = (tpos - spos)[coord - 1] / jpl._au / 1.0e3;
                    // These should be very exact
                    // Allow for errors of only ~ 1e-12
                    let maxerr = 1.0e-12;
                    let err = ((truth - calc) / truth).abs();
                    assert!(err < maxerr);
                }
                // Comparing velocities
                else {
                    let calc = (tvel - svel)[coord - 4] / jpl._au / 1.0e3 * 86400.0;
                    let maxerr: f64 = 1.0e-12;
                    let err: f64 = ((truth - calc) / truth).abs();
                    assert!(err < maxerr);
                }
            }
        }
    }
}
