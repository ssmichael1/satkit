use crate::utils::{self, download_if_not_exist};

use super::{Error, Result};

use crate::mathtypes::*;

use std::sync::OnceLock;

#[derive(Debug)]
pub struct IERSTable {
    data: [DMatrix<f64>; 6],
}

/// Identifier for the three IERS-2010 IAU precession-nutation tables that
/// satkit holds as singletons. Each maps to a `tab5.2X.txt` file in the
/// data directory (see IERS Technical Note 36 §5):
///
/// * [`Tab5A`](Self::Tab5A) — CIP X-coordinate series
/// * [`Tab5B`](Self::Tab5B) — CIP Y-coordinate series
/// * [`Tab5D`](Self::Tab5D) — CIO locator s series
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IersTableId {
    Tab5A,
    Tab5B,
    Tab5D,
}

impl IersTableId {
    /// Default filename under [`datadir`](crate::utils::datadir) used by
    /// the lazy default-resolver.
    pub const fn default_filename(self) -> &'static str {
        match self {
            Self::Tab5A => "tab5.2a.txt",
            Self::Tab5B => "tab5.2b.txt",
            Self::Tab5D => "tab5.2d.txt",
        }
    }
}

static TAB5A_INSTANCE: OnceLock<IERSTable> = OnceLock::new();
static TAB5B_INSTANCE: OnceLock<IERSTable> = OnceLock::new();
static TAB5D_INSTANCE: OnceLock<IERSTable> = OnceLock::new();

fn instance_for(id: IersTableId) -> &'static OnceLock<IERSTable> {
    match id {
        IersTableId::Tab5A => &TAB5A_INSTANCE,
        IersTableId::Tab5B => &TAB5B_INSTANCE,
        IersTableId::Tab5D => &TAB5D_INSTANCE,
    }
}

/// Return the IERS table singleton for `id`, loading from
/// [`datadir`](crate::utils::datadir) on first access.
pub fn table(id: IersTableId) -> &'static IERSTable {
    // This backs the per-transform IERS reduction and cannot return a `Result`
    // without threading it through every `q*2*` frame-transform signature, so a
    // missing/corrupt table file — a setup error — panics with an actionable
    // message rather than an opaque `unwrap`.
    instance_for(id).get_or_init(|| {
        let fname = id.default_filename();
        IERSTable::from_file(fname).unwrap_or_else(|e| {
            panic!(
                "Failed to load IERS table \"{fname}\": {e}. Ensure the data \
                 files are present (set the SATKIT_DATA environment variable to \
                 your data directory, or run satkit::utils::update_datafiles to \
                 download them)."
            )
        })
    })
}

/// Load all three IERS precession-nutation tables now, returning an error
/// (instead of the panic the lazy [`table`] accessor raises) if a file is
/// missing or unreadable. Idempotent: tables that are already initialized
/// are left untouched. The Python bindings and
/// [`Precomputed`](crate::orbitprop::Precomputed) call this so a missing
/// data file surfaces as a normal error before any transform runs.
pub fn preload() -> Result<()> {
    for id in [IersTableId::Tab5A, IersTableId::Tab5B, IersTableId::Tab5D] {
        let cell = instance_for(id);
        if cell.get().is_none() {
            let parsed = IERSTable::from_file(id.default_filename())?;
            // A concurrent lazy init may have won the race; either way the
            // table is now loaded, so an `Err` from `set` is not a failure.
            let _ = cell.set(parsed);
        }
    }
    Ok(())
}

/// Initialize the IERS table singleton for `id` from an in-memory byte
/// buffer.
///
/// The bytes must be a valid `tab5.2X.txt` text file (UTF-8). Must be
/// called *before* any frame transform that depends on this table,
/// otherwise the lazy default-resolver init has already won and this
/// returns [`Error::IersTableAlreadyInitialized`].
pub fn init_from_bytes(id: IersTableId, bytes: &[u8]) -> Result<()> {
    let parsed = IERSTable::from_bytes(bytes)?;
    instance_for(id)
        .set(parsed)
        .map_err(|_| Error::IersTableAlreadyInitialized { id })
}

/// Initialize the IERS table singleton for `id` from a file at `path`.
///
/// Same semantics as [`init_from_bytes`]; see that function for details.
pub fn init_from_path(id: IersTableId, path: &std::path::Path) -> Result<()> {
    let parsed = IERSTable::from_path(path)?;
    instance_for(id)
        .set(parsed)
        .map_err(|_| Error::IersTableAlreadyInitialized { id })
}

/// IERS Table
///
/// This struct is used to store the IERS tables used in the IERS 2010 conventions.
/// See the IERS Conventions 2010 document for more information.
///
/// Should not be used directly, but through the `FrameTransform` struct.
///
impl IERSTable {
    /// Load an IERS table from a file under
    /// [`datadir`](crate::utils::datadir) by basename. Auto-downloads via
    /// [`download_if_not_exist`] if missing.
    pub fn from_file(fname: &str) -> Result<Self> {
        // Precedence: a copy in the data directory wins (so an updated table
        // can be dropped in without rebuilding); otherwise the compiled-in
        // copy; a download is only attempted for a name that is not embedded.
        if let Some(path) = utils::find_data_file(fname) {
            return Self::from_path(&path);
        }
        if let Some(bytes) = utils::embedded::get(fname) {
            return Self::from_bytes(&bytes);
        }
        let path = utils::datadir()?.join(fname);
        download_if_not_exist(&path, None)?;
        Self::from_path(&path)
    }

    /// Load an IERS table from a file at `path`. No download is attempted.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Load an IERS table from an in-memory byte buffer. The buffer must
    /// be a valid `tab5.2X.txt` text file (UTF-8).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(std::str::from_utf8(bytes)?)
    }

    /// Parse an IERS table from a `tab5.2X.txt` text string.
    pub fn parse(text: &str) -> Result<Self> {
        let mut table = Self {
            data: [
                DMatrix::<f64>::zeros(0, 0),
                DMatrix::<f64>::zeros(0, 0),
                DMatrix::<f64>::zeros(0, 0),
                DMatrix::<f64>::zeros(0, 0),
                DMatrix::<f64>::zeros(0, 0),
                DMatrix::<f64>::zeros(0, 0),
            ],
        };

        let mut tnum: i32 = -1;
        let mut rowcnt: usize = 0;

        for line in text.lines() {
            let tline = line.trim();
            if tline.len() < 10 {
                continue;
            }
            if tline.starts_with("j =") {
                // Expected form: "j = <tnum> ... <tsize>"; read tokens rather
                // than byte-slicing so a non-ASCII/corrupt line can't panic.
                let s: Vec<&str> = tline.split_whitespace().collect();
                if s.len() < 3 {
                    return Err(Error::InvalidIersTableDef {
                        fname: String::from("<buffer>"),
                    });
                }
                tnum = s[2].parse()?;
                let tsize: usize = s[s.len() - 1].parse().unwrap_or(0);
                if !(0..=5).contains(&tnum) || tsize == 0 {
                    return Err(Error::InvalidIersTableDef {
                        fname: String::from("<buffer>"),
                    });
                }
                table.data[tnum as usize] = DMatrix::<f64>::zeros(tsize, 17);
                rowcnt = 0;
                continue;
            } else if tnum >= 0 {
                if table.data[tnum as usize].ncols() < 17 {
                    return Err(Error::IersTableNotInitialized {
                        fname: String::from("<buffer>"),
                    });
                }
                // Propagate a bad numeric token instead of panicking.
                let vals: Vec<f64> = tline
                    .split_whitespace()
                    .map(|x| x.parse())
                    .collect::<std::result::Result<Vec<f64>, _>>()?;
                // Guard against a file with more rows or columns than the
                // declared table dimensions.
                if rowcnt >= table.data[tnum as usize].nrows() {
                    return Err(Error::InvalidIersTableDef {
                        fname: String::from("<buffer>"),
                    });
                }
                for (c, &val) in vals.iter().enumerate().take(17) {
                    table.data[tnum as usize][(rowcnt, c)] = val;
                }
                rowcnt += 1;
            }
        }
        Ok(table)
    }

    pub fn compute(&self, t_tt: f64, delaunay: &numeris::Vector<f64, 14>) -> f64 {
        let mut retval: f64 = 0.0;
        for i in 0..6 {
            // return if finished
            if self.data[i].ncols() == 0 {
                continue;
            }

            let mut tmult: f64 = 1.0;
            for _ in 0..i {
                tmult *= t_tt;
            }

            for j in 0..self.data[i].nrows() {
                //double argVal = 0;
                let mut argval: f64 = 0.0;
                for k in 0..13 {
                    argval += self.data[i][(j, k + 3)] * delaunay[k];
                }
                let sval = f64::sin(argval);
                let cval = f64::cos(argval);
                retval += tmult * self.data[i][(j, 1)].mul_add(sval, self.data[i][(j, 2)] * cval);
            }
        }
        retval
    }
}

#[cfg(test)]
mod tests {
    use super::IERSTable;
    use anyhow::Result;

    #[test]
    fn test_parse_bad_token_errors_not_panics() {
        // A non-numeric token in a data row must return an error rather than
        // panicking (the row parse previously used `.unwrap()`).
        let text = "j = 0  amp  1\n\
                    1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 NOTNUM\n";
        assert!(IERSTable::parse(text).is_err());
    }

    #[test]
    fn load_table() -> Result<()> {
        let t = IERSTable::from_file("tab5.2a.txt");
        if t.is_err() {
            anyhow::bail!("Could not load IERS table");
        }
        if t.unwrap().data[0].ncols() < 17 {
            anyhow::bail!("Error loading table");
        }
        Ok(())
    }
}
