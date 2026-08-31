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
    // without threading it through every `q*2*` frame-transform signature.
    // With the tables compiled in and a corrupt on-disk copy falling back to
    // them (`from_path_or_embedded`), this panic is unreachable short of a
    // build defect in the embedded blobs; the message stays actionable anyway.
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
            return Self::from_path_or_embedded(&path, fname);
        }
        if let Some(bytes) = utils::embedded::get(fname) {
            return Self::from_bytes(&bytes);
        }
        let path = utils::datadir()?.join(fname);
        download_if_not_exist(&path, None)?;
        Self::from_path(&path)
    }

    /// Load the table at `path`; if it is unreadable or does not parse and a
    /// compiled-in copy of `fname` exists, warn and use that instead.
    ///
    /// A corrupt, truncated or substituted file in a search directory is a
    /// setup problem, not a reason to lose the frame chain: the compiled-in
    /// table is the same IERS 2010 series, so falling back to it is exact,
    /// unlike degrading to an approximate transform. The warning (suppressed
    /// by `SATKIT_QUIET=1`) says which file to fix. With this fallback the
    /// only way [`table`] can still panic is a compiled-in table that fails
    /// to inflate or parse, which is a build defect, not a runtime condition.
    fn from_path_or_embedded(path: &std::path::Path, fname: &str) -> Result<Self> {
        let err = match Self::from_path(path) {
            Ok(table) => return Ok(table),
            Err(e) => e,
        };
        let Some(bytes) = utils::embedded::get(fname) else {
            return Err(err);
        };
        if std::env::var_os("SATKIT_QUIET").is_none() {
            eprintln!(
                "Warning: IERS table {} could not be loaded ({err}); using the compiled-in \
                 copy of {fname} instead. Delete or replace the file to silence this.",
                path.display()
            );
        }
        Self::from_bytes(&bytes)
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
        // A table whose header promised more rows than the text delivered is
        // a truncated file; the missing rows would otherwise stay zero and
        // silently shift every transform that uses the series.
        let truncated = |table: &Self, tnum: i32, rowcnt: usize| -> Result<()> {
            if tnum >= 0 && rowcnt != table.data[tnum as usize].nrows() {
                return Err(Error::InvalidIersTableDef {
                    fname: String::from("<buffer>"),
                });
            }
            Ok(())
        };

        for line in text.lines() {
            let tline = line.trim();
            if tline.len() < 10 {
                continue;
            }
            if tline.starts_with("j =") {
                truncated(&table, tnum, rowcnt)?;
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
        truncated(&table, tnum, rowcnt)?;
        // No `j =` header at all means this was never an IERS table (a proxy
        // notice page, an empty file): six empty series would otherwise load
        // fine and make every transform silently skip the nutation terms.
        if tnum < 0 {
            return Err(Error::InvalidIersTableDef {
                fname: String::from("<buffer>"),
            });
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

    /// One well-formed series: header declaring one row, then that row.
    const MINIMAL_TABLE: &str = "j = 0  Number of terms = 1\n\
        1  -6844318.44  1328.67  0 0 0 0 1 0 0 0 0 0 0 0 0\n";

    #[test]
    fn parse_accepts_a_minimal_table() {
        let t = IERSTable::parse(MINIMAL_TABLE).unwrap();
        assert_eq!(t.data[0].nrows(), 1);
        assert_eq!(t.data[0].ncols(), 17);
    }

    #[test]
    fn parse_rejects_text_with_no_table_header() {
        // Six empty series used to load "successfully" and make every
        // transform skip the nutation terms without a word.
        for text in [
            "",
            "<!DOCTYPE html>\n<html><body>Review Usage Policy</body></html>\n",
            "some unrelated text file with long enough lines to be considered\n",
        ] {
            assert!(IERSTable::parse(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn parse_rejects_a_truncated_table() {
        // Header promises two rows, the file delivers one: a cut-off transfer.
        let text = "j = 0  Number of terms = 2\n\
            1  -6844318.44  1328.67  0 0 0 0 1 0 0 0 0 0 0 0 0\n";
        assert!(IERSTable::parse(text).is_err());
        // Same, with the truncation before a second series starts.
        let text =
            format!("{text}j = 1  Number of terms = 1\n1 1.0 1.0 0 0 0 0 1 0 0 0 0 0 0 0 0\n");
        assert!(IERSTable::parse(&text).is_err());
    }

    #[test]
    fn corrupt_file_in_data_dir_falls_back_to_the_embedded_table() {
        let dir = std::env::temp_dir().join(format!("satkit_iers_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("tab5.2a.txt");
        std::fs::write(&bad, "<!DOCTYPE html>\n<html><body>blocked</body></html>\n").unwrap();

        // A known table name: the compiled-in copy takes over, fully populated.
        let t = IERSTable::from_path_or_embedded(&bad, "tab5.2a.txt").unwrap();
        assert!(
            t.data[0].nrows() > 1000,
            "embedded tab5.2a series is ~1300 rows"
        );
        // A name with no compiled-in copy: the original error is reported.
        assert!(IERSTable::from_path_or_embedded(&bad, "no-such-table.txt").is_err());
        // A missing file (not just a corrupt one) also falls back.
        let missing = dir.join("tab5.2b.txt");
        assert!(IERSTable::from_path_or_embedded(&missing, "tab5.2b.txt").is_ok());

        let _ = std::fs::remove_dir_all(&dir);
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
