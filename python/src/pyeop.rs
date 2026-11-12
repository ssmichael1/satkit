//! `satkit.eop`: the Earth Orientation Parameters table, under the names
//! `satkit.spaceweather` uses for its table.

use pyo3::prelude::*;

use crate::pyinstant::PyInstant;
use crate::pyutils::{instant_from_pyany, io_error_with_path};
use satkit::earth_orientation_params as eop;

/// `(dut1, xp, yp, lod, dx, dy)`, as Python receives the parameters
pub type EopTuple = (f64, f64, f64, f64, f64, f64);

/// Earth Orientation Parameters at the given time
///
/// Interpolated linearly between the two bracketing daily rows of the IERS
/// ``finals2000A.all`` table. Same values as
/// ``satkit.frametransform.earth_orientation_params``.
///
/// Args:
///     time (satkit.time|datetime.datetime|numpy.datetime64): Time at which to
///         query the parameters (a single time; a list, even of one element,
///         raises TypeError)
///
/// Returns:
///     (float, float, float, float, float, float) | None: tuple with elements
///
///     * 0 : UT1 - UTC, seconds
///     * 1 : X polar motion, arcsec
///     * 2 : Y polar motion, arcsec
///     * 3 : LOD: excess length of day, -d(UT1-UTC)/dt, seconds per day
///     * 4 : dX wrt IAU-2000A nutation, milliarcsec
///     * 5 : dY wrt IAU-2000A nutation, milliarcsec
///
///     or None before the table's first row (1973-01-02) or when no table is
///     loaded. After the last row the last row's values are returned
///     unchanged; :func:`status` tells these cases apart.
///
/// Raises:
///     TypeError: If ``time`` is not a single time
///
/// Example:
///     >>> dut1, xp, yp, lod, dx, dy = satkit.eop.get(satkit.time(2024, 1, 1))
#[pyfunction]
pub fn get(time: &Bound<'_, PyAny>) -> PyResult<Option<EopTuple>> {
    let t = instant_from_pyany(time)?;
    Ok(eop::get(&t).map(|r| (r[0], r[1], r[2], r[3], r[4], r[5])))
}

/// Time bounds of the loaded Earth Orientation Parameters table
///
/// Returns:
///     (satkit.time, satkit.time, satkit.time) | None: ``(first, last_observed, last)``:
///     the first row, the last *observed* row (rows after it are IERS
///     predictions) and the last row; None if no table is loaded. Epochs
///     after ``last`` use that row's values held constant.
///
/// Example:
///     >>> first, last_obs, last = satkit.eop.coverage()
#[pyfunction]
pub fn coverage() -> Option<(PyInstant, PyInstant, PyInstant)> {
    eop::coverage().map(|c| {
        (
            PyInstant(c.first),
            PyInstant(c.last_observed),
            PyInstant(c.last),
        )
    })
}

/// Where a time falls relative to the loaded Earth Orientation Parameters table
///
/// Args:
///     time (satkit.time|datetime.datetime|numpy.datetime64): Time to classify (a
///         single time; a list, even of one element, raises TypeError)
///
/// Returns:
///     str: One of ``"observed"`` (on or before the last observed row),
///     ``"predicted"`` (inside the IERS predictions), ``"extrapolated"``
///     (after the table end: the last row is held constant, and accuracy
///     degrades by ~0.1 arcsec / ~10 ms over a few months), ``"before_table"``
///     (before the first row, 1973-01-02 for ``finals2000A.all``: zeros are
///     used, UT1 = UTC) or ``"not_loaded"`` (no table: zeros are used).
///
/// Raises:
///     TypeError: If ``time`` is not a single time
///
/// Example:
///     >>> satkit.eop.status(satkit.time(2024, 1, 1))
///     'observed'
#[pyfunction]
pub fn status(time: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let t = instant_from_pyany(time)?;
    Ok(status_str(eop::status(&t)))
}

/// The Python spelling of an [`eop::EopStatus`]
pub fn status_str(s: eop::EopStatus) -> &'static str {
    use eop::EopStatus::*;
    match s {
        Observed => "observed",
        Predicted => "predicted",
        Extrapolated => "extrapolated",
        BeforeTable => "before_table",
        NotLoaded => "not_loaded",
    }
}

/// Refresh the Earth Orientation Parameters file and reload the table
///
/// Downloads IERS ``finals2000A.all`` into the data directory unless the
/// copy there was fetched within the last 24 hours, then loads it. Run this
/// (or ``satkit.utils.update_datafiles``) periodically for current values.
///
/// Raises:
///     RuntimeError: If the data directory is not writable, the download
///         fails, or the downloaded file cannot be read
#[pyfunction]
pub fn update() -> anyhow::Result<()> {
    Ok(eop::update()?)
}

/// Load the Earth Orientation Parameters table from a file, replacing whatever is loaded
///
/// The file must be an IERS ``finals2000A.all`` (Bulletin A) file. From the
/// next call on, everything that reads EOP (the ITRF frame transforms, UT1,
/// the propagators) uses this table. This is how to pin satkit to one fixed
/// input file when a comparison must not move with the daily refresh. The
/// one-time warnings are not re-armed for the new table.
///
/// Args:
///     path (str | os.PathLike): File to load
///
/// Raises:
///     FileNotFoundError: If the file does not exist (other read failures
///         raise the matching ``OSError`` subclass)
///     RuntimeError: If the file is not a valid ``finals2000A.all`` file
///         (the loaded table is then unchanged)
#[pyfunction]
pub fn init_from_path(path: std::path::PathBuf) -> PyResult<()> {
    match eop::init_from_path(&path) {
        Ok(()) => Ok(()),
        Err(eop::Error::Io(e)) => Err(io_error_with_path(&e, &path)),
        Err(e) => Err(anyhow::Error::from(e).into()),
    }
}

/// Load the Earth Orientation Parameters table from bytes, replacing whatever is loaded
///
/// Same format and effect as :func:`init_from_path`.
///
/// Args:
///     data (bytes): File contents
///
/// Raises:
///     RuntimeError: If the bytes are not a valid ``finals2000A.all`` file
///         (the loaded table is then unchanged)
#[pyfunction]
pub fn init_from_bytes(data: &[u8]) -> anyhow::Result<()> {
    Ok(eop::init_from_bytes(data)?)
}
