use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

use crate::pyutils::instant_from_pyany;
use satkit::spaceweather;

/// Space-weather record for the given time
///
/// Returns the daily space-weather record closest to and not after the
/// given time, from the GFZ Potsdam observed record, then the NOAA/SWPC
/// 45-day forecast, then the NASA MSAFE monthly forecast. These are
/// the values the NRLMSISE-00 density model consumes when
/// `use_spaceweather` is enabled — exposing them lets you inspect or
/// reproduce the propagator's density inputs.
///
/// Args:
///     time (satkit.time|datetime.datetime): Time for which to return the record
///
/// Returns:
///     dict: Space-weather record with the following keys:
///
///         * ``date`` (satkit.time) — date of the record
///         * ``kp`` (list[int]) — eight 3-hourly Kp indices (x10)
///         * ``kp_sum`` (int) — daily Kp sum
///         * ``ap`` (list[int]) — eight 3-hourly Ap indices
///         * ``ap_avg`` (int) — daily average Ap
///         * ``f10p7_obs`` (float) — observed F10.7 solar flux, sfu (10^-22 W m^-2 Hz^-1)
///         * ``f10p7_adj`` (float) — F10.7 adjusted to 1 AU, sfu
///         * ``f10p7_obs_c81`` / ``f10p7_obs_l81`` (float) — 81-day centered / last-81-day observed averages, sfu
///         * ``f10p7_adj_c81`` / ``f10p7_adj_l81`` (float) — 81-day centered / last-81-day adjusted averages, sfu
///         * ``isn`` (int) — international sunspot number
///         * ``cp`` (float) — planetary daily character figure
///         * ``c9`` (int) — Cp scaled to [0, 9]
///         * ``bsrn`` (int) — Bartels solar rotation number
///         * ``nd`` (int) — day within the Bartels rotation
///         * ``data_type`` (str) — provenance of the row: ``"OBS"`` measured,
///           ``"INT"`` interpolated, ``"PRD"`` daily prediction, ``"PRM"``
///           monthly prediction, ``""`` unknown
///
///     Note: fields not yet published for predicted (future) rows are ``-1``.
///     MSAFE ``"PRM"`` rows carry a climatological daily Ap in every ``ap``
///     slot and ``-1`` for ``kp``. Use :func:`coverage` / :func:`status` to
///     find out which regime an epoch is in before propagating.
///
/// Raises:
///     RuntimeError: If no space-weather record is available for the date
#[pyfunction]
pub fn get(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    let tm = instant_from_pyany(time)?;
    let rec = spaceweather::get(&tm)?;
    pyo3::Python::attach(|py| -> anyhow::Result<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("date", crate::pyinstant::PyInstant(rec.date))?;
        d.set_item("kp", rec.kp.to_vec())?;
        d.set_item("kp_sum", rec.kp_sum)?;
        d.set_item("ap", rec.ap.to_vec())?;
        d.set_item("ap_avg", rec.ap_avg)?;
        d.set_item("f10p7_obs", rec.f10p7_obs)?;
        d.set_item("f10p7_adj", rec.f10p7_adj)?;
        d.set_item("f10p7_obs_c81", rec.f10p7_obs_c81)?;
        d.set_item("f10p7_obs_l81", rec.f10p7_obs_l81)?;
        d.set_item("f10p7_adj_c81", rec.f10p7_adj_c81)?;
        d.set_item("f10p7_adj_l81", rec.f10p7_adj_l81)?;
        d.set_item("isn", rec.isn)?;
        d.set_item("cp", rec.cp)?;
        d.set_item("c9", rec.c9)?;
        d.set_item("bsrn", rec.bsrn)?;
        d.set_item("nd", rec.nd)?;
        d.set_item("data_type", rec.data_type.as_str())?;
        Ok(d.into_py_any(py)?)
    })
}

/// Refresh the space-weather files and reload the in-memory table
///
/// The GFZ observed record (every 3 h), the SWPC 45-day forecast (daily)
/// and the MSAFE monthly forecast are each re-fetched only when older than
/// their publication cadence. Run this (or `satkit.utils.update_datafiles`)
/// periodically for current values.
#[pyfunction]
pub fn update() -> anyhow::Result<()> {
    Ok(spaceweather::update()?)
}

/// Time bounds of the loaded space-weather table
///
/// ``last_daily`` is the boundary that matters for atmospheric drag: past it
/// the table holds only monthly MSAFE rows — a 13-month-smoothed F10.7 and a
/// climatological daily Ap, with no 3-hourly structure and no storm timing.
///
/// Returns:
///     (satkit.time, satkit.time, satkit.time, satkit.time) | None:
///     ``(first, last_observed, last_daily, last)``, or None if no
///     space-weather table is loaded.
///
/// Example:
///     >>> first, last_obs, last_daily, last = satkit.spaceweather.coverage()
#[pyfunction]
pub fn coverage() -> Option<(
    crate::pyinstant::PyInstant,
    crate::pyinstant::PyInstant,
    crate::pyinstant::PyInstant,
    crate::pyinstant::PyInstant,
)> {
    spaceweather::coverage().map(|c| {
        (
            crate::pyinstant::PyInstant(c.first),
            crate::pyinstant::PyInstant(c.last_observed),
            crate::pyinstant::PyInstant(c.last_daily),
            crate::pyinstant::PyInstant(c.last),
        )
    })
}

/// Where a time falls relative to the loaded space-weather table
///
/// Args:
///     time (satkit.time|datetime.datetime): Time to classify
///
/// Returns:
///     str: One of ``"observed"`` (measured), ``"predicted_daily"`` (inside
///     the NOAA/SWPC 45-day forecast), ``"predicted_monthly"`` (the MSAFE monthly
///     forecast: smoothed F10.7 and a climatological Ap, no storm timing),
///     ``"extrapolated"`` (past the table; the last row is returned
///     unchanged), ``"before_table"``, or ``"not_loaded"``.
#[pyfunction]
pub fn status(time: &Bound<'_, PyAny>) -> anyhow::Result<&'static str> {
    use satkit::spaceweather::SpaceWeatherStatus::*;
    let tm = instant_from_pyany(time)?;
    Ok(match spaceweather::status(&tm) {
        Observed => "observed",
        PredictedDaily => "predicted_daily",
        PredictedMonthly => "predicted_monthly",
        Extrapolated => "extrapolated",
        BeforeTable => "before_table",
        NotLoaded => "not_loaded",
    })
}

/// Disable the warnings about out-of-range or missing space-weather data
///
/// Three one-time warnings exist: an epoch past the daily predictions (only
/// monthly F10.7, no geomagnetic data), an epoch past the end of the table,
/// and no table loaded at all. Each is shown at most once per process; this
/// suppresses all of them.
///
/// Example:
///     >>> satkit.spaceweather.disable_space_weather_time_warning()
#[pyfunction]
pub fn disable_space_weather_time_warning() {
    spaceweather::disable_space_weather_time_warning();
}

/// Load the space-weather table from a file, replacing whatever is loaded
///
/// Accepts the GFZ ``Kp_ap_Ap_SN_F107_since_1932.txt`` table (observed
/// only). This is how to pin satkit to one fixed input file when a
/// comparison must not move with the daily refresh.
///
/// Args:
///     path (str | os.PathLike): File to load
#[pyfunction]
pub fn init_from_path(path: std::path::PathBuf) -> anyhow::Result<()> {
    Ok(spaceweather::init_from_path(&path)?)
}

/// Load the space-weather table from bytes, replacing whatever is loaded
///
/// Same formats as :func:`init_from_path`.
///
/// Args:
///     data (bytes): File contents
#[pyfunction]
pub fn init_from_bytes(data: &[u8]) -> anyhow::Result<()> {
    Ok(spaceweather::init_from_bytes(data)?)
}
