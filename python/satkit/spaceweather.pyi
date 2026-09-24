"""
Space weather data access

Daily space-weather records (Kp/Ap geomagnetic indices, F10.7 solar flux and
81-day averages) from CelesTrak's ``SW-All.csv``, plus the NOAA/SWPC
solar-cycle forecast used for future-epoch predictions. These are the inputs
the NRLMSISE-00 density model consumes when ``use_spaceweather`` is enabled.
"""

from __future__ import annotations

from .satkit import time as _time, TimeScalar

def get(time: TimeScalar) -> dict:
    """Space-weather record for the given time

    Returns the daily record closest to and not after the given time.

    Args:
        time (satkit.time | datetime.datetime): Time for which to return the record

    Returns:
        dict: Space-weather record with keys:

            * ``date`` (satkit.time) — date of the record
            * ``kp`` (list[int]) — eight 3-hourly Kp indices (x10)
            * ``kp_sum`` (int) — daily Kp sum
            * ``ap`` (list[int]) — eight 3-hourly Ap indices
            * ``ap_avg`` (int) — daily average Ap
            * ``f10p7_obs`` (float) — observed F10.7 solar flux, sfu (10^-22 W m^-2 Hz^-1)
            * ``f10p7_adj`` (float) — F10.7 adjusted to 1 AU, sfu
            * ``f10p7_obs_c81`` / ``f10p7_obs_l81`` (float) — 81-day centered / last-81-day observed averages, sfu
            * ``f10p7_adj_c81`` / ``f10p7_adj_l81`` (float) — 81-day centered / last-81-day adjusted averages, sfu
            * ``isn`` (int) — international sunspot number
            * ``cp`` (float) — planetary daily character figure
            * ``c9`` (int) — Cp scaled to [0, 9]
            * ``bsrn`` (int) — Bartels solar rotation number
            * ``nd`` (int) — day within the Bartels rotation
            * ``data_type`` (str) — provenance: ``"OBS"`` measured, ``"INT"``
              interpolated, ``"PRD"`` daily prediction, ``"PRM"`` monthly
              prediction, ``""`` unknown

        Fields not yet published for predicted (future) rows are ``-1``.
        ``"PRM"`` rows carry F10.7 but **no** geomagnetic data at all — every
        ``kp``/``ap`` entry is ``-1`` — so NRLMSISE-00 falls back to a
        quiet-time ``Ap = 4``. Use :func:`coverage` / :func:`status` to check
        before propagating.

    Raises:
        RuntimeError: If no space-weather record is available for the date
    """
    ...

def predicted_f107(time: TimeScalar) -> float | None:
    """Predicted F10.7 solar flux for a (future) time

    Linearly interpolates the NOAA/SWPC monthly solar-cycle forecast — the
    value the NRLMSISE-00 density model falls back to for future dates.

    Args:
        time (satkit.time | datetime.datetime): Time for which to return predicted F10.7

    Returns:
        float | None: Predicted F10.7 solar flux in sfu (10^-22 W m^-2 Hz^-1),
        or None if the time is outside the forecast range
    """
    ...

def update() -> None:
    """Download the latest space-weather file and reload the in-memory data

    Space weather is updated daily; run this (or
    ``satkit.utils.update_datafiles``) periodically for current values.
    """
    ...

def coverage() -> tuple[_time, _time, _time, _time] | None:
    """Time bounds of the loaded space-weather table

    ``last_daily`` is the boundary that matters for atmospheric drag: past it
    the table holds only monthly rows, which carry F10.7 but no Kp/ap, so
    NRLMSISE-00 runs on a quiet-time ``Ap = 4`` with no storm information.

    Returns:
        tuple[satkit.time, satkit.time, satkit.time, satkit.time] | None:
        ``(first, last_observed, last_daily, last)``, or None if no
        space-weather table is loaded.

    Example:
        >>> first, last_obs, last_daily, last = satkit.spaceweather.coverage()
    """
    ...

def status(time: TimeScalar) -> str:
    """Where a time falls relative to the loaded space-weather table

    Args:
        time (satkit.time | datetime.datetime): Time to classify

    Returns:
        str: One of ``"observed"``, ``"predicted_daily"`` (inside the
        NOAA/SWPC 45-day forecast), ``"predicted_monthly"`` (only monthly
        F10.7 — **no geomagnetic data**, NRLMSISE-00 uses ``Ap = 4``),
        ``"extrapolated"`` (past the table; the last row is returned
        unchanged), ``"before_table"``, or ``"not_loaded"``.
    """
    ...

def disable_space_weather_time_warning() -> None:
    """Disable the warnings about out-of-range or missing space-weather data

    Three one-time warnings exist: an epoch past the daily predictions, an
    epoch past the end of the table, and no table loaded at all. Each is shown
    at most once per process; this suppresses all of them.
    """
    ...
