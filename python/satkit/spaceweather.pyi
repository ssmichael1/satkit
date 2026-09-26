"""
Space weather data access

Daily space-weather records (Kp/Ap geomagnetic indices, F10.7 solar flux and
81-day averages) assembled from three primary sources: the GFZ Potsdam
observed record (CC BY 4.0), the NOAA/SWPC 45-day forecast and the NASA MSFC
MSAFE monthly forecast (both public domain). These are the inputs the
NRLMSISE-00 density model consumes when ``use_spaceweather`` is enabled.
"""

from __future__ import annotations

import os

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
            * ``isn`` (int) — international sunspot number; ``-1`` throughout the
              default table (GFZ's sunspot number is CC BY-NC and not ingested)
            * ``cp`` (float) — planetary daily character figure
            * ``c9`` (int) — Cp scaled to [0, 9]
            * ``bsrn`` (int) — Bartels solar rotation number
            * ``nd`` (int) — day within the Bartels rotation
            * ``data_type`` (str) — provenance: ``"OBS"`` measured and
              definitive, ``"OBS-P"`` measured but still preliminary, ``"INT"``
              interpolated, ``"PRD"`` daily prediction, ``"PRM"`` monthly
              prediction, ``""`` unknown

        Fields not yet published for predicted (future) rows are ``-1``.
        MSAFE ``"PRM"`` rows carry a climatological daily Ap in every ``ap``
        slot and ``-1`` for ``kp``. Use :func:`coverage` / :func:`status` to
        check which regime an epoch is in before propagating.

    Raises:
        RuntimeError: If no space-weather record is available for the date
    """
    ...


def update() -> None:
    """Refresh the space-weather files and reload the in-memory table

    The GFZ observed record (every 3 h), the SWPC 45-day forecast (daily)
    and the MSAFE monthly forecast are each re-fetched only when older than
    their publication cadence. Run this (or ``satkit.utils.update_datafiles``)
    periodically for current values.
    """
    ...

def init_from_path(path: str | os.PathLike) -> None:
    """Load the space-weather table from a file, replacing whatever is loaded

    Accepts the GFZ ``Kp_ap_Ap_SN_F107_since_1932.txt`` table (observed only).
    This is how to pin satkit to one fixed input file when a comparison must
    not move with the daily refresh.

    Args:
        path (str | os.PathLike): File to load
    """
    ...

def init_from_bytes(data: bytes) -> None:
    """Load the space-weather table from bytes, replacing whatever is loaded

    Same formats as :func:`init_from_path`.

    Args:
        data (bytes): File contents
    """
    ...

def coverage() -> tuple[_time, _time, _time, _time] | None:
    """Time bounds of the loaded space-weather table

    ``last_daily`` is the boundary that matters for atmospheric drag: past it
    the table holds only monthly MSAFE rows — a 13-month-smoothed F10.7 and a
    climatological daily Ap, with no 3-hourly structure and no storm timing.

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
        NOAA/SWPC 45-day forecast), ``"predicted_monthly"`` (the MSAFE monthly
        forecast: smoothed F10.7 and a climatological Ap, no storm timing),
        ``"extrapolated"`` (past the table; the last row is returned
        unchanged), ``"before_table"``, or ``"not_loaded"``.
    """
    ...

def disable_space_weather_time_warning() -> None:
    """Disable the warnings about out-of-range or missing space-weather data

    Four one-time warnings exist: an epoch past the daily predictions, an
    epoch past the end of the table, no table loaded at all, and an index
    NRLMSISE-00 has to take its default for (an epoch before the table starts,
    or F10.7 before 1947). Each is shown at most once per process; this
    suppresses all of them. They are logged to the ``satkit.spaceweather``
    logger; silencing that logger (or ``satkit``) with :mod:`logging` works
    too.
    """
    ...
