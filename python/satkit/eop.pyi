"""
Earth Orientation Parameters (EOP)

The IERS ``finals2000A.all`` table (Bulletin A: observed values, then about a
year of predictions) behind UT1 and the ITRF frame transforms: UT1 - UTC,
polar motion, excess length of day and the celestial pole offsets. The table
is loaded from the data directory on first use and refreshed by ``update`` or
``satkit.utils.update_datafiles``.

These functions mirror ``satkit.spaceweather``. The EOP functions in
``satkit.frametransform`` (``earth_orientation_params``, ``eop_coverage``,
``eop_status``, ``disable_eop_time_warning``) remain.
"""

from __future__ import annotations

import os
from typing import Literal

from .satkit import time as _time, TimeScalar

def get(time: TimeScalar) -> tuple[float, float, float, float, float, float] | None:
    """Earth Orientation Parameters at the given time

    Interpolated linearly between the two bracketing daily rows of the IERS
    ``finals2000A.all`` table. Same values as
    ``satkit.frametransform.earth_orientation_params``.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to
            query the parameters (a single time; a list, even of one element,
            raises ``TypeError``)

    Returns:
        (float, float, float, float, float, float) | None: tuple with elements

            * 0 : UT1 - UTC, seconds
            * 1 : X polar motion, arcsec
            * 2 : Y polar motion, arcsec
            * 3 : LOD: excess length of day, -d(UT1-UTC)/dt, seconds per day
            * 4 : dX wrt IAU-2000A nutation, milliarcsec
            * 5 : dY wrt IAU-2000A nutation, milliarcsec

        or None before the table's first row (1973-01-02) or when no table is
        loaded. After the last row the last row's values are returned
        unchanged; ``status`` tells these cases apart.

    Raises:
        TypeError: If ``time`` is not a single time

    Example:
        >>> dut1, xp, yp, lod, dx, dy = satkit.eop.get(satkit.time(2024, 1, 1))
    """
    ...

def coverage() -> tuple[_time, _time, _time] | None:
    """Time bounds of the loaded Earth Orientation Parameters table

    Returns:
        tuple[satkit.time, satkit.time, satkit.time] | None: ``(first, last_observed, last)``:
        the first row, the last *observed* row (rows after it are IERS
        predictions) and the last row; None if no table is loaded. Epochs
        after ``last`` use that row's values held constant.

    Example:
        >>> first, last_obs, last = satkit.eop.coverage()
    """
    ...

def status(
    time: TimeScalar,
) -> Literal["observed", "predicted", "extrapolated", "before_table", "not_loaded"]:
    """Where a time falls relative to the loaded Earth Orientation Parameters table

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time to classify (a
            single time; a list, even of one element, raises ``TypeError``)

    Returns:
        str: One of ``"observed"`` (on or before the last observed row),
        ``"predicted"`` (inside the IERS predictions), ``"extrapolated"``
        (after the table end: the last row is held constant, and accuracy
        degrades by ~0.1 arcsec / ~10 ms over a few months), ``"before_table"``
        (before the first row, 1973-01-02 for ``finals2000A.all``: zeros are
        used, UT1 = UTC) or ``"not_loaded"`` (no table: zeros are used).

    Raises:
        TypeError: If ``time`` is not a single time

    Example:
        >>> satkit.eop.status(satkit.time(2024, 1, 1))
        'observed'
    """
    ...

def update() -> None:
    """Refresh the Earth Orientation Parameters file and reload the table

    Downloads IERS ``finals2000A.all`` into the data directory unless the
    copy there was fetched within the last 24 hours, then loads it. Run this
    (or ``satkit.utils.update_datafiles``) periodically for current values.

    Raises:
        RuntimeError: If the data directory is not writable, the download
            fails, or the downloaded file cannot be read
    """
    ...

def init_from_path(path: str | os.PathLike[str]) -> None:
    """Load the Earth Orientation Parameters table from a file, replacing whatever is loaded

    The file must be an IERS ``finals2000A.all`` (Bulletin A) file. From the
    next call on, everything that reads EOP (the ITRF frame transforms, UT1,
    the propagators) uses this table. This is how to pin satkit to one fixed
    input file when a comparison must not move with the daily refresh. The
    one-time warnings are not re-armed for the new table.

    Args:
        path (str | os.PathLike): File to load

    Raises:
        FileNotFoundError: If the file does not exist (other read failures
            raise the matching ``OSError`` subclass)
        RuntimeError: If the file is not a valid ``finals2000A.all`` file
            (the loaded table is then unchanged)
    """
    ...

def init_from_bytes(data: bytes) -> None:
    """Load the Earth Orientation Parameters table from bytes, replacing whatever is loaded

    Same format and effect as ``init_from_path``.

    Args:
        data (bytes): File contents

    Raises:
        RuntimeError: If the bytes are not a valid ``finals2000A.all`` file
            (the loaded table is then unchanged)
    """
    ...
