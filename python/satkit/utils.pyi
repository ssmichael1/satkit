"""
Utility functions for SatKit
"""

from __future__ import annotations

import os


def update_datafiles(
    *, overwrite: bool = False, dir: str | os.PathLike[str] | None = None
) -> None:
    """Download & store data files needed for "satkit" computations

    Not required for normal use: the IERS nutation tables and gravity models
    are compiled into satkit, the JPL ephemeris is downloaded on first use,
    and the Earth-orientation / space-weather files are fetched on first use.
    Call this to provision everything up front (a container image, a machine
    that will later be offline) or to refresh the daily files.

    Keyword Args:

      overwrite (bool):  Re-download static files even when a verified copy is already present
      dir (str | os.PathLike): Target directory for files.  Uses ``datadir()`` if not specified

    Raises:
        TypeError: for any other keyword argument (e.g. ``force=True``).
        RuntimeError: under offline mode (``SATKIT_OFFLINE=1`` or
            ``set_offline(True)``), before anything is printed or fetched;
            or when the target directory is not writable (read-only
            filesystem, or owned by another user) — the message names it.

    Static files are fetched according to the data manifest compiled into
    satkit (``data/manifest.json``): each is tried from ``SATKIT_DATA_URL``
    (if set), then the GitHub release asset, the origin server, and the
    legacy bucket, and is only kept when its size and SHA-256 match the
    manifest. Files already present with the right hash are skipped.

    Notes:
        - Files downloaded:
            - ``linux_p1550p2650.440`` : JPL Ephemeris version 440 (~ 100 MB)
            - ``finals2000A.all`` : Earth orientation parameters (IERS Bulletin A), updated daily
            - ``Kp_ap_Ap_SN_F107_since_1932.txt``, ``45-day-forecast.txt``, ``msafe-f10-prd.txt`` : Space weather (GFZ observed record, SWPC and MSAFE forecasts)

        - The IERS nutation tables (``tab5.2a/b/d.txt``) and the gravity
          models (EGM96, EGM2008, JGM2, JGM3 — to degree 70) are compiled
          into satkit and are not downloaded; ITU_GRACE16 (CC BY 4.0) is
          fetched on first use of ``gravmodel.itugrace16``. A full-degree
          gravity file or an updated IERS table placed in the data directory
          still takes precedence over the compiled-in copy.

        - The space weather and Earth-orientation files are downloaded once
          per update rather than transferring the whole table on every call: no
          request is made while the local copy is inside its publication
          cadence (3 h for the GFZ record, 24 h for the SWPC forecast and the Earth-orientation file, a week for MSAFE), and past
          that the request carries ``If-Modified-Since``, so an unchanged file
          costs a ``304``. ``overwrite=True`` forces a full re-fetch of these
          too. Calling this at the start of every script is therefore fine.

    Example:
        ```python
        # Download all data files to the default data directory
        satkit.utils.update_datafiles()

        # Force re-download of all files
        satkit.utils.update_datafiles(overwrite=True)
        ```
    """
    ...

def datadir() -> str | None:
    """Directory where downloaded data files are written

    The core data (IERS nutation tables, gravity models to degree 70) is
    compiled into satkit, so a data directory is only needed for the JPL
    ephemeris (downloaded on first use, SHA-256 verified) and the regularly
    refreshed Earth-orientation / space-weather files.

    Files are *looked up* across several locations (see ``data_search_dirs``),
    but downloads go to exactly one place — ``SATKIT_DATA`` if set, else the
    directory given to ``set_datadir``, else the platform user-data directory:

    - macOS: ``~/Library/Application Support/satkit-data``
    - Linux: ``$XDG_DATA_HOME/satkit-data`` (default ``~/.local/share/satkit-data``)
    - Windows: ``%LOCALAPPDATA%\\satkit-data``

    satkit never writes next to its own extension module or inside
    ``site-packages``. Set ``SATKIT_OFFLINE=1`` to forbid downloads entirely
    (a missing file then raises ``RuntimeError`` naming its sources), and
    ``SATKIT_DATA_URL`` to fetch the manifest-pinned files (the JPL
    ephemeris, ITU_GRACE16) from a mirror; the Earth-orientation and
    space-weather refreshes always go to their producers.

    Returns:
        str | None: directory downloads are written to (created on first use),
        or ``None`` if none could be determined

    Example:
        ```python
        print(satkit.utils.datadir())
        # /Users/user/Library/Application Support/satkit-data
        ```
    """
    ...

def data_search_dirs() -> list[str]:
    """Directories searched for data files, in order

    A file is used from the first directory that contains it; any of these
    may be read-only (a system-wide directory, a directory next to the
    extension). Downloads go only to ``datadir()``.

    1. ``SATKIT_DATA`` (environment; also the write location)
    2. the directory given to ``set_datadir`` (also the write location)
    3. directories added with ``add_search_dir``
    4. ``<dir of the satkit extension>/satkit-data``
    5. the platform user-data directory (the default write location)
    6. ``~/.satkit-data`` (legacy)
    7. ``/usr/share/satkit-data`` (not on Windows)
    8. macOS: ``/Library/Application Support/satkit-data``

    Returns:
        list[str]: search directories in order
    """
    ...

def add_search_dir(path: str) -> None:
    """Add a read-only directory to the data-file search list

    Tried after ``SATKIT_DATA`` / ``set_datadir`` and before the platform
    locations. Downloads are never written here. Use it for a shared or
    provisioned copy of the data files.

    Args:
        path (str): Directory to search
    """
    ...

def set_offline(enabled: bool) -> None:
    """Forbid (or re-allow) downloads for this process

    Offline mode blocks *downloads only*: the explicit ``update_datafiles()``
    and every lazy first-use fetch (the JPL ephemeris, the Earth-orientation
    and space-weather refresh, any non-embedded file), and the element-set
    fetches ``TLE.from_url`` / ``omm_from_url``. It does not change
    where files are searched, and the compiled-in core data (IERS nutation
    tables, gravity models) is unaffected. A blocked download raises
    ``RuntimeError`` naming the file and its sources — the same error a
    build without the ``download`` feature gives.

    Precedence: the last call to ``set_offline`` wins; if it was never
    called, the ``SATKIT_OFFLINE`` environment variable is consulted
    (``1``/anything except ``0``, ``false`` or empty means offline).

    Args:
        enabled (bool): True to forbid downloads, False to allow them
    """
    ...

def is_offline() -> bool:
    """Whether downloads are currently forbidden

    Reflects ``set_offline`` if it was ever called, else the
    ``SATKIT_OFFLINE`` environment variable.

    Returns:
        bool: True if downloads are forbidden
    """
    ...

def set_datadir(datadir: str) -> None:
    """Set the data directory

    The directory becomes the first search location (after ``SATKIT_DATA``)
    and the location downloads are written to.

    Args:
        datadir (str): Path to the data directory

    Raises:
        RuntimeError: If the directory does not exist
    """
    ...

def datafiles_exist() -> bool:
    """Check whether a JPL ephemeris file is present in any search directory

    The ephemeris is the only data satkit needs that is neither compiled in
    nor refreshed daily, so its presence marks a provisioned data location.
    Everything else (frames, gravity, SGP4, time, Kepler, Lambert) works
    without any data files.

    Returns:
        bool: True if an ephemeris file is found, False otherwise
    """
    ...

def dylib_path() -> str:
    """Return path to the compiled satkit library

    Returns:
        str: Path to the compiled library
    """
    ...

def githash() -> str:
    """Return git hash of this satkit build

    ``"unknown"`` unless satkit was built from a git checkout of satkit
    itself (not from an sdist or a copy vendored inside another repository).
    Use ``satkit.__version__`` for the release version.

    Returns:
        str: Git hash of this satkit build, or ``"unknown"``
    """
    ...

def version() -> str:
    """Return version of this satkit library as a string

    Returns:
        str: Version of this satkit library
    """
    ...
