"""
Utility functions for SatKit
"""

from __future__ import annotations


def update_datafiles(**kwargs) -> None:
    """Download & store data files needed for "satkit" computations

    Not required for normal use: the IERS nutation tables and gravity models
    are compiled into satkit, the JPL ephemeris is downloaded on first use,
    and the Earth-orientation / space-weather files are fetched on first use.
    Call this to provision everything up front (a container image, a machine
    that will later be offline) or to refresh the daily files. Raises
    ``RuntimeError`` if ``SATKIT_OFFLINE=1`` is set.

    Keyword Args:

      overwrite (bool):  Re-download static files even when a verified copy is already present
      dir(string): Target directory for files.  Uses ``datadir()`` if not specified

    Static files are fetched according to the data manifest compiled into
    satkit (``data/manifest.json``): each is tried from ``SATKIT_DATA_URL``
    (if set), then the GitHub release asset, the origin server, and the
    legacy bucket, and is only kept when its size and SHA-256 match the
    manifest. Files already present with the right hash are skipped.

    Notes:
        - Files include:
            - ``EGM96.gfc`` :   EGM-96 Gravity Model Coefficients
            - ``JGM3.gfc`` :    JGM-3 Gravity Model Coefficients
            - ``JGM2.gfc`` :    JGM-2 Gravity Model Coefficients
            - ``ITU_GRACE16.gfc`` : ITU Grace 16 Gravity
            - ``tab5.2a.txt`` : Coefficients for GCRS to GCRF conversion
            - ``tab5.2b.txt`` : Coefficients for GCRS to GCRF conversion
            - ``tab5.2d.txt`` : Coefficients for GCRS to GCRF conversion
            - ``SW-ALL.csv`` : Space weather data, updated daily
            - ``predicted-solar-cycle.json`` : NOAA/SWPC solar cycle forecast (~5 years of predicted F10.7)
            - ``leap-seconds.list`` : Leap seconds (UTC vs TAI); reference only — the runtime table is compiled in
            - ``EOP-All.csv`` : Earth orientation parameters, updated daily
            - ``linux_p1550p2650.440`` : JPL Ephemeris version 440 (~ 100 MB)

        - The space weather and earth orientation parameters files are updated
          daily and will always be downloaded regardless of the overwrite flag

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
    ``SATKIT_DATA_URL`` to fetch from a mirror.

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
    may be read-only (a system-wide directory, the optional ``satkit-data``
    package inside ``site-packages``). Downloads go only to ``datadir()``.

    1. ``SATKIT_DATA`` (environment; also the write location)
    2. the directory given to ``set_datadir`` (also the write location)
    3. directories added with ``add_search_dir``
    4. ``<dir of the satkit extension>/satkit-data``
    5. ``<site-packages>/satkit_data/data`` (the ``satkit-data`` pip package)
    6. the platform user-data directory (the default write location)
    7. ``~/.satkit-data`` (legacy)
    8. ``/usr/share/satkit-data`` (not on Windows)
    9. macOS: ``/Library/Application Support/satkit-data``

    Returns:
        list[str]: search directories in order
    """
    ...

def add_search_dir(path: str) -> None:
    """Add a read-only directory to the data-file search list

    Tried after ``SATKIT_DATA`` / ``set_datadir`` and before the platform
    locations. Downloads are never written here. The ``satkit`` package uses
    this itself to register the optional ``satkit_data`` bundle.

    Args:
        path (str): Directory to search
    """
    ...

def set_offline(enabled: bool) -> None:
    """Forbid (or re-allow) downloads for this process

    Offline mode blocks *downloads only*: the explicit ``update_datafiles()``
    and every lazy first-use fetch (the JPL ephemeris, the Earth-orientation
    and space-weather refresh, any non-embedded file). It does not change
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

    Returns:
        str: Git hash of this satkit build
    """
    ...

def build_date() -> str:
    """Return build date of this satkit library as a string

    Returns:
        str: Build date of this satkit library
    """
    ...

def version() -> str:
    """Return version of this satkit library as a string

    Returns:
        str: Version of this satkit library
    """
    ...
