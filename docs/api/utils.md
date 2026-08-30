# Utility Functions

Data-file management lives here. The short version — see [Data Files](../getting-started/datafiles.md) for the full search order per platform:

- Frames and gravity need **no data files**: the IERS nutation tables and the gravity models are compiled in.
- The JPL ephemeris is **downloaded on first use** (SHA-256 verified) into `datadir()` — the platform user-data directory, or `SATKIT_DATA` / `set_datadir()` if given. Nothing is ever written next to the extension module or inside `site-packages`.
- Files are **looked up** across `data_search_dirs()`, which includes an installed `satkit-data` package and `/usr/share/satkit-data` (read-only), so an offline bundle is picked up automatically.
- `update_datafiles()` provisions everything up front and refreshes the daily Earth-orientation / space-weather files.
- `SATKIT_OFFLINE=1` forbids downloads (a missing file raises `RuntimeError` naming its sources); `SATKIT_DATA_URL` names a mirror; `SATKIT_JPLEPHEM_FILE` selects the ephemeris.

::: satkit.utils
