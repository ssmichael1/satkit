from .satkit import *  # type: ignore
from ._types import OMMDict
from .satkit import __version__

# The core data (IERS nutation tables, public-domain gravity models to degree 70) is
# compiled into the extension, so satkit works with no data directory at all.
# The JPL ephemeris is downloaded (SHA-256 verified) on first use into
# `satkit.utils.datadir()`, and the Earth-orientation / space-weather files
# are refreshed by `satkit.utils.update_datafiles()`.
