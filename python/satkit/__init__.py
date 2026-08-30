from importlib.metadata import version

__version__ = version("satkit")

from .satkit import *  # type: ignore

# The core data (IERS nutation tables, gravity models to degree 70) is
# compiled into the extension, so satkit works with no data directory at all.
# The JPL ephemeris is downloaded (SHA-256 verified) on first use into
# `satkit.utils.datadir()`, and the Earth-orientation / space-weather files
# are refreshed by `satkit.utils.update_datafiles()`.
#
# If the optional offline bundle (`pip install satkit[data]`, i.e. the
# `satkit_data` package) is importable, register its data directory as a
# read-only search location so its ephemeris and files are used wherever
# the package happens to be installed. Downloads still go to `datadir()`.
try:
    import os as _os

    import satkit_data as _satkit_data

    _data_path = _os.path.join(_os.path.dirname(_satkit_data.__file__), "data")
    if _os.path.isdir(_data_path):
        utils.add_search_dir(_data_path)  # type: ignore
    del _data_path, _satkit_data, _os
except ImportError:
    pass
