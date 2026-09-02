from importlib.metadata import version

__version__ = version("satkit")

from .satkit import *  # type: ignore
from ._types import OMMDict

# The core data (IERS nutation tables, gravity models to degree 70) is
# compiled into the extension, so satkit works with no data directory at all.
# The JPL ephemeris is downloaded (SHA-256 verified) on first use into
# `satkit.utils.datadir()`, and the Earth-orientation / space-weather files
# are refreshed by `satkit.utils.update_datafiles()`.
#
# If the optional offline bundle (the `satkit_data` package, from
# `pip install satkit-data` or the conda `satkit-data` package) is importable,
# register its `data/` directory as a read-only search location so its
# ephemeris and files are used wherever the package happens to be installed.
# Downloads still go to `datadir()`.


def _optional_data_bundle_dirs(module) -> list:
    """Candidate ``data`` directories of an installed ``satkit_data`` package.

    The conda package (and any directory without ``__init__.py``) imports as
    a *namespace* package, whose ``__file__`` is ``None`` — only ``__path__``
    is meaningful. The PyPI wheel is a regular package with ``__file__``.
    Both layouts put the files under ``<package>/data``.
    """
    import os

    roots = []
    file = getattr(module, "__file__", None)
    if file:
        roots.append(os.path.dirname(file))
    for entry in getattr(module, "__path__", None) or []:
        if entry not in roots:
            roots.append(entry)
    return [os.path.join(r, "data") for r in roots if os.path.isdir(os.path.join(r, "data"))]


try:
    import satkit_data as _satkit_data
except ImportError:
    pass
else:
    try:
        for _d in _optional_data_bundle_dirs(_satkit_data):
            utils.add_search_dir(_d)  # type: ignore
    except Exception:  # never let optional data discovery break `import satkit`
        pass
    del _satkit_data
