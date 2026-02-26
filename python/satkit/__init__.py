from importlib.metadata import version

__version__ = version("satkit")

from .satkit import *  # type: ignore

import os
import sys

# Try to locate data files from the satkit_data pip package
if not utils.datafiles_exist():  # type: ignore
    try:
        import satkit_data

        _data_path = os.path.join(os.path.dirname(satkit_data.__file__), "data")
        if os.path.isdir(_data_path):
            utils.set_datadir(_data_path)  # type: ignore
        del _data_path
    except (ImportError, RuntimeError):
        pass

if not utils.datafiles_exist():  # type: ignore
    print(f"Could not find necessary data files.", file=sys.stderr)
    print(f"Install satkit-data package via pypi or", file=sys.stderr)
    print(
        f'Run "satkit.utils.update_datafiles()" to download necessary files',
        file=sys.stderr,
    )
    print(f"This includes JPL ephemeris, gravity, space weather, ", file=sys.stderr)
    print(
        f"Earth orientation parameters, leap seconds, and coefficients", file=sys.stderr
    )
    print(f"for inertial-to-Earth-fixed transforms", file=sys.stderr)
