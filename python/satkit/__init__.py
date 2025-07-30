from .satkit import * # type: ignore

from ._version import *
import sys

if utils.datafiles_exist() == False:  # type: ignore
    print(f"Could not find necessary data files.", file=sys.stderr)
    print(f"Install satkit-data package via pypi or", file=sys.stderr)
    print(f'Run "satkit.utils.update_datafiles()" to download necessary files', file=sys.stderr)
    print(f"This includes JPL ephemeris, gravity, space weather, ", file=sys.stderr)
    print(f"Earth orientation parameters, leap seconds, and coefficients", file=sys.stderr)
    print(f"for inertial-to-Earth-fixed transforms", file=sys.stderr)
