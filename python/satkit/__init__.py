from .satkit import *

from ._version import *

if utils.datafiles_exist() == False:  # type: ignore
    print(f"Could not find necessary data files.")
    print(f'Run "satkit.utils.update_datafiles()" to download necessary files')
    print(f"This includes JPL ephemeris, gravity, space weather, ")
    print(f"Earth orientation parameters, leap seconds, and coefficients")
    print(f"for inertial-to-Earth-fixed transforms")
