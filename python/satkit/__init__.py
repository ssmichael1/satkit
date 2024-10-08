from .satkit import *

__version__ = utils.version()[1:]

if utils.datafiles_exist() == False:
    print(f"Could not find necessary data files.")
    print(f'Run "satkit.utils.update_datafiles()" to download necessary files')
    print(f"This includes JPL ephemeris, gravity, space weather, ")
    print(f"Earth orientation parameters, leap seconds, and coefficients")
    print(f"for inertial-to-Earth-fixed transforms")
