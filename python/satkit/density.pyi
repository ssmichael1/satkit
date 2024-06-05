"""
Air density models

Currently only contains NRL MSISE-00 air density model
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit


@typing.overload
def nrlmsise(itrf: satkit.itrfcoord, time: satkit.time | None) -> typing.Tuple[float, float]:
    """
    NRL MSISE-00 Atmosphere Density Model
    
    https://en.wikipedia.org/wiki/NRLMSISE-00
    
    or for more detail:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002JA009430
    
    Args: 
    
        itrf (satkit.itrfcoord):  position at which to compute density & temperature                  
        time (satkit.time|numpy.ndarray|list):  Optional instant(s) at which to compute density & temperature.
               "Space weather" data at this time will be used in model 
               computation.  Note: at satellite altitudes, density can
               change by > 10 X depending on solar cycle
                 
    Returns:
        tuple: (rho, T) where rho is mass density in kg/m^3 and T is temperature in Kelvin
    """
    
@typing.overload
def nrlmsise(altitude_meters: float,
             latitude_rad: float,
             longitude_rad: float,
             time: satkit.time | None) -> typing.Tuple[float, float]:
    """
    NRL MSISE-00 Atmosphere Density Model
    
    https://en.wikipedia.org/wiki/NRLMSISE-00
    
    or for more detail:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002JA009430
    
    Args:
        altitude_meters (float):  Altitude in meters
        latitude_rad (float, optional):  Latitude in radians. Default is 0.
        longitude_rad (float, optional):  Longitude in radians.  Default is 0.
        time (satkit.time|numpy.ndarray|list, optional):  Optional instant(s) at which to compute density & temperature.
               "Space weather" data at this time will be used in model 
               computation.  Note: at satellite altitudes, density can
               change by > 10 X depending on solar cycle
                 
    Returns:
        tuple: (rho, T) where rho is mass density in kg/m^3 and T is temperature in Kelvin
    """