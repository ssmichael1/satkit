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
    
    Inputs: 
    
        itrf:  satkit.itrfcoord representing position at which to 
                compute density & temperature
                  
        time:  Optional instant at which to compute density & temperature.
               "Space weather" data at this time will be used in model 
               computation.  Note: at satellite altitudes, density can
               change by > 10 X depending on solar cycle
                 
    Outputs:
        
        rho:  Atmosphere mass density in kg / m^3
        
          T:  Atmosphere temperature in Kelvin
    
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
    
    Inputs: 
    
    altitude_meters:  Altitude in meters
    
       latitude_rad:  Latitude in radians
       
      longitude_rad:  Longitude in radians
                 
               time:  Optional instant at which to compute density & temperature.
                      "Space weather" data at this time will be used in model 
                      computation.  Note: at satellite altitudes, density can
                      change by > 10 X depending on solar cycle                 
    Outputs:
        
        rho:  Atmosphere mass density in kg / m^3
        
          T:  Atmosphere temperature in Kelvin
    
    """