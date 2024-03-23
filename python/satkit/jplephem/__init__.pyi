"""
High-precision JPL ephemerides for solar-system bodies

For details, see:
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

def geocentric_pos(body: satkit.solarsystem, tm: satkit.time|list[satkit.time]|npt.ArrayLike[satkit.time]) -> npt.ArrayLike[np.float64]:
    """
    Return geocentric (Earth-centered) positionfor a given solar-system body
    at the given time or times in the GCRF frame 
    
    If time is a single value, 3-element numpy array is returned.
    Units are meters.
    
    If time is a list of values or a numpy array of values, a numpy array of size Nx3
    are returned where N is the number of time values
    
    Example:
    
    from satkit import jplephem
    from satkit import solarsystem

    timearray = [satkit.time(2022, 1, 1), satkit.time(2022, 1, 2)]
    print(jplephem.geocentric_pos(solarsystem.Sun, timearray))    
    
    [[ 2.61298635e+10 -1.32825366e+11 -5.75794120e+10]
    [ 2.87014564e+10 -1.32376376e+11 -5.73848916e+10]]    
    """
    
def barycentric_pos(body: satkit.solarsystem, tm: satkit.time|list[satkit.time]|npt.ArrayLike[satkit.time]) -> npt.ArrayLike[np.float64]:
    """
    Return barycentric position for a given solar-system body
    at the given time or times.  The barycentric coordinate system origin 
    is the center of mass of the solar system, and axis are aligned with the
    GCRF frame
    
    If time is a single value, 3-element numpy array is returned.
    Units are meters.
    
    If time is a list of values or a numpy array of values, a numpy array of size Nx3
    are returned where N is the number of time values
    
    Example:
    
    from satkit import jplephem
    from satkit import solarsystem

    timearray = [satkit.time(2022, 1, 1), satkit.time(2022, 1, 2)]
    print(jplephem.barycentric(solarsystem.Sun, timearray))    
    
    [[-1.28367505e+09  4.49091935e+08  2.22928070e+08]
     [-1.28417613e+09  4.47924236e+08  2.22445460e+08]]
    """    

def geocentric_state(body: satkit.solarsystem, 
                     tm: satkit.time|list[satkit.time]|npt.ArrayLike[satkit.time]) -> typing.Tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]:
    """
    Return geocentric (Earth-centered) state (position and velocity) for a given solar-system body
    at the given time or times in the GCRF frame 
    
    If time is a single value, a tuple of numpy arrays (position, velocity) are returned.
    Units are meters and meters / second.
    
    If time is a list of values or a numpy array of values, a tuple of numpy arrays of size Nx3
    are returned where N is the number of time values
    
    Example:
    
    from satkit import jplephem
    from satkit import solarsystem

    timearray = [satkit.time(2022, 1, 1), satkit.time(2022, 1, 2)]
    print(jplephem.geocentric_state(solarsystem.Sun, timearray))
    
    (array([[ 2.61298635e+10, -1.32825366e+11, -5.75794120e+10],
       [ 2.87014564e+10, -1.32376376e+11, -5.73848916e+10]]),
       array([[29812.13116606,  4956.22306944,  2147.11845068],
       [29713.79583428,  5436.80972595,  2355.57132841]]))
    
    
    """
    
def barycentric_state(body: satkit.solarsystem, 
                     tm: satkit.time|list[satkit.time]|npt.ArrayLike[satkit.time]) -> typing.Tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]:
    """
    Return barycentric position and velocity for a given solar-system body
    at the given time or times.  The barycentric coordinate system origin 
    is the center of mass of the solar system, and axis are aligned with the
    GCRF frame
    
    If time is a single value, a tuple of numpy arrays (position, velocity) are returned.
    Units are meters and meters / second.
    
    If time is a list of values or a numpy array of values, a tuple of numpy arrays of size Nx3
    are returned where N is the number of time values
    
    Example:
    
    from satkit import jplephem
    from satkit import solarsystem

    timearray = [satkit.time(2022, 1, 1), satkit.time(2022, 1, 2)]
    print(jplephem.barycentric_state(solarsystem.Sun, timearray))
    
    (array([[-1.28367505e+09,  4.49091935e+08,  2.22928070e+08],
       [-1.28417613e+09,  4.47924236e+08,  2.22445460e+08]]),
       array([[ -5.80935403, -13.51319934,  -5.58473785],
       [ -5.78987442, -13.51686542,  -5.58678539]]))
    
    
    """    