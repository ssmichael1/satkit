"""
satkitdynamic calculations related to the moon
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

@typing.overload
def pos_gcrf(time: satkit.time) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the GCRF Frame

    From Vallado Algorithm 31

    Input:

    time:  satkit.time object,
    Output:

    3-element numpy array representing moon position in GCRF frame
    at given time.  Units are meters

    Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
    and 1275 km in range
    """

@typing.overload
def pos_gcrf(
    time: npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the GCRF Frame

    From Vallado Algorithm 31

    Input:

    time:  satkit.time list, or numpy array
            for which to compute position

    Output:

    Nx3 numpy array representing moon position in GCRF frame
    at given times.  Units are meters

    Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
    and 1275 km in range
    """
