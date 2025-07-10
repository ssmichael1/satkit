"""
High-precision JPL ephemerides for solar-system bodies

For details, see: https://ssd.jpl.nasa.gov/
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

def geocentric_pos(
    body: satkit.solarsystem, tm: satkit.time | list[satkit.time] | npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the GCRF coordinate system (origin is Earth center)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of cartesian Geocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array
    """

def barycentric_pos(
    body: satkit.solarsystem,
    tm: satkit.time | list[satkit.time] | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the Barycentric coordinate system (origin is solarsystem barycenter)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the solar system barycenter. If input is list or numpy array of N times, then r will be Nx3 array

    Notes:
     * Positions for all bodies are natively relative to solar system barycenter,
       with exception of moon, which is computed in Geocentric system
     * EMB (2) is the Earth-Moon barycenter
     * The sun position is relative to the solar system barycenter
       (it will be close to origin)
    """

def geocentric_state(
    body: satkit.solarsystem,
    tm: satkit.time | list[satkit.time] | npt.ArrayLike,
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the position and velocity of the given body in Geocentric coordinate system (GCRF)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays
    """

def barycentric_state(
    body: satkit.solarsystem,
    tm: satkit.time | list[satkit.time] | npt.ArrayLike,
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the position & velocity the given body in the barycentric coordinate system (origin is solar system barycenter)


    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second, with the origin at the solar system barycenter.  If input is list or numpy array of N times, then r and v will be Nx3 arrays

    Notes:
     * Positions for all bodies are natively relative to solar system barycenter,
       with exception of moon, which is computed in Geocentric system
     * EMB (2) is the Earth-Moon barycenter
     * The sun position is relative to the solar system barycenter
       (it will be close to origin)

    """
