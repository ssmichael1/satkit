"""
Low-precision planetary ephemerides from https://ssd.jpl.nasa.gov/?planet_pos
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

def heliocentric_pos(
    body: satkit.solarsystem,
    tm: satkit.time | list[satkit.time] | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the Heliocentric coordinate system (origin is the Sun)

    Note: This function is only valid for the Sun and the 8 planets

    Note: This is less accurate than using the jpl ephemeris, but involves fewer calculations

    Note: See https://ssd.jpl.nasa.gov/?planet_pos for more information and accuracy details

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the Sun.
                       If input is list or numpy array of N times, then r will be Nx3 array
    """
