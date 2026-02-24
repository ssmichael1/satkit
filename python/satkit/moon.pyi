"""
Astrodynamic calculations related to the moon
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
from typing import ClassVar

import satkit

class moonphase:
    """
    Enum representing moon phases
    """

    NewMoon: ClassVar[moonphase]
    """New Moon (0 - 22.5)"""

    WaxingCrescent: ClassVar[moonphase]
    """Waxing Crescent (22.5 - 67.5)"""

    FirstQuarter: ClassVar[moonphase]
    """First Quarter (67.5 - 112.5)"""

    WaxingGibbous: ClassVar[moonphase]
    """Waxing Gibbous (112.5 - 157.5)"""

    FullMoon: ClassVar[moonphase]
    """Full Moon (157.5 - 202.5)"""

    WaningGibbous: ClassVar[moonphase]
    """Waning Gibbous (202.5 - 247.5)"""

    LastQuarter: ClassVar[moonphase]
    """Last Quarter (247.5 - 292.5)"""

    WaningCrescent: ClassVar[moonphase]
    """Waning Crescent (292.5 - 337.5)"""

@typing.overload
def pos_gcrf(time: satkit.time) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the GCRF Frame

    From Vallado Algorithm 31

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing moon position in GCRF frame
        at given time.  Units are meters

    Notes:
        Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
        and 1275 km in range

    Example:
        ```python
        import numpy as np
        t = satkit.time(2024, 1, 1)
        moon = satkit.moon.pos_gcrf(t)
        print(f"Moon distance: {np.linalg.norm(moon)/1e3:.0f} km")
        ```
    """
    ...

@typing.overload
def pos_gcrf(
    time: npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the GCRF Frame

    From Vallado Algorithm 31

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time
            for which to compute position

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing moon position in GCRF frame
        at given times.  Units are meters

    Notes:
        Accurate to 0.3 degree in ecliptic longitude, 0.2 degree in ecliptic latitude,
        and 1275 km in range
    """
    ...

def illumination(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64] | float:
    """
    Fractional illumination of moon

    Args:
        time (satkit.time | npt.ArrayLike | list[satkit.time]): time object, list, or numpy array
            for which to compute illumination

    Returns:
        float | npt.NDArray[np.float64]: float or numpy array of floats representing fractional illumination of moon at given time(s).

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        illum = satkit.moon.illumination(t)
        print(f"Moon illumination: {illum*100:.1f}%")
        ```
    """
    ...

def phase(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64] | float:
    """
    Phase of moon in radians

    Args:
        time (satkit.time | npt.ArrayLike | list[satkit.time]): time object, list, or numpy array
            for which to compute phase

    Returns:
        float | npt.NDArray[np.float64]: float or numpy array of floats representing moon phase in radians at given time(s).
    """
    ...

def phase_name(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> moonphase | list[moonphase]:
    """
    Phase name of moon

    Args:
        time (satkit.time | npt.ArrayLike | list[satkit.time]): time object, list, or numpy array
            for which to compute phase name

    Returns:
        moonphase | list[moonphase]: moonphase or list of moonphase representing moon phase name at given time(s).

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        p = satkit.moon.phase_name(t)
        print(p)
        # e.g., moonphase.WaningGibbous
        ```
    """
    ...
