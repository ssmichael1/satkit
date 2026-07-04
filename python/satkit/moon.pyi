"""
Astrodynamic calculations related to the moon
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
from typing import ClassVar

import satkit
from .satkit import TimeScalar, TimeArrayLike, TimeInput

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
def pos_gcrf(time: TimeScalar) -> npt.NDArray[np.float64]:
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
    time: TimeArrayLike,
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

@typing.overload
def illumination(time: TimeScalar) -> float:
    """
    Fractional illumination of moon

    Args:
        time (satkit.time | datetime.datetime): scalar time at which to compute illumination

    Returns:
        float: fractional illumination of moon at the given time

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        illum = satkit.moon.illumination(t)
        print(f"Moon illumination: {illum*100:.1f}%")
        ```
    """
    ...

@typing.overload
def illumination(time: TimeArrayLike) -> list[float]:
    """
    Fractional illumination of moon

    Args:
        time (TimeArrayLike): list or numpy array of times at which to compute illumination

    Returns:
        list[float]: fractional illumination of moon at each given time
    """
    ...

@typing.overload
def phase(time: TimeScalar) -> float:
    """
    Phase of moon in radians

    Args:
        time (satkit.time | datetime.datetime): scalar time at which to compute phase

    Returns:
        float: moon phase in radians at the given time
    """
    ...

@typing.overload
def phase(time: TimeArrayLike) -> list[float]:
    """
    Phase of moon in radians

    Args:
        time (TimeArrayLike): list or numpy array of times at which to compute phase

    Returns:
        list[float]: moon phase in radians at each given time
    """
    ...

@typing.overload
def phase_name(time: TimeScalar) -> moonphase:
    """
    Phase name of moon

    Args:
        time (satkit.time | datetime.datetime): scalar time at which to compute phase name

    Returns:
        moonphase: moon phase name at the given time
    """
    ...

@typing.overload
def phase_name(time: TimeArrayLike) -> list[moonphase]:
    """
    Phase name of moon

    Args:
        time (TimeArrayLike): list or numpy array of times at which to compute phase name

    Returns:
        list[moonphase]: moon phase name at each given time
    """
    ...

