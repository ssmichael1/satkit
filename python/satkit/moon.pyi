"""
satkitdynamic calculations related to the moon
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
import satkit

from satkit import static_property

class moonphase:
    """
    Enum representing moon phases
    """

    @static_property
    def NewMoon(self) -> moonphase:
        """
        New Moon (0° - 22.5°)
        """

    @static_property
    def WaxingCrescent(self) -> moonphase:
        """
        Waxing Crescent (22.5° - 67.5°)
        """

    @static_property
    def FirstQuarter(self) -> moonphase:
        """
        First Quarter (67.5° - 112.5°)
        """

    @static_property
    def WaxingGibbous(self) -> moonphase:
        """
        Waxing Gibbous (112.5° - 157.5°)
        """

    @static_property
    def FullMoon(self) -> moonphase:
        """
        Full Moon (157.5° - 202.5°)
        """

    @static_property
    def WaningGibbous(self) -> moonphase:
        """
        Waning Gibbous (202.5° - 247.5°)
        """

    @static_property
    def LastQuarter(self) -> moonphase:
        """
        Last Quarter (247.5° - 292.5°)
        """

    @static_property
    def WaningCrescent(self) -> moonphase:
        """
        Waning Crescent (292.5° - 337.5°)
        """

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

def illumination(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64] | float:
    """
    Fractional illumination of moon

    Input:

    time:  satkit.time object, list, or numpy array
            for which to compute illumination

    Output:

    float or numpy array of floats representing fractional illumination of moon at given time(s).
    """

def phase(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64] | float:
    """
    Phase of moon in radians

    Input:

    time:  satkit.time object, list, or numpy array
            for which to compute phase

    Output:

    float or numpy array of floats representing moon phase in radians at given time(s).
    """

def phase_name(
    time: satkit.time | npt.ArrayLike | list[satkit.time],
) -> moonphase | list[moonphase]:
    """
    Phase name of moon

    Input:

    time:  satkit.time object, list, or numpy array
            for which to compute phase name

    Output:

    moonphase or list of moonphase representing moon phase name at given time(s).
    """
