"""
satkitdynamic calculations related to the sun
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

@typing.overload
def pos_gcrf(time: satkit.time) -> npt.NDArray[np.float64]:
    """
    Sun position in the Geocentric Celestial Reference Frame (GCRF)

    Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Input:

    time:  satkit.time object representing time
            at which to compute position

    Output:

    3-element numpy array representing sun position in GCRF frame
    at given time[s].  Units are meters


    From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050

    """

@typing.overload
def pos_gcrf(
    time: npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64]:
    """
    Sun position in the Geocentric Celestial Reference Frame (GCRF)

    Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Input:

    time:  list or numpy array of satkit.time objects representing times
            at which to compute position

    Output:

    Nx3 numpy array representing sun position in GCRF frame
    at the "N" given times.  Units are meters


    From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
    """


@typing.overload
def pos_mod(time: satkit.time) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Algorithm 29 from Vallado for sun in Mean of Date (MOD)

    Args:

        time:  satkit.time object representing time
                aatwhich to compute position

    Returns:

    3-element numpy array representing sun position in MOD frame
    at given time.  Units are meters

    From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050

    """

@typing.overload
def pos_mod(
    time: npt.ArrayLike | list[satkit.time],
) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Algorithm 29 from Vallado for sun in Mean of Date (MOD)

    Args:

        time (time|npt.ArrayLike[time]):  list or numpy arra of satkit.time objects,
             representing times at which to compute positions

    Returns:

        (npt.ArrayLike[float]): Nx3 numpy array representing sun position in MOD frame
        at given times.  Units are meters


    Notes:

        From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
    """

def rise_set(
    time: satkit.time, coord: satkit.itrfcoord, sigma: float = 90.0 + 50.0 / 60.0
) -> typing.Tuple[satkit.time, satkit.time]:
    """
    Sunrise and sunset times on the day given by input time
    and at the given location.

    Time is a "date" at location, and should have hours, minutes, and seconds
    set to zero

    Vallado Algorithm 30

    Args:

        Time:   satkit.time representing date for which to compute
                sunrise & sunset

        coord:   satkit.itrfcoord representing location for which to compute
                sunrise & sunset

        sigma:   optional angle in degrees between noon & rise/set:
                Common Values:
                            "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
                        "Civil Twilight": 96 deg
                    "Nautical Twilight": 102 deg
                "satkitnomical Twilight": 108 deg

                If not passed in, "Standard" is used (90.0 + 50.0/60.0)

    Returns:

        (sunrise: satkit.time, sunset: satkit.time)

    """

def shadowfunc(
    sunpos: npt.NDArray[np.float64], satpos: npt.NDArray[np.float64]
) -> float:
    """
    Is satellite in Earth shadow given sun position

    Args:
        sunpos: geocentric Sun position, meters
        satpos: geocentric satellite position, meters,

    Notes:
    * See algorithm in Section 3.4.2 of Montenbruck and Gill for calculation

    Returns:
        (float) : number in range [0,1] indication no sun or full sun (no occlusion) hitting satellite

    """
