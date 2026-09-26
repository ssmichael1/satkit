"""
Astrodynamic calculations related to the sun
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit
from .satkit import TimeScalar, TimeArrayLike, TimeInput

@typing.overload
def pos_gcrf(time: TimeScalar) -> npt.NDArray[np.float64]:
    """
    Sun position in the Geocentric Celestial Reference Frame (GCRF)

    Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing sun position in GCRF frame
        at given time.  Units are meters

    Notes:
        From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050

    Example:
        ```python
        import numpy as np
        t = satkit.time(2024, 6, 21)
        sun = satkit.sun.pos_gcrf(t)
        print(f"Sun distance: {np.linalg.norm(sun)/1e9:.3f} million km")
        ```
    """
    ...

@typing.overload
def pos_gcrf(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Sun position in the Geocentric Celestial Reference Frame (GCRF)

    Algorithm 29 from Vallado for sun in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time objects
            representing times at which to compute position

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing sun position in GCRF frame
        at the "N" given times.  Units are meters

    Notes:
        From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
    """
    ...

@typing.overload
def pos_mod(time: TimeScalar) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Algorithm 29 from Vallado for sun in Mean of Date (MOD)

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing sun position in MOD frame
        at given time.  Units are meters

    Notes:
        From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
    """
    ...

@typing.overload
def pos_mod(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Algorithm 29 from Vallado for sun in Mean of Date (MOD)

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time objects,
             representing times at which to compute positions

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing sun position in MOD frame
        at given times.  Units are meters

    Notes:
        From Vallado: Valid with accuracy of .01 degrees from 1950 to 2050
    """
    ...

def rise_set(
    time: TimeScalar, coord: satkit.itrfcoord, sigma: float | None = None
) -> tuple[satkit.time, satkit.time]:
    """
    Sunrise and sunset times on a calendar date at the given location.

    The input selects the UTC calendar date (its time of day is ignored);
    the returned sunrise and sunset are those of that date at the location's
    longitude, as UTC times.  At far-west longitudes the sunset can fall on
    the next UTC date (and at far-east longitudes the sunrise on the
    previous one).

    To get the events of a *local* date, pass a timezone-aware datetime at
    local noon: for time zones UTC-11 to UTC+11 local noon falls on the same
    UTC date, while local midnight falls on the previous UTC date east of
    Greenwich.

    Vallado Algorithm 30

    Args:
        time (satkit.time | datetime.datetime): time whose UTC calendar date selects the day
        coord (satkit.itrfcoord): location for which to compute sunrise & sunset
        sigma (float, optional): angle in degrees between noon & rise/set.
            Common Values:
                        "Standard": 90 deg, 50 arcmin (90.0+50.0/60.0)
                    "Civil Twilight": 96 deg
                "Nautical Twilight": 102 deg
            "Astronomical Twilight": 108 deg

            If None or not passed in, "Standard" is used (90.0 + 50.0/60.0)

    Returns:
        tuple[satkit.time, satkit.time]: (sunrise, sunset)

    Example:
        ```python
        from datetime import datetime, timedelta, timezone

        # Sunrise and sunset on 2024-10-14 in Honolulu: pass local noon on that
        # date (a ZoneInfo("Pacific/Honolulu") tzinfo works the same way)
        honolulu = timezone(timedelta(hours=-10))
        coord = satkit.itrfcoord(latitude_deg=21.31, longitude_deg=-157.86)
        noon = datetime(2024, 10, 14, 12, tzinfo=honolulu)
        sunrise, sunset = satkit.sun.rise_set(noon, coord)
        print(f"Sunrise: {sunrise.to_datetime().astimezone(honolulu)}")
        print(f"Sunset:  {sunset.to_datetime().astimezone(honolulu)}")
        ```
    """
    ...

def shadowfunc(
    sunpos: npt.NDArray[np.float64], satpos: npt.NDArray[np.float64]
) -> float:
    """
    Is satellite in Earth shadow given sun position

    Args:
        sunpos (npt.NDArray[np.float64]): geocentric Sun position, meters
        satpos (npt.NDArray[np.float64]): geocentric satellite position, meters

    Notes:
        - See algorithm in Section 3.4.2 of Montenbruck and Gill for calculation
        - Beyond ~1.4 million km on the anti-Sun side the Earth's disc is smaller
          than the Sun's, and on the shadow axis the eclipse is annular:
          1 - b^2/a^2, with a and b the apparent radii of the Sun and the Earth
        - A position at or below the Earth's surface is lit when the Sun is above
          its local horizon plane; the Earth's center returns 0

    Returns:
        float: number in range [0,1] indicating no sun or full sun (no occlusion) hitting satellite

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        sun = satkit.sun.pos_gcrf(t)
        sat_pos = np.array([6.781e6, 0, 0])  # satellite position, GCRF, meters
        shadow = satkit.sun.shadowfunc(sun, sat_pos)
        if shadow < 0.01:
            print("Satellite is in Earth shadow")
        else:
            print(f"Illumination fraction: {shadow:.2f}")
        ```
    """
    ...
