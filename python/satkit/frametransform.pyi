"""
Transformations between coordinate frames, and associated utility functions

Coordinate frame transforms are mostly pulled from Vallado:
https://www.google.com/books/edition/Fundamentals_of_Astrodynamics_and_Applic/PJLlWzMBKjkC?hl=en&gbpv=0

or the IERS:
https://www.iers.org/

"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
import datetime

from .satkit import time, quaternion

@typing.overload
def gmst(tm: time | datetime.datetime) -> float:
    """Greenwich Mean Sidereal Time

    Notes:
        * GMST is the angle between the vernal equinox and the Greenwich meridian
        * Vallado algorithm 15
        * GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

    Args:
        tm (satkit.time | datetime.datetime): scalar time at which to calculate output

    Returns:
        float: Greenwich Mean Sidereal Time, radians, at input time

    Example:
        ```python
        import math
        t = satkit.time(2024, 1, 1)
        theta = satkit.frametransform.gmst(t)
        print(f"GMST: {math.degrees(theta):.4f} deg")
        ```
    """
    ...

@typing.overload
def gmst(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.NDArray[np.float64]:
    """Greenwich Mean Sidereal Time

    Notes:
        * GMST is the angle between the vernal equinox and the Greenwich meridian
        * Vallado algorithm 15
        * GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        float | npt.ArrayLike[np.float]: Greenwich Mean Sidereal Time, radians, at input time(s)
    """
    ...

@typing.overload
def gast(
    tm: time | datetime.datetime,
) -> float:
    """Greenwich Apparent Sidereal Time

    Args:
        tm (satkit.time): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        float : Greenwich apparent sidereal time, radians, at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        theta = satkit.frametransform.gast(t)
        ```
    """
    ...

@typing.overload
def gast(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.NDArray[np.float64]:
    """Greenwich Apparent Sidereal Time

    Args:
        tm (npt.ArrayLike[datetime.datetime] | npt.ArrayLike[time]): list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        npt.ArrayLike[np.float]: Greenwich apparent sidereal time, radians, at input time(s)
    """
    ...

@typing.overload
def earth_rotation_angle(
    tm: time | datetime.datetime,
) -> float:
    """Earth Rotation Angle

    Notes:
        * See: IERS Technical Note 36, Chapter 5, Equation 5.15
        * Calculation Details:
            * Let t be UT1 Julian date
            * let f be fractional component of t (fraction of day)
            * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))

    Args:
        tm (satkit.time|datetime.datetime: Time[s] at which to calculate Earth Rotation Angle

    Returns:
        float: Earth Rotation Angle at input time[s] in radians

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        era = satkit.frametransform.earth_rotation_angle(t)
        ```
    """
    ...

@typing.overload
def earth_rotation_angle(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.NDArray[np.float64]:
    """Earth Rotation Angle

    Notes:
        * See: IERS Technical Note 36, Chapter 5, Equation 5.15
        * Calculation Details:
            * Let t be UT1 Julian date
            * let f be fractional component of t (fraction of day)
            * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0)

    Args:
        tm (npt.ArrayLike[datetime.datetime] | npt.ArrayLike[time]): list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        npt.ArrayLike[np.float]: Earth Rotation Angle at input time[s] in radians
    """
    ...

@typing.overload
def qitrf2tirs(
    tm: time,
) -> quaternion:
    """Rotation from Terrestrial Intermediate Reference System to Celestial Intermediate Reference Systems

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to TIRS at input time(s)
    """
    ...

@typing.overload
def qitrf2tirs(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Rotation from Terrestrial Intermediate Reference System to Celestial Intermediate Reference Systems

    Args:
        tm (npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to TIRS at input time(s)
    """
    ...

@typing.overload
def qteme2gcrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Rotation from True Equator Mean Equinox (TEME) to Geocentric Celestial Reference Frame (GCRF)

    Args:
        tm (satkit.time| datetime.datetime ): Time[s] at which to calculate the quaternion

    Returns:
        quaternion : Quaternion representing rotation from TEME to GCRF at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qteme2gcrf(t)
        ```
    """
    ...

@typing.overload
def qteme2gcrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Rotation from True Equator Mean Equinox (TEME) to Geocentric Celestial Reference Frame (GCRF)

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from TEME to GCRF at input time(s)
    """
    ...

@typing.overload
def qcirs2gcrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from CIRS to GCRF at input time(s)
    """
    ...

@typing.overload
def qcirs2gcrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from CIRS to GCRF at input time(s)
    """
    ...

@typing.overload
def qtirs2cirs(
    tm: time | datetime.datetime,
) -> quaternion:
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from TIRS to CIRS at input time(s)
    """
    ...

@typing.overload
def qtirs2cirs(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from TIRS to CIRS at input time(s)
    """
    ...

@typing.overload
def qgcrf2itrf_approx(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Accurate to approx. 1 arcsec

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

@typing.overload
def qgcrf2itrf_approx(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Accurate to approx. 1 arcsec

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

@typing.overload
def qitrf2gcrf_approx(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        * Accurate to approx. 1 arcsec

    Args:
        tm (satkit.time  | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

@typing.overload
def qitrf2gcrf_approx(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        * Accurate to approx. 1 arcsec

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

@typing.overload
def qgcrf2itrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)

    Example:
        ```python
        import numpy as np

        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qgcrf2itrf(t)

        # Rotate a GCRF position vector to ITRF
        pos_gcrf = np.array([6.781e6, 0, 0])
        pos_itrf = q * pos_gcrf
        ```
    """
    ...

@typing.overload
def qgcrf2itrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

@typing.overload
def qitrf2gcrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive

    Args:
        tm (satkit.time  datetime.datetime): Time[s] at which to calculate the quaternion
    Returns:
        quaternion : Quaternion representing rotation from ITRF to GCRF at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qitrf2gcrf(t)
        ```
    """
    ...

@typing.overload
def qitrf2gcrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

@typing.overload
def qteme2itrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * This is equation 3-90 in Vallado
        * TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from TEME to ITRF at input time(s)

    Example:
        ```python
        import numpy as np

        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qteme2itrf(t)

        # Convert SGP4 TEME output to ITRF
        pos_teme = np.array([6.781e6, 0, 0])
        pos_itrf = q * pos_teme
        ```
    """
    ...

@typing.overload
def qteme2itrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * This is equation 3-90 in Vallado
        * TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from TEME to ITRF at input time(s)
    """
    ...

def earth_orientation_params(
    time: time,
) -> tuple[float, float, float, float, float, float]:
    """Get Earth Orientation Parameters at given instant

    Args:
        time (satkit.time): Instant at which to query parameters

    Returns:
        (float, float, float, float, float, float) | None: Tuple with following elements:
            0 : (UT1 - UTC) in seconds
            1 : X polar motion in arcsecs
            2 : Y polar motion in arcsecs
            3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
            4 : dX wrt IAU-2000A nutation, milli-arcsecs
            5 : dY wrt IAU-2000A nutation, milli-arcsecs

    Notes:
        * Returns None if the time is outside the range of available Earth Orientation Parameters (EOP)
        * EOP are only available from 1962 to current, and predict to current + ~ 4 months
        * If EOP are not available, all values will be set to zero, and a warning will be printed to stderr
        * See: https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        eop = satkit.frametransform.earth_orientation_params(t)
        if eop is not None:
            ut1_utc, xp, yp, lod, dx, dy = eop
            print(f"UT1-UTC: {ut1_utc:.6f} s")
        ```
    """
    ...

def disable_eop_time_warning() -> None:
    """Disable the warning printed to stderr when Earth Orientation Parameters (EOP) are not available for a given time.

    Notes:
        * This function is used to disable the warning printed when EOP are not available for a given time.
        * If not disabled, warning will be shown only once per library load,
    """
    ...
