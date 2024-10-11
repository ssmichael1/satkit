"""
Transformations between coordinate frames, and associated utility functions

Coordinate frame transforms are mostly pulled from Vallado:
https://www.google.com/books/edition/Fundamentals_of_satkitdynamics_and_Applic/PJLlWzMBKjkC?hl=en&gbpv=0

or the IERS:
https://www.iers.org/

"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

from .satkit import time, quaternion

import datetime

def gmst(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """Greenwich Mean Sidereal Time

    Notes:
        * GMST is the angle between the vernal equinox and the Greenwich meridian
        * Vallado algorithm 15
        * GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        float | npt.ArrayLike[np.float]: Greenwich Mean Sideral Time, radians, at intput time(s)
    """


def gast(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """ Greenwich Apparent Sideral Time

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        float | npt.ArrayLike[np.float]: Greenwich apparant sidereal time, radians, at input time(s)    
    """

def earth_rotation_angle(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """Earth Rotation Angle

    Notes:
        * See: IERS Technical Note 36, Chapter 5, Equation 5.15
        * Calculation Details:
            * Let t be UT1 Julian date
            * let f be fractional component of t (fraction of day)
            * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))

    Args:
        tm (satkit.time|datetime.datetime|list|numpy.array): Time[s] at which to calculate Earth Rotation Angle

    Returns:
        float|numpy.array: Earth Rotation Angle at input time[s] in radians
    """

def qitrf2tirs(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Rotation from Terrestrial Intermediate Reference System to Celestial Intermediate Reference Systems
    
    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to TIRS at input time(s)
    """

def qcirs2gcrf(tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
    ) -> quaternion | npt.ArrayLike[quaternion]:
    """ Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from CIRS to GCRF at input time(s)
    """


def qtirs2cirs(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion
    
    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from TIRS to CIRS at input time(s)
    """

def qgcrf2itrf_approx(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Accurate to approx. 1 arcsec

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion
        
    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """


def qitrf2gcrf_approx(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes: 
        * Accurate to approx. 1 arcsec
    
    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """


def qgcrf2itrf(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """

def qitrf2gcrf(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        * Uses full IAU2010 Reduction
        * See IERS Technical Note 36, Chapter 5
        * Does not include solid tides, ocean tides
        * Very computationally expensive
    
    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion
    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """


def qteme2itrf(
    tm: time | npt.ArrayLike[time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> quaternion | npt.ArrayLike[quaternion]:
    """ Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        * This is equation 3-90 in Vallado
        * TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        quaternion | npt.ArrayLike[quaternion]: Quaternion representing rotation from TEME to ITRF at input time(s)
    """


def earth_orientation_params(time: time) -> tuple[float, float, float, float] | None:
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
            
    """
