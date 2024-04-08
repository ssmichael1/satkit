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

import satkit
import datetime

def gmst(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """
    Greenwich Mean Sidereal Time
    
    Vallado algorithm 15:
    
    GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)
    
    
    # Arguments
    
      * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
              representing time at which to calculate output
    
    # Returns
    
    * Greenwich Mean Sideral Time, radians, at intput time(s)
    
    """

def gast(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """
    Greenwich apparant sidereal time, radians
    
    # Arguments:
    
      * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
              representing time at which to calculate output
    
    # Returns:
    
     * Greenwich apparant sidereal time, radians, at input time(s)
    
    """

def earth_rotation_angle(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> float | npt.ArrayLike[np.float]:
    """
    Earth Rotation Angle
    
    See
    [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf?__blob=publicationFile&v=1)
    Equation 5.15
    
    # Arguments:
    
     * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
             representing time at which to calculate output
    
    # Returns:
    
     * Earth rotation angle, in radians, at input time(s)
    
    # Calculation Details
    
    * Let t be UT1 Julian date
    * let f be fractional component of t (fraction of day)
    * ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t − 2451545.0))
    
    """

def qitrf2tirs(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Rotation from Terrestrial Intermediate Reference System to
    Celestial Intermediate Reference Systems
    
    # Arguments:
    
     * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
             representing time at which to calculate output
    
    # Returns:
    
     * Quaternion representing rotation from TIRS to CIRS at input time(s)
    
    """

@typing.overload
def qcirs2gcrf(tm: satkit.time | datetime.datetime) -> satkit.quaternion:
    """  
    Rotation from Celestial Intermediate Reference System
    to Geocentric Celestial Reference Frame
    
    # Arguments:
    
     * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
             representing time at which to calculate output
    
    # Returns:
    
     * Quaternion representing rotation from CIRS to GCRF at input time(s)
    
    """

@typing.overload
def qcirs2gcrf(tm: npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]) -> npt.ArrayLike[satkit.quaternion]:
    """
    Rotation from Celestial Intermediate Reference System
    to Geocentric Celestial Reference Frame
    
    # Arguments:
    
     * `tm`: scalar, list, or numpy array of astro.time or datetime.datetime 
             representing time at which to calculate output
    
    # Returns:
    
     * Quaternion representing rotation from CIRS to GCRF at input time(s)
    
    """

def qtirs2cirs(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Rotation from Terrestrial Intermediate Reference System (TIRS)
    to the Celestial Intermediate Reference System (CIRS)


    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is satkit.quaternion or numpy array of satkit.quaternion representiong
    rotations from itrf to tirs matched element-wise to the input times
    """

def qgcrf2itrf_approx(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Quaternion representing approximate rotation from the
    Geocentric Celestial Reference Frame (GCRF)
    to the International Terrestrial Reference Frame (ITRF)

    # Notes:

    * Accurate to approx. 1 arcsec

    # Arguments:

    * `tm` - satkit.time object or list or numpy array of satkit.time objects.

    # Outputs:

        * satkit.quaternion or numpy array of satkit.quaternion representiong
        rotations from gcrf to itrf matched element-wise to the input times
    """

def qitrf2gcrf_approx(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Quaternion representing approximate rotation from the
    International Terrestrial Reference Frame (ITRF)
    to the Geocentric Celestial Reference Frame (GCRF)

    # Notes:

    * Accurate to approx. 1 arcsec

    # Arguments:

    * `tm` - satkit.time object or list or numpy array of satkit.time objects.

    # Outputs:

        * satkit.quaternion or numpy array of satkit.quaternion representiong
        rotations from itrf to gcrf matched element-wise to the input times
    """

def qgcrf2itrf(
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Quaternion representing rotation from the
    Geocentric Celestial Reference Frame (GCRF)
    to the International Terrestrial Reference Frame (ITRF)

    Uses full IAU2006 Reduction
    See IERS Technical Note 36, Chapter 5

    but does not include solid tides, ocean tides

    Note: Very computationally expensive

    # Arguments:

    * `tm` - satkit.time object or list or numpy array of satkit.time objects.

    # Outputs:

        * satkit.quaternion or numpy array of satkit.quaternion representiong
        rotations from gcrf to itrf matched element-wise to the input times
    """

def qitrf2gcrf(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Quaternion representing rotation from the
    International Terrestrial Reference Frame (ITRF)
    to the Geocentric Celestial Reference Frame (GCRF)

    Uses full IAU2006 Reduction
    See IERS Technical Note 36, Chapter 5

    but does not include solid tides, ocean tides

    Note: Very computationally expensive

    # Arguments:

    * `tm` - satkit.time object or list or numpy array of satkit.time objects.

    # Outputs:

        * satkit.quaternion or numpy array of satkit.quaternion representiong
        rotations from itrf to gcrf matched element-wise to the input times
    """

def qteme2itrf(
    tm: satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Quaternion representing rotation from the
    True Equator Mean Equinox (TEME) frame
    to the International Terrestrial Reference Frame (ITRF)

    This is equation 3-90 in Vallado

    Note: TEME is the output frame of the SGP4 propagator used to
    compute position from two-line element sets.

    # Arguments:

    * `tm` - satkit.time object or list or numpy array of satkit.time objects.

    # Outputs:

        * satkit.quaternion or numpy array of satkit.quaternion representiong
        rotations from teme to itrf matched element-wise to the input times
    """

def earth_orientation_params(time: satkit.time) -> tuple[float, float, float, float] | None:
    """
    Get Earth Orientation Parameters at given instant

    Arguments:
        time: Instant at which to query parameters

    Return:
        Tuple with following elements:
            0 : (UT1 - UTC) in seconds
            1 : X polar motion in arcsecs
            2 : Y polar motion in arcsecs
            3 : LOD: instantaneous rate of change in (UT1-UTC), msec/day
            4 : dX wrt IAU-2000A nutation, milli-arcsecs
            5 : dY wrt IAU-2000A nutation, milli-arcsecs

    If time is out of bounds, None is returned
    """
