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
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> float | npt.ArrayLike[np.float]:
    """
    Greenwich Mean Sidereal Time

    Vallado algorithm 15:

    GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * -6.2e-6)

    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is float or numpy array of floats with GMST in radians matched element-wise
    to the input times.
    """

def gast(
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> float | npt.ArrayLike[np.float]:
    """
    Greenwich Apparent Sidereal Time

    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is float or numpy array of floats with GAST in radians matched element-wise
    to the input times.
    """

def earth_rotation_angle(
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> float | npt.ArrayLike[np.float]:
    """
    Earth rotation angle

    See:
    https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf?__blob=publicationFile&v=1

    Equation 5.15

    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is float or numpy array of floats with Earth Rotation Angle in radians
    matched element-wise to the input times.
    """

def qitrf2tirs(
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Rotation from International Terrestrial Reference Frame
    (ITRF) to the Terrestrial Intermediate Reference System (TIRS)
    represented as satkit.quaterinion object

    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is satkit.quaternion or numpy array of satkit.quaternion representiong
    rotations from itrf to tirs matched element-wise to the input times
    """

@typing.overload
def qcirs2gcrf(tm: satkit.time) -> satkit.quaternion:
    """
    Rotate from Celestial Intermediate Reference System
    to Geocentric Celestial Reference Frame
    """

@typing.overload
def qcirs2gcrf(tm: npt.ArrayLike[satkit.time]) -> npt.ArrayLike[satkit.quaternion]:
    """
    Rotate from Celestial Intermediate Reference System
    to Geocentric Celestial Reference Frame
    """

def qtirs2cirs(
    tm: satkit.time | npt.ArrayLike[satkit.time],
) -> satkit.quaternion | npt.ArrayLike[satkit.quaternion]:
    """
    Rotation from Terrestrial Intermediate Reference System (TIRS)
    to the Celestial Intermediate Reference System (CIRS)


    Input is satkit.time object or list or numpy array of satkit.time objects.

    Output is satkit.quaternion or numpy array of satkit.quaternion representiong
    rotations from itrf to tirs matched element-wise to the input times
    """

def qgcrf2itrf_approx(
    tm: satkit.time | npt.ArrayLike[satkit.time],
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
    tm: satkit.time | npt.ArrayLike[satkit.time],
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
    tm: satkit.time | npt.ArrayLike[satkit.time],
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
    tm: satkit.time | npt.ArrayLike[satkit.time],
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
