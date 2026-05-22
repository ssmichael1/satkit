"""
Transformations between coordinate frames, and associated utility functions

Coordinate frame transforms are mostly pulled from Vallado:
<https://www.google.com/books/edition/Fundamentals_of_Astrodynamics_and_Applic/PJLlWzMBKjkC?hl=en&gbpv=0>

or the IERS:
<https://www.iers.org/>

"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
import datetime

from .satkit import time, quaternion, frame

@typing.overload
def gmst(tm: time | datetime.datetime) -> float:
    """Greenwich Mean Sidereal Time

    Notes:
        - GMST is the angle between the vernal equinox and the Greenwich meridian
        - Vallado algorithm 15
        - GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

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
        - GMST is the angle between the vernal equinox and the Greenwich meridian
        - Vallado algorithm 15
        - GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        float | npt.ArrayLike[np.float]: Greenwich Mean Sidereal Time, radians, at input time(s)
    """
    ...

def gmst(*args, **kwargs):
    """Greenwich Mean Sidereal Time

    Notes:
        - GMST is the angle between the vernal equinox and the Greenwich meridian
        - Vallado algorithm 15
        - GMST = 67310.5481 + (876600h + 8640184.812866) * tᵤₜ₁ * (0.983104 + tᵤₜ₁ * −6.2e−6)

    Args:
        tm (satkit.time | datetime.datetime): scalar time at which to calculate output

    Returns:
        Greenwich Mean Sidereal Time, radians, at input time

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

def gast(*args, **kwargs):
    """Greenwich Apparent Sidereal Time

    Args:
        tm (satkit.time): scalar, list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        Greenwich apparent sidereal time, radians, at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        theta = satkit.frametransform.gast(t)
        ```
    """
    ...

@typing.overload
def earth_rotation_angle(
    tm: time | datetime.datetime,
) -> float:
    """Earth Rotation Angle

    Notes:
        - See: IERS Technical Note 36, Chapter 5, Equation 5.15
        - Calculation Details:
            - Let t be UT1 Julian date
            - let f be fractional component of t (fraction of day)
            - ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))

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
        - See: IERS Technical Note 36, Chapter 5, Equation 5.15
        - Calculation Details:
            - Let t be UT1 Julian date
            - let f be fractional component of t (fraction of day)
            - ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0)

    Args:
        tm (npt.ArrayLike[datetime.datetime] | npt.ArrayLike[time]): list, or numpy array of astro.time or datetime.datetime representing time at which to calculate output

    Returns:
        npt.ArrayLike[np.float]: Earth Rotation Angle at input time[s] in radians
    """
    ...

def earth_rotation_angle(*args, **kwargs):
    """Earth Rotation Angle

    Notes:
        - See: IERS Technical Note 36, Chapter 5, Equation 5.15
        - Calculation Details:
            - Let t be UT1 Julian date
            - let f be fractional component of t (fraction of day)
            - ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))

    Args:
        tm (satkit.time|datetime.datetime: Time[s] at which to calculate Earth Rotation Angle

    Returns:
        Earth Rotation Angle at input time[s] in radians

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        era = satkit.frametransform.earth_rotation_angle(t)
        ```
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

def qitrf2tirs(*args, **kwargs):
    """Rotation from Terrestrial Intermediate Reference System to Celestial Intermediate Reference Systems

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from ITRF to TIRS at input time(s)
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

def qteme2gcrf(*args, **kwargs):
    """Rotation from True Equator Mean Equinox (TEME) to Geocentric Celestial Reference Frame (GCRF)

    Args:
        tm (satkit.time| datetime.datetime ): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from TEME to GCRF at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qteme2gcrf(t)
        ```
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

def qcirs2gcrf(*args, **kwargs):
    """Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        tm (satkit.time | npt.ArrayLike[satkit.time] | datetime.datetime | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from CIRS to GCRF at input time(s)
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

def qtirs2cirs(*args, **kwargs):
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from TIRS to CIRS at input time(s)
    """
    ...

@typing.overload
def qgcrf2itrf_approx(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Accurate to approx. 1 arcsec
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between GCRF and ITRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`gcrf_to_itrf_state` / :func:`itrf_to_gcrf_state` for
          full state (position + velocity) transforms.

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
        - Accurate to approx. 1 arcsec

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

def qgcrf2itrf_approx(*args, **kwargs):
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Accurate to approx. 1 arcsec

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

@typing.overload
def qitrf2gcrf_approx(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Accurate to approx. 1 arcsec
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

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
        - Accurate to approx. 1 arcsec

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

def qitrf2gcrf_approx(*args, **kwargs):
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Accurate to approx. 1 arcsec

    Args:
        tm (satkit.time  | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

@typing.overload
def qgcrf2itrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

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
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from GCRF to ITRF at input time(s)
    """
    ...

def qgcrf2itrf(*args, **kwargs):
    """Quaternion representing rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from GCRF to ITRF at input time(s)

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
def qitrf2gcrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

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
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from ITRF to GCRF at input time(s)
    """
    ...

def qitrf2gcrf(*args, **kwargs):
    """Quaternion representing rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Uses full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation)
        - See IERS Technical Note 36, Chapter 5
        - Does not include solid tides, ocean tides
        - Very computationally expensive
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use :func:`itrf_to_gcrf_state` / :func:`gcrf_to_itrf_state` for
          full state (position + velocity) transforms.

    Args:
        tm (satkit.time  datetime.datetime): Time[s] at which to calculate the quaternion
    Returns:
        Quaternion representing rotation from ITRF to GCRF at input time(s)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qitrf2gcrf(t)
        ```
    """
    ...

@typing.overload
def qteme2itrf(
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - This is equation 3-90 in Vallado
        - TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

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
        - This is equation 3-90 in Vallado
        - TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        tm (npt.ArrayLike[satkit.time] | npt.ArrayLike[datetime.datetime]): Time[s] at which to calculate the quaternion

    Returns:
        npt.ArrayLike[quaternion]: Quaternion representing rotation from TEME to ITRF at input time(s)
    """
    ...

def qteme2itrf(*args, **kwargs):
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - This is equation 3-90 in Vallado
        - TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        tm (satkit.time | datetime.datetime): Time[s] at which to calculate the quaternion

    Returns:
        Quaternion representing rotation from TEME to ITRF at input time(s)

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
        - Returns None if the time is before the range of available EOP data
        - For times after the last available EOP data, the last entry's values are returned (constant extrapolation)
        - EOP data is available from 1962 to current, with predictions ~4 months ahead
        - See: <https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html>

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

@typing.overload
def qmod2gcrf(tm: time | datetime.datetime) -> quaternion:
    """Quaternion rotating Mean-of-Date (MOD) → GCRF at the given time.

    Mean-of-Date accounts for precession but not nutation. For the
    precession+nutation pair see :func:`qcirs2gcrf` or :func:`qitrf2gcrf`.
    """
    ...

@typing.overload
def qmod2gcrf(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Quaternion rotating Mean-of-Date (MOD) → GCRF at the given times."""
    ...

def qmod2gcrf(*args, **kwargs):
    """Quaternion rotating Mean-of-Date (MOD) → GCRF at the given time(s).

    Mean-of-Date accounts for precession but not nutation.
    """
    ...

@typing.overload
def qtod2mod_approx(tm: time | datetime.datetime) -> quaternion:
    """Approximate True-of-Date (TOD) → Mean-of-Date (MOD) rotation at the
    given time. Accounts for nutation only.
    """
    ...

@typing.overload
def qtod2mod_approx(
    tm: npt.ArrayLike | list[time] | list[datetime.datetime],
) -> npt.ArrayLike:
    """Approximate True-of-Date (TOD) → Mean-of-Date (MOD) rotation at
    the given times. Accounts for nutation only.
    """
    ...

def qtod2mod_approx(*args, **kwargs):
    """Approximate True-of-Date (TOD) → Mean-of-Date (MOD) rotation.

    Accounts for nutation only.
    """
    ...

def to_gcrf(
    frame: frame,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the 3x3 DCM that transforms a vector from a satellite-local
    orbital frame into GCRF at the current state.

    This is the unified dispatch for satellite-local orbital frames.
    Supported values:

    - ``frame.GCRF`` — returns the 3x3 identity matrix (trivial case)
    - ``frame.LVLH`` — Local Vertical / Local Horizontal
    - ``frame.RTN``  — Radial / In-track / Cross-track (= RSW = RTN)
    - ``frame.NTW``  — Normal-to-velocity / Tangent / Cross-track

    For an arbitrary frame-to-frame rotation, compose with
    :func:`from_gcrf`::

        # NTW -> RIC
        dcm = sk.frametransform.from_gcrf(sk.frame.RTN, pos, vel) @ \\
              sk.frametransform.to_gcrf(sk.frame.NTW, pos, vel)

    Args:
        frame: Source satellite-local frame
        pos: 3-element position vector in GCRF [m]
        vel: 3-element velocity vector in GCRF [m/s]

    Returns:
        numpy.ndarray: 3x3 rotation matrix (frame → GCRF)

    Raises:
        RuntimeError: if ``frame`` is not a satellite-local orbital frame.
            Earth-fixed / celestial frames (ITRF, TEME, EME2000, etc.) need
            a time argument for their rotation to GCRF and must use the
            dedicated quaternion helpers (:func:`qitrf2gcrf`,
            :func:`qteme2gcrf`, etc.) instead.

    Example:
        ```python
        import satkit as sk
        dcm = sk.frametransform.to_gcrf(sk.frame.NTW, pos_gcrf, vel_gcrf)
        v_gcrf = dcm @ v_ntw
        ```
    """
    ...

def itrf_to_gcrf_state(
    pos_itrf: npt.ArrayLike,
    vel_itrf: npt.ArrayLike,
    time: time | list[time] | npt.NDArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a satellite state (position + velocity) from ITRF to GCRF.

    Accepts either a single state (``pos``/``vel`` are length-3 vectors and
    ``time`` is a single ``satkit.time``) or a batch of ``N`` states
    (``pos``/``vel`` are shape ``(N, 3)`` arrays and ``time`` is a length-``N``
    array/list of times). The output shape matches the input.

    Unlike the raw :func:`qitrf2gcrf` quaternion, this function correctly
    handles the Earth-rotation contribution to velocity. A point at rest
    on Earth's surface has zero velocity in ITRF but ~465 m/s in GCRF at
    the equator, and this function accounts for that term.

    The IERS 2010 ITRF → GCRF reduction decomposes into three stages:
    polar motion (ITRF → TIRS), Earth rotation about the CIO polar axis
    (TIRS → CIRS), and precession-nutation (CIRS → GCRF). The
    Earth-rotation sweep term ``omega_earth x r`` is computed in
    **TIRS** — not ITRF or GCRF — because TIRS is defined such that
    Earth's rotation axis is exactly along its +z axis. Computing the
    sweep anywhere else would introduce either a polar-motion-sized
    error (~0.3 arcsec in ITRF) or a precession-sized error (tens of
    degrees in GCRF).

    Implementation:

    1. Rotate ``pos_itrf`` and ``vel_itrf`` into TIRS via polar motion.
    2. Add ``omega_earth x r_tirs`` to the velocity in TIRS, where
       ``omega_earth = (0, 0, OMEGA_EARTH)`` exactly.
    3. Rotate TIRS → CIRS → GCRF via the full IERS 2010 chain.

    Uses the full IERS 2010 reduction (polar motion + Earth rotation +
    precession-nutation with dX/dY corrections from Earth orientation
    parameters).

    Args:
        pos_itrf: 3-element position vector in ITRF [m]
        vel_itrf: 3-element velocity vector *as observed in ITRF* [m/s]
            (zero for a point at rest on Earth's surface)
        time: Epoch of the state

    Returns:
        A 2-tuple ``(pos_gcrf, vel_gcrf)`` of numpy arrays with the
        state expressed in GCRF.

    Example:
        ```python
        import satkit as sk
        import numpy as np

        # Geostationary satellite, stationary in ITRF
        t = sk.time(2024, 1, 1)
        pos_itrf = np.array([42164.17e3, 0.0, 0.0])
        vel_itrf = np.array([0.0, 0.0, 0.0])
        pos_gcrf, vel_gcrf = sk.frametransform.itrf_to_gcrf_state(
            pos_itrf, vel_itrf, t)
        # |vel_gcrf| ≈ 3075 m/s (the GEO orbital speed)
        ```
    """
    ...

def gcrf_to_itrf_state(
    pos_gcrf: npt.ArrayLike,
    vel_gcrf: npt.ArrayLike,
    time: time | list[time] | npt.NDArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a satellite state (position + velocity) from GCRF to ITRF.

    Inverse of :func:`itrf_to_gcrf_state`. Rotates the state through
    GCRF → CIRS → TIRS, subtracts the Earth-rotation ``omega_earth x r``
    term **in TIRS** (where Earth's rotation axis is exactly along +z),
    then applies inverse polar motion to reach ITRF. A geostationary
    satellite (whose GCRF velocity is pure orbital motion) produces
    zero velocity in ITRF. Uses the full IERS 2010 reduction.

    Accepts either a single state or a batch of ``N`` states: when
    ``pos``/``vel`` are shape ``(N, 3)`` arrays, ``time`` must be a
    length-``N`` array/list of times, and the returned arrays have
    shape ``(N, 3)``.

    Args:
        pos_gcrf: 3-element position vector in GCRF [m]
        vel_gcrf: 3-element velocity vector in GCRF [m/s]
        time: Epoch of the state

    Returns:
        A 2-tuple ``(pos_itrf, vel_itrf)`` where ``vel_itrf`` is the
        velocity as observed in ITRF.
    """
    ...

def itrf_to_gcrf_state_approx(
    pos_itrf: npt.ArrayLike,
    vel_itrf: npt.ArrayLike,
    time: time | list[time] | npt.NDArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Approximate ITRF → GCRF state transform using the IAU-76/FK5
    reduction (accurate to ~1 arcsec on position).

    Faster alternative to :func:`itrf_to_gcrf_state` when the full IERS
    2010 precision is not required. Neglects polar motion, so the
    Earth-rotation sweep ``omega_earth x r`` is evaluated in ITRF directly.
    Accepts scalar or batched inputs like :func:`itrf_to_gcrf_state`.
    """
    ...

def gcrf_to_itrf_state_approx(
    pos_gcrf: npt.ArrayLike,
    vel_gcrf: npt.ArrayLike,
    time: time | list[time] | npt.NDArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Approximate GCRF → ITRF state transform using the IAU-76/FK5
    reduction. Inverse of :func:`itrf_to_gcrf_state_approx`; accurate to
    ~1 arcsec on position. Accepts scalar or batched inputs.
    """
    ...

def from_gcrf(
    frame: frame,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the 3x3 DCM that transforms a vector from GCRF into a
    satellite-local orbital frame at the current state.

    Transpose of :func:`to_gcrf`. See that function for the list of
    supported frames, composition examples, and error conditions.

    Args:
        frame: Destination satellite-local frame
        pos: 3-element position vector in GCRF [m]
        vel: 3-element velocity vector in GCRF [m/s]

    Returns:
        numpy.ndarray: 3x3 rotation matrix (GCRF → frame)

    Raises:
        RuntimeError: if ``frame`` is not a satellite-local orbital frame.

    Example:
        ```python
        import satkit as sk
        dcm = sk.frametransform.from_gcrf(sk.frame.RTN, pos_gcrf, vel_gcrf)
        v_ric = dcm @ v_gcrf
        ```
    """
    ...

def disable_eop_time_warning() -> None:
    """Disable the warning printed to stderr when Earth Orientation Parameters (EOP) are not available for a given time.

    Notes:
        - This function is used to disable the warning printed when EOP are not available for a given time.
        - If not disabled, warning will be shown only once per library load,
    """
    ...

# ── Frame-enum dispatch (new in 0.17.0) ─────────────────────────────────

def rotation(
    from_frame: frame,
    to_frame: frame,
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion rotating a vector from ``from_frame`` to ``to_frame`` at
    ``tm``. Full IERS 2010 reduction.

    Uses the shortest path through the frame graph for each pair (does not
    always pivot through GCRF). Pairs involving orbit-dependent frames
    (``LVLH``, ``RTN``, ``NTW``) require state and are not supported here —
    use :func:`to_gcrf` / :func:`from_gcrf` for those.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        tm: Epoch

    Returns:
        Rotation from ``from_frame`` to ``to_frame`` at ``tm``.

    Raises:
        RuntimeError: if the pair involves LVLH / RTN / NTW.
    """
    ...

def rotation_approx(
    from_frame: frame,
    to_frame: frame,
    tm: time | datetime.datetime,
) -> quaternion:
    """Quaternion rotating a vector from ``from_frame`` to ``to_frame`` using
    the IAU-76/FK5 approximate reduction (~1 arcsec).

    Only valid between ``ITRF`` and the inertial cluster (``GCRF``,
    ``EME2000``, ``ICRF``, ``TEME``). ``TIRS`` and ``CIRS`` are defined by
    the IERS 2010 reduction and have no FK5 analogue.

    Raises:
        RuntimeError: if either frame is ``TIRS`` / ``CIRS``, or if the pair
            involves orbit-dependent frames.
    """
    ...

def transform_state(
    from_frame: frame,
    to_frame: frame,
    tm: time | datetime.datetime,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """State (position + velocity) transform from ``from_frame`` to
    ``to_frame`` at ``tm``. Properly handles the Earth-rotation sweep term
    when transitioning between rotating (ITRF) and inertial frames.

    Currently supported pairs: identity, ``ITRF``↔{``GCRF``, ``EME2000``,
    ``ICRF``, ``TEME``}, and within-inertial pairs. Other pairs raise
    ``RuntimeError`` in this version.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        tm: Epoch
        pos: 3-element position vector [m]
        vel: 3-element velocity vector [m/s]

    Returns:
        ``(pos, vel)`` in ``to_frame``.
    """
    ...

def transform_state_approx(
    from_frame: frame,
    to_frame: frame,
    tm: time | datetime.datetime,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """State transform using the IAU-76/FK5 approximate reduction. Same
    supported-pair set as :func:`transform_state`.
    """
    ...
