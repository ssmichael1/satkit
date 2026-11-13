"""
Transformations between coordinate frames, and associated utility functions

Coordinate frame transforms are mostly pulled from Vallado:
<https://www.google.com/books/edition/Fundamentals_of_Astrodynamics_and_Applic/PJLlWzMBKjkC?hl=en&gbpv=0>

or the IERS:
<https://www.iers.org/>

Time arguments
--------------
A single time is a ``satkit.time``, a ``datetime.datetime`` or a
``numpy.datetime64``. A *naive* ``datetime.datetime`` (no ``tzinfo``) is taken
as **local time**, as Python's own ``datetime.timestamp()`` does; pass an aware
datetime (``tzinfo=datetime.timezone.utc``) or a ``satkit.time`` for UTC.
``numpy.datetime64`` values are UTC. Functions that take several times accept a
list, a tuple, or a 1-D numpy array of ``datetime64`` or of time objects.

The time parameter is named ``time``. Its old keyword spelling ``tm=`` still
works but emits a ``DeprecationWarning``; positional calls are unaffected.
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np
import datetime
from typing_extensions import deprecated

from .satkit import time, quaternion, frame, TimeScalar, TimeArrayLike, TimeInput

@typing.overload
def gmst(time: TimeScalar) -> float:
    """Greenwich Mean Sidereal Time

    Notes:
        - GMST is the angle between the vernal equinox and the Greenwich meridian
        - Vallado algorithm 15
        - GMST = 67310.54841 + (876600ʰ + 8640184.812866) tᵤₜ₁ + 0.093104 tᵤₜ₁² − 6.2e−6 tᵤₜ₁³ (seconds of time; tᵤₜ₁ = Julian centuries of UT1 from J2000.0)

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): scalar time at which to calculate output

    Returns:
        float: Greenwich Mean Sidereal Time, radians, at input time. Not normalised to
        [0, 2π): the value lies in (-2π, 2π) and is negative for most epochs
        before J2000 (1990-01-01 gives -4.53); use ``gmst(t) % (2 * math.pi)``
        for an angle in [0, 2π)

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
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """Greenwich Mean Sidereal Time

    Notes:
        - GMST is the angle between the vernal equinox and the Greenwich meridian
        - Vallado algorithm 15
        - GMST = 67310.54841 + (876600ʰ + 8640184.812866) tᵤₜ₁ + 0.093104 tᵤₜ₁² − 6.2e−6 tᵤₜ₁³ (seconds of time; tᵤₜ₁ = Julian centuries of UT1 from J2000.0)

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate output

    Returns:
        npt.NDArray[np.float64]: Greenwich Mean Sidereal Time, radians, one per input
        time, as a 1-D array (a list up to 0.24.3). Not normalised to [0, 2π); see the
        scalar overload
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def gmst(
    *,
    tm: TimeInput,
) -> float | npt.NDArray[np.float64]: ...

@typing.overload
def eqeq(time: TimeScalar) -> float:
    """Equation of the Equinoxes

    The equation of the equinoxes is the difference between apparent and mean
    sidereal time (GAST - GMST), arising from nutation of the Earth's axis.

    Notes:
        - Two-term approximation, dPsi cos(eps) with dPsi = -17.2" sin(Omega)
          - 1.3" sin(2L) (Vallado 2013, §3.7.3). Against the IAU 1994
          equation of the equinoxes with the full IAU 1980 nutation (ERFA
          ``eqeq94``) it is good to about 0.6" (0.65" max, 43 ms of time,
          over 1950-2100); use ``rotation`` for the full reduction.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): scalar time at which to calculate output

    Returns:
        float: Equation of the equinoxes, radians, at input time
    """
    ...

@typing.overload
def eqeq(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """Equation of the Equinoxes

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate output

    Returns:
        npt.NDArray[np.float64]: Equation of the equinoxes, radians, one per input
        time, as a 1-D array (a list up to 0.24.3)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def eqeq(
    *,
    tm: TimeInput,
) -> float | npt.NDArray[np.float64]: ...

@typing.overload
def gast(
    time: TimeScalar,
) -> float:
    """Greenwich Apparent Sidereal Time

    GMST (IAU 1982) plus the two-term equation of the equinoxes
    (``eqeq``), so good to about 0.6" (0.65" max, 43 ms of time,
    over 1950-2100) against ERFA ``gst94``.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): scalar time at which to calculate output

    Returns:
        float: Greenwich apparent sidereal time, radians, at input time. Like ``gmst``
        it is not normalised to [0, 2π): it is negative for most epochs before
        J2000

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        theta = satkit.frametransform.gast(t)
        ```
    """
    ...

@typing.overload
def gast(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """Greenwich Apparent Sidereal Time

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate output

    Returns:
        npt.NDArray[np.float64]: Greenwich apparent sidereal time, radians, one per
        input time, as a 1-D array (a list up to 0.24.3). Not normalised to [0, 2π)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def gast(
    *,
    tm: TimeInput,
) -> float | npt.NDArray[np.float64]: ...

@typing.overload
def earth_rotation_angle(
    time: TimeScalar,
) -> float:
    """Earth Rotation Angle

    Notes:
        - See: IERS Technical Note 36, Chapter 5, Equation 5.15
        - Calculation Details:
            - Let t be UT1 Julian date
            - let f be fractional component of t (fraction of day)
            - ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0))

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate Earth Rotation Angle

    Returns:
        float: Earth Rotation Angle at the input time, radians, in [0, 2π)

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        era = satkit.frametransform.earth_rotation_angle(t)
        ```
    """
    ...

@typing.overload
def earth_rotation_angle(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """Earth Rotation Angle

    Notes:
        - See: IERS Technical Note 36, Chapter 5, Equation 5.15
        - Calculation Details:
            - Let t be UT1 Julian date
            - let f be fractional component of t (fraction of day)
            - ERA = 2𝜋 ((0.7790572732640 + f + 0.00273781191135448 * (t - 2451545.0)

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate output

    Returns:
        npt.NDArray[np.float64]: Earth Rotation Angle, radians, in [0, 2π), one per
        input time, as a 1-D array (a list up to 0.24.3)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def earth_rotation_angle(
    *,
    tm: TimeInput,
) -> float | npt.NDArray[np.float64]: ...

@typing.overload
def qitrf2tirs(
    time: TimeScalar,
) -> quaternion:
    """Rotation from the ITRF to the Terrestrial Intermediate Reference System (polar motion)

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from ITRF to TIRS at the input time
    """
    ...

@typing.overload
def qitrf2tirs(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Rotation from the ITRF to the Terrestrial Intermediate Reference System (polar motion)

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from ITRF to TIRS, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qitrf2tirs(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qteme2gcrf(
    time: TimeScalar,
) -> quaternion:
    """Rotation from True Equator Mean Equinox (TEME) to Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - **Approximate**: the same as ``rotation_approx(TEME, GCRF)``, not
          ``rotation(TEME, GCRF)``. TEME (a quasi-inertial frame) is rotated
          to PEF by GMST82 alone, then to GCRF by the approximate chain of
          ``qitrf2gcrf_approx`` (two-term equation of the equinoxes and
          nutation, IAU 2006 precession, no frame bias). No polar motion is
          involved. Accurate to 0.55" max against the full IERS 2010
          reduction (1973-2026): ~19 m at LEO, ~110 m at GEO.
        - For the full reduction use ``rotation(TEME, GCRF)`` (GMST82, polar
          motion, then the IERS 2010 ITRF -> GCRF chain; matches ERFA to
          ~5 uas), at a higher cost.
        - TEME is the output frame of the SGP4 propagator

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from TEME to GCRF at the input time

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qteme2gcrf(t)
        ```
    """
    ...

@typing.overload
def qteme2gcrf(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Rotation from True Equator Mean Equinox (TEME) to Geocentric Celestial Reference Frame (GCRF)

    Approximate (0.55" max); see the scalar overload.

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from TEME to GCRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qteme2gcrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qcirs2gcrf(
    time: TimeScalar,
) -> quaternion:
    """Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from CIRS to GCRF at the input time
    """
    ...

@typing.overload
def qcirs2gcrf(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Rotation from Celestial Intermediate Reference System to Geocentric Celestial Reference Frame

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from CIRS to GCRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qcirs2gcrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qtirs2cirs(
    time: TimeScalar,
) -> quaternion:
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from TIRS to CIRS at the input time
    """
    ...

@typing.overload
def qtirs2cirs(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Rotation from Terrestrial Intermediate Reference System (TIRS) to the Celestial Intermediate Reference System (CIRS)

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from TIRS to CIRS, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qtirs2cirs(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qgcrf2itrf_approx(
    time: TimeScalar,
) -> quaternion:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Accurate to about 1 arcsec (1.0" max against the full IERS 2010
          reduction, 1973-2026; ~35 m at LEO), of which up to 0.6" is polar
          motion, which this chain neglects
        - The chain is GAST (GMST82 + the two-term ``eqeq``), the
          two-term nutation of ``qtod2mod_approx`` and the IAU 2006
          precession of ``qmod2gcrf`` (no frame bias). It is often
          labelled "IAU-76/FK5", but it is neither the IAU 1976 precession
          nor the 106-term IAU 1980 nutation series
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between GCRF and ITRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use ``gcrf_to_itrf_state`` / ``itrf_to_gcrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from GCRF to ITRF at the input time
    """
    ...

@typing.overload
def qgcrf2itrf_approx(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Quaternion representing approximate rotation from the Geocentric Celestial Reference Frame (GCRF) to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - Accurate to about 1 arcsec; see the scalar overload

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from GCRF to ITRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qgcrf2itrf_approx(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qitrf2gcrf_approx(
    time: TimeScalar,
) -> quaternion:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Accurate to about 1 arcsec (1.0" max against the full IERS 2010
          reduction, 1973-2026; ~35 m at LEO), of which up to 0.6" is polar
          motion, which this chain neglects
        - The chain is GAST (GMST82 + the two-term ``eqeq``), the
          two-term nutation of ``qtod2mod_approx`` and the IAU 2006
          precession of ``qmod2gcrf`` (no frame bias). It is often
          labelled "IAU-76/FK5", but it is neither the IAU 1976 precession
          nor the 106-term IAU 1980 nutation series
        - **Velocity transforms**: this quaternion rotates *position* vectors
          between ITRF and GCRF but **is not sufficient for velocity** on
          its own. ITRF is a rotating frame, so the velocity transform
          picks up an extra ``omega_earth x r`` term (~470 m/s at LEO).
          Use ``itrf_to_gcrf_state`` / ``gcrf_to_itrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from ITRF to GCRF at the input time
    """
    ...

@typing.overload
def qitrf2gcrf_approx(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Quaternion representing approximate rotation from the International Terrestrial Reference Frame (ITRF) to the Geocentric Celestial Reference Frame (GCRF)

    Notes:
        - Accurate to about 1 arcsec; see the scalar overload

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from ITRF to GCRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qitrf2gcrf_approx(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qgcrf2itrf(
    time: TimeScalar,
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
          Use ``itrf_to_gcrf_state`` / ``gcrf_to_itrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from GCRF to ITRF at the input time

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
    time: TimeArrayLike,
) -> list[quaternion]:
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
          Use ``itrf_to_gcrf_state`` / ``gcrf_to_itrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from GCRF to ITRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qgcrf2itrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qitrf2gcrf(
    time: TimeScalar,
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
          Use ``itrf_to_gcrf_state`` / ``gcrf_to_itrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion
    Returns:
        quaternion: Quaternion representing rotation from ITRF to GCRF at the input time

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        q = satkit.frametransform.qitrf2gcrf(t)
        ```
    """
    ...

@typing.overload
def qitrf2gcrf(
    time: TimeArrayLike,
) -> list[quaternion]:
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
          Use ``itrf_to_gcrf_state`` / ``gcrf_to_itrf_state`` for
          full state (position + velocity) transforms.

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from ITRF to GCRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qitrf2gcrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qteme2itrf(
    time: TimeScalar,
) -> quaternion:
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - This is equation 3-90 in Vallado: the GMST (IAU 1982) rotation
          TEME -> PEF followed by polar motion PEF -> ITRF. No
          precession-nutation is involved, so it is exact to the model (it
          matches ERFA to ~10 uas) and identical to ``rotation(TEME, ITRF)``
          and ``rotation_approx(TEME, ITRF)``.
        - TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to calculate the quaternion

    Returns:
        quaternion: Quaternion representing rotation from TEME to ITRF at the input time

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
    time: TimeArrayLike,
) -> list[quaternion]:
    """Quaternion representing rotation from the True Equator Mean Equinox (TEME) frame to the International Terrestrial Reference Frame (ITRF)

    Notes:
        - This is equation 3-90 in Vallado: the GMST (IAU 1982) rotation
          TEME -> PEF followed by polar motion PEF -> ITRF. No
          precession-nutation is involved, so it is exact to the model (it
          matches ERFA to ~10 uas) and identical to ``rotation(TEME, ITRF)``
          and ``rotation_approx(TEME, ITRF)``.
        - TEME is the output frame of the SGP4 propagator used to compute position from two-line element sets.

    Args:
        time (list | tuple | numpy.ndarray): list, tuple or 1-D numpy array of times (satkit.time, datetime.datetime or numpy.datetime64) at which to calculate the quaternions

    Returns:
        list[quaternion]: Quaternions representing rotation from TEME to ITRF, one per input time
        (a one-element list for a one-element input)
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qteme2itrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

def earth_orientation_params(
    time: TimeScalar,
) -> tuple[float, float, float, float, float, float] | None:
    """Get Earth Orientation Parameters at given instant

    ``satkit.eop`` is the home for EOP queries: this is the same as
    ``satkit.eop.get``.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Instant at which to
            query parameters (a single time; a list, even of one element, raises
            ``TypeError``)

    Returns:
        (float, float, float, float, float, float) | None: Tuple with following elements:
            0 : (UT1 - UTC) in seconds
            1 : X polar motion in arcsecs
            2 : Y polar motion in arcsecs
            3 : LOD: excess length of day, -d(UT1-UTC)/dt, seconds per day
            4 : dX wrt IAU-2000A nutation, milli-arcsecs
            5 : dY wrt IAU-2000A nutation, milli-arcsecs

    Notes:
        - Returns None if the time is before the range of available EOP data, or if no EOP table is loaded
        - For times after the last available EOP data, the last entry's values are returned (constant
          extrapolation) and a one-time warning is logged; use ``eop_status`` / ``eop_coverage`` to check
        - EOP data is available from 1973-01-02 (IERS ``finals2000A.all``) to current, with
          predictions up to a year ahead; refresh with ``satkit.utils.update_datafiles()``
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
def qmod2gcrf(time: TimeScalar) -> quaternion:
    """Quaternion rotating Mean-of-Date (MOD) → GCRF at the given time.

    Mean-of-Date accounts for precession but not nutation. For the
    precession+nutation pair see ``qcirs2gcrf`` or ``qitrf2gcrf``.

    Notes:
        - Precession only: the IAU 2006 angles zeta_A, z_A, theta_A
          (Capitaine et al. 2003; Vallado Eqs. 3-88, 3-89), not the IAU 1976
          precession. It matches the precession matrix of ERFA ``bp06`` to
          0.1 uas.
        - No frame bias: the target is the J2000 mean equator and equinox
          (EME2000), which differs from GCRF by the constant 23 mas frame
          bias (0.8 m at LEO). Compose with
          ``rotation(EME2000, GCRF)`` for a true MOD -> GCRF.
    """
    ...

@typing.overload
def qmod2gcrf(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Quaternion rotating Mean-of-Date (MOD) → GCRF at the given times."""
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qmod2gcrf(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def qtod2mod_approx(time: TimeScalar) -> quaternion:
    """Approximate True-of-Date (TOD) → Mean-of-Date (MOD) rotation at the
    given time. Accounts for nutation only.

    Notes:
        - Two-term nutation (the 18.6-year and semi-annual terms; Vallado
          2013, §3.7.3), good to 0.9" (0.88" max over 1950-2100) against the
          IAU 2006/2000A nutation (ERFA ``nut06a``) and equally against the
          IAU 1980 series
    """
    ...

@typing.overload
def qtod2mod_approx(
    time: TimeArrayLike,
) -> list[quaternion]:
    """Approximate True-of-Date (TOD) → Mean-of-Date (MOD) rotation at
    the given times. Accounts for nutation only.
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def qtod2mod_approx(
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

def to_gcrf(
    frame: frame,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the 3x3 DCM that transforms a vector from a satellite-local
    orbital frame (or an inertial frame) into GCRF at the current state.

    This is the unified dispatch for satellite-local orbital frames.
    Supported values:

    - ``frame.GCRF`` — returns the 3x3 identity matrix (trivial case)
    - ``frame.LVLH`` — Local Vertical / Local Horizontal
    - ``frame.RTN``  — Radial / In-track / Cross-track (= RSW = RIC)
    - ``frame.NTW``  — Normal-to-velocity / Tangent / Cross-track
    - ``frame.ICRF`` — the 3x3 identity (GCRF has the ICRF axes)
    - ``frame.EME2000`` — the constant IERS 2010 frame bias (about 23 mas),
      the same rotation as ``rotation(frame.EME2000, frame.GCRF, time)`` at
      any time

    ``pos`` and ``vel`` are used only by ``LVLH``, ``RTN`` and ``NTW``.

    For an arbitrary frame-to-frame rotation, compose with
    ``from_gcrf``::

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
        RuntimeError: if ``frame`` is time-dependent (``ITRF``, ``TIRS``,
            ``CIRS``, ``TEME``): use ``rotation(frame, satkit.frame.GCRF, time)``
            (or ``rotation_with_state``) for those.
        ValueError: if ``pos`` or ``vel`` is not a 3-element vector, holds a
            non-finite value, or (for ``LVLH`` / ``RTN`` / ``NTW``) the two do
            not define an orbit plane: either is zero or they are parallel.
        TypeError: if ``pos`` or ``vel`` is not a real numeric array-like
            (e.g. a complex array).

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
    time: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a satellite state (position + velocity) from ITRF to GCRF.

    Accepts either a single state (``pos``/``vel`` are length-3 vectors and
    ``time`` is a single ``satkit.time``) or a batch of ``N`` states
    (``pos``/``vel`` are shape ``(N, 3)`` arrays and ``time`` is a length-``N``
    array/list of times). The output shape matches the input.

    Unlike the raw ``qitrf2gcrf`` quaternion, this function correctly
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
        time: Epoch of the state: a single time for a single state (a
            one-element list is not a single time and raises ``TypeError``),
            or a length-``N`` list / array of times for a batch

    Returns:
        A 2-tuple ``(pos_gcrf, vel_gcrf)`` of numpy arrays with the
        state expressed in GCRF: position in meters, velocity in m/s
        (shape ``(3,)`` each, or ``(N, 3)`` for batched input).

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
    time: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a satellite state (position + velocity) from GCRF to ITRF.

    Inverse of ``itrf_to_gcrf_state``. Rotates the state through
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
        time: Epoch of the state: a single time for a single state, or a
            length-``N`` list / array of times for a batch

    Returns:
        A 2-tuple ``(pos_itrf, vel_itrf)`` where ``vel_itrf`` is the
        velocity as observed in ITRF: position in meters, velocity in m/s
        (shape ``(3,)`` each, or ``(N, 3)`` for batched input).
    """
    ...

def itrf_to_gcrf_state_approx(
    pos_itrf: npt.ArrayLike,
    vel_itrf: npt.ArrayLike,
    time: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Approximate ITRF → GCRF state transform using the approximate
    reduction of ``qitrf2gcrf_approx`` (accurate to ~1 arcsec on position).

    Faster alternative to ``itrf_to_gcrf_state`` when the full IERS
    2010 precision is not required. Neglects polar motion, so the
    Earth-rotation sweep ``omega_earth x r`` is evaluated in ITRF directly.
    Accepts scalar or batched inputs like ``itrf_to_gcrf_state``.

    Args:
        pos_itrf: ``(3,)`` or ``(N, 3)`` position vector in ITRF, meters
        vel_itrf: ``(3,)`` or ``(N, 3)`` velocity vector as observed in ITRF, m/s
        time: Epoch of the state (length-``N`` array/list for batched input)

    Returns:
        A 2-tuple ``(pos_gcrf, vel_gcrf)``: position in meters, velocity in m/s
    """
    ...

def gcrf_to_itrf_state_approx(
    pos_gcrf: npt.ArrayLike,
    vel_gcrf: npt.ArrayLike,
    time: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Approximate GCRF → ITRF state transform using the approximate
    reduction of ``qgcrf2itrf_approx``. Inverse of ``itrf_to_gcrf_state_approx``; accurate to
    ~1 arcsec on position. Accepts scalar or batched inputs.

    Args:
        pos_gcrf: ``(3,)`` or ``(N, 3)`` position vector in GCRF, meters
        vel_gcrf: ``(3,)`` or ``(N, 3)`` velocity vector in GCRF, m/s
        time: Epoch of the state (length-``N`` array/list for batched input)

    Returns:
        A 2-tuple ``(pos_itrf, vel_itrf)``: position in meters, velocity
        as observed in ITRF in m/s
    """
    ...

def from_gcrf(
    frame: frame,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the 3x3 DCM that transforms a vector from GCRF into a
    satellite-local orbital frame at the current state.

    Transpose of ``to_gcrf``. See that function for the list of
    supported frames, composition examples, and error conditions.

    Args:
        frame: Destination satellite-local frame
        pos: 3-element position vector in GCRF [m]
        vel: 3-element velocity vector in GCRF [m/s]

    Returns:
        numpy.ndarray: 3x3 rotation matrix (GCRF → frame)

    Raises:
        RuntimeError: if ``frame`` is time-dependent (``ITRF``, ``TIRS``,
            ``CIRS``, ``TEME``).
        ValueError: if ``pos`` or ``vel`` is not a 3-element vector, holds a
            non-finite value, or does not define an orbit plane (see ``to_gcrf``).
        TypeError: if ``pos`` or ``vel`` is not a real numeric array-like.

    Example:
        ```python
        import satkit as sk
        dcm = sk.frametransform.from_gcrf(sk.frame.RTN, pos_gcrf, vel_gcrf)
        v_ric = dcm @ v_gcrf
        ```
    """
    ...

def disable_eop_time_warning() -> None:
    """Disable the warnings about Earth Orientation Parameters (EOP) availability.

    ``satkit.eop`` is the home for EOP queries (``get``, ``coverage``,
    ``status``, ``update``, ``init_from_path``); this switch stays here.

    Notes:
        - Four one-time warnings exist: epoch before the EOP table, epoch after the table end
          (last values held constant), epoch in the predictions of a table not refreshed for
          over 30 days, and no EOP table loaded at all (zeros used).
        - Each is shown at most once per process; this call suppresses all of them.
        - They are logged to the ``satkit.earth_orientation_params`` logger; silencing that
          logger (or ``satkit``) with ``logging`` works too.
    """
    ...

def eop_coverage() -> tuple[time, time, time] | None:
    """Time bounds of the loaded Earth Orientation Parameters (EOP) table.

    ``satkit.eop`` is the home for EOP queries: this is the same as
    ``satkit.eop.coverage``.

    Returns:
        (satkit.time, satkit.time, satkit.time) | None: ``(first, last_observed, last)`` — the first
        row, the last *observed* row (rows after it are IERS predictions), and the last row of the
        table; ``None`` if no EOP table is loaded. Epochs after ``last`` use that row's values held
        constant (see ``eop_status``); refresh with ``satkit.utils.update_datafiles()``.

    Example:
        >>> first, last_observed, last = satkit.frametransform.eop_coverage()
    """
    ...

@deprecated("the EOP table is always finals2000A.all; use eop_coverage()")
def eop_source() -> str | None:
    """Deprecated since 0.24, removed in 0.25: the EOP table is always IERS
    ``finals2000A.all``. Use ``eop_coverage`` to check whether a table is loaded.

    Returns:
        str | None: ``"finals2000A"`` when a table is loaded, ``None`` otherwise.
    """

@typing.overload
def eop_status(
    time: TimeScalar,
) -> typing.Literal[
    "observed", "predicted", "extrapolated", "before_table", "not_loaded"
]:
    """Classify an epoch against the loaded Earth Orientation Parameters (EOP) table.

    ``satkit.eop`` is the home for EOP queries: this is the same as
    ``satkit.eop.status``.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): Epoch to classify (a
            single time; a list, even of one element, raises ``TypeError``). The
            keyword ``tm=`` is a deprecated alias of ``time=``.

    Returns:
        str: one of ``"observed"`` (inside the table, on or before the last observed row),
        ``"predicted"`` (inside the table, IERS prediction), ``"extrapolated"`` (after the table
        end — the last row is held constant; accuracy degrades by ~0.1 arcsec / ~10 ms per few
        months, refresh the data files), ``"before_table"`` (before the table's first row — 1973-01-02 for
        ``finals2000A.all`` — zeros used, so UT1 = UTC), or
        ``"not_loaded"`` (no EOP table loaded; zeros used).

    Raises:
        TypeError: if ``time`` is not a single time
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def eop_status(
    *,
    tm: TimeScalar,
) -> typing.Literal[
    "observed", "predicted", "extrapolated", "before_table", "not_loaded"
]: ...

# ── Frame-enum dispatch (new in 0.17.0) ─────────────────────────────────

@typing.overload
def rotation(
    from_frame: frame,
    to_frame: frame,
    time: TimeScalar,
) -> quaternion:
    """Quaternion rotating a vector from ``from_frame`` to ``to_frame`` at
    ``time``. Full IERS 2010 reduction.

    Uses the shortest path through the frame graph for each pair (does not
    always pivot through GCRF). Pairs involving orbit-dependent frames
    (``LVLH``, ``RTN``, ``NTW``) require state and are not supported here —
    use ``to_gcrf`` / ``from_gcrf`` for those.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        time: Epoch

    Returns:
        Rotation from ``from_frame`` to ``to_frame`` at ``time``.

    Raises:
        RuntimeError: if the pair involves LVLH / RTN / NTW.
    """
    ...

@typing.overload
def rotation(
    from_frame: frame,
    to_frame: frame,
    time: TimeArrayLike,
) -> list[quaternion]:
    """Quaternions rotating a vector from ``from_frame`` to ``to_frame`` at
    each of the times ``time``. Full IERS 2010 reduction; see the scalar
    overload.

    Returns:
        One rotation per input time (a one-element list for a one-element
        input).
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def rotation(
    from_frame: frame,
    to_frame: frame,
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def rotation_with_state(
    from_frame: frame,
    to_frame: frame,
    time: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> quaternion:
    """Quaternion rotating a vector from ``from_frame`` to ``to_frame`` — the
    unified front door supporting **all** frames, both the time-parameterised
    Earth chain (``ITRF``, ``TIRS``, ``CIRS``, ``GCRF``, ``TEME``, ``EME2000``,
    ``ICRF``) and the orbit-dependent frames (``LVLH``, ``RTN``, ``NTW``), in a
    single call.

    Unlike ``rotation`` (which rejects the orbit frames) and ``to_gcrf``
    (which rejects the Earth frames), this accepts any pair. It does **not**
    always pivot through GCRF: a purely Earth-frame pair delegates to
    ``rotation``, which takes the shortest path through the frame graph;
    only pairs involving an orbit-dependent frame compose through GCRF. The
    orbit state (``pos``, ``vel``, both in GCRF) is only consulted when an
    orbit-dependent frame is involved.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        time: Epoch (a single time; a list, even of one element, raises ``TypeError``)
        pos: 3-element GCRF position vector [m]
        vel: 3-element GCRF velocity vector [m/s]

    Returns:
        Rotation from ``from_frame`` to ``to_frame`` at ``time``.

    Raises:
        ValueError: if an orbit-dependent frame is involved and ``pos`` /
            ``vel`` do not define an orbit plane (either is zero, non-finite,
            or they are parallel), or either is not a 3-element vector.
        TypeError: if ``time`` is not a single time, or ``pos`` / ``vel`` is
            not a real numeric array-like.
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def rotation_with_state(
    from_frame: frame,
    to_frame: frame,
    *,
    tm: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> quaternion: ...

@typing.overload
def rotation_approx(
    from_frame: frame,
    to_frame: frame,
    time: TimeScalar,
) -> quaternion:
    """Quaternion rotating a vector from ``from_frame`` to ``to_frame`` using
    the approximate reduction of ``qitrf2gcrf_approx`` (~1 arcsec;
    TEME <-> GCRF / EME2000 / ICRF 0.55", as ``qteme2gcrf``; TEME <-> ITRF
    is exact, as ``qteme2itrf``).

    Only valid between ``ITRF`` and the inertial cluster (``GCRF``,
    ``EME2000``, ``ICRF``, ``TEME``). ``TIRS`` and ``CIRS`` are defined by
    the IERS 2010 reduction and have no analogue in the approximate chain.

    Raises:
        RuntimeError: if either frame is ``TIRS`` / ``CIRS``, or if the pair
            involves orbit-dependent frames.
    """
    ...

@typing.overload
def rotation_approx(
    from_frame: frame,
    to_frame: frame,
    time: TimeArrayLike,
) -> list[quaternion]:
    """Approximate rotations from ``from_frame`` to ``to_frame`` at each of
    the times ``time``; see the scalar overload.

    Returns:
        One rotation per input time (a one-element list for a one-element
        input).
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def rotation_approx(
    from_frame: frame,
    to_frame: frame,
    *,
    tm: TimeInput,
) -> quaternion | list[quaternion]: ...

@typing.overload
def transform_state(
    from_frame: frame,
    to_frame: frame,
    time: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """State (position + velocity) transform from ``from_frame`` to
    ``to_frame`` at ``time``. Properly handles the Earth-rotation sweep term
    when transitioning between rotating (ITRF) and inertial frames.

    Supported pairs: any two of the time-parameterised Earth frames
    ``ITRF``, ``TIRS``, ``CIRS``, ``GCRF``, ``TEME``, ``EME2000`` and ``ICRF``
    (the identity included). A pair involving an orbit-dependent frame
    (``LVLH``, ``RTN``, ``NTW``) raises ``RuntimeError``.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        time: Epoch (a single time; a list, even of one element, raises ``TypeError``)
        pos: 3-element position vector [m]
        vel: 3-element velocity vector [m/s]

    Returns:
        ``(pos, vel)`` in ``to_frame``: position in meters, velocity in m/s.

    Raises:
        RuntimeError: if either frame is ``LVLH``, ``RTN`` or ``NTW``.
        TypeError: if ``time`` is not a single time, or ``pos`` / ``vel`` is not
            a real numeric array-like.
        ValueError: if ``pos`` or ``vel`` is not a 3-element vector.
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def transform_state(
    from_frame: frame,
    to_frame: frame,
    *,
    tm: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

@typing.overload
def transform_state_approx(
    from_frame: frame,
    to_frame: frame,
    time: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """State transform using the approximate reduction of
    ``rotation_approx``. TEME <-> ITRF does not use the approximate chain
    and equals the full transform.

    Supported pairs: those of ``rotation_approx``, a smaller set than
    ``transform_state``: any two of ``ITRF``, ``GCRF``, ``TEME``, ``EME2000``
    and ``ICRF``. ``TIRS`` and ``CIRS`` have no analogue in the approximate
    chain and raise ``RuntimeError`` (except paired with themselves), as do
    the orbit-dependent frames.

    Args:
        from_frame: Source frame
        to_frame: Destination frame
        time: Epoch (a single time; a list, even of one element, raises ``TypeError``)
        pos: 3-element position vector, meters
        vel: 3-element velocity vector, m/s

    Returns:
        ``(pos, vel)`` in ``to_frame``: position in meters, velocity in m/s.

    Raises:
        RuntimeError: if either frame is ``TIRS`` / ``CIRS`` (and the two
            differ), or ``LVLH`` / ``RTN`` / ``NTW``.
        TypeError: if ``time`` is not a single time, or ``pos`` / ``vel`` is not
            a real numeric array-like.
        ValueError: if ``pos`` or ``vel`` is not a 3-element vector.
    """
    ...

@typing.overload
@deprecated("the keyword 'tm' is deprecated; use 'time'")
def transform_state_approx(
    from_frame: frame,
    to_frame: frame,
    *,
    tm: TimeScalar,
    pos: npt.ArrayLike,
    vel: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
