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

    Each value covers a range of the moon phase angle (see :func:`phase`),
    given here in degrees.
    """

    NewMoon: ClassVar[moonphase]
    """New Moon (phase angle 0 - 22.5 degrees, or 337.5 - 360 degrees)"""

    WaxingCrescent: ClassVar[moonphase]
    """Waxing Crescent (phase angle 22.5 - 67.5 degrees)"""

    FirstQuarter: ClassVar[moonphase]
    """First Quarter (phase angle 67.5 - 112.5 degrees)"""

    WaxingGibbous: ClassVar[moonphase]
    """Waxing Gibbous (phase angle 112.5 - 157.5 degrees)"""

    FullMoon: ClassVar[moonphase]
    """Full Moon (phase angle 157.5 - 202.5 degrees)"""

    WaningGibbous: ClassVar[moonphase]
    """Waning Gibbous (phase angle 202.5 - 247.5 degrees)"""

    LastQuarter: ClassVar[moonphase]
    """Last Quarter (phase angle 247.5 - 292.5 degrees)"""

    WaningCrescent: ClassVar[moonphase]
    """Waning Crescent (phase angle 292.5 - 337.5 degrees)"""

@typing.overload
def pos_gcrf(time: TimeScalar) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the GCRF Frame

    Algorithm 31 from Vallado for the moon in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing moon position in GCRF frame
        at given time.  Units are meters

    Notes:
        Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst case
        against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude,
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

    Algorithm 31 from Vallado for the moon in Mean of Date (MOD), then rotated
    from MOD to GCRF via Equations 3-88 and 3-89 in Vallado

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time
            for which to compute position

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing moon position in GCRF frame
        at given times.  Units are meters

    Notes:
        Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst case
        against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude,
        and 1275 km in range
    """
    ...

@typing.overload
def pos_mod(time: TimeScalar) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the Mean-of-Date (MOD) Frame

    Algorithm 31 from Vallado for the moon in Mean of Date (MOD)

    ``pos_gcrf(t) = satkit.frametransform.qmod2gcrf(t) * pos_mod(t)``

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing moon position in MOD frame
        at given time.  Units are meters

    Notes:
        Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst case
        against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude,
        and 1275 km in range

        Useful when comparing with Vallado's worked examples, which are given in
        MOD coordinates

    Example:
        ```python
        import numpy as np
        t = satkit.time(2024, 1, 1)
        moon = satkit.moon.pos_mod(t)
        print(f"Moon distance: {np.linalg.norm(moon)/1e3:.0f} km")
        ```
    """
    ...

@typing.overload
def pos_mod(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Approximate Moon position in the Mean-of-Date (MOD) Frame

    Algorithm 31 from Vallado for the moon in Mean of Date (MOD)

    ``pos_gcrf(t) = satkit.frametransform.qmod2gcrf(t) * pos_mod(t)``

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time
            for which to compute position

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing moon position in MOD frame
        at given times.  Units are meters

    Notes:
        Accurate to about 0.3 degree in ecliptic longitude (0.36 degree worst case
        against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude,
        and 1275 km in range

        Useful when comparing with Vallado's worked examples, which are given in
        MOD coordinates
    """
    ...

@typing.overload
def illumination(time: TimeScalar) -> float:
    """
    Fractional illumination of moon

    Args:
        time (satkit.time | datetime.datetime): scalar time at which to compute illumination

    Returns:
        float: fractional illumination of moon at the given time, unitless, range 0.0 to 1.0

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
        list[float]: fractional illumination of moon at each given time, unitless, range 0.0 to 1.0
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


def rise_set(
    time: TimeScalar, coord: satkit.itrfcoord, *, use_jpl: bool = False
) -> tuple[satkit.time | None, satkit.time | None]:
    """
    Moonrise and moonset times on a calendar date at the given location.

    The input selects the UTC calendar date (its time of day is ignored);
    the returned moonrise and moonset are those of that date at the
    location's longitude, i.e. in the local mean day from
    0h UTC - longitude / 15 hours to 24 hours later, as UTC times.  At
    far-west longitudes an event can fall on the next UTC date (and at
    far-east longitudes on the previous one).  These are the same day
    semantics as ``satkit.sun.rise_set``.

    To get the events of a *local* date, pass a timezone-aware datetime at
    local noon: for time zones UTC-11 to UTC+11 local noon falls on the same
    UTC date, while local midnight falls on the previous UTC date east of
    Greenwich.

    The Moon rises about 50 minutes later each day, so about once a month
    there is no moonrise on a date (and on another no moonset): that event
    is None.  At high latitudes the Moon can stay up or down all day,
    giving (None, None), and a day can hold two moonrises (or moonsets):
    the first is returned.

    Notes:
        * Rise and set are when the Moon's upper limb touches a sea-level
          horizon with 34 arcmin of refraction (the USNO / Astronomical
          Almanac definition): the topocentric altitude of the Moon's centre
          is -34' - asin(R_moon / d), with d the topocentric distance and
          R_moon = 1737.4 km.  Parallax is exact; altitude is measured from
          the plane normal to the geodetic vertical, with no dip correction
          for an elevated site.
        * The altitude is sampled every 10 minutes (and near-horizon extrema
          checked for a grazing Moon), and events refined to 0.05 s.
        * Accuracy against Skyfield (DE440s) over 2024, 35 S to 62 N: with
          use_jpl, 0.1 s (1.6 s for a grazing Moon at 62 N); built-in,
          52 s at the equator, 89 s at 35 deg, 128 s at 52 deg, a few
          minutes at 62 deg.  Real refraction varies with the weather by a
          minute or more of time.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): time whose UTC calendar date selects the day
        coord (satkit.itrfcoord): location at which to compute moonrise and moonset
        use_jpl (bool, optional): use the JPL ephemeris (apparent position) and the full
            IERS 2010 Earth orientation instead of the built-in analytic Moon.  Default False

    Returns:
        tuple[satkit.time | None, satkit.time | None]: moonrise and moonset, UTC, each None
        if it does not happen that day

    Raises:
        RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the date

    Example:
        ```python
        from datetime import datetime, timedelta, timezone

        # Moonrise and moonset on 2024-03-15 (local date) in New York
        eastern = timezone(timedelta(hours=-4))
        coord = satkit.itrfcoord(latitude_deg=40.71, longitude_deg=-74.01)
        noon = datetime(2024, 3, 15, 12, tzinfo=eastern)
        rise, set = satkit.moon.rise_set(noon, coord)
        print(f"Moonrise: {rise.to_datetime().astimezone(eastern) if rise else None}")
        print(f"Moonset:  {set.to_datetime().astimezone(eastern) if set else None}")
        ```
    """
    ...

def phase_times(
    start: TimeScalar, end: TimeScalar, *, use_jpl: bool = False
) -> list[tuple[moonphase, satkit.time]]:
    """
    Times of the principal Moon phases in a time interval

    Returns every New Moon, First Quarter, Full Moon and Last Quarter in
    [start, end), in time order.

    Notes:
        * The principal phases are when the Moon's geocentric apparent
          ecliptic longitude exceeds the Sun's by 0, 90, 180 and 270 degrees
          (the Astronomical Almanac / USNO definition).  Full Moon is
          therefore not exactly the instant of greatest illuminated fraction.
        * With use_jpl, apparent JPL positions (light time and aberration,
          which move the Sun by 20 arcsec and the phase times by ~40 s) in
          the mean ecliptic of date; otherwise the built-in analytic Sun and
          Moon (see phase)
        * Accuracy over 2024: with use_jpl, 0.1 s of Skyfield (DE440s) and
          40 s of USNO's minute-rounded times; built-in, up to 22 minutes
          (the analytic Moon's 0.36 deg worst-case longitude error allows 42)

    Args:
        start (satkit.time | datetime.datetime | numpy.datetime64): start of the interval (inclusive)
        end (satkit.time | datetime.datetime | numpy.datetime64): end of the interval (exclusive)
        use_jpl (bool, optional): use apparent JPL positions instead of the built-in
            analytic Sun and Moon.  Default False

    Returns:
        list[tuple[moonphase, satkit.time]]: (phase, time) pairs in time order; empty if end <= start

    Raises:
        RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the interval

    Example:
        ```python
        # The full moons of 2024
        for phase, t in satkit.moon.phase_times(satkit.time(2024, 1, 1), satkit.time(2025, 1, 1)):
            if phase == satkit.moon.moonphase.FullMoon:
                print(t)
        ```
    """
    ...

def next_phase(
    time: TimeScalar, phase: moonphase, *, use_jpl: bool = False
) -> satkit.time:
    """
    Time of the next occurrence of a principal Moon phase

    Returns the first time at or after `time` when the Moon reaches `phase`.
    See phase_times for the definition and accuracy.

    Args:
        time (satkit.time | datetime.datetime | numpy.datetime64): time from which to search
        phase (moonphase): NewMoon, FirstQuarter, FullMoon or LastQuarter
        use_jpl (bool, optional): use apparent JPL positions instead of the built-in
            analytic Sun and Moon.  Default False

    Returns:
        satkit.time: time of the phase, UTC

    Raises:
        ValueError: if phase is not one of the four principal phases (e.g. WaxingCrescent,
            which spans a range of phases)
        RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the search

    Example:
        ```python
        full = satkit.moon.next_phase(satkit.time(2024, 1, 1), satkit.moon.moonphase.FullMoon)
        print(f"Next full moon: {full}")
        ```
    """
    ...
