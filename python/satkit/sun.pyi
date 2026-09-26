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

    ``pos_mod`` rotated from mean of date to GCRF (Vallado Equations 3-88
    and 3-89); see ``pos_mod`` for the model

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing sun position in GCRF frame
        at given time.  Units are meters

    Notes:
        - The direction includes the annual aberration and no nutation (the
          apparent direction in mean-of-date terms): it is 20.5 arcsec behind
          the geometric direction of ``satkit.jplephem.geocentric_pos`` along
          the ecliptic
        - Against JPL DE440 over 1900-2100: within 3.6 arcsec in ecliptic
          longitude (0.9 arcsec RMS) and 1600 km in distance

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

    ``pos_mod`` rotated from mean of date to GCRF (Vallado Equations 3-88
    and 3-89); see ``pos_mod`` for the model

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time objects
            representing times at which to compute position

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing sun position in GCRF frame
        at the "N" given times.  Units are meters

    Notes:
        - The direction includes the annual aberration and no nutation (the
          apparent direction in mean-of-date terms): it is 20.5 arcsec behind
          the geometric direction of ``satkit.jplephem.geocentric_pos`` along
          the ecliptic
        - Against JPL DE440 over 1900-2100: within 3.6 arcsec in ecliptic
          longitude (0.9 arcsec RMS) and 1600 km in distance
    """
    ...

@typing.overload
def pos_mod(time: TimeScalar) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Mean equator and equinox of date; see the notes for the model

    Args:
        time (satkit.time): time at which to compute position

    Returns:
        npt.NDArray[np.float64]: 3-element numpy array representing sun position in MOD frame
        at given time.  Units are meters

    Notes:
        - Meeus, "Astronomical Algorithms", 2nd ed., ch. 25 (low-accuracy
          solar coordinates), plus the largest planetary (Venus, Jupiter,
          Mars) and lunar (Earth-Moon barycentre) terms of VSOP87D
          (Bretagnon & Francou 1988).  This replaces Vallado's Algorithm 29
          (up to 43 arcsec off)
        - As before, the direction includes the annual aberration (-20.5
          arcsec in longitude) and no nutation: the apparent direction
          referred to the mean equator and equinox of date.  The distance is
          geometric
        - Against JPL DE440 over 1900-2100: within 3.6 arcsec in ecliptic
          longitude (0.9 arcsec RMS), 1 arcsec in latitude and 1600 km in
          distance
    """
    ...

@typing.overload
def pos_mod(
    time: TimeArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Sun position in the Mean-of-Date Frame

    Mean equator and equinox of date; see the notes for the model

    Args:
        time (npt.ArrayLike | list[satkit.time]): list or numpy array of satkit.time objects,
             representing times at which to compute positions

    Returns:
        npt.NDArray[np.float64]: Nx3 numpy array representing sun position in MOD frame
        at given times.  Units are meters

    Notes:
        - Meeus, "Astronomical Algorithms", 2nd ed., ch. 25 (low-accuracy
          solar coordinates), plus the largest planetary (Venus, Jupiter,
          Mars) and lunar (Earth-Moon barycentre) terms of VSOP87D
          (Bretagnon & Francou 1988).  This replaces Vallado's Algorithm 29
          (up to 43 arcsec off)
        - As before, the direction includes the annual aberration (-20.5
          arcsec in longitude) and no nutation: the apparent direction
          referred to the mean equator and equinox of date.  The distance is
          geometric
        - Against JPL DE440 over 1900-2100: within 3.6 arcsec in ecliptic
          longitude (0.9 arcsec RMS), 1 arcsec in latitude and 1600 km in
          distance
    """
    ...

def rise_set(
    time: TimeScalar,
    coord: satkit.itrfcoord,
    sigma: float | None = None,
    *,
    use_jpl: bool = False,
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

    Raises RuntimeError if the Sun stays above the threshold all day (polar
    day) or below it all day (polar night).

    Notes:
        * Rise and set are when the topocentric apparent Sun's centre is at
          zenith distance ``sigma``, by default 90 deg 50 arcmin: 34' of
          refraction plus a fixed 16' semidiameter below a sea-level horizon
          (the almanac convention of USNO and Skyfield).
        * Vallado Algorithm 30, repeated at the computed event until it moves
          less than 0.1 s.  The built-in model uses the analytic apparent Sun
          of ``pos_mod`` with nutation and solar parallax, and UTC in place of
          UT1.  With ``use_jpl=True`` the Sun is the apparent one from the JPL
          ephemeris as seen from the site (light time, aberration, exact
          parallax), with the full IERS 2010 Earth orientation (UT1,
          precession-nutation, polar motion).
        * Against Skyfield (DE421), every other day of 2024 at latitudes 60 S
          to 65 N: within 0.5 s for the built-in model (typically 0.2 s), and
          0.01 s with ``use_jpl=True``.  The built-in model's errors grow to
          1 s, and a few seconds on the most grazing days, between 65 N and
          the polar-day and polar-night thresholds.
        * The horizon is at sea level: the observer's altitude is ignored.  An
          elevated observer sees the horizon lowered by the dip,
          dip ≈ 1.76' × sqrt(h), h in meters above the surrounding terrain or
          sea, which makes sunrise earlier and sunset later.  To account for
          it, pass ``sigma = 90.0 + (50.0 + 1.76 * math.sqrt(h)) / 60.0``.

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
        use_jpl (bool, optional): use the JPL ephemeris (apparent,
            topocentric) and the full IERS 2010 Earth orientation instead of
            the built-in analytic Sun.  The ephemeris is downloaded on first
            use.  Default False

    Returns:
        tuple[satkit.time, satkit.time]: (sunrise, sunset)

    Raises:
        RuntimeError: for polar day or night, or, with ``use_jpl``, if the JPL
            ephemeris is unavailable or does not cover the date (there is no
            fallback to the built-in model)

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

        # From the JPL ephemeris
        sunrise, sunset = satkit.sun.rise_set(noon, coord, use_jpl=True)
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
