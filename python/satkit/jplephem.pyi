"""
High-precision JPL ephemerides for solar-system bodies

For details, see: <https://ssd.jpl.nasa.gov/>

Which file is loaded
--------------------
The ephemeris is loaded lazily on the first query and cached for the process.
satkit looks in every data search directory (see ``satkit.utils.datadir``)
for a JPL Linux-format binary ephemeris (``linux_p*.4XX`` / ``lnxp*.4XX``) and
picks the highest DE version; if none is found it downloads DE440
(``linux_p1550p2650.440``, ~102 MB, SHA-256 verified) into the data write
directory. Set ``SATKIT_JPLEPHEM_FILE`` to an absolute path or a basename to
choose a different file — e.g. ``lnxp1900p2053.421`` selects DE421 (14 MB,
1900–2053), which is downloaded on demand like DE440. A wrong value (a path
that does not exist, an unknown bare name, or a file that is not a JPL binary
ephemeris) raises ``RuntimeError`` from the first query, naming the resolved
path; there is no silent fallback to another ephemeris. A pinned file already
on disk is hashed once on first load (≈0.2 s for DE440; a ``.sha256-verified``
sidecar makes later loads skip the hash) and re-downloaded if corrupt — or, under
offline mode, refused with the expected hash. ``SATKIT_OFFLINE=1``
or ``satkit.utils.set_offline(True)`` turns a needed download into that same
error without touching the network.

Time scale
----------
The ephemeris is evaluated at TDB (the JPL files' ``T_eph``); a ``satkit.time``
argument is converted to TDB internally, whatever scale it was built in.
Releases before 0.24 evaluated it at TT, which moves the Moon by up to ~2 m.
"""

from __future__ import annotations
import numpy.typing as npt
import numpy as np

import satkit
from .satkit import TimeScalar, TimeArrayLike, TimeInput

def geocentric_pos(
    body: satkit.solarsystem, tm: TimeInput
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the GCRF coordinate system (origin is Earth center)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position

    Returns:
        numpy.ndarray: 3-vector of cartesian Geocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array

    Raises:
        RuntimeError: if a time is outside the span of the loaded ephemeris
            (``"Invalid Julian date"``; about 1550 to 2650 for DE440), or
            the ephemeris file is unavailable (see the module documentation)
        TypeError: if ``tm`` is not a time or a list / array of times

    Example:
        ```python
        import numpy as np
        t = satkit.time(2024, 1, 1)
        pos = satkit.jplephem.geocentric_pos(satkit.solarsystem.Sun, t)
        print(f"Sun distance from Earth: {np.linalg.norm(pos)/1e9:.3f} million km")
        ```
    """
    ...

def barycentric_pos(
    body: satkit.solarsystem,
    tm: TimeInput,
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the Barycentric coordinate system (origin is solarsystem barycenter)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position

    Returns:
        numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the solar system barycenter. If input is list or numpy array of N times, then r will be Nx3 array

    Raises:
        RuntimeError: if a time is outside the span of the loaded ephemeris
            (``"Invalid Julian date"``; about 1550 to 2650 for DE440), or
            the ephemeris file is unavailable (see the module documentation)
        TypeError: if ``tm`` is not a time or a list / array of times

    Notes:
     - Every body, the Moon included, is returned relative to the solar system
       barycenter, in the ICRF axes. (The DE files tabulate the Moon
       geocentrically; it is combined with the Earth-Moon barycenter here.
       Releases up to 0.24.3 returned the geocentric Moon from this function:
       use ``geocentric_pos`` / ``geocentric_state`` for that.)
     - EMB is the Earth-Moon barycenter; there is no ``Earth`` member. The
       Earth is ``barycentric(EMB) - geocentric(Moon) / (1 + EMRAT)``, with
       ``EMRAT = satkit.jplephem.consts("EMRAT")``, or equivalently
       ``barycentric(body) - geocentric(body)`` for any body
     - The sun position is relative to the solar system barycenter
       (it will be close to origin)
    """
    ...

def geocentric_state(
    body: satkit.solarsystem,
    tm: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the position and velocity of the given body in Geocentric coordinate system (GCRF)

    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays

    Raises:
        RuntimeError: if a time is outside the span of the loaded ephemeris
            (``"Invalid Julian date"``; about 1550 to 2650 for DE440), or
            the ephemeris file is unavailable (see the module documentation)
        TypeError: if ``tm`` is not a time or a list / array of times

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        pos, vel = satkit.jplephem.geocentric_state(satkit.solarsystem.Moon, t)
        print(f"Moon position (GCRF): {pos} m")
        print(f"Moon velocity (GCRF): {vel} m/s")
        ```
    """
    ...

def barycentric_state(
    body: satkit.solarsystem,
    tm: TimeInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the position & velocity the given body in the barycentric coordinate system (origin is solar system barycenter)


    Args:
        body (satkit.solarsystem): Solar system body for which to return position
        tm (satkit.time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): Time, or list / tuple / 1-D numpy array of times, at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second, with the origin at the solar system barycenter.  If input is list or numpy array of N times, then r and v will be Nx3 arrays

    Raises:
        RuntimeError: if a time is outside the span of the loaded ephemeris
            (``"Invalid Julian date"``; about 1550 to 2650 for DE440), or
            the ephemeris file is unavailable (see the module documentation)
        TypeError: if ``tm`` is not a time or a list / array of times

    Notes:
     - Every body, the Moon included, is returned relative to the solar system
       barycenter, in the ICRF axes. (The DE files tabulate the Moon
       geocentrically; it is combined with the Earth-Moon barycenter here.
       Releases up to 0.24.3 returned the geocentric Moon from this function:
       use ``geocentric_pos`` / ``geocentric_state`` for that.)
     - EMB is the Earth-Moon barycenter; there is no ``Earth`` member. The
       Earth is ``barycentric(EMB) - geocentric(Moon) / (1 + EMRAT)``, with
       ``EMRAT = satkit.jplephem.consts("EMRAT")``, or equivalently
       ``barycentric(body) - geocentric(body)`` for any body
     - The sun position is relative to the solar system barycenter
       (it will be close to origin)

    """
    ...

def consts(name: str) -> float | None:
    """Return a named constant from the loaded JPL ephemeris file

    The DE ephemeris files carry a table of constants used in their
    construction — e.g. "AU" (astronomical unit, km), "EMRAT" (Earth/Moon
    mass ratio), and "GM1".."GM9" / "GMS" / "GMB" (GM values in au^3/day^2).

    Args:
        name (str): Constant name (case-sensitive, as in the DE file header)

    Returns:
        float | None: The constant's value, or None if the name is not present
        (or the ephemeris file could not be loaded)
    """
    ...

def constants() -> dict[str, float]:
    """All named constants of the loaded JPL ephemeris file

    The DE file header's whole constant table, as ``consts`` looks it up one
    name at a time: e.g. "AU" (astronomical unit, km), "EMRAT" (Earth/Moon
    mass ratio), "GM1".."GM9" and "GMS"/"GMB" (GM values in au^3/day^2).
    A missing name is a ``KeyError`` on the returned dict, where ``consts``
    returns None.

    Returns:
        dict[str, float]: Constant name (as in the DE file header) to value;
        a new dict on each call

    Raises:
        RuntimeError: if the ephemeris file is unavailable

    Example:
        >>> c = satkit.jplephem.constants()
        >>> c["EMRAT"] == satkit.jplephem.consts("EMRAT")
        True
    """
    ...
