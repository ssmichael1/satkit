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
path; there is no silent fallback to another ephemeris. ``SATKIT_OFFLINE=1``
or ``satkit.utils.set_offline(True)`` turns a needed download into that same
error without touching the network.
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
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of cartesian Geocentric position in meters. If input is list or numpy array of N times, then r will be Nx3 array

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
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the solar system barycenter. If input is list or numpy array of N times, then r will be Nx3 array

    Notes:
     - Positions for all bodies are natively relative to solar system barycenter,
       with exception of moon, which is computed in Geocentric system
     - EMB (2) is the Earth-Moon barycenter
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
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second.  If input is list or numpy array of N times, then r and v will be Nx3 arrays

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
        tm (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        tuple: (r, v) where r is the position in meters and v is the velocity in meters / second, with the origin at the solar system barycenter.  If input is list or numpy array of N times, then r and v will be Nx3 arrays

    Notes:
     - Positions for all bodies are natively relative to solar system barycenter,
       with exception of moon, which is computed in Geocentric system
     - EMB (2) is the Earth-Moon barycenter
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
    """
    ...
