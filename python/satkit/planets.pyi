"""
Low-precision planetary ephemerides from the Keplerian-element approximations of
E. M. Standish and J. G. Williams, "Keplerian Elements for Approximate Positions
of the Major Planets", JPL Solar System Dynamics:
https://ssd.jpl.nasa.gov/planets/approx_pos.html
"""

from __future__ import annotations
import numpy.typing as npt
import numpy as np

import satkit
from .satkit import TimeScalar, TimeArrayLike, TimeInput

def heliocentric_pos(
    planet: satkit.solarsystem,
    time: TimeInput,
) -> npt.NDArray[np.float64]:
    """Return the position of the given body in the Heliocentric coordinate system (origin is the Sun)

    Axes are ICRF (J2000 equatorial).

    Note: Valid bodies are Mercury through Pluto and the Earth-Moon barycenter
    (``satkit.solarsystem.EMB``); the Sun and the Moon raise ``RuntimeError``

    Note: Valid from 3000 BC to 3000 AD, with a more accurate element set for 1800 AD to 2050 AD

    Note: This is less accurate than using the jpl ephemeris, but involves fewer calculations.
    JPL's approximate errors in heliocentric ecliptic longitude / latitude / range run from
    15" / 1" / 1000 km (Mercury, 1800-2050) to 2000" / 30" / 8 million km (Uranus,
    3000 BC-3000 AD); see https://ssd.jpl.nasa.gov/planets/approx_pos.html.  The Uranus,
    Neptune and Pluto elements follow the orbit about the solar-system barycenter, so from
    1800 to 2050 their heliocentric errors are dominated by the Sun's unmodeled barycentric
    motion (up to ~2 arcmin and ~2.3 million km).

    Args:
        planet (satkit.solarsystem): Mercury through Pluto, or EMB
        time (satkit.time|numpy.ndarray|list): Time[s] at which to return position

    Returns:
        numpy.ndarray: 3-vector of Cartesian position in meters, with the origin at the Sun.
                       If input is list or numpy array of N times, then r will be Nx3 array

    Example:
        ```python
        import numpy as np
        t = satkit.time(2024, 1, 1)
        pos = satkit.planets.heliocentric_pos(satkit.solarsystem.Mars, t)
        print(f"Mars distance from Sun: {np.linalg.norm(pos)/satkit.consts.au:.2f} AU")
        ```
    """
    ...
