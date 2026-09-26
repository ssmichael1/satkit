"""
Air density models

Currently only contains NRL MSISE-00 air density model
"""

from __future__ import annotations
import typing

import satkit
from .satkit import TimeScalar

@typing.overload
def nrlmsise(
    itrf: satkit.itrfcoord, time: TimeScalar | None = None, /
) -> tuple[float, float]:
    """
    NRL MSISE-00 Atmosphere Density Model

    <https://en.wikipedia.org/wiki/NRLMSISE-00>

    or for more detail:
    <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002JA009430>

    Args:

        itrf (satkit.itrfcoord):  position at which to compute density & temperature
        time (satkit.time|datetime.datetime, optional):  Instant at which to compute
               density & temperature. "Space weather" data at this time will be
               used in model computation.  Note: at satellite altitudes, density can
               change by > 10 X depending on solar cycle. Without a time the model
               runs on its default indices (F10.7 = F10.7A = 150, Ap = 4).

    Returns:
        tuple: (rho, T) where rho is mass density in kg/m^3 and T is temperature in Kelvin

    Raises:
        TypeError: If ``time`` is not a ``satkit.time``, ``datetime.datetime`` or ``None``

    Example:
        ```python
        t = satkit.time(2024, 1, 1)
        coord = satkit.itrfcoord(latitude_deg=0, longitude_deg=0, altitude=400e3)
        rho, temp = satkit.density.nrlmsise(coord, t)
        print(f"Density: {rho:.2e} kg/m^3")
        print(f"Temperature: {temp:.1f} K")
        ```
    """
    ...

@typing.overload
def nrlmsise(
    altitude_meters: float,
    latitude_rad: float = 0.0,
    longitude_rad: float = 0.0,
    time: TimeScalar | None = None,
    /,
) -> tuple[float, float]:
    """
    NRL MSISE-00 Atmosphere Density Model

    <https://en.wikipedia.org/wiki/NRLMSISE-00>

    or for more detail:
    <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002JA009430>

    Args:
        altitude_meters (float):  Altitude in meters
        latitude_rad (float, optional):  Latitude in radians. Default is 0.
        longitude_rad (float, optional):  Longitude in radians.  Default is 0.
        time (satkit.time|datetime.datetime, optional):  Instant at which to compute
               density & temperature. "Space weather" data at this time will be
               used in model computation.  Note: at satellite altitudes, density can
               change by > 10 X depending on solar cycle. Without a time the model
               runs on its default indices (F10.7 = F10.7A = 150, Ap = 4). The time
               may also directly follow the altitude or the latitude.

    Returns:
        tuple: (rho, T) where rho is mass density in kg/m^3 and T is temperature in Kelvin

    Raises:
        TypeError: If an angle is not a real number, or ``time`` is not a
            ``satkit.time``, ``datetime.datetime`` or ``None``
    """
    ...

@typing.overload
def nrlmsise(
    altitude_meters: float, time: TimeScalar | None, /
) -> tuple[float, float]: ...

@typing.overload
def nrlmsise(
    altitude_meters: float, latitude_rad: float, time: TimeScalar | None, /
) -> tuple[float, float]: ...
