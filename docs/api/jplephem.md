# JPL Ephemerides

Positions (m) and velocities (m/s) of the Sun, Moon, planets, Pluto and the Earth-Moon barycenter from the JPL Development Ephemerides, in the ICRF / GCRF axes, relative to the Earth's center (`geocentric_*`) or the solar-system barycenter (`barycentric_*`). Bodies are named by [`satkit.solarsystem`](#satkit.solarsystem); times are converted to TDB internally.

The first query downloads the DE440 file (102 MB) into the data directory; [Selecting a JPL ephemeris file](../getting-started/datadirs.md#selecting-a-jpl-ephemeris-file) describes the smaller DE421 alternative. The [`sun`](sun.md), [`moon`](moon.md) and [`planets`](planets.md) modules are analytic, lower-precision models that need no file. See the [Planetary Ephemerides](../tutorials/Planetary%20Ephemerides.ipynb) tutorial.

::: satkit.jplephem

::: satkit.solarsystem
