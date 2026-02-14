# Constants

## Units

| Symbol | Unit |
|---|---|
| `wgs84_a`, `earth_radius`, `au`, `sun_radius`, `moon_radius`, `geo_r`, `jgm3_a` | `m` |
| `mu_earth`, `mu_moon`, `mu_sun`, `GM`, `jgm3_mu` | `m^3/s^2` |
| `omega_earth` | `rad/s` |
| `c` | `m/s` |
| `wgs84_f`, `earth_moon_mass_ratio`, `jgm3_j2` | unitless |

## API Reference

```{eval-rst}
.. autoclass:: satkit.consts

.. rubric:: Reference / Shape Constants

.. autoattribute:: satkit.consts.wgs84_a
.. autoattribute:: satkit.consts.wgs84_f
.. autoattribute:: satkit.consts.earth_radius

.. rubric:: Gravitational Constants

.. autoattribute:: satkit.consts.mu_earth
.. autoattribute:: satkit.consts.mu_moon
.. autoattribute:: satkit.consts.mu_sun
.. autoattribute:: satkit.consts.GM

.. rubric:: Rotation, Light, and Distance Constants

.. autoattribute:: satkit.consts.omega_earth
.. autoattribute:: satkit.consts.c
.. autoattribute:: satkit.consts.au
.. autoattribute:: satkit.consts.geo_r

.. rubric:: Body Size / Ratio Constants

.. autoattribute:: satkit.consts.sun_radius
.. autoattribute:: satkit.consts.moon_radius
.. autoattribute:: satkit.consts.earth_moon_mass_ratio

.. rubric:: JGM3 Model Constants

.. autoattribute:: satkit.consts.jgm3_mu
.. autoattribute:: satkit.consts.jgm3_a
.. autoattribute:: satkit.consts.jgm3_j2
```
