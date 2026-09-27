# Sun, Moon and Ephemerides

satkit has two sources of solar-system positions: the JPL Development Ephemerides (`jplephem`), accurate to meters but read from a large file, and analytic low-precision models (`lpephem`), which need no data and are good to tens of arcseconds (Sun) and a few arcminutes (Moon). The [Planetary Ephemerides](../tutorials/Planetary%20Ephemerides.ipynb) tutorial compares them.

## Low-precision Sun and Moon: `lpephem`

- `lpephem::sun::pos_gcrf(&t)` and `lpephem::moon::pos_gcrf(&t)`: geocentric GCRF positions in meters (`pos_mod` for the mean-of-date frame).
- `sun::riseset(&t, &coord, sigma)`: sunrise and sunset on the UTC calendar date of `t` at an `ITRFCoord`, as UTC `Instant`s. `sigma` is the Sun's zenith angle at the event in degrees, `None` for the standard 90° 50′; 96°, 102° and 108° give civil, nautical and astronomical twilight. Where the Sun stays up or down all day it returns `lpephem::Error::NoSunriseOrSunset`.
- `moon::phase(&t)`, `moon::illumination(&t)` and `moon::phase_name(&t)`: the Sun–Moon elongation, the illuminated fraction, and the named phase.
- `sun::shadowfunc(&sun_pos, &sat_pos)`: the fraction of the Sun's disc visible from a satellite (1 in sunlight, 0 in the umbra, in between in the penumbra).
- `lpephem::heliocentric_pos(body, &t)`: approximate heliocentric planet positions from Keplerian elements.

```rust
--8<-- "examples/sun_moon.rs"
```

```text
$ cargo run --example sun_moon
Sun  distance = 1.016219 AU
Moon distance = 386478 km
Moon phase 160.9 deg (Full Moon), 97.3% illuminated
Boston sunrise 09:07 UTC, sunset 00:24 UTC
civil twilight 08:32 to 00:59 UTC
Tromso: midnight Sun, no sunset
shadow function: sunward 1.0, behind the Earth 0.0
```

## JPL ephemerides: `jplephem`

`jplephem::geocentric_pos(body, &t)`, `geocentric_state`, `barycentric_pos` and `barycentric_state` return positions (m) and velocities (m/s) of a `SolarSystem` body in the ICRF/GCRF axes. The first query loads the DE440 file `linux_p1550p2650.440` (1550–2650), downloading it (102 MB, SHA-256 verified) if it is not in a data directory; `SATKIT_JPLEPHEM_FILE` selects another file, such as the 14 MB DE421 ([Selecting a JPL ephemeris file](../getting-started/datadirs.md#selecting-a-jpl-ephemeris-file)). Offline with no copy on disk, the queries return `jplephem::Error`.

```rust
--8<-- "examples/ephemeris.rs"
```

```text
$ cargo run --example ephemeris
Moon: 386605 km away, moving at 1.021 km/s
Venus: 0.7165 AU from the barycenter
Mars: 1.3874 AU from the barycenter
Jupiter: 5.0174 AU from the barycenter
low-precision Sun:  17 arcsec from DE440
low-precision Moon: 306 arcsec from DE440
AU in the DE440 header: 149597870.7 km
```

## See also

- [Planetary Ephemerides](../tutorials/Planetary%20Ephemerides.ipynb), [Sunrise & Sunset](../tutorials/riseset.ipynb) and [Solar Eclipse Predictions](../tutorials/Solar%20Eclipse%20Predictions.ipynb).
- [`jplephem`](https://docs.rs/satkit/latest/satkit/jplephem/) and [`lpephem`](https://docs.rs/satkit/latest/satkit/lpephem/) on docs.rs.
