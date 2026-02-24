# satkit

**Satellite astrodynamics in Rust, with full Python bindings.**

![Build](https://github.com/ssmichael1/satkit/actions/workflows/build.yml/badge.svg)
![Release](https://github.com/ssmichael1/satkit/actions/workflows/release.yml/badge.svg)
![License: MIT](https://img.shields.io/github/license/ssmichael1/satkit)

[![Crates.io](https://img.shields.io/crates/v/satkit)](https://crates.io/crates/satkit)
[![Crates.io Downloads](https://img.shields.io/crates/dr/satkit)](https://crates.io/crates/satkit)
[![PyPI](https://img.shields.io/pypi/v/satkit)](https://pypi.org/project/satkit/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/satkit)](https://pypi.org/project/satkit/)
[![Python](https://img.shields.io/pypi/pyversions/satkit)](https://pypi.org/project/satkit/)

---

Satkit is a high-performance orbital mechanics library written in Rust with complete Python bindings via PyO3. It handles coordinate transforms, orbit propagation, time systems, gravity models, atmospheric density, and JPL ephemerides -- everything needed for satellite astrodynamics work.

**[Python documentation and tutorials](https://ssmichael1.github.io/satkit/)** | **[Rust API docs](https://docs.rs/satkit/)**

## Installation

**Rust:**
```bash
cargo add satkit
```

**Python:**
```bash
pip install satkit
```

Pre-built wheels are available for Linux, macOS, and Windows on Python 3.10--3.14.

After installing, download the required data files (gravity models, ephemerides, Earth orientation parameters):

```python
import satkit as sk
sk.utils.update_datafiles()  # one-time download; re-run periodically for fresh EOP/space weather
```

## Quick Examples

### SGP4 propagation (Python)

```python
import satkit as sk

tle = sk.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
])

pos, vel = sk.sgp4(tle, sk.time(2024, 1, 2))
```

### High-precision propagation (Python)

```python
import satkit as sk
import numpy as np

r0 = 6378e3 + 500e3  # 500 km altitude
v0 = np.sqrt(sk.consts.mu_earth / r0)

settings = sk.propsettings()
settings.gravity_model = sk.gravmodel.JGM3
settings.gravity_order = 8

result = sk.propagate(
    np.array([r0, 0, 0, 0, v0, 0]),
    sk.time(2024, 1, 1),
    end=sk.time(2024, 1, 1) + sk.duration.from_days(1),
    propsettings=settings,
)

state = result.interp(sk.time(2024, 1, 1) + sk.duration.from_hours(6))
```

### Coordinate transforms (Python)

```python
import satkit as sk

time = sk.time(2024, 1, 1, 12, 0, 0)
coord = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=100.0)

q = sk.frametransform.qitrf2gcrf(time)
gcrf_pos = q * coord.vector
```

### Planetary ephemerides (Rust)

```rust
use satkit::{Instant, SolarSystem, jplephem};

let time = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0)?;
let (pos, vel) = jplephem::geocentric_state(SolarSystem::Moon, &time)?;
```

## Features

### Coordinate Frames

Full IAU-2006/2000 reduction with Earth orientation parameters:

| Frame | Description |
|-------|-------------|
| ITRF | International Terrestrial Reference Frame (Earth-fixed) |
| GCRF | Geocentric Celestial Reference Frame (inertial) |
| TEME | True Equator Mean Equinox (SGP4 output frame) |
| CIRS | Celestial Intermediate Reference System |
| TIRS | Terrestrial Intermediate Reference System |
| Geodetic | Latitude / longitude / altitude (WGS-84) |

Plus ENU, NED, and geodesic distance (Vincenty) utilities.

### Orbit Propagation

- **Numerical** -- Adaptive Runge-Kutta 9(8) with dense output, state transition matrix, and configurable force models
- **SGP4** -- Standard TLE/OMM propagator with TLE fitting from precision states
- **Keplerian** -- Analytical two-body propagation

### Force Models

- **Earth gravity**: JGM2, JGM3, EGM96, ITU GRACE16 (spherical harmonics up to degree/order 360)
- **Third-body gravity**: Sun and Moon via JPL DE440/441 ephemerides
- **Atmospheric drag**: NRLMSISE-00 with automatic space weather data
- **Solar radiation pressure**: Cannonball model with shadow function

### Time Systems

Seamless conversion between UTC, TAI, TT, TDB, UT1, and GPS time scales with full leap-second handling.

### Solar System

- JPL DE440/DE441 ephemerides for all planets, Sun, Moon, and barycenters
- Fast analytical Sun/Moon models for lower-precision work
- Sunrise/sunset and Moon phase calculations

### Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `omm-xml` | yes | XML OMM deserialization via `quick-xml` |
| `chrono` | no | `TimeLike` impl for `chrono::DateTime` |

## Data Files

Satkit needs external data for gravity models, ephemerides, and Earth orientation. Call `update_datafiles()` to download them automatically.

**Downloaded once:** JPL DE440/441 (~100 MB), gravity model coefficients, IERS nutation tables

**Update periodically:** Space weather indices (F10.7, Ap) and Earth orientation parameters (polar motion, UT1-UTC) -- both sourced from [Celestrak](https://celestrak.org/SpaceData/).

## Testing and Validation

The library is validated against:

- **Vallado** test cases for SGP4, coordinate transforms, and Keplerian elements
- **JPL** test vectors for DE440/441 ephemeris interpolation (10,000+ cases)
- **ICGEM** reference values for gravity field calculations
- **GPS SP3** precise ephemerides for multi-day numerical propagation

99 unit tests and 35 doc-tests run on every commit across Linux, macOS, and Windows.

## Documentation

- **Rust**: [docs.rs/satkit](https://docs.rs/satkit/)
- **Python**: [ssmichael1.github.io/satkit](https://ssmichael1.github.io/satkit/) -- tutorials, Jupyter notebooks, and API reference

## References

- D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013
- O. Montenbruck & E. Gill, *Satellite Orbits: Models, Methods, Applications*, 2000
- J. Verner, [Runge-Kutta integration coefficients](https://www.sfu.ca/~jverner/)

## License

MIT
