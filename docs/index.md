# SatKit

**Satellite astrodynamics in Rust, with full Python bindings.**

![PyPI - Version](https://img.shields.io/pypi/v/satkit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/satkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/satkit)
[![Crates.io](https://img.shields.io/crates/v/satkit)](https://crates.io/crates/satkit)
![License: MIT](https://img.shields.io/github/license/ssmichael1/satkit)

SatKit is a high-performance orbital mechanics library written in Rust with complete Python bindings via PyO3. It handles coordinate transforms, orbit propagation, time systems, gravity models, atmospheric density, and JPL ephemerides -- everything needed for satellite astrodynamics work.

Pre-built wheels are available for **Linux**, **macOS**, and **Windows** on Python 3.10--3.14.

## Quick Start

```bash
pip install satkit
```

The `satkit_data` package is installed automatically as a dependency and includes gravity models, JPL ephemerides, and Earth orientation parameters -- no extra download step needed. To update space weather and Earth orientation parameters to the latest values, run:

```python
import satkit as sk
sk.utils.update_datafiles()
```

## Quick Examples

### SGP4 propagation

```python
import satkit as sk

tle = sk.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
])

pos, vel = sk.sgp4(tle, sk.time(2024, 1, 2))
```

### High-precision propagation

```python
import satkit as sk
import numpy as np

r0 = 6378e3 + 500e3  # 500 km altitude
v0 = np.sqrt(sk.consts.mu_earth / r0)

settings = sk.propsettings(
    gravity_model=sk.gravmodel.jgm3,
    gravity_degree=8,
)

result = sk.propagate(
    np.array([r0, 0, 0, 0, v0, 0]),
    sk.time(2024, 1, 1),
    end=sk.time(2024, 1, 1) + sk.duration.from_days(1),
    propsettings=settings,
)

state = result.interp(sk.time(2024, 1, 1) + sk.duration.from_hours(6))
```

### Coordinate transforms

```python
import satkit as sk

time = sk.time(2024, 1, 1, 12, 0, 0)
coord = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=100.0)

q = sk.frametransform.qitrf2gcrf(time)
gcrf_pos = q * coord.vector
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

- **Numerical** -- Adaptive Runge-Kutta integrators (9(8), 8(7), 6(5), 5(4)) with dense output, state transition matrix, and configurable force models
- **SGP4** -- Standard TLE/OMM propagator with TLE fitting from precision states
- **Keplerian** -- Analytical two-body propagation
- **Lambert** -- Multi-revolution Lambert targeting for orbit transfer design

### Force Models

- **Earth gravity**: JGM2, JGM3, EGM96, ITU GRACE16 (spherical harmonics up to degree/order 360)
- **Third-body gravity**: Sun and Moon via JPL DE440/441 ephemerides
- **Atmospheric drag**: NRLMSISE-00 (pure Rust) with automatic space weather data
- **Solar radiation pressure**: Cannonball model with shadow function

### Time Systems

Seamless conversion between UTC, TAI, TT, TDB, UT1, and GPS time scales with full leap-second handling.

### Solar System

- JPL DE440/DE441 ephemerides for all planets, Sun, Moon, and barycenters
- Fast analytical Sun/Moon models for lower-precision work
- Sunrise/sunset and Moon phase calculations

## Quick Links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | Install from PyPI or build from source |
| **[Data Files](getting-started/datafiles.md)** | Required data files for calculations |
| **[User Guide](guide/time.md)** | Learn how to use the library |
| **[Tutorials](tutorials/index.md)** | Interactive Jupyter notebook examples |
| **[API Reference](api/index.md)** | Full Python API documentation |
| **[Rust API (docs.rs)](https://docs.rs/satkit/)** | Rust API reference |
| **[GitHub](https://github.com/ssmichael1/satkit)** | Source code and issue tracker |

## Author

Steven Michael (ssmichael@gmail.com)

Please reach out if you find errors in code or calculations, are interested in contributing to this repository, or have suggestions for improvements to the API.
