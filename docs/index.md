# SatKit

**Satellite astrodynamics in Rust, with full Python bindings.**

![PyPI - Version](https://img.shields.io/pypi/v/satkit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/satkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/satkit)
[![Crates.io](https://img.shields.io/crates/v/satkit)](https://crates.io/crates/satkit)
[![docs.rs](https://img.shields.io/docsrs/satkit)](https://docs.rs/satkit)
![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

📚 **API documentation:** [**Python**](api/index.md) on this site · [**Rust**](https://docs.rs/satkit) on docs.rs.

SatKit is a high-performance orbital mechanics library written in Rust with complete Python bindings via PyO3. It handles coordinate transforms, orbit propagation, time systems, gravity models, atmospheric density, and JPL ephemerides -- everything needed for satellite astrodynamics work.

Pre-built wheels are available for **Linux** (x86_64, aarch64), **macOS** (Apple silicon), and **Windows** (x86_64) on Python 3.11--3.15; Intel Macs build from source or use conda-forge.

## Quick Start

```bash
pip install satkit        # or: conda install -c conda-forge satkit
```

Frames, gravity, SGP4 and time scales work straight away: the IERS nutation tables and gravity models are compiled into the package. The JPL ephemeris, Earth orientation and space weather are downloaded on first use — see [Data Files](getting-started/datafiles.md).

```python
import satkit as sk

tle = sk.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
])[0]

pos, vel = sk.sgp4(tle, sk.time(2024, 1, 2))
```

## Features

- **Coordinate frames** — the full IERS 2010 reduction (IAU 2006/2000A) between ITRF, GCRF, TEME, CIRS, TIRS and geodetic coordinates, with Earth orientation parameters, plus ENU / NED and geodesic distance: [Coordinate Frames](tutorials/Coordinate%20Frames.ipynb)
- **Numerical propagation** — adaptive Runge-Kutta, RODAS4 and Gauss-Jackson 8 integrators with dense output and the state transition matrix: [ODE Integrators](guide/integrators.md), [State Vectors, STM & Covariance](guide/satstate.md)
- **Force models** — EGM2008 / EGM96 / JGM gravity to degree 70 with solid tides, Sun and Moon from JPL DE440, NRLMSISE-00 drag with automatic space weather, solar radiation pressure and relativity: [Force Model](guide/forces.md), validated against [GMAT](guide/gmat_validation.md)
- **SGP4, Kepler and Lambert** — TLE / OMM propagation and TLE fitting, two-body propagation, multi-revolution Lambert targeting: [TLEs, SGP4 & OMMs](guide/tle.md), [Keplerian Elements](guide/kepler.md), [Lambert's Problem](guide/lambert.md)
- **Time systems** — UTC, TAI, TT, TDB, UT1 and GPS with full leap-second handling: [Time Systems](tutorials/Time%20Systems.ipynb)
- **Solar system** — JPL DE440/441 ephemerides, fast analytic Sun / Moon models, sunrise / sunset and Moon phase: [Planetary Ephemerides](tutorials/Planetary%20Ephemerides.ipynb)

## Quick Links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | Install from PyPI or build from source |
| **[Data Files](getting-started/datafiles.md)** | Required data files for calculations |
| **[Learn](tutorials/index.md)** | Tutorials and theory — from basics to advanced topics |
| **[API Reference](api/index.md)** | Full Python API documentation |
| **[References](guide/references.md)** | Sources for every model and algorithm |
| **[Rust API (docs.rs)](https://docs.rs/satkit/)** | Rust API reference |
| **[GitHub](https://github.com/ssmichael1/satkit)** | Source code and issue tracker |

## Author

Steven Michael (ssmichael@gmail.com)

Please reach out if you find errors in code or calculations, are interested in contributing to this repository, or have suggestions for improvements to the API.
