# SatKit

![PyPI - Version](https://img.shields.io/pypi/v/satkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/satkit)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/satkit)
![PyPI - Status](https://img.shields.io/pypi/status/satkit)

**SatKit** is a Python library providing tools that enable computation and prediction of satellite orbits, satellite maneuvers, and satellite attitude dynamics.

The SatKit core code is written in [Rust](https://www.rust-lang.org) for speed and safety. The Python bindings are done via the [PyO3](https://pyo3.rs) package. All calculations are performed natively in Rust, making the package much faster than a pure-Python equivalent.

## Features

- High-precision coordinate transforms between:
    - International Terrestrial Reference Frame (ITRF)
    - Geocentric Celestial Reference Frame (GCRF) using IAU-2000 reduction
    - True-Equinox Mean Equator (TEME) frame used in SGP4 propagation of TLEs
    - Celestial Intermediate Reference Frame (CIRF)
    - Terrestrial Intermediate Reference Frame (TIRF)
    - Terrestrial Geodetic frame (latitude, longitude)
- Geodesic distances
- SGP4 and Keplerian orbit propagation
- JPL high-precision planetary ephemerides
- High-order gravity models
- High-precision, high-speed numerical satellite orbit propagation with high-order efficient Runge-Kutta solvers, ability to solve for state transition matrix, and inclusion of the following forces:
    - High-order Earth gravity with multiple models
    - Solar gravity
    - Lunar gravity
    - Drag (NRL MSISE-00 density model)
    - Radiation pressure

## Quick Links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | Install from PyPI or build from source |
| **[Data Files](getting-started/datafiles.md)** | Required data files for calculations |
| **[User Guide](guide/time.md)** | Learn how to use the library |
| **[Tutorials](tutorials/index.md)** | Interactive Jupyter notebook examples |
| **[API Reference](api/index.md)** | Full Python API documentation |

## Author

Steven Michael (ssmichael@gmail.com)

Please reach out if you find errors in code or calculations, are interested in contributing to this repository, or have suggestions for improvements to the API.
