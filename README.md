# satkit

**Satellite astrodynamics in Rust, with full Python bindings.**

![Build](https://github.com/ssmichael1/satkit/actions/workflows/build.yml/badge.svg)
![Release](https://github.com/ssmichael1/satkit/actions/workflows/release.yml/badge.svg)
![License: MIT OR Apache-2.0](https://img.shields.io/github/license/ssmichael1/satkit)

[![Crates.io](https://img.shields.io/crates/v/satkit)](https://crates.io/crates/satkit)
[![Crates.io Downloads](https://img.shields.io/crates/dr/satkit)](https://crates.io/crates/satkit)
[![PyPI](https://img.shields.io/pypi/v/satkit)](https://pypi.org/project/satkit/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/satkit)](https://pypi.org/project/satkit/)
[![Python](https://img.shields.io/pypi/pyversions/satkit)](https://pypi.org/project/satkit/)

---

Satkit is a high-performance orbital mechanics library written in Rust with complete Python bindings via PyO3. It handles coordinate transforms, orbit propagation, time systems, gravity models, atmospheric density, and JPL ephemerides -- everything needed for satellite astrodynamics work.

**[Documentation and tutorials](https://satkit.dev/)** (Python examples, but the concepts and API apply equally to Rust) | **[Rust API reference](https://docs.rs/satkit/)**


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

settings = sk.propsettings(
    gravity_model=sk.gravmodel.egm96,  # default; also jgm3, jgm2, itugrace16
    gravity_degree=8,
    integrator=sk.integrator.rkv98,    # default; also rkv87, rkv65, rkts54,
                                       # gauss_jackson8 (fixed-step multistep)
)

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

Full IERS 2010 Conventions reduction (IAU 2006/2000A precession-nutation) with Earth orientation parameters:

| Frame | Description |
|-------|-------------|
| ITRF | International Terrestrial Reference Frame (Earth-fixed) |
| GCRF | Geocentric Celestial Reference Frame (inertial) |
| TEME | True Equator Mean Equinox (SGP4 output frame) |
| CIRS | Celestial Intermediate Reference System |
| TIRS | Terrestrial Intermediate Reference System |
| EME2000 / ICRF | J2000 mean equator and the International Celestial Reference Frame |
| Geodetic | Latitude / longitude / altitude (WGS-84) |

Plus satellite-local RTN, NTW, and LVLH frames (maneuvers, covariance), and ENU, NED, and geodesic distance (Vincenty) utilities.

### Orbit Propagation

- **Numerical** -- Selectable adaptive Runge-Kutta integrators (9(8), 8(7), 6(5), 5(4)) plus RODAS4 (stiff) and Gauss-Jackson 8 (fixed-step multistep for high-precision long-duration propagation), with dense output, state transition matrix, and configurable force models. With matched force models it agrees with NASA GMAT to a few centimetres over 7 days in LEO, MEO, and GEO (see [Testing and Validation](#testing-and-validation))
- **SGP4** -- Standard TLE/OMM propagator with TLE fitting from precision states
- **Keplerian** -- Analytical two-body propagation

### Orbit Maneuvers

- **Impulsive maneuvers** -- Instantaneous delta-v applied at a scheduled time during propagation. Supported frames: GCRF (inertial), RTN (radial/tangential/normal — the CCSDS OEM convention, also exposed as `RSW` and `RIC` aliases), NTW (velocity-aligned — natural for prograde burns on eccentric orbits, where a pure +T delta-v adds exactly Δv to |v|), and LVLH (Local Vertical / Local Horizontal). Ergonomic helpers `add_prograde` / `add_retrograde` / `add_radial` / `add_normal` for common scalar-magnitude burns.
- **Continuous thrust** -- Constant-acceleration thrust arcs over time windows in any of the frames above, integrated directly into the force model
- **Automatic segmentation** -- Propagation through maneuver sequences is handled transparently, including backward propagation

### Force Models

- **Earth gravity**: JGM2, JGM3, EGM96, ITU GRACE16 (spherical harmonics up to degree/order 40)
- **Solid Earth tides**: IERS 2010 Step-1 corrections to the gravity field
- **Third-body gravity**: Sun and Moon via JPL DE440/441 ephemerides
- **Atmospheric drag**: NRLMSISE-00 with automatic space weather data
- **Solar radiation pressure**: Cannonball model with shadow function
- **Relativity**: IERS 2010 Eq. 10.12 — Schwarzschild, geodesic (de Sitter) precession, and Lense–Thirring

### Time Systems

Seamless conversion between UTC, TAI, TT, TDB, UT1, and GPS time scales with full leap-second handling.

### Solar System

- JPL DE440/DE441 ephemerides for all planets, Sun, Moon, and barycenters
- Fast analytical Sun/Moon models for lower-precision work
- Sunrise/sunset and Moon phase calculations

### Linear Algebra

SatKit uses [numeris](https://crates.io/crates/numeris) for all linear algebra (vectors, matrices, quaternions, ODE integration). If you also use nalgebra in your project, enable the `nalgebra` feature on numeris for zero-cost `From`/`Into` conversions between types:

```toml
numeris = { version = "0.5.18", features = ["nalgebra"] }
```

### Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `omm-xml` | yes | XML OMM deserialization via `quick-xml` |
| `download` | yes | Data-file downloader (`update_datafiles`) via `ureq` |
| `chrono` | no | `TimeLike` impl for `chrono::DateTime` |

## Data Files

Satkit needs external data for gravity models, ephemerides, and Earth orientation. Call `update_datafiles()` to download them automatically.

**Downloaded once:** JPL DE440/441 (~100 MB), gravity model coefficients, IERS nutation tables

**Update periodically:** Space weather indices (F10.7, Ap) and Earth orientation parameters (polar motion, UT1-UTC) -- both sourced from [Celestrak](https://celestrak.org/SpaceData/).

## Testing and Validation

The library is validated against:

- **Vallado** test cases for SGP4, coordinate transforms, and Keplerian elements
- **JPL** test vectors for DE440/441 ephemeris interpolation (10,000+ cases)
- **NASA GMAT** reference trajectories for the high-precision propagator (see below)
- **ICGEM** reference values for gravity field calculations
- **GPS SP3** precise ephemerides for multi-day numerical propagation

Around 300 Rust tests and 150 Python tests run on every commit across Linux, macOS, and Windows.

### GMAT comparison

The numerical propagator is regression-tested against NASA's General Mission Analysis Tool (GMAT R2026A). The corpus in `tests/gmat/` holds 17 seven-day reference trajectories -- ISS-like LEO, sun-synchronous, GPS MEO, Molniya, GEO, the lunar-resonant TESS orbit, and a 300,000 km cislunar orbit -- each with a low-degree gravity model, a 36×36 EGM96 + solid tides model, and (for three orbits) relativity. GMAT cannot run in CI, so the trajectories are generated offline (`tests/gmat/generate.py`, SPICE DE440, `EarthICRF`) and committed; `tests/gmat_regression.rs` and `python/test/test_gmat.py` replay them hour by hour and gate on the worst residual.

With matched force models the two agree to 3 cm (ISS), 2 cm (SSO), 8 cm (GPS), and 13 cm (GEO, Molniya) over 7 days. At 200,000 km and beyond the residual is ~1 m, which is GMAT's own integration floor (its point-mass runs differ from the analytic Kepler solution by the same amount). The remaining differences with tides and relativity enabled are documented with the tolerances in `tests/gmat/README.md`: GMAT omits the anelastic phase lag in its solid-tide Love numbers that satkit includes, while the relativity cases sit at the same floors (both tools apply the full IERS 2010 Eq. 10.12 correction).

### Running Tests Locally

Tests require two sets of external data: the **astro-data** files (gravity models, ephemerides, etc.) and the **test vectors** (reference outputs for validation). Download both before running:

```bash
# Install the download helper
pip install requests

# Download data files and test vectors into the current directory
python python/test/download_data.py astro-data
python python/test/download_testvecs.py satkit-testvecs
```

Then run tests with the environment variables pointing to the downloaded directories:

```bash
# Rust tests
SATKIT_DATA=astro-data SATKIT_TESTVEC_ROOT=satkit-testvecs cargo test

# Python tests (after `pip install -e ".[test]"`)
SATKIT_DATA=astro-data SATKIT_TESTVEC_ROOT=satkit-testvecs pytest python/test/
```

The GMAT regression tests need only the data files; their reference trajectories are checked in.

## Documentation

- **Rust**: [docs.rs/satkit](https://docs.rs/satkit/)
- **Python**: [satkit.dev](https://satkit.dev/) -- tutorials, Jupyter notebooks, and API reference

## References

- D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013
- O. Montenbruck & E. Gill, *Satellite Orbits: Models, Methods, Applications*, 2000
- J. Verner, [Runge-Kutta integration coefficients](https://www.sfu.ca/~jverner/)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
