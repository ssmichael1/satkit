# Satellite Toolkit (satkit)

**A comprehensive, high-performance satellite astrodynamics library combining the speed of Rust with the convenience of Python.**

Satkit provides robust, high-performance satellite orbital mechanics calculations with a clean, intuitive API. Built from the ground up in Rust for maximum performance and memory safety, it offers complete Python bindings for all functionality, making advanced orbital mechanics accessible to both systems programmers and data scientists.

----- 

![Build Passing?](https://github.com/ssmichael1/satkit/actions/workflows/build.yml/badge.svg)
![Wheel Passing?](https://github.com/ssmichael1/satkit/actions/workflows/wheels.yml/badge.svg)
![GitHub License](https://img.shields.io/github/license/ssmichael1/satkit)

![Crates.io Version](https://img.shields.io/crates/v/satkit)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/satkit)

![PyPI - Version](https://img.shields.io/pypi/v/satkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/satkit)
![PyPI - Status](https://img.shields.io/pypi/status/satkit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/satkit)
![Read the Docs](https://img.shields.io/readthedocs/satellite-toolkit)
 
------

## Language Bindings

- **Native Rust**: Available as a crate on [crates.io](https://crates.io/crates/satkit)
- **Python**: Comprehensive Python bindings via PyO3, combining Rust performance with Python convenience
  - Install with `pip install satkit`
  - Binary packages available for Windows, macOS (Intel & ARM), and Linux (x86_64 & ARM64)
  - Python versions 3.8 through 3.14 supported
  - Full documentation at <https://satellite-toolkit.readthedocs.io/latest/>

## Key Capabilities

### Coordinate Frame Transformations
High-precision conversions between multiple reference frames with full support for time-varying Earth orientation:
  - **ITRF** - International Terrestrial Reference Frame (Earth-fixed)
  - **GCRF** - Geocentric Celestial Reference Frame using IAU-2000/2006 reduction (inertial)
  - **TEME** - True Equinox Mean Equator frame used in SGP4 propagation
  - **CIRF** - Celestial Intermediate Reference Frame (IAU-2006 intermediate)
  - **TIRF** - Terrestrial Intermediate Reference Frame (Earth-rotation intermediate)
  - **Geodetic** - Latitude, longitude, altitude with WGS-84 ellipsoid
  
### Orbit Propagation
Multiple propagation methods optimized for different accuracy and performance requirements:
  - **Numerical Integration**: High-precision propagation using adaptive Runge-Kutta 9(8) methods with dense output
    - Supports state transition matrix computation for covariance propagation
    - Configurable force models and integration tolerances
    - Efficient interpolation for arbitrary epoch queries
  - **SGP4**: Industry-standard propagator for Two-Line Element (TLE) sets
    - Full AFSPC and improved mode support
    - TLE fitting from high-precision states with drag estimation
    - Batch processing for multiple satellites
  - **Keplerian**: Fast analytical two-body propagation for preliminary analysis
  
### Force Models
Comprehensive perturbation modeling for high-fidelity orbit propagation:
  - **Earth Gravity**: Spherical harmonic models up to degree/order 360
    - Multiple models: JGM2, JGM3, EGM96, ITU GRACE16
    - Efficient computation with configurable truncation order
    - Gravity gradient support for state transition matrix
  - **Third-Body Gravity**: Solar and lunar perturbations using JPL ephemerides
  - **Atmospheric Drag**: NRLMSISE-00 density model with space weather integration
    - Automatic space weather data updates (F10.7, Ap index)
    - Configurable ballistic coefficients
  - **Solar Radiation Pressure**: Cannon-ball model with shadow function
  
### Ephemerides
Access to high-precision solar system body positions:
  - **JPL DE440/DE441**: State-of-the-art planetary ephemerides
    - Chebyshev polynomial interpolation for accuracy
    - Support for all major planets, sun, moon, and solar system barycenter
  - **Low-Precision Models**: Fast analytical models for sun and moon when high precision isn't required
  
### Time Systems
Comprehensive support for all standard astronomical time scales:
  - **UTC** - Coordinated Universal Time with leap second handling
  - **TAI** - International Atomic Time
  - **TT** - Terrestrial Time
  - **TDB** - Barycentric Dynamical Time
  - **UT1** - Universal Time with Earth orientation corrections
  - **GPS** - GPS Time
  - Automatic conversion between all time scales with microsecond precision
  
### Geodetic Utilities
- **Geodesic Calculations**: Accurate distance and azimuth between ground locations using Vincenty's formulae
- **Coordinate Conversions**: ITRF ↔ Geodetic ↔ East-North-Up ↔ North-East-Down
- **Elevation/Azimuth**: Topocentric coordinate transformations for ground station analysis

### Sun / Moon Calculations
- **Sun rise / set**: Compute sun rise / set times as function of day & location
- **Moon Phase**: Phase of moon and fraction illuminated
- **Ephemeris**: Fast low-precision ephemeris for sun & moon

## Technical Details

### ODE Solvers

The numerical orbit propagation engine employs adaptive Runge-Kutta methods for integration of ordinary differential equations. The pure-Rust ODE solver features:

- **Adaptive Step Size Control**: Automatically adjusts step size based on error tolerance
- **Dense Output**: Efficient interpolation for state queries at arbitrary times without re-integration
- **High-Order Methods**: Runge-Kutta 9(8) pairs for optimal accuracy and stability
- **State Transition Matrix**: Optional computation for covariance propagation and sensitivity analysis

Integration coefficients are based on the work of Jim Verner: [https://www.sfu.ca/~jverner/](https://www.sfu.ca/~jverner/)

### Performance Characteristics

- **Zero-Cost Abstractions**: Rust's ownership model enables safe code without runtime overhead
- **SIMD-Friendly**: Designed to take advantage of modern CPU vector instructions
- **Memory Efficient**: Static typing and stack allocation minimize heap pressure
- **Parallel Processing**: Thread-safe APIs enable concurrent propagation of multiple satellites
- **Python Integration**: Near-native performance in Python via PyO3 bindings with minimal overhead

## References, Models, and External Data

### Theoretical Foundation

The equations and many unit tests are based on the following authoritative sources:

- **"Fundamentals of Astrodynamics and Applications, Fourth Edition"**, D. Vallado, Microcosm Press and Springer, 2013.<br>
  [https://celestrak.org/software/vallado-sw.php](https://celestrak.org/software/vallado-sw.php)
- **"Satellite Orbits: Models, Methods, Applications"**, O. Montenbruck and E. Gill, Springer, 2000.<br>
  [https://doi.org/10.1007/978-3-642-58351-3](https://doi.org/10.1007/978-3-642-58351-3)

### Data Dependencies

The library requires external data files for various calculations. These are automatically downloaded by the `update_datafiles()` function:

#### Core Data Files (One-Time Download)

- **JPL Ephemerides** ([JPL Solar System Dynamics](https://ssd.jpl.nasa.gov/ephem.html))
  - DE440/DE441 planetary ephemerides (~100 MB)
  - Provides positions of sun, moon, planets, and solar system barycenter
  - Valid for years 1550-2650 CE
  
- **Gravity Models** ([ICGEM](http://icgem.gfz-potsdam.de/home))
  - JGM2, JGM3, EGM96, ITU GRACE16 spherical harmonic coefficients
  - International Centre for Global Earth Models standardized format
  - Up to degree/order 360 for high-fidelity propagation
  
- **IERS Nutation Tables** ([IERS Conventions](https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html))
  - IAU-2006 nutation series coefficients
  - Required for GCRF ↔ ITRF transformations
  - Technical Note 36 reference tables

#### Regularly Updated Data Files (Daily Updates Recommended)

- **Space Weather Indices** ([Celestrak](https://celestrak.org/SpaceData/))
  - F10.7 solar flux (past and predicted)
  - Ap geomagnetic index
  - Critical for atmospheric density modeling and drag calculations
  - Updated daily by NOAA Space Weather Prediction Center
  
- **Earth Orientation Parameters** ([Celestrak](https://celestrak.org/SpaceData/))
  - Polar motion (x, y)
  - UT1-UTC time difference
  - Length of day variations
  - Essential for high-precision GCRF ↔ ITRF conversions
  - Updated daily by IERS

The `update_datafiles()` function intelligently manages these files:
- Downloads missing files
- Always refreshes space weather and EOP data
- Skips existing files to save bandwidth
- Stores files in a platform-specific data directory

## Verification and Testing

The library includes comprehensive test suites ensuring correctness of calculations:

### Test Coverage
- **JPL Ephemerides**: Validated against JPL-provided test vectors for Chebyshev polynomial interpolation
  - Over 10,000 test cases covering all planets and time ranges
  - Accuracy verified to within JPL's published tolerances (sub-meter precision)
  
- **SGP4**: Verified using official test vectors from the original C++ distribution
  - All test cases from Vallado's SGP4 implementation
  - Includes edge cases and error conditions
  
- **Coordinate Transformations**: Cross-validated against multiple reference implementations
  - SOFA library comparisons for IAU-2006 transformations
  - Vallado test cases for GCRF ↔ ITRF conversions
  
- **Numerical Propagation**: Validated against high-precision commercial tools
  - Orbit fits to GPS SP3 ephemerides
  - Multi-day propagations with sub-meter accuracy

### Continuous Integration
- Automated testing on multiple platforms (Linux, macOS, Windows)
- Python versions 3.8-3.14 tested on each platform
- Clippy linting for code quality
- Documentation build verification

## Getting Started

### Installation

#### Rust

Add satkit to your `Cargo.toml`:

```toml
[dependencies]
satkit = "0.8"
```

Or use cargo add:

```bash
cargo add satkit
```

#### Python

Install from PyPI using pip:

```bash
pip install satkit
```

Pre-built binary wheels are available for:
- **Windows**: AMD64
- **macOS**: Intel (x86_64) and Apple Silicon (ARM64)  
- **Linux**: x86_64 and ARM64 (aarch64)
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14

### Initial Setup

After installation, download the required [data files](#data-dependencies) needed for calculations. This is a one-time operation, though space weather and Earth orientation parameters should be updated periodically (daily updates available).

**Rust:**
```rust
use satkit::utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download data files to default location
    utils::update_datafiles(None, false)?;
    Ok(())
}
```

**Python:**
```python
import satkit as sk

# Download data files to default location
sk.utils.update_datafiles()
```

### Quick Start Examples

#### Coordinate Transformations (Python)

```python
import satkit as sk
from datetime import datetime, timezone

# Create a time instant
time = sk.time(2024, 1, 1, 12, 0, 0)

# Define position in ITRF (Earth-fixed) frame
itrf_pos = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=100.0)

# Get the ITRF to GCRF (inertial) rotation quaternion
q = sk.frametransform.qitrf2gcrf(time)

# Transform to GCRF
gcrf_pos = q * itrf_pos.vector
print(f"GCRF position: {gcrf_pos}")
```

#### Orbit Propagation (Python)

```python
import satkit as sk
import numpy as np

# Initial state vector [x, y, z, vx, vy, vz] in GCRF frame
r0 = 6378e3 + 500e3  # 500 km altitude
v0 = np.sqrt(sk.consts.mu_earth / r0)
state0 = np.array([r0, 0, 0, 0, v0, 0])

# Start time
time0 = sk.time(2024, 1, 1)

# Propagation settings
settings = sk.propsettings()
settings.gravity_model = sk.gravmodel.JGM3
settings.gravity_order = 8

# Propagate for 1 day
result = sk.propagate(
    state0, 
    time0, 
    stop=time0 + sk.duration.from_days(1),
    propsettings=settings
)

# Query state at any time
query_time = time0 + sk.duration.from_hours(6)
state = result.interp(query_time)
print(f"State after 6 hours: {state}")
```

#### SGP4 Propagation (Python)

```python
import satkit as sk

# Load TLE
lines = [
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
]
tle = sk.TLE.from_lines(lines)

# Propagate TLE
time = sk.time(2024, 1, 2)
pos, vel = sk.sgp4(tle, time)
print(f"ISS position: {pos}")
```

#### Planetary Ephemerides (Rust)

```rust
use satkit::{Instant, SolarSystem, jplephem};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create time instant
    let time = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0)?;
    
    // Get Moon position and velocity in GCRF
    let (pos, vel) = jplephem::geocentric_state(SolarSystem::Moon, &time)?;
    
    println!("Moon position: {:?}", pos);
    println!("Moon velocity: {:?}", vel);
    
    Ok(())
}
```

## Documentation

- **Rust API Documentation**: Available on [docs.rs](https://docs.rs/satkit/)
- **Python Documentation**: Comprehensive guide at [satellite-toolkit.readthedocs.io](https://satellite-toolkit.readthedocs.io/)
  - Getting started tutorials
  - API reference with examples
  - Theory and implementation notes
  - Data file management guide

## Use Cases

Satkit is suitable for a wide range of applications:

- **Satellite Operations**: Real-time tracking and orbit determination
- **Mission Planning**: Trajectory design and optimization
- **Space Situational Awareness**: Conjunction assessment and collision avoidance
- **Ground Station Management**: Visibility predictions and pass planning
- **Research and Education**: Orbital mechanics analysis and experimentation
- **Simulation**: High-fidelity orbit propagation for testing and validation

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes and improvements
- New features and capabilities
- Documentation enhancements
- Additional test cases
- Performance optimizations

See the [GitHub repository](https://github.com/ssmichael1/satkit) for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Steven Michael** - [ssmichael@gmail.com](mailto:ssmichael@gmail.com)

For questions, bug reports, or feature requests, please open an issue on GitHub or contact the author directly.

## Acknowledgments

This work builds upon the theoretical foundations established by:
- Dr. David Vallado - "Fundamentals of Astrodynamics and Applications"
- Dr. Oliver Montenbruck & Dr. Eberhard Gill - "Satellite Orbits: Models, Methods, Applications"
- Dr. Jim Verner - Runge-Kutta integration coefficients
- The international space community for maintaining critical data products (IERS, JPL, NOAA, ICGEM)
