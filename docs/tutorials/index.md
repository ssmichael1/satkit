# Tutorials

Interactive Jupyter notebook tutorials demonstrating various features of the `satkit` package. Tutorials are organized from foundational concepts to advanced applications.

## Foundations

Core concepts that underpin the rest of the library.

| Tutorial | Description |
|----------|-------------|
| [Time Systems](Time%20Systems.ipynb) | UTC, TAI, TT, TDB, UT1, GPS conversions and why they matter |
| [Quaternions](Quaternions.ipynb) | Constructing, composing, and interpolating 3D rotations |
| [ITRF Coordinates](ITRF%20Coordinates.ipynb) | Working with geodetic and Cartesian coordinates |
| [Coordinate Frame Transforms](Coordinate%20Frame%20Transforms.ipynb) | GCRF, ITRF, and TEME rotations with comparison of approximate vs full IAU-2006 |
| [Keplerian Elements](Keplerian%20Elements.ipynb) | Orbital elements, Cartesian conversion, and two-body vs numerical propagation |

## Orbit Propagation

Propagating satellite orbits using analytical and numerical methods.

| Tutorial | Description |
|----------|-------------|
| [Two-Line Element Set](Two-Line%20Element%20Set.ipynb) | Loading and using TLEs with SGP4 |
| [High Precision Propagation](High%20Precision%20Propagation.ipynb) | Numerical orbit propagation with force models |
| [SGP4 vs Numerical Propagation](SGP4%20vs%20Numerical%20Propagation.ipynb) | Comparing analytical and numerical orbit propagation |
| [TLE Fitting](TLE%20Fitting.ipynb) | Fitting TLEs from state vectors |

## Applications

Common tasks built on top of the core library.

| Tutorial | Description |
|----------|-------------|
| [Plots](Plots.ipynb) | Plotting satellite orbits and ground tracks |
| [Satellite Ground Contacts](Satellite%20Ground%20Contacts.ipynb) | Computing satellite ground contacts and visibility |
| [Eclipse](Eclipse.ipynb) | Computing satellite eclipse times |
| [Sunrise & Sunset](riseset.ipynb) | Computing sunrise, sunset, and twilight times |
| [Optical Observations](Optical%20Observations%20of%20Satellites.ipynb) | Simulating optical satellite observations |

## Advanced Topics

Specialized capabilities for mission analysis and design.

| Tutorial | Description |
|----------|-------------|
| [Planetary Ephemerides](Planetary%20Ephemerides.ipynb) | JPL DE440 and low-precision Sun/Moon/planet positions |
| [Atmospheric Density](Atmospheric%20Density.ipynb) | NRLMSISE-00 density profiles, solar activity effects, and drag |
| [Covariance Propagation](Covariance%20Propagation.ipynb) | State transition matrix, uncertainty growth, and LVLH frame |
| [Lambert Targeting](Lambert%20Targeting.ipynb) | Orbit transfer design with delta-v computation and pork-chop plots |
| [Orbital Mean-Element Message](Orbital%20Mean-Element%20Message.ipynb) | Working with OMM records |
