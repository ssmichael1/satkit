# Learn

Interactive tutorials and reference material for the `satkit` package, organized from foundational concepts to advanced applications. Tutorials are Jupyter notebooks with runnable code; theory pages provide deeper mathematical background.

## Foundations

Core concepts that underpin the rest of the library.

| Tutorial | Description |
|----------|-------------|
| [Time Systems](Time%20Systems.ipynb) | UTC, TAI, TT, TDB, UT1, GPS conversions and why they matter |
| [Quaternions](Quaternions.ipynb) | Constructing, composing, and interpolating 3D rotations |
| [Coordinate Frames](Coordinate%20Frames.ipynb) | GCRF, ITRF, TEME, and the rotations between them |
| [Geodetic Coordinates](Geodetic%20Coordinates.ipynb) | The `itrfcoord` data type: geodetic, Cartesian, and local tangent planes |
| [Keplerian Elements](Keplerian%20Elements.ipynb) | Orbital elements, Cartesian conversion, and two-body vs numerical propagation |

## SGP4 Propagation

Analytic mean-element propagation from TLEs and OMMs.

| | Description |
|----------|-------------|
| [Theory: TLEs, SGP4 & OMMs](../guide/tle.md) | TLE/OMM formats, SGP4 history, and mean-element ephemeris concepts |
| [Two-Line Element Set](Two-Line%20Element%20Set.ipynb) | Loading and using TLEs with SGP4 |
| [TLE Fitting](TLE%20Fitting.ipynb) | Fitting TLEs from state vectors |
| [SGP4 vs Numerical Propagation](SGP4%20vs%20Numerical%20Propagation.ipynb) | Comparing analytical and numerical orbit propagation |

## Numerical Propagation

High-fidelity force-model propagation with Runge-Kutta and Gauss-Jackson integrators.

| | Description |
|----------|-------------|
| [Theory: Force Model](../guide/forces.md) | Modeled forces: gravity, third-body, drag, SRP, solid Earth tides, GR |
| [Theory: ODE Integrators](../guide/integrators.md) | RKV / GJ8 integrators, step-size selection, tolerances |
| [Theory: State Vectors, STM & Covariance](../guide/satstate.md) | satstate class, state transition matrix, covariance, maneuvers |
| [Theory: Maneuver Coordinate Frames](../guide/maneuver_frames.md) | RTN / NTW / LVLH definitions and when to use each |
| [GPS Example](GPS%20Example.ipynb) | GPS orbit fit against ESA SP3 truth, integrator comparison |
| [Orbit Maneuvers](Orbit%20Maneuvers.ipynb) | Impulsive maneuvers, frame choice, low-thrust orbit raising |

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

| | Description |
|----------|-------------|
| [Theory: Lambert's Problem](../guide/lambert.md) | Algorithm details: Izzo's method, Lancaster-Blanchard parameterization, multi-revolution solutions |
| [Lambert Targeting](Lambert%20Targeting.ipynb) | Orbit transfer design with delta-v computation and pork-chop plots |
| [Planetary Ephemerides](Planetary%20Ephemerides.ipynb) | JPL DE440 and low-precision Sun/Moon/planet positions |
| [Atmospheric Density](Atmospheric%20Density.ipynb) | NRLMSISE-00 density profiles, solar activity effects, and drag |
| [Covariance Propagation](Covariance%20Propagation.ipynb) | State transition matrix, uncertainty growth, and orbital frame analysis |
| [Orbital Mean-Element Message](Orbital%20Mean-Element%20Message.ipynb) | Working with OMM records |
