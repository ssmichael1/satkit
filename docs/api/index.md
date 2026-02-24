# API Reference

Python API documentation for satkit, auto-generated from type stubs.

## Time & Duration

| Symbol | Description |
|--------|-------------|
| [`time`](time.md) | Representation of an instant in time with multiple time scales |
| [`duration`](time.md#satkit.duration) | Time duration for arithmetic with `time` objects |
| [`timescale`](time.md#satkit.timescale) | Enumeration of supported time scales (UTC, TAI, TT, etc.) |

## Attitude & Coordinates

| Symbol | Description |
|--------|-------------|
| [`quaternion`](quaternion.md) | Unit quaternion for 3D rotations |
| [`itrfcoord`](itrfcoord.md) | ITRF coordinate with geodetic ↔ Cartesian conversion |
| [`frametransform`](frametransform.md) | Coordinate frame rotation functions |

## Orbit Propagation

| Symbol | Description |
|--------|-------------|
| [`TLE`](tle.md) | Two-Line Element set representation |
| [`sgp4`](tle.md#satkit.sgp4) | SGP4 orbit propagation from TLE/OMM |
| [`propagate`](satprop.md) | High-precision numerical orbit propagation |
| [`propsettings`](satprop.md#satkit.propsettings) | Propagator configuration |
| [`satstate`](satstate.md) | Satellite state with built-in propagation |
| [`kepler`](kepler.md) | Keplerian orbital elements |

## Environment Models

| Symbol | Description |
|--------|-------------|
| [`sun`](sun.md) | Sun position, sunrise/sunset, shadow function |
| [`moon`](moon.md) | Moon position, illumination, and phase |
| [`planets`](planets.md) | Low-precision planetary ephemerides |
| [`density`](density.md) | NRL MSISE-00 atmospheric density model |
| [`gravity`](gravity.md) | Earth gravity acceleration |
| [`gravmodel`](gravity.md#satkit.gravmodel) | Gravity model selection enum |
| [`jplephem`](jplephem.md) | High-precision JPL planetary ephemerides |

## Utilities

| Symbol | Description |
|--------|-------------|
| [`consts`](consts.md) | Physical and astrodynamic constants |
| [`utils`](utils.md) | Utility functions (data file management, etc.) |
