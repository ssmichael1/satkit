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
| [`frame`](frame.md) | Coordinate reference frame identifier (GCRF, ITRF, RTN, NTW, LVLH, …) |
| [`frametransform`](frametransform.md) | Coordinate frame rotation functions |

## Orbit Propagation

| Symbol | Description |
|--------|-------------|
| [`TLE`](tle.md) | Two-Line Element set representation |
| [`sgp4`](tle.md#satkit.sgp4) | SGP4 orbit propagation from TLE/OMM |
| [`omm_from_url`, `omm_from_file`, `omm_from_text`](tle.md#satkit.omm_from_url) | Read OMMs (JSON, XML, CSV) as dictionaries |
| [`TLEWarning`](tle.md#satkit.TLEWarning) | Warning for a TLE record skipped with `on_error="warn"` |
| [`propagate`](satprop.md) | High-precision numerical orbit propagation |
| [`propresult`](satprop.md#satkit.propresult) | Propagation result: final state, dense-output interpolation, statistics |
| [`propsettings`](satprop.md#satkit.propsettings) | Propagator configuration |
| [`satproperties`](satprop.md#satkit.satproperties) | Drag, radiation-pressure (cannonball / ECOM) and thrust properties |
| [`ecomparams`](satprop.md#satkit.ecomparams) | ECOM empirical solar-radiation-pressure coefficients |
| [`thrust`](satprop.md#satkit.thrust) | Continuous-thrust arc |
| [`satstate`](satstate.md) | Satellite state with built-in propagation |
| [`kepler`](kepler.md) | Keplerian orbital elements |
| [`lambert`](lambert.md) | Lambert two-point boundary value problem solver |

## Environment Models

| Symbol | Description |
|--------|-------------|
| [`sun`](sun.md) | Sun position, sunrise/sunset, shadow function |
| [`moon`](moon.md) | Moon position, illumination, phase, rise/set and phase times |
| [`planets`](planets.md) | Low-precision planetary ephemerides |
| [`density`](density.md) | NRLMSISE-00 atmospheric density model |
| [`spaceweather`](spaceweather.md) | Space-weather indices ($F_{10.7}$, $A_p$) and their coverage |
| [`gravity`](gravity.md) | Earth gravity acceleration |
| [`gravmodel`](gravity.md#satkit.gravmodel) | Gravity model selection enum |
| [`jplephem`](jplephem.md) | High-precision JPL planetary ephemerides |

## Utilities

| Symbol | Description |
|--------|-------------|
| [`consts`](consts.md) | Physical and astrodynamic constants |
| [`utils`](utils.md) | Utility functions (data file management, etc.) |
