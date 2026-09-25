# Coordinate Reference Frames

The `satkit.frame` enum identifies a coordinate reference frame throughout
the satkit API — most visibly in the maneuver, thrust, uncertainty, and
frame-transform functions. Frames are passed by value, e.g.:

```python
import satkit as sk

sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
sat.set_pos_uncertainty(sigma, frame=sk.frame.RTN)
dcm = sk.frametransform.to_gcrf(sk.frame.NTW, pos, vel)
```

## Supported values

### Earth and celestial frames

| Name | Type | Description |
|------|------|-------------|
| `GCRF` | Inertial | Geocentric Celestial Reference Frame — the default inertial frame |
| `ICRF` | Inertial | International Celestial Reference Frame — same axes as GCRF, origin at the solar-system barycenter (frame transforms rotate only; they do not shift the origin) |
| `EME2000` | Inertial | Earth Mean Equator and Equinox of J2000.0 (the J2000 frame) |
| `TEME` | Quasi-inertial | True Equator Mean Equinox (SGP4 output) |
| `CIRS` | Intermediate | Celestial Intermediate Reference System |
| `TIRS` | Intermediate | Terrestrial Intermediate Reference System |
| `ITRF` | Earth-fixed | International Terrestrial Reference Frame |

### Satellite frames

| Name | Description |
|------|-------------|
| `RTN` | Radial / Transverse / Normal — R along position (CCSDS OEM / CDM convention) |
| `RSW` | Alias for `RTN` — Vallado's name for the same frame |
| `RIC` | Alias for `RTN` — older NASA / Clohessy-Wiltshire name |
| `NTW` | Normal / Tangent / Cross-track — T along velocity |
| `LVLH` | Local Vertical / Local Horizontal — RTN relabeled (x = T, y = −N, z = −R) |

RTN follows the CCSDS Orbit Data Messages convention ([CCSDS 502.0-B-3](../guide/references.md#ccsds502)), RSW and NTW are defined in [Vallado (2013)](../guide/references.md#vallado2013), §3.3, and RIC descends from [Clohessy & Wiltshire (1960)](../guide/references.md#clohessy1960).

`frame.RSW` and `frame.RIC` are class-level aliases that resolve to the
same enum value as `frame.RTN`, so `sk.frame.RSW == sk.frame.RTN` is
`True` and all three can be used interchangeably.

The [Coordinate Frames](../tutorials/Coordinate Frames.ipynb) tutorial
describes both groups in more detail, [Frame Transforms](frametransform.md)
covers which function rotates between which frames, and
[Theory: Maneuver Coordinate Frames](../guide/maneuver_frames.md) compares
the satellite frames for maneuvers and covariance.

## Enum reference

::: satkit.frame
