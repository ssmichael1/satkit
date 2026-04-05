# Coordinate Reference Frames

The `satkit.frame` enum identifies a coordinate reference frame throughout
the satkit API — most visibly in the maneuver, thrust, uncertainty, and
frame-transform functions. Frames are passed by value, e.g.:

```python
import satkit as sk

sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
sat.set_pos_uncertainty(sigma, frame=sk.frame.LVLH)
dcm = sk.frametransform.to_gcrf(sk.frame.NTW, pos, vel)
```

## Supported values

| Name | Type | Description |
|------|------|-------------|
| `GCRF` | Inertial | Geocentric Celestial Reference Frame — the default inertial frame |
| `ITRF` | Earth-fixed | International Terrestrial Reference Frame |
| `TEME` | Quasi-inertial | True Equator Mean Equinox (SGP4 output) |
| `CIRS` | Intermediate | Celestial Intermediate Reference System |
| `TIRS` | Intermediate | Terrestrial Intermediate Reference System |
| `EME2000` | Inertial | Earth Mean Equator 2000 (J2000 inertial) |
| `ICRF` | Inertial | International Celestial Reference Frame |
| `LVLH` | Satellite-local | Local Vertical / Local Horizontal (z = nadir) |
| `RTN` | Satellite-local | Radial / Tangential / Normal (CCSDS OEM convention) |
| `RSW` | Alias | Alias for `RTN` — Vallado's name for the same frame |
| `RIC` | Alias | Alias for `RTN` — older NASA / Clohessy-Wiltshire name |
| `NTW` | Satellite-local | Normal / Tangent / Cross-track (velocity-aligned) |

`frame.RSW` and `frame.RIC` are class-level aliases that resolve to the
same enum value as `frame.RTN`, so `sk.frame.RSW == sk.frame.RTN` is
`True` and all three can be used interchangeably.

See the [Theory: Maneuver Coordinate Frames](../guide/maneuver_frames.md)
guide for a side-by-side comparison of the four satellite-local frames
(RTN, NTW, LVLH, GCRF) and guidance on when to use each.

## Enum reference

::: satkit.frame
