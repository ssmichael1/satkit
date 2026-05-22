# Coordinate Frame Transforms

The `satkit.frametransform` module provides functions for transforming between various coordinate
frames used in satellite tracking and orbit determination. These include multiple variations of "inertial"
coordinate frames, and multiple versions of "Earth-fixed" coordinate frames.

Some notes:

- Most of the algorithms in this module are from the book "Fundamentals of Astrodynamics and Applications" by David Vallado.
- The frame transforms are defined as arbitrary rotations in a 3-dimensional space. The rotations are a function of time, and are represented as quaternions.
- The rotation from the Geocentric Celestial Reference Frame (GCRF) to the Earth-Centered Inertial (ECI) frame is defined by the International Astronomical Union (IAU), available at <https://www.iers.org/>. See IERS Technical Note 36 for the latest values.

## Dispatch API

The recommended entry points are the frame-enum dispatch functions, which
take a source and destination [`frame`](frame.md) and pick the appropriate
rotation internally:

```python
import satkit as sk

t = sk.time(2024, 1, 1, 12, 0, 0)

# Full IERS 2010 reduction
q = sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, t)

# IAU-76/FK5 approximation (~1 arcsec), inertial cluster + ITRF only
q_approx = sk.frametransform.rotation_approx(sk.frame.ITRF, sk.frame.GCRF, t)

# Position + velocity (handles the Earth-rotation sweep term)
pos_gcrf, vel_gcrf = sk.frametransform.transform_state(
    sk.frame.ITRF, sk.frame.GCRF, t, pos_itrf, vel_itrf
)
```

`rotation` accepts any pair of `ITRF`, `GCRF`, `TEME`, `EME2000`, `ICRF`,
`TIRS`, `CIRS` and picks the shortest path through the frame graph (it does
not always pivot through GCRF). Pairs involving the orbit-dependent frames
`LVLH`, `RTN`, `NTW` need a state and so go through
[`to_gcrf`](#satkit.frametransform.to_gcrf) /
[`from_gcrf`](#satkit.frametransform.from_gcrf) instead.

The per-pair functions below (`qitrf2gcrf`, `qteme2itrf`, `qcirs2gcrf`, …)
remain available for direct use when the source / destination pair is
hard-coded in the surrounding code.

::: satkit.frametransform
