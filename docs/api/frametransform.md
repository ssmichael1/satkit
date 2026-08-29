# Coordinate Frame Transforms

The `satkit.frametransform` module provides functions for transforming between various coordinate
frames used in satellite tracking and orbit determination. These include multiple variations of "inertial"
coordinate frames, and multiple versions of "Earth-fixed" coordinate frames.

Some notes:

- The IAU-76/FK5 (`_approx`) reduction, GMST (Algorithm 15, Eq. 3-45), the MOD and TEME rotations (Eqs. 3-88 to 3-90) follow [Vallado (2013)](../guide/references.md#vallado2013).
- The frame transforms are defined as arbitrary rotations in a 3-dimensional space. The rotations are a function of time, and are represented as quaternions.
- The full ITRF↔GCRF reduction is the CIO-based procedure of the IERS Conventions (2010), Chapter 5 ([Petit & Luzum 2010](../guide/references.md#petit2010)): polar motion (Eq. 5.3), the Earth rotation angle (Eq. 5.15), the IAU 2006/2000A CIP coordinates $X$, $Y$ and CIO locator $s$ (Tables 5.2a/5.2b/5.2d), and the frame bias to EME2000 (§5.5.4, Eq. 5.36), with Earth orientation parameters from the IERS.

## Dispatch API

The recommended entry points are the frame-enum dispatch functions, which
take a source and destination [`frame`](frame.md) and pick the appropriate
rotation internally:

```python
import satkit as sk

t = sk.time(2024, 1, 1, 12, 0, 0)

# Full IERS 2010 reduction. Keyword arguments are recommended at the call
# site so the source / destination direction is unambiguous; positional
# args work too once you know the order (from, to, tm).
q = sk.frametransform.rotation(
    from_frame=sk.frame.ITRF, to_frame=sk.frame.GCRF, tm=t,
)

# IAU-76/FK5 approximation (~1 arcsec), inertial cluster + ITRF only
q_approx = sk.frametransform.rotation_approx(
    from_frame=sk.frame.ITRF, to_frame=sk.frame.GCRF, tm=t,
)

# Position + velocity (handles the Earth-rotation sweep term)
pos_gcrf, vel_gcrf = sk.frametransform.transform_state(
    from_frame=sk.frame.ITRF, to_frame=sk.frame.GCRF,
    tm=t, pos=pos_itrf, vel=vel_itrf,
)
```

`rotation` accepts any pair of `ITRF`, `GCRF`, `TEME`, `EME2000`, `ICRF`,
`TIRS`, `CIRS` and picks the shortest path through the frame graph (it does
not always pivot through GCRF). Pairs involving the orbit-dependent frames
`LVLH`, `RTN`, `NTW` need a state and so go through
[`to_gcrf`](#satkit.frametransform.to_gcrf) /
[`from_gcrf`](#satkit.frametransform.from_gcrf) instead.

### Which function do I call?

There are three related quaternion entry points; pick by what your frames need:

| Function | Frames it handles | Extra arguments | Returns |
|---|---|---|---|
| [`rotation`](#satkit.frametransform.rotation) | Earth chain only (`ITRF`, `TIRS`, `CIRS`, `GCRF`, `TEME`, `EME2000`, `ICRF`) | — | `quaternion` |
| [`to_gcrf`](#satkit.frametransform.to_gcrf) / [`from_gcrf`](#satkit.frametransform.from_gcrf) | Orbit frames only (`LVLH`, `RTN`, `NTW`) | `pos`, `vel` (GCRF) | 3×3 matrix |
| [`rotation_with_state`](#satkit.frametransform.rotation_with_state) | **All** frames (Earth *and* orbit) | `pos`, `vel` (GCRF) | `quaternion` |

Use [`rotation_with_state`](#satkit.frametransform.rotation_with_state) when a
pair mixes an Earth frame and an orbit frame — e.g. going straight from `TEME`
to `RTN` — without manually composing two transforms through GCRF. Note that
the solution does **not** always go through GCRF: a purely Earth-frame pair
delegates to [`rotation`](#satkit.frametransform.rotation), which picks the
shortest path through the frame graph (e.g. `ITRF`↔`TIRS` is a single
polar-motion rotation, with no IERS reduction paid at all); only pairs that
involve an orbit-dependent frame compose through GCRF. The orbit state is only
consulted when an orbit frame is involved:

```python
# TEME (Earth-fixed SGP4 frame) directly to RTN (orbit-local) in one call.
q = sk.frametransform.rotation_with_state(
    from_frame=sk.frame.TEME, to_frame=sk.frame.RTN,
    tm=t, pos=pos_gcrf, vel=vel_gcrf,
)
```

Any of these functions accept either a `satkit.time` or a `datetime.datetime`
for the `tm` argument.

The per-pair functions below (`qitrf2gcrf`, `qteme2itrf`, `qcirs2gcrf`, …)
remain available for direct use when the source / destination pair is
hard-coded in the surrounding code.

::: satkit.frametransform
