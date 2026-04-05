# Satellite-Local Coordinate Frames for Maneuvers

When you schedule an impulsive burn or a continuous thrust arc in `satkit`, you
specify the delta-v as a 3-vector *plus* a reference frame. The frame tells
satkit how to interpret the components of the vector relative to the
satellite's instantaneous state at the burn time.

`satkit` supports four frames for maneuvers:

| Frame | Axis definition | Best used for |
|---|---|---|
| `GCRF` | Inertial Earth-centred Cartesian | Burns specified in inertial axes |
| `RIC` (a.k.a. `RSW`, `RTN`) | Radial / In-track / Cross-track (tied to position) | Covariance, relative motion, radial/cross-track components |
| `NTW` | Normal / Tangent / Cross-track (tied to velocity) | Prograde/retrograde burns, Hohmann transfers |
| `LVLH` | Local Vertical / Local Horizontal (nadir-pointing) | Porting crewed-spaceflight / GN&C code |

For circular orbits the three non-inertial frames all boil down to the same
three directions. For **eccentric orbits** they differ in subtle but
physically meaningful ways, and choosing the wrong frame will give you a
measurably wrong answer. This guide explains the differences.

## The orbital plane and the three natural directions

Every satellite orbit lies (instantaneously) in a plane determined by the
position vector $\vec{r}$ and the velocity vector $\vec{v}$. The angular
momentum $\vec{h} = \vec{r} \times \vec{v}$ is perpendicular to that plane.
This gives us three natural directions:

1. **Radial** — along $\hat{r}$, outward from Earth's centre
2. **Tangent** — along $\hat{v}$, the direction of motion
3. **Cross-track** — along $\hat{h}$, out of the orbital plane

The three non-inertial frames all use some permutation and sign of these
three directions, but they do **not** all use the same "in-plane prograde"
axis. That's where the subtlety lives.

## The flight-path angle

For a **circular** orbit, the position vector $\vec{r}$ and velocity vector
$\vec{v}$ are exactly perpendicular — the satellite moves along the
circumference. For an **eccentric** orbit, $\vec{r}$ and $\vec{v}$ are only
perpendicular at perigee and apogee. At all other points the velocity has
a radial component, and the angle between $\vec{v}$ and the local
"horizontal" (perpendicular to $\vec{r}$) is called the **flight-path
angle** $\gamma$:

$$
\tan\gamma = \frac{e \sin\nu}{1 + e \cos\nu}
$$

where $e$ is eccentricity and $\nu$ is true anomaly. For a low-eccentricity
LEO ($e \sim 0.001$) the flight-path angle is tiny (~0.06°) and the
distinction below doesn't matter in practice. For a GTO ($e = 0.73$) at
mid-anomaly the flight-path angle can reach ~45° and the distinction
matters a lot.

## RIC (RSW / RTN) — position-tied

```
R = r̂                    (radial, outward)
I = ĥ × r̂                (in-track, perpendicular to R in the orbit plane)
C = ĥ                    (cross-track, along angular momentum)
```

The **in-track** (I) axis is perpendicular to the position vector, *not* to
the velocity vector. For a circular orbit these are the same thing, but for
an eccentric orbit at non-apsidal anomaly they differ by $\gamma$. Other
common names for this frame: **RSW** (Vallado's convention) and **RTN**
(CCSDS OEM/ODM spec). The axes are identical; only the name differs.

**When to use RIC for maneuvers:**

- You want to specify burn components in the "radial / along-track /
  cross-track" basis that's conventional for **relative motion**
  (Clohessy-Wiltshire, Hill's equations).
- You're reading or generating **CCSDS OEM / OMM / ODM messages**, whose
  covariance and delta-v blocks are standardised on the RTN frame.
- You're translating formulas from Vallado or the CCSDS spec that use RSW
  or RTN.

!!! note "Covariance convention in `satkit`"
    `satkit`'s state-vector uncertainty API — `SatState.set_pos_uncertainty`
    and `set_vel_uncertainty` — accepts any of `GCRF`, `LVLH`, `RIC`, or
    `NTW` via a `frame` parameter. Pass `frame=sk.frame.RIC` if you're
    loading covariance values from a CCSDS OEM file, or
    `frame=sk.frame.LVLH` (the default) if you're thinking in
    nadir/along-track/cross-track sigmas.

**When *not* to use RIC:**

- You want "10 m/s along velocity" to mean exactly that, for an eccentric
  orbit. A RIC +I burn of 10 m/s only increases $|\vec{v}|$ by
  $10\cos\gamma$, not by 10. Use NTW instead.

`satkit` exposes RSW and RTN as compile-time aliases for RIC — you can
write `Frame::RIC`, `Frame::RSW`, or `Frame::RTN` in Rust and they all
resolve to the same value. In Python, use `satkit.frame.RIC`.

## NTW — velocity-tied

```
T = v̂                    (tangent, along velocity)
W = ĥ                    (cross-track, along angular momentum — same as RIC's C)
N = T̂ × Ŵ                (in-plane normal to velocity)
```

The **tangent** (T) axis is parallel to the velocity vector *always*,
regardless of eccentricity. That's what makes NTW the natural frame for
"thrust along velocity" scenarios.

The **N** axis is perpendicular to velocity in the orbit plane. For a
circular orbit N coincides with the outward radial direction; for an
eccentric orbit N leans off-radial by the flight-path angle.

**When to use NTW for maneuvers:**

- Prograde / retrograde burns — a pure +T burn of magnitude $\Delta v$ adds
  *exactly* $\Delta v$ to $|\vec{v}|$. This is the only frame for which
  that's true.
- Hohmann transfers, bi-elliptic transfers, station-keeping burns — any
  burn whose intent is "add energy to the orbit along the current
  velocity direction".
- Electric propulsion mission profiles with a long tangential thrust arc.
- Eccentric transfer orbits (GTO, lunar transfer) where the flight-path
  angle matters.

This is the frame described in Vallado §3.3 (eq 3-31).

## LVLH — body-pointing

```
z = −r̂                   (nadir; points toward Earth's centre)
y = −ĥ                   (opposite angular momentum)
x = ŷ × ẑ = ĥ × r̂        (completes the right-handed system)
```

The classical **Local Vertical / Local Horizontal** frame, used by the ISS
and most crewed / Earth-pointing vehicles for attitude control. The
distinguishing feature is that **z points down** — the frame assumes you
want the satellite pointing at Earth.

Geometrically LVLH spans the same orbital plane as RIC. The axes are just
relabeled and sign-flipped:

- LVLH $+\hat{x}$ = RIC $+\hat{I}$ (in-track; perpendicular to R, not V)
- LVLH $-\hat{z}$ = RIC $+\hat{R}$ (radial outward)
- LVLH $-\hat{y}$ = RIC $+\hat{C}$ (cross-track)

So LVLH has the same "in-track is not quite velocity" caveat as RIC for
eccentric orbits.

**When to use LVLH for maneuvers:**

- You're translating GN&C code originally written in LVLH body-frame
  conventions.
- You think in "nadir / in-track / out-of-plane" rather than
  "radial / in-track / cross-track".

For everything else, RIC or NTW will be more natural.

## The key distinction, in numbers

Consider an eccentric orbit with semi-major axis $a = 8000$ km,
eccentricity $e = 0.3$, at true anomaly $\nu = 60°$. The flight-path angle
at this point is $\gamma \approx 12.7°$. Apply a 10 m/s delta-v in each
frame with the "prograde-like" component equal to 10 m/s:

| Frame | Specification | $\Delta |\vec{v}|$ actually added |
|---|---|---|
| NTW | `(0, 10, 0)` — +T | **10.000 m/s** (exact) |
| RIC | `(0, 10, 0)` — +I | 9.755 m/s |
| LVLH | `(10, 0, 0)` — +x | 9.755 m/s (same as RIC +I) |

The loss of 0.245 m/s in the RIC / LVLH case is exactly $10(1 -
\cos\gamma)$. For a high-precision Hohmann transfer injection burn, a
quarter-meter-per-second error is a lot. This is why NTW is the right
choice when you care about "energy added along velocity".

## Cheat sheet

- **"I want to add 10 m/s prograde on any orbit"** → NTW, or
  `sat.add_prograde(time, 10.0)` (Python) / `ImpulsiveManeuver::prograde`
  (Rust). These helpers pick NTW for you.
- **"I want a radial-out burn of 1 m/s"** → NTW +N
  (`sat.add_radial(time, 1.0)`), or RIC +R if you want strict outward-r.
- **"I want a cross-track burn of 0.5 m/s"** → NTW +W or RIC +C — they're
  the same axis.
- **"I'm porting code from Vallado that uses RSW"** → RIC (`frame.RIC`
  in Python; `Frame::RSW` or `Frame::RIC` in Rust, they're the same value).
- **"I'm porting code from CCSDS OEM that uses RTN"** → RIC (`Frame::RTN`
  is an alias in Rust).
- **"My delta-v vector is already in ECI / J2000 inertial"** → GCRF.
- **"I'm porting GN&C code written in LVLH"** → LVLH.

## Summary table

| Property | GCRF | RIC / RSW / RTN | NTW | LVLH |
|---|---|---|---|---|
| Inertial? | Yes | No | No | No |
| Prograde axis parallel to $\hat{v}$? | — | Only if $\gamma = 0$ | **Always** | Only if $\gamma = 0$ |
| Cross-track axis | — | $+\hat{h}$ | $+\hat{h}$ | $-\hat{h}$ |
| Radial-out axis | — | $+\hat{R}$ | $+\hat{N}$ (only on circular) | $-\hat{z}$ |
| `satkit` covariance uncertainty API | **Yes** | **Yes** (default) | **Yes** | **Yes** |
| CCSDS OEM covariance convention | No | **Yes** (as RTN) | No | No |
| Natural for prograde burns | No | Only on circular | **Yes** | Only on circular |
| Natural for crewed-flight GN&C | No | No | No | **Yes** |

## Reference

- Vallado, D., *Fundamentals of Astrodynamics and Applications*, 4th ed.,
  §3.3 — definitions of RSW and NTW frames.
- CCSDS 502.0-B-3, *Orbit Data Messages*, Annex C — RTN frame in CCSDS
  OEM/OMM messages.
- Montenbruck, O. and Gill, E., *Satellite Orbits*, §2.4 — LVLH frame
  definitions and relative motion.
