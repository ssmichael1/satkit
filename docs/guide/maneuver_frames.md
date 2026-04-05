# Satellite-Local Coordinate Frames for Maneuvers

When you schedule an impulsive burn or a continuous thrust arc in `satkit`, you
specify the delta-v as a 3-vector *plus* a reference frame. The frame tells
satkit how to interpret the components of the vector relative to the
satellite's instantaneous state at the burn time.

`satkit` supports four frames for maneuvers:

| Frame | Axis definition | Best used for |
|---|---|---|
| `GCRF` | Inertial Earth-centred Cartesian | Burns specified in inertial axes |
| `RTN` (a.k.a. `RSW`, `RIC`) | Radial / Tangential / Normal (tied to position) | CCSDS OEM covariance, relative motion, radial/cross-track components |
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

![RTN, NTW, and LVLH axes on a circular orbit](../images/frames_circular.svg)

On a circular orbit ($\gamma = 0$) all three frames coincide up to sign
conventions — RTN's radial is LVLH's $-\hat{z}$, NTW's tangent is RTN's
$\hat{I}$, and so on. The axes overlap in the figure above. The difference
only appears on eccentric orbits.

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

![RTN vs NTW axes on an eccentric orbit showing the flight-path angle γ](../images/frames_eccentric.svg)

On the eccentric orbit ($e = 0.3$, $\nu = 60°$) above, RTN's $\hat{T}$ and
NTW's $\hat{T}$ are no longer the same direction — they differ by the
flight-path angle $\gamma \approx 12.7°$. Likewise RTN's radial $\hat{R}$
and NTW's in-plane-normal $\hat{N}$ differ by the same angle. This is the
geometric origin of every issue in the rest of this guide.

## RTN (RSW / RIC) — position-tied

```
R = r̂                    (radial, outward)
T = ĥ × r̂                (tangential / in-track, perpendicular to R in the orbit plane)
N = ĥ                    (normal / cross-track, along angular momentum)
```

The **tangential** (T) axis is perpendicular to the position vector, *not* to
the velocity vector. For a circular orbit these are the same thing, but for
an eccentric orbit at non-apsidal anomaly they differ by $\gamma$. Two other
common names for this same frame:

- **RSW** — Vallado's convention (R, S=Ŵ×R̂, W=ĥ). The prevailing name in
  astrodynamics textbooks.
- **RIC** — older NASA usage (Radial / In-track / Cross-track), common in
  Clohessy-Wiltshire relative-motion literature.

satkit's canonical name is **RTN** (matching the CCSDS OEM/OMM/ODM
convention); `Frame::RSW` and `Frame::RIC` are provided as compile-time
aliases so code can spell the frame whichever way matches the source
it's transcribing from. In Python, `sk.frame.RSW` and `sk.frame.RIC` are
class-level aliases that compare equal to `sk.frame.RTN`, so
`sk.frame.RSW == sk.frame.RTN` is `True`.

**When to use RTN for maneuvers:**

- You're reading or generating **CCSDS OEM / OMM / ODM messages**, whose
  covariance and delta-v blocks are standardised on the RTN frame.
- You want to specify burn components in the "radial / along-track /
  cross-track" basis that's conventional for **relative motion**
  (Clohessy-Wiltshire, Hill's equations — these papers usually write it
  as RIC).
- You're translating formulas from Vallado (written as RSW) or the CCSDS
  spec (written as RTN).

!!! note "Covariance convention in `satkit`"
    `satkit`'s state-vector uncertainty API — `SatState.set_pos_uncertainty`
    and `set_vel_uncertainty` — accepts any of `GCRF`, `LVLH`, `RTN`, or
    `NTW` via a `frame` parameter. Pass `frame=sk.frame.RTN` if you're
    loading covariance values from a CCSDS OEM file (or use the
    equivalent `sk.frame.RIC` / `sk.frame.RSW` aliases), or
    `frame=sk.frame.LVLH` if you're thinking in nadir/along-track/
    cross-track sigmas.

**When *not* to use RTN:**

- You want "10 m/s along velocity" to mean exactly that, for an eccentric
  orbit. A RTN +T burn of 10 m/s only increases $|\vec{v}|$ by
  $10\cos\gamma$, not by 10. Use NTW instead.

## NTW — velocity-tied

```
T = v̂                    (tangent, along velocity)
W = ĥ                    (cross-track, along angular momentum — same as RTN's N)
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

Geometrically LVLH spans the same orbital plane as RTN. The axes are just
relabeled and sign-flipped:

- LVLH $+\hat{x}$ = RTN $+\hat{T}$ (tangential; perpendicular to R, not V)
- LVLH $-\hat{z}$ = RTN $+\hat{R}$ (radial outward)
- LVLH $-\hat{y}$ = RTN $+\hat{N}$ (cross-track)

So LVLH has the same "in-track is not quite velocity" caveat as RTN for
eccentric orbits.

**When to use LVLH for maneuvers:**

- You're translating GN&C code originally written in LVLH body-frame
  conventions.
- You think in "nadir / in-track / out-of-plane" rather than
  "radial / tangential / normal".

For everything else, RTN or NTW will be more natural.

## The key distinction, in numbers

Consider an eccentric orbit with semi-major axis $a = 8000$ km,
eccentricity $e = 0.3$, at true anomaly $\nu = 60°$. The flight-path angle
at this point is $\gamma \approx 12.7°$. Apply a 10 m/s delta-v in each
frame with the "prograde-like" component equal to 10 m/s:

| Frame | Specification | $\Delta |\vec{v}|$ actually added |
|---|---|---|
| NTW | `(0, 10, 0)` — +T | **10.000 m/s** (exact) |
| RTN | `(0, 10, 0)` — +T (tangential) | 9.755 m/s |
| LVLH | `(10, 0, 0)` — +x | 9.755 m/s (same as RTN +T) |

The loss of 0.245 m/s in the RTN / LVLH case is exactly $10(1 -
\cos\gamma)$. For a high-precision Hohmann transfer injection burn, a
quarter-meter-per-second error is a lot. This is why NTW is the right
choice when you care about "energy added along velocity".

![Velocity-space view of NTW +T vs RTN +T burns on an eccentric orbit](../images/frames_burn_comparison.svg)

The figure above shows a velocity-space view of the same eccentric state.
The black arrow is the current velocity $\vec{v}$; the orange and blue
arrows are two hypothetical 1.5 km/s $\Delta v$ burns (exaggerated from
10 m/s for visibility) specified in NTW and RTN respectively. Both burns
have the same nominal magnitude, but the NTW burn points *along* $\vec{v}$
while the RTN burn points *perpendicular to the position vector* — and
those are not the same direction on an eccentric orbit. The dotted
circles show the resulting $|\vec{v}|$ after each burn; the NTW circle is
farther from the origin, meaning it added more speed.

## Cheat sheet

- **"I want to add 10 m/s prograde on any orbit"** → NTW, or
  `sat.add_prograde(time, 10.0)` (Python) / `ImpulsiveManeuver::prograde`
  (Rust). These helpers pick NTW for you.
- **"I want a radial-out burn of 1 m/s"** → NTW +N
  (`sat.add_radial(time, 1.0)`), or RTN +R if you want strict outward-r.
- **"I want a cross-track burn of 0.5 m/s"** → NTW +W or RTN +N — they're
  the same axis.
- **"I'm porting code from CCSDS OEM that uses RTN"** → RTN
  (`frame.RTN` in Python; `Frame::RTN` in Rust).
- **"I'm porting code from Vallado that uses RSW"** → use `frame.RSW` /
  `Frame::RSW` — it's an alias for the same frame as RTN.
- **"I'm porting code from the Clohessy-Wiltshire / relative-motion
  literature that uses RIC"** → use `frame.RIC` / `Frame::RIC` — also
  an alias for the same frame.
- **"My delta-v vector is already in ECI / J2000 inertial"** → GCRF.
- **"I'm porting GN&C code written in LVLH"** → LVLH.

## Summary table

| Property | GCRF | RTN / RSW / RIC | NTW | LVLH |
|---|---|---|---|---|
| Inertial? | Yes | No | No | No |
| Prograde axis parallel to $\hat{v}$? | — | Only if $\gamma = 0$ | **Always** | Only if $\gamma = 0$ |
| Cross-track axis | — | $+\hat{h}$ | $+\hat{h}$ | $-\hat{h}$ |
| Radial-out axis | — | $+\hat{R}$ | $+\hat{N}$ (only on circular) | $-\hat{z}$ |
| `satkit` covariance uncertainty API | **Yes** | **Yes** | **Yes** | **Yes** |
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
