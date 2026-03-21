# Lambert's Problem

Lambert's problem is one of the fundamental problems in orbital mechanics: given two position vectors and the time of flight between them, determine the orbit that connects them. The solution yields the departure and arrival velocity vectors, which are essential for trajectory design, rendezvous planning, and interplanetary mission analysis.

## Problem Statement

Given:

- $\vec{r}_1$ — position at departure (meters)
- $\vec{r}_2$ — position at arrival (meters)
- $\Delta t$ — time of flight (seconds)
- $\mu$ — gravitational parameter of the central body (m$^3$/s$^2$)

Find the velocity vectors $\vec{v}_1$ and $\vec{v}_2$ such that a Keplerian orbit passes through $\vec{r}_1$ at time $t_0$ with velocity $\vec{v}_1$ and through $\vec{r}_2$ at time $t_0 + \Delta t$ with velocity $\vec{v}_2$.

## Transfer Geometry

The transfer angle $\Delta\theta$ between the two position vectors determines the geometry of the transfer. An important special case is the **180-degree transfer** (e.g., Hohmann transfer), where $\vec{r}_1$ and $\vec{r}_2$ are anti-parallel and the orbit plane is not uniquely defined by the two positions alone. In this case, the `prograde` parameter determines the orbit plane.

### Prograde vs. Retrograde

The `prograde` flag resolves the short-way / long-way ambiguity:

- **Prograde** (`prograde=True`): the satellite moves counterclockwise when viewed from above the orbital plane (positive angular momentum in the $z$-direction). This is the standard direction for most Earth-orbiting satellites.
- **Retrograde** (`prograde=False`): the satellite moves clockwise.

## Algorithm

The `satkit` Lambert solver implements **Izzo's algorithm** (2015), the contemporary standard used by ESA's pykep, poliastro, and most modern astrodynamics libraries.

### Lancaster-Blanchard Formulation

The problem is parameterized by a single variable $x \in (-1, 1)$ for elliptic orbits, with $x > 1$ for hyperbolic transfers. The geometry is captured by the parameter $\lambda$:

$$
\lambda^2 = 1 - \frac{c}{s}, \quad s = \frac{|\vec{r}_1| + |\vec{r}_2| + c}{2}
$$

where $c = |\vec{r}_2 - \vec{r}_1|$ is the chord length and $s$ is the semiperimeter. An auxiliary variable $y$ relates $x$ and $\lambda$:

$$
y(x) = \sqrt{1 - \lambda^2(1 - x^2)}
$$

### Time-of-Flight Equation

The non-dimensional time of flight $T = \Delta t \sqrt{2\mu / s^3}$ is expressed as:

$$
T(x) = \frac{\psi + M\pi}{\sqrt{|1-x^2|}} \cdot \frac{1}{1-x^2} + \frac{\lambda y - x}{1-x^2}
$$

where $\psi = \cos^{-1}(xy + \lambda(1-x^2))$ for elliptic orbits and $M$ is the number of complete revolutions. Near the parabolic boundary ($x \approx 1$), a Battin hypergeometric series avoids the numerical singularity.

### Householder Iteration

The equation $T(x) = T_\text{target}$ is solved using **Householder's 4th-order method**, which typically converges in 2-3 iterations. The first three derivatives of $T(x)$ are computed via Izzo's recurrence relations:

$$
T' = \frac{3Tx - 2 + 2\lambda^3 x / y}{1 - x^2}
$$

$$
T'' = \frac{3T + 5xT' + 2(1-\lambda^2)\lambda^3 / y^3}{1 - x^2}
$$

$$
T''' = \frac{7xT'' + 8T' - 6(1-\lambda^2)\lambda^5 x / y^5}{1-x^2}
$$

### Velocity Reconstruction

Velocities are decomposed into radial and tangential components using the solution $x$:

$$
v_{r,1} = \frac{\gamma}{r_1}\left[(\lambda y - x) - \rho(\lambda y + x)\right], \quad v_{t} = \frac{\gamma \sigma (y + \lambda x)}{r}
$$

where $\gamma = \sqrt{\mu s / 2}$, $\rho = (r_1 - r_2)/c$, and $\sigma = \sqrt{1 - \rho^2}$. Angular momentum conservation is guaranteed by construction since the tangential momentum $r \cdot v_t = \gamma \sigma (y + \lambda x)$ is the same at both endpoints.

### Multi-Revolution Solutions

For sufficiently long times of flight, multiple orbits can connect the same two positions with different numbers of complete revolutions $M$. For each $M \geq 1$, up to two solutions exist (short-period and long-period), provided $T > T_\text{min}(M)$. The solver automatically finds all valid multi-revolution solutions.

## Usage

### Python

```python
import satkit
import numpy as np

# Position vectors (meters)
r1 = np.array([7000e3, 0, 0])
r2 = np.array([0, 7000e3, 0])

# Solve Lambert's problem (1-hour transfer)
solutions = satkit.lambert(r1, r2, 3600.0)
v1, v2 = solutions[0]

print(f"Departure velocity: {v1} m/s")
print(f"Arrival velocity:   {v2} m/s")
```

### Hohmann Transfer

```python
import satkit
import numpy as np

r1_mag = 7000e3   # LEO radius (meters)
r2_mag = 42164e3  # GEO radius (meters)

r1 = np.array([r1_mag, 0, 0])
r2 = np.array([-r2_mag, 0, 0])  # 180-degree transfer

# Hohmann transfer time
a_transfer = (r1_mag + r2_mag) / 2
tof = np.pi * np.sqrt(a_transfer**3 / satkit.consts.MU_EARTH)

solutions = satkit.lambert(r1, r2, tof)
v1, v2 = solutions[0]

# Compute delta-v
v_circ_1 = np.sqrt(satkit.consts.MU_EARTH / r1_mag)
v_circ_2 = np.sqrt(satkit.consts.MU_EARTH / r2_mag)
dv1 = np.linalg.norm(v1) - v_circ_1
dv2 = v_circ_2 - np.linalg.norm(v2)
print(f"Delta-v at departure: {dv1:.1f} m/s")
print(f"Delta-v at arrival:   {dv2:.1f} m/s")
print(f"Total delta-v:        {dv1 + dv2:.1f} m/s")
```

### Rust

```rust
use satkit::lambert::lambert;
use satkit::consts::MU_EARTH;

let r1 = numeris::vector![7000.0e3, 0.0, 0.0];
let r2 = numeris::vector![0.0, 7000.0e3, 0.0];

let solutions = lambert(&r1, &r2, 3600.0, MU_EARTH, true).unwrap();
let (v1, v2) = &solutions[0];
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `r1` | array (3,) | *required* | Departure position vector (meters) |
| `r2` | array (3,) | *required* | Arrival position vector (meters) |
| `tof` | float | *required* | Time of flight (seconds), must be positive |
| `mu` | float | Earth $\mu$ | Gravitational parameter (m$^3$/s$^2$) |
| `prograde` | bool | `True` | Prograde (counterclockwise) or retrograde transfer |

### Return Value

A list of `(v1, v2)` tuples, where `v1` and `v2` are 3-element numpy arrays representing the departure and arrival velocity vectors in m/s. Currently returns the zero-revolution solution.

## Applications

- **Orbit transfer design**: compute the delta-v budget for orbit maneuvers
- **Rendezvous planning**: determine phasing and approach trajectories
- **Interplanetary trajectories**: by using the Sun's $\mu$ and heliocentric positions
- **Pork-chop plots**: sweep over departure/arrival dates to find optimal launch windows
- **Initial orbit determination**: given multiple position observations, estimate the orbit

## References

- D. Izzo, "Revisiting Lambert's problem," *Celestial Mechanics and Dynamical Astronomy*, vol. 121, pp. 1-15, 2015.
- R.H. Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, AIAA, 1999.
- D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm Press, 2013, Chapter 7.
