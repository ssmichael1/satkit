# State Vectors, STM & Covariance

This page covers the state representation used by satkit's numerical propagator, the state transition matrix, and how to propagate covariance and impulsive maneuvers.

## State Vector

The propagator integrates a 6-dimensional state $\vec{s} = (\vec{p}, \vec{v})$ — position and velocity in the GCRF (inertial) frame, in SI units. The free function [`satkit.propagate()`](../api/satprop.md) operates directly on these 6-vectors.

For most workflows the higher-level [`satstate`](../api/satstate.md) class is more ergonomic — it bundles state with time, optional covariance, optional maneuvers, and a `propagate()` method that handles everything automatically.

## State Transition Matrix

The state transition matrix $\Phi$ describes the partial derivative of propagated state with respect to initial state:

$$
\Phi(t; t_0)~=~\frac{\partial(\vec{p}(t),\vec{v}(t))}{\partial(\vec{p}_0,\vec{v}_0)}
$$

This 6×6 matrix is computed by augmenting the ODE integration with the partial derivatives of each force term (Montenbruck & Gill Ch. 7). It is the central object for:

- **Covariance propagation** — $\sigma^2_{p,v} = \Phi\,\sigma^2_{p_0,v_0}\,\Phi^T$
- **State estimation** — linearization for least-squares fits and Kalman filters
- **Sensitivity analysis** — "how does the orbit at time $t$ change if I nudge the initial state?"

Pass `output_phi=True` to [`satkit.propagate()`](../api/satprop.md) to compute it. (`rodas4` and `gauss_jackson8` integrators do **not** support STM propagation — use one of the `rkv*` family.)

## The `satstate` Class

Compare:

| Feature | `propagate()` | `satstate.propagate()` |
|---|---|---|
| Position + velocity | Yes | Yes |
| Covariance propagation | Manual (`output_phi` + matrix math) | Automatic via STM |
| Impulsive maneuvers | Not supported | Automatic segmentation |
| Continuous thrust | Via `satproperties` | Via `satproperties` |

### Basic Usage

```python
import satkit as sk
import numpy as np

r = sk.consts.earth_radius + 500e3
v = np.sqrt(sk.consts.mu_earth / r)
sat = sk.satstate(sk.time(2024, 1, 1), np.array([r, 0, 0]), np.array([0, v, 0]))

# Propagate forward 6 hours
new_state = sat.propagate(sat.time + sk.duration.from_hours(6))
print(new_state.pos)  # position at t + 6h
```

### Covariance Propagation

Attach position and/or velocity uncertainty and it will be propagated automatically via the STM. The `set_pos_uncertainty` and `set_vel_uncertainty` methods accept 1-sigma components along any supported satellite-local or inertial frame:

```python
# 1-sigma position uncertainty in LVLH (frame is required — no default)
sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)

# Or in RTN — the convention used in CCSDS OEM messages (RSW and RIC
# are Python-level aliases for the same frame)
sat.set_pos_uncertainty(np.array([10.0, 200.0, 30.0]), frame=sk.frame.RTN)

# Or directly in GCRF
sat.set_pos_uncertainty(np.array([150.0, 150.0, 150.0]), frame=sk.frame.GCRF)

# Set velocity uncertainty the same way — the position block is preserved
sat.set_vel_uncertainty(np.array([0.1, 0.2, 0.05]), frame=sk.frame.LVLH)

# Or set the full 6x6 covariance matrix directly (in GCRF)
sat.cov = my_6x6_matrix

# Propagate -- covariance propagates automatically
new_state = sat.propagate(sat.time + sk.duration.from_hours(6))
print(new_state.cov)  # 6x6 covariance at the new time
```

Supported uncertainty frames are `GCRF`, `LVLH`, `RIC` (= RSW = RTN), and `NTW`. See the [Maneuver Coordinate Frames](maneuver_frames.md) guide for a side-by-side comparison and guidance on which to use.

### Impulsive Maneuvers

Add delta-v events at scheduled times. The propagator automatically segments at each maneuver time and applies the burn:

```python
t_burn = sat.time + sk.duration.from_hours(1)

# Option 1: ergonomic constructors (recommended for the common cases).
# These pick the right coordinate frame for you.
sat.add_prograde(t_burn, 10.0)     # +10 m/s along velocity (NTW +T)
sat.add_retrograde(t_burn, 5.0)    # -5 m/s along velocity
sat.add_radial(t_burn, 2.0)        # +2 m/s radial-out (NTW +N)
sat.add_normal(t_burn, 1.0)        # +1 m/s cross-track (NTW +W)

# Option 2: explicit delta-v vector in a chosen frame.
sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.NTW)  # tangent = along velocity
sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)  # tangential ≠ velocity on eccentric orbits
sat.add_maneuver(t_burn, [0, 0, 5], frame=sk.frame.GCRF)  # 5 m/s in inertial +Z

# Propagate past the burn -- delta-v is applied at t_burn
new_state = sat.propagate(sat.time + sk.duration.from_hours(3))
```

Multiple maneuvers can be added and will be applied in chronological order. Backward propagation reverses the maneuvers automatically.

Supported maneuver frames are `frame.GCRF`, `frame.RTN` (a.k.a. RSW, RIC),
`frame.NTW`, and `frame.LVLH`. For circular orbits the three non-inertial
frames are equivalent; for eccentric orbits they differ by the flight-path
angle in ways that matter for precision maneuver planning. See the
[Maneuver Coordinate Frames](maneuver_frames.md) guide for a detailed
comparison and recommendations on which frame to use.

## See Also

- **Tutorial**: [Orbit Maneuvers](../tutorials/Orbit Maneuvers.ipynb) — impulsive maneuver examples; [Covariance Propagation](../tutorials/Covariance Propagation.ipynb) — STM-based covariance evolution.
- **Theory**: [Force Model](forces.md) for what's being integrated; [ODE Integrators](integrators.md) for the integration mechanics; [Maneuver Coordinate Frames](maneuver_frames.md) for RTN/NTW/LVLH definitions.
- **API**: [`satkit.satstate`](../api/satstate.md), [`satkit.propagate`](../api/satprop.md).
