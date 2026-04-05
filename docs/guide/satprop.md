# High-Precision Orbit Propagation

The `satkit` package includes a high-precision orbit propagator, which predicts future (and past) positions and velocities of satellites by integrating the known forces acting upon the satellite.

The propagator and force models follow very closely the excellent and detailed description provided in the book [**"Satellite Orbits: Models, Methods, Applications"**](https://doi.org/10.1007/978-3-642-58351-3) by O. Montenbruck and E. Gill. A brief description is provided below; for more detail please consult this reference.

The propagator, like the rest of the package, is written natively in [Rust](https://www.rust-lang.org). This includes both the force model and the Runge-Kutta ODE integrator. This allows the propagator to run extremely fast, even when being called from Python.

## Mathematical Description

The orbit propagator integrates the forces acting upon the satellite to produce a change in velocity, then integrates the velocity to produce a change in position. Mathematically, this is:

$$
\vec{v}(t_1)~=~\vec{v}(t_0) + \int_{t_0}^{t_1}~\vec{a}\left ( t,~\vec{p}_{t},~\vec{v}_{t} \right ) ~dt
$$

$$
\vec{p}(t_1)~=~\vec{p}(t_0) + \int_{t_0}^{t_1}~\vec{v}(t)~dt
$$

where $\vec{p}(t)$ and $\vec{v}(t)$ are the position and velocity vectors, respectively, of the satellite, and $\vec{a}\left (t,~\vec{p}(t),~\vec{v}(t) \right )$ is the acceleration vector, which is simply the forces acting upon the satellite (a function of time and satellite position & velocity) divided by the satellite mass.

## Modelled Forces

For a ballistic satellite orbiting the Earth, the forces acting upon the satellite are accurately known. These are:

### Earth Gravity

The Earth's gravity is the largest force acting on the satellite. For simpler Keplerian orbit models, the Earth is approximated as a point mass, which is valid if the Earth were spherical with a constant density. However, the Earth actually has a much more complex shape. The force of gravity is computed by taking an expansion of Legendre polynomials with coefficients determined by shape and density of the Earth. For example, the Earth bulges at the center, creating extra mass that pulls inclined orbits toward the equator and causes *precession*. This is commonly known as the **J2** term in the Legendre expansion.

Multiple experiments have attempted to measure the Legendre coefficients for Earth gravity. The University of Potsdam maintains a catalog of gravity models [here](https://icgem.gfz-potsdam.de/home). The `satkit` package is able to compute gravity using several of the models published at this site with a user-settable degree of accuracy.

### Solar Gravity

The sun acts as a point mass pulling the satellite toward it. The sun also pulls the earth towards it, so the force from the sun produces an acceleration in the geocentric frame that must be subtracted from the acceleration due to the Earth:

$$
\vec{a}~=~GM_{sun}~\left [ \frac{\vec{p} - \vec{p}_{sun}}{|\vec{p} - \vec{p}_{sun}|^3}  - \frac{\vec{p}_{sun}}{|\vec{p}_{sun}|^3} \right ]
$$

where $G$ is the gravitational constant, $M_{sun}$ is the mass of the sun, and $\vec{p}_{sun}$ is the position of the sun.

### Lunar Gravity

The moon, like the sun, acts as a point mass pulling the satellite towards it, and the expression for the acceleration of the satellite due to the moon is very similar to above:

$$
\vec{a}~=~GM_{moon}~\left [ \frac{\vec{p} - \vec{p}_{moon}}{|\vec{p} - \vec{p}_{moon}|^3}  - \frac{\vec{p}_{moon}}{|\vec{p}_{moon}|^3} \right ]
$$

where $G$ is the gravitational constant, $M_{moon}$ is the mass of the moon, and $\vec{p}_{moon}$ is the position of the moon.

### Drag

At about 600km altitude and below, there is enough atmosphere to impose a drag force on the satellite. The force takes a standard form:

$$
\vec{a}~=~-\frac{1}{2}~C_d~\frac{A}{m}~\rho~\vec{v}_r~|\vec{v}_r|
$$

where $C_d$ is the unitless coefficient of drag (generally a number between 1.5 and 3), $A$ is the satellite cross-sectional area, $m$ is the mass, $\rho$ is the air density, and $\vec{v}_r$ is the satellite velocity relative to the surrounding air (which is generally assumed to be zero in the *Earth-fixed* frame). The propagator uses the [NRL-MSISE00](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/) density model, and includes space weather effects.

### Solar Radiation Pressure

Momentum transfer to the satellite from solar photons that are scattered or absorbed adds an additional force:

$$
\vec{a}~=~-P_{sun}~\cos(\theta)~\frac{A}{m}\left [ (1-\epsilon) \hat{p}_{sun} + 2 \epsilon \cos(\theta) \hat{n} \right ]
$$

where $P_{sun}\approx 4.56\cdot10^{-6}~Nm^{-2}$ is the solar radiation pressure in the vicinity of the Earth, $A\cos(\theta)$ is the cross-section of the satellite illuminated by the sun ($\theta$ is the incidence angle), $\epsilon$ is the fraction of light scattered by the satellite (1-$\epsilon$ is absorption), $m$ is the satellite mass, and $\hat{n}$ is the half-angle between the incoming and reflected rays. The propagator includes an additional computation that considers if the sun is shadowed by the Earth. Satellite surfaces can be complex, so the most-accurate representation of the expression above would integrate $dA \cos(\theta)$ over the full cross-sectional area.

The `satkit` package greatly simplifies the above expression by assuming that the surface normals all point in the direction of the sun (this is *mostly* true for active satellites that have large solar panels pointed at the sun). The expression above is then simplified:

$$
\vec{a}~=~-P_{sun}~C_R\frac{A}{m}\hat{p}_{sun}
$$

To include this force, the user sets a static $C_R\frac{A}{m}$ value in the `satproperties` object used in the propagation. The integrator takes care of computing the sun position and Earth occlusion.

## Un-modeled Forces

The high-precision propagator does not include several additional forces that are generally small. These include:

- Solid tides of the Earth
- Radiation pressure of Earth albedo
- Gravitational force of other planets
- Relativistic effects

## ODE Solver

The high-precision propagator supports two families of integrators:

1. **Adaptive Runge-Kutta methods** (the default), with embedded error estimation for automatic step-size control. A proportional-integral-derivative (PID) controller adjusts the step size to keep errors within user-defined bounds. The Butcher tableaux are provided by the *delightful* web page of [Jim Verner](https://www.sfu.ca/~jverner/).
2. **Gauss-Jackson 8**, an 8th-order fixed-step multistep predictor-corrector method specialised for 2nd-order ODEs (Berry & Healy 2004). This is the integrator used in GMAT, STK, ODTK, and the US Space Surveillance Network for high-precision orbit propagation. For smooth long-duration propagation it typically uses 3-10× fewer force evaluations than `rkv98` at comparable accuracy.

### Integrator Choices

Several integrators are available, selected via the `integrator` parameter of `propsettings`:

| Integrator | Order | Type | Dense Output | Notes |
|---|---|---|---|---|
| `rkv98` | 9(8) | adaptive RK, 26 stages | 9th-order | Default. Best accuracy for precision work. |
| `rkv98_nointerp` | 9(8) | adaptive RK, 16 stages | None | Same stepping accuracy, faster when interpolation is not needed. |
| `rkv87` | 8(7) | adaptive RK, 21 stages | 8th-order | Good balance of speed and accuracy. |
| `rkv65` | 6(5) | adaptive RK, 10 stages | None | Faster, moderate accuracy. |
| `rkts54` | 5(4) | adaptive RK, 7 stages | None | Fastest. Good for quick propagations. |
| `rodas4` | 4(3) | Rosenbrock, 6 stages | None | L-stable (implicit). For stiff problems. No STM support. |
| `gauss_jackson8` | 8 | fixed-step multistep | 5th-order Hermite | High-efficiency for smooth long-duration propagation. No STM support. |

Higher-order integrators can take larger time steps for the same accuracy, so despite more stages per step, they often require fewer total function evaluations. For most orbit propagation tasks, the default `rkv98` is recommended. For stiff problems (re-entry, very low perigee), `rodas4` uses an implicit method with analytical Jacobian. For long-duration high-precision propagation of smooth orbits (days to months), `gauss_jackson8` with an appropriate fixed step is typically the fastest choice.

```python
import satkit as sk

# Use the faster Tsitouras 5(4) integrator
settings = sk.propsettings(integrator=sk.integrator.rkts54)

# Use the 8(7) integrator with EGM96 gravity
settings = sk.propsettings(
    integrator=sk.integrator.rkv87,
    gravity_model=sk.gravmodel.egm96,
    gravity_degree=16,
)

# Use RODAS4 for a very low orbit with high drag
# (implicit solver handles stiff dynamics from rapid density changes)
settings = sk.propsettings(
    integrator=sk.integrator.rodas4,
    gravity_degree=8,
)
satprops = sk.satproperties(cd_a_over_m=2.2 * 0.01 / 1.0)

# Use Gauss-Jackson 8 for a long-duration GEO propagation with
# a 120-second fixed step (good for MEO/GEO regimes)
settings = sk.propsettings(
    integrator=sk.integrator.gauss_jackson8,
    gj_step_seconds=120.0,
)
```

!!! note
    The `rodas4` and `gauss_jackson8` integrators do not support state
    transition matrix propagation (`output_phi=True`). Attempting to use
    `output_phi=True` with either will raise a `RuntimeError`.

!!! note "Integrator step budget: `max_steps`"
    Every integrator stops with a max-steps error once it exceeds
    `propsettings.max_steps` total steps. This is a runaway-propagation
    safeguard, not a quality knob. The default of `1_000_000` comfortably
    covers the longest realistic arcs — roughly 700 days of `gauss_jackson8`
    at a 60 s step, or millions of adaptive RK steps at typical tolerances.
    Lower it if you want to fail-fast on configurations that would take a
    very long time; raise it if you hit the limit on a genuine long-arc
    propagation.

!!! note "Gauss-Jackson step-size selection"
    `gauss_jackson8` uses a fixed step size (`gj_step_seconds`) which the
    user must choose based on the orbit regime. Typical values:

    * **LEO** (400-800 km): 30-60 s
    * **MEO**: 60-300 s
    * **GEO**: 300-600 s
    * **HEO / eccentric transfer**: use `rkv98` instead — GJ8's fixed step
      wastes accuracy at apogee and misses resolution at perigee.

    GJ8 is also unsuitable for propagation across discontinuities such as
    eclipse boundaries or impulsive maneuvers — use an adaptive RK method
    for those cases. The integrator needs ≥ 9 steps of startup, so
    propagations shorter than ~`9 × gj_step_seconds` will fail; use an
    adaptive RK integrator for such short intervals.

### Gravity Model Selection

The gravity model used in propagation can be selected via the `gravity_model` parameter. Available models are:

| Model | Description |
|---|---|
| `egm96` | Earth Gravitational Model 1996 (default) |
| `jgm3` | Joint Gravity Model 3 |
| `jgm2` | Joint Gravity Model 2 |
| `itugrace16` | ITU GRACE 2016 |

The `gravity_degree` and `gravity_order` parameters control the maximum degree and order of the spherical harmonic expansion.

## Future Propagation

When propagating into the future (beyond the date range of downloaded data files), the following behavior applies:

- **Earth Orientation Parameters** ($\Delta UT1$, $x_p$, $y_p$): The last available values from the EOP file are used (constant extrapolation). This is much more accurate than defaulting to zero, since these parameters change slowly.

- **Space Weather** (F10.7 solar flux, Ap geomagnetic index): When historical space weather data is not available, the [NOAA/SWPC solar cycle forecast](https://www.swpc.noaa.gov/products/solar-cycle-progression) is used for predicted F10.7 values (out ~5 years). The Ap geomagnetic index is not included in the forecast and defaults to 4. If neither source is available, fixed defaults are used (F10.7 = 150, Ap = 4). Run `satkit.utils.update_datafiles()` to download the latest forecast.

## State Transition Matrix

The state transition matrix, $\Phi$ describes the partial derivative of the propagated position and velocity with respect to the initial position and velocity:

$$
\Phi~=~\frac{\partial (\vec{p},\vec{v})}{\partial (\vec{p}_0,\vec{v}_0)}
$$

This 6x6 matrix can be computed by numerically integrating the partial derivatives of the accelerations described above, and is useful for "propagating" the 6x6 state covariance, via the equation below. Details for computing $\Phi$ are found in Montenbruck & Gill.

$$
\sigma^2_{p,v}~=~\Phi~\sigma^2_{p_0,v_0}~\Phi^T
$$

The state transition matrix is also useful when estimating a satellite state from a series of observations (e.g., radar or optical).

The `satkit` package includes the option to compute the state transition matrix when solving for the new state.

## The `satstate` Class

The free function `propagate()` operates on raw 6-element state vectors. The `satstate` class wraps a state vector with additional capabilities:

| Feature | `propagate()` | `satstate.propagate()` |
|---|---|---|
| Position + velocity | Yes | Yes |
| Covariance propagation | Manual (output_phi + matrix math) | Automatic via STM |
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

Attach position and/or velocity uncertainty, and it will be propagated automatically via the state transition matrix. The `set_pos_uncertainty` and `set_vel_uncertainty` methods accept the 1-sigma components along any supported satellite-local or inertial frame:

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

Supported uncertainty frames are `GCRF`, `LVLH`, `RIC` (= RSW = RTN), and `NTW`. See the [Maneuver Coordinate Frames](maneuver_frames.md) guide for a side-by-side comparison of the orbital frames and guidance on which to use.

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

## Forces vs Altitude

The plot below, modeled on a similar plot in Montenbruck and Gill, gives a sense of the various contributors to satellite acceleration as a function of altitude:

![Acceleration vs Altitude](../images/force_vs_altitude.svg)
