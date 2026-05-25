# ODE Integrators

The numerical propagator integrates the equations of motion forward (or backward) in time using one of two families of solvers:

1. **Adaptive Runge-Kutta** (default) — embedded error estimation, PID step-size controller, accept/reject logic. Butcher tableaux courtesy of the *delightful* web page of [Jim Verner](https://www.sfu.ca/~jverner/).
2. **Gauss-Jackson 8** — 8th-order fixed-step multistep predictor-corrector specialised for 2nd-order ODEs (Berry & Healy 2004). The dominant integrator in operational astrodynamics codes (GMAT, STK, ODTK, U.S. Space Surveillance Network). For smooth long-duration propagation it typically uses 3-10× fewer force evaluations than `rkv98` at comparable accuracy.

## Integrator Choices

Select via the `integrator` parameter of `propsettings`:

| Integrator | Order | Type | Dense Output | Notes |
|---|---|---|---|---|
| `rkv98` | 9(8) | adaptive RK, 26 stages | 9th-order | Default. Best accuracy for precision work. |
| `rkv98_nointerp` | 9(8) | adaptive RK, 16 stages | None | Same stepping accuracy, faster when interpolation is not needed. |
| `rkv87` | 8(7) | adaptive RK, 21 stages | 8th-order | Good balance of speed and accuracy. |
| `rkv65` | 6(5) | adaptive RK, 10 stages | None | Faster, moderate accuracy. |
| `rkts54` | 5(4) | adaptive RK, 7 stages | None | Fastest. Good for quick propagations. |
| `rodas4` | 4(3) | Rosenbrock, 6 stages | None | L-stable (implicit). For stiff problems. No STM support. |
| `gauss_jackson8` | 8 | fixed-step multistep | 5th-order Hermite | High-efficiency for smooth long-duration propagation. No STM support. |

Higher-order RK methods can take larger steps for the same accuracy, so despite more stages per step they often require *fewer* total function evaluations. The default `rkv98` covers almost all situations. Reach for `rodas4` on stiff problems (very low perigee, re-entry); reach for `gauss_jackson8` on long smooth arcs.

```python
import satkit as sk

# Faster adaptive RK
settings = sk.propsettings(integrator=sk.integrator.rkts54)

# 8(7) integrator with a higher-degree gravity model
settings = sk.propsettings(
    integrator=sk.integrator.rkv87,
    gravity_model=sk.gravmodel.egm96,
    gravity_degree=16,
)

# Implicit Rosenbrock for stiff low-perigee dynamics
settings = sk.propsettings(
    integrator=sk.integrator.rodas4,
    gravity_degree=8,
)

# Gauss-Jackson 8 for long-duration GEO with a 120 s fixed step
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

## Error Tolerances

The adaptive RK integrators (and `rodas4`) accept the same step every time both:

- the **absolute error** estimate falls below `abs_error` (default `1e-8`)
- the **relative error** estimate falls below `rel_error` (default `1e-8`)

For sub-meter precision over a day, tighten both to `1e-10` to `1e-13`. For coarse mission planning, `1e-6` to `1e-8` is usually fine.

`gauss_jackson8` ignores both — its accuracy is set by the fixed step `gj_step_seconds`.

## See Also

- **Tutorial**: [GPS Example](../tutorials/GPS Example.ipynb) — runs all integrators against the same GPS arc and compares accuracy / cost.
- **Theory**: [Force Model](forces.md) for the right-hand side of the ODE; [State Vectors, STM & Covariance](satstate.md) for what's being integrated.
- **API**: [`satkit.integrator`](../api/satprop.md), [`satkit.propsettings`](../api/satprop.md).
