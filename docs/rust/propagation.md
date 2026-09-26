# Numerical Propagation

The high-precision propagator integrates the full force model (spherical-harmonic gravity, Sun and Moon, solid tides, relativity, drag, solar radiation pressure, thrust) in the GCRF. The [Force Model](../guide/forces.md), [ODE Integrators](../guide/integrators.md) and [State Vectors, STM & Covariance](../guide/satstate.md) guides describe the models; this page shows the Rust API.

!!! note "Data files"
    The propagator needs the JPL ephemeris for the Sun and Moon (a one-time 102 MB download on first use) and the Earth orientation table, and drag reads the space-weather tables. With the `download` feature these are fetched on first use; see [Data Files](../getting-started/datafiles.md). Without Earth orientation data `propagate` returns `orbitprop::Error::EopUnavailable` rather than integrate with a mis-oriented gravity field.

## Settings: `PropSettings`

`PropSettings::default()` has degree-4 EGM2008 gravity, Sun and Moon gravity, solid tides, relativity, space-weather-driven drag, the Verner 9(8) integrator, 1e-8 tolerances and dense output on. Set fields directly (`abs_error`, `rel_error`, `integrator`, `use_spaceweather`, ...) and the gravity degree with `set_gravity(degree, order)`, which validates it (degree 70 at most). `require_eop_coverage` makes a propagation outside the Earth orientation table an error instead of a warning. For many propagations over one interval, `precompute_terms(&begin, &end)` caches the Sun, Moon and frame rotations.

## Satellite properties: `SatProperties`

Drag and radiation pressure come from a `SatProperties`, passed as `Some(&props)` (or `None` for neither). `SatPropertiesSimple::new(cd_a_over_m, cr_a_over_m)` holds constant values in m²/kg, with optional continuous thrust (`with_thrust`) and ECOM coefficients (`with_ecom`). Implement the `SatProperties` trait yourself for values that change with time or state (attitude-dependent area, a mass that decreases with thrust); its methods receive the time and the state.

## `propagate` and dense output

`orbitprop::propagate(&state, &begin, &end, &settings, satprops)` integrates a 6-vector state (GCRF position in m, velocity in m/s; `SimpleState`), or a 6×7 state whose last six columns carry the state transition matrix (`CovState`, initialized to identity). It can run backwards (`end` before `begin`). The `PropagationResult` holds the final state (`state_end`), step counts and, with `enable_interp`, the dense output: `interp(&t)` and `interp_batch(&times)` return the state at any time inside the interval without re-integrating.

## `SatState`: maneuvers and covariance

`SatState` carries a time, a GCRF state, an optional 6×6 covariance and a list of impulsive maneuvers. `SatState::propagate(&t, Some(&settings), satprops)` splits the arc at each maneuver and applies its Δv, and propagates the covariance through the state transition matrix (P₁ = Φ P₀ Φᵀ). Set the covariance with `set_pos_uncertainty` / `set_vel_uncertainty` (1-σ values in GCRF, RTN, NTW or LVLH) or `set_cov`; add burns with `ImpulsiveManeuver::prograde`, `retrograde`, `radial_out`, `normal`, or a vector in a chosen frame.

## Example

```rust
--8<-- "examples/propagate_leo.rs"
```

```text
$ cargo run --release --example propagate_leo
264 accepted steps, 5544 function evaluations
after 1 day: |r| = 6871.383 km, |v| = 7.6096 km/s
interpolated at 12 h: |r| = 6875.495 km
drag vs no drag after 1 day: 7104.5 m apart
1 m/s prograde burn at 6 h: RTN offset after 1 day = [0.7, -192.5, 0.0] km
1-sigma after 1 day (RTN): 56 m, 3830 m, 9 m
```

The drag figures depend on the space-weather record for June 2024. Build with `--release` for propagation: the debug build is much slower.

## See also

- [Force Model](../guide/forces.md), [ODE Integrators](../guide/integrators.md), [State Vectors, STM & Covariance](../guide/satstate.md) and [Maneuver Coordinate Frames](../guide/maneuver_frames.md).
- [GPS Example](../tutorials/GPS%20Example.ipynb), [Orbit Maneuvers](../tutorials/Orbit%20Maneuvers.ipynb) and [Covariance Propagation](../tutorials/Covariance%20Propagation.ipynb).
- [`orbitprop`](https://docs.rs/satkit/latest/satkit/orbitprop/) on docs.rs.
