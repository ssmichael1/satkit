# High-Precision Orbit Propagator

API reference for the numerical (force-model) propagator: `satkit.propagate`
integrates a Cartesian state under a configurable force model (gravity field,
Sun/Moon, drag, radiation pressure, tides, relativity). It is one of three
ways to move a satellite forward in time in satkit — pick by what you have and
how accurate you need to be:

| you have | use | what it does |
|---|---|---|
| a Cartesian state and a force model | **this page** — `satkit.propagate` / `satstate.propagate` | numerical integration, the most accurate; needs the ephemeris and EOP data files |
| a TLE / OMM element set | [`satkit.sgp4`](tle.md) | the analytic SGP4/SDP4 theory the elements were fitted with; km-level accuracy for days; no data files |
| classical elements and an unperturbed (two-body) model | [`satkit.kepler.propagate`](kepler.md) | advances the anomaly only; instantaneous, no data files; see the [Keplerian Elements guide](../guide/kepler.md) |

For background on the numerical propagator, see:

- [Force Model](../guide/forces.md) — modeled forces and physical motivation
- [Empirical SRP: ECOM](../guide/ecom.md) — the experimental ECOM solar-radiation-pressure model behind `satproperties.ecom`
- [ODE Integrators](../guide/integrators.md) — solver families, step-size selection, tolerances
- [State Vectors, STM & Covariance](../guide/satstate.md) — state representation, STM, covariance propagation, maneuvers
- Tutorial: [GPS Example](../tutorials/GPS Example.ipynb)
- [References](../guide/references.md) — primary sources for each force model and integrator
- Tutorial: [ECOM Solar Radiation Pressure](../tutorials/ECOM Solar Radiation Pressure.ipynb)

::: satkit.propagate

::: satkit.propresult

::: satkit.propsettings

::: satkit.integrator

::: satkit.tidemodel

::: satkit.propstats

::: satkit.satproperties

::: satkit.ecomparams

::: satkit.thrust
