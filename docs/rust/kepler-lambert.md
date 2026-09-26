# Kepler and Lambert

Two-body tools that need no data files. The [Keplerian Elements](../guide/kepler.md) and [Lambert's Problem](../guide/lambert.md) guides cover the conventions and algorithms; this page shows the Rust API.

## Keplerian elements: `kepler::Kepler`

`Kepler` holds the semi-major axis `a` (m), eccentricity `eccen`, inclination `incl`, RAAN `raan`, argument of periapsis `argp` and true anomaly `nu` (radians), and the gravitational parameter `mu` (Earth by default; `with_mu` / `from_pv_with_mu` for another body). Construct it with:

- `Kepler::try_new(a, e, i, raan, argp, anomaly)`, which validates the elements and returns `kepler::Error::InvalidElement` for one out of its domain, or `Kepler::new`, which does not; `anomaly` is `Anomaly::True`, `Anomaly::Mean` or `Anomaly::Eccentric`;
- `Kepler::from_pv(r, v)` from an inertial position and velocity (elliptical orbits only).

`to_pv()` returns the Cartesian state, `propagate(&dt)` moves the anomaly along a two-body orbit, and the derived quantities are methods: `period()`, `mean_motion()`, `mean_anomaly()`, `eccentric_anomaly()`, `periapsis()`, `apoapsis()`, `semiparameter()`, `specific_energy()`, `angular_momentum()`, `flight_path_angle()`, `argument_of_latitude()`, `true_longitude()`. `SatState::from_kepler` starts a numerical propagation from elements.

## Lambert's problem: `lambert::lambert`

`lambert(&r1, &r2, tof, mu, prograde)` returns the departure and arrival velocities of every transfer from `r1` to `r2` in `tof` seconds, using Izzo's algorithm: the zero-revolution solution first, then a short- and long-period pair per revolution count that fits in the time of flight. `prograde` picks the transfer direction (angular momentum along +z or −z). Out-of-domain input returns a `lambert::Error`.

## Example

```rust
--8<-- "examples/kepler_lambert.rs"
```

```text
$ cargo run --example kepler_lambert
a = 7000.0 km, period = 97.14 min, perigee alt = 614.9 km
|r| = 6993.940 km, |v| = 7.5526 km/s, round trip da = 3.7e-9 m
mean anomaly after T/4: 120.000 deg (true anomaly 120.099 deg)
Lambert (1 solution(s)): v1 = [644.1, 8451.9, 0.0] m/s, v2 = [-2724.5, -4120.0, 0.0] m/s
transfer: a = 9482.4 km, e = 0.2719, perigee alt = 526.3 km, arrival miss = 1.86e-8 m
```

## See also

- [Keplerian Elements](../guide/kepler.md) and [Lambert's Problem](../guide/lambert.md), with the [Keplerian Elements](../tutorials/Keplerian%20Elements.ipynb) and [Lambert Targeting](../tutorials/Lambert%20Targeting.ipynb) tutorials.
- [`kepler`](https://docs.rs/satkit/latest/satkit/kepler/) and [`lambert`](https://docs.rs/satkit/latest/satkit/lambert/) on docs.rs.
