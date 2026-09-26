# Rust API Reference

The Rust API reference is on **[docs.rs/satkit](https://docs.rs/satkit)**, built from the doc comments of each release.

Where to find what:

- **Crate root**: the most used types, re-exported: `Instant`, `Duration`, `TimeScale`, `Frame`, `ITRFCoord`, `TLE`, `Kepler`, `SatState`, `PropSettings`, `propagate`, `Vector3`, `Quaternion`. [`satkit::prelude`](https://docs.rs/satkit/latest/satkit/prelude/) brings these and a few more into scope with `use satkit::prelude::*;`.
- **Modules** by topic: [`frametransform`](https://docs.rs/satkit/latest/satkit/frametransform/) (rotations), [`itrfcoord`](https://docs.rs/satkit/latest/satkit/itrfcoord/), [`tle`](https://docs.rs/satkit/latest/satkit/tle/), [`omm`](https://docs.rs/satkit/latest/satkit/omm/), [`sgp4`](https://docs.rs/satkit/latest/satkit/sgp4/), [`orbitprop`](https://docs.rs/satkit/latest/satkit/orbitprop/) (numerical propagation), [`jplephem`](https://docs.rs/satkit/latest/satkit/jplephem/), [`lpephem`](https://docs.rs/satkit/latest/satkit/lpephem/), [`kepler`](https://docs.rs/satkit/latest/satkit/kepler/), [`lambert`](https://docs.rs/satkit/latest/satkit/lambert/), [`earthgravity`](https://docs.rs/satkit/latest/satkit/earthgravity/), [`nrlmsise`](https://docs.rs/satkit/latest/satkit/nrlmsise/) (atmospheric density), [`consts`](https://docs.rs/satkit/latest/satkit/consts/).
- **Data management**: [`utils`](https://docs.rs/satkit/latest/satkit/utils/) (`update_datafiles`, data directories, offline mode), [`earth_orientation_params`](https://docs.rs/satkit/latest/satkit/earth_orientation_params/) and [`spaceweather`](https://docs.rs/satkit/latest/satkit/spaceweather/) (loading and coverage of the tables).
- **Errors**: each module's `Error` enum is documented in that module; see [Errors](index.md#errors).

The Python API reference is under [API Reference](../api/index.md). The two APIs mirror each other closely, with Python-style names (`satkit.time` for `Instant`, `satkit.satstate` for `SatState`); the migration guide lists [Rust-only changes](../migration.md#rust-api) per release.
