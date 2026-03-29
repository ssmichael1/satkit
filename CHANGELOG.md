# Changelog


## 0.15.0 - 2026-03-29

### Orbit Maneuvers

- **Impulsive maneuvers**: Add `ImpulsiveManeuver` to `SatState` — instantaneous delta-v applied at a scheduled time during propagation. Supports GCRF and RIC frames. Propagation automatically segments at maneuver times and applies delta-v, including backward propagation with sign reversal.
- **Continuous thrust**: Add `ContinuousThrust` and `ThrustProfile` types for constant-acceleration thrust arcs over time windows. Integrated into the force model via the `SatProperties` trait. Supports GCRF and RIC frames with automatic frame rotation.
- **RIC frame transforms**: Add `ric_to_gcrf()` and `gcrf_to_ric()` rotation matrix functions to `frametransform`
- **`Frame::RIC` and `Frame::LVLH`**: New coordinate frame variants with explicit axis definitions in all docs

### Python Bindings

- `satstate.add_maneuver(time, delta_v, frame)` — add impulsive maneuvers to satellite state
- `thrust.constant(accel, start, end, frame)` — create continuous thrust arcs
- `satproperties(thrusts=[...])` — attach thrust arcs to satellite properties
- `frametransform.ric_to_gcrf(pos, vel)` and `frametransform.gcrf_to_ric(pos, vel)`
- Full type stubs and docstrings for all new types

### Code Quality

- **Frame safety**: Replace `_ =>` catch-all match arms in thrust/maneuver frame handling with explicit variants; unsupported frames now panic with descriptive messages
- **Pickle round-trip**: `satstate` pickle now serializes maneuvers; `satproperties` pickle now serializes thrust arcs (backwards compatible with old format)
- **Coordinate frame docs**: Add explicit LVLH and RIC axis definitions across Rust doc comments, Python docstrings, and type stubs
- **Test split**: Split monolithic `test.py` (1,563 lines) into 6 domain-specific test files (`test_time`, `test_coordinates`, `test_frames`, `test_ephemeris`, `test_propagation`, `test_sgp4`)
- **`__init__.pyi`**: Fix `__all__` export list — add missing `frame`, `weekday`, `sgp4_error`, `sgp4_gravconst`, `sgp4_opsmode`, `geodetic`, `gravity_and_partials`, `nrlmsise00`
- **CI**: Update `build.yml` and `release.yml` to discover all test files via `pytest python/test/`

### Documentation

- New tutorial: Orbit Maneuvers (Jupyter notebook)
- Unified Learn section merging User Guide and Tutorials

### Internal

- 133 Rust tests + 37 doc-tests pass
- 71 Python tests pass (4 new pickle round-trip tests, 10 new maneuver/thrust tests)

## 0.14.3 - 2026-03-25

### Performance

- **Binary search interpolation**: Replace linear scan with `partition_point` for dense output step lookup (O(log n) vs O(n))
- **Batch interpolation**: Add `interp_batch` method to `PropagationResult` — single-pass walk over sorted query times (O(n+m) vs O(m log n))
- **Vectorized Python interp**: `result.interp(time_list)` now returns an Nx6 numpy array in one FFI round-trip instead of N individual calls (~4x speedup at 10k points)
- Requires numeris 0.5.7 (`interpolate_batch` support)

## 0.14.2 - 2026-03-25

### Bug Fixes

- **`from_gps_week_and_second`**: Fix week constant (was 168 days instead of 7 days)
- **GPS MJD conversions**: `as_mjd(GPS)` and `from_mjd(GPS)` now return proper Modified Julian Dates instead of days since GPS epoch
- **J2000 constant**: Fix ~96-second error where TT offset was applied in the wrong direction
- **TDB coefficient**: Fix 10x error in `from_mjd(TDB)` amplitude (`0.01657` -> `0.001657`)
- **TDB J2000 reference**: Fix JD-to-MJD offset (`2400000.4` -> `2400000.5`)
- **RFC 3339 timezone offsets**: `from_rfc3339` now correctly parses timezone offsets (e.g., `-05:00`, `+05:30`) instead of silently ignoring them
- **`from_string` parser**: Remove debug `println!`, fix microsecond default causing `-0.000001s` error, guard out-of-bounds access, fix index-shifting bug when removing parsed tokens
- **Cartopy `DownloadWarning`**: Fix warning filter in tutorial notebooks (`category=UserWarning` didn't match cartopy's `DownloadWarning`)
- **Python stubs**: Mark `quaternion.angle` and `quaternion.axis` as `@property` (were incorrectly declared as methods)

### Improvements

- **Duration display**: Show remaining units instead of cumulative totals (e.g., "1 days 1 hours" instead of "1 days 25 hours")
- **Typo fix**: "Coordinate Univeral Time" -> "Coordinated Universal Time"

### Documentation

- New tutorials: Time Systems, Quaternions, SGP4 vs Numerical Propagation
- Add 5 missing tutorials to mkdocs navigation
- Update homepage with quick-start examples and feature summary
- New Rust tests for GPS week/second and RFC 3339 timezone parsing
- New Python test for GPS week/second

## 0.14.1 - 2026-03-21

### Lambert Targeting

- Add Lambert's problem solver using Izzo's algorithm (2015) with Householder 4th-order iteration
- Handles all orbit types (elliptic, parabolic, hyperbolic) and 180-degree transfers
- Multi-revolution solution support
- Rust API: `satkit::lambert::lambert()` in new `lambert` module
- Python API: `satkit.lambert()` with full type stub and docstring
- 9 Rust tests, 6 Python tests

### Documentation Overhaul

- Switch all tutorial plots from Plotly to matplotlib with SciencePlots
- STIX serif fonts, SVG output, colorblind-friendly palette matching numeris docs
- Match numeris site theme: blue grey header with navigation tabs
- Muted steel blue link color (#4a7c96)
- Static SVG plots (density, forces) generated at build time via `docs/examples/gen_plots.py`
- Add Lambert Targeting tutorial with delta-v analysis, pork-chop plot, and orbit visualization
- Add Lambert solver to User Guide and API Reference
- Fix docstring formatting across all Python stubs: bullet style, Returns sections, clickable URLs, Notes admonitions
- Add concrete defs for all `@typing.overload` functions so mkdocstrings renders them
- Add `@typing.overload` signatures to `propresult.interp` for all call patterns
- CI/CD: add cartopy/certifi system deps, generate plots before build, remove plotly

### Internal

- 114 Rust tests pass (9 new Lambert)
- 58 Python tests pass (6 new Lambert)

## 0.14.0 - 2026-03-20

### Breaking: Replace nalgebra with numeris

The `nalgebra` dependency has been replaced with `numeris` 0.5.6 for all linear algebra. The built-in ODE solver module (`src/ode/`) has been removed in favor of the ODE solvers provided by `numeris`. This is a **breaking change** for Rust API consumers; the Python API is unchanged.

### Rust API Changes

- **Math types** (`satkit::mathtypes`): All type aliases now point to `numeris` types instead of `nalgebra`. `Vector<N>`, `Matrix<M,N>`, `Quaternion`, and `DMatrix<T>` remain available with the same names.
- **Vector construction**: `Vector3::new(x, y, z)` is replaced by `numeris::vector![x, y, z]`
- **Matrix construction**: `Matrix3::new(a,b,c,d,e,f,g,h,i)` is replaced by `Matrix3::new([[a,b,c],[d,e,f],[g,h,i]])`
- **Identity matrix**: `Matrix::identity()` is replaced by `Matrix::eye()`
- **Quaternion axis rotations**: `Quaternion::from_axis_angle(&Vector3::z_axis(), θ)` is replaced by `Quaternion::rotz(θ)` (also `rotx`, `roty`)
- **Quaternion vector rotation**: `q.transform_vector(&v)` is replaced by `q * v`
- **Quaternion storage order**: Changed from nalgebra's `[x,y,z,w]` to numeris `[w,x,y,z]` (scalar-first). Component access uses `.w`, `.x`, `.y`, `.z` fields.
- **Block extraction**: `m.fixed_view::<R,C>(i,j)` is replaced by `m.block::<R,C>(i,j)` (returns owned copy)
- **Block insertion**: `m.fixed_view_mut::<R,C>(i,j).copy_from(&src)` is replaced by `m.set_block(i, j, &src)`
- **Matrix inverse**: `m.try_inverse()` (returning `Option`) is replaced by `m.inverse()` (returning `Result`)
- **Cholesky**: `m.cholesky()` now returns `Result<CholeskyDecomposition, LinalgError>` instead of `Option`

### ODE Module Removed

The `satkit::ode` module (7 adaptive solvers, Rosenbrock, ODEState trait, ~2,500 lines) has been removed. Orbit propagation now uses `numeris::ode` solvers directly. The same solver algorithms are available (RKF45, RKTS54, RKV65, RKV87, RKV98, RKV98NoInterp, RODAS4). The `PropSettings` API and `Integrator` enum are unchanged.

### nalgebra Interoperability

If you need nalgebra types for interoperability with other crates, enable the `nalgebra` feature on `numeris`:

```toml
numeris = { version = "0.5.6", features = ["nalgebra"] }
```

This provides zero-cost `From`/`Into` conversions between numeris and nalgebra matrix, vector, and dynamic matrix types. Both libraries use identical column-major storage, so conversions are a `memcpy`.

### Python Bindings

- No breaking changes to the Python API
- All internal conversions updated for numeris types
- Quaternion component order handling updated internally (transparent to Python users)

### Code Simplification

- Use `numeris::vector!` macro throughout instead of `Vector3::from_array([...])`
- Use `Quaternion::rotation_between()` from numeris instead of hand-rolled helper
- Remove `qrot_xcoord`/`qrot_ycoord`/`qrot_zcoord` wrappers; use `Quaternion::rotx`/`roty`/`rotz` directly
- Remove `satkit::filters` module (UKF); use `numeris::estimate::Ukf` instead
- Enable `estimate` feature on numeris

### Dependency Cleanup

- Remove `nalgebra` dependency entirely
- Remove `ndarray` dependency (unused; `numpy` crate re-exports it for Python bindings)
- Remove `cty` dependency; use `std::ffi::{c_double, c_int}` instead
- Remove `once_cell` dependency; use `std::sync::OnceLock` (stable since Rust 1.70)
- Remove redundant reference-based `Add`/`Sub` operator impls from `ITRFCoord` (both types are `Copy`)

### Internal

- Net reduction of ~11,000 lines of code (removed ODE module and filters module)
- 141 Rust tests pass (105 lib + 36 doc)
- 52 Python tests pass


## 0.13.0 - 2026-03-15

### Integrator and Gravity Model Selection

- Add `Integrator` enum to `PropSettings` for selecting ODE solver (RKV98, RKV87, RKV65, RKTS54, RODAS4)
- Add `GravityModel` enum for selecting Earth gravity model (JGM3, JGM2, EGM96, ITU GRACE16)
- Optimize ODE solver hot paths


## 0.12.0 - 2026-03-02

### API Improvements

- **`Instant::utc()` convenience constructor** — shorter alias for `from_datetime()`: `Instant::utc(2024, 1, 1, 12, 0, 0.0)`
- **Explicit time scale method names** — add `as_mjd_utc()`, `as_jd_utc()`, `from_mjd_utc()`, `from_jd_utc()` with explicit scale in the name. Deprecate the old implicit-UTC methods `as_mjd()`, `as_jd()`, `from_mjd()`, `from_jd()` with messages pointing to the new names
- **`Geodetic` named struct** — new `Geodetic { latitude_rad, longitude_rad, height_m }` struct with `latitude_deg()`/`longitude_deg()` helpers and `Display` impl. Add `ITRFCoord::to_geodetic()` returning it. Exported from crate root and prelude
- **`ITRFCoord::distance_to()`** — convenience method returning geodesic distance in meters (wraps `geodesic_distance().0`)
- **`Kepler` convenience constructors** — add `with_true_anomaly()`, `with_mean_anomaly()`, `with_eccentric_anomaly()` to avoid requiring the `Anomaly` enum directly
- **`PropSettings::set_gravity()`** — validated setter for gravity degree/order, returns error if `order > degree`
- **`Duration` now derives `Debug`**
- Remove 10 redundant `&Duration` / `&mut Instant` operator impl variants — both types are `Copy`, so ref-based operators are unnecessary
- **`Display` for `Kepler`** — shows all 6 elements in a readable format

### Breaking Changes

- **`ITRFCoord`: `From<&[f64]>` replaced with `TryFrom<&[f64]>`** — the old impl panicked via `assert!` on wrong-length slices; now returns `Result` with a descriptive error message
- **`as_mjd()`, `as_jd()`, `from_mjd()`, `from_jd()`** are deprecated (still functional). Use `as_mjd_utc()`, `as_jd_utc()`, `from_mjd_utc()`, `from_jd_utc()` or the `_with_scale()` variants

### Performance

- **`drag.rs`**: replace separate `itrf.hae()` + `itrf.latitude_rad()` + `itrf.longitude_rad()` calls with single `itrf.to_geodetic_rad()` (eliminates 2 redundant iterative Bowring geodetic conversions per call)
- **`drag.rs`**: cache `vrel.norm()` in `drag_and_partials` (was computed 3 times, now 1)
- **`point_gravity.rs`**: cache `rsnorm2 * rsnorm` as `rsnorm3` (was computed twice per call)
- **`jplephem.rs`**: replace `(m.transpose() * t)[(0,0)]` with `m.column(0).dot(&t)` in Chebyshev evaluation (avoids transposed-matrix allocation for a dot product)

### Documentation

- **Frame transform reference table** — add module-level docs to `frametransform` with accuracy and description table for all available transforms (ITRF, GCRF, TEME, TIRS, CIRS, MOD)
- **`qteme2gcrf` accuracy note** — prominently document ~1 arcsec approximate accuracy in function docs
- Fix ~20 spelling/grammar errors across the codebase: "Runga-Kutta" → "Runge-Kutta", "Dorumund-Prince" → "Dormand-Prince", "coeffeicient" → "coefficient", "velcocity" → "velocity", and others

### Internal

- **`jplephem.rs`**: extract `ChebySetup` struct and `dispatch_ncoeff!` macro, deduplicating ~40 lines across `body_pos_optimized`/`body_state_optimized`/`barycentric_pos`/`barycentric_state`
- **`itrfcoord.rs`**: deduplicate geodetic conversions in `geodesic_distance()` and `move_with_heading()`
- Curate prelude with explicit imports instead of wildcard re-exports
- Add root-level re-exports for `Frame`, `ITRFCoord`, `Kepler`, `Quaternion`, `Vector3`, `propagate`, `PropSettings`, `SatState`, `SolarSystem`, `TLE`
- Fix "attractur" → "attractor" typo in `point_gravity.rs`

### Python Binding Improvements

- **`satkit.geodetic` class** — new Python class wrapping the Rust `Geodetic` struct with named fields `latitude_rad`, `longitude_rad`, `height_m` and computed properties `latitude_deg`, `longitude_deg`
- **`itrfcoord.geodetic` property** — replaces `geodetic_rad` and `geodetic_deg` tuple properties with a single `geodetic` property returning `satkit.geodetic`
- **`propsettings.precompute_terms`**: `step` argument now accepts `satkit.duration`, `float` (seconds), or `datetime.timedelta`; adds `Precomputed::new_with_step` and `PropSettings::precompute_terms_with_step` in Rust core
- **`propresult.interp`**: `time` argument now accepts `satkit.time` or `datetime.datetime`; also accepts a `list` of either type, returning a `list` of interpolated state arrays
- **`instant_from_pyany`**: internal utility for extracting a single `satkit::Instant` from either `satkit.time` or `datetime.datetime`, for use across Python binding functions


## 0.11.0 - 2026-02-27

### Configurable gravity degree/order and third-body toggles

- **Breaking:** Rename `gravity_order` to `gravity_degree` in `PropSettings` (Rust) and `propsettings` (Python)
- Add separate `gravity_order` parameter for spherical harmonic order (defaults to `gravity_degree`; must be ≤ `gravity_degree`)
- Add `use_sun_gravity` and `use_moon_gravity` toggles to enable/disable third-body perturbations (default `true`)
- Update `gravity()` and `gravity_and_partials()` Python functions: rename `order` kwarg to `degree`, add new `order` kwarg (defaults to `degree`)
- Update Rust `earthgravity` API to accept separate `degree` and `order` parameters
- Add 7 new Rust tests and 3 new Python tests for the new functionality
- Update documentation, type stubs, README, and tutorial notebook


## 0.10.4 - 2026-02-26

### Data directory fix
- Fix data directory resolution for `satkit_data` pip package: the `data/` subdirectory was not being found because the path was missing the `/data` suffix and used the wrong parent level relative to the dylib
- Python `__init__.py` now uses Python's import system to locate `satkit_data` package and set the data directory, rather than relying solely on Rust-side path heuristics
- Update documentation to reflect correct data directory search order


## 0.10.3

  - Clean up LICENSE file so GitHub correctly detects MIT license
  - Point homepage and documentation URLs to satkit.dev
  - Add CNAME for GitHub Pages custom domain (satkit.dev)

## 0.10.1

  - Fix PyO3 0.28 deprecation warnings: add `from_py_object` to all `#[pyclass]` types deriving `Clone`
  - Use SPDX `license = "MIT"` in Cargo.toml and pyproject.toml (fixes crates.io license detection)
  - Fix broken release badge in README (wheels.yml was renamed to release.yml)
  - Rewrite README for clarity and conciseness
  - Fix `release.toml` regex so `cargo release` correctly syncs all version files

## 0.10.0

Starting with this release, Rust and Python versions are tracked with identical version numbers.

  - Add 20 unit tests for core physics modules: frame transforms, point gravity, atmospheric drag, ITRFCoord, Kepler elements, and earth gravity
  - Test count increases from 79 to 99 library tests

## Python 0.9.4
  - ensure python packages are built in release mode (they have been, but adding additoinal flags)
  - Restructure into Cargo workspace with separate Python bindings crate
  - Fix and modernize Python `.pyi` stub files: explicit typed signatures for `time`, `duration`, `itrfcoord`, `quaternion`, `kepler`, `propsettings`, and `propagate` (replaces `*args`/`**kwargs`)
  - Migrate documentation from Sphinx/Read the Docs to MkDocs + Material theme + GitHub Pages
  - Replace `pytz` with stdlib `zoneinfo` in sunrise/sunset tutorial
  - Add `cargo-release` config to sync `pyproject.toml` version from `Cargo.toml`
  - Drop Python 3.8 and 3.9 support; minimum is now Python 3.10
  - Add explanatory markdown to all Jupyter notebook tutorials
  - Fix MathJax rendering in notebook tutorials
  - Remove ReadTheDocs workflow; docs now deployed via GitHub Pages
  - Streamline CI: merge `wheels.yml` and `cargo-publish.yml` into single `release.yml` on `v*` tags
  - Add Python bindings test job to `build.yml` (runs on every push)
  - Move cibuildwheel testing out of `pyproject.toml` into dedicated CI job
  - Update PyO3 to 0.28.2 and numpy to 0.28

## Rust 0.9.4
  - Add prelude to expose commonly-used structs and methods
  - OMM no-longer exposed at top crate level (but is expose via prelude)
  - Make `epoch_instant` function public in OMM
  - OMM supports import of xml files with `omm-xml` feature (enabled by default)

## Python 0.9.3
  - bugfix: Fixed transposed state transition matrix in python bindings for high-precision propagator

## Rust 0.9.3, Python 0.9.2
  - bugfix: Fix (and document in python) the `to_enu` and `to_ned` functions in ITRFCoord

## Rust 0.9.2, Python 0.9.1
  - bugfix: handle high-precision propagation without error when duration is zero
  - Fix Rust documentation for SGP4 to accurately represent sgp4 source
  - Add a "zero" static function for duration to represent zero duration

## Rust 0.9.1
  - Functions that accept time as input now accept structs that implement the new `TimeLike` trait.
  - Add a `chrono` feature that enables interoperability with the chrono crate.  implement `TimeLike` for `chrono::DateTime`

## Rust 0.9.0, Python 0.9.0
  - Support Orbital Mean-Element Messages (OMM) in JSON format
  - Add OMM documentation, tests, and python example
  - Structured output of SGP4 propagator in rust
  - add "as_datetime" function in python for satkit.time (for consistent nomenclature)
  - Rename time-interval boundary nomenclature from `start/stop` to `begin/end` across Rust + Python APIs (including propagation functions and related settings/results).

-----------------

## Rust 0.8.4, Python 0.8.5
  - Add "x", "y", "z", "w" property getters to quaternion in python
  - Add pickle serialize/deserialize tests in python testing
  - allow for duration division by float and by other duration in python

## Rust 0.8.3, Python 0.8.4
  - Fix small python typing bugs to allow setting of properties without errors
  - add additional option for RK integrator
  - Add functions for phase of moon & fraction of moon illuminated
  - typo corrections in code

## Python 0.8.3
  - Use pyo3 0.27.1 (transparent to user but annoying API updates)
  - add typing for property setters that were left out of python bindings (e.g., propsettings)
  - Allow user to set values for kepler object following object creation

## Rust 0.8.2
  - Cleanup of referencing of nalgebra types: commonly used aliases are now all referenced from mathtypes.rs
  - Improve documentation

## Rust 0.8.1
  - Remove un-used dependencies

## Python 0.8.2

### Allow TLE setting of parameters
  - Allow setting of TLE parameters in python bindings

### Support for Python 3.14
  - Build wheels for Python 3.14


## Python 0.8.1

### Fix _version.py
  - export "version" and "_version"

## Python 0.8.0.  Rust 0.8.0

### TLE export to lines
  - functions for generating 69-character TLE lines from a TLE object

### Python use of "anyhow::Result"
  - Use "anyhow::Result" where appropriate in python bindings, as it is used in main code branch and compatible with pyo3

### Code Cleanup
  - Remove "clippy" warnings, mainly in comment indentation and a few inefficient code idioms

### Error Checking on from_datetime
  - Bounds checking on month, day, hour, minute, second

### TLE Fitting
  - Add functionality to generate TLE from high-precision state vectors

### Day of Year
  - Add functionality to compute 1-based day of year in "Instant" structure

## Python 0.7.4 Rust 0.7.4

### Separate Rust and python versions
 - Rust and python versions will now evolve independently

### Fix EOP-induced crashes in frame transform
 - Earth orientation parameters query returns None for invalid dates, which caused frame transform functions that rely on it to panic.  Instead, print warning to stderr in query when out of bounds, and for frame transforms set all values to zero when out of bounds.  Also add function to disable the warning from being shown even once

### Add workflow_dispatch trigger to GitHub Actions
 - Suggestion & contribution by "DeflateAwning"

### Docs cleanup and fix path for download of python scripts
 - Contribution by "DeflateAwning"

### No panic if TLE lines are too short
 - Return error rather than panic if not enough characters in TLE
 - Contribution by "DeflateAwning"


## 0.7.3 - 2025-07-30

### Python data file warning to stderr
 - Print warning to stderr in python bindings if importing with missing datafiles

## 0.7.1 - 2025-07-22
## 0.7.2 - 2025-07-22

### Quaternion python bindings
 - Add ability to create quaternion from (w,x,y,z) values in python bindings
 - Did not merge correctly, so upped version twice.

## 0.7.0 - 2025-07-17

### satkit-data package for python bindings
- For python bindings, necessary data files are now included in a separate package, ``satkit-data``, which is a dependency of ``satkit``


## 0.6.2 - 2025-07-14

### Jupyter notebook example fixes
- Fix jupyter notebook examples to work with time and duration casting as properties


## 0.6.1 - 2025-07-12

### Python comparison operators
- Add complete set of comparison operators for python bindings of time and duration

## 0.5.9 - 2025-07-10

### Python typing fixes
- Python typing fixes, too many to enumerate

### Python duration casting
- Python binding duration casting (e.g., seconds, minutes, hours, days) are now properties, not function calls, e.g. d.seconds instead of d.seconds()


## 0.5.8 - 2025-07-07

### Python fixes
- Fix issue with quaternion rotation on multiple vectors failing if memory is non-contiguous
- Fix indexing error with SGP4 when propagating with multiple TLEs

### Clippy Warnings
- minor code changes to remove rust clippy warnings


## 0.5.7 - 2025-04-25

### Linux ARM
- Include Python binaries for Linux 64-bit ARM

## 0.5.6 - 2025-04-6

### Anyhow
- Use "anyhow" crate for error handling, both in core code and in python bindings (it is very nice!)

## 0.5.5 - 2025-01-27

### Low-Precision Ephemeris
 - Coefficients for low-precision planetary ephemerides did not match paper or JPL website referenced in documentation.  Not sure where original nu mbers came from.  Some plantes (e.g., Mercury) matched.  Others (e.g., Mars) did not, although numbers for Mars did match a google search. Very strange ... regardless, update so they match.  Coefficients for years 1800 to 2050 AD are correct and remain unchanged.
 - Add Pluto as a planet for low-precision ephemerides (it is included in reference paper, but not JPL website)

### Two-Line Element Sets
- Fix issue where two-line element sets confuse satellite name for a line number if satellite name starts with a "1" or a "2"
