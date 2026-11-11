# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## 0.24.3 - 2026-11-11

The one change in behaviour from 0.24.2 (text read as OMM CSV) is marked "0.24.3" in [Migrating to 0.24](https://satkit.dev/migration/).

### Added

- `TLE.from_lines`, `TLE.from_file` and `TLE.from_url` take `on_error="raise" | "warn" | "skip"`: `"warn"` and `"skip"` keep the good element sets of a file with bad ones, `"warn"` reporting each one left out as a `satkit.TLEWarning`; Rust gains `TLE::records_from_url`, the per-record form of `from_url` ([#287](https://github.com/ssmichael1/satkit/pull/287))
- OMM CSV: `omm_from_text`, `omm_from_file` and `omm_from_url` (Rust `OMM::from_text`, `from_file`, `from_url`, and the new `OMM::from_csv_string`) read the header-row-plus-one-row-per-object table CelesTrak and Space-Track serve, which is what a CelesTrak GP query returns when no `FORMAT` is given ([#288](https://github.com/ssmichael1/satkit/pull/288))

### Fixed

- Docs: a guide section on catalog numbers above 99,999 (Alpha-5, CelesTrak's 404 for TLEs, OMM as the format to use), troubleshooting entries for that 404 and for one bad line losing a TLE file, CSV in the OMM tutorial, and that tutorial's XML example, whose code cell had been overwritten with a copy of the next one ([#289](https://github.com/ssmichael1/satkit/pull/289))

## 0.24.2 - 2026-11-02

Results that change from 0.24.1 are marked "0.24.2" in [Migrating to 0.24](https://satkit.dev/migration/).

### Added

- `earth_orientation_params::set_table` installs an EOP table given by value, a slice of the new `EopRecord` rows (another provider, a subset or smoothed IERS table, a synthetic table in a test; rows need not be a day apart), and `reset` discards it and loads the data-directory `finals2000A.all` again; Rust only ([#284](https://github.com/ssmichael1/satkit/pull/284))

### Changed

- **Breaking (Rust):** `earth_orientation_params::Error` is `#[non_exhaustive]` and gains `InvalidTable` (rows rejected by `set_table`); an exhaustive match outside satkit needs a wildcard arm ([#284](https://github.com/ssmichael1/satkit/pull/284))
- Type stubs mark every satkit class `@final`, matching the runtime, so type checkers reject subclassing them ([#280](https://github.com/ssmichael1/satkit/pull/280))
- PyO3 0.30 (and numpy 0.30); `pyo3-log`, which does not support PyO3 0.30 yet, is replaced by a small in-tree bridge with the same behaviour: satkit's `Info`-and-above records go to the `satkit.<module>` Python loggers, and records logged while the interpreter is shutting down are dropped instead of panicking ([#279](https://github.com/ssmichael1/satkit/pull/279))

### Fixed

- `TLE.fit_from_states` fits in equinoctial elements, which are not singular at zero eccentricity or inclination: a near-equatorial (e.g. geostationary) arc fitted at an epoch other than its element set's collapsed the inclination to ~1e-10° with the node left at its two-body seed, kilometres off (GOES-19, a day of its own SGP4 states re-epoched by 7 h: 3.9 km, now 0.4 m, the rest being SGP4's own epoch dependence); other orbits give the same elements in about half the iterations. The finite-difference step of the dimensionless elements now scales with their size, so an exactly equatorial arc converges (8 cm) instead of running out of iterations at ~1 m ([#283](https://github.com/ssmichael1/satkit/pull/283))
- Solid Earth tides: the degree-3 order-3 normalization factor was mistyped (0.1394338 instead of sqrt(7/360) = 0.1394433, 7e-5 relative), scaling the smallest tide term; a unit test now checks every normalization factor against its formula
- CI: rustdoc on Rust 1.99 rejected two redundant intra-doc link targets in `orbitprop::srp`; stubtest on mypy 2.4 reports PyO3's auto-generated reflected operator slots, now allowlisted (test extra requires `mypy>=2.4`); rustls 0.23.46 for RUSTSEC-2026-0346/0347/0348 ([#278](https://github.com/ssmichael1/satkit/pull/278))
- Dependencies: rustls-webpki 0.103.16 and webpki-roots / webpki-root-certs 1.0.11, which distrust the compromised emSign intermediates (RUSTSEC-2026-0381/0382/0383) ([#282](https://github.com/ssmichael1/satkit/pull/282))

## 0.24.1 - 2026-09-30

Results that change from 0.24.0 are marked "0.24.1" in [Migrating to 0.24](https://satkit.dev/migration/).

### Added

- Moonrise / moonset (`moon::riseset`, Python `moon.rise_set`: upper limb with refraction, `None` on days without the event) and principal Moon phase times (`moon::phase_times` / `next_phase`), built-in analytic or `use_jpl` ([#270](https://github.com/ssmichael1/satkit/pull/270))
- `sun.rise_set(..., use_jpl=True)` (Rust `sun::riseset_with`) takes the Sun from the JPL ephemeris, within 0.01 s of Skyfield; the analytic Sun (`sun.pos_mod`, `pos_gcrf`, `rise_set`) adds the largest VSOP87 planetary and lunar terms to Meeus's solar coordinates, 3.6″ from JPL (was 43″), so `rise_set` is within 0.5 s (was 2.8 s) ([#272](https://github.com/ssmichael1/satkit/pull/272))

### Changed

- Sun and Moon third-body gravity use Battin's form, which avoids subtracting the nearly equal direct and indirect terms: the perturbation is now at machine precision (was 1e-12 relative for the Sun in LEO); orbits change only at the rounding level ([#275](https://github.com/ssmichael1/satkit/pull/275))

### Fixed

- `propresult.interp` interpolates every time in a `numpy.datetime64` array (0.24.0 used only the first); type stubs: `sgp4` returns a 2- or 3-tuple by `errflag`, and `list[TLE]`, `list[np.datetime64]`, `list[int]` days, array-like `fit_from_states` states and a run-time `output_phi` bool type-check; the API docs list the overloaded functions (`sgp4`, `frametransform`, `sun`/`moon` positions, `propresult.interp`, `time` operators), which were missing ([#274](https://github.com/ssmichael1/satkit/pull/274))
- `sun.rise_set` iterates Algorithm 30 at the event and adds nutation and solar parallax: within 3 s of Skyfield up to 65° latitude (was up to 35 s); clearer polar day/night error; horizon (sea level), dip and UT1 documented ([#268](https://github.com/ssmichael1/satkit/pull/268))

### Docs

- New tutorial on solar eclipse prediction: Besselian elements from the JPL ephemerides, the path of totality with its limits and duration contours for 2024 and 2045, checked against NASA's elements and path table (within 1 km) ([#269](https://github.com/ssmichael1/satkit/pull/269))
- New Rust section on satkit.dev: quick start (Cargo features, data files, logging, errors) and topic pages built on runnable programs in `examples/`, linked from the docs.rs front page ([#271](https://github.com/ssmichael1/satkit/pull/271))
- The eclipse tutorial is renamed "Solar Eclipse Predictions", explains the algorithm step by step with four diagrams, and adds local circumstances for eleven cities; it replaces the older "Eclipse" tutorial, whose averaged Moon distance made totality too short ([#273](https://github.com/ssmichael1/satkit/pull/273))

### CI

- The docs check also runs on pull requests that change the type stubs or the griffe overload extension, which the API pages are built from ([#277](https://github.com/ssmichael1/satkit/pull/277))

## 0.24.0 - 2026-09-26

Upgrading from 0.23: see [Migrating to 0.24](https://satkit.dev/migration/) for what to check and change.

### Added

- `TLE` keeps its classification letter (line 1, column 8: `U`/`C`/`S`), round-tripped by `to_2line()` and `OMM`; Python `TLE.classification` property ([#264](https://github.com/ssmichael1/satkit/pull/264))
- Python `moon.pos_mod` (mean-of-date Moon position), mirroring `sun.pos_mod` ([#254](https://github.com/ssmichael1/satkit/pull/254))
- Warnings go through the Rust `log` facade and, in Python, the `logging` module (`satkit.*` loggers); with no logger installed Rust still prints them to stderr ([#262](https://github.com/ssmichael1/satkit/pull/262))
- The first-use JPL ephemeris download is announced as a warning, so it shows by default in Python ([#265](https://github.com/ssmichael1/satkit/pull/265))
- Every Python function that takes times accepts `numpy.datetime64` scalars and 1-D arrays of any unit, read as UTC; `quaternion * Nx3` and `sgp4` no longer copy their output arrays ([#263](https://github.com/ssmichael1/satkit/pull/263))

### Changed

- **Breaking, wrong results:** `quaternion * Nx3` applied the inverse rotation (0.14.1–0.23.1); `satproperties()` is keyword-only, since positional calls swapped drag and SRP; one-element time lists give one-element results ([#222](https://github.com/ssmichael1/satkit/pull/222))
- **Breaking, wrong results:** `lambert` was wrong for long-way, retrograde, hyperbolic and near-parabolic transfers, and `kepler.from_pv` at inclination π; non-finite input and `r1 == r2` raise ([#250](https://github.com/ssmichael1/satkit/pull/250))
- **Breaking, wrong results:** `moon.pos_gcrf` returns GCRF (it returned mean of date), `planets.heliocentric_pos` uses the J2000 obliquity and correctly scaled outer-planet terms, and `sun.rise_set` returns the input's UTC date ([#251](https://github.com/ssmichael1/satkit/pull/251))
- **Breaking, wrong results:** SGP4 defaults to WGS72 everywhere (Rust `sgp4()` and TLE fitting used WGS84), and its cached initialization follows element and setting changes; mismatched TLE lines and non-TEME / non-Earth OMMs are errors ([#253](https://github.com/ssmichael1/satkit/pull/253))
- **Breaking, wrong results:** drag applies up to 1,000 km (was 700 km); a missing previous-day F10.7 no longer drops the day's space weather (density 35–39 % low); `density.nrlmsise()` uses a `datetime` time ([#249](https://github.com/ssmichael1/satkit/pull/249))
- **Breaking:** Python 3.10 is no longer supported (end of life October 2026); wheels are built for CPython 3.11–3.15 ([#260](https://github.com/ssmichael1/satkit/pull/260))
- **Breaking:** unknown keywords raise `TypeError` (was `ValueError`) in `duration()`, `propsettings()`, `sgp4()`, `propagate()`, `gravity()` and the other keyword bindings ([#236](https://github.com/ssmichael1/satkit/pull/236))
- **Breaking:** Python `TLE.from_lines()` / `from_file()` / `from_url()` always return `list[TLE]`; parse errors name the line and satellite; new Rust `TLE::records()` and optional checksum checks ([#240](https://github.com/ssmichael1/satkit/pull/240))
- **Breaking:** EOP comes only from IERS `finals2000A.all`, so there is none before 1973-01-02; Rust `EopSource` / `source()` are removed and Python `eop_source()` is deprecated ([#235](https://github.com/ssmichael1/satkit/pull/235))
- **Breaking:** the optional `satkit[data]` bundle is dropped, `require_eop_coverage` also checks the table start, and an EOP parser panic is fixed ([#242](https://github.com/ssmichael1/satkit/pull/242))
- **Breaking:** `sgp4([tle], t)` keeps the TLE axis, bad binding inputs raise specific exceptions, and `sgp4` is thread-safe on a shared TLE ([#245](https://github.com/ssmichael1/satkit/pull/245))
- **Breaking:** `time.strptime()`, `from_string()` and `time(<str>)` raise `ValueError` (was `RuntimeError`) for an unparseable string ([#247](https://github.com/ssmichael1/satkit/pull/247))
- **Breaking:** time arithmetic beyond ±292,000 years raises `OverflowError` in Python (it wrapped) and saturates in Rust; `propsettings(gravity_order > gravity_degree)` raises; truncated `finals2000A.all` files are rejected ([#252](https://github.com/ssmichael1/satkit/pull/252))
- **Breaking:** `utils.build_date()` is removed so builds are reproducible; `githash()` / `gittag()` are `"unknown"` outside satkit's own checkout ([#255](https://github.com/ssmichael1/satkit/pull/255))
- **Breaking:** Python `TLE.to_2line()` raises `ValueError` (was `RuntimeError`) when `satnum` is 340000 or above and has no Alpha-5 representation ([#261](https://github.com/ssmichael1/satkit/pull/261))
- **Breaking (Rust):** `DataDirReadOnly` is `DataDirReadOnly { path, reason }`; `update_datafiles()` rejects unknown keywords and fails up front when offline ([#218](https://github.com/ssmichael1/satkit/pull/218))
- **Breaking (experimental ECOM):** coefficients are referred to 1 AU and scale by `(AU / d)²` ([#213](https://github.com/ssmichael1/satkit/pull/213), [#210](https://github.com/ssmichael1/satkit/issues/210))
- **Behaviour change, wrong results:** time parsing applies the UTC offsets `from_rfc3339` / `from_string` dropped or misread, and rejects trailing input and malformed offsets ([#243](https://github.com/ssmichael1/satkit/pull/243))
- **Behaviour change:** float → microsecond conversions round instead of truncating, and the `strptime` `%z` sign is fixed; 0.24 pickles do not load in 0.23 ([#225](https://github.com/ssmichael1/satkit/pull/225))
- **Behaviour change:** UTC before 1972 follows the 1961–1971 "rubber second" model, so pre-1972 labels move by up to 9.9 s ([#223](https://github.com/ssmichael1/satkit/pull/223))

### Deprecated

- Reminder: the Python `as_*` conversions on `time` and `quaternion`, deprecated in 0.23, are removed in 0.25; use `to_*` ([#200](https://github.com/ssmichael1/satkit/pull/200))

### Fixed

- **Wrong results:** TDB − TT had a ~57-year period (up to 3.3 ms off), the JPL ephemerides are evaluated at TDB (was TT), and leap-second edge cases are fixed ([#217](https://github.com/ssmichael1/satkit/pull/217))
- `utils.version()` returns the release version, like `satkit.__version__` (it returned the `git describe` tag, `"unknown"` in the wheels) ([#256](https://github.com/ssmichael1/satkit/pull/256))
- Of several copies of `finals2000A.all` across the search directories, the freshest is read ([#229](https://github.com/ssmichael1/satkit/pull/229))
- `orbitprop::propagate` has its rustdoc again, and the crates.io publish job only runs for `v*` tags ([#226](https://github.com/ssmichael1/satkit/pull/226))
- `help(satkit.satstate)` shows the class documentation again ([#237](https://github.com/ssmichael1/satkit/pull/237))
- Alpha-5 satellite numbers with a sign or malformed digits (`"A-123"` became 99877) are rejected, and the ≥ 340000 error points to OMM ([#261](https://github.com/ssmichael1/satkit/pull/261))
- Speed regressions since 0.23.1 in the EOP, space-weather and UTC conversion hot paths are fixed, with identical results: LEO drag propagation, `gmst`, `qteme2itrf` and NRLMSISE-00 are back to 0.23.1 speed or faster ([#266](https://github.com/ssmichael1/satkit/pull/266))

### Docs

- Covariance docs use RTN instead of LVLH, and the Coordinate Frames tutorial opens with frame tables ([#214](https://github.com/ssmichael1/satkit/pull/214))
- The Time Systems page is the single reference for time in satkit ([#215](https://github.com/ssmichael1/satkit/pull/215))
- New Troubleshooting & FAQ page ([#216](https://github.com/ssmichael1/satkit/pull/216))
- De-duplicated docs, README, CONTRIBUTING, crate docs and CHANGELOG ([#228](https://github.com/ssmichael1/satkit/pull/228))
- `jplephem::barycentric_pos` / `barycentric_state` docs say barycentric, not heliocentric ([#231](https://github.com/ssmichael1/satkit/pull/231))
- ECOM Solar Radiation Pressure tutorial re-executed against current main ([#238](https://github.com/ssmichael1/satkit/pull/238))
- `help()` shows class documentation for the previously undocumented Python classes ([#239](https://github.com/ssmichael1/satkit/pull/239))
- TLE loader docs de-duplicated ([#241](https://github.com/ssmichael1/satkit/pull/241))
- New Migrating to 0.24 page, and time-parsing docstrings brought up to date ([#246](https://github.com/ssmichael1/satkit/pull/246))
- Migration page and CHANGELOG edited for the release: one-line "check old results" list, merged duplicate entries ([#258](https://github.com/ssmichael1/satkit/pull/258))
- Rustdoc has no broken intra-doc links (`RUSTDOCFLAGS="-D warnings" cargo doc`), checked in CI ([#259](https://github.com/ssmichael1/satkit/pull/259))

### CI

- Releases are gated on a green Build; Build uses cargo-deny, and Dependabot updates the Actions pins ([#211](https://github.com/ssmichael1/satkit/pull/211))
- Dependabot: cibuildwheel 4.2.0 → 4.2.1 ([#212](https://github.com/ssmichael1/satkit/pull/212))
- stubtest checks every stub module, and the docs and docstring examples run in CI ([#220](https://github.com/ssmichael1/satkit/pull/220))
- The data cache key includes the download script ([#224](https://github.com/ssmichael1/satkit/pull/224))
- Raising notebooks fail the docs build, docs-only PRs get a docs check, and the Python tests run on more versions and platforms ([#244](https://github.com/ssmichael1/satkit/pull/244))
- Release: `cargo publish --dry-run` before publishing, and PyPI publishes only after crates.io succeeds ([#248](https://github.com/ssmichael1/satkit/pull/248))

### Tests

- Differential tests against ERFA for time scales and frames (new test dependency `pyerfa`) ([#219](https://github.com/ssmichael1/satkit/pull/219))
- Property tests for time, UT1, frames and the Python bindings (new test dependency `hypothesis`) ([#221](https://github.com/ssmichael1/satkit/pull/221))
- Rust time tests slimmed with no coverage lost ([#232](https://github.com/ssmichael1/satkit/pull/232))
- Python tests de-duplicated with no coverage lost ([#233](https://github.com/ssmichael1/satkit/pull/233))

### Internal

- No behaviour change: `rustfmt.toml` packs numeric tables, unused NRLMSISE-00 code removed ([#227](https://github.com/ssmichael1/satkit/pull/227))
- No behaviour change: Python bindings simplified ([#230](https://github.com/ssmichael1/satkit/pull/230))
- No behaviour change: Rust core de-duplicated ([#234](https://github.com/ssmichael1/satkit/pull/234))
- No behaviour change: builds without default features are warning-free, and CI checks it ([#257](https://github.com/ssmichael1/satkit/pull/257))

## 0.23.1 - 2026-09-24

### Fixed

- Cannonball solar radiation pressure scales the 1 AU pressure by `(AU / d)²` instead of holding it at 1 AU (up to ±3.4 %, a few metres per day at GPS); the 1367 W/m² reference is `consts::SOLAR_PRESSURE_1AU` / `consts.solar_pressure_1au` ([#207](https://github.com/ssmichael1/satkit/pull/207), [#206](https://github.com/ssmichael1/satkit/issues/206))
- With `SATKIT_OFFLINE=1` (or without the `download` feature) a missing JPL ephemeris no longer prints a "downloading ..." notice before the offline error ([#208](https://github.com/ssmichael1/satkit/pull/208), [#205](https://github.com/ssmichael1/satkit/issues/205))
