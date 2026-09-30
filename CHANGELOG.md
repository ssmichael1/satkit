# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## Unreleased

### Added

- Moonrise / moonset (`moon::riseset`, Python `moon.rise_set`: upper limb with refraction, `None` on days without the event) and principal Moon phase times (`moon::phase_times` / `next_phase`), built-in analytic or `use_jpl` ([#270](https://github.com/ssmichael1/satkit/pull/270))
- `sun.rise_set(..., use_jpl=True)` (Rust `sun::riseset_with`) takes the Sun from the JPL ephemeris, within 0.01 s of Skyfield; the analytic Sun (`sun.pos_mod`, `pos_gcrf`, `rise_set`) adds the largest VSOP87 planetary and lunar terms to Meeus's solar coordinates, 3.6″ from JPL (was 43″), so `rise_set` is within 0.5 s (was 2.8 s) ([#272](https://github.com/ssmichael1/satkit/pull/272))

### Changed

- Sun and Moon third-body gravity use Battin's form, which avoids subtracting the nearly equal direct and indirect terms: the perturbation is now at machine precision (was 1e-12 relative for the Sun in LEO); orbits change only at the rounding level ([#275](https://github.com/ssmichael1/satkit/pull/275))

### Fixed

- `propresult.interp` interpolates every time in a `numpy.datetime64` array (0.24.0 used only the first); type stubs: `sgp4` returns a 2- or 3-tuple by `errflag`, and `list[TLE]`, `list[np.datetime64]`, `list[int]` days, array-like `fit_from_states` states and a run-time `output_phi` bool type-check ([#274](https://github.com/ssmichael1/satkit/pull/274))
- `sun.rise_set` iterates Algorithm 30 at the event and adds nutation and solar parallax: within 3 s of Skyfield up to 65° latitude (was up to 35 s); clearer polar day/night error; horizon (sea level), dip and UT1 documented ([#268](https://github.com/ssmichael1/satkit/pull/268))

### Docs

- New tutorial on solar eclipse prediction: Besselian elements from the JPL ephemerides, the path of totality with its limits and duration contours for 2024 and 2045, checked against NASA's elements and path table (within 1 km) ([#269](https://github.com/ssmichael1/satkit/pull/269))
- New Rust section on satkit.dev: quick start (Cargo features, data files, logging, errors) and topic pages built on runnable programs in `examples/`, linked from the docs.rs front page ([#271](https://github.com/ssmichael1/satkit/pull/271))
- The eclipse tutorial is renamed "Solar Eclipse Predictions", explains the algorithm step by step with four diagrams, and adds local circumstances for eleven cities; it replaces the older "Eclipse" tutorial, whose averaged Moon distance made totality too short ([#273](https://github.com/ssmichael1/satkit/pull/273))

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

## 0.23.0 - 2026-09-24

### Added

- Space weather from its primary sources (GFZ Potsdam observations, NOAA/SWPC 45-day and NASA MSFC MSAFE monthly forecasts) instead of CelesTrak's merged file, ending the silent `Ap = 4` fallback past the 45-day forecast; `isn` is now `-1`; `data_type`, `coverage()` / `status()` and one-time warnings mirror the EOP API. **Breaking:** `tle::Error` is `#[non_exhaustive]`, and `solar_cycle_forecast` / `spaceweather.predicted_f107()` are removed ([#203](https://github.com/ssmichael1/satkit/pull/203), [#202](https://github.com/ssmichael1/satkit/issues/202))

### Changed

- Lockfile refreshed for the release (4 minor crate updates, no new dependencies) ([#204](https://github.com/ssmichael1/satkit/pull/204))
- **Breaking:** the default gravity model is EGM2008 (was EGM96) and the degree/order cap is 70 (was 40): default-model propagations shift by a few metres per day at LEO; pass `gravity_model=gravmodel.egm96` for old results; each acceleration costs about 3× the degree-40 one ([#201](https://github.com/ssmichael1/satkit/pull/201))
- **Breaking:** EGM2008 is added (compiled in to degree 70) and ITU_GRACE16 is no longer compiled in (downloaded on first use, so offline with no copy is an error); solid Earth tides respect each model's tide system (no double-counted permanent tide for `jgm3` / `itugrace16`); the ICGEM parser reads `D` exponents and skips time-variable rows ([#196](https://github.com/ssmichael1/satkit/pull/196), [#195](https://github.com/ssmichael1/satkit/issues/195), [#183](https://github.com/ssmichael1/satkit/issues/183))
- EOP and space-weather refreshes follow [CelesTrak's usage policy](https://celestrak.org/usage-policy.php): no request inside the file's publication cadence (3 h space weather, 24 h EOP) and a conditional `If-Modified-Since` request outside it; `update_datafiles(overwrite=True)` still forces a fetch; CI test jobs no longer refresh ([#199](https://github.com/ssmichael1/satkit/pull/199))
- Earth orientation parameters come from IERS `finals2000A.all` (about a year of predictions instead of six months), with CelesTrak's `EOP-All.csv` as the fallback; `eop_source()` reports the loaded file. Fixed: the `EOP-All.csv` dX/dY celestial-pole offsets were applied 1000× too small (~1 cm LEO, ~5 cm GEO) ([#198](https://github.com/ssmichael1/satkit/pull/198), [#164](https://github.com/ssmichael1/satkit/issues/164))

### Deprecated

- Python: the `as_X` conversions on `time` and `quaternion` (`as_datetime`, `as_mjd`, `as_rotation_matrix`, …) are renamed `to_X` to pair with the `from_X` constructors; the old names emit `DeprecationWarning` and are removed in 0.25 ([#200](https://github.com/ssmichael1/satkit/pull/200))

## 0.22.1 - 2026-09-20

### Changed

- **Breaking (Python):** `quaternion.conj` / `conjugate` are methods (`q.conj()`), and `quaternion.norm`, `tlefitstatus.converged`, `time.day_of_year` and `time.weekday` are properties; the property-vs-method rule is in CONTRIBUTING.md, enforced by a stub-parsing test ([#187](https://github.com/ssmichael1/satkit/pull/187))
- The one-time "using the compiled-in copy of tab5.2a.txt" note is removed (it fired on every fresh install with nothing to act on); `SATKIT_QUIET=1` still silences the corrupt-table warning ([#194](https://github.com/ssmichael1/satkit/pull/194))
- `Precomputed` stores its table behind an `Arc`, so cloning `PropSettings` (every Python `propagate` call) no longer copies it — 55 MB and ~2 ms per call for a one-year table ([#193](https://github.com/ssmichael1/satkit/pull/193), [#190](https://github.com/ssmichael1/satkit/issues/190))

### Fixed

- `TLE::fit_from_states` could return a negative eccentricity and stall on near-circular orbits: the damping is Marquardt-scaled, steps across e = 0 are mapped back, and the result is validated; `TLE::to_2line` rejects e outside [0, 1) ([#192](https://github.com/ssmichael1/satkit/pull/192))
- Lockfile refreshed for the release (13 minor crate updates, no new dependencies) ([#197](https://github.com/ssmichael1/satkit/pull/197))
- Lockfile: rustls 0.23.44 → 0.23.45 for [RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285) (TLS 1.3 handshake, medium), reached through `ureq` by the data downloader in the published wheels ([#188](https://github.com/ssmichael1/satkit/pull/188))

### Docs

- Notebook stderr on satkit.dev renders as a neutral block with an amber edge instead of a red error background ([#191](https://github.com/ssmichael1/satkit/pull/191))
- README: conda-forge version and download badges ([#185](https://github.com/ssmichael1/satkit/pull/185))
- satkit is on conda-forge (`conda install -c conda-forge satkit`, [conda-forge/satkit-feedstock](https://github.com/conda-forge/satkit-feedstock)); the installation docs say so and the in-repo recipe copy is removed ([#184](https://github.com/ssmichael1/satkit/pull/184))
- `THIRDPARTY-DATA.md` states the source, citation, licence and truncation of every compiled-in dataset; it ships in the sdist and wheels and is linked from the README ([#182](https://github.com/ssmichael1/satkit/pull/182))

## 0.22.0 - 2026-09-12

### Added

- `propsettings.initial_step_secs` / `propresult.next_step_secs`: the first adaptive step is derived from the state, tolerances and integrator order instead of numeris' scale-sensitive heuristic, and a follow-on arc can warm-start; `rkv98` with `enable_interp=False` runs the 16-stage `rkv98_nointerp` tableau (24% fewer force evaluations). Requires numeris 0.6 ([#178](https://github.com/ssmichael1/satkit/pull/178))

### Distribution

- **Breaking (Python packaging):** macOS wheels are arm64 only (deployment target 11.0); Intel Macs install from source (`pip install --no-binary satkit satkit`) or from conda-forge's `osx-64` build ([#172](https://github.com/ssmichael1/satkit/pull/172))

### Changed

- **Breaking:** Kepler: `w` is renamed `argp` (Python `w` still works with a `DeprecationWarning`), elements carry their own `mu`, the constructor and setters raise `ValueError` for out-of-domain values, `kepler::Error` is `#[non_exhaustive]`; adds derived quantities (`periapsis`, `apoapsis`, …) and `SatState::from_kepler` ([#168](https://github.com/ssmichael1/satkit/pull/168))
- OMM overhaul: `omm_from_url` returns every field the source provided, new `omm_from_file` / `omm_from_text` (JSON or XML), `TLE.from_omm` / `to_omm`, and `sgp4` dict inputs share the Rust parser. **Breaking (Rust):** `OMM.epoch` is an `Instant`, `omm::Error` is `#[non_exhaustive]`, `Default` is removed ([#173](https://github.com/ssmichael1/satkit/pull/173))
- Propagation steps ~3x faster (LEO, 40x40 field, drag, rkv98): FMA gravity kernels on x86-64, fewer NRLMSISE-00 recomputations, a single Bowring geodetic refinement, degree-limited gravity parsing (contributed by @scottshambaugh, [#175](https://github.com/ssmichael1/satkit/pull/175))
- `update_datafiles()` no longer downloads files compiled into the library (the IERS tables and gravity models are `default: false` in the manifest), and the unused `leap-seconds.list` is removed from the manifest ([#163](https://github.com/ssmichael1/satkit/pull/163))

### Fixed

- Space-weather records are indexed by UTC calendar day; since `Instant` counts leap seconds, the last 37 s of each UTC day read the next day's record (contributed by @scottshambaugh, [#176](https://github.com/ssmichael1/satkit/pull/176))
- `sgp4()` rejects SGP4-XP element sets (TLE ephemeris type 4, OMM `EPHEMERIS_TYPE` 4) instead of running classic SGP4 on the wrong inputs ([#174](https://github.com/ssmichael1/satkit/pull/174))
- `density.nrlmsise(altitude_m, latitude_rad, longitude_rad, time)` converts radians to the degrees the model takes (60° N was evaluated at 1.05° N) ([#171](https://github.com/ssmichael1/satkit/pull/171))
- A corrupt or truncated `tab5.2*.txt` no longer panics the first frame transform: satkit warns and uses the compiled-in copy, and the parser rejects text with no header or too few rows ([#166](https://github.com/ssmichael1/satkit/pull/166))

### Docs

- Every Python docstring states units for dimensioned arguments, returns and attributes (SI, radians unless the name ends in `_deg`); `sgp4` outputs are metres and m/s in TEME, not km ([#169](https://github.com/ssmichael1/satkit/pull/169))
