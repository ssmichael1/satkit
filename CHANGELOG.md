# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## Unreleased

Upgrading from 0.23: see [Migrating to 0.24](https://satkit.dev/migration/) for what to check and change.

### Changed

- **Breaking:** `satproperties()` is keyword-only (positional calls swapped drag and SRP: re-check them); one-element time lists give one-element results. **Wrong results:** `quaternion * Nx3` applied the inverse rotation in 0.14.1–0.23.1 ([#222](https://github.com/ssmichael1/satkit/pull/222))
- **Breaking:** `duration()`, `propsettings()`, `itrfcoord()`, `sgp4()`, `propagate()`, `satstate.propagate()`, `gravity()`, `gravity_and_partials()` and `nrlmsise00()` raise `TypeError` (was `ValueError`) for an unknown keyword ([#236](https://github.com/ssmichael1/satkit/pull/236))
- **Breaking:** Python `TLE.from_lines()` / `from_file()` / `from_url()` always return `list[TLE]` (zero TLEs: `ValueError`); parse errors name the input line and satellite; new Rust `TLE::records()` and optional checksum checks ([#240](https://github.com/ssmichael1/satkit/pull/240))
- **Breaking:** `sgp4([tle], t)` keeps the TLE axis (`(1, 3)`, was `(3,)`); bad inputs raise specific exceptions instead of `RuntimeError` or a meaningless value; `sgp4` is thread-safe on a shared TLE ([#245](https://github.com/ssmichael1/satkit/pull/245))
- **Breaking:** EOP comes only from IERS `finals2000A.all` (not CelesTrak `EOP-All.csv`), so there is none before 1973-01-02 (UT1 = UTC); Rust `EopSource`, `source()`, `CELESTRAK_FILE` and the `InvalidEntry` / `ParseFloat` errors are removed; Python `frametransform.eop_source()` is deprecated (use `eop_coverage()`; removed in 0.25) ([#235](https://github.com/ssmichael1/satkit/pull/235))
- **Breaking:** the optional PyPI `satkit-data` bundle (`satkit[data]`) is dropped; `require_eop_coverage` also refuses spans before the table (Rust `Error::EopCoverage` gains `span_start` / `table_start`); an EOP parser panic and six smaller data-layer defects fixed ([#242](https://github.com/ssmichael1/satkit/pull/242))
- **Breaking:** `time.strptime()`, `time.from_string()` and `time(<str>)` raise `ValueError` (was `RuntimeError`) for a string they cannot parse, like `from_rfc3339` ([#247](https://github.com/ssmichael1/satkit/pull/247))
- **Breaking (Rust):** `DataDirReadOnly` is `DataDirReadOnly { path, reason }` in the EOP, space-weather and `update_data` errors; `update_datafiles()` rejects unknown keywords, and offline, read-only-directory and data warnings say why and where ([#218](https://github.com/ssmichael1/satkit/pull/218))
- **Breaking (experimental ECOM):** coefficients are referred to 1 AU and scale by `(AU / d)²`; old coefficients convert by `(d / AU)²` at their epoch ([#213](https://github.com/ssmichael1/satkit/pull/213), [#210](https://github.com/ssmichael1/satkit/issues/210))
- **Behaviour change:** UTC before 1972 follows the USNO / ERFA "rubber second" model from 1961, so pre-1972 labels convert up to 9.9 s differently; Rust `Instant::UNIX_EPOCH.raw` is 8,000,082 (was 0) ([#223](https://github.com/ssmichael1/satkit/pull/223))
- **Behaviour change:** float → microsecond conversions round instead of truncating, the `strptime` `%z` sign is fixed, and `datetime` / chrono interop and pickles are exact; pickles written by 0.24 do not load in 0.23 ([#225](https://github.com/ssmichael1/satkit/pull/225))
- **Behaviour change:** time parsing applies the UTC offsets `from_rfc3339` / `from_string` dropped or misread, rejects trailing input and malformed offsets, and rounds fractions beyond 6 digits; exact `day_of_week` and dates before −4712 ([#243](https://github.com/ssmichael1/satkit/pull/243))
- Of several copies of `finals2000A.all` across the search directories, the one with the latest observed row is read ([#229](https://github.com/ssmichael1/satkit/pull/229))

### Deprecated

- Reminder: the Python `as_*` conversions on `time` and `quaternion`, deprecated in 0.23, are removed in 0.25; use `to_*` ([#200](https://github.com/ssmichael1/satkit/pull/200))

### Fixed

- **Wrong results, values change:** `lambert` returned wrong velocities for long-way (> 180°) and retrograde transfers and for hyperbolic and near-parabolic ones; `kepler.from_pv` was wrong at inclination π. Non-finite inputs and `r1 == r2` now raise (Rust `lambert::Error` is `#[non_exhaustive]`, with `NonFinite` / `CoincidentPositions`) ([#250](https://github.com/ssmichael1/satkit/pull/250))
- Time scales: TDB − TT had a ~57-year instead of a one-year period (up to 1.7 ms), the JPL ephemerides are evaluated at TDB instead of TT (Moon ~2 m), and several leap-second edge cases are fixed ([#217](https://github.com/ssmichael1/satkit/pull/217))
- `orbitprop::propagate` has its rustdoc again, and the crates.io publish job only runs for `v*` tags ([#226](https://github.com/ssmichael1/satkit/pull/226))
- `help(satkit.satstate)` shows the class documentation again, and a stray doc line is removed from the TLE bindings ([#237](https://github.com/ssmichael1/satkit/pull/237))

### Docs

- Covariance docs use RTN instead of LVLH, the Covariance Propagation tutorial compares RTN and NTW on an eccentric orbit, and the Coordinate Frames tutorial opens with frame tables ([#214](https://github.com/ssmichael1/satkit/pull/214))
- The Time Systems page is rewritten as the single reference for time in satkit: scales, storage, leap seconds, UT1 / EOP coverage and TDB ([#215](https://github.com/ssmichael1/satkit/pull/215))
- New Troubleshooting & FAQ page under Getting Started, organised by symptom ([#216](https://github.com/ssmichael1/satkit/pull/216))
- De-duplicated docs, README, CONTRIBUTING, crate docs and CHANGELOG, and one CI data-download composite action ([#228](https://github.com/ssmichael1/satkit/pull/228))
- `jplephem::barycentric_pos` / `barycentric_state` docs say barycentric, not "Heliocentric" ([#231](https://github.com/ssmichael1/satkit/pull/231))
- ECOM Solar Radiation Pressure tutorial re-executed against current main, with the prose numbers updated to match ([#238](https://github.com/ssmichael1/satkit/pull/238))
- `help()` shows class documentation for `consts`, `frame`, `weekday`, `moon.moonphase`, `satproperties`, `sgp4_error`, `sgp4_gravconst` and `sgp4_opsmode` ([#239](https://github.com/ssmichael1/satkit/pull/239))
- TLE loader docs de-duplicated: `TLE::records()` / Python `TLE.from_lines()` hold the parsing rules ([#241](https://github.com/ssmichael1/satkit/pull/241))
- New Migrating to 0.24 page; troubleshooting entries for the 0.24 changes, time-parsing docstrings brought up to date (and `from_rfc3339` errors give the reason), and stale docs facts fixed ([#246](https://github.com/ssmichael1/satkit/pull/246))

### CI

- Release `preflight` job (version strings + a green Build on the tagged commit) gates the PyPI publish; Build drops its duplicate release build and uses cargo-deny; Dependabot updates the Actions pins ([#211](https://github.com/ssmichael1/satkit/pull/211))
- Dependabot: cibuildwheel 4.2.0 → 4.2.1 ([#212](https://github.com/ssmichael1/satkit/pull/212))
- stubtest checks all eleven stub modules, and the Python examples in the docs and docstrings run in CI ([#220](https://github.com/ssmichael1/satkit/pull/220))
- CI data cache: the cache key includes the download script ([#224](https://github.com/ssmichael1/satkit/pull/224))
- Notebooks that raise fail the docs build, docs-only PRs get a docs-check job, the Python tests also run on 3.10 / 3.14, macOS and Windows, the release workflow is hardened, and the caches are split and pinned ([#244](https://github.com/ssmichael1/satkit/pull/244))
- Release: `cargo publish --dry-run` in the preflight and in Build, PyPI publishes only after crates.io succeeds, and a re-run skips files PyPI already has; migration-page and docs corrections ([#248](https://github.com/ssmichael1/satkit/pull/248))

### Tests

- Differential tests against ERFA (`pyerfa`, a new test dependency) for the time scales, Earth rotation, precession–nutation, TEME and geodetic conversion ([#219](https://github.com/ssmichael1/satkit/pull/219))
- Property tests (proptest, and hypothesis as a new test dependency) for leap seconds, time scales, pickles and frame transforms, plus a weekly 100k-case run ([#221](https://github.com/ssmichael1/satkit/pull/221))
- Rust time tests slimmed with no coverage lost (~215 fewer lines) ([#232](https://github.com/ssmichael1/satkit/pull/232))
- Python tests de-duplicated with no coverage lost (~350 fewer lines) ([#233](https://github.com/ssmichael1/satkit/pull/233))

### Internal

- No behaviour change: `rustfmt.toml` packs numeric tables, unused NRLMSISE-00 code removed, `Instant` / `Duration` ordering derived (~3,200 fewer lines) ([#227](https://github.com/ssmichael1/satkit/pull/227))
- No behaviour change: Python bindings simplified with shared helpers and safe numpy reshapes (~590 fewer lines) ([#230](https://github.com/ssmichael1/satkit/pull/230))
- No behaviour change: Rust core de-duplicated (~100 fewer lines) ([#234](https://github.com/ssmichael1/satkit/pull/234))

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

## 0.21.2 - 2026-08-30

### Changed

- **Breaking (Rust):** `utils::download::Error` and its field-carrying variants are `#[non_exhaustive]`: downstream matches need a wildcard arm and struct patterns a `..` ([#161](https://github.com/ssmichael1/satkit/pull/161))

### Fixed

- Downloads verify against the operating system's trust store (TLS-inspecting corporate proxies work), `SATKIT_CA_BUNDLE` overrides it, and errors name the failing URL; the daily CelesTrak files are parsed before replacing the copy on disk, so an HTML notice page can no longer overwrite the EOP table ([#160](https://github.com/ssmichael1/satkit/pull/160))

### Docs

- GMAT validation page: removed the "What the corpus found" note ([#159](https://github.com/ssmichael1/satkit/pull/159))
- GMAT validation page and README describe the drag corpus: drag orbits, constant/file-driven force models, measured agreement against the drag-only displacement, anomalous-oxygen and F10.7-timing floors ([#157](https://github.com/ssmichael1/satkit/pull/157))

### CI

- Build workflow runs once per change: pull requests build on the PR event only, `main` builds on the merge commit, and a new push cancels the superseded run ([#158](https://github.com/ssmichael1/satkit/pull/158))
