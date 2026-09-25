# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## Unreleased

### Changed

- **Breaking (experimental ECOM):** ECOM coefficients are referred to 1 AU and the ECOM acceleration is scaled by `(AU / d)²` with the satellite–Sun distance, like the cannonball term, so fitted coefficients no longer drift ±3.4 % with the season; coefficients from an unscaled implementation convert by `(d / AU)²` at their epoch ([#213](https://github.com/ssmichael1/satkit/pull/213), [#210](https://github.com/ssmichael1/satkit/issues/210))

### Fixed

- Time scales: TDB − TT had a ~57-year instead of a one-year period (up to 1.7 ms wrong), and the JPL ephemerides are now evaluated at TDB instead of TT (Moon ~2 m); 00:00:00 UTC after a leap second was built 1 s early, `23:59:60.x` could not be entered, the 1972-01-01 leap-second entry was 9 s late, pre-1970 times of day printed negative, and UT1 − UTC was interpolated across the leap-second step on the preceding day (0.5 s off); docs now say that a naive `datetime` is local time, as in Python ([#217](https://github.com/ssmichael1/satkit/pull/217))
- Messages, stubs and docs cleanup: `utils.update_datafiles()` raises `TypeError` for an unknown keyword (was silently ignored) and fails before printing anything under offline mode; offline errors say whether `set_offline(True)` or `SATKIT_OFFLINE` is the cause; an unwritable data directory (read-only filesystem, another user's directory) is reported with its path (Rust: `DataDirReadOnly` gains `path` / `reason`); EOP and space-weather warnings and the `propagate` EOP errors name the Python functions, and no longer suggest a refresh for epochs before 1973 or past the MSAFE forecast; the CelesTrak throttle hint is limited to GP queries; space weather reads the freshest copy of each file across the data search directories; the `kepler.w` `DeprecationWarning` points at the caller; `sgp4` stub overloads for `errflag` (an `int32` array) and a runnable example; docs fixes for `SATKIT_DATA_URL` scope, EOP coverage from 1973, tides in `propagate`, and the ephemeris trigger ([#218](https://github.com/ssmichael1/satkit/pull/218))

### Docs

- Covariance docs use RTN (R exactly along position) instead of LVLH, whose x axis is only approximately along-track, and the Covariance Propagation tutorial adds an eccentric-orbit comparison of RTN and NTW (T exactly along velocity); the Coordinate Frames tutorial opens with tables of the Earth/celestial and satellite frames ([#214](https://github.com/ssmichael1/satkit/pull/214))
- New Troubleshooting & FAQ page under Getting Started, organised by symptom: EOP/space-weather warnings, data directories and downloads, TLS/proxy/offline setups, installation, API renames and results that look wrong ([#216](https://github.com/ssmichael1/satkit/pull/216))

### CI

- Release workflow: a `preflight` job replaces the redundant pre-release test job — it verifies the version strings and waits for a green Build run on the tagged commit — and the PyPI publish now depends on it, so a mis-versioned tag can no longer reach PyPI while being refused by crates.io. The Build workflow drops its duplicate `cargo build --release` pass, runs `cargo doc` on Ubuntu only, caches the sdist job's compile, and replaces the per-PR cargo-audit job with cargo-deny (RustSec advisories plus a dependency licence allowlist and crate-source check; policy in `deny.toml`, weekly cargo-audit schedule kept). Dependabot keeps the GitHub Actions pins current ([#211](https://github.com/ssmichael1/satkit/pull/211))

## 0.23.1 - 2026-09-24

### Fixed

- Cannonball solar radiation pressure scales the 1 AU pressure by `(AU / d)²` with the satellite–Sun distance instead of holding it at its 1 AU value, a seasonal error of up to ±3.4 % (a few metres per day at GPS); the reference stays at 1367 W/m², now `consts::SOLAR_PRESSURE_1AU` / `consts.solar_pressure_1au` ([#207](https://github.com/ssmichael1/satkit/pull/207), [#206](https://github.com/ssmichael1/satkit/issues/206))
- With `SATKIT_OFFLINE=1` (or without the `download` feature) a missing JPL ephemeris no longer prints a "downloading ..." notice before the offline error ([#208](https://github.com/ssmichael1/satkit/pull/208), [#205](https://github.com/ssmichael1/satkit/issues/205))

## 0.23.0 - 2026-09-24

### Added

- Space weather from its primary sources: the observed record is GFZ Potsdam's `Kp_ap_Ap_SN_F107_since_1932.txt` (CC BY 4.0; its CC BY-NC sunspot column is not ingested, so `isn` is now `-1`), the 45-day forecast NOAA/SWPC's and the monthly forecast NASA MSFC's MSAFE, assembled into one table with the 81-day averages computed across the forecast (they reproduce CelesTrak's published columns to 0.24 sfu). MSAFE's monthly rows carry a climatological Ap, which ends the silent quiet-time `Ap = 4` fallback past the 45-day forecast (up to 2x too little density during a storm). `SpaceWeatherRecord::data_type` (`OBS`/`OBS-P`/`INT`/`PRD`/`PRM`), `spaceweather::coverage()` / `status()` / `disable_space_weather_time_warning()` and one-time warnings mirror the EOP API; `init_from_path` / `init_from_bytes` are exposed to Python and read the GFZ table; CelesTrak's merged space-weather file is no longer downloaded. The Data Files documentation is split into Data Files / Data Directories / Downloads and Refresh / Data Coverage and `satkit.spaceweather` gains an API reference page. ([#203](https://github.com/ssmichael1/satkit/pull/203), [#202](https://github.com/ssmichael1/satkit/issues/202))
- EGM2008 gravity model (`GravityModel::EGM2008` / `gravmodel.egm2008`, Pavlis et al. 2012, public domain), compiled in to degree 70 like EGM96 / JGM2 / JGM3; `Gravity::tide_system` and `earthgravity::TideSystem` record each model's permanent-tide convention from the ICGEM `tide_system` header or, for the headerless JGM files, from the C20 value; `earthgravity::ensure_loaded` / `is_loaded` load a model with a typed error instead of a panic ([#196](https://github.com/ssmichael1/satkit/pull/196), [#195](https://github.com/ssmichael1/satkit/issues/195), [#183](https://github.com/ssmichael1/satkit/issues/183))

### Changed

- Lockfile refreshed for the release (4 minor crate updates, no new dependencies) ([#204](https://github.com/ssmichael1/satkit/pull/204))
- **Breaking (Rust):** `tle::Error` is `#[non_exhaustive]`, like `kepler::Error`, `omm::Error` and `download::Error`: downstream matches need a wildcard arm, so the variants 0.22.1 added (and future ones) land without breaking compiles ([#203](https://github.com/ssmichael1/satkit/pull/203))
- **Breaking:** the `solar_cycle_forecast` module, `spaceweather.predicted_f107()` and the `predicted-solar-cycle.json` download are removed: the NOAA/SWPC solar-cycle JSON was the density model's fallback for dates past the space-weather record, and with MSAFE in the table it was no longer reached. `update_datafiles()` no longer fetches it ([#203](https://github.com/ssmichael1/satkit/pull/203))
- **Breaking:** the default gravity model is EGM2008 (was EGM96) and the gravity degree/order cap is 70 (was 40; the compiled-in models were already stored to degree 70, so nothing new is downloaded). Propagations that relied on the default model shift by a few metres per day at LEO and centimetres at GPS; pass `gravity_model=gravmodel.egm96` to reproduce old results. Degree 41–70 is worth tens of metres per day at 400 km and a few metres at 800 km; each acceleration costs about 3× the degree-40 one ([#201](https://github.com/ssmichael1/satkit/pull/201))
- **Breaking:** ITU_GRACE16 is no longer compiled in — its CC BY 4.0 licence attached to the library and every package built from it — but stays available as `gravmodel.itugrace16`: the 1.8 MB file is downloaded (SHA-256 verified) on first use, so selecting it offline with no copy on disk is now a `RuntimeError` / `orbitprop::Error::Gravity` at `propagate` entry rather than working from the embedded copy; `THIRDPARTY-DATA.md` lists only public-domain and IERS data ([#196](https://github.com/ssmichael1/satkit/pull/196), [#183](https://github.com/ssmichael1/satkit/issues/183))
- Solid Earth tides are tide-system aware: for a zero-tide gravity model (`jgm3`, `itugrace16`) the propagator removes the permanent tide (IERS 2010 Eq. 6.13, A₀H₀k₂₀ = −4.201e-9 in C̄20) from the Step 1 correction instead of counting it twice — J2-only EGM96 vs JGM3 over a day at 500 km with tides on goes from 8.5 m to 5 cm; results with `tidemodel.none`, and with the tide-free `egm96` / `egm2008` / `jgm2`, are unchanged ([#196](https://github.com/ssmichael1/satkit/pull/196), [#195](https://github.com/ssmichael1/satkit/issues/195))
- The ICGEM `.gfc` parser accepts Fortran `D` exponents (EGM2008's `1.0d0` row, the GGM05 headers) and Latin-1 headers, reads ICGEM 2.0 `gfct` rows as the static field at the reference epoch and skips the `trnd` / `asin` / `acos` / `dot` time-variable rows, which it used to read as coefficients ([#196](https://github.com/ssmichael1/satkit/pull/196), [#195](https://github.com/ssmichael1/satkit/issues/195))
- EOP and space-weather refreshes follow [CelesTrak's usage policy](https://celestrak.org/usage-policy.php) instead of re-downloading the whole 1957-to-present table on every call: `utils::refresh_file` makes no request while the local copy is inside the file's publication cadence (3 h for the space-weather file, 24 h for the EOP file, `finals2000A.all` or `EOP-All.csv`) and a conditional `If-Modified-Since` request outside it, so an unchanged file costs a `304`; `update_datafiles(overwrite=True)` still forces a full fetch. CI stopped refreshing in the test jobs entirely (every EOP test works from the table's own bounds) and the docs/release workflows only refresh a cached copy over a week old, down from ~12 full-file downloads per push; `python/test/download_data.py` now identifies itself instead of sending `python-requests/x.y` ([#199](https://github.com/ssmichael1/satkit/pull/199))
- Earth orientation parameters are now read from the IERS Bulletin A combined file `finals2000A.all` (fetched from the USNO mirror, then the IERS data centre), with CelesTrak's `EOP-All.csv` as the fallback when both mirrors are unreachable and still read when present; when both files are on disk the one whose observed record runs later is used, with the CSV's 1962–1972 rows kept in front of the IERS table. The IERS file carries about a year of predictions instead of six months, so the silent constant extrapolation past the table end starts later. The source order is the manifest's new `eop` section; `earth_orientation_params::source()` / `satkit.frametransform.eop_source()` report which file is loaded, and `earth_orientation_params::refresh_into` / `load_from_dir` expose the refresh and load steps. A data directory that only holds `EOP-All.csv` keeps working ([#198](https://github.com/ssmichael1/satkit/pull/198), [#164](https://github.com/ssmichael1/satkit/issues/164))

### Deprecated

- Python: the `as_X` conversion methods on `time` (`as_date`, `as_gregorian`, `as_datetime`, `as_mjd`, `as_jd`, `as_unixtime`, `as_iso8601`, `as_rfc3339`) and `quaternion` (`as_rotation_matrix`, `as_euler`) are renamed `to_X` to pair with the `from_X` constructors; the old names still work but emit a `DeprecationWarning` and are removed in 0.25 ([#200](https://github.com/ssmichael1/satkit/pull/200))

### Fixed

- The IAU 2000A celestial-pole offsets dX/dY from `EOP-All.csv` were read as milliarcseconds but the file holds arcseconds, so the correction applied in the CIRS→GCRS rotation was 1000× too small (about 0.2 mas, i.e. ~1 cm at LEO and ~5 cm at GEO); `earth_orientation_params(t)[4:6]` now returns milliarcseconds as documented. LOD is documented as seconds per day, which is what the files hold ([#198](https://github.com/ssmichael1/satkit/pull/198))

### Docs

- Force-model guide: JGM2 is tide-free, not zero-tide (its C20 is EGM96's to 1e-10); the gravity-model table now lists each model's tide system and how it is provided, and the tide-system note describes the automatic handling ([#196](https://github.com/ssmichael1/satkit/pull/196))

## 0.22.1 - 2026-09-20

### Changed

- **Breaking (Python):** `quaternion.conj` and `quaternion.conjugate` are methods (`q.conj()`), matching numpy's `a.conj()`, scipy's `Rotation.inv()` and satkit's own `inverse()`; `quaternion.norm`, `tlefitstatus.converged`, `time.day_of_year` and `time.weekday` are properties (`q.norm`, `status.converged`, `t.day_of_year`, `t.weekday`), like `quaternion.angle` and `propresult.can_interp`. The rule ("a property tells you something about the object; a method gives you something to use instead of it") is in CONTRIBUTING.md and enforced by a stub-parsing test ([#187](https://github.com/ssmichael1/satkit/pull/187))
- The one-time "using the compiled-in copy of tab5.2a.txt" note is gone: the embedded IERS tables and gravity files are byte-identical to the downloadable ones and `update_datafiles()` deliberately does not install them, so the note fired on every fresh install (and as a red stderr block in every satkit.dev tutorial) while describing nothing to act on. `SATKIT_QUIET=1` still silences the warning for a corrupt table that is replaced by the compiled-in copy, and the docs build sets it ([#194](https://github.com/ssmichael1/satkit/pull/194))
- `Precomputed` stores its interpolation table behind an `Arc`, so cloning `PropSettings` (which the Python `propagate` does on every call) shares the table instead of copying it — with a one-year table that was a 55 MB allocation and ~2 ms per call, multiplied by the thread count since the GIL is released ([#193](https://github.com/ssmichael1/satkit/pull/193), [#190](https://github.com/ssmichael1/satkit/issues/190))

### Fixed

- `TLE::fit_from_states` could return a negative eccentricity and stall far from the optimum on near-circular orbits (the doc example fitted a 400 km circular arc at 7.5 km RMS with e = -1.2e-4): the Levenberg-Marquardt damping is now Marquardt-scaled and solved in column-scaled form, a trial step across e = 0 is mapped onto the equivalent orbit with e ≥ 0, and the result is validated (`Error::FitElementOutOfRange`); `TLE::to_2line` now rejects an eccentricity outside [0, 1) (`Error::EccentricityOutOfRange`) instead of silently writing |e| ([#192](https://github.com/ssmichael1/satkit/pull/192))
- Lockfile refreshed for the release (13 minor crate updates, no new dependencies) ([#197](https://github.com/ssmichael1/satkit/pull/197))
- Lockfile: rustls 0.23.44 → 0.23.45 for [RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285) (TLS 1.3 handshake messages accepted across encryption-level boundaries, medium); reached through `ureq`, so it affects the data downloader in the published wheels ([#188](https://github.com/ssmichael1/satkit/pull/188))

### Docs

- Notebook stderr output (Python warnings, the stale-EOP notice) on satkit.dev renders as a neutral code block with an amber edge instead of JupyterLab's red error background, in both light and dark themes ([#191](https://github.com/ssmichael1/satkit/pull/191))
- README: conda-forge version and download badges ([#185](https://github.com/ssmichael1/satkit/pull/185))
- satkit is on conda-forge (`conda install -c conda-forge satkit`, built by [conda-forge/satkit-feedstock](https://github.com/conda-forge/satkit-feedstock)); the installation docs say so and the in-repo recipe copy is removed, since the feedstock is now the source of truth and version bumps arrive there as bot PRs ([#184](https://github.com/ssmichael1/satkit/pull/184))
- `THIRDPARTY-DATA.md` states the source, citation, licence and truncation of every dataset compiled into the library (ITU_GRACE16 is CC BY 4.0; EGM96, JGM-2/3 and the IERS tables are public-domain / freely redistributable); it ships in the sdist and wheels as a licence file and is linked from the README ([#182](https://github.com/ssmichael1/satkit/pull/182))

## 0.22.0 - 2026-09-12

### Added

- Kepler: checked constructor `Kepler::try_new` / `Kepler::validate` (Python: the constructor and the `a`/`eccen`/`inclination`/`mu` setters raise `ValueError` for a non-finite value, `a <= 0`, `eccen` outside [0, 1), `incl` outside [0, π] or `mu <= 0` instead of producing NaN); per-instance gravitational parameter `mu` (`Kepler::with_mu`, `from_pv_with_mu`; Python `kepler(..., mu=)`, `kepler.mu`, `from_pv(..., mu=)`) so lunar and heliocentric elements have a correct period, `propagate` and `to_pv`; derived quantities `periapsis`, `apoapsis`, `specific_energy`, `angular_momentum`, `flight_path_angle`, `argument_of_latitude`, `true_longitude`; `SatState::from_kepler` / `satstate.from_kepler(time, kepler)`; a one-line `repr` for Python `kepler`; `Kepler` derives `PartialEq` and serde (`mu` defaults to Earth's when absent), `Anomaly` derives `Copy`/`PartialEq` ([#168](https://github.com/ssmichael1/satkit/pull/168))
- `propsettings.initial_step_secs` and `propresult.next_step_secs` (Rust: `PropSettings::initial_step_secs`,
  `PropagationResult::next_step_secs`). The adaptive integrators no longer use numeris' starting-step heuristic,
  which is scale sensitive and started `rkv98` at a fraction of a millisecond for an orbit in metres and seconds —
  about half the force evaluations of a one-hour arc at 1e-9 tolerance were spent growing that first step
  ([#175](https://github.com/ssmichael1/satkit/pull/175) discussion). The default first step is now derived from the
  initial state, the tolerances and the integrator order (`1.5·|r|/|v|·tol^(1/(p+1))`, within ~2.5× of the settled
  stride across the RK integrators from 1e-6 to 1e-12); `initial_step_secs` overrides it, and `next_step_secs` reports the
  integrator's working stride at the end of an arc so a follow-on arc can warm-start at full stride
  (`ps.initial_step_secs = res.next_step_secs`). Requires numeris 0.6 ([#178](https://github.com/ssmichael1/satkit/pull/178))
- `integrator.rkv98` with `enable_interp=False` now runs the 16-stage `rkv98_nointerp` tableau automatically:
  the five extra stages of the 21-stage tableau exist only to build the interpolant, so this is the same order
  and error control at 24% fewer force evaluations per step. Results change at the tolerance level for that
  combination (a different tableau takes different steps) ([#178](https://github.com/ssmichael1/satkit/pull/178))

### Distribution

- **Breaking (Python packaging):** macOS wheels are arm64 (Apple silicon) only; the `x86_64-apple-darwin` wheels are no longer built, matching SciPy/Polars and the arm64-only GitHub runners, and the macOS deployment target moves from 10.12 to 11.0. Intel-Mac users install from source (`pip install --no-binary satkit satkit`, stable Rust toolchain required) or from conda-forge, which builds `osx-64` ([#172](https://github.com/ssmichael1/satkit/pull/172))

### Fixed

- Space-weather records are indexed by UTC calendar day. The lookup counted continuous days from the first record, and since `Instant` counts leap seconds the last 37 seconds of each UTC day read the *next* day's record (F10.7, Ap, and the 3-hourly Ap history fed to NRLMSISE-00). Density, and hence drag, in that window changes very slightly (contributed by @scottshambaugh, [#176](https://github.com/ssmichael1/satkit/pull/176))
- `sgp4()` rejects SGP4-XP element sets (TLE ephemeris type 4, OMM `EPHEMERIS_TYPE` 4) with a clear error instead of propagating them: an SGP4-XP line 1 stores agom and a B term in the columns a classic TLE uses for nddot and B*, so the old behaviour ran classic SGP4 on the wrong inputs and returned a plausible but wrong state ([#174](https://github.com/ssmichael1/satkit/pull/174))
- `satkit.density.nrlmsise(altitude_m, latitude_rad, longitude_rad, time)` converts its radian latitude/longitude to the degrees the model takes; the values were passed through unchanged, so a caller following the stub at 60° N was evaluated at 1.05° N (the `itrfcoord` overload and `nrlmsise00(latitude_deg=...)` were already correct) ([#171](https://github.com/ssmichael1/satkit/pull/171))
- A corrupt or truncated `tab5.2*.txt` in a data directory no longer panics the first frame transform: satkit warns and uses the compiled-in copy of the same IERS table (exact, not an approximation), and the parser now rejects text with no table header or fewer rows than declared — an HTML notice page saved under the table's name previously loaded as six empty series and silently dropped the nutation terms ([#166](https://github.com/ssmichael1/satkit/pull/166))

### Changed

- **Breaking (Rust):** `Kepler.w` is renamed `Kepler.argp`, the struct gains a `mu` field (struct-literal construction must supply it — prefer `Kepler::new(...).with_mu(...)`), and `kepler::Error` is `#[non_exhaustive]` (new `InvalidElement` variant). Python: `argp` is the constructor parameter and property name; `w` still works as a constructor keyword and as a property (kept indefinitely) but emits `DeprecationWarning`. **Breaking (Python):** `kepler.from_pv` raises `ValueError` instead of `RuntimeError` for a hyperbolic/parabolic or rectilinear state, and every element setter (`a`, `eccen`, `inclination`, `raan`, `argp`, `nu`, `mu`) raises `ValueError` for an out-of-domain or non-finite value instead of accepting it ([#168](https://github.com/ssmichael1/satkit/pull/168))
- OMM interface overhaul: `omm_from_url` returns every field the source provided (metadata, `COMMENT`, Space-Track extras such as `OBJECT_TYPE`/`RCS_SIZE`/`TLE_LINE1`, XML `USER_DEFINED` parameters) instead of 17 keys; new `omm_from_file` / `omm_from_text` loaders (JSON or XML, detected from content) replace the `xmltodict` workaround; `TLE.from_omm` / `TLE.to_omm` convert between the representations; `sgp4` dict inputs go through the same serde parser as Rust (quoted numbers, `null`/empty optionals, case-insensitive metadata, any level of an xmltodict tree, `EPOCH` as `time`/`datetime`); propagation rejects `EPHEMERIS_TYPE` 4 (SGP4-XP) instead of running classic SGP4 on it; stubs gain the `OMMDict` type. Rust: `OMM.epoch` is an `Instant` (was a string; `epoch_instant()` deprecated), `OMM` derives `Serialize`, gains `from_mean_elements`, `from_tle`/`to_tle`, `from_json_value`, `from_text`/`from_file`, `reset_cache`; `from_json_string` accepts a bare object; `omm::Error` is `#[non_exhaustive]` with a single `InvalidField` variant; `Default` is removed ([#173](https://github.com/ssmichael1/satkit/pull/173))
- Propagation step time is ~3x faster (LEO, 40x40 field, drag, rkv98): the spherical-harmonic gravity kernels dispatch at runtime to a copy compiled with FMA on x86-64 (their `mul_add` calls were libm function calls on the baseline x86-64 that wheels target; ~5x on the 40x40 field, no change on arm64 where FMA is native), NRLMSISE-00 computes its latitude, longitude and local-time terms once per evaluation instead of in each of its 14 inner calls, the ITRF-to-geodetic conversion uses a single Bowring refinement (double precision from 1000 km below the surface to beyond lunar distance), gravity-model parsing skips coefficients above the requested degree, and `satkit.__version__` is read from the extension module instead of an `importlib.metadata` lookup at import time (contributed by @scottshambaugh, [#175](https://github.com/ssmichael1/satkit/pull/175))
- `update_datafiles()` no longer downloads files that are compiled into the library: the IERS tables and gravity models are `default: false` in the manifest (still pinned and fetchable by name), and the unused `leap-seconds.list` (nothing ever read it — the runtime leap-second table is a compiled-in constant) is removed from the manifest entirely. The only static download left is the JPL ephemeris, alongside the daily EOP / space-weather / solar-cycle refreshes ([#163](https://github.com/ssmichael1/satkit/pull/163))

### Docs

- Every Python docstring (`.pyi` stubs and runtime `__doc__`) states units of measure for dimensioned arguments, returns and attributes (SI: meters, m/s, m/s², radians unless the name ends in `_deg`); `sgp4` outputs explicitly noted as meters / m/s in the TEME frame, not the km / km/s of other SGP4 libraries ([#169](https://github.com/ssmichael1/satkit/pull/169))

## 0.21.2 - 2026-08-30

### Changed

- **Breaking (Rust):** `utils::download::Error` and its field-carrying variants are `#[non_exhaustive]`: downstream matches need a wildcard arm and struct patterns a `..`, so future variants and fields (this release alone added `Request`, `ContentRejected`, and `AllSourcesFailed.hint`) land without breaking compiles ([#161](https://github.com/ssmichael1/satkit/pull/161))

### Fixed

- Downloads verify servers against the operating system's trust store (macOS keychain, Windows certificate store, `/etc/ssl` on Unix) instead of the Mozilla root list compiled into the HTTP client: on a network whose TLS is inspected by a corporate proxy every download failed with `io: invalid peer certificate: UnknownIssuer`, because such a proxy re-signs traffic with a private CA that can only ever live in the system store. `SATKIT_CA_BUNDLE` overrides the choice with a PEM file, `platform`, or `webpki` (the compiled-in roots, for a container with no system store); `SSL_CERT_FILE` is deliberately not consulted. Download errors also name the URL that failed and, for a certificate failure, say what to do about it once (`download::Error` gains a `Request` variant, and `AllSourcesFailed` a `hint` field) ([#160](https://github.com/ssmichael1/satkit/pull/160))

- The daily CelesTrak files are checked before they replace the copy on disk: `EOP-All.csv` and `SW-All.csv` are run through the parser that will later read them, and any unverified download is rejected if it opens with an HTML document. A filtering proxy or captive portal that answers `200 OK` with a notice page can no longer overwrite a good EOP table with a web page — the partial download is discarded and the existing file left in place (`download::Error::ContentRejected`) ([#160](https://github.com/ssmichael1/satkit/pull/160))

### Docs

- GMAT validation page: removed the "What the corpus found" note ([#159](https://github.com/ssmichael1/satkit/pull/159))
- GMAT validation page and README describe the drag corpus: drag orbits, constant/file-driven force models, measured agreement against the drag-only displacement, anomalous-oxygen and F10.7-timing floors ([#157](https://github.com/ssmichael1/satkit/pull/157))

### CI

- Build workflow runs once per change: pull requests build on the PR event only (branch pushes without a PR no longer build), `main` builds on the merge commit, the redundant run on PR close is gone, and a new push to a PR cancels its superseded run ([#158](https://github.com/ssmichael1/satkit/pull/158))
