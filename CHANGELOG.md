# Changelog


## Unreleased

### High-precision propagator: force model unified across all integrators

- **A single `force_model()` replaces four duplicated force closures.** The RKV `ydot`, the RODAS4 `ydot_vec` and `jac_fn`, and the Gauss-Jackson 8 `accel_fn` each carried a near-identical copy of the acceleration physics (Earth gravity, Sun/Moon third-body, solid Earth tides, GR Schwarzschild, SRP, drag, thrust) — ~200 lines duplicated four ways. Adding a force term (as 0.18.0's tides and GR did) meant editing every copy in lockstep and risked them drifting apart. The physics now lives in one `force_model()` function in `orbitprop::propagator`; each integrator closure is a thin adapter that unpacks its own state layout (`Matrix<6, C>` / `Vector<f64, 6>` / separate `(r, v)`), calls `force_model()`, and packs the result. **No behavior change** — end states are bit-identical and the full Rust suite (including the ESA SP3 GPS regression) passes unchanged.
- **`ForceEval` mode selector** (`Accel` / `AccelAndPartials` / `PartialsOnly`) keeps each integrator's cost identical to the prior hand-specialized code. Every call site passes a compile-time-constant mode and `force_model` is `#[inline]`, so the compiler constant-folds the partials/accel branches away and regenerates the original specialization — no extra call, no runtime dispatch on the enum. The RODAS4 Jacobian path (`PartialsOnly`) still skips the tide/GR/SRP/thrust terms — they contribute no partials — so it does not regress.
- **Drag altitude gate uses `norm_squared()`** against a precomputed squared limit (`DRAG_RADIUS_LIMIT_M`) instead of `norm()`, removing a square root from every force evaluation regardless of whether drag is active.

### Earth gravity: stored coefficient table capped at the evaluation degree

- **`Gravity::parse` caps the coefficient table at degree 44** (`MAX_COEFF_DIM`) rather than storing the file's full resolution. The evaluator dispatches at degree ≤ 40 and the Cunningham recursion / divisor tables are 44×44, so higher-degree coefficients were never used — yet the default EGM96 model (file degree 360) was holding a 361×361 `DMatrix` (~1 MB). Capping it drops that to ~15 KB and, more importantly, shrinks the column-major stride from 361 to 44 so the strided S-coefficient reads `coeffs[(m-1, n)]` stay resident in L1 instead of scattering across ~3 KB jumps. **Results are bit-identical** (coefficients for n,m ≤ 43 are untouched, and computation never exceeds degree 40); a microbenchmark on EGM96 shows ~5–15% faster `accel` / `accel_and_partials` and, notably, eliminates the cache-miss timing variance in the hot loop. The cap is coupled to the pre-existing degree-40 dispatch and 44×44 divisor-table limits, and is documented as such so a future degree bump touches all three together.


## 0.18.0 - 2026-05-25

### Solid Earth tides (IERS 2010 §6.2 Step 1) in the high-precision propagator

- **New `orbitprop::tides` module** implementing the IERS Conventions 2010 §6.2.1 Step 1 frequency-independent solid Earth tide model — Love-number response of the Earth's gravity field to lunar and solar attraction. Exposes `TideModel` (`None` / `SolidStep1` / `SolidFull`), `TideDeltas` (ΔC̄ₙₘ, ΔS̄ₙₘ for n=2,3,4), `solid_tide_deltas()` and `tide_accel()`. Resolves [#16](https://github.com/ssmichael1/satkit/issues/16).
- **Enabled by default.** `PropSettings::default()` now ships `tide_model: TideModel::SolidStep1`. Consistent with the project's existing "high-fidelity by default" stance (sun gravity, moon gravity, space weather are all on by default). The ~5% per-ydot overhead is dominated by Sun/Moon ITRF rotations rather than the tide math itself. **Behavior change for callers holding state-vector regression tests**: end-state positions will shift by ~tens of cm to several meters depending on regime and arc length. Pass `tide_model=TideModel::None` to reproduce pre-0.18 numerics.
- **Step 2 (frequency-dependent corrections from Tables 6.5a/b/c) deferred.** Selecting `TideModel::SolidFull` currently falls through to Step 1 behavior; the 71-constituent frequency-dependent correction is the next planned increment. ~99% of the solid-tide signal lives in Step 1, so the deferral has minimal accuracy cost for most use cases.
- **Wired into every integrator force closure** — RKV (`ydot`), RODAS4 (`ydot_vec`), and Gauss-Jackson 8 (`accel_fn`). The tide acceleration is computed in ITRF (where IERS §6.2 formulas are written) and rotated back to GCRF via the existing precomputed `qgcrf2itrf` quaternion. Tide partials are skipped from the STM update — ∂a_tide/∂r is ≲1e-12 of the J2 partial and well below filter-noise thresholds.
- **Validated against ESA's `ESA0OPSFIN` SP3 GPS truth file.** Max per-axis residual over a 1-day GPS-20 arc drops from 6.42 m (no tides) to 5.71 m (Step 1 tides) at degree-4 gravity — an 11% improvement. The `test_gps` regression threshold tightened from 8 m to 6.5 m to reflect the gain, and a new strict assertion enforces *with-tides residual < without-tides residual* so a future regression that breaks the wiring fails loudly.
- **Python bindings.** New `satkit.tidemodel` enum (`none` / `solid_step1` / `solid_full`) and `satkit.propsettings.tide_model` field, plumbed through both the kwarg constructor and getter/setter. `satkit.pyi` updated with full docstrings and the new constructor argument. Four new tests in `TestSolidTides` cover enum members, default value, kwarg/setter round-trip, and the with-vs-without functional difference.

### General-relativistic Schwarzschild correction in the high-precision propagator

- **New `orbitprop::relativity` module** implementing the IERS Conventions 2010 §10.3 Eq. 10.12 Schwarzschild (post-Newtonian, β = γ = 1) acceleration. Single `gr_schwarzschild_accel(pos_gcrf, vel_gcrf, mu_e) -> Vector3` function — ~10 flops, trivial computational cost. Resolves [#109](https://github.com/ssmichael1/satkit/issues/109).
- **Enabled by default.** `PropSettings::default()` now ships `use_relativistic_correction: true`. Schwarzschild is universal physics, not a model choice — the flag exists for reproducibility against pre-0.18 numerics, not because the term is optional in practice.
- **~1 m/day position drift at GPS altitude** if omitted; ~3 m/day at GEO. The empirical NTW acceleration fit in the high-precision propagation tutorial (`docs/tutorials/High Precision Propagation.ipynb`) was previously absorbing this signal — explicit GR modeling shifts the empirical from "catch-all for ~0.17 m of un-modeled physics" to "smaller catch-all for the rest (higher-order SRP, ocean tides, …)."
- **Lense-Thirring and de Sitter currently omitted** (sub-cm-class effects at LEO/MEO). Reserved for future expansion behind the same `use_relativistic_correction` flag.
- **Wired into every integrator force closure** (RKV, RODAS4, GJ8). GR partials are ~1e-15/m vs J2's ~1e-7/m and are skipped in the STM update — same precision tradeoff as the tide block.
- **GPS regression test threshold tightened to 2.5 m** (was 6.5 m with tides only, 8 m before tides). Hardcoded `v0` + `Cr*A/m` refitted against ESA SP3 truth using the updated force model. The previous "with-tides residual < without-tides residual" strict assertion has been dropped from this test — once the IC is fit to the full force model, the initial state absorbs enough of the constant-shift part of the tide signal that toggling tides off can produce a *smaller* residual on this particular arc. The independent `test_solid_tides_perturb_orbit` test still guards that tides change the propagation.
- **Python bindings.** New `propsettings.use_relativistic_correction: bool` field exposed via kwarg constructor and getter/setter. `.pyi` stub updated with docstrings. Three new tests in `TestRelativisticCorrection` cover the default, setter/kwarg round-trip, and a with-vs-without propagation diff at GPS altitude.

### Bytes/path init for every data-file subsystem (embedded / non-filesystem support)

- **New `init_from_bytes(&[u8])` and `init_from_path(&Path)` on each of the six data-file subsystems** — `jplephem`, `earthgravity`, `frametransform::ierstable`, `spaceweather`, `solar_cycle_forecast`, `earth_orientation_params`. Lets callers populate satkit's runtime state from a database row, an `importlib.resources` buffer, a memory-mapped file, or any other non-filesystem source, instead of forcing every data file through `datadir()`. Motivated by two concrete use cases: (a) embedded contexts where the ephemeris and EOP live in SQLite/blob storage rather than on disk, and (b) conda-forge packaging where the data file is bundled as a Python package resource and accessed via `pkgutil` / `importlib.resources` returning bytes. The bytes APIs take borrowed `&[u8]` so callers don't pay a double allocation.
- **Two semantics, picked per subsystem.** *Static* subsystems (`jplephem`, `earthgravity`, `frametransform::ierstable`) wrap a `OnceLock` and return `Err(AlreadyInitialized)` if init is called after the lazy default load has already won — the data is mathematical constants and replacing it mid-process makes no sense. *Refreshable* subsystems (`spaceweather`, `solar_cycle_forecast`, `earth_orientation_params`) wrap a `RwLock<Option<T>>` and always succeed, replacing any previously loaded data — CelesTrak / NOAA / IERS publish daily-to-monthly updates and refresh-in-place is the intended model.
- **`earth_orientation_params` drops `AtomicPtr` + intentional-leak for safe `RwLock<Option<...>>`.** The old EOP loader stored a `Vec<EOPEntry>` behind an `AtomicPtr` and leaked the previous allocation on every `update()` since other threads might still hold the lock-free reference. The new pattern matches `spaceweather` and `solar_cycle_forecast` — one `RwLock`, one `std::sync::Once` for the silent default-load attempt, no `unsafe` in the read path. EOP reads are cheap enough that the lock cost is unmeasurable next to the binary-search-plus-interpolation work; benchmarks would not detect the change.
- **`spaceweather` consolidates `OnceLock<RwLock<Result<_>>>` to a const-init `RwLock<Option<_>>`.** Same destination as the EOP cleanup — one mechanism across all three refreshable subsystems instead of two-and-a-half. The `Result` that used to live inside the lock was capturing initial-load failures for later readers to encounter; with the new init API, load failures bubble up to whoever called `init_from_*`/`update`, and `get()` simply returns `NoRecordForDate` if the singleton stays empty.
- **Parsers factored out for testability.** Each subsystem's loader is now split into a pure parser (`parse_csv(&str)` / `JPLEphem::parse(&[u8])` / `Gravity::parse(&str)` / etc.) plus thin `from_file` / `from_path` / `from_bytes` wrappers around it. The byte-buffer entry point is the natural primitive; everything else is composition on top.
- **Rust-only.** No Python bindings — embedded users live in Rust and notebook/script users have `satkit.utils.update_datafiles()` already.
- **Integration tests in `tests/`.** Each subsystem gets a separate-binary integration test that verifies bytes-init populates the singleton, the relevant query function reads from the just-installed data, and the static-vs-refreshable second-init semantics work as specified. Six new test files; all run cleanly even when the underlying data files are absent (they skip with a notice rather than failing).

### JPL ephemeris file selection is now configurable; DE405–DE441 all supported

- **`SATKIT_JPLEPHEM_FILE` env var.** Either an absolute path, or a basename to resolve under `datadir()`. Lets a user point satkit at a specific ephemeris file without dropping it at the legacy filename.
- **Autodetect from `datadir()`.** When the env var isn't set, scan the data directory for any file matching `linux_p<start>p<stop>.4XX` *or* `lnxp<start>p<stop>.4XX` (DE421 and earlier used the older `lnxp` prefix; DE430 and later use `linux_p`). The highest DE-version suffix wins.
- **Legacy fallback unchanged.** With neither the env var nor any matching file present, the singleton falls back to `linux_p1550p2650.440` under `datadir()` and triggers the existing GCS auto-download. Existing installs are not affected.
- **Auto-download is now scoped to the legacy default name only.** Previously, any filename passed through `from_file` would trigger a GCS request if the file was missing — guaranteed to 404 for anything other than the DE440 full-span binary. The new behaviour: download only for the legacy name; for any other resolved path, the file must already exist (or be supplied via `init_from_*`).
- **Parser is fully header-driven.** `de_version`, `n_con`, the interpolation-pointer table `ipt[15][3]`, the kernel size, and the JD span are all read from the file header rather than hard-coded. DE405 / DE421 / DE430 / DE440 / DE441 all parse through the same code path. The one special branch in the parser (`if de_version > 430 && n_con != 400`) handles the extended constants block introduced in DE440; DE421-and-earlier files take the older path.
- **DE421 end-to-end verification.** A file-gated integration test loads `lnxp1900p2053.421` via `init_from_path` and walks the official `testpo.421` truth values — 380 position vectors match to <1e-10 relative error, confirming the smaller-`n_con` layout is parsed correctly. The DE421 binary is ~13 MB vs the full DE440 at ~98 MB, making it a viable default for distribution channels (e.g. conda-forge) where the 100 MB blob is a friction point.
- **Out-of-range queries are explicit.** Calling `geocentric_state` / `barycentric_state` outside the loaded file's `[jd_start, jd_stop]` window returns `Error::InvalidJulianDate(jd)` instead of returning silently wrong values. This was already the existing behaviour but is now load-bearing: a DE421 install (range 1899–2053) will start rejecting queries past 2053, and that's the correct failure mode.


## 0.17.0 - 2026-05-22

### Frame-enum dispatch: `rotation` / `rotation_approx` / `transform_state`

- **New `frametransform::rotation(from, to, t)` and `transform_state(from, to, t, pos, vel)`** plus their `_approx` companions. Single-call frame-to-frame dispatch — `rotation(Frame::ITRF, Frame::GCRF, &t)` replaces having to know which of `qitrf2gcrf` / `qteme2itrf` / `qcirs2gcrs` / … is the right function for a given pair. Matches the convention every modern peer library (SPICE `pxform`, Orekit `Frame.getTransformTo`, Astropy `SkyCoord.transform_to`, ANISE `Almanac::transform`) has used for years; satkit's previous "named function per directional pair" pattern was the outlier.
- **Shortest-path dispatch, not always-through-GCRF.** The hand-coded match arms take the natural shortest path through the frame graph for each pair: ITRF↔TIRS pays only polar motion, ITRF↔CIRS skips the expensive precession/nutation step, and ITRF↔GCRF uses the existing amortised direct function (one EOP lookup feeds both polar motion and the dX/dY nutation correction). No transform pays full IERS 2010 reduction unnecessarily.
- **EME2000 and ICRF are now real implementations** rather than stub variants. EME2000↔GCRF uses the J2000 frame-bias matrix (IERS 2010 §5.4.4 — three small constant angles totalling ~17 milliarcsec). The matrix is pinned in tests against an independent numpy computation of the IERS reference: bit-perfect match (max element diff ~1e-16 rad, pure floating-point rounding). ICRF↔GCRF is treated as identity (the actual offset is < 0.1 arcsec and below satkit's other modelling errors). Previously both variants existed in the `Frame` enum but every call site that touched them panicked or returned `UnsupportedFrame`.
- **`rotation_approx` is honest about its narrower domain.** TIRS and CIRS are defined by the IERS 2010 reduction and have no FK5 analogue — `rotation_approx` returns `Error::ApproxNotSupportedForFrame` rather than silently falling back. Orbit-dependent frames (LVLH, RTN, NTW) return `Error::OrbitFrameRequiresState` with a pointer at the existing `to_gcrf` / `from_gcrf` helpers.
- **`transform_state` covers all non-orbit frame pairs** — identity, ITRF↔inertial, TIRS/CIRS↔inertial, within-inertial, and ITRF↔TIRS — with proper sweep-term bookkeeping. The `ω⊕ × r` sweep is evaluated in TIRS where ω⊕ is exactly along +ẑ, then composed onto the requested endpoints. Polar motion between ITRF and TIRS is treated as a static rotation: its rate is ~1.7e-9 rad/s, contributing sub-mm/s at LEO, and is neglected to match the existing `itrf_to_gcrf_state` convention. Orbit-dependent frames remain `to_gcrf` / `from_gcrf` territory.
- **Python bindings + type stubs.** `satkit.frametransform.rotation`, `rotation_approx`, `transform_state`, `transform_state_approx`. Both `rotation` and `rotation_approx` accept either a scalar time or an array of times — returning a single quaternion or a list of quaternions respectively, matching the per-pair functions' batch shape. The original per-pair functions remain; this layer is purely additive.
- **Docs.** All major tutorials updated to use the dispatch API in examples. The *first* call in each introductory page uses keyword arguments (`rotation(from_frame=ITRF, to_frame=GCRF, tm=t)`) so the source / destination direction is unambiguous at first sight; subsequent calls in the same tutorial fall back to positional form for brevity. A new "Dispatch API" section in `docs/api/frametransform.md` introduces all four entry points.
- **No `_ =>` catch-alls on `Frame`.** Both the dispatch match arms and the internal `is_earth_rotating` / `is_orbit_dependent` classifiers enumerate every variant explicitly so the compiler will flag any future `Frame` additions.
- **Direction pins.** Each TEME-involving dispatch arm is compared against the canonical `qteme2gcrf` / `qitrf2tirs` / `qteme2itrf` composition so a future change that flips a sign fails immediately — the roundtrip test alone can't catch direction errors because `rotation(b, a)` is always the conjugate of `rotation(a, b)` regardless of which is "right".

### `ITRFCoord::to_enu` / `to_ned`: parameter renamed `ref_coord` → `origin`, docstrings overhauled

- **Parameter rename.** `ITRFCoord::to_enu(&self, ref_coord)` and `to_ned(&self, ref_coord)` now take `origin` instead. The ENU/NED triad is *anchored* at this argument — calling it "origin" matches the standard local-tangent-frame terminology and makes call sites read like prose: `satellite.to_enu(&station)` ("ENU of the satellite, with the station as the origin"). Rust callers are unaffected (positional args); Python callers using `origin=` as a kwarg will need to update from `refcoord=` / `other=`. Resolves [#91](https://github.com/ssmichael1/satkit/issues/91).
- **Docstrings rewritten.** Lead with a one-line "FROM `origin` TO `self`" statement and an explicit sign convention for the Up/Down component. Example renamed from `itrf1`/`itrf2` to `station`/`satellite` so the canonical sat/ground-station case (the source of the issue) is the worked example. Mirrored across the Rust source, the PyO3 binding, and the `.pyi` stub.
- **Latent Python binding inconsistency fixed.** `to_ned`'s parameter was `other` in code but documented as `refcoord` — both now consistently `origin`.

### `satkit::Error` façade is deprecated

- **`satkit::Error` and `satkit::Result` marked `#[deprecated(since = "0.17.0")]`.** The top-level error enum was added in 0.16.x (PR #86) as a convenience for downstream apps that wanted a single result type to unify the per-module typed errors. In practice the module-scoped errors (`tle::Error`, `orbitprop::Error`, …) plus a downstream-defined `enum AppError` or `anyhow::Result` cover the use case more cleanly and don't lock satkit into a public surface that has to grow in lockstep with every new module-level error variant. Both `Error` and `Result` are still exported and functional; the deprecation just nudges callers toward the more durable pattern. Removal is planned for a future release.

### `update_datafiles` uses conditional GETs to skip unchanged files

- **`If-Modified-Since` on every refresh download.** `download_file` now formats the local file's mtime as an HTTP-date and sends it as `If-Modified-Since`. On a `304 Not Modified` response, the existing file is left untouched and the function returns `Ok(false)`. The two regularly-refreshed files (`EOP-All.csv` and `SW-All.csv`, both served by celestrak.org) had been re-downloaded on every `update_datafiles()` call regardless of whether they had changed — now they only transfer when the server reports a newer `Last-Modified`. Bandwidth-constrained users (e.g. on cellular) see ~3 MB/run drop to a pair of HEAD-sized 304s. Resolves [#97](https://github.com/ssmichael1/satkit/issues/97).
- **New `%a` format code in `Instant::strftime`** (`src/time/instantparse.rs`). Abbreviated weekday name — `Sun`, `Mon`, ..., `Sat` — parallel to the existing full-name `%A`. Added to support the RFC 7231 IMF-fixdate format (`%a, %d %b %Y %H:%M:%S GMT`) used by `If-Modified-Since`. No new dependencies.
- **Behavior of `overwrite_if_exists=false` is unchanged**: the local-existence fast path still short-circuits before any network call. The flag now effectively means "ask the server whether to re-fetch" when true, rather than "always re-fetch".

### Python type stubs: `TLE.from_lines` accepts any `Sequence[str]`

- **`TLE.from_lines` stub widened from `list[str]` to `Sequence[str]`** (`python/satkit/satkit.pyi`). The Rust binding already accepted any Python sequence at runtime — only the stub was narrow, so callers passing a `tuple[str, str]` (the natural shape for a 2-line TLE) got spurious type-checker complaints. Resolves [#93](https://github.com/ssmichael1/satkit/issues/93).
- **`@overload` for fixed-length tuples.** `tuple[str, str]` and `tuple[str, str, str]` (the 2-line and name+2-line forms) are now statically typed as returning a single `TLE`; other sequences still return `TLE | list[TLE]`. Callers writing `tle = TLE.from_lines((line1, line2))` no longer need to narrow the result.

### `ureq` is now an optional dep behind the `download` feature

- **New `download` Cargo feature** (default-on) gates `ureq`. `TLE::from_url`, `OMM::from_url`, `solar_cycle_forecast::update`, and the `satkit::utils::{download_file, download_file_async, download_to_string, download_if_not_exist, update_datafiles}` helpers all live behind it. Default builds are unchanged. Users who want a slim dependency tree can opt out via `cargo build --no-default-features --features omm-xml`, which drops ~25 transitive crates (`ureq`, `rustls`, `ring`, `rustls-webpki`, `webpki-roots`, `flate2`, etc.). Without the feature, `download_if_not_exist` will succeed if the file already exists and otherwise return an error; the other download helpers always return an error.
- **Python builds are unaffected.**

### Licensing

- **Dual-licensed under MIT OR Apache-2.0.** Previously MIT-only. Apache-2.0 adds an explicit patent grant — important now that the project is taking external contributions, since it binds contributors to a patent peace clause that bare MIT does not. Matches the Rust ecosystem convention. `LICENSE` was renamed to `LICENSE-MIT` and a new `LICENSE-APACHE` was added; `Cargo.toml` and `pyproject.toml` `license` fields updated to `"MIT OR Apache-2.0"`. Downstream users may continue to use the project under either license at their option — no action required.

### Workspace-wide `cargo fmt`

- **CI now enforces `cargo fmt`.** A new `lint` job runs `cargo fmt --all -- --check` on ubuntu before the build matrix kicks off, so formatting failures surface on a single runner instead of three. Contributors should run `cargo fmt` locally or enable format-on-save in their editor before pushing.
- **Workspace reformatted in one mechanical commit.** No functional changes; 52 files touched (+5741 / −1336). Kicked off by [@parker-research](https://github.com/parker-research) in #89.
- **`.git-blame-ignore-revs` added** to mask the bulk-format commit from `git blame`. GitHub's web blame honors it automatically. Locally, opt in with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.


## 0.16.2 - 2026-04-13

### Frame Transforms

- **Batched `itrf_to_gcrf_state` / `gcrf_to_itrf_state` (Python).** Both functions now accept either a single state (length-3 `pos`/`vel` + scalar `time`) or a batch of `N` states (shape `(N, 3)` arrays + length-`N` time array/list) and return matching-shape output. Previously required a Python loop.
- **New `itrf_to_gcrf_state_approx` / `gcrf_to_itrf_state_approx` (Rust + Python).** Approximate IAU-76/FK5 variants of the full-state transforms (~1 arcsec on position, <1 m/s on velocity vs. full IERS 2010). Substantially cheaper when full precision isn't required; polar motion is neglected so the `ω⊕ × r` sweep is evaluated in ITRF directly. Scalar and batched inputs supported in Python.

### Documentation

- **Nomenclature sweep: "IAU 2006" / "IAU-2000" → "IERS 2010" / "IAU 2000A".** The full reduction chain satkit implements is the IERS 2010 Conventions, which adopt the IAU 2006 precession with the IAU 2000A nutation series — referring to it as "IAU-2006 reduction" or "IAU-2000 nutation" was imprecise. Updated across `README.md`, `docs/index.md`, crate-level `lib.rs` docs, `src/earth_orientation_params.rs` doc comments, and the **Coordinate Frames** and **Plots** tutorials.


## 0.16.1 - 2026-04-05

### Dependency Cleanup

- **`rmpfit` removed.** `TLE::fit_from_states` previously used the `rmpfit` crate (a thin wrapper around the `cmpfit` C library) for Levenberg-Marquardt. Replaced with a small local LM loop built on top of `numeris` fixed-size linear algebra (7×7 normal equations, finite-difference Jacobian, numeris `LuDecomposition` for the damped solve). SGP4 failures on perturbed or trial parameters are now handled as step rejections rather than hard errors, so fits that previously aborted mid-way (e.g. the 400 km LEO + drag test case) now converge.
- **`rmpfit::MPStatus` → `satkit::tle::TleFitResult`.** New public types `TleFitStatus` (enum) and `TleFitResult` (struct with `orig_norm`, `best_norm`, `grad_norm`, `n_iter`, `n_res_evals`) replace the rmpfit-specific return type. The Python `tle.fit_from_states` now returns a dict with keys `status`, `converged`, `orig_norm`, `best_norm`, `grad_norm`, `n_iter`, `n_res_evals`, and the Python `mpsuccess` class is replaced by `tlefitstatus` with variants `GradientConverged`, `StepConverged`, `CostConverged`, `MaxIterations`, `DampingSaturated` plus a `.converged()` helper.
- **`itertools` removed.** The crate was used for a single method (`take_while_ref`) at 5 call sites in `time/instantparse.rs`. Replaced with a local `take_while_peek` helper on `Peekable<Chars>`.
- **`json` crate removed.** The two usages (`utils/update_data.rs`, `solar_cycle_forecast.rs`) migrated to `serde_json`, which was already a dependency. Reduces crate graph by one JSON parser.
- **`serde-pickle` removed from the top-level crate.** It was only actually used in `python/src/pypropresult.rs`; the root `Cargo.toml` declaration was dead. Still a dependency of the `satkit-python` crate.

### Bug Fixes

- **Clippy**: fix `clone_on_copy` on a `Copy` error enum in `pysgp4.rs`; factor a `LambertSolution` type alias in `pylambert.rs` to silence `type_complexity` warnings. Full workspace now builds clean under `cargo clippy --all-targets`.

### Documentation

- **README.md** and **lib.rs** crate-level docs updated for 0.16: canonical `Frame::RTN` (with `RIC`/`RSW` aliases) and new `NTW` / `LVLH` frames in the maneuver list, Gauss-Jackson 8 in the integrator list, updated Python version range (3.10–3.14), corrected docs URL (<https://satkit.dev/>), bumped `numeris` example version to 0.5.7, and refreshed test counts (157 Rust + 81 Python).
- **High Precision Propagation** tutorial: dropped three unused imports (`math`, `numpy.typing`, `scipy.optimize.minimize_scalar`).
- **Two-Line Element Set**, **Orbital Mean-Element Message**, and **Optical Observations of Satellites** tutorials: replaced hand-rolled `requests.get` / `xmltodict` / hardcoded TLE-line blocks with the built-in `sk.TLE.from_url(url)` and `sk.omm_from_url(url)` helpers introduced in 0.15.1. The OMM notebook no longer depends on `requests` or `xmltodict` at all; the Optical Observations notebook now derives all sample times from `tle.epoch` so the fit is reproducible against the current TLE.


## 0.16.0 - 2026-04-05

### Gauss-Jackson 8 Integrator

- **`Integrator::GaussJackson8`**: fixed-step 8th-order multistep predictor-corrector specialised for orbit propagation. Typically 3-10x fewer force evaluations than RKV98 for smooth long-duration runs. Combined Gauss-Jackson + Summed-Adams formulation handles velocity-dependent forces (drag, SRP) natively. Per-step dense output via quintic Hermite interpolation.
- Selected with `Integrator::GaussJackson8` and a user-supplied `gj_step_seconds` (Rust) / `propsettings(integrator=satkit.integrator.gauss_jackson8, gj_step_seconds=...)` (Python). No STM support.
- Lives in a new `satkit::orbitprop::ode` submodule (astrodynamics-specific enough that it doesn't belong in `numeris`).
- **Precompute bounds fix**: `Precomputed::new_padded` now takes an explicit padding; `PropSettings::required_precompute_padding` automatically extends interp-table bounds to cover the GJ8 backward startup stencil (4 × `gj_step_seconds` on each end). Previously silently failed for `gj_step_seconds > 60`.

### Coordinate Frames: RTN canonical, NTW, LVLH maneuvers

- **Breaking: `Frame::RIC` renamed to `Frame::RTN`** as the canonical name (matches the CCSDS OEM convention). `Frame::RIC` and `Frame::RSW` remain as compile-time aliases (`pub const RIC: Self = Self::RTN`), so existing code using either name still compiles and `Frame::RIC == Frame::RTN` is `true`. Python exposes `frame.RIC` and `frame.RSW` as class-level aliases of `frame.RTN`.
- **`Frame::NTW`** (velocity-aligned: T=v̂, W=ĥ, N=T×W). Natural for prograde/retrograde burns on eccentric orbits: a pure +T delta-v of magnitude Δv adds exactly Δv to |v|, while an in-track RTN burn of the same magnitude loses a factor of cos γ where γ is the flight-path angle. Accepted by maneuvers, thrust, uncertainty, and frame transforms.
- **LVLH** is now a supported maneuver/thrust coordinate frame (previously valid only for uncertainty).
- **Unified frame-transform API**: `frametransform::to_gcrf(frame, pos, vel)` and `from_gcrf(frame, pos, vel)` replace the combinatorial explosion of per-frame helpers. `state_to_gcrf` / `gcrf_to_state` handle the position+velocity pair with the correct **TIRS-frame** Earth-rotation term (`ω⊕ × r_tirs`) for ITRF↔GCRF. Validated against Vallado Example 3-14.
- **New guide: "Theory: Maneuver Coordinate Frames"** (`docs/guide/maneuver_frames.md`) — side-by-side GCRF / RTN / NTW / LVLH comparison with flight-path-angle derivation, a worked numeric example on an e=0.3 orbit showing the 0.245 m/s discrepancy, a cheat sheet, and a summary table.

### Breaking: Unified Uncertainty API

- **`SatState::set_pos_uncertainty(sigma, frame)` and `set_vel_uncertainty(sigma, frame)`** replace the four per-frame methods (`set_lvlh_pos_uncertainty`, `set_lvlh_vel_uncertainty`, etc.). Supports `GCRF`, `LVLH`, `RTN`, and `NTW`. Each call preserves the 3×3 block it is not updating, so pos-then-vel correctly builds a full 6×6 covariance — the old methods silently overwrote the whole matrix. **Old methods are removed, not deprecated.**
- **Doc fix**: the default covariance frame is `LVLH`, not `RIC` as previously documented in several places.

### Breaking: Python API Parity with Rust

- `satstate.add_maneuver(time, delta_v, frame)`, `set_pos_uncertainty`, `set_vel_uncertainty`, and `thrust.constant` now all require an **explicit** `frame` argument from Python, matching Rust (no silent defaults).
- Added ergonomic helpers on `satstate`: `add_prograde`, `add_retrograde`, `add_radial`, `add_normal` alongside the generic `add_maneuver`.
- Added matching Rust constructors on `ImpulsiveManeuver`: `prograde`, `retrograde`, `radial_out`, `normal`, plus `gcrf` / `rtn` / `ntw` for arbitrary-vector burns.

### Default Gravity Model: EGM96

- **Breaking (subtle)**: `PropSettings::default()` now uses `GravityModel::EGM96` instead of `JGM3`. EGM96 is a more modern and more widely used model; the numerical difference for typical LEO propagation is sub-meter over a day, but the default selection changes. Python `propsettings()` and the standalone `gravity()` / `gravity_and_partials()` helpers pick up the new default automatically.

### Documentation

- **Coordinate Frame Transforms tutorial** rewritten: explicit GCRS/ICRF and ITRS definitions (quasar VLBI realisation vs. ground-tracking realisation), geodetic-vs-geocentric explanation, ground-track overlay on a cartopy `PlateCarree` map, time-series plot of `qgcrf2itrf_approx` vs full IERS 2010 error over 30 years. Dropped the low-value 24-hour Earth-rotation section.
- **New API reference page**: `docs/api/frame.md` documenting the `Frame` enum with all variants and aliases.
- **MathJax** now accepts both `\(...\)` / `\[...\]` and `$...$` / `$$...$$` delimiters, so equations render correctly in Jupyter-notebook tutorials (previously broken in `Quaternions.ipynb` and others).
- **Covariance Propagation** tutorial simplified to use the unified uncertainty API (dropped the hand-rolled LVLH→GCRF rotation).
- **High Precision Propagation**, **satprop guide**, **maneuver/covariance examples** updated for the new API.

### Tutorial Reorganization

- **Renamed "Coordinate Frame Transforms" → "Coordinate Frames"**. The tutorial is primarily a description of the frames themselves (GCRF, ITRF, TEME), not just the rotations between them.
- **Renamed "ITRF Coordinates" → "Geodetic Coordinates"**. This tutorial is about the `itrfcoord` data type (geodetic / Cartesian / ENU / NED / geodesic distance), not the ITRF reference frame. The old name made it sound like two views of the same topic as Coordinate Frames.
- Nav reordered so **Coordinate Frames** (frame theory) comes before **Geodetic Coordinates** (data type built on top). Reciprocal cross-reference notes added at the top of both tutorials.
- **Expanded TEME section** in Coordinate Frames: origin of the "True Equator, Mean Equinox" name (intentional half-and-half construction), the three practical awkwardness points (not uniquely defined, time-dependent orientation, positions cannot be compared directly), API table, and Vallado 2006 reference.
- **"Why yet another time type?" section** added at the top of the Time Systems tutorial, covering time-scale-as-first-class, microsecond-precision i64 representation, correct leap-second handling, built-in UT1/TDB, and the single-type-across-Rust-and-Python story.

### Propagator: Configurable `max_steps`

- **`PropSettings::max_steps`** (Rust) / **`propsettings(max_steps=...)`** (Python): configurable maximum number of integrator steps before the propagator aborts with a max-steps error. Applies uniformly to the adaptive Runge-Kutta / Rosenbrock solvers (via `numeris::ode::AdaptiveSettings::max_steps`) and the Gauss-Jackson 8 solver (via its own internal settings). Default: 1_000_000, which matches the previous hard-coded Gauss-Jackson 8 ceiling and is a loosening of the previously inherited numeris RK default of 100_000. This covers very long arcs (≈700 days of GJ8 at 60 s step) with headroom; lower for a tighter runaway-propagation safeguard.

### Release Tooling

- **`release.yml` `check_version`**: now verifies the tag matches *all three* version strings — root `Cargo.toml`, `python/Cargo.toml`, and `pyproject.toml`. Previously only the root Cargo.toml was checked, which allowed `python/Cargo.toml` to drift silently if a version bump was applied by hand instead of through `cargo release`. Any future drift will hard-block the release workflow.

### Bug Fixes

- **MathJax in Jupyter notebooks**: remove the `ignoreHtmlClass` / `processHtmlClass: "arithmatex"` restriction in `docs/javascripts/mathjax.js` that caused MathJax to skip notebook HTML entirely (mkdocs-jupyter does not wrap notebook-cell math in an `.arithmatex` span). Equations in the Quaternions tutorial and all other notebooks now render. Plain markdown pages still work because pymdownx.arithmatex (generic mode) emits raw delimiters that MathJax picks up under default scanning.
- **`test_gravity`**: explicitly pin to `model=sk.gravmodel.jgm3`. The ICGEM reference values in that test are for JGM3 specifically; previously they relied on the default, which switched to EGM96 in this release.

### Internal

- `Frame` derives `Copy + PartialEq + Eq`; `PyFrame::NTW`, `PyIntegrator::gauss_jackson8`, `PyPropSettings::gj_step_seconds`, `PyPropResult::gj_dense` exposed in the Python bindings.
- `.pyi` stubs updated throughout: new frame variants, new integrator variant, `gj_step_seconds`, unified uncertainty API, ergonomic maneuver helpers, corrected frame docstrings.
- All 20 tutorial notebooks re-executed and stripped of outputs (mkdocs-jupyter re-executes at build time).
- 157 Rust tests + 81 Python tests pass (up from 133 / 71 at 0.15.1).


## 0.15.1 - 2026-03-29

### URL Loading

- **`TLE.from_url(url)`**: Load TLE(s) directly from a URL returning plain-text TLE data
- **`OMM.from_url(url)`** (Rust): Load OMM(s) from a URL with auto-detection of JSON vs XML format
- **`omm_from_url(url)`** (Python): Fetch OMMs from a URL and return as a list of dictionaries, compatible with `sgp4()`

### SatState Documentation

- Expose `set_lvlh_vel_uncertainty()` and `set_gcrf_vel_uncertainty()` in Python bindings
- Rewrite `SatState` struct and type stub documentation: when to use it vs `propagate()`, units, pickle support
- Add `SatState` section to user guide with comparison table and code examples
- Add all missing method stubs to `satkit.pyi`

### Plot Styling

- Extract shared matplotlib style to `docs/satkit.mplstyle`, replacing duplicated `rcParams` blocks across 18 notebooks (22 occurrences, -386 lines)

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
