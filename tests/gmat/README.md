# GMAT regression tests

Regression tests for satkit's high-precision orbit propagator against
reference trajectories from NASA's General Mission Analysis Tool (GMAT).

GMAT cannot run in CI, so the references are generated offline with a local
GMAT install and committed here as JSON. Both language front-ends replay them on
every commit:

| | file | runs in |
|---|---|---|
| Rust | `tests/gmat_regression.rs` — one `#[test]` per case | `cargo test` (`rust` CI job) |
| Python | `python/test/test_gmat.py` — parametrized over the same files | `pytest python/test/` (`python` CI job) |

## Layout

| file | role |
|---|---|
| `cases/*.json` | the corpus: one reference trajectory per case (see format below) |
| `cases.py` | the case matrix — orbits × force models — and the tolerance gates |
| `generate.py` | writes GMAT scripts, runs `GmatConsole`, parses the reports, writes `cases/*.json` |
| `prop_baseline.script` | the original hand-written TESS script that exposed the `MU_MOON` error |

## What a test does

For each case it takes GMAT's state at *t = 0*, propagates with satkit to the
next hourly sample, compares position and velocity, then continues **from its
own propagated state** (not GMAT's) to the next sample — so the residual
accumulates over the full 7 days exactly as a real propagation would. The gate is
the maximum residual over all 168 samples. On failure the whole residual history
is printed so the log shows *when* the divergence begins (e.g. a perigee pass or a
lunar encounter), not just that it happened.

satkit settings used on the replay side: `RKV98NoInterp`, `abs_error = rel_error
= 1e-13`, space weather off (no drag or SRP in any case), everything else from
the case's `force_model` block.

## Orbital regimes

All cases share the epoch 2023-05-16 20:00:00 UTC and run for 7 days with hourly
samples. Elements are Keplerian in EarthICRF (km, deg); TESS is given as a
Cartesian state.

| orbit | a (km) | e | i (°) | period | what it stresses |
|---|---|---|---|---|---|
| `leo_iss` | 6778 | 0.001 | 51.6 | 93 min | high-degree gravity, tides, ~109 revs of J2 precession |
| `sso_800` | 7178 | 0.0012 | 98.6 | 101 min | near-polar: zonal/tesseral mix, sun-synchronous node rate |
| `meo_gps` | 26560 | 0.01 | 55.0 | 12 h | 2:1 tesseral resonance, Sun/Moon, same regime as the SP3 test |
| `molniya` | 26600 | 0.74 | 63.4 | 12 h | 500 km perigee → 39,000 km apogee: adaptive step control, tides at perigee |
| `geo` | 42164 | 0.0003 | 0.1 | 24 h | lunisolar-dominated, near-degenerate elements |
| `tess` | Cartesian | ~0.55 | ~37 | 13.7 d | 2:1 lunar-resonant HEO — the case that exposed the `MU_MOON` error |
| `cislunar` | 300000 | 0.0 | 20.0 | 19 d | third-body dominated, closest approach to the Moon ~85,000 km |

## Force models

| name | Earth gravity | Sun/Moon | solid tides | relativity | purpose |
|---|---|---|---|---|---|
| `j2` | EGM96 2×2 | on | off | off | isolates μ, ephemeris, frame orientation, time |
| `full` | EGM96 36×36 | on | on (`Solid` / `SolidStep1`) | off | everything both tools model the same way |
| `gr` | EGM96 36×36 | on | on | on | known model gap, see below |

Cases: every orbit × `j2` and `full`, plus `gr` for `leo_iss`, `tess` and
`cislunar` — 17 in total.

## GMAT configuration

* GMAT R2026A, `GmatConsole` headless, `RungeKutta89`, `Accuracy = 1e-14`,
  `ErrorControl = RSSStep`, `MaxStep = 2700`.
* **Frame**: `EarthICRF`, not `EarthMJ2000Eq` — matches satkit's GCRF without
  the ~23 mas frame bias.
* **Ephemeris**: SPICE `de440.bsp` (GMAT bundles only DE405/421/424; satkit uses
  DE440).
* **Body GMs** pinned to the DE440 values satkit uses (`Luna.Mu = 4902.800118`,
  etc.) and recorded in the JSON, so a wrong constant in satkit shows up as a
  residual against a reviewable reference value. The harmonic field's μ
  (398600.4415) comes from the coefficient file on both sides.
* Gravity file `EGM96.cof`; its coefficients are byte-identical to satkit's
  `EGM96.gfc`.

## Measured floors and the gates

Gates are ~3× the residual measured when the corpus was generated and are
recorded per case in `cases.py` (and copied into each JSON). The floors:

| source | size over 7 days | cause |
|---|---|---|
| GMAT integration error | 3 cm LEO, 13 cm GEO, 0.6–1.0 m at 200,000–300,000 km | GMAT's own point-mass runs deviate from the analytic Kepler solution by exactly these amounts (satkit matches Kepler to <1 cm); independent of GMAT's `Accuracy`, `MaxStep`, or integrator |
| solid tides (`full`, `gr`) | +0.4–0.7 m LEO, ~2 m Molniya | GMAT's `Solid` includes the IERS 2010 Step-2 frequency-dependent terms; satkit's `SolidStep1` omits them |
| relativity (`gr`) | +0.2 m LEO, +1 m at 200,000 km | GMAT adds geodesic precession and Lense–Thirring; satkit implements Schwarzschild only |

Everything else — μ's, DE440, the GCRF↔ITRF frame, 36×36 gravity, third-body
— agrees at the centimetre level. (Building this corpus found that the
propagator's precomputed frame table used the ~1″ IAU-76/FK5 approximation,
which drifted the ISS case by 50 m; fixed alongside this corpus.)

## Regenerating

```bash
# one-time: DE440 kernel
curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp

python tests/gmat/generate.py --gmat "~/Projects/GMAT R2026A" --spk ./de440.bsp
cargo test --test gmat_regression
```

All 17 cases regenerate in under a minute; the dlopen warnings GMAT prints
about missing proprietary plugins are benign. Useful variations:

```bash
python tests/gmat/generate.py ... --only tess_full leo_iss_j2   # a subset
python tests/gmat/generate.py --update-tolerances               # gates only, no GMAT
SATKIT_GMAT_CASE_DIR=/scratch/cases cargo test --test gmat_regression -- --ignored report --nocapture
```

The last form evaluates ad-hoc cases without touching the corpus or the test
list — handy for bisecting a discrepancy by force-model component.

**Adding a case**: append it to `cases.py`, run `generate.py`, add the name to
`gmat_cases!` in `tests/gmat_regression.rs` (the `every_case_file_has_a_test`
test enforces the two stay in sync). The Python test picks it up from the glob.
**Tightening a gate** should accompany the model improvement that earns it.

## JSON format

```json
{
  "name": "meo_gps_full",
  "gmat":  { "version": "...", "ephemeris": "SPICE de440.bsp (DE440)", "coordinate_system": "EarthICRF",
             "integrator": "RungeKutta89", "accuracy": 1e-14,
             "mu_earth_km3s2": 398600.4418, "mu_moon_km3s2": 4902.800118, "mu_sun_km3s2": 132712440041.27942 },
  "epoch_utc": "2023-05-16T20:00:00",
  "orbit": { "kep": [26560.0, 0.01, 55.0, 200.0, 30.0, 100.0], "days": 7 },
  "force_model": { "gravity_model": "EGM96", "gravity_degree": 36, "gravity_order": 36,
                   "sun": true, "moon": true, "tides": "SolidStep1", "relativity": false },
  "tolerance": { "pos_m": 0.3, "vel_mps": 5e-5 },
  "samples": [[0.0, x, y, z, vx, vy, vz], [3600.0000004, ...], ...]
}
```

`samples` rows are `[elapsed_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms]` in
EarthICRF; `elapsed_s` is GMAT's reported value (its stop condition overshoots
by ~4e-7 s), and the tests use it as the true sample time.
