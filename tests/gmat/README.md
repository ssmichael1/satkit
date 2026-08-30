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
accumulates over the full arc (7 days, or 3 days for the drag cases) exactly as a
real propagation would. The gate is the maximum residual over all samples. On failure the whole residual history
is printed so the log shows *when* the divergence begins (e.g. a perigee pass or a
lunar encounter), not just that it happened.

satkit settings used on the replay side: `RKV98NoInterp`, `abs_error = rel_error
= 1e-13`, no SRP in any case, everything else from the case's `force_model`
block. The drag cases pass `SatPropertiesSimple(Cd·A/m)` from the case's
`spacecraft` block; only the file-driven ones (`drag_*_sw`) turn on
`use_spaceweather`, so the rest of the corpus does not depend on `SW-All.csv`.

## Orbital regimes

The gravity/third-body cases share the epoch 2023-05-16 20:00:00 UTC and run for
7 days with hourly samples. Elements are Keplerian in EarthICRF (km, deg); TESS
is given as a Cartesian state.

| orbit | a (km) | e | i (°) | period | what it stresses |
|---|---|---|---|---|---|
| `leo_iss` | 6778 | 0.001 | 51.6 | 93 min | high-degree gravity, tides, ~109 revs of J2 precession |
| `sso_800` | 7178 | 0.0012 | 98.6 | 101 min | near-polar: zonal/tesseral mix, sun-synchronous node rate |
| `meo_gps` | 26560 | 0.01 | 55.0 | 12 h | 2:1 tesseral resonance, Sun/Moon, same regime as the SP3 test |
| `molniya` | 26600 | 0.74 | 63.4 | 12 h | 500 km perigee → 39,000 km apogee: adaptive step control, tides at perigee |
| `geo` | 42164 | 0.0003 | 0.1 | 24 h | lunisolar-dominated, near-degenerate elements |
| `tess` | Cartesian | ~0.55 | ~37 | 13.7 d | 2:1 lunar-resonant HEO — the case that exposed the `MU_MOON` error |
| `cislunar` | 300000 | 0.0 | 20.0 | 19 d | third-body dominated, closest approach to the Moon ~85,000 km |

The drag orbits start at 2023-03-01 00:00:00 UTC — well inside the *observed*
block of the CelesTrak space-weather file on both sides, two days after a G2
storm (Ap 91 on 2023-02-27) so the 3-hourly ap history still matters on day 1 —
and run for 3 days (drag error grows ~t², and 7 days at 300 km would be
dominated by it). All carry the same spacecraft: `Cd = 2.2`, `DragArea = 10 m²`,
`DryMass = 1000 kg` (satkit: `Cd·A/m = 0.022 m²/kg`).

| orbit | a (km) | e | i (°) | perigee alt | what it stresses |
|---|---|---|---|---|---|
| `iss_420` | 6798 | 0.0005 | 51.6 | 420 km | ISS regime; every local solar time sampled |
| `leo_300` | 6678 | 0.0005 | 45.0 | 300 km | strongest drag: 1,400 km along-track over 3 days |
| `sso_550` | 6928 | 0.001 | 97.6 | 550 km | sun-synchronous: fixed local time, polar passes, helium/anomalous-oxygen regime |
| `gto_250` | 24396 | 0.728 | 27.0 | 250 km | drag as a perigee impulse; adaptive step through a 1,000 s drag pass |

## Force models

| name | Earth gravity | Sun/Moon | solid tides | relativity | purpose |
|---|---|---|---|---|---|
| `j2` | EGM96 2×2 | on | off | off | isolates μ, ephemeris, frame orientation, time |
| `full` | EGM96 36×36 | on | on (`Solid` / `SolidStep1`) | off | everything both tools model the same way |
| `gr` | EGM96 36×36 | on | on | on | full IERS 2010 eq. 10.12 relativity on both sides |
| `drag_const` | as `full` | on | on | off | + NRLMSISE-00 drag with fixed F10.7 = F10.7A = 150, Ap = 4: the density model and drag force alone |
| `drag_sw` | as `full` | on | on | off | + NRLMSISE-00 drag driven by the CelesTrak space-weather file on both sides: the whole chain, including each tool's F10.7/Ap feed conventions |

Cases: every gravity orbit × `j2` and `full`, plus `gr` for `leo_iss`, `tess` and
`cislunar` (17), and every drag orbit × `drag_const` and `drag_sw` (8) — 25 in
total.

The constant-weather values are exactly what satkit's NRLMSISE-00 assumes with
`use_spaceweather = false`, so no new API is needed to isolate the density model
from the feed. GMAT's `Drag.MagneticIndex` is *Kp*, converted internally with a
table lookup; Kp = 1 maps to Ap = 4 exactly, and Ap = 4 is also where the daily-Ap
and 3-hourly-ap formulations of NRLMSISE-00 coincide, so the formulation cannot
leak into the constant cases.

## GMAT configuration

* GMAT R2026A, `GmatConsole` headless, `RungeKutta89`, `Accuracy = 1e-14`,
  `ErrorControl = RSSStep`, `MaxStep = 2700`. GMAT's models are documented in
  the GMAT Mathematical Specifications (see [References](https://satkit.dev/guide/references/#gmatspec)).
* **Frame**: `EarthICRF`, not `EarthMJ2000Eq` — matches satkit's GCRF
  exactly. GMAT's `EarthMJ2000Eq` is *not* the IERS constant-bias EME2000
  (satkit's `Frame::EME2000`, 23 mas from GCRF): GMAT realizes it through the
  IAU-76/FK5 precession model via `data/icrf/ICRF_Table.txt`, and its offset
  from ICRF is time-varying — ≈ 44 mas (1.5 m at 7000 km) at the corpus epoch,
  growing ≈ 2.5 mas/yr (satkit's own measurement from GMAT's table, consistent
  with the IAU-76 precession-rate error).
* **Ephemeris**: SPICE `de440.bsp` (GMAT bundles only DE405/421/424; satkit uses
  DE440).
* **Body GMs** pinned to the DE440 values satkit uses (`Luna.Mu = 4902.800118`,
  etc.) and recorded in the JSON, so a wrong constant in satkit shows up as a
  residual against a reviewable reference value. The harmonic field's μ
  (398600.4415) comes from the coefficient file on both sides.
* Gravity file `EGM96.cof`; its coefficients are byte-identical to satkit's
  `EGM96.gfc`.
* **Drag** (`drag_*` cases): `Drag.AtmosphereModel = 'NRLMSISE00'`,
  `Drag.DragModel = 'Spherical'`; constant weather via
  `HistoricWeatherSource = PredictedWeatherSource = 'ConstantFluxAndGeoMag'`
  with `Drag.F107 = Drag.F107A = 150`, `Drag.MagneticIndex = 1` (Kp);
  file-driven via `'CSSISpaceWeatherFile'` and `Drag.CSSISpaceWeatherFile`
  pointing at CelesTrak's `SW-All.txt` (the `.txt` twin of satkit's
  `SW-All.csv`; the file's `UPDATED` stamp is recorded in the JSON).
  `drag_leo300_sw` runs at `Accuracy = 1e-13`: GMAT's RK89 refuses 1e-14
  through the file-driven weather steps at 300 km ("Accuracy settings will
  be violated").

## Measured floors and the gates

Gates are ~3× the residual measured when the corpus was generated and are
recorded per case in `cases.py` (and copied into each JSON). The floors:

| source | size over 7 days | cause |
|---|---|---|
| GMAT integration error | 3 cm LEO, 13 cm GEO, 0.6–1.0 m at 200,000–300,000 km | GMAT's own point-mass runs deviate from the analytic Kepler solution by exactly these amounts (satkit matches Kepler to <1 cm); independent of GMAT's `Accuracy`, `MaxStep`, or integrator |
| solid tides (`full`, `gr`) | +0.4–0.7 m LEO, ~2 m Molniya | satkit's `SolidStep1` uses the IERS 2010 Table 6.3 anelastic Love numbers with their imaginary (phase-lag) parts; GMAT's `Solid` uses real Love numbers only. Zeroing satkit's imaginary parts reproduces GMAT to the `j2` floor — GMAT omits the lag, satkit does not |
| relativity (`gr`) | none | both tools apply Schwarzschild + geodesic precession + Lense–Thirring (IERS 2010 eq. 10.12); `gr` residuals equal the `full` (LEO) or `j2` (high-orbit) floors |
| NRLMSISE-00 implementation (`drag_*_const`) | 26 m ISS, 293 m at 300 km, 10 m GTO over 3 days = 0.5–2 × 10⁻⁴ of the drag-only displacement | fixed-index density along the ISS orbit agrees to +0.01 % mean / 0.06 % rms (GMAT `AtmosDensity` vs satkit at the same lat/lon/alt/time); the residual is the integrated effect of that plus integration noise |
| anomalous oxygen (`drag_sso550_const`) | 34 m over 3 days = 1.8 × 10⁻³ of the drag-only displacement | GMAT evaluates NRLMSISE-00 through `gtd7`, whose total density omits the anomalous-oxygen component; satkit uses `gtd7d`, the entry point the model's authors specify for drag. The difference is +0.3 % mean at 550–600 km and +1 % over the high-latitude (southern, in March) polar passes; reproducing `gtd7` in satkit brings the density difference to +0.005 % mean / 0.09 % rms. The ISS orbit at 420 km sees < 0.05 % |
| space-weather feed (`drag_*_sw`) | 325 m ISS, 2.3 km at 300 km, 37 m SSO, 307 m GTO peak over 3 days = 1.3–2.3 × 10⁻³ of the drag-only displacement (198 m, 2.1 km, 18 m, 113 m at the end of the arc) | Both tools feed NRLMSISE-00 the 3-hourly ap history (7-element array, switch 9 = −1; satkit since the 3-hourly feed was adopted, current-day daily Ap in element 0 on both sides). What remains is the F10.7 timing: GMAT interpolates the daily F10.7 linearly between 20:00 UT nodes and switches F10.7A at 08:00 UT, satkit steps both at 00:00 UT. The residual oscillates rather than accumulating (it peaks mid-arc and shrinks towards the end), as a timing offset in a daily-stepped index would. Before the 3-hourly feed the same cases sat at 6.8 km / 57 km / 1.1 km / 12 km (3.5–4.8 %): along the ISS orbit two days after the G2 storm the daily-Ap formulation was −1.5 % mean / 5.3 % rms / 24 % max in density against GMAT, and replicating GMAT's whole feed (ap array + F10.7 timing) inside satkit's model brought it to +0.04 % / 0.5 % / 9.7 % |

Everything else — μ's, DE440, the GCRF↔ITRF frame, 36×36 gravity, third-body
— agrees at the centimetre level. (Building this corpus found that the
propagator's precomputed frame table used the ~1″ IAU-76/FK5 approximation,
which drifted the ISS case by 50 m; fixed alongside this corpus. The drag cases
found that `drag.rs` passed geodetic latitude/longitude to NRLMSISE-00 in
radians instead of degrees — density wrong by up to +200 % pointwise, 8 km over
3 days at ISS altitude — and that the space-weather feed used the 1 AU-adjusted
instead of the observed F10.7 and the previous day's Ap; both fixed alongside.
The `drag_*_sw` gates were re-measured when satkit adopted the 3-hourly ap
history; the GMAT trajectories themselves were not regenerated.)

The drag gates are large in metres because the drag effect itself is large; read
them against the drag-only displacement listed per case in `cases.py`. The
`report` diagnostic prints that displacement and the ratio for every drag case.

## Regenerating

```bash
# one-time: DE440 kernel and the CelesTrak space-weather file (do not commit either)
curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp
curl -O https://celestrak.org/SpaceData/SW-All.txt

python tests/gmat/generate.py --gmat "~/Projects/GMAT R2026A" --spk ./de440.bsp --sw-file ./SW-All.txt
cargo test --test gmat_regression
```

All 25 cases regenerate in a few minutes (the drag cases take ~15 s each); the dlopen warnings GMAT prints
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
                   "sun": true, "moon": true, "tides": "SolidStep1", "relativity": false,
                   "drag": null },
  "tolerance": { "pos_m": 0.3, "vel_mps": 5e-5 },
  "samples": [[0.0, x, y, z, vx, vy, vz], [3600.0000004, ...], ...]
}
```

`samples` rows are `[elapsed_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms]` in
EarthICRF; `elapsed_s` is GMAT's reported value (its stop condition overshoots
by ~4e-7 s), and the tests use it as the true sample time.

Drag cases add `"epoch": ...` and `"spacecraft": {"cd", "drag_area_m2",
"dry_mass_kg"}` to `orbit`, and `force_model.drag` is either
`{"atmosphere": "NRLMSISE00", "weather": "constant", "f107": 150.0, "f107a": 150.0,
"ap": 4.0, "gmat_kp": 1.0}` or `{"atmosphere": "NRLMSISE00", "weather":
"CSSISpaceWeatherFile"}`; the file-driven cases also record
`gmat.space_weather_file` (URL and the file's `UPDATED` stamp).
