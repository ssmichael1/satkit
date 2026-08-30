# Validation: GMAT Comparison

The numerical propagator is checked on every commit against reference trajectories from NASA's [General Mission Analysis Tool (GMAT)](https://gmat.atlassian.net/), an independent, operationally used high-fidelity propagator whose own verification against STK and FreeFlyer is described by [Hughes et al. (2014)](references.md#hughes2014) and whose models are documented in the [GMAT Mathematical Specifications](references.md#gmatspec). This page describes what is compared, how well the two agree, and where the remaining differences come from.

## Why a committed corpus

GMAT is a large GUI-oriented application that cannot run inside a CI job. The comparison is therefore done the same way satkit validates against SP3 and Vallado test vectors: GMAT trajectories are generated offline with a local install, committed to the repository as JSON under `tests/gmat/cases/`, and replayed under `cargo test` and `pytest`. Each file records the epoch, initial state, force model, GMAT version and settings, the body GMs used, an hourly state history, and the tolerance that gates the test — so a reviewer can see exactly what the reference is.

## GMAT configuration

| setting | value | why |
|---|---|---|
| GMAT | R2026A, `GmatConsole` headless | models per the [GMAT Mathematical Specifications](references.md#gmatspec) |
| integrator | `RungeKutta89`, `Accuracy = 1e-14`, `ErrorControl = RSSStep` | tight enough that GMAT's own error is well below the gates for most cases (see [floors](#known-differences-and-the-gates)) |
| frame | `EarthICRF` | matches satkit's GCRF exactly; GMAT's `EarthMJ2000Eq` is an IAU-76/FK5 realization whose offset from ICRF is time-varying (≈ 44 mas ≈ 1.5 m at 7000 km in 2023), not the IERS constant 23 mas bias ([Petit & Luzum 2010](references.md#petit2010), §5.5.4) of satkit's `EME2000` |
| ephemeris | SPICE `de440.bsp` | GMAT bundles only DE405/421/424; satkit uses DE440 ([Park et al. 2021](references.md#park2021)) |
| body GMs | pinned to the DE440 values satkit uses and recorded in the JSON | a wrong constant in satkit shows up as a residual against a reviewable reference value |
| gravity file | `EGM96.cof` | coefficients identical to satkit's `EGM96.gfc`; the field's $GM$ comes from the file on both sides |
| drag (`drag_*` cases) | `AtmosphereModel = 'NRLMSISE00'`, `DragModel = 'Spherical'`; constant weather via `ConstantFluxAndGeoMag` (F10.7 = F10.7A = 150, `MagneticIndex` Kp = 1 ⇒ Ap = 4) or file-driven via `CSSISpaceWeatherFile` pointing at CelesTrak's `SW-All.txt` | the same model satkit uses ([Picone et al. 2002](references.md#picone2002)), fed either fixed indices or the `.txt` twin of satkit's `SW-All.csv`; `drag_leo300_sw` needs `Accuracy = 1e-13` (GMAT's RK89 refuses 1e-14 through the weather steps at 300 km) |

The last point is what makes the corpus a *constants* test as well as a *dynamics* test: the erroneous `MU_MOON` that motivated this work (a $4 \times 10^{-4}$ relative error, 1.3 km over 7 days on a lunar-resonant orbit) would fail the `tess` cases immediately.

## Orbital regimes

The gravity/third-body cases share the epoch 2023-05-16 20:00:00 UTC and run for 7 days with hourly samples.

| orbit | $a$ (km) | $e$ | $i$ (°) | period | what it stresses |
|---|---|---|---|---|---|
| `leo_iss` | 6778 | 0.001 | 51.6 | 93 min | high-degree gravity, tides, ~109 revolutions of J2 precession |
| `sso_800` | 7178 | 0.0012 | 98.6 | 101 min | near-polar: zonal/tesseral mix, sun-synchronous node rate |
| `meo_gps` | 26560 | 0.01 | 55.0 | 12 h | 2:1 tesseral resonance, Sun/Moon; same regime as the SP3 test |
| `molniya` | 26600 | 0.74 | 63.4 | 12 h | 500 km perigee to 39,000 km apogee: step control, tides at perigee |
| `geo` | 42164 | 0.0003 | 0.1 | 24 h | lunisolar-dominated, near-degenerate elements |
| `tess` | (Cartesian) | ~0.55 | ~37 | 13.7 d | 2:1 lunar-resonant HEO — the case that exposed the `MU_MOON` error |
| `cislunar` | 300000 | 0.0 | 20.0 | 19 d | third-body dominated; closest approach to the Moon ~85,000 km |

The drag orbits start at 2023-03-01 00:00:00 UTC — inside the *observed* block of the CelesTrak space-weather file on both sides, two days after a G2 storm (Ap 91 on 2023-02-27) so the 3-hourly ap history still matters on day 1 — and run for 3 days: drag error grows roughly as $t^2$, and 7 days at 300 km would be dominated by it. All carry the same spacecraft, $C_d = 2.2$, area 10 m², mass 1000 kg ($C_d A/m = 0.022$ m²/kg).

| orbit | $a$ (km) | $e$ | $i$ (°) | perigee altitude | what it stresses |
|---|---|---|---|---|---|
| `iss_420` | 6798 | 0.0005 | 51.6 | 420 km | ISS regime; every local solar time sampled |
| `leo_300` | 6678 | 0.0005 | 45.0 | 300 km | strongest drag: 1,400 km along-track over 3 days |
| `sso_550` | 6928 | 0.001 | 97.6 | 550 km | sun-synchronous: fixed local time, polar passes, helium / anomalous-oxygen regime |
| `gto_250` | 24396 | 0.728 | 27.0 | 250 km | drag as a perigee impulse; adaptive step through a ~1,000 s drag pass |

## Force models

Each orbit is run under two or three force models so that a discrepancy can be attributed to a component:

| name | Earth gravity | Sun/Moon | solid tides | relativity | purpose |
|---|---|---|---|---|---|
| `j2` | EGM96 2×2 | on | off | off | isolates $GM$ values, ephemeris, frame orientation and time |
| `full` | EGM96 36×36 | on | on | off | everything both tools model the same way |
| `gr` | EGM96 36×36 | on | on | on | exercises the relativistic correction (below) |
| `drag_const` | as `full` | on | on | off | + NRLMSISE-00 drag with fixed F10.7 = F10.7A = 150, Ap = 4: the density model and drag force alone |
| `drag_sw` | as `full` | on | on | off | + NRLMSISE-00 drag driven by the CelesTrak space-weather file on both sides: the whole chain, including each tool's F10.7 / Ap feed conventions |

Every gravity orbit is run with `j2` and `full`; `leo_iss`, `tess` and `cislunar` are also run with `gr` (17 cases); every drag orbit is run with `drag_const` and `drag_sw` (8 cases) — 25 in total. Solar radiation pressure is off throughout: its inputs (shadow model, reflectivity conventions) cannot be matched closely enough between the two tools for the residual to say anything about satkit.

The constant-weather values are exactly what satkit's NRLMSISE-00 assumes with `use_spaceweather = false`, so no special API is needed to isolate the density model from the feed. Ap = 4 (GMAT's `MagneticIndex` is Kp; Kp = 1 maps to Ap = 4 exactly) is also where the daily-Ap and 3-hourly-ap formulations of NRLMSISE-00 coincide, so the feed formulation cannot leak into the constant cases.

## How the test works

The test is implemented twice — `tests/gmat_regression.rs` (one `#[test]` per case) and `python/test/test_gmat.py` (parametrized over the same files) — so both the Rust core and the Python bindings are exercised.

For each case the test takes GMAT's state at $t = 0$, propagates with satkit to the next hourly sample and compares position and velocity, then continues **from its own propagated state** to the next sample. Errors therefore accumulate over the full arc (7 days, or 3 days for the drag cases) exactly as they would in a real propagation; the gate is the maximum residual over all samples. When a case fails, the whole residual history is printed so the log shows *when* the divergence begins — a lunar perigee passage or a resonance — rather than only that it happened.

The replay uses `rkv98_nointerp` with `abs_error = rel_error = 1e-13`, so satkit's own integration error is negligible against the gates. The drag cases pass `SatPropertiesSimple(Cd·A/m)` from the case's `spacecraft` block; only the file-driven ones turn on `use_spaceweather`.

## Measured agreement

With matched force models (`j2`: low-degree gravity, Sun and Moon), satkit and GMAT agree over 7 days to:

| orbit | max position residual | max velocity residual |
|---|---|---|
| `leo_iss` | 3 cm | 3 × 10⁻⁵ m/s |
| `sso_800` | 2 cm | 2 × 10⁻⁵ m/s |
| `meo_gps` | 8 cm | 1 × 10⁻⁵ m/s |
| `molniya` | 13 cm | 5 × 10⁻⁵ m/s |
| `geo` | 13 cm | 9 × 10⁻⁶ m/s |
| `tess` | 1.0 m | 3 × 10⁻⁶ m/s |
| `cislunar` | 0.6 m | 3 × 10⁻⁶ m/s |

The 36×36 field adds nothing measurable (36×36 without tides agrees to 2 cm at LEO); the increases under `full` and `gr` are the two model differences described next.

For the drag cases the residual has to be read against the size of the drag effect itself — the along-track displacement between a satkit run with and without drag — because the metres are large only because drag is:

| case | drag-only displacement | end-of-arc residual | max residual | fraction of the drag effect |
|---|---|---|---|---|
| `drag_iss_const` | 152 km | 26 m | 26 m | 1.7 × 10⁻⁴ |
| `drag_leo300_const` | 1374 km | 293 m | 293 m | 2.1 × 10⁻⁴ |
| `drag_sso550_const` | 19 km | 34 m | 34 m | 1.8 × 10⁻³ |
| `drag_gto_const` | 106 km | 6 m | 10 m | 5.5 × 10⁻⁵ |
| `drag_iss_sw` | 194 km | 198 m | 325 m | 1.6 × 10⁻³ |
| `drag_leo300_sw` | 1643 km | 2.1 km | 2.3 km | 1.3 × 10⁻³ |
| `drag_sso550_sw` | 27 km | 18 m | 37 m | 1.3 × 10⁻³ |
| `drag_gto_sw` | 126 km | 113 m | 307 m | 2.3 × 10⁻³ |

With fixed indices the two NRLMSISE-00 implementations agree to +0.01 % mean / 0.06 % rms in density along the ISS orbit (GMAT's `AtmosDensity` report against satkit at the same latitude, longitude, altitude and time); the residual is the integrated effect of that plus integration noise. The remaining differences are the model-level ones described next.

## Known differences and the gates

Gates are set at roughly three times the residual measured when the corpus was generated, so a real regression trips them while the following documented floors do not.

**GMAT's own integration error.** The 0.6–1.0 m residuals on `tess` and `cislunar` are not satkit's: GMAT's point-mass-only runs on those orbits deviate from the analytic Kepler solution by exactly those amounts (13 cm at GEO, 3 cm at LEO), independent of GMAT's `Accuracy`, `MaxStep`, or choice of integrator, while satkit matches the analytic solution to under a centimetre. This sets the floor for the high-altitude cases.

**Solid Earth tides.** satkit's `SolidStep1` uses the IERS 2010 Table 6.3 *anelastic* Love numbers ([Petit & Luzum 2010](references.md#petit2010)), including their imaginary (phase-lag) parts (see [Force Model](forces.md#solid-earth-tides)); GMAT's `Solid` model uses real-valued Love numbers only, so it omits the lag. The lag is a small secular along-track effect: under `full` it accounts for 0.4–0.7 m at LEO over 7 days and about 2 m on Molniya, whose perigee passes sample the tidal field most strongly. Zeroing satkit's imaginary parts reproduces GMAT to the `j2` floor, so this is a term GMAT drops rather than one satkit is missing.

**Relativity.** Both tools apply the full IERS 2010 Eq. 10.12 correction ([Petit & Luzum 2010](references.md#petit2010); [GMAT Mathematical Specifications](references.md#gmatspec), §4.2.6) — Schwarzschild, geodesic (de Sitter) precession and Lense–Thirring (see [Force Model](forces.md#general-relativistic-correction)). The `gr` cases therefore add no residual of their own: they sit at the `full` floor at LEO and at GMAT's integration floor at 200,000 km and beyond. (Before the geodesic and Lense–Thirring terms were added, satkit's Schwarzschild-only model left ~1 m over 7 days at 200,000 km, where the geodesic term is the dominant relativistic acceleration.)

**Anomalous oxygen.** GMAT evaluates NRLMSISE-00 through `gtd7`, whose total density omits the anomalous-oxygen component; satkit uses `gtd7d`, the entry point the model's authors specify for drag ([Picone et al. 2002](references.md#picone2002)). The difference is +0.3 % mean at 550–600 km and +1 % over the high-latitude polar passes, which is why `drag_sso550_const` sits at 1.8 × 10⁻³ while the other constant cases are at 10⁻⁴; reproducing `gtd7` in satkit brings the density difference to +0.005 % mean / 0.09 % rms. At 420 km the effect is below 0.05 %.

**Space-weather feed.** Both tools feed NRLMSISE-00 the 7-element 3-hourly ap history (model switch 9 = −1, current-day daily Ap in element 0). What remains is the F10.7 timing: GMAT interpolates the daily F10.7 linearly between 20:00 UT nodes and switches F10.7A at 08:00 UT, while satkit steps both at 00:00 UT. The residual oscillates rather than accumulating — it peaks mid-arc and shrinks toward the end — as a timing offset in a daily-stepped index would. Before satkit adopted the 3-hourly history the same cases sat at 6.8 km / 57 km / 1.1 km / 12 km (3.5–4.8 % of the drag effect): along the ISS orbit two days after the G2 storm the daily-Ap formulation was −1.5 % mean / 5.3 % rms / 24 % max in density against GMAT, and replicating GMAT's whole feed inside satkit's own model brought it to +0.04 % / 0.5 % / 9.7 %.

## Regenerating the corpus

The corpus only needs regenerating when a case is added or the reference configuration changes; tightening a gate is a one-line edit. With a local GMAT install and NAIF's `de440.bsp`:

```bash
python tests/gmat/generate.py --gmat "~/Projects/GMAT R2026A" --spk ./de440.bsp
cargo test --test gmat_regression
```

All 25 cases regenerate in a few minutes (the drag cases also need CelesTrak's `SW-All.txt`, which like the kernel is not committed). The case matrix, gates and the measured floors live in `tests/gmat/cases.py`; operational details — adding a case, evaluating ad-hoc cases against a scratch directory, the JSON format — are in `tests/gmat/README.md` in the repository.

## See Also

- **Theory**: [Force Model](forces.md) for each term being compared; [ODE Integrators](integrators.md) for the replay settings.
- **Tutorial**: [GPS Example](../tutorials/GPS Example.ipynb) — validation of a different kind, fitting a GPS orbit against ESA SP3 truth.
- **Tutorial**: [SGP4 vs Numerical Propagation](../tutorials/SGP4 vs Numerical Propagation.ipynb).
- **References**: [GMAT Mathematical Specifications](references.md#gmatspec); [Hughes et al. 2014](references.md#hughes2014); [Petit & Luzum 2010](references.md#petit2010); [Park et al. 2021](references.md#park2021).
