# Validation: GMAT Comparison

The numerical propagator is checked on every commit against reference trajectories from NASA's [General Mission Analysis Tool (GMAT)](https://gmat.atlassian.net/), an independent, operationally used high-fidelity propagator. This page describes what is compared, how well the two agree, and where the remaining differences come from.

## Why a committed corpus

GMAT is a large GUI-oriented application that cannot run inside a CI job. The comparison is therefore done the same way satkit validates against SP3 and Vallado test vectors: GMAT trajectories are generated offline with a local install, committed to the repository as JSON under `tests/gmat/cases/`, and replayed under `cargo test` and `pytest`. Each file records the epoch, initial state, force model, GMAT version and settings, the body GMs used, an hourly state history, and the tolerance that gates the test — so a reviewer can see exactly what the reference is.

## GMAT configuration

| setting | value | why |
|---|---|---|
| GMAT | R2026A, `GmatConsole` headless | |
| integrator | `RungeKutta89`, `Accuracy = 1e-14`, `ErrorControl = RSSStep` | tight enough that GMAT's own error is well below the gates for most cases (see [floors](#known-differences-and-the-gates)) |
| frame | `EarthICRF` | matches satkit's GCRF; `EarthMJ2000Eq` differs by the ~23 mas frame bias |
| ephemeris | SPICE `de440.bsp` | GMAT bundles only DE405/421/424; satkit uses DE440 |
| body GMs | pinned to the DE440 values satkit uses and recorded in the JSON | a wrong constant in satkit shows up as a residual against a reviewable reference value |
| gravity file | `EGM96.cof` | coefficients identical to satkit's `EGM96.gfc`; the field's $GM$ comes from the file on both sides |

The last point is what makes the corpus a *constants* test as well as a *dynamics* test: the erroneous `MU_MOON` that motivated this work (a $4 \times 10^{-4}$ relative error, 1.3 km over 7 days on a lunar-resonant orbit) would fail the `tess` cases immediately.

## Orbital regimes

All cases share the epoch 2023-05-16 20:00:00 UTC and run for 7 days with hourly samples.

| orbit | $a$ (km) | $e$ | $i$ (°) | period | what it stresses |
|---|---|---|---|---|---|
| `leo_iss` | 6778 | 0.001 | 51.6 | 93 min | high-degree gravity, tides, ~109 revolutions of J2 precession |
| `sso_800` | 7178 | 0.0012 | 98.6 | 101 min | near-polar: zonal/tesseral mix, sun-synchronous node rate |
| `meo_gps` | 26560 | 0.01 | 55.0 | 12 h | 2:1 tesseral resonance, Sun/Moon; same regime as the SP3 test |
| `molniya` | 26600 | 0.74 | 63.4 | 12 h | 500 km perigee to 39,000 km apogee: step control, tides at perigee |
| `geo` | 42164 | 0.0003 | 0.1 | 24 h | lunisolar-dominated, near-degenerate elements |
| `tess` | (Cartesian) | ~0.55 | ~37 | 13.7 d | 2:1 lunar-resonant HEO — the case that exposed the `MU_MOON` error |
| `cislunar` | 300000 | 0.0 | 20.0 | 19 d | third-body dominated; closest approach to the Moon ~85,000 km |

## Force models

Each orbit is run under two or three force models so that a discrepancy can be attributed to a component:

| name | Earth gravity | Sun/Moon | solid tides | relativity | purpose |
|---|---|---|---|---|---|
| `j2` | EGM96 2×2 | on | off | off | isolates $GM$ values, ephemeris, frame orientation and time |
| `full` | EGM96 36×36 | on | on | off | everything both tools model the same way |
| `gr` | EGM96 36×36 | on | on | on | exercises the relativistic correction (a known model gap, below) |

Every orbit is run with `j2` and `full`; `leo_iss`, `tess` and `cislunar` are also run with `gr` — 17 cases in total. Drag and solar radiation pressure are off in all cases: their inputs (atmosphere model variant, space-weather source, shadow model) cannot be matched closely enough between the two tools for the residual to say anything about satkit.

## How the test works

The test is implemented twice — `tests/gmat_regression.rs` (one `#[test]` per case) and `python/test/test_gmat.py` (parametrized over the same files) — so both the Rust core and the Python bindings are exercised.

For each case the test takes GMAT's state at $t = 0$, propagates with satkit to the next hourly sample and compares position and velocity, then continues **from its own propagated state** to the next sample. Errors therefore accumulate over the full 7 days exactly as they would in a real propagation; the gate is the maximum residual over all 168 samples. When a case fails, the whole residual history is printed so the log shows *when* the divergence begins — a lunar perigee passage or a resonance — rather than only that it happened.

The replay uses `rkv98_nointerp` with `abs_error = rel_error = 1e-13`, so satkit's own integration error is negligible against the gates.

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

## Known differences and the gates

Gates are set at roughly three times the residual measured when the corpus was generated, so a real regression trips them while the following documented floors do not.

**GMAT's own integration error.** The 0.6–1.0 m residuals on `tess` and `cislunar` are not satkit's: GMAT's point-mass-only runs on those orbits deviate from the analytic Kepler solution by exactly those amounts (13 cm at GEO, 3 cm at LEO), independent of GMAT's `Accuracy`, `MaxStep`, or choice of integrator, while satkit matches the analytic solution to under a centimetre. This sets the floor for the high-altitude cases.

**Solid Earth tides.** satkit's `SolidStep1` uses the IERS 2010 Table 6.3 *anelastic* Love numbers, including their imaginary (phase-lag) parts (see [Force Model](forces.md#solid-earth-tides)); GMAT's `Solid` model uses real-valued Love numbers only, so it omits the lag. The lag is a small secular along-track effect: under `full` it accounts for 0.4–0.7 m at LEO over 7 days and about 2 m on Molniya, whose perigee passes sample the tidal field most strongly. Zeroing satkit's imaginary parts reproduces GMAT to the `j2` floor, so this is a term GMAT drops rather than one satkit is missing.

**Relativity.** Both tools apply the full IERS 2010 Eq. 10.12 correction — Schwarzschild, geodesic (de Sitter) precession and Lense–Thirring (see [Force Model](forces.md#general-relativistic-correction)). The `gr` cases therefore add no residual of their own: they sit at the `full` floor at LEO and at GMAT's integration floor at 200,000 km and beyond. (Before the geodesic and Lense–Thirring terms were added, satkit's Schwarzschild-only model left ~1 m over 7 days at 200,000 km, where the geodesic term is the dominant relativistic acceleration.)

!!! note "What the corpus found"
    Building this comparison uncovered a defect in the propagator itself. The precomputed table of GCRF→ITRF rotations used by the force model was built from the IAU-76/FK5 approximation (`qgcrf2itrf_approx`), which neglects polar motion and is accurate to about 1 arcsecond. A 1″ tilt of the axis about which J2 makes the orbit precess is not small when integrated over a hundred revolutions: the ISS case drifted 50 m over 7 days relative to GMAT, and Molniya 117 m. The table is now the full IAU 2006/2000A chain — precession-nutation with EOP corrections and polar motion sampled hourly, the Earth rotation angle evaluated exactly — at the same cost as the old approximation. Those cases now agree to 3 cm and 13 cm.

## Regenerating the corpus

The corpus only needs regenerating when a case is added or the reference configuration changes; tightening a gate is a one-line edit. With a local GMAT install and NAIF's `de440.bsp`:

```bash
python tests/gmat/generate.py --gmat "~/Projects/GMAT R2026A" --spk ./de440.bsp
cargo test --test gmat_regression
```

All 17 cases regenerate in under a minute. The case matrix, gates and the measured floors live in `tests/gmat/cases.py`; operational details — adding a case, evaluating ad-hoc cases against a scratch directory, the JSON format — are in `tests/gmat/README.md` in the repository.

## See Also

- **Theory**: [Force Model](forces.md) for each term being compared; [ODE Integrators](integrators.md) for the replay settings.
- **Tutorial**: [GPS Example](../tutorials/GPS Example.ipynb) — validation of a different kind, fitting a GPS orbit against ESA SP3 truth.
- **Tutorial**: [SGP4 vs Numerical Propagation](../tutorials/SGP4 vs Numerical Propagation.ipynb).
