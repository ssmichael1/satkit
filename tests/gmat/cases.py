"""
Case matrix for the satkit-vs-GMAT regression corpus.

Each entry becomes one JSON file in ``tests/gmat/cases/`` (via
``generate.py``) and one ``#[test]``-equivalent iteration in
``tests/gmat_regression.rs``.

Orbits are given as Keplerian elements in EarthICRF (km, deg) or as a
Cartesian EarthICRF state (km, km/s).  Force models are named so that the
satkit side can be reconstructed from the JSON alone.

Tolerances are the *gate*: 3-5x the residual measured when the case was
generated, so that a real regression trips them but integrator noise and
known model differences do not.  Tightening a tolerance is a reviewed change.
"""

EPOCH_UTC = "2023-05-16T20:00:00"  # all cases share an epoch (EOP + SW data exist)
SAMPLE_SECONDS = 3600.0

# --- Body GMs (km^3/s^2).  DE440 values, identical to satkit's src/consts.rs.
# GMAT's own defaults are the DE405 set (Luna 4902.8005821478, Earth
# 398600.4415, ...); we pin them so the comparison isolates *model* differences
# and so the reference values are visible in the JSON for review.  A wrong
# constant in satkit still shows up as a residual against these.
MU_EARTH_KM3 = 398600.4418  # only used by GMAT for point-mass/GR; the harmonic field's mu (398600.4415) comes from the coefficient file on both sides
MU_MOON_KM3 = 4902.800118
MU_SUN_KM3 = 132712440041.27942

# --- Force models -------------------------------------------------------------
# tides: "None" | "SolidStep1"    (GMAT: 'None' | 'Solid')
FORCE_MODELS = {
    # Low-degree field + Sun/Moon: isolates mu, ephemeris, frame, time.
    "j2": dict(gravity_model="EGM96", gravity_degree=2, gravity_order=2,
               sun=True, moon=True, tides="None", relativity=False),
    # Everything satkit and GMAT model identically (GR off).
    "full": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
                 sun=True, moon=True, tides="SolidStep1", relativity=False),
    # As "full" with GR on.  satkit implements Schwarzschild only; GMAT adds
    # geodesic precession + Lense-Thirring (MathSpec eq. 4.109).  The residual
    # is a known floor (~1 m at 200,000 km) -- tighten when satkit adds those.
    "gr": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
               sun=True, moon=True, tides="SolidStep1", relativity=True),
}

# --- Orbits -------------------------------------------------------------------
ORBITS = {
    "leo_iss":   dict(kep=(6778.0, 0.001, 51.6, 30.0, 40.0, 50.0), days=7),
    "sso_800":   dict(kep=(7178.0, 0.0012, 98.6, 120.0, 90.0, 0.0), days=7),
    "meo_gps":   dict(kep=(26560.0, 0.01, 55.0, 200.0, 30.0, 100.0), days=7),
    "molniya":   dict(kep=(26600.0, 0.74, 63.4, 250.0, 270.0, 0.0), days=7),
    "geo":       dict(kep=(42164.0, 0.0003, 0.1, 0.0, 0.0, 45.0), days=7),
    # TESS: 2:1 lunar-resonant HEO -- the case that exposed the MU_MOON error.
    "tess":      dict(cart=(-130142.3736645361, 120626.1878655891, 126520.4971633258,
                            -0.308744045658656, -1.40299418814228, -0.0616245044905574416),
                      days=7),
    "cislunar":  dict(kep=(300000.0, 0.0, 20.0, 60.0, 0.0, 180.0), days=7),
}

# --- Cases: (orbit, force model, tolerance) -----------------------------------
# tolerance = dict(pos_m=..., vel_mps=...), gate on max over all samples.
#
# Gates are ~3x the residual measured at generation (satkit 0.20.4 + exact
# frame table, GMAT R2026A).  Measured floors and their known causes:
#
#   * Earth-only / j2 cases: 2-3 cm at LEO, 8 cm MEO, 13 cm GEO, 0.6-1.0 m at
#     200,000-300,000 km.  This is *GMAT's* integration error: GMAT's own
#     point-mass runs deviate from the analytic Kepler solution by exactly
#     these amounts (satkit matches Kepler to < 1 cm), independent of GMAT's
#     Accuracy, MaxStep, or integrator choice.
#   * "full" (tides on): +0.4-0.7 m at LEO, ~2 m at Molniya perigee over 7
#     days.  satkit's SolidStep1 uses the IERS 2010 Table 6.3 *anelastic*
#     Love numbers, including their imaginary (phase-lag) parts; GMAT's
#     'Solid' uses real Love numbers only.  The lag is a secular along-track
#     effect.  Zeroing satkit's imaginary parts reproduces GMAT to the j2
#     floor (verified), so this is GMAT omitting a term, not satkit.
#   * "gr": +0.2 m LEO, +1 m at 200,000 km: GMAT's geodesic + Lense-Thirring
#     terms, which satkit does not model.
def _c(orbit, fm, pos_m, vel_mps):
    return dict(name=f"{orbit}_{fm}", orbit=orbit, force_model=fm,
                tolerance=dict(pos_m=pos_m, vel_mps=vel_mps))

CASES = [
    _c("leo_iss",  "j2",   0.10, 1e-4),   # measured 0.028 m / 3.2e-5
    _c("leo_iss",  "full", 1.50, 2e-3),   # measured 0.50 m / 5.7e-4 (tides)
    _c("leo_iss",  "gr",   2.00, 2.5e-3), # measured 0.66 m / 7.4e-4 (tides + GR)
    _c("sso_800",  "j2",   0.10, 1e-4),   # measured 0.021 m / 2.2e-5
    _c("sso_800",  "full", 1.50, 1.5e-3), # measured 0.43 m / 4.4e-4 (tides)
    _c("meo_gps",  "j2",   0.30, 5e-5),   # measured 0.081 m / 1.2e-5
    _c("meo_gps",  "full", 0.30, 5e-5),   # measured 0.067 m / 9.7e-6
    _c("molniya",  "j2",   0.50, 2e-4),   # measured 0.13 m / 4.9e-5
    _c("molniya",  "full", 6.00, 5e-3),   # measured 1.97 m / 1.6e-3 (tides at perigee)
    _c("geo",      "j2",   0.50, 4e-5),   # measured 0.13 m / 9.4e-6
    _c("geo",      "full", 0.50, 4e-5),   # measured 0.13 m / 9.6e-6
    _c("tess",     "j2",   3.00, 1e-5),   # measured 1.01 m / 2.9e-6 (GMAT floor)
    _c("tess",     "full", 3.00, 1e-5),   # measured 1.01 m / 2.9e-6
    _c("tess",     "gr",   6.00, 2e-5),   # measured 2.08 m / 6.2e-6 (GR model)
    _c("cislunar", "j2",   2.00, 1e-5),   # measured 0.63 m / 3.3e-6 (GMAT floor)
    _c("cislunar", "full", 2.00, 1e-5),   # measured 0.63 m / 3.3e-6
    _c("cislunar", "gr",   6.00, 3e-5),   # measured 1.85 m / 8.0e-6 (GR model)
]
