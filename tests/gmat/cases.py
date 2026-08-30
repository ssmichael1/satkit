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

EPOCH_UTC = "2023-05-16T20:00:00"  # default epoch (EOP + SW data exist)
# Drag cases: well inside the *observed* block of the CelesTrak space-weather
# file on both sides (GMAT's SW-All.txt, satkit's SW-All.csv), so neither tool
# uses predicted indices.  2023-02-27 was a G2 storm (Ap 91); the arc starts
# two days after it, so the 3-hourly ap history still matters on day 1.
EPOCH_UTC_DRAG = "2023-03-01T00:00:00"
SAMPLE_SECONDS = 3600.0
GMAT_ACCURACY = 1e-14  # RK89 relative accuracy; a case may override with gmat_accuracy=

# Spacecraft ballistic properties for the drag cases (GMAT Cd / DragArea /
# DryMass; satkit uses the product Cd*A/m = 0.022 m^2/kg).
SPACECRAFT = dict(cd=2.2, drag_area_m2=10.0, dry_mass_kg=1000.0)

# CelesTrak space-weather source used by the file-driven drag cases.  GMAT
# reads the fixed-width .txt, satkit the .csv; both are the same data set.
SW_TXT_URL = "https://celestrak.org/SpaceData/SW-All.txt"

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
# drag:  None | dict(atmosphere="NRLMSISE00", weather="constant", f107=, f107a=, ap=, gmat_kp=)
#             | dict(atmosphere="NRLMSISE00", weather="CSSISpaceWeatherFile")
#
# Constant weather uses F10.7 = F10.7A = 150, Ap = 4: exactly what satkit's
# NRLMSISE-00 assumes with `use_spaceweather = False`, so no new API is
# needed to isolate the density model from the space-weather feed.  GMAT's
# Drag.MagneticIndex is *Kp*, converted with its Kp->Ap table lookup
# (AtmosphereModel::ConvertKpToAp: index = int((kp + 0.01) * 3)); Kp = 1 maps
# to Ap = 4 exactly.  Ap = 4 also makes NRLMSISE-00's daily-Ap and 3-hourly
# ap-array formulations coincide (both geomagnetic terms vanish at Ap = 4),
# so the choice of formulation cannot leak into the constant cases.
DRAG_CONST = dict(atmosphere="NRLMSISE00", weather="constant",
                  f107=150.0, f107a=150.0, ap=4.0, gmat_kp=1.0)
DRAG_FILE = dict(atmosphere="NRLMSISE00", weather="CSSISpaceWeatherFile")

FORCE_MODELS = {
    # Low-degree field + Sun/Moon: isolates mu, ephemeris, frame, time.
    "j2": dict(gravity_model="EGM96", gravity_degree=2, gravity_order=2,
               sun=True, moon=True, tides="None", relativity=False, drag=None),
    # Everything satkit and GMAT model identically (GR off).
    "full": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
                 sun=True, moon=True, tides="SolidStep1", relativity=False, drag=None),
    # As "full" with GR on.  Both tools apply IERS 2010 eq. 10.12 in full
    # (Schwarzschild + geodesic precession + Lense-Thirring; GMAT MathSpec
    # Table 4.1), so the gr residual sits at the full/j2 floor.
    "gr": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
               sun=True, moon=True, tides="SolidStep1", relativity=True, drag=None),
    # "full" + NRLMSISE-00 drag with constant space weather: tests the
    # density model implementation and the drag force alone.
    "drag_const": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
                       sun=True, moon=True, tides="SolidStep1", relativity=False,
                       drag=DRAG_CONST),
    # "full" + NRLMSISE-00 drag driven by the CelesTrak space-weather file on
    # both sides: tests the whole chain, including each tool's F10.7 / Ap
    # feed conventions (see the README floors).
    "drag_sw": dict(gravity_model="EGM96", gravity_degree=36, gravity_order=36,
                    sun=True, moon=True, tides="SolidStep1", relativity=False,
                    drag=DRAG_FILE),
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
    # Drag orbits: 3-day arcs (drag error grows ~t^2; 7 days at 300 km would
    # be dominated by it), own epoch, and a spacecraft block.
    "iss_420":   dict(kep=(6798.0, 0.0005, 51.6, 30.0, 40.0, 50.0), days=3,
                      epoch=EPOCH_UTC_DRAG, spacecraft=SPACECRAFT),
    "leo_300":   dict(kep=(6678.0, 0.0005, 45.0, 120.0, 90.0, 0.0), days=3,
                      epoch=EPOCH_UTC_DRAG, spacecraft=SPACECRAFT),
    "sso_550":   dict(kep=(6928.0, 0.001, 97.6, 200.0, 90.0, 0.0), days=3,
                      epoch=EPOCH_UTC_DRAG, spacecraft=SPACECRAFT),
    # GTO: 250 km perigee x 35,786 km apogee -- drag is a perigee impulse.
    "gto_250":   dict(kep=(24396.0, 0.72832, 27.0, 60.0, 180.0, 0.0), days=3,
                      epoch=EPOCH_UTC_DRAG, spacecraft=SPACECRAFT),
}

# --- Cases: (orbit, force model, tolerance) -----------------------------------
# tolerance = dict(pos_m=..., vel_mps=...), gate on max over all samples.
#
# Gates are ~3x the residual measured at generation (satkit 0.20.4 + exact
# frame table, GMAT R2026A; the drag_*_sw gates were re-measured after satkit
# adopted the 3-hourly ap history).  Measured floors and their known causes:
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
#   * "gr": no additional floor.  Both sides model Schwarzschild + geodesic
#     precession + Lense-Thirring (IERS 2010 eq. 10.12); the gr residuals
#     equal the corresponding full (LEO) or j2 (high orbit) residuals.
def _c(orbit, fm, pos_m, vel_mps, name=None):
    return dict(name=name or f"{orbit}_{fm}", orbit=orbit, force_model=fm,
                tolerance=dict(pos_m=pos_m, vel_mps=vel_mps))


def _d(orbit, weather, pos_m, vel_mps, **extra):
    """Drag case: ``drag_<orbit>_<const|sw>`` on the ``drag_<weather>`` model."""
    short = {"iss_420": "iss", "leo_300": "leo300", "sso_550": "sso550", "gto_250": "gto"}[orbit]
    return dict(_c(orbit, f"drag_{weather}", pos_m, vel_mps, name=f"drag_{short}_{weather}"), **extra)

CASES = [
    _c("leo_iss",  "j2",   0.10, 1e-4),   # measured 0.028 m / 3.2e-5
    _c("leo_iss",  "full", 1.50, 2e-3),   # measured 0.50 m / 5.7e-4 (tides)
    _c("leo_iss",  "gr",   1.50, 2e-3),   # measured 0.50 m / 5.8e-4 (tides)
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
    _c("tess",     "gr",   3.00, 1e-5),   # measured 1.01 m / 2.9e-6 (GMAT floor)
    _c("cislunar", "j2",   2.00, 1e-5),   # measured 0.63 m / 3.3e-6 (GMAT floor)
    _c("cislunar", "full", 2.00, 1e-5),   # measured 0.63 m / 3.3e-6
    _c("cislunar", "gr",   2.00, 1e-5),   # measured 0.63 m / 3.3e-6 (GMAT floor)
    # Drag (3 days).  Gates ~3x the measured residual.  The drag-only
    # displacement (satkit with drag minus satkit without, 3 days) is what each
    # residual should be read against; see the README floors.
    #                                      measured max |dr| / |dv|   drag-only   ratio
    _d("iss_420", "const",  80.0, 0.10),   # 25.8 m   / 2.9e-2       152 km      1.7e-4
    _d("iss_420", "sw",    1.0e3, 1.1),    # 325 m    / 0.37         201 km      1.6e-3 (F10.7 timing; final 198 m)
    _d("leo_300", "const", 900.0, 1.0),    # 293 m    / 0.34         1374 km     2.1e-4
    # GMAT's RK89 cannot hold 1e-14 through the file-driven weather steps at
    # 300 km ("Accuracy settings will be violated"), so this case runs at 1e-13.
    _d("leo_300", "sw",    7.0e3, 8.0, gmat_accuracy=1e-13),  # 2.28 km / 2.64  1697 km  1.3e-3 (F10.7 timing; final 2.07 km)
    _d("sso_550", "const", 100.0, 0.12),   # 34.2 m   / 3.7e-2       19.2 km     1.8e-3 (LST, see README)
    _d("sso_550", "sw",    110.0, 0.12),   # 37.0 m   / 4.0e-2       28.0 km     1.3e-3 (F10.7 timing; final 18 m)
    _d("gto_250", "const",  30.0, 2.5e-2), # 9.9 m    / 7.5e-3       106 km      5.5e-5 (final 5.9 m)
    _d("gto_250", "sw",    1.0e3, 0.7),    # 307 m    / 0.23         132 km      2.3e-3 (F10.7 timing; final 113 m)
]
