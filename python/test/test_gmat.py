"""
Regression tests against NASA GMAT reference trajectories.

Python mirror of ``tests/gmat_regression.rs``: replays the committed corpus in
``tests/gmat/cases/`` through the Python ``propagate`` API. See
``tests/gmat/README.md`` for how the corpus is generated and what each
case isolates.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import satkit as sk

CASE_DIR = Path(__file__).resolve().parents[2] / "tests" / "gmat" / "cases"
CASES = sorted(CASE_DIR.glob("*.json"))

GRAVITY = {
    "EGM96": sk.gravmodel.egm96,
    "JGM3": sk.gravmodel.jgm3,
    "JGM2": sk.gravmodel.jgm2,
    "ITUGrace16": sk.gravmodel.itugrace16,
}
TIDES = {
    "None": sk.tidemodel.none,
    "SolidStep1": sk.tidemodel.solid_step1,
    "SolidFull": sk.tidemodel.solid_full,
}


def _use_spaceweather(drag) -> bool:
    """satkit's fixed indices (``use_spaceweather=False``) are F10.7 = F10.7A =
    150, Ap = 4 -- what the ``constant`` cases were generated with; the
    file-driven cases read SW-All.csv."""
    if drag is None:
        return False
    assert drag["atmosphere"] == "NRLMSISE00"
    if drag["weather"] == "constant":
        assert (drag["f107"], drag["f107a"], drag["ap"]) == (150.0, 150.0, 4.0)
        return False
    assert drag["weather"] == "CSSISpaceWeatherFile", drag["weather"]
    return True


def _satproperties(case: dict):
    """``Cd * A / m`` from the GMAT spacecraft block, for drag cases only."""
    if case["force_model"].get("drag") is None:
        return None
    sc = case["orbit"]["spacecraft"]
    return sk.satproperties(cdaoverm=sc["cd"] * sc["drag_area_m2"] / sc["dry_mass_kg"])


def _settings(fm: dict) -> "sk.propsettings":
    # Mirrors tests/gmat_regression.rs: tolerances 10x tighter than the
    # tightest gate, no dense output, space weather only for the
    # file-driven drag cases.
    assert fm["gravity_degree"] <= 40 and fm["gravity_order"] <= fm["gravity_degree"]
    s = sk.propsettings()
    s.gravity_model = GRAVITY[fm["gravity_model"]]
    s.gravity_degree = fm["gravity_degree"]
    s.gravity_order = fm["gravity_order"]
    s.use_sun_gravity = fm["sun"]
    s.use_moon_gravity = fm["moon"]
    s.tide_model = TIDES[fm["tides"]]
    s.use_relativistic_correction = fm["relativity"]
    s.use_spaceweather = _use_spaceweather(fm.get("drag"))
    s.integrator = sk.integrator.rkv98_nointerp
    s.abs_error = 1e-13
    s.rel_error = 1e-13
    s.enable_interp = False
    return s


def _check_gms(case: dict) -> None:
    """The reference was generated with these GMs; they must be satkit's."""
    for body, gmat_km3, satkit_m3 in (
        ("Earth", case["gmat"]["mu_earth_km3s2"], sk.consts.mu_earth),
        ("Moon", case["gmat"]["mu_moon_km3s2"], sk.consts.mu_moon),
        ("Sun", case["gmat"]["mu_sun_km3s2"], sk.consts.mu_sun),
    ):
        rel = abs(gmat_km3 * 1e9 - satkit_m3) / satkit_m3
        assert rel < 1e-9, f"{case['name']}: {body} GM differs from satkit.consts by {rel:.2e}"


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_gmat_case(path: Path):
    case = json.loads(path.read_text())
    assert case["name"] == path.stem
    samples = np.asarray(case["samples"], dtype=float)
    assert samples.shape[0] >= 2 and samples.shape[1] == 7
    assert samples[0, 0] == 0.0
    assert np.all(np.diff(samples[:, 0]) > 0), "elapsed times must be strictly increasing"
    _check_gms(case)

    epoch = sk.time.from_string(case["epoch_utc"])
    settings = _settings(case["force_model"])
    satprops = _satproperties(case)
    tol = case["tolerance"]
    assert tol["pos_m"] > 0 and tol["vel_mps"] > 0

    # km, km/s -> m, m/s; re-propagate segment by segment from our own state
    state = samples[0, 1:] * 1e3
    t_prev = epoch + sk.duration.from_seconds(samples[0, 0])
    rows = []
    for sample in samples[1:]:
        t = epoch + sk.duration.from_seconds(sample[0])
        state = sk.propagate(state, t_prev, end=t, propsettings=settings, satproperties=satprops).state
        t_prev = t
        truth = sample[1:] * 1e3
        rows.append((sample[0], np.linalg.norm(state[:3] - truth[:3]), np.linalg.norm(state[3:] - truth[3:])))

    res = np.array(rows)
    worst_pos, worst_vel = res[:, 1].max(), res[:, 2].max()
    if worst_pos > tol["pos_m"] or worst_vel > tol["vel_mps"]:
        table = "\n".join(f"  {r[0]:>9.0f}  {r[1]:>12.5f}  {r[2]:>12.4e}" for r in res)
        pytest.fail(
            f"{case['name']}: exceeds GMAT tolerance (pos {worst_pos:.4f} m > {tol['pos_m']} m "
            f"or vel {worst_vel:.3e} > {tol['vel_mps']:.1e} m/s)\n"
            f"residuals vs GMAT (elapsed s, |dr| m, |dv| m/s):\n{table}"
        )


def test_corpus_present():
    """The corpus must be checked in; an empty glob would silently skip everything."""
    assert len(CASES) >= 25, f"expected the GMAT corpus in {CASE_DIR}"
