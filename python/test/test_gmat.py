"""
Regression tests against NASA GMAT reference trajectories.

Python mirror of ``tests/gmat_regression.rs``: replays the committed corpus in
``tests/gmat/cases/`` through the Python ``propagate`` API. See
``tests/gmat/README.md`` for how the corpus is generated and what each
case isolates.
"""

import json
from datetime import datetime
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


def _settings(fm: dict) -> "sk.propsettings":
    s = sk.propsettings()
    s.gravity_model = GRAVITY[fm["gravity_model"]]
    s.gravity_degree = fm["gravity_degree"]
    s.gravity_order = fm["gravity_order"]
    s.use_sun_gravity = fm["sun"]
    s.use_moon_gravity = fm["moon"]
    s.tide_model = TIDES[fm["tides"]]
    s.use_relativistic_correction = fm["relativity"]
    s.use_spaceweather = False
    s.integrator = sk.integrator.rkv98_nointerp
    s.abs_error = 1e-13
    s.rel_error = 1e-13
    s.enable_interp = False
    return s


def _epoch(iso: str) -> "sk.time":
    d = datetime.fromisoformat(iso)
    return sk.time(d.year, d.month, d.day, d.hour, d.minute, d.second + d.microsecond * 1e-6)


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_gmat_case(path: Path):
    case = json.loads(path.read_text())
    assert case["name"] == path.stem
    samples = np.asarray(case["samples"], dtype=float)
    assert samples.shape[0] >= 2 and samples.shape[1] == 7

    epoch = _epoch(case["epoch_utc"])
    settings = _settings(case["force_model"])
    tol = case["tolerance"]

    # km, km/s -> m, m/s; re-propagate segment by segment from our own state
    state = samples[0, 1:] * 1e3
    t_prev = epoch + sk.duration.from_seconds(samples[0, 0])
    rows = []
    for sample in samples[1:]:
        t = epoch + sk.duration.from_seconds(sample[0])
        state = sk.propagate(state, t_prev, end=t, propsettings=settings).state
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
    assert len(CASES) >= 17, f"expected the GMAT corpus in {CASE_DIR}"
