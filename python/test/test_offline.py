"""
What satkit can do with no data directory and no network.

CI runs this file with ``SATKIT_DATA`` pointing at an empty directory and
``SATKIT_OFFLINE=1``; in a normal developer environment the same tests run
against the on-disk files. Everything here must pass either way — it is the
contract for the compiled-in core data (IERS nutation tables, gravity models
to degree 70).

Not offline-capable, by design: the JPL ephemeris (downloaded on first use)
and the Earth-orientation / space-weather files (refreshed from CelesTrak).
``test_missing_ephemeris_is_typed_error`` checks that asking for the
ephemeris in that state is a clean ``RuntimeError``, not a hang or a crash.
"""

import math
import os

import numpy as np
import pytest

import satkit as sk


def _offline_env() -> bool:
    v = os.environ.get("SATKIT_OFFLINE", "")
    return v not in ("", "0", "false", "False")


def test_gravity_acceleration_all_models():
    pos = np.array([7000.0e3, 0.0, 0.0])
    ref = None
    for model in (sk.gravmodel.egm96, sk.gravmodel.jgm3, sk.gravmodel.jgm2, sk.gravmodel.itugrace16):
        a = sk.gravity(pos, model=model, degree=20)
        mag = np.linalg.norm(a)
        assert abs(mag - 8.135) < 0.02, (model, mag)  # ~GM/r^2
        assert a[0] < 0.0
        ref = ref or mag
        assert abs(mag - ref) < 0.01


def test_iers_tables_and_precession_nutation():
    t = sk.time(2024, 6, 1, 0, 0, 0)
    q = sk.frametransform.rotation(sk.frame.CIRS, sk.frame.GCRF, t)
    # The CIP has moved ~480" ≈ 0.13° since J2000 (IERS X/Y series).
    angle = math.degrees(q.angle)
    angle = min(angle, 360.0 - angle)
    assert 0.10 < angle < 0.20, angle


def test_time_scales():
    t = sk.time(2024, 1, 1, 0, 0, 0)
    tai = t.as_mjd(sk.timescale.TAI)
    utc = t.as_mjd(sk.timescale.UTC)
    assert abs((tai - utc) * 86400.0 - 37.0) < 1e-6
    tt = t.as_mjd(sk.timescale.TT)
    assert abs((tt - tai) * 86400.0 - 32.184) < 1e-6


def test_sgp4():
    tle = sk.TLE.from_lines(
        [
            "ISS (ZARYA)",
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        ]
    )
    tle = tle[0] if isinstance(tle, list) else tle
    p, v = sk.sgp4(tle, tle.epoch + sk.duration.from_hours(1.0))
    assert abs(np.linalg.norm(p) - 6.78e6) < 5e4
    assert abs(np.linalg.norm(v) - 7.66e3) < 1e2


def test_kepler_and_lambert():
    k = sk.kepler(7000.0e3, 0.001, 0.9, 0.1, 0.2, 0.3)
    r0, v0 = k.to_pv()
    r1, _v1 = k.propagate(sk.duration.from_seconds(600.0)).to_pv()
    sols = sk.lambert(np.asarray(r0), np.asarray(r1), 600.0)
    v1 = sols[0][0]
    assert np.linalg.norm(np.asarray(v1) - np.asarray(v0)) < 0.5


def test_data_dirs_api():
    dirs = sk.utils.data_search_dirs()
    assert isinstance(dirs, list) and len(dirs) >= 1
    if os.environ.get("SATKIT_DATA"):
        assert dirs[0] == os.environ["SATKIT_DATA"]
        assert sk.utils.datadir() == os.environ["SATKIT_DATA"]
    # Never a write location inside the package or site-packages.
    d = sk.utils.datadir()
    assert d is None or "site-packages" not in d or os.environ.get("SATKIT_DATA", "") == d


def test_set_offline_round_trip():
    """The programmatic setter overrides the environment in both directions."""
    before = sk.utils.is_offline()
    try:
        sk.utils.set_offline(True)
        assert sk.utils.is_offline() is True
        sk.utils.set_offline(False)
        assert sk.utils.is_offline() is False
    finally:
        # Restore whatever the environment implied for the rest of the suite.
        sk.utils.set_offline(before)


@pytest.mark.skipif(
    not _offline_env() or sk.utils.datafiles_exist(),
    reason="only meaningful with an empty data directory and SATKIT_OFFLINE=1",
)
def test_missing_ephemeris_is_typed_error():
    t = sk.time(2024, 1, 1, 0, 0, 0)
    with pytest.raises(RuntimeError) as ei:
        sk.jplephem.geocentric_pos(sk.solarsystem.Moon, t)
    msg = str(ei.value)
    assert "SATKIT_OFFLINE" in msg
    assert "linux_p1550p2650.440" in msg
    assert "https://" in msg
