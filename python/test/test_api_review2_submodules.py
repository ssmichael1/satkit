"""API review round 2, submodules and shared plumbing: the ``satkit.eop``
module, ``TimeScalar`` for the frametransform EOP queries, ``to_gcrf`` /
``from_gcrf`` for ICRF and EME2000, ``FileNotFoundError`` from the
``init_from_path`` loaders, ``jplephem.constants`` and the ``tm`` -> ``time``
parameter rename."""

import datetime
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import satkit as sk

T = sk.time(2024, 1, 1)


# ──────────────────────────────── satkit.eop ────────────────────────────────


class TestEopModule:
    def test_importable(self):
        import satkit.eop

        assert satkit.eop is sk.eop
        assert sys.modules["satkit.eop"] is sk.eop
        assert sk.eop.__name__ == "satkit.eop"
        assert sk.eop.get.__module__ == "satkit.eop"
        assert sk.eop.__doc__

    def test_get_matches_frametransform(self):
        v = sk.eop.get(T)
        assert isinstance(v, tuple) and len(v) == 6
        assert v == sk.frametransform.earth_orientation_params(T)
        assert sk.eop.get(sk.time(1960, 1, 1)) is None

    def test_coverage_matches_frametransform(self):
        cov = sk.eop.coverage()
        assert cov == sk.frametransform.eop_coverage()
        first, last_observed, last = cov
        assert first == sk.time(1973, 1, 2)
        assert first < last_observed <= last

    def test_status(self):
        _, _, last = sk.eop.coverage()
        assert sk.eop.status(T) == "observed"
        assert sk.eop.status(sk.time(1960, 1, 1)) == "before_table"
        assert sk.eop.status(last + sk.duration.from_days(10)) == "extrapolated"
        assert sk.eop.status(T) == sk.frametransform.eop_status(T)

    @pytest.mark.parametrize(
        "t",
        [
            T,
            datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
            np.datetime64("2024-01-01T00:00:00"),
        ],
    )
    def test_time_scalar_inputs(self, t):
        assert sk.eop.get(t) == sk.eop.get(T)
        assert sk.eop.status(t) == "observed"

    @pytest.mark.parametrize("bad", [[T], 5, "2024-01-01"])
    def test_not_a_single_time(self, bad):
        with pytest.raises(TypeError):
            sk.eop.get(bad)
        with pytest.raises(TypeError):
            sk.eop.status(bad)

    def test_init_from_path_missing_file(self, tmp_path):
        before = sk.eop.coverage()
        missing = tmp_path / "no_such_finals2000A.all"
        with pytest.raises(FileNotFoundError) as e:
            sk.eop.init_from_path(missing)
        assert e.value.filename == str(missing)
        assert "no_such_finals2000A.all" in str(e.value)
        assert sk.eop.coverage() == before  # table unchanged

    def test_init_from_bytes_invalid_leaves_table(self):
        before = sk.eop.coverage()
        with pytest.raises(RuntimeError):
            sk.eop.init_from_bytes(b"DATE,MJD,X,Y\n")  # CelesTrak CSV is rejected
        with pytest.raises(RuntimeError):
            sk.eop.init_from_bytes(b"\xff\xfe not utf-8")
        assert sk.eop.coverage() == before

    def test_init_from_path_and_bytes_replace_the_table(self, tmp_path):
        """A valid load replaces the process-wide table, so it runs in a child
        process: nothing here can leave a one-row table installed."""
        code = r"""
import sys
import satkit as sk
from pathlib import Path

src = None
for d in sk.utils.data_search_dirs():
    f = Path(d) / "finals2000A.all"
    if f.is_file():
        src = f
        break
if src is None:
    print("SKIP")
    sys.exit(0)
lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln[7:15].strip() in ("60310.00", "60311.00", "60312.00")]
assert len(lines) == 3, lines
part = Path(sys.argv[1]) / "part.all"
part.write_text("\n".join(lines) + "\n", encoding="utf-8")
full = sk.eop.get(sk.time(2024, 1, 2))

sk.eop.init_from_path(str(part))
first, last_obs, last = sk.eop.coverage()
assert first == sk.time(2024, 1, 1) and last == sk.time(2024, 1, 3), (first, last)
assert sk.eop.get(sk.time(2024, 1, 2)) == full
assert sk.eop.status(sk.time(2023, 6, 1)) == "before_table"
assert sk.frametransform.eop_status(sk.time(2023, 6, 1)) == "before_table"

sk.eop.init_from_bytes(src.read_bytes())
assert sk.eop.coverage()[0] == sk.time(1973, 1, 2)
print("OK")
"""
        r = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr
        if r.stdout.strip() == "SKIP":
            pytest.skip("no finals2000A.all in the data search directories")
        assert r.stdout.strip().endswith("OK"), r.stdout

    def test_update_is_documented_not_run(self):
        # never called here: it downloads
        assert callable(sk.eop.update)
        assert "Refresh" in sk.eop.update.__doc__

    def test_members(self):
        public = sorted(n for n in dir(sk.eop) if not n.startswith("_"))
        assert public == [
            "coverage",
            "get",
            "init_from_bytes",
            "init_from_path",
            "status",
            "update",
        ]

    def test_frametransform_docstrings_point_to_eop(self):
        for f in (
            sk.frametransform.earth_orientation_params,
            sk.frametransform.eop_coverage,
            sk.frametransform.eop_status,
            sk.frametransform.disable_eop_time_warning,
        ):
            assert "satkit.eop" in f.__doc__, f.__name__


# ────────────────── frametransform EOP queries take TimeScalar ──────────────────


class TestEopTimeScalar:
    @pytest.mark.parametrize(
        "t",
        [
            datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
            np.datetime64("2024-01-01T00:00:00"),
        ],
    )
    def test_datetime_and_datetime64(self, t):
        assert sk.frametransform.eop_status(t) == "observed"
        assert sk.frametransform.earth_orientation_params(
            t
        ) == sk.frametransform.earth_orientation_params(T)

    def test_list_still_type_error(self):
        with pytest.raises(TypeError):
            sk.frametransform.eop_status([T])
        with pytest.raises(TypeError):
            sk.frametransform.earth_orientation_params([T])


# ─────────────────────── to_gcrf / from_gcrf: ICRF, EME2000 ───────────────────────

POS = np.array([7.0e6, 1.0e5, 2.0e5])
VEL = np.array([-50.0, 7.5e3, 100.0])


class TestToGcrfInertial:
    @pytest.mark.parametrize("frame", [sk.frame.EME2000, sk.frame.ICRF])
    def test_matches_rotation(self, frame):
        dcm = sk.frametransform.to_gcrf(frame, POS, VEL)
        q = sk.frametransform.rotation(frame, sk.frame.GCRF, T)
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(dcm @ v, q * v, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(sk.frametransform.from_gcrf(frame, POS, VEL), dcm.T)

    def test_icrf_identity_eme2000_bias(self):
        np.testing.assert_array_equal(sk.frametransform.to_gcrf(sk.frame.ICRF, POS, VEL), np.eye(3))
        b = sk.frametransform.to_gcrf(sk.frame.EME2000, POS, VEL)
        # the ~23 mas frame bias, constant: the state does not matter
        angle = np.arccos((np.trace(b) - 1) / 2)
        assert 0.5e-7 < angle < 1.5e-7
        np.testing.assert_array_equal(
            sk.frametransform.to_gcrf(sk.frame.EME2000, np.zeros(3), np.zeros(3)), b
        )

    @pytest.mark.parametrize("frame", [sk.frame.ITRF, sk.frame.TIRS, sk.frame.CIRS, sk.frame.TEME])
    def test_time_dependent_frames_still_raise(self, frame):
        for func in ("to_gcrf", "from_gcrf"):
            with pytest.raises(RuntimeError, match="time-dependent"):
                getattr(sk.frametransform, func)(frame, POS, VEL)


# ───────────────────── spaceweather.init_from_path(missing) ─────────────────────


def test_spaceweather_init_from_path_missing(tmp_path):
    missing = tmp_path / "no_such_Kp_file.txt"
    before = sk.spaceweather.coverage()
    with pytest.raises(FileNotFoundError) as e:
        sk.spaceweather.init_from_path(missing)
    assert e.value.filename == str(missing)
    assert "no_such_Kp_file.txt" in str(e.value)
    assert sk.spaceweather.coverage() == before


def test_spaceweather_init_from_path_other_errors_unchanged(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe not utf-8")
    before = sk.spaceweather.coverage()
    with pytest.raises(RuntimeError):
        sk.spaceweather.init_from_path(bad)
    assert sk.spaceweather.coverage() == before


# ──────────────────────────── jplephem.constants ────────────────────────────


def test_jplephem_constants():
    c = sk.jplephem.constants()
    assert isinstance(c, dict)
    assert c["EMRAT"] == sk.jplephem.consts("EMRAT")
    assert c["AU"] == sk.jplephem.consts("AU")
    for name, value in c.items():
        assert isinstance(name, str) and isinstance(value, float)
        assert sk.jplephem.consts(name) == value
    with pytest.raises(KeyError):
        c["NOT_A_CONSTANT"]
    # a fresh copy each time
    c["EMRAT"] = 0.0
    assert sk.jplephem.constants()["EMRAT"] == sk.jplephem.consts("EMRAT")


def test_stub_files_exist():
    stub_dir = pathlib.Path(sk.__file__).parent
    assert (stub_dir / "eop.pyi").is_file()
