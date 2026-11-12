"""API review round 2, submodules and shared plumbing: the ``satkit.eop``
module, ``TimeScalar`` for the frametransform EOP queries, ``to_gcrf`` /
``from_gcrf`` for ICRF and EME2000, ``FileNotFoundError`` from the
``init_from_path`` loaders, ``jplephem.constants`` and the ``tm`` -> ``time``
parameter rename."""

import ast
import datetime
import inspect
import pathlib
import subprocess
import sys
import warnings

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


# ───────────────────────── tm -> time (deprecated alias) ─────────────────────────

ft = sk.frametransform
_P = np.array([7.0e6, 0.0, 0.0])
_V = np.array([0.0, 7.5e3, 0.0])
_FR = (sk.frame.ITRF, sk.frame.GCRF)

# name -> (function, positional arguments before the time, keyword arguments after it)
RENAMED = {
    **{
        f"frametransform.{n}": (getattr(ft, n), (), {})
        for n in (
            "gmst", "eqeq", "gast", "earth_rotation_angle", "qitrf2tirs", "qtirs2cirs",
            "qcirs2gcrf", "qitrf2gcrf", "qgcrf2itrf", "qgcrf2itrf_approx",
            "qitrf2gcrf_approx", "qteme2itrf", "qteme2gcrf", "qmod2gcrf",
            "qtod2mod_approx", "eop_status",
        )
    },
    "frametransform.rotation": (ft.rotation, _FR, {}),
    "frametransform.rotation_approx": (ft.rotation_approx, _FR, {}),
    **{
        f"frametransform.{n}": (getattr(ft, n), _FR, {"pos": _P, "vel": _V})
        for n in ("rotation_with_state", "transform_state", "transform_state_approx")
    },
    **{
        f"jplephem.{n}": (getattr(sk.jplephem, n), (sk.solarsystem.Sun,), {})
        for n in ("geocentric_pos", "geocentric_state", "barycentric_pos", "barycentric_state")
    },
}


def _same(a, b):
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, sk.quaternion):
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(a * v, b * v)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("name", sorted(RENAMED))
class TestTmRenamedToTime:
    def test_signature(self, name):
        f, _, _ = RENAMED[name]
        params = inspect.signature(f).parameters
        assert "time" in params
        assert params["tm"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_positional_and_time_keyword_do_not_warn(self, name):
        f, pre, post = RENAMED[name]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a = f(*pre, T, **post)
            b = f(*pre, time=T, **post)
        _same(a, b)

    def test_tm_keyword_warns_and_works(self, name):
        f, pre, post = RENAMED[name]
        with pytest.warns(DeprecationWarning, match="'tm' is deprecated; use 'time'"):
            c = f(*pre, tm=T, **post)
        _same(c, f(*pre, T, **post))

    def test_both_is_type_error(self, name):
        f, pre, post = RENAMED[name]
        with pytest.raises(TypeError, match="both 'time' and its deprecated alias 'tm'"):
            f(*pre, time=T, tm=T, **post)
        with pytest.raises(TypeError, match="both"):
            f(*pre, T, tm=T, **post)

    def test_missing_is_type_error(self, name):
        f, pre, post = RENAMED[name]
        with pytest.raises(TypeError, match="missing required argument: 'time'"):
            f(*pre, **post)

    def test_docstring_names_time(self, name):
        f, _, _ = RENAMED[name]
        assert "tm (" not in f.__doc__
        assert "time (" in f.__doc__


@pytest.mark.parametrize("name", ["rotation_with_state", "transform_state", "transform_state_approx"])
def test_state_functions_still_require_pos_vel(name):
    f = getattr(ft, name)
    with pytest.raises(TypeError, match="missing required argument: 'pos'"):
        f(*_FR, T)
    with pytest.raises(TypeError, match="missing required argument: 'vel'"):
        f(*_FR, T, _P)


def test_time_list_by_keyword():
    ts = [T, T + sk.duration.from_hours(1)]
    np.testing.assert_array_equal(ft.gmst(time=ts), ft.gmst(ts))
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(ft.gmst(tm=ts), ft.gmst(ts))


@pytest.mark.parametrize("stub, expected", [("frametransform.pyi", 21), ("jplephem.pyi", 4)])
def test_stub_tm_only_in_deprecated_overloads(stub, expected):
    path = pathlib.Path(sk.__file__).parent / stub
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seen = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        a = node.args
        names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        if "tm" not in names:
            continue
        seen += 1
        assert any(
            isinstance(d, ast.Call) and getattr(d.func, "id", "") == "deprecated"
            for d in node.decorator_list
        ), f"{stub}: {node.name} has a non-deprecated 'tm'"
        assert a.kwonlyargs[0].arg == "tm" and "time" not in names
    assert seen == expected
