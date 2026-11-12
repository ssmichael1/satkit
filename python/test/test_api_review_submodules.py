"""Regression tests for the Python API review: native submodules and the
shared argument helpers (time inputs, array inputs).

Sections follow the review worklist: behaviour bugs first, then additions,
then the changes that alter results or exception types.
"""

import datetime
import math
import warnings

import numpy as np
import pytest

import satkit as sk

T0 = sk.time(2024, 3, 1, 6, 0, 0)
POS = np.array([7.0e6, 0.0, 0.0])
VEL = np.array([0.0, 7.5e3, 0.0])
ORBIT_FRAMES = [sk.frame.LVLH, sk.frame.RTN, sk.frame.NTW]


# ───────────────────────── bug 8: barycentric Moon ─────────────────────────


class TestBarycentricMoon:
    """``jplephem.barycentric_pos/state(Moon)`` returned the geocentric Moon."""

    def test_moon_is_barycentric(self):
        moon = sk.solarsystem.Moon
        emrat = sk.jplephem.consts("EMRAT")
        assert emrat == pytest.approx(81.3, abs=0.01)

        emb, emb_v = sk.jplephem.barycentric_state(sk.solarsystem.EMB, T0)
        geo, geo_v = sk.jplephem.geocentric_state(moon, T0)
        bary, bary_v = sk.jplephem.barycentric_state(moon, T0)

        # ~1 au from the barycenter, not 384,000 km
        assert np.linalg.norm(bary) / sk.consts.au == pytest.approx(1.0, abs=0.03)
        assert 3.5e8 < np.linalg.norm(geo) < 4.1e8

        k = emrat / (1.0 + emrat)
        np.testing.assert_allclose(bary, emb + geo * k, rtol=0, atol=1e-3)
        np.testing.assert_allclose(bary_v, emb_v + geo_v * k, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(sk.jplephem.barycentric_pos(moon, T0), bary)

    def test_earth_is_consistent_across_bodies(self):
        """barycentric - geocentric is the Earth, whichever body it is taken from"""
        earth = [
            sk.jplephem.barycentric_pos(b, T0) - sk.jplephem.geocentric_pos(b, T0)
            for b in (sk.solarsystem.Moon, sk.solarsystem.Sun, sk.solarsystem.Mars)
        ]
        np.testing.assert_allclose(earth[0], earth[1], rtol=0, atol=1e-3)
        np.testing.assert_allclose(earth[0], earth[2], rtol=0, atol=1e-3)

    def test_array_input(self):
        times = [T0, T0 + sk.duration(days=3)]
        bary = sk.jplephem.barycentric_pos(sk.solarsystem.Moon, times)
        assert bary.shape == (2, 3)
        np.testing.assert_array_equal(bary[0], sk.jplephem.barycentric_pos(sk.solarsystem.Moon, T0))
        pos, vel = sk.jplephem.barycentric_state(sk.solarsystem.Moon, times)
        np.testing.assert_array_equal(pos, bary)
        assert vel.shape == (2, 3)

    def test_geocentric_moon_unchanged(self):
        """Value from 0.24.3, which this change must not move"""
        geo = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, sk.time(2024, 1, 1))
        np.testing.assert_allclose(
            geo, [-3.67980874e08, 1.42721023e08, 8.93144218e07], rtol=1e-8
        )


# ───────────────────────── bug 10: rise_set sigma ─────────────────────────


class TestRiseSetSigma:
    coord = sk.itrfcoord(latitude_deg=40.0, longitude_deg=-70.0)

    @pytest.mark.parametrize("sigma", [float("nan"), float("inf"), -float("inf"), 0.0, -5.0, 180.0, 200.0])
    def test_bad_sigma(self, sigma):
        with pytest.raises(ValueError, match="0 < sigma < 180"):
            sk.sun.rise_set(T0, self.coord, sigma)

    def test_good_sigma(self):
        std = sk.sun.rise_set(T0, self.coord)
        np.testing.assert_array_equal(
            [t.to_mjd() for t in sk.sun.rise_set(T0, self.coord, None)],
            [t.to_mjd() for t in std],
        )
        civil = sk.sun.rise_set(T0, self.coord, 96.0)
        assert civil[0] < std[0] < std[1] < civil[1]


# ─────────────────── bug 14: to_gcrf / from_gcrf degenerate ───────────────────


class TestOrbitFrameDegenerate:
    @pytest.mark.parametrize("frame", ORBIT_FRAMES)
    @pytest.mark.parametrize("func", ["to_gcrf", "from_gcrf"])
    @pytest.mark.parametrize(
        "pos, vel, what",
        [
            (np.zeros(3), np.zeros(3), "pos is zero"),
            (np.zeros(3), VEL, "pos is zero"),
            (POS, np.zeros(3), "vel is zero"),
            (POS, 2.0e-3 * POS, "parallel"),
            (POS, -POS, "parallel"),
            (POS, [np.nan, 0, 0], "finite"),
            ([np.inf, 0, 0], VEL, "finite"),
        ],
    )
    def test_raises_instead_of_nan(self, frame, func, pos, vel, what):
        with pytest.raises(ValueError, match=what) as e:
            getattr(sk.frametransform, func)(frame, pos, vel)
        assert str(e.value).startswith(func + ":")

    def test_gcrf_needs_no_orbit_plane(self):
        np.testing.assert_array_equal(
            sk.frametransform.to_gcrf(sk.frame.GCRF, np.zeros(3), np.zeros(3)), np.eye(3)
        )

    @pytest.mark.parametrize("frame", ORBIT_FRAMES)
    def test_valid_state_still_a_rotation(self, frame):
        dcm = sk.frametransform.to_gcrf(frame, POS, VEL)
        np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-14)
        np.testing.assert_array_equal(sk.frametransform.from_gcrf(frame, POS, VEL), dcm.T)
        # lists and integer arrays are array-likes too
        np.testing.assert_array_equal(
            sk.frametransform.to_gcrf(frame, [7000000, 0, 0], (0, 7500, 0)), dcm
        )

    @pytest.mark.parametrize("func", ["to_gcrf", "from_gcrf"])
    @pytest.mark.parametrize(
        "frame", [sk.frame.ITRF, sk.frame.TEME, sk.frame.EME2000, sk.frame.ICRF]
    )
    def test_unsupported_frame_message_names_the_function(self, func, frame):
        with pytest.raises(RuntimeError) as e:
            getattr(sk.frametransform, func)(frame, POS, VEL)
        assert str(e.value).startswith(func + ":")

    def test_rotation_with_state(self):
        with pytest.raises(ValueError, match="pos is zero"):
            sk.frametransform.rotation_with_state(
                sk.frame.RTN, sk.frame.GCRF, T0, np.zeros(3), np.zeros(3)
            )
        # Earth-frame pairs never look at the state
        q = sk.frametransform.rotation_with_state(
            sk.frame.ITRF, sk.frame.GCRF, T0, np.zeros(3), np.zeros(3)
        )
        assert q.angle == pytest.approx(sk.frametransform.qitrf2gcrf(T0).angle, abs=1e-12)


# ───────────────────────── bug 15: complex arrays ─────────────────────────


class TestComplexRejected:
    """A complex array was truncated to its real part with only a ComplexWarning."""

    cpos = POS.astype(complex)

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: sk.frametransform.itrf_to_gcrf_state(c, VEL, T0),
            lambda c: sk.frametransform.itrf_to_gcrf_state(POS, c, T0),
            lambda c: sk.frametransform.gcrf_to_itrf_state(c, VEL, T0),
            lambda c: sk.frametransform.itrf_to_gcrf_state_approx(c, VEL, T0),
            lambda c: sk.frametransform.gcrf_to_itrf_state_approx(POS, c, T0),
            lambda c: sk.frametransform.transform_state(sk.frame.ITRF, sk.frame.GCRF, T0, c, VEL),
            lambda c: sk.frametransform.transform_state_approx(sk.frame.ITRF, sk.frame.GCRF, T0, POS, c),
            lambda c: sk.frametransform.rotation_with_state(sk.frame.RTN, sk.frame.GCRF, T0, c, VEL),
            lambda c: sk.frametransform.to_gcrf(sk.frame.RTN, c, VEL),
            lambda c: sk.frametransform.from_gcrf(sk.frame.RTN, POS, c),
            lambda c: sk.sun.shadowfunc(c, POS),
            lambda c: sk.sun.shadowfunc(POS, c),
            # callers of the shared extractor outside the submodules
            lambda c: sk.kepler.from_pv(c, VEL),
            lambda c: sk.thrust.constant(c, T0, T0 + sk.duration(seconds=60), sk.frame.GCRF),
        ],
    )
    def test_complex_is_a_type_error(self, call):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a ComplexWarning would be the old bug
            with pytest.raises(TypeError, match="real numeric"):
                call(self.cpos)

    def test_batch_complex(self):
        with pytest.raises((TypeError, ValueError)):
            sk.frametransform.itrf_to_gcrf_state(
                np.tile(POS, (2, 1)), np.tile(VEL, (2, 1)).astype(complex), [T0, T0]
            )

    def test_strings_and_bools(self):
        with pytest.raises(TypeError):
            sk.sun.shadowfunc("abc", POS)
        with pytest.raises(TypeError):
            sk.sun.shadowfunc(np.array([True, False, True]), POS)

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="sunpos must be a 3-element vector"):
            sk.sun.shadowfunc([1.0, 2.0], POS)
        with pytest.raises(ValueError, match="vel must be a 3-element vector"):
            sk.frametransform.to_gcrf(sk.frame.RTN, POS, np.zeros((3, 3)))

    def test_real_array_likes_accepted(self):
        sun = sk.sun.pos_gcrf(T0)
        ref = sk.sun.shadowfunc(sun, POS)
        assert sk.sun.shadowfunc(list(sun), tuple(POS)) == ref
        assert sk.sun.shadowfunc(sun.astype(np.float32).astype(float), POS.astype(np.int64)) == ref
        p, v = sk.frametransform.itrf_to_gcrf_state([7000000, 0, 0], (0, 7500, 0), T0)
        p2, v2 = sk.frametransform.itrf_to_gcrf_state(POS, VEL, T0)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(v, v2)


# ─────────────────── bug 16: a one-element list is not a time ───────────────────


class TestSingleTime:
    @pytest.mark.parametrize(
        "call",
        [
            lambda t: sk.spaceweather.get(t),
            lambda t: sk.spaceweather.status(t),
            lambda t: sk.frametransform.transform_state(sk.frame.ITRF, sk.frame.GCRF, t, POS, VEL),
            lambda t: sk.frametransform.transform_state_approx(sk.frame.ITRF, sk.frame.GCRF, t, POS, VEL),
            lambda t: sk.frametransform.rotation_with_state(sk.frame.RTN, sk.frame.GCRF, t, POS, VEL),
            lambda t: sk.frametransform.itrf_to_gcrf_state(POS, VEL, t),
            lambda t: sk.nrlmsise00(400.0, time=t),
        ],
    )
    @pytest.mark.parametrize(
        "make",
        [
            lambda: [T0],
            lambda: [T0, T0],
            lambda: np.array([T0], dtype=object),
            lambda: np.array(["2024-03-01T06:00"], dtype="datetime64[s]"),
        ],
        ids=["list1", "list2", "objarray", "datetime64array"],
    )
    def test_list_is_not_a_scalar_time(self, call, make):
        with pytest.raises(TypeError, match="single time") as e:
            call(make())
        assert "numpy.datetime64" in str(e.value)

    @pytest.mark.parametrize(
        "t",
        [
            T0,
            datetime.datetime(2024, 3, 1, 6, tzinfo=datetime.timezone.utc),
            np.datetime64("2024-03-01T06:00:00"),
            np.array("2024-03-01T06:00:00", dtype="datetime64[s]"),  # 0-d array
        ],
        ids=["time", "datetime", "datetime64", "datetime64-0d"],
    )
    def test_scalar_times_accepted(self, t):
        assert sk.spaceweather.status(t) == sk.spaceweather.status(T0)
        assert sk.spaceweather.get(t)["date"] == sk.spaceweather.get(T0)["date"]
        p, _ = sk.frametransform.transform_state(sk.frame.ITRF, sk.frame.GCRF, t, POS, VEL)
        p0, _ = sk.frametransform.transform_state(sk.frame.ITRF, sk.frame.GCRF, T0, POS, VEL)
        np.testing.assert_array_equal(p, p0)


# ═════════════════════════════ additions ═════════════════════════════

# ─────────────────── native submodules are importable ───────────────────

import importlib
import os
import pathlib
import pickle
import subprocess
import sys
import types

NATIVE_SUBMODULES = [
    "density",
    "frametransform",
    "jplephem",
    "moon",
    "planets",
    "spaceweather",
    "sun",
    "utils",
]


class TestSubmodulesImportable:
    def test_the_list_is_complete(self):
        found = sorted(
            n for n, m in vars(sk.satkit).items() if isinstance(m, types.ModuleType)
        )
        assert found == NATIVE_SUBMODULES

    @pytest.mark.parametrize("name", NATIVE_SUBMODULES)
    def test_import(self, name):
        mod = importlib.import_module(f"satkit.{name}")
        assert mod is getattr(sk, name)
        assert mod is sys.modules[f"satkit.{name}"]
        assert mod.__name__ == f"satkit.{name}"

    @pytest.mark.parametrize("name", NATIVE_SUBMODULES)
    def test_members_know_their_module(self, name):
        mod = getattr(sk, name)
        public = [n for n in dir(mod) if not n.startswith("_")]
        assert public
        for attr in public:
            obj = getattr(mod, attr)
            assert obj.__module__ == f"satkit.{name}", attr
            # ... which is what makes them picklable (by reference)
            assert pickle.loads(pickle.dumps(obj)) is obj

    @pytest.mark.parametrize("name", NATIVE_SUBMODULES)
    def test_from_import_in_a_fresh_interpreter(self, name):
        """`import satkit.x` / `from satkit.x import y` as the first import"""
        member = next(n for n in dir(getattr(sk, name)) if not n.startswith("_"))
        code = (
            f"import satkit.{name}\n"
            f"from satkit.{name} import {member}\n"
            f"import satkit\n"
            f"assert {member} is satkit.{name}.{member}\n"
            f"assert satkit.{name}.__name__ == 'satkit.{name}'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_pickled_function_loads_in_a_fresh_interpreter(self):
        blob = pickle.dumps((sk.sun.pos_gcrf, sk.moon.moonphase.FullMoon, sk.solarsystem.Mars))
        code = (
            "import pickle, sys\n"
            "f, phase, body = pickle.loads(sys.stdin.buffer.read())\n"
            "import satkit\n"
            "assert f is satkit.sun.pos_gcrf\n"
            "assert phase == satkit.moon.moonphase.FullMoon\n"
            "assert body == satkit.solarsystem.Mars\n"
            "assert f(satkit.time(2024, 1, 1)).shape == (3,)\n"
        )
        subprocess.run([sys.executable, "-c", code], input=blob, check=True)


# ───────────────────────── enums: hash, module ─────────────────────────


class TestSubmoduleEnums:
    @pytest.mark.parametrize(
        "cls, module, n",
        [(sk.solarsystem, "satkit", 11), (sk.moon.moonphase, "satkit.moon", 8)],
    )
    def test_hashable_frozen_and_named(self, cls, module, n):
        members = [
            getattr(cls, a)
            for a in dir(cls)
            if not a.startswith("_") and isinstance(getattr(cls, a), cls)
        ]
        assert len(members) == n
        assert cls.__module__ == module
        assert len(set(members)) == n
        assert {m: int(m) for m in members}[members[0]] == int(members[0])
        for m in members:
            assert m == int(m) and hash(m) == hash(pickle.loads(pickle.dumps(m)))
            with pytest.raises(AttributeError):
                m.x = 1

    def test_values(self):
        assert int(sk.solarsystem.Mercury) == 0
        assert int(sk.solarsystem.EMB) == 2
        assert int(sk.solarsystem.Sun) == 10
        assert int(sk.moon.moonphase.NewMoon) == 0
        assert int(sk.moon.moonphase.WaningCrescent) == 7
        by_phase = {sk.moon.moonphase.FullMoon: "full"}
        full = sk.moon.next_phase(T0, sk.moon.moonphase.FullMoon)
        assert by_phase[sk.moon.phase_name(full)] == "full"


# ───────────────────────── PathLike arguments ─────────────────────────


class TestPathLike:
    def test_utils_directories(self, tmp_path):
        """In a child process: these change process-wide search state"""
        code = (
            "import pathlib, sys, satkit\n"
            "d = pathlib.Path(sys.argv[1])\n"
            "(d / 'a').mkdir(); (d / 'b').mkdir()\n"
            "satkit.utils.add_search_dir(d / 'a')\n"
            "satkit.utils.add_search_dir(str(d / 'a'))\n"
            "assert str(d / 'a') in satkit.utils.data_search_dirs()\n"
            "satkit.utils.set_datadir(d / 'b')\n"
            "assert satkit.utils.datadir() == str(d / 'b')\n"
            "satkit.utils.set_datadir(str(d / 'b'))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "SATKIT_DATA"}
        subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, env=env)

    def test_not_a_path(self):
        with pytest.raises(TypeError):
            sk.spaceweather.init_from_path(5)
        with pytest.raises(TypeError):
            sk.utils.set_datadir(5)
        with pytest.raises(TypeError):
            sk.utils.add_search_dir(None)


# ───────────────────────── tuples of times ─────────────────────────


class TestTimeTuples:
    times = [T0, T0 + sk.duration(hours=3), T0 + sk.duration(days=2)]

    def test_tuple_equals_list(self):
        tup = tuple(self.times)
        np.testing.assert_array_equal(sk.sun.pos_gcrf(tup), sk.sun.pos_gcrf(self.times))
        np.testing.assert_array_equal(sk.moon.pos_gcrf(tup), sk.moon.pos_gcrf(self.times))
        np.testing.assert_array_equal(
            sk.frametransform.gmst(tup), sk.frametransform.gmst(self.times)
        )
        np.testing.assert_array_equal(
            sk.jplephem.geocentric_pos(sk.solarsystem.Sun, tup),
            sk.jplephem.geocentric_pos(sk.solarsystem.Sun, self.times),
        )
        np.testing.assert_array_equal(
            sk.planets.heliocentric_pos(sk.solarsystem.Mars, tup),
            sk.planets.heliocentric_pos(sk.solarsystem.Mars, self.times),
        )
        qs = sk.frametransform.qitrf2gcrf_approx(tup)
        assert isinstance(qs, list) and len(qs) == 3
        assert sk.moon.phase_name(tup) == sk.moon.phase_name(self.times)

    def test_mixed_and_batch(self):
        mixed = (T0, T0.to_datetime(), np.datetime64("2024-03-01T06:00:00"))
        pos = sk.sun.pos_gcrf(mixed)
        np.testing.assert_allclose(pos[1], pos[0], rtol=1e-12)
        np.testing.assert_array_equal(pos[2], pos[0])
        p, v = sk.frametransform.itrf_to_gcrf_state(
            np.tile(POS, (3, 1)), np.tile(VEL, (3, 1)), tuple(self.times)
        )
        assert p.shape == v.shape == (3, 3)

    def test_one_element_and_empty_tuples_are_arrays(self):
        assert sk.sun.pos_gcrf((T0,)).shape == (1, 3)
        assert sk.sun.pos_gcrf(()).shape == (0, 3)
        with pytest.raises(TypeError, match="single time"):
            sk.spaceweather.status((T0,))

    def test_tuple_of_non_times(self):
        with pytest.raises(TypeError, match="list or tuple"):
            sk.sun.pos_gcrf((1.0, 2.0))


# ─────────────── stub docstrings: Args names are parameters ───────────────

import ast
import re

STUB_DIR = pathlib.Path(__file__).resolve().parents[1] / "satkit"

# Known failures: functions whose stub docstring documents an argument that
# no overload of the function has. Each entry is
# "<stub file>::<qualified name>": [names]. Empty at the time of writing:
# every stub passes. Add an entry only for a mismatch that cannot be fixed
# in the same change, and delete it with the fix
# (test_known_mismatches_are_current fails for an entry that no longer
# applies).
KNOWN_ARGS_MISMATCHES: dict[str, list[str]] = {}

_SECTION = re.compile(
    r"^(\s*)(Args|Arguments|Parameters|Keyword Args|Keyword Arguments):\s*$"
)
_ENTRY = re.compile(r"^(\s*)\*{0,2}([A-Za-z_]\w*)\s*(\([^)]*\))?\s*:")


def _documented_args(doc: str) -> list[str]:
    """Names listed in the Google-style ``Args:`` sections of a docstring"""
    names: list[str] = []
    lines = doc.expandtabs().splitlines()
    i = 0
    while i < len(lines):
        m = _SECTION.match(lines[i])
        i += 1
        if not m:
            continue
        base = len(m.group(1))
        entry_indent = None
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base:
                break  # next section
            if entry_indent is None:
                entry_indent = indent
            if indent == entry_indent:
                e = _ENTRY.match(line)
                if e:
                    names.append(e.group(2))
            i += 1
    return names


def _signature_names(fn) -> set[str]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _stub_functions():
    """(stub file name, qualified name, FunctionDef) for every function and
    method in every stub, each overload separately"""

    def walk(fname, body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                yield from walk(fname, node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield fname, prefix + node.name, node

    for path in sorted(STUB_DIR.glob("*.pyi")):
        tree = ast.parse(path.read_text(), filename=str(path))
        yield from walk(path.name, tree.body, "")


def _args_mismatches() -> dict[str, list[str]]:
    """Documented argument names that are not parameters. The overloads of a
    function are taken together: one docstring often documents all the forms
    (``time.__init__`` lists ``string`` and the calendar fields), so a name
    is fine if any overload has it."""
    params: dict[str, set[str]] = {}
    for fname, qual, fn in _stub_functions():
        params.setdefault(f"{fname}::{qual}", set()).update(_signature_names(fn))
    out: dict[str, list[str]] = {}
    for fname, qual, fn in _stub_functions():
        key = f"{fname}::{qual}"
        doc = ast.get_docstring(fn, clean=False)
        if not doc:
            continue
        for n in _documented_args(doc):
            if n not in params[key] and n not in out.setdefault(key, []):
                out[key].append(n)
    return {k: v for k, v in out.items() if v}


class TestStubDocstringArgs:
    """Every name in a stub docstring's ``Args:`` section is a parameter of
    that function (the cross-cutting check from the API review: the stub and
    the Rust ``///`` docstring are written separately and drift)."""

    def test_parser(self):
        doc = """Summary

        Args:
            a (int): first
                continuation: not an entry
            *rest: more
            b: second

        Returns:
            c (int): not an argument
        """
        assert _documented_args(doc) == ["a", "rest", "b"]

    def test_stubs_are_scanned(self):
        seen = {f for f, _q, _fn in _stub_functions()}
        assert {"satkit.pyi", "sun.pyi", "frametransform.pyi", "utils.pyi"} <= seen
        documented = sum(
            1
            for _f, _q, fn in _stub_functions()
            if _documented_args(ast.get_docstring(fn, clean=False) or "")
        )
        assert documented > 150

    def test_a_stray_name_is_caught(self):
        fn = ast.parse(
            'def f(tm, *, use_jpl=False):\n    """S\n\n    Args:\n        time: t\n        use_jpl: u\n    """\n'
        ).body[0]
        doc = ast.get_docstring(fn, clean=False)
        assert [n for n in _documented_args(doc) if n not in _signature_names(fn)] == ["time"]

    def test_args_are_parameters(self):
        found = _args_mismatches()
        unexpected = {k: v for k, v in found.items() if k not in KNOWN_ARGS_MISMATCHES}
        assert not unexpected, (
            "stub docstrings document arguments that are not in the signature: "
            f"{unexpected}"
        )

    def test_known_mismatches_are_current(self):
        found = _args_mismatches()
        stale = {
            k: v
            for k, v in KNOWN_ARGS_MISMATCHES.items()
            if sorted(found.get(k, [])) != sorted(v)
        }
        assert not stale, f"entries of KNOWN_ARGS_MISMATCHES to update or delete: {stale}"


# ═══════════════ changes of return type and exception type ═══════════════

# ───────────── scalar functions of time return ndarray for arrays ─────────────

FLOAT_OF_TIME = [
    ("gmst", sk.frametransform.gmst),
    ("gast", sk.frametransform.gast),
    ("eqeq", sk.frametransform.eqeq),
    ("earth_rotation_angle", sk.frametransform.earth_rotation_angle),
    ("moon.phase", sk.moon.phase),
    ("moon.illumination", sk.moon.illumination),
]


class TestFloatOfTimeReturnsArray:
    times = [T0 + sk.duration(hours=7 * i) for i in range(5)]

    @pytest.mark.parametrize("fn", [f for _, f in FLOAT_OF_TIME], ids=[n for n, _ in FLOAT_OF_TIME])
    def test_scalar_in_float_out(self, fn):
        for t in (T0, T0.to_datetime(), np.datetime64("2024-03-01T06:00:00")):
            out = fn(t)
            assert type(out) is float
        assert fn(T0) == fn(np.datetime64("2024-03-01T06:00:00"))

    @pytest.mark.parametrize("fn", [f for _, f in FLOAT_OF_TIME], ids=[n for n, _ in FLOAT_OF_TIME])
    def test_array_in_ndarray_out(self, fn):
        expect = np.array([fn(t) for t in self.times])
        d64 = np.array([t.to_datetime().replace(tzinfo=None) for t in self.times], dtype="datetime64[us]")
        for arg in (self.times, tuple(self.times), np.array(self.times, dtype=object), d64):
            out = fn(arg)
            assert isinstance(out, np.ndarray)
            assert out.dtype == np.float64 and out.shape == (5,)
            np.testing.assert_array_equal(out, expect)
        one = fn([T0])
        assert isinstance(one, np.ndarray) and one.shape == (1,) and one[0] == fn(T0)
        empty = fn([])
        assert isinstance(empty, np.ndarray) and empty.shape == (0,) and empty.dtype == np.float64

    def test_ranges_are_not_normalised_for_sidereal_time(self):
        """gmst / gast keep their sign: negative before J2000 (documented)"""
        before, after = sk.time(1990, 1, 1), sk.time(2024, 1, 1)
        g = sk.frametransform.gmst([before, after])
        assert g[0] == pytest.approx(-4.531136705964937, abs=1e-9)
        assert g[0] % (2 * math.pi) == pytest.approx(1.7520486012146493, abs=1e-9)
        assert 0 < g[1] < 2 * math.pi
        assert sk.frametransform.gast(before) < 0
        era = sk.frametransform.earth_rotation_angle([before, after])
        assert np.all((era >= 0) & (era < 2 * math.pi))

    def test_moon_phase_range(self):
        days = [sk.time(2024, 1, 1) + sk.duration(days=d) for d in range(60)]
        ph = sk.moon.phase(days)
        assert np.all((ph >= 0) & (ph < 2 * math.pi))
        np.testing.assert_allclose(sk.moon.illumination(days), 0.5 * (1 - np.cos(ph)), atol=1e-15)
        new = sk.moon.next_phase(sk.time(2024, 1, 1), sk.moon.moonphase.NewMoon)
        full = sk.moon.next_phase(sk.time(2024, 1, 1), sk.moon.moonphase.FullMoon)
        assert min(sk.moon.phase(new), 2 * math.pi - sk.moon.phase(new)) < 1e-6
        assert sk.moon.phase(full) == pytest.approx(math.pi, abs=1e-6)

    def test_phase_name_is_still_a_list(self):
        names = sk.moon.phase_name(self.times)
        assert isinstance(names, list) and len(names) == 5


# ───────────────────────── exception types ─────────────────────────


class TestTimeArgumentTypeErrors:
    """Something that is not a time is a TypeError (it was RuntimeError)."""

    @pytest.mark.parametrize(
        "bad",
        [5, 5.0, "2024-01-01", None, np.array([1.0, 2.0]), np.float64(3.0), {"t": 1},
         np.array([5, "a"], dtype=object)],
        ids=["int", "float", "str", "None", "floatarray", "npfloat", "dict", "objarray"],
    )
    @pytest.mark.parametrize(
        "call",
        [
            sk.frametransform.gmst,
            sk.frametransform.qitrf2gcrf_approx,
            sk.sun.pos_gcrf,
            sk.moon.pos_gcrf,
            sk.moon.phase,
            lambda t: sk.planets.heliocentric_pos(sk.solarsystem.Mars, t),
            lambda t: sk.jplephem.geocentric_pos(sk.solarsystem.Sun, t),
            lambda t: sk.jplephem.barycentric_state(sk.solarsystem.Sun, t),
            lambda t: sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, t),
            lambda t: sk.frametransform.transform_state(sk.frame.ITRF, sk.frame.GCRF, t, POS, VEL),
            sk.spaceweather.status,
            sk.spaceweather.get,
        ],
    )
    def test_not_a_time(self, call, bad):
        with pytest.raises(TypeError, match="numpy.datetime64"):
            call(bad)

    def test_value_errors_are_kept(self):
        """A time of the right type but a bad value is not a TypeError"""
        with pytest.raises(ValueError, match="NaT"):
            sk.frametransform.gmst(np.datetime64("NaT", "s"))
        with pytest.raises(ValueError, match="1-D"):
            sk.frametransform.gmst(np.zeros((2, 2), dtype="datetime64[s]"))


class TestSubmoduleExceptionTypes:
    @pytest.mark.parametrize("body", [sk.solarsystem.Sun, sk.solarsystem.Moon])
    def test_planets_invalid_body(self, body):
        for t in (T0, [T0, T0]):
            with pytest.raises(ValueError, match="Invalid body"):
                sk.planets.heliocentric_pos(body, t)

    def test_planets_time_out_of_range_is_runtime(self):
        with pytest.raises(RuntimeError, match="Time out of range"):
            sk.planets.heliocentric_pos(sk.solarsystem.Mars, sk.time(3500, 1, 1))
        assert sk.planets.heliocentric_pos(sk.solarsystem.EMB, T0).shape == (3,)

    def test_jplephem_outside_span_is_runtime(self):
        """Documented: a time outside the loaded DE file's span is RuntimeError"""
        for fn in (sk.jplephem.geocentric_pos, sk.jplephem.barycentric_pos,
                   sk.jplephem.geocentric_state, sk.jplephem.barycentric_state):
            with pytest.raises(RuntimeError, match="Invalid Julian date"):
                fn(sk.solarsystem.Moon, sk.time(3000, 1, 1))

    def test_set_datadir_missing(self, tmp_path):
        missing = tmp_path / "no" / "such" / "dir"
        for arg in (missing, str(missing)):
            with pytest.raises(FileNotFoundError, match="does not exist") as e:
                sk.utils.set_datadir(arg)
            assert str(missing) in str(e.value)
            assert isinstance(e.value, OSError)
        # a file is not a directory either
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(FileNotFoundError):
            sk.utils.set_datadir(f)


# ───────────────────────── stub return types ─────────────────────────


class TestStubReturnTypes:
    """The fixed sets of strings the stubs declare as Literal are the sets
    the functions can return (read from the stub, compared with the binding)."""

    @staticmethod
    def _literal(stub: str, func: str) -> set[str]:
        tree = ast.parse((STUB_DIR / stub).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
        ret = fn.returns
        assert isinstance(ret, ast.Subscript) and ast.unparse(ret.value).endswith("Literal")
        return {c.value for c in ast.walk(ret.slice) if isinstance(c, ast.Constant)}

    def test_eop_status(self):
        lit = self._literal("frametransform.pyi", "eop_status")
        assert lit == {"observed", "predicted", "extrapolated", "before_table", "not_loaded"}
        first, last_obs, last = sk.frametransform.eop_coverage()
        got = {
            sk.frametransform.eop_status(t)
            for t in (first - sk.duration(days=10), first, last_obs, last, last + sk.duration(days=10))
        }
        assert got == {"before_table", "observed", "predicted", "extrapolated"}
        assert got <= lit

    def test_spaceweather_status(self):
        lit = self._literal("spaceweather.pyi", "status")
        assert lit == {
            "observed", "predicted_daily", "predicted_monthly", "extrapolated",
            "before_table", "not_loaded",
        }
        first, last_obs, last_daily, last = sk.spaceweather.coverage()
        got = {
            sk.spaceweather.status(t)
            for t in (first - sk.duration(days=10), last_obs, last + sk.duration(days=40))
        }
        assert got == {"before_table", "observed", "extrapolated"}
        assert got <= lit

    def test_spaceweather_record_keys(self):
        tree = ast.parse((STUB_DIR / "spaceweather.pyi").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpaceWeatherRecord")
        keys = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
        rec = sk.spaceweather.get(T0)
        assert list(rec) == keys and len(keys) == 17
        assert isinstance(rec["date"], sk.time)
        assert len(rec["kp"]) == len(rec["ap"]) == 8
        # kp_sum is the sum of the x10 Kp values, to rounding of the thirds
        assert abs(rec["kp_sum"] - sum(rec["kp"])) <= 8
        assert sk.spaceweather.get(sk.time(1940, 1, 1))["f10p7_obs"] == -1.0
        assert not hasattr(sk.spaceweather, "SpaceWeatherRecord")  # type-check only
