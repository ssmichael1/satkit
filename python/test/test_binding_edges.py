"""Binding edge cases: argument forms the stubs promise and the bindings must
accept (or reject with a clear error).

Covers the keyword-only ``satproperties`` constructor, list-in / list-out for
one-element time lists, real numeric array-likes for quaternion rotation,
explicit ``None`` for optional keywords, ``datetime`` wherever a scalar time is
accepted, offline mode for the element-set URL fetches, misspelt keywords, enum
pickling and ``duration`` arithmetic.
"""

import copy
import datetime
import pickle

import numpy as np
import pytest

import satkit as sk
from shared import ISS_2024, same

T0 = sk.time(2024, 1, 1, 12, 0, 0)
DT0 = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

# ─────────────────────────── satproperties ───────────────────────────


class TestSatPropertiesKeywordOnly:
    @pytest.mark.parametrize("args", [(0.01,), (0.01, 0.02), (0.01, 0.02, None)])
    def test_positional_raises_naming_the_keywords(self, args):
        with pytest.raises(TypeError, match=r"keyword arguments only: cdaoverm=, craoverm=, thrusts=, ecom="):
            sk.satproperties(*args)

    def test_keywords_bind_by_name(self):
        p = sk.satproperties(craoverm=0.02, cdaoverm=0.01)
        assert (p.cdaoverm, p.craoverm) == (0.01, 0.02)
        p = sk.satproperties()
        assert (p.cdaoverm, p.craoverm) == (0.0, 0.0)

    def test_explicit_none(self):
        p = sk.satproperties(cdaoverm=0.01, thrusts=None, ecom=None)
        assert p.thrusts == [] and p.ecom is None

    def test_unknown_keyword_rejected(self):
        with pytest.raises(TypeError, match="cdaovrm"):
            sk.satproperties(cdaovrm=0.01)

    def test_signature_is_keyword_only(self):
        import inspect

        params = inspect.signature(sk.satproperties).parameters.values()
        assert [p.name for p in params] == ["cdaoverm", "craoverm", "thrusts", "ecom"]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)


# ─────────────────────── one-element time lists ───────────────────────


def _scalar_and_list_cases():
    ft = sk.frametransform
    return [
        ("gmst", ft.gmst),
        ("gast", ft.gast),
        ("eqeq", ft.eqeq),
        ("earth_rotation_angle", ft.earth_rotation_angle),
        ("qitrf2gcrf", ft.qitrf2gcrf),
        ("qteme2gcrf", ft.qteme2gcrf),
        ("rotation", lambda t: ft.rotation(sk.frame.ITRF, sk.frame.GCRF, t)),
        ("rotation_approx", lambda t: ft.rotation_approx(sk.frame.TEME, sk.frame.GCRF, t)),
        ("sun.pos_gcrf", sk.sun.pos_gcrf),
        ("moon.pos_gcrf", sk.moon.pos_gcrf),
        ("moon.phase", sk.moon.phase),
        ("moon.illumination", sk.moon.illumination),
        ("heliocentric_pos", lambda t: sk.planets.heliocentric_pos(sk.solarsystem.Mars, t)),
    ]


CASES = _scalar_and_list_cases()


class TestOneElementTimeLists:
    @pytest.mark.parametrize("fn", [f for _, f in CASES], ids=[n for n, _ in CASES])
    def test_list_in_list_out(self, fn):
        scalar = fn(T0)
        assert not isinstance(scalar, list)
        assert not (isinstance(scalar, np.ndarray) and scalar.ndim == 2)
        for arg in ([T0], np.array([T0]), [DT0]):
            out = fn(arg)
            assert isinstance(out, (list, np.ndarray)) and len(out) == 1, (arg, out)
            assert same(out[0], scalar)
        two = fn([T0, T0])
        assert len(two) == 2

    def test_vector_functions_keep_the_time_axis(self):
        assert sk.sun.pos_gcrf(T0).shape == (3,)
        assert sk.sun.pos_gcrf([T0]).shape == (1, 3)

    def test_sgp4(self):
        tle = sk.TLE.from_lines(ISS_2024)[0]
        p, v = sk.sgp4(tle, T0)
        assert p.shape == v.shape == (3,)
        p1, v1 = sk.sgp4(tle, [T0])
        assert p1.shape == v1.shape == (1, 3)
        np.testing.assert_array_equal(p1[0], p)
        p2, _ = sk.sgp4([tle, tle], [T0])
        assert p2.shape == (2, 1, 3)
        p3, _ = sk.sgp4([tle, tle], T0)
        assert p3.shape == (2, 3)

    def test_jplephem(self):
        try:
            scalar = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, T0)
        except Exception as e:  # no ephemeris file available
            pytest.skip(f"JPL ephemeris unavailable: {e}")
        out = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, [T0])
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out[0], scalar)

    def test_empty_list(self):
        assert sk.frametransform.gmst([]) == []
        assert sk.sun.pos_gcrf([]).shape == (0, 3)


# ─────────────────────── quaternion array-likes ───────────────────────


class TestQuaternionArrayLikes:
    q = sk.quaternion.rotz(np.pi / 2)

    @pytest.mark.parametrize(
        "v",
        [np.array([1, 0, 0]), [1, 0, 0], (1, 0, 0), np.array([1, 0, 0], dtype=np.int32),
         np.array([1, 0, 0], dtype=np.uint8), np.array([1, 0, 0], dtype=np.float32)],
        ids=["int64", "list", "tuple", "int32", "uint8", "float32"],
    )
    def test_rotate_vector(self, v):
        np.testing.assert_allclose(self.q * v, [0, 1, 0], atol=1e-15)

    def test_rotate_nx3(self):
        out = self.q * np.array([[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(out, [[0, 1, 0], [-1, 0, 0]], atol=1e-15)
        np.testing.assert_allclose(self.q * [[1, 0, 0]], [[0, 1, 0]], atol=1e-15)

    def test_nx3_matches_row_by_row(self):
        """Each row of q * V is q * v_i (the Nx3 path used to apply the
        inverse rotation)."""
        q = sk.quaternion.from_axis_angle([1, 2, 3], 0.7)
        v = np.random.default_rng(3).normal(size=(5, 3))
        rows = np.array([q * r for r in v])
        np.testing.assert_allclose(q * v, rows, rtol=0, atol=1e-15)
        np.testing.assert_allclose(q * v, v @ q.to_rotation_matrix().T, rtol=0, atol=1e-15)
        np.testing.assert_allclose(q * np.asfortranarray(v), rows, rtol=0, atol=1e-15)

    def test_float_path_unchanged(self):
        v = np.array([0.3, -1.2, 2.5])
        np.testing.assert_array_equal(self.q * v, self.q * v.tolist())
        np.testing.assert_array_equal(self.q * v[::-1], self.q * v[::-1].copy())

    def test_rotation_between(self):
        for v1, v2 in (([1, 0, 0], [0, 1, 0]), (np.array([1, 0, 0]), np.array([0, 1, 0]))):
            q = sk.quaternion.rotation_between(v1, v2)
            np.testing.assert_allclose(q * v1, [0, 1, 0], atol=1e-15)

    def test_from_axis_angle_and_matrix(self):
        q = sk.quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(q * [1, 0, 0], [0, 1, 0], atol=1e-15)
        q = sk.quaternion.from_rotation_matrix(np.eye(3, dtype=int))
        assert abs(q.angle) < 1e-15

    @pytest.mark.parametrize(
        "bad, exc",
        [(np.array([1j, 0, 0]), TypeError), (["a", "b", "c"], TypeError), ([1, 2], ValueError),
         (np.zeros((2, 2)), ValueError), (2, ValueError)],
        ids=["complex", "strings", "len2", "Nx2", "scalar"],
    )
    def test_rejects(self, bad, exc):
        with pytest.raises(exc):
            self.q * bad

    def test_rotation_between_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="v1 must be a 3-element vector"):
            sk.quaternion.rotation_between([1, 0], [0, 1, 0])


# ─────────────────────────── explicit None ───────────────────────────


class TestExplicitNone:
    def test_satstate_propagate(self):
        s = sk.satstate(T0, np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
        dt = sk.duration(seconds=60)
        ref = s.propagate(dt)
        for kw in ({"propsettings": None}, {"satproperties": None}, {"propsettings": None, "satproperties": None}):
            np.testing.assert_array_equal(s.propagate(dt, **kw).pos, ref.pos)

    def test_nrlmsise00(self):
        assert sk.nrlmsise00(400.0, time=None) == sk.nrlmsise00(400.0)

    def test_propagate(self):
        state = np.array([7e6, 0, 0, 0, 7.5e3, 0])
        ref = sk.propagate(state, T0, duration_secs=60)
        for kw in (
            {"end": None, "duration_secs": 60},
            {"duration": None, "duration_secs": 60, "duration_days": None},
            {"propsettings": None, "satproperties": None, "duration_secs": 60},
        ):
            np.testing.assert_array_equal(sk.propagate(state, T0, **kw).state_end, ref.state_end)
        np.testing.assert_array_equal(sk.propagate(state, T0, None, duration_secs=60).state_end, ref.state_end)


# ─────────────────────── datetime as a scalar time ───────────────────────


class TestDatetimeTimes:
    def test_satstate_propagate(self):
        s = sk.satstate(T0, np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
        a = s.propagate(T0 + sk.duration(seconds=90))
        b = s.propagate(DT0 + datetime.timedelta(seconds=90))
        assert a.time == b.time
        np.testing.assert_array_equal(a.pos, b.pos)

    def test_satstate_constructors_and_maneuvers(self):
        a = sk.satstate(DT0, np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
        assert a.time == T0
        k = sk.kepler(7e6, 0.01, 0.5, 0.1, 0.2, 0.3)
        assert sk.satstate.from_kepler(DT0, k).time == T0
        a.add_prograde(DT0 + datetime.timedelta(seconds=10), 1.0)
        a.add_maneuver(DT0 + datetime.timedelta(seconds=20), [0, 1.0, 0], sk.frame.RTN)
        assert a.num_maneuvers == 2
        assert a.maneuvers[0]["time"] == T0 + sk.duration(seconds=10)

    def test_bad_type(self):
        s = sk.satstate(T0, np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
        with pytest.raises(TypeError, match="satkit.time, datetime.datetime or satkit.duration"):
            s.propagate(60.0)


# ─────────────────────── offline URL fetches ───────────────────────


@pytest.fixture
def offline():
    was = sk.utils.is_offline()
    sk.utils.set_offline(True)
    try:
        yield
    finally:
        sk.utils.set_offline(was)


class TestOfflineUrlFetches:
    # `.invalid` never resolves, so even a missed offline check could not
    # reach the network; the message check tells the two failures apart.
    URL = "https://celestrak.invalid/NORAD/elements/gp.php?CATNR=25544&FORMAT="
    MSG = r"not fetched: network access is forbidden \(offline mode was turned on with satkit.utils.set_offline"

    def test_tle_from_url(self, offline):
        with pytest.raises(RuntimeError, match=self.MSG):
            sk.TLE.from_url(self.URL + "tle")

    def test_omm_from_url(self, offline):
        with pytest.raises(RuntimeError, match=self.MSG):
            sk.omm_from_url(self.URL + "json")


# ─────────────────────── misspelt keywords ───────────────────────

_POS = np.array([7000e3, 0.0, 0.0])
_STATE = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])

# Each binding whose keywords are real keyword-only parameters, called with
# one misspelt keyword: Python's own TypeError, naming the keyword.
MISSPELT_KEYWORD_CALLS = {
    "duration": lambda: sk.duration(day=1),
    "propsettings": lambda: sk.propsettings(gravity_degre=8),
    "satproperties": lambda: sk.satproperties(cdaovrm=0.01),
    "itrfcoord": lambda: sk.itrfcoord(latitude_deg=1.0, longitude_deg=2.0, alttiude=3.0),
    "sgp4": lambda: sk.sgp4(ISS_2024, T0, gravconstt=sk.sgp4_gravconst.wgs84),
    "nrlmsise00": lambda: sk.nrlmsise00(400.0, latitude=10.0),
    "propagate": lambda: sk.propagate(_STATE, T0, duration_secs=60.0, propsetings=None),
    "satstate.propagate": lambda: sk.satstate(T0, _POS, _STATE[3:]).propagate(T0, propsetings=None),
}


@pytest.mark.parametrize("call", MISSPELT_KEYWORD_CALLS.values(), ids=MISSPELT_KEYWORD_CALLS.keys())
def test_misspelt_keyword_raises_typeerror(call):
    with pytest.raises(TypeError, match="unexpected keyword argument '(day|gravity_degre|cdaovrm|alttiude|gravconstt|latitude|propsetings)'"):
        call()


@pytest.mark.parametrize(
    "fn",
    [sk.duration, sk.propsettings, sk.itrfcoord, sk.sgp4, sk.nrlmsise00, sk.propagate, sk.satstate.propagate,
     sk.gravity, sk.gravity_and_partials],
    ids=lambda f: f.__qualname__,
)  # fmt: skip
def test_signature_has_no_var_keyword(fn):
    import inspect

    kinds = [p.kind for p in inspect.signature(fn).parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds




class TestGravityKeywords:
    pos = np.array([7000e3, 0.0, 0.0])

    @pytest.mark.parametrize("fn", [sk.gravity, sk.gravity_and_partials], ids=["gravity", "gravity_and_partials"])
    def test_misspelt_keyword(self, fn):
        with pytest.raises(TypeError, match="'degre'"):
            fn(self.pos, degre=8)
        with pytest.raises(TypeError, match="'odrer'"):
            fn(self.pos, odrer=4, modle=sk.gravmodel.jgm3)

    def test_valid_keywords(self):
        a = sk.gravity(self.pos, model=sk.gravmodel.jgm3, degree=8, order=4)
        b, _ = sk.gravity_and_partials(self.pos, model=sk.gravmodel.jgm3, degree=8, order=4)
        np.testing.assert_allclose(a, b, rtol=1e-12)


# ─────────────────────────── enum pickling ───────────────────────────

ENUMS = [
    sk.frame, sk.timescale, sk.weekday, sk.gravmodel, sk.integrator, sk.tidemodel,
    sk.sgp4_gravconst, sk.sgp4_opsmode, sk.sgp4_error, sk.solarsystem, sk.tlefitstatus,
    sk.moon.moonphase,
]  # fmt: skip


def _members(cls):
    return [(n, getattr(cls, n)) for n in dir(cls) if not n.startswith("_") and type(getattr(cls, n)) is cls]


@pytest.mark.parametrize("cls", ENUMS, ids=lambda c: c.__name__)
def test_enum_pickle_roundtrip(cls):
    members = _members(cls)
    assert members
    for name, m in members:
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            m2 = pickle.loads(pickle.dumps(m, protocol=proto))
            assert type(m2) is cls and m2 == m, (name, proto)
        assert copy.deepcopy(m) == m and copy.copy(m) == m
    # inside a container, as a multiprocessing / joblib payload would be
    payload = {"members": [m for _, m in members]}
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_enum_alias_pickles_to_its_value():
    assert pickle.loads(pickle.dumps(sk.frame.RIC)) == sk.frame.RTN


# ─────────────────────────── duration ───────────────────────────


class TestDurationArithmetic:
    d = sk.duration(seconds=10)

    @pytest.mark.parametrize("k", [2, 2.0, np.int64(2), np.int32(2), np.float64(2.0), np.float32(2.0)])
    def test_divide_by_real(self, k):
        assert (self.d / k) == sk.duration(seconds=5)

    def test_divide_by_duration(self):
        assert self.d / sk.duration(seconds=4) == 2.5

    def test_multiply(self):
        assert self.d * 3 == sk.duration(seconds=30)
        assert self.d * np.int64(3) == sk.duration(seconds=30)

    @pytest.mark.parametrize("bad", [1.0, 1, "x"])
    def test_add_non_duration_is_type_error(self, bad):
        with pytest.raises(TypeError, match="unsupported operand"):
            self.d + bad

    def test_divide_by_non_number_is_type_error(self):
        with pytest.raises(TypeError, match="unsupported operand"):
            self.d / "x"

    def test_add_time(self):
        assert self.d + T0 == T0 + self.d
