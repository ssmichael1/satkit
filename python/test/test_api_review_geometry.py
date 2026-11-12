"""Regression tests for the Python API review, geometry area: quaternion,
kepler, itrfcoord / geodetic, consts, frame, gravity, lambert, nrlmsise00 and
the density submodule."""

import math
import pickle

import numpy as np
import pytest

import satkit as sk


# --- geodesics (Vincenty iterated to convergence, antipodal fallback) ---------

# Reference values from GeographicLib (Karney 2013), WGS-84:
# lat1, lon1, lat2, lon2 (deg), distance (m), initial and final azimuth (deg)
GEODESICS = [
    (42.36, -71.06, 40.71, -74.01, 306867.9261, -125.678572512, -127.635097300),
    (51.4778, -0.0015, -33.8568, 151.2153, 16982642.2418, 60.560478098, 139.170422320),
    (-41.32, 174.81, 40.96, -5.5, 19959679.2674, 161.067669986, 18.825195123),
    (0.0, 0.0, 0.0, 90.0, 10018754.1714, 90.0, 90.0),
    (0.0, 0.0, 0.0, 179.0, 19926188.8520, 90.0, 90.0),
    (0.0, 0.0, 0.5, 179.7, 19944127.4208, 15.556882793, 164.442513891),
    (30.0, 0.0, -29.9, 179.95, 19992657.1400, 4.496427384, 175.508078616),
]


def _wrap(angle_rad):
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize("lat1,lon1,lat2,lon2,dist,az1,az2", GEODESICS)
def test_geodesic_reference(lat1, lon1, lat2, lon2, dist, az1, az2):
    a = sk.itrfcoord(latitude_deg=lat1, longitude_deg=lon1)
    b = sk.itrfcoord(latitude_deg=lat2, longitude_deg=lon2)
    s, h1, h2 = a.geodesic_distance(b)
    assert s == pytest.approx(dist, abs=1e-3)
    assert abs(_wrap(h1 - math.radians(az1))) < 1e-7
    assert abs(_wrap(h2 - math.radians(az2))) < 1e-7
    assert a.distance_to(b) == s
    assert b.distance_to(a) == pytest.approx(dist, abs=1e-3)


def test_geodesic_antipodal():
    # Half the meridian ellipse; five fixed iterations gave 19903593.39 m
    a = sk.itrfcoord(latitude_deg=0, longitude_deg=0)
    b = sk.itrfcoord(latitude_deg=0, longitude_deg=180)
    assert a.distance_to(b) == pytest.approx(20003931.4586, abs=1e-3)
    # ... and 179.5 deg came out shorter than 179 deg
    d179 = a.distance_to(sk.itrfcoord(latitude_deg=0, longitude_deg=179.0))
    d1795 = a.distance_to(sk.itrfcoord(latitude_deg=0, longitude_deg=179.5))
    assert d1795 == pytest.approx(19980861.9089, abs=1e-3)
    assert d179 < d1795 < 20003931.4586
    n = sk.itrfcoord(latitude_deg=90, longitude_deg=0)
    s = sk.itrfcoord(latitude_deg=-90, longitude_deg=0)
    assert n.distance_to(s) == pytest.approx(20003931.4586, abs=1e-3)


@pytest.mark.parametrize("lat,lon", [(0, 0), (42.36, -71.06), (-89, 179), (90, 0)])
def test_geodesic_identical_points(lat, lon):
    p = sk.itrfcoord(latitude_deg=lat, longitude_deg=lon)
    assert p.geodesic_distance(p) == (0.0, 0.0, 0.0)
    assert p.distance_to(p) == 0.0


def test_move_with_heading_round_trip():
    start = sk.itrfcoord(latitude_deg=-41.32, longitude_deg=174.81)
    for dist, heading in [(1e3, 0.2), (5e6, 2.1), (19e6, -1.4)]:
        end = start.move_with_heading(dist, heading)
        s, h1, _ = start.geodesic_distance(end)
        assert s == pytest.approx(dist, abs=1e-3)
        assert h1 == pytest.approx(heading, abs=1e-9)


# --- quaternion stays unit ----------------------------------------------------


def test_quaternion_constructor_normalizes():
    q = sk.quaternion(1, 1, 1, 1)
    assert q.norm == pytest.approx(1.0, abs=1e-15)
    assert (q.w, q.x, q.y, q.z) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    # a rotation preserves length: this used to return [-3, 4, 0]
    assert q * [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-15)
    assert sk.quaternion(2, 0, 0, 0).w == 1.0
    big = sk.quaternion(1e200, 1e200, 0, 0)
    assert big.norm == pytest.approx(1.0, abs=1e-15)
    tiny = sk.quaternion(0, 1e-200, 0, 0)
    assert (tiny.w, tiny.x) == (0.0, 1.0)


@pytest.mark.parametrize(
    "args",
    [(0, 0, 0, 0), (math.nan, 0, 0, 1), (1, math.inf, 0, 0), (0.0, -0.0, 0.0, 0.0)],
)
def test_quaternion_constructor_rejects_zero_and_nonfinite(args):
    with pytest.raises(ValueError, match="finite and not all zero"):
        sk.quaternion(*args)


def test_quaternion_from_rotation_matrix_validates():
    q0 = sk.quaternion.from_axis_angle([1, 2, 3], 0.7)
    r = q0.to_rotation_matrix()
    q = sk.quaternion.from_rotation_matrix(r)
    assert q.norm == pytest.approx(1.0, abs=1e-15)
    assert q.to_rotation_matrix() == pytest.approx(r, abs=1e-14)
    # single-precision and integer input is accepted
    q32 = sk.quaternion.from_rotation_matrix(r.astype(np.float32))
    assert q32.norm == pytest.approx(1.0, abs=1e-15)
    assert sk.quaternion.from_rotation_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).angle == pytest.approx(
        math.pi / 2
    )
    bad = [
        2 * np.eye(3),  # scaled: used to give norm 1.32
        np.zeros((3, 3)),
        np.diag([1.0, 1.0, -1.0]),  # reflection
        np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]]),  # shear
        np.full((3, 3), np.nan),
    ]
    for m in bad:
        with pytest.raises(ValueError, match="not a rotation matrix"):
            sk.quaternion.from_rotation_matrix(m)


def test_quaternion_factories_stay_unit():
    with pytest.raises(ValueError):
        sk.quaternion.rotz(math.nan)
    with pytest.raises(ValueError):
        sk.quaternion.from_axis_angle([0, 0, 1], math.inf)
    with pytest.raises(ValueError):
        sk.quaternion.from_axis_angle([math.nan, 0, 1], 1.0)
    with pytest.raises(ValueError):
        sk.quaternion.from_euler(0.1, math.nan, 0.2)
    with pytest.raises(ValueError):
        sk.quaternion.rotz(1.0).slerp(sk.quaternion.rotx(1.0), math.nan)
    # zero axis: identity, whatever the angle
    q = sk.quaternion.from_axis_angle([0, 0, 0], 1.0)
    assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0)
    # nearly parallel vectors: a unit quaternion, with the small angle kept
    q = sk.quaternion.rotation_between([1, 0, 0], [1, 1e-12, 0])
    assert q.norm == pytest.approx(1.0, abs=1e-15)
    assert q.angle == pytest.approx(1e-12, rel=1e-6)
    q = sk.quaternion.rotation_between([1, 0, 0], [2, 0, 0])
    assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0)
    q = sk.quaternion.rotation_between([3, 0, 0], [0, 0, 5])
    assert q * [1, 0, 0] == pytest.approx([0, 0, 1], abs=1e-15)
    for frac in (-0.5, 0.0, 0.3, 1.0, 2.0):
        assert sk.quaternion.rotz(0.2).slerp(sk.quaternion.roty(2.5), frac).norm == pytest.approx(
            1.0, abs=1e-15
        )
    assert pickle.loads(pickle.dumps(sk.quaternion(1, 2, 3, 4))).x == sk.quaternion(1, 2, 3, 4).x


# --- itrfcoord constructor ----------------------------------------------------


def test_itrfcoord_constructor_forms():
    ref = sk.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100.0)
    v = ref.vector
    for c in (
        sk.itrfcoord(v),
        sk.itrfcoord(list(v)),
        sk.itrfcoord(tuple(v)),
        sk.itrfcoord(float(v[0]), float(v[1]), float(v[2])),
        sk.itrfcoord(latitude_rad=math.radians(42.44), longitude_rad=math.radians(-71.15), height=100.0),
        sk.itrfcoord(latitude_deg=42.44, longitude_rad=math.radians(-71.15), altitude=100.0),
    ):
        assert c.vector == pytest.approx(v, abs=1e-8)
    # integer input of every kind
    assert sk.itrfcoord(np.array([7000000, 0, 0])).vector.tolist() == [7.0e6, 0.0, 0.0]
    assert sk.itrfcoord((7000000, 0, 0)).vector.tolist() == [7.0e6, 0.0, 0.0]
    assert sk.itrfcoord(7000000, 0, 0).vector.tolist() == [7.0e6, 0.0, 0.0]
    assert sk.itrfcoord(latitude_deg=90, longitude_deg=0).latitude_deg == pytest.approx(90.0)
    assert sk.itrfcoord(latitude_rad=-math.pi / 2, longitude_rad=0).latitude_deg == pytest.approx(-90.0)
    assert sk.itrfcoord(latitude_deg=10, longitude_deg=370).longitude_deg == pytest.approx(10.0)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (([1.0e6, 2.0e6, 3.0e6],), dict(latitude_deg=3, longitude_deg=4)),  # positional was dropped
        ((1.0e6, 2.0e6, 3.0e6), dict(altitude=5.0)),
        ((), dict(latitude_deg=3, latitude_rad=0.5, longitude_deg=4)),  # rad used to win
        ((), dict(latitude_deg=3, longitude_deg=4, longitude_rad=0.5)),
        ((), dict(latitude_deg=3, longitude_deg=4, altitude=1.0, height=5.0)),
        ((), dict(latitude_deg=3)),
        ((), dict(longitude_rad=3)),
        ((), dict(altitude=3)),
        ((), {}),
        ((1.0, 2.0), {}),
        ((1.0, 2.0, 3.0, 4.0), {}),
        (("abc",), {}),
        ((["a", "b", "c"],), {}),
        ((1.0, "b", 3.0), {}),
    ],
)
def test_itrfcoord_constructor_type_errors(args, kwargs):
    with pytest.raises(TypeError):
        sk.itrfcoord(*args, **kwargs)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), dict(latitude_deg=95, longitude_deg=4)),  # used to wrap to 85 deg, lon + 180
        ((), dict(latitude_deg=-90.001, longitude_deg=4)),
        ((), dict(latitude_rad=1.6, longitude_rad=0)),
        ((), dict(latitude_deg=math.nan, longitude_deg=4)),
        ((), dict(latitude_deg=3, longitude_deg=math.nan)),
        ((), dict(latitude_deg=3, longitude_deg=math.inf)),
        ((), dict(latitude_deg=3, longitude_deg=4, altitude=math.nan)),
        ((), dict(latitude_deg=3, longitude_deg=4, height=math.inf)),
        (([1.0, 2.0],), {}),
        (([1.0, 2.0, 3.0, 4.0],), {}),
        ((np.zeros((3, 1)),), {}),
        (([1.0, math.nan, 3.0],), {}),
        ((1.0, 2.0, math.inf), {}),
    ],
)
def test_itrfcoord_constructor_value_errors(args, kwargs):
    with pytest.raises(ValueError):
        sk.itrfcoord(*args, **kwargs)


def test_itrfcoord_pickle_round_trip():
    c = sk.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100.0)
    assert pickle.loads(pickle.dumps(c)) == c


# --- gravity ------------------------------------------------------------------


@pytest.mark.parametrize("func", [sk.gravity, sk.gravity_and_partials])
def test_gravity_argument_validation(func):
    pos = [7.0e6, 1.0e5, -2.0e5]
    with pytest.raises(ValueError, match="order 6 exceeds the degree"):
        func(pos, degree=4, order=6)  # was silently evaluated at order 4
    with pytest.raises(ValueError, match="origin"):
        func([0, 0, 0])  # was [nan, nan, nan]
    with pytest.raises(ValueError, match="origin"):
        func(sk.itrfcoord(0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        func([math.nan, 0, 0])
    with pytest.raises(ValueError, match="exceeds the maximum"):
        func(pos, degree=71)
    # order == degree and order < degree are fine
    func(pos, degree=4, order=4)
    func(pos, degree=4, order=0)


def test_gravity_shapes():
    pos = sk.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=0)
    g = sk.gravity(pos)
    assert g.shape == (3,)
    assert np.linalg.norm(g) == pytest.approx(9.8, abs=0.05)
    g2, partials = sk.gravity_and_partials(pos.vector)
    assert g2 == pytest.approx(g)
    assert partials.shape == (3, 3)
    assert sk.gravity((7000000, 0, 0)) == pytest.approx(sk.gravity(np.array([7.0e6, 0, 0])))


# --- additions: hashable enums, __eq__, immutable consts, array-likes ----------


def test_frame_and_gravmodel_hashable():
    assert {sk.frame.GCRF: 1, sk.frame.RTN: 2}[sk.frame.RIC] == 2
    assert hash(sk.frame.RSW) == hash(sk.frame.RTN)
    assert len({sk.gravmodel.jgm3, sk.gravmodel.egm96, sk.gravmodel.jgm3}) == 2
    assert int(sk.frame.GCRF) == sk.frame.GCRF
    assert sk.gravmodel.__module__ == "satkit" and sk.frame.__module__ == "satkit"
    assert pickle.loads(pickle.dumps(sk.gravmodel.egm96)) == sk.gravmodel.egm96
    assert pickle.loads(pickle.dumps(sk.frame.NTW)) == sk.frame.NTW


def test_consts_not_reassignable():
    assert sk.consts.__module__ == "satkit"
    with pytest.raises((TypeError, AttributeError)):
        sk.consts.wgs84_a = 5.0
    assert sk.consts.wgs84_a == 6378137.0


def test_quaternion_and_geodetic_eq_hash():
    q = sk.quaternion.rotz(0.3)
    assert q == sk.quaternion.rotz(0.3)
    assert q != sk.quaternion.rotz(0.3000001)
    assert q != 3 and q != None  # noqa: E711
    assert hash(q) == hash(sk.quaternion.rotz(0.3))
    assert q.conj() == q.conjugate()
    c = sk.itrfcoord(latitude_deg=10, longitude_deg=20, altitude=30)
    assert c.geodetic == c.geodetic
    assert hash(c.geodetic) == hash(c.geodetic)
    assert c.geodetic != sk.itrfcoord(latitude_deg=10, longitude_deg=21).geodetic
    assert pickle.loads(pickle.dumps(c.geodetic)) == c.geodetic


def test_lambert_accepts_array_likes():
    ref = sk.lambert(np.array([7000e3, 0, 0]), np.array([0, 7000e3, 0]), 3600.0)
    for r1, r2 in [
        ([7000e3, 0, 0], [0, 7000e3, 0]),
        ((7000000, 0, 0), (0, 7000000, 0)),
        (np.array([7000000, 0, 0]), np.array([0, 9, 7.0e6, 9, 0, 9])[::2]),
    ]:
        sol = sk.lambert(r1, r2, 3600)
        assert sol[0][0] == pytest.approx(ref[0][0])
        assert sol[0][1] == pytest.approx(ref[0][1])
    with pytest.raises(TypeError):
        sk.lambert("abc", [0, 7e6, 0], 3600.0)
    with pytest.raises(ValueError):
        sk.lambert(np.zeros(4), [0, 7e6, 0], 3600.0)


def test_density_module_doc_and_signature():
    assert "NRL MSISE-00" in sk.density.__doc__
    assert sk.density.nrlmsise.__text_signature__ != "(*args)"
    t = sk.time(2024, 1, 1)
    c = sk.itrfcoord(latitude_deg=0, longitude_deg=0, altitude=400e3)
    ref = sk.density.nrlmsise(c, t)
    assert sk.density.nrlmsise(400e3, 0.0, 0.0, t) == pytest.approx(ref, rel=1e-9)
    assert sk.density.nrlmsise(400e3, t) == pytest.approx(ref, rel=1e-9)
    assert sk.density.nrlmsise(400e3, 0.0, t) == pytest.approx(ref, rel=1e-9)
    assert sk.density.nrlmsise(400e3, None) == sk.density.nrlmsise(400e3)
    with pytest.raises(TypeError):
        sk.density.nrlmsise()
    with pytest.raises(TypeError):
        sk.density.nrlmsise(400e3, t, 0.1)
