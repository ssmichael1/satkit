"""API review round 2, geometry area: itrfcoord named constructors and
keyword types, quaternion comparison and conversions, kepler aliases, and
the duration types accepted by kepler.propagate and lambert."""

import datetime
import math

import numpy as np
import pytest

import satkit as sk


# --- itrfcoord ----------------------------------------------------------------


def test_itrfcoord_from_geodetic_matches_keyword_constructor():
    a = sk.itrfcoord.from_geodetic(latitude_deg=42.44, longitude_deg=-71.15, height=100.0)
    b = sk.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100.0)
    assert a == b
    r = sk.itrfcoord.from_geodetic(
        latitude_rad=math.radians(42.44), longitude_rad=math.radians(-71.15), height=100.0
    )
    assert np.allclose(r.vector, b.vector, atol=1e-6)
    z = sk.itrfcoord.from_geodetic(latitude_deg=10.0, longitude_deg=20.0)
    assert z.height == pytest.approx(0.0, abs=1e-6)
    # exactly 90 degrees is accepted
    sk.itrfcoord.from_geodetic(latitude_deg=90, longitude_deg=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(latitude_deg=3),
        dict(longitude_deg=3),
        dict(),
        dict(latitude_deg=3, latitude_rad=0.1, longitude_deg=4),
        dict(latitude_deg=3, longitude_deg=4, longitude_rad=0.1),
        dict(latitude_deg="a", longitude_deg=4),
        dict(latitude_deg=3, longitude_deg=4, height="x"),
        dict(latitude_deg=3, longitude_deg=4, altitude=5.0),  # not a keyword here
    ],
)
def test_itrfcoord_from_geodetic_type_errors(kwargs):
    with pytest.raises(TypeError):
        sk.itrfcoord.from_geodetic(**kwargs)


def test_itrfcoord_from_geodetic_is_keyword_only():
    with pytest.raises(TypeError):
        sk.itrfcoord.from_geodetic(42.0, -71.0)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(latitude_deg=95, longitude_deg=4),
        dict(latitude_rad=1.6, longitude_rad=0),
        dict(latitude_deg=math.nan, longitude_deg=4),
        dict(latitude_deg=3, longitude_deg=math.inf),
        dict(latitude_deg=3, longitude_deg=4, height=math.nan),
    ],
)
def test_itrfcoord_from_geodetic_value_errors(kwargs):
    with pytest.raises(ValueError):
        sk.itrfcoord.from_geodetic(**kwargs)


def test_itrfcoord_from_vector():
    v = [1523128.63570828, -4461395.28873207, 4281865.94218203]
    c = sk.itrfcoord.from_vector(v)
    assert c == sk.itrfcoord(v)
    assert sk.itrfcoord.from_vector(np.array(v)) == c
    assert sk.itrfcoord.from_vector(c.vector) == c
    assert sk.itrfcoord.from_vector((1, 2, 3)) == sk.itrfcoord(1.0, 2.0, 3.0)
    with pytest.raises(TypeError):
        sk.itrfcoord.from_vector("abc")
    with pytest.raises(ValueError):
        sk.itrfcoord.from_vector([1.0, 2.0])
    with pytest.raises(ValueError):
        sk.itrfcoord.from_vector([1.0, math.nan, 3.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(latitude_deg="a", longitude_deg=4),
        dict(latitude_deg=3, longitude_deg=[1, 2]),
        dict(latitude_rad=None, longitude_rad="b", latitude_deg=1),
        dict(latitude_deg=3, longitude_deg=4, altitude="x"),
        dict(latitude_deg=3, longitude_deg=4, height=object()),
    ],
)
def test_itrfcoord_keyword_wrong_type_is_type_error(kwargs):
    with pytest.raises(TypeError, match="must be a real number"):
        sk.itrfcoord(**kwargs)


def test_itrfcoord_move_with_heading_drops_height():
    start = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=5000.0)
    dest = start.move_with_heading(10e3, 0.0)
    assert dest.height == pytest.approx(0.0, abs=1e-6)


# --- quaternion ---------------------------------------------------------------


def _neg(q):
    return sk.quaternion(-q.w, -q.x, -q.y, -q.z)


def test_quaternion_angle_to():
    q1 = sk.quaternion.rotz(math.radians(10))
    q2 = sk.quaternion.rotz(math.radians(40))
    assert q1.angle_to(q2) == pytest.approx(math.radians(30), abs=1e-14)
    assert q2.angle_to(q1) == pytest.approx(math.radians(30), abs=1e-14)
    assert q1.angle_to(q1) == 0.0
    assert q1.angle_to(_neg(q1)) == pytest.approx(0.0, abs=1e-15)
    # always in [0, pi]: 350 degrees apart is 10 degrees the other way
    q3 = sk.quaternion.rotx(math.radians(350))
    assert sk.quaternion().angle_to(q3) == pytest.approx(math.radians(10), abs=1e-13)
    assert sk.quaternion().angle_to(sk.quaternion.roty(math.pi)) == pytest.approx(math.pi)
    # small angles keep full precision
    assert sk.quaternion().angle_to(sk.quaternion.rotx(1e-10)) == pytest.approx(1e-10, rel=1e-9)


def test_quaternion_isclose():
    q = sk.quaternion.from_axis_angle([1, 2, 3], 0.7)
    assert q.isclose(q)
    assert q.isclose(_neg(q))
    assert not q == _neg(q)
    other = sk.quaternion.from_axis_angle([1, 2, 3], 0.7 + 1e-6)
    assert not q.isclose(other)
    assert q.isclose(other, atol=2e-6)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            q.isclose(q, atol=bad)
    with pytest.raises(TypeError):
        q.isclose(q, 1e-3)  # type: ignore[misc]


def test_quaternion_to_axis_angle_round_trip():
    q = sk.quaternion.from_axis_angle([0, 0, 2], 0.5)
    axis, angle = q.to_axis_angle()
    assert np.allclose(axis, [0, 0, 1])
    assert angle == pytest.approx(0.5)
    assert np.array_equal(axis, q.axis)
    assert angle == q.angle
    assert sk.quaternion.from_axis_angle(*q.to_axis_angle()).isclose(q, atol=1e-14)
    axis, angle = sk.quaternion().to_axis_angle()
    assert np.array_equal(axis, [1.0, 0.0, 0.0])
    assert angle == 0.0


def test_quaternion_to_from_array():
    q = sk.quaternion.from_axis_angle([1, -2, 0.5], 1.2)
    arr = q.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.tolist() == [q.w, q.x, q.y, q.z]
    # renormalized on the way back in, so equal to rounding
    back = sk.quaternion.from_array(arr)
    assert np.allclose(back.to_array(), arr, rtol=0, atol=1e-15)
    assert back.isclose(q)
    assert back == sk.quaternion(*arr)
    assert sk.quaternion().to_array().tolist() == [1.0, 0.0, 0.0, 0.0]
    # same normalization as the constructor
    assert sk.quaternion.from_array([2, 0, 0, 0]) == sk.quaternion(2, 0, 0, 0)
    assert sk.quaternion.from_array((1, 1, 1, 1)) == sk.quaternion(1, 1, 1, 1)
    with pytest.raises(TypeError):
        sk.quaternion.from_array("abcd")
    for bad in ([1, 0, 0], [[1, 0, 0, 0]], [0, 0, 0, 0], [1, math.nan, 0, 0]):
        with pytest.raises(ValueError):
            sk.quaternion.from_array(bad)


# --- kepler -------------------------------------------------------------------


def test_kepler_eccentricity_alias():
    k = sk.kepler(7.0e6, 0.01, 0.5, 0.1, 0.2, 0.3)
    assert k.eccentricity == k.eccen == 0.01
    k.eccentricity = 0.02
    assert k.eccen == 0.02
    k.eccen = 0.03
    assert k.eccentricity == 0.03
    with pytest.raises(ValueError):
        k.eccentricity = 1.5
    assert k.eccentricity == 0.03
    assert k.inclination == 0.5


def test_kepler_true_anomaly_setter():
    k = sk.kepler(7.0e6, 0.01, 0.5, 0.1, 0.2, 0.3)
    k.true_anomaly = 1.25
    assert k.nu == 1.25
    assert k.true_anomaly == 1.25
    with pytest.raises(ValueError):
        k.true_anomaly = math.nan
    assert k.nu == 1.25


def test_kepler_propagate_duration_types():
    k = sk.kepler(7.0e6, 0.01, 0.5, 0.1, 0.2, 0.3)
    ref = k.propagate(sk.duration.from_seconds(600))
    assert k.propagate(600) == ref
    assert k.propagate(600.0) == ref
    assert k.propagate(datetime.timedelta(minutes=10)) == ref
    with pytest.raises(TypeError):
        k.propagate("600")


# --- lambert ------------------------------------------------------------------


def test_lambert_tof_duration_types():
    r1 = [7000e3, 0, 0]
    r2 = [0, 7000e3, 0]
    ref = sk.lambert(r1, r2, 3600.0)
    for tof in (3600, sk.duration.from_hours(1), datetime.timedelta(hours=1)):
        sol = sk.lambert(r1, r2, tof)
        assert len(sol) == len(ref)
        assert np.array_equal(sol[0][0], ref[0][0])
        assert np.array_equal(sol[0][1], ref[0][1])
    with pytest.raises(TypeError):
        sk.lambert(r1, r2, "3600")
    with pytest.raises(ValueError):
        sk.lambert(r1, r2, datetime.timedelta(hours=-1))
    # mu and prograde stay positional-or-keyword
    a = sk.lambert(r1, r2, 3600.0, sk.consts.mu_earth, False)
    b = sk.lambert(r1, r2, 3600.0, mu=sk.consts.mu_earth, prograde=False)
    assert np.array_equal(a[0][0], b[0][0])
