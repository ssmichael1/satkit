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
