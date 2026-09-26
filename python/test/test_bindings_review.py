"""Regression tests for the Python-bindings review fixes: thread safety of
``sgp4``, stub/runtime agreement, exception types, pickling and edge cases."""

import copy
import pickle
import threading

import numpy as np
import pytest

import satkit as sk
from shared import ISS_2024

T0 = sk.time(2024, 1, 1, 12, 0, 0)
STATE = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])


def iss():
    return sk.TLE.from_lines(ISS_2024)[0]


class TestSgp4Threads:
    """``sgp4`` used to hold a mutable borrow of the TLE while the GIL was
    released, so a second thread using the same TLE panicked
    (``PanicException``, not caught by ``except Exception``) or got
    ``ValueError: Invalid TLE: Already borrowed``."""

    def test_concurrent_calls_on_one_tle(self):
        tle = iss()
        big = [tle.epoch + i * 1e-3 for i in range(50_000)]
        want_big = sk.sgp4(iss(), big)[0]
        want_one = sk.sgp4([iss()], tle.epoch)[0]
        errors = []
        results = []

        def single():
            for _ in range(3):
                try:
                    results.append(("big", sk.sgp4(tle, big)[0]))
                except BaseException as e:  # noqa: BLE001 - PanicException too
                    errors.append(repr(e))

        def listpath():
            for _ in range(100):
                try:
                    results.append(("one", sk.sgp4([tle], tle.epoch)[0]))
                except BaseException as e:  # noqa: BLE001
                    errors.append(repr(e))

        def reader():
            for _ in range(2000):
                try:
                    _ = tle.epoch, tle.eccen
                except BaseException as e:  # noqa: BLE001
                    errors.append(repr(e))

        threads = [threading.Thread(target=f) for f in (single, single, listpath, reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        assert not any(t.is_alive() for t in threads)
        assert errors == []
        assert len(results) == 3 + 3 + 100
        for kind, p in results:
            np.testing.assert_array_equal(p, want_big if kind == "big" else want_one)

    def test_tle_unchanged_and_reusable(self):
        tle = iss()
        before = tle.to_2line()
        p1 = sk.sgp4(tle, T0)[0]
        p2 = sk.sgp4([tle], T0)[0][0]
        np.testing.assert_array_equal(p1, p2)
        assert tle.to_2line() == before


class TestSgp4Shapes:
    def test_one_element_tle_list_keeps_its_axis(self):
        tle = iss()
        p, v = sk.sgp4([tle], T0)
        assert p.shape == v.shape == (1, 3)
        np.testing.assert_array_equal(p[0], sk.sgp4(tle, T0)[0])
        p, v, e = sk.sgp4([tle], [T0, T0 + 0.1], errflag=True)
        assert p.shape == (1, 2, 3)
        assert e.shape == (1, 2)
        p, _, e = sk.sgp4([tle], T0, errflag=True)
        assert e.shape == (1,)

    def test_empty_inputs_give_empty_arrays(self):
        tle = iss()
        assert sk.sgp4(tle, [])[0].shape == (0, 3)
        assert sk.sgp4([tle], [])[0].shape == (1, 0, 3)
        assert sk.sgp4([tle, tle], [])[0].shape == (2, 0, 3)
        assert sk.sgp4([], T0)[0].shape == (0, 3)
        assert sk.sgp4([], [T0])[0].shape == (0, 1, 3)
        _, _, e = sk.sgp4([tle], [], errflag=True)
        assert e.shape == (1, 0)


class TestStubRuntimeAgreement:
    def test_rotation_list_gives_list(self):
        for fn in (sk.frametransform.rotation, sk.frametransform.rotation_approx):
            out = fn(sk.frame.ITRF, sk.frame.GCRF, [T0, T0 + 0.1])
            assert isinstance(out, list) and len(out) == 2
            assert all(isinstance(q, sk.quaternion) for q in out)
            assert isinstance(fn(sk.frame.ITRF, sk.frame.GCRF, T0), sk.quaternion)

    def test_interp_list_is_one_array(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0)
        out = r.interp([T0 + 0.001, T0 + 0.002])
        assert isinstance(out, np.ndarray) and out.shape == (2, 6)
        np.testing.assert_array_equal(out[1], r.interp(T0 + 0.002))
        assert r.interp([]).shape == (0, 6)

    def test_interp_output_phi_without_stm_raises(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0)
        with pytest.raises(ValueError, match="state transition matrix"):
            r.interp(T0 + 0.001, output_phi=True)
        with pytest.raises(ValueError, match="state transition matrix"):
            r.interp([T0 + 0.001], output_phi=True)

    def test_interp_output_phi_with_stm(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0, output_phi=True)
        s, phi = r.interp(T0 + 0.001, output_phi=True)
        assert s.shape == (6,) and phi.shape == (6, 6)
        out = r.interp([T0 + 0.001, T0 + 0.002], output_phi=True)
        assert isinstance(out, list) and len(out) == 2
        assert all(isinstance(x, tuple) and len(x) == 2 for x in out)

    def test_time_minus_list_of_times(self):
        t1 = T0 + 1.5
        out = t1 - [T0, T0 + 1]
        assert isinstance(out, np.ndarray) and out.dtype == object and out.shape == (2,)
        assert out[0] == t1 - T0 and out[1] == t1 - (T0 + 1)
        assert all(isinstance(d, sk.duration) for d in out)

    def test_time_minus_float_array_is_object_array_of_times(self):
        out = T0 - np.array([0.5, 1.0])
        assert out.dtype == object
        assert out[1] == T0 - 1.0

    @pytest.mark.parametrize(
        "pos",
        [[7e6, 0, 0], (7e6, 0, 0), np.array([7000000, 0, 0]), np.array([7e6, 0, 0], dtype=np.float32)],
    )
    def test_gravity_array_likes(self, pos):
        want = sk.gravity(np.array([7e6, 0.0, 0.0]))
        np.testing.assert_allclose(sk.gravity(pos), want, rtol=1e-6)
        g, p = sk.gravity_and_partials(pos)
        assert g.shape == (3,) and p.shape == (3, 3)

    def test_gravity_bad_input(self):
        with pytest.raises(ValueError):
            sk.gravity(np.array([7e6, 0.0]))
        with pytest.raises(TypeError):
            sk.gravity("abc")

    def test_batch_state_transform_integer_arrays(self):
        pi = np.array([[7000000, 0, 0], [0, 7000000, 0]])
        vi = np.array([[0, 7500, 0], [-7500, 0, 0]])
        times = [T0, T0 + 0.01]
        pf, vf = sk.frametransform.itrf_to_gcrf_state(pi.astype(float), vi.astype(float), times)
        p, v = sk.frametransform.itrf_to_gcrf_state(pi, vi, times)
        np.testing.assert_array_equal(p, pf)
        np.testing.assert_array_equal(v, vf)
        # Nested lists are (N, 3) array-likes too
        p, _ = sk.frametransform.gcrf_to_itrf_state(pi.tolist(), vi.tolist(), times)
        assert p.shape == (2, 3)


class TestExceptionTypes:
    q = sk.quaternion.rotz(0.3)

    @pytest.mark.parametrize("rhs", [2.0, 2, np.float64(2.0), "abc", None])
    def test_quaternion_times_non_vector_is_type_error(self, rhs):
        with pytest.raises(TypeError):
            self.q * rhs

    @pytest.mark.parametrize("rhs", [np.array([1.0, 2, 3, 4]), [1.0, 2.0], np.ones((2, 4)), np.array(2.0)])
    def test_quaternion_times_wrong_shape_is_value_error(self, rhs):
        with pytest.raises(ValueError):
            self.q * rhs

    def test_quaternion_times_vector_still_works(self):
        np.testing.assert_allclose(self.q * [1.0, 0, 0], self.q * np.array([1.0, 0, 0]))
        assert (self.q * np.zeros((0, 3))).shape == (0, 3)

    def test_vector_length_errors_are_value_errors(self):
        with pytest.raises(ValueError):
            sk.quaternion.rotation_between(np.array([1.0, 0]), np.array([0, 1.0, 0]))
        with pytest.raises(ValueError):
            sk.quaternion.from_axis_angle(np.array([0, 0, 1.0, 0]), 0.5)

    def test_satstate_propagate_non_time_is_type_error(self):
        st = sk.satstate(T0, STATE[:3], STATE[3:])
        with pytest.raises(TypeError):
            st.propagate(1.0)

    def test_duration_invalid_operands(self):
        d = sk.duration(seconds=1)
        with pytest.raises(TypeError):
            d + 1.0
        with pytest.raises(TypeError):
            d / "x"
        with pytest.raises(ZeroDivisionError):
            d / 0
        with pytest.raises(ZeroDivisionError):
            d / sk.duration(seconds=0)


class TestPickleNewTypes:
    def test_geodetic(self):
        g = sk.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=123.4).geodetic
        for g2 in (pickle.loads(pickle.dumps(g)), copy.deepcopy(g), copy.copy(g)):
            assert isinstance(g2, type(g))
            assert (g2.latitude_rad, g2.longitude_rad, g2.height_m) == (
                g.latitude_rad,
                g.longitude_rad,
                g.height_m,
            )

    def test_propstats(self):
        s = sk.propagate(STATE, T0, duration_secs=600.0).stats
        for s2 in (pickle.loads(pickle.dumps(s)), copy.deepcopy(s)):
            assert isinstance(s2, type(s))
            assert (s2.num_eval, s2.num_accept, s2.num_reject) == (
                s.num_eval,
                s.num_accept,
                s.num_reject,
            )
            assert s2.num_eval > 0


class TestEdgeCases:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_time_plus_non_finite_days(self, bad):
        with pytest.raises(ValueError, match="finite"):
            T0 + bad
        with pytest.raises(ValueError, match="finite"):
            T0 - bad
        with pytest.raises(ValueError, match="finite"):
            T0 + [1.0, bad]
        with pytest.raises(ValueError, match="finite"):
            T0 - np.array([bad])

    def test_time_plus_huge_days_overflows(self):
        with pytest.raises(OverflowError):
            T0 + 1e300

    @pytest.mark.parametrize("kw", ["days", "hours", "minutes", "seconds"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_duration_non_finite(self, kw, bad):
        with pytest.raises(ValueError, match="finite"):
            sk.duration(**{kw: bad})
        with pytest.raises(ValueError, match="finite"):
            getattr(sk.duration, f"from_{kw}")(bad)

    def test_duration_finite_still_works(self):
        assert sk.duration(days=1.5).seconds == 1.5 * 86400
        assert sk.duration.from_milliseconds(2.5).microseconds == 2500
        assert (T0 + 0.5) - T0 == sk.duration(hours=12)

    def test_propagate_without_end_time(self):
        with pytest.raises(TypeError, match="end time"):
            sk.propagate(STATE, T0)
        with pytest.raises(ValueError, match="finite"):
            sk.propagate(STATE, T0, duration_secs=float("nan"))

    def test_tle_epoch_rejects_list(self):
        tle = iss()
        with pytest.raises(TypeError):
            tle.epoch = [T0 + 1, T0 + 2]
        assert tle.epoch == iss().epoch
        tle.epoch = T0 + 1
        assert tle.epoch == T0 + 1
