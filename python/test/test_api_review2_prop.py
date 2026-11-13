"""Tests for the propagation part of the Python API review, round 2:
``propagate``, ``propresult``, ``propsettings``, ``satstate``, ``thrust``
and ``satproperties``.

``TestRound1*`` pin the exception types and validation of round 1 (the
"Breaking (prop)" changes), which shipped without dedicated tests."""

import datetime
import math

import numpy as np
import pytest

import satkit as sk

T0 = sk.time(2024, 1, 1, 12, 0, 0)
STATE = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
POS, VEL = STATE[:3], STATE[3:]
UNSUPPORTED_FRAMES = [sk.frame.ITRF, sk.frame.TEME, sk.frame.CIRS, sk.frame.EME2000]


def secs(s):
    return sk.duration(seconds=s)


class TestRound1ThrustAccel:
    def test_accel_is_ndarray(self):
        t = sk.thrust.constant([0.0, 1e-4, 0.0], T0, T0 + secs(60), sk.frame.RTN)
        a = t.accel
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.float64
        assert a.shape == (3,)
        np.testing.assert_array_equal(a, [0.0, 1e-4, 0.0])

    def test_accel_is_a_new_array(self):
        t = sk.thrust.constant([0.0, 1e-4, 0.0], T0, T0 + secs(60), sk.frame.RTN)
        t.accel[1] = 5.0
        assert t.accel[1] == 1e-4


class TestRound1PropagateTypeErrors:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"begin": "2024-01-01", "duration_secs": 60.0},
            {"begin": 60.0, "duration_secs": 60.0},
            {"begin": T0, "end": "tomorrow"},
            {"begin": T0, "end": 60.0},
            {"begin": T0, "duration": "1 hour"},
            {"begin": T0, "duration_secs": "60"},
            {"begin": T0, "duration_days": "1"},
            {"begin": T0, "duration_secs": 60.0, "output_phi": "yes"},
            {"begin": T0, "duration_secs": 60.0, "propsettings": {"abs_error": 1e-8}},
            {"begin": T0, "duration_secs": 60.0, "propsettings": sk.satproperties()},
            {"begin": T0, "duration_secs": 60.0, "satproperties": {"cdaoverm": 0.01}},
            {"begin": T0, "duration_secs": 60.0, "satproperties": sk.propsettings()},
        ],
        ids=[
            "begin-str",
            "begin-float",
            "end-str",
            "end-float",
            "duration-str",
            "duration_secs-str",
            "duration_days-str",
            "output_phi-str",
            "propsettings-dict",
            "propsettings-satproperties",
            "satproperties-dict",
            "satproperties-propsettings",
        ],
    )
    def test_wrong_type(self, kwargs):
        with pytest.raises(TypeError):
            sk.propagate(STATE, **kwargs)

    def test_non_numeric_state(self):
        with pytest.raises(TypeError):
            sk.propagate(["a", "b", "c", "d", "e", "f"], T0, duration_secs=60.0)


class TestRound1SatstatePropagateTypeErrors:
    @pytest.mark.parametrize("bad", ["2024-01-01", None, [T0], object()])
    def test_timedur_wrong_type(self, bad):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(TypeError):
            s.propagate(bad)

    def test_propsettings_wrong_type(self):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(TypeError):
            s.propagate(T0 + secs(60), propsettings=sk.satproperties())

    def test_satproperties_wrong_type(self):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(TypeError):
            s.propagate(T0 + secs(60), satproperties=sk.propsettings())


class TestRound1SatpropertiesTypeErrors:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cdaoverm": "0.01"},
            {"craoverm": "0.01"},
            {"ecom": "ecom"},
            {"ecom": sk.propsettings()},
        ],
        ids=["cdaoverm", "craoverm", "ecom-str", "ecom-propsettings"],
    )
    def test_wrong_type(self, kwargs):
        with pytest.raises(TypeError):
            sk.satproperties(**kwargs)


class TestRound1PropsettingsTypeErrors:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gravity_model": "egm96"},
            {"gravity_model": sk.integrator.rkv98},
            {"integrator": "rkv98"},
            {"integrator": sk.tidemodel.none},
            {"tide_model": "none"},
            {"tide_model": sk.gravmodel.egm96},
        ],
        ids=[
            "gravity_model-str",
            "gravity_model-integrator",
            "integrator-str",
            "integrator-tidemodel",
            "tide_model-str",
            "tide_model-gravmodel",
        ],
    )
    def test_wrong_type(self, kwargs):
        with pytest.raises(TypeError):
            sk.propsettings(**kwargs)


class TestRound1ShapeErrors:
    @pytest.mark.parametrize("state", [STATE[:5], np.zeros(7), np.zeros((2, 3))])
    def test_propagate_state_shape(self, state):
        with pytest.raises(ValueError, match="6-element"):
            sk.propagate(state, T0, duration_secs=60.0)

    @pytest.mark.parametrize("which", ["pos", "vel"])
    @pytest.mark.parametrize("bad", [np.zeros(2), np.zeros(4)])
    def test_propagate_pos_vel_shape(self, which, bad):
        kw = {"pos": POS, "vel": VEL}
        kw[which] = bad
        with pytest.raises(ValueError):
            sk.propagate(begin=T0, duration_secs=60.0, **kw)

    @pytest.mark.parametrize("which", ["pos", "vel"])
    @pytest.mark.parametrize("bad", [np.zeros(2), np.zeros(4), [1.0, 2.0]])
    def test_satstate_pos_vel_shape(self, which, bad):
        args = {"pos": POS, "vel": VEL}
        args[which] = bad
        with pytest.raises(ValueError):
            sk.satstate(T0, args["pos"], args["vel"])

    @pytest.mark.parametrize("bad", [np.eye(5), np.eye(7), np.zeros(36), np.zeros((6, 7))])
    def test_satstate_cov_shape(self, bad):
        with pytest.raises(ValueError, match="6x6"):
            sk.satstate(T0, POS, VEL, cov=bad)
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(ValueError, match="6x6"):
            s.cov = bad

    @pytest.mark.parametrize("method", ["set_pos_uncertainty", "set_vel_uncertainty"])
    @pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    def test_uncertainty_sigma_shape(self, method, bad):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(ValueError):
            getattr(s, method)(bad, sk.frame.RTN)

    @pytest.mark.parametrize("method", ["set_pos_uncertainty", "set_vel_uncertainty"])
    def test_uncertainty_sigma_not_numeric(self, method):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(TypeError):
            getattr(s, method)(["a", "b", "c"], sk.frame.RTN)


class TestRound1UnsupportedFrames:
    @pytest.mark.parametrize("frame", UNSUPPORTED_FRAMES, ids=str)
    def test_thrust_constant(self, frame):
        with pytest.raises(ValueError, match="unsupported frame"):
            sk.thrust.constant([0.0, 1e-4, 0.0], T0, T0 + secs(60), frame)

    @pytest.mark.parametrize("method", ["set_pos_uncertainty", "set_vel_uncertainty"])
    @pytest.mark.parametrize("frame", UNSUPPORTED_FRAMES, ids=str)
    def test_uncertainty(self, method, frame):
        s = sk.satstate(T0, POS, VEL)
        with pytest.raises(ValueError, match="unsupported frame"):
            getattr(s, method)([1.0, 2.0, 3.0], frame)
        assert s.cov is None

    @pytest.mark.parametrize("frame", [sk.frame.GCRF, sk.frame.RTN, sk.frame.NTW, sk.frame.LVLH], ids=str)
    def test_supported(self, frame):
        sk.thrust.constant([0.0, 1e-4, 0.0], T0, T0 + secs(60), frame)
        s = sk.satstate(T0, POS, VEL)
        s.set_pos_uncertainty([1.0, 2.0, 3.0], frame)
        s.set_vel_uncertainty([0.1, 0.2, 0.3], frame)
        assert s.cov.shape == (6, 6)


class TestRound1Values:
    @pytest.mark.parametrize(
        "step", [0.0, -60.0, secs(0), secs(-60), datetime.timedelta(0), math.nan, math.inf]
    )
    def test_precompute_terms_bad_step(self, step):
        ps = sk.propsettings()
        with pytest.raises(ValueError):
            ps.precompute_terms(T0, T0 + secs(3600), step)

    @pytest.mark.parametrize("name", ["abs_error", "rel_error"])
    @pytest.mark.parametrize("bad", [-1e-8, math.nan])
    def test_tolerance(self, name, bad):
        with pytest.raises(ValueError, match=name):
            sk.propsettings(**{name: bad})
        ps = sk.propsettings()
        with pytest.raises(ValueError, match=name):
            setattr(ps, name, bad)
        assert getattr(ps, name) == 1e-8

    @pytest.mark.parametrize("name", ["abs_error", "rel_error"])
    def test_tolerance_zero_allowed(self, name):
        ps = sk.propsettings(**{name: 0.0})
        assert getattr(ps, name) == 0.0

    @pytest.mark.parametrize("bad", [-0.01, math.nan])
    def test_cdaoverm(self, bad):
        with pytest.raises(ValueError, match="cdaoverm"):
            sk.satproperties(cdaoverm=bad)
        p = sk.satproperties(cdaoverm=0.01)
        with pytest.raises(ValueError, match="cdaoverm"):
            p.cdaoverm = bad
        assert p.cdaoverm == 0.01

    def test_cdaoverm_zero_allowed(self):
        assert sk.satproperties(cdaoverm=0.0).cdaoverm == 0.0

    def test_propagate_zero_position(self):
        with pytest.raises(ValueError, match="all zeros"):
            sk.propagate(np.array([0.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]), T0, duration_secs=60.0)
        with pytest.raises(ValueError, match="all zeros"):
            sk.propagate(pos=np.zeros(3), vel=VEL, begin=T0, duration_secs=60.0)

    def test_satstate_propagate_zero_position(self):
        s = sk.satstate(T0, np.zeros(3), VEL)
        with pytest.raises(ValueError, match="all zeros"):
            s.propagate(T0 + secs(60))
        # Propagating to its own time does nothing, so it is allowed
        s2 = s.propagate(T0)
        np.testing.assert_array_equal(s2.pos, np.zeros(3))


class TestDurationConvention:
    """``propagate(duration=)`` and ``satstate.propagate(timedur)`` take the
    same durations: a satkit.duration, a datetime.timedelta or seconds."""

    @pytest.mark.parametrize(
        "dur",
        [
            secs(600.5),
            datetime.timedelta(seconds=600, microseconds=500000),
            600.5,
            np.float64(600.5),
        ],
        ids=["duration", "timedelta", "float", "np.float64"],
    )
    def test_propagate_duration(self, dur):
        ref = sk.propagate(STATE, T0, duration_secs=600.5)
        r = sk.propagate(STATE, T0, duration=dur)
        assert r.time_end == ref.time_end
        np.testing.assert_array_equal(r.state, ref.state)

    def test_propagate_duration_int_seconds(self):
        r = sk.propagate(STATE, T0, duration=600)
        assert r.time_end == T0 + secs(600)

    def test_timedelta_is_exact(self):
        td = datetime.timedelta(days=1, seconds=7, microseconds=3)
        r = sk.propagate(STATE, T0, duration=td, propsettings=sk.propsettings(enable_interp=False))
        assert (r.time_end - T0).microseconds == 86_407_000_003
        s = sk.satstate(T0, POS, VEL).propagate(td)
        assert (s.time - T0).microseconds == 86_407_000_003

    def test_negative_timedelta(self):
        r = sk.propagate(STATE, T0, duration=datetime.timedelta(seconds=-300))
        assert r.time_end == T0 - secs(300)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_seconds(self, bad):
        with pytest.raises(ValueError, match="finite"):
            sk.propagate(STATE, T0, duration=bad)
        with pytest.raises(ValueError, match="finite"):
            sk.satstate(T0, POS, VEL).propagate(bad)

    @pytest.mark.parametrize("bad", ["600", True, [600.0], T0])
    def test_propagate_duration_wrong_type(self, bad):
        with pytest.raises(TypeError, match="duration"):
            sk.propagate(STATE, T0, duration=bad)

    def test_duration_none_is_not_given(self):
        r = sk.propagate(STATE, T0, duration=None, duration_secs=60.0)
        assert r.time_end == T0 + secs(60)

    @pytest.mark.parametrize(
        "dur",
        [secs(600.5), datetime.timedelta(seconds=600.5), 600.5, np.float64(600.5)],
        ids=["duration", "timedelta", "float", "np.float64"],
    )
    def test_satstate_propagate(self, dur):
        sat = sk.satstate(T0, POS, VEL)
        ref = sat.propagate(T0 + secs(600.5))
        s = sat.propagate(dur)
        assert s.time == ref.time
        np.testing.assert_array_equal(s.pos, ref.pos)
        np.testing.assert_array_equal(s.vel, ref.vel)

    def test_satstate_int_seconds(self):
        s = sk.satstate(T0, POS, VEL).propagate(60)
        assert s.time == T0 + secs(60)

    def test_satstate_bool_refused(self):
        with pytest.raises(TypeError, match="timedur"):
            sk.satstate(T0, POS, VEL).propagate(True)

    def test_satstate_and_propagate_agree(self):
        s = sk.satstate(T0, POS, VEL).propagate(900.0)
        r = sk.propagate(STATE, T0, duration=900.0)
        assert s.time == r.time_end
        np.testing.assert_array_equal(s.pos, r.pos)


class TestToSatstate:
    def test_end_state(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0)
        s = r.to_satstate()
        assert isinstance(s, sk.satstate)
        assert s.time == r.time_end
        np.testing.assert_array_equal(s.pos, r.pos)
        np.testing.assert_array_equal(s.vel, r.vel)
        assert s.cov is None
        assert s.maneuvers == []

    def test_with_phi(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0, output_phi=True)
        s = r.to_satstate()
        np.testing.assert_array_equal(s.pos, r.pos)
        assert s.cov is None

    def test_continue(self):
        r = sk.propagate(STATE, T0, duration_secs=600.0)
        s = r.to_satstate().propagate(600.0)
        assert s.time == T0 + secs(1200)

    def test_backward(self):
        r = sk.propagate(STATE, T0, duration_secs=-600.0)
        assert r.to_satstate().time == T0 - secs(600)

    def test_is_a_method(self):
        r = sk.propagate(STATE, T0, duration_secs=60.0)
        assert callable(r.to_satstate)


class TestStateWithPosVel:
    def test_all_three_refused(self):
        with pytest.raises(TypeError, match="not both"):
            sk.propagate(STATE, T0, pos=POS, vel=VEL, duration_secs=60.0)

    def test_state_with_one_refused(self):
        with pytest.raises(TypeError):
            sk.propagate(STATE, T0, pos=POS, duration_secs=60.0)
        with pytest.raises(TypeError):
            sk.propagate(STATE, T0, vel=VEL, duration_secs=60.0)


class TestCraoverm:
    @pytest.mark.parametrize("bad", [-0.01, math.nan, -math.inf])
    def test_negative_or_nan(self, bad):
        with pytest.raises(ValueError, match="craoverm"):
            sk.satproperties(craoverm=bad)
        p = sk.satproperties(craoverm=0.02)
        with pytest.raises(ValueError, match="craoverm"):
            p.craoverm = bad
        assert p.craoverm == 0.02

    def test_zero_allowed(self):
        p = sk.satproperties(craoverm=0.0)
        p.craoverm = 0.0
        assert p.craoverm == 0.0


class TestManeuversCopy:
    def test_copy_and_rebuild(self):
        s = sk.satstate(T0, POS, VEL)
        s.add_maneuver(T0 + secs(60), [0.0, 1.0, 0.0], sk.frame.RTN)
        s.add_prograde(T0 + secs(120), 2.0)
        ms = s.maneuvers
        ms.clear()
        assert s.num_maneuvers == 2
        # The documented way to drop one: clear and add back the rest
        kept = s.maneuvers[1:]
        s.clear_maneuvers()
        for m in kept:
            s.add_maneuver(**m)
        assert s.num_maneuvers == 1
        assert s.maneuvers[0]["time"] == T0 + secs(120)
        assert s.maneuvers[0]["frame"] == sk.frame.NTW


class TestInterpTimeTypes:
    def setup_method(self):
        self.r = sk.propagate(STATE, T0, duration_secs=600.0)

    def test_tuple_is_a_list_of_times(self):
        out = self.r.interp((T0 + secs(100), T0 + secs(200)))
        assert out.shape == (2, 6)
        np.testing.assert_array_equal(out, self.r.interp([T0 + secs(100), T0 + secs(200)]))

    @pytest.mark.parametrize("bad", [60.0, 5, "2024-01-01", None, np.array([1.0]), [1.0]])
    def test_type_error(self, bad):
        with pytest.raises(TypeError):
            self.r.interp(bad)
