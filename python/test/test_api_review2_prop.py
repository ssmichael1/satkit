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
