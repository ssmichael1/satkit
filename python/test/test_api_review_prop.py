"""Regression tests for the propagation part of the Python API review:
``propagate``, ``propresult``, ``propsettings``, ``satstate``, ``thrust``,
``ecomparams`` and ``satproperties``."""

import copy
import pickle

import numpy as np
import pytest

import satkit as sk

T0 = sk.time(2024, 1, 1, 12, 0, 0)
STATE = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
POS, VEL = STATE[:3], STATE[3:]


def secs(s):
    return sk.duration(seconds=s)


class TestCanInterp:
    """``can_interp`` was True whenever a Runge-Kutta solution was kept, with
    or without dense output, and ``interp`` disagreed with it."""

    def test_true_with_dense_output(self):
        r = sk.propagate(STATE, T0, duration_secs=600)
        assert r.can_interp
        assert "Can Interp: true" in str(r)
        assert r.interp(T0 + secs(300)).shape == (6,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_interp": False},
            {"integrator": sk.integrator.rodas4},
            {"integrator": sk.integrator.rkv98_nointerp},
            {"integrator": sk.integrator.rkv98_nointerp, "enable_interp": False},
            {"integrator": sk.integrator.gauss_jackson8, "enable_interp": False},
        ],
        ids=["no-interp", "rodas4", "rkv98_nointerp", "rkv98_nointerp-off", "gj8-off"],
    )
    def test_false_without_dense_output(self, kwargs):
        r = sk.propagate(STATE, T0, duration_secs=600, propsettings=sk.propsettings(**kwargs))
        assert not r.can_interp
        assert "Can Interp: false" in str(r)
        # ... and interp agrees, at every time including the begin time
        for t in (T0 + secs(300), T0, [T0, T0 + secs(300)], []):
            with pytest.raises(ValueError, match="no dense output"):
                r.interp(t)

    def test_gauss_jackson_dense_output(self):
        ps = sk.propsettings(integrator=sk.integrator.gauss_jackson8)
        r = sk.propagate(STATE, T0, duration_secs=600, propsettings=ps)
        assert r.can_interp
        assert r.interp(T0 + secs(300)).shape == (6,)

    def test_zero_duration_is_consistent(self):
        r = sk.propagate(STATE, T0, duration_secs=0)
        assert not r.can_interp
        assert "Can Interp: false" in str(r)
        with pytest.raises(ValueError, match="no dense output"):
            r.interp(T0)
        with pytest.raises(ValueError, match="no dense output"):
            r.interp([T0])
        np.testing.assert_array_equal(r.state, STATE)

    def test_rkv98_nointerp_same_cost_as_rkv98_without_interp(self):
        ps = sk.propsettings(enable_interp=False)
        a = sk.propagate(STATE, T0, duration_secs=6000, propsettings=ps)
        ps.integrator = sk.integrator.rkv98_nointerp
        b = sk.propagate(STATE, T0, duration_secs=6000, propsettings=ps)
        assert a.stats.num_eval == b.stats.num_eval
        np.testing.assert_array_equal(a.state, b.state)


class TestGravityOrderFollowsDegree:
    """``ps.gravity_degree = 20`` used to leave the order at 4."""

    def test_setter_raises_order_with_degree(self):
        ps = sk.propsettings()
        ps.gravity_degree = 20
        assert (ps.gravity_degree, ps.gravity_order) == (20, 20)
        ps.gravity_degree = 8
        assert (ps.gravity_degree, ps.gravity_order) == (8, 8)

    def test_constructor_default_follows(self):
        ps = sk.propsettings(gravity_degree=10)
        assert ps.gravity_order == 10
        ps.gravity_degree = 30
        assert ps.gravity_order == 30

    def test_explicit_order_in_constructor_is_kept(self):
        ps = sk.propsettings(gravity_degree=10, gravity_order=6)
        ps.gravity_degree = 30
        assert (ps.gravity_degree, ps.gravity_order) == (30, 6)
        # ... lowered only to stay valid
        ps.gravity_degree = 4
        assert (ps.gravity_degree, ps.gravity_order) == (4, 4)
        ps.gravity_degree = 30
        assert ps.gravity_order == 4

    def test_assigned_order_is_kept(self):
        ps = sk.propsettings()
        ps.gravity_degree = 12
        ps.gravity_order = 2
        ps.gravity_degree = 20
        assert (ps.gravity_degree, ps.gravity_order) == (20, 2)

    def test_same_field_as_the_constructor(self):
        a = sk.propsettings()
        a.gravity_degree = 20
        b = sk.propsettings(gravity_degree=20, gravity_order=20)
        ra = sk.propagate(STATE, T0, duration_secs=3000, propsettings=a)
        rb = sk.propagate(STATE, T0, duration_secs=3000, propsettings=b)
        np.testing.assert_array_equal(ra.state, rb.state)
        c = sk.propsettings(gravity_degree=20, gravity_order=4)
        rc = sk.propagate(STATE, T0, duration_secs=3000, propsettings=c)
        assert np.linalg.norm(ra.pos - rc.pos) > 1e-3

    @pytest.mark.parametrize("explicit", [False, True])
    def test_survives_pickle_and_copy(self, explicit):
        ps = sk.propsettings(gravity_degree=6, gravity_order=6 if explicit else None)
        for dup in (pickle.loads(pickle.dumps(ps)), copy.deepcopy(ps), copy.copy(ps)):
            dup.gravity_degree = 20
            assert dup.gravity_order == (6 if explicit else 20)


    def test_reads_pickle_state_without_the_flag(self):
        # satkit <= 0.24.3 pickled the settings bytes alone
        for order, follows in ((6, True), (3, False)):
            old_state = sk.propsettings(gravity_degree=6, gravity_order=order).__getstate__()[0]
            assert isinstance(old_state, bytes)
            ps = sk.propsettings()
            ps.__setstate__(old_state)
            assert (ps.gravity_degree, ps.gravity_order) == (6, order)
            ps.gravity_degree = 20
            assert ps.gravity_order == (20 if follows else order)


def sym_cov():
    """A symmetric 6x6 whose off-diagonal 3x3 blocks are not symmetric and
    whose every element is distinct, so a transposed block would show"""
    a = np.arange(36, dtype=float).reshape(6, 6) + 1.0
    return a @ a.T + np.diag([1e4, 2e4, 3e4, 4.0, 5.0, 6.0])


class TestCovariance:
    """The getter read the column-major storage as row-major."""

    def test_getter_returns_what_was_set(self):
        cov = sym_cov()
        assert not np.array_equal(cov[:3, 3:], cov[:3, 3:].T)
        sat = sk.satstate(T0, POS, VEL)
        sat.cov = cov
        np.testing.assert_array_equal(sat.cov, cov)
        np.testing.assert_array_equal(sat.cov[:3, 3:], cov[:3, 3:])
        np.testing.assert_array_equal(sat.cov[3:, :3], cov[3:, :3])
        np.testing.assert_array_equal(sk.satstate(T0, POS, VEL, cov).cov, cov)

    def test_roundoff_asymmetry_is_kept_exactly(self):
        # Asymmetric at round-off level only: accepted, and returned element
        # for element (the old getter returned the transpose)
        cov = sym_cov()
        cov[0, 4] *= 1 + 1e-13
        cov[5, 2] *= 1 - 1e-13
        assert not np.array_equal(cov, cov.T)
        sat = sk.satstate(T0, POS, VEL)
        sat.cov = cov
        np.testing.assert_array_equal(sat.cov, cov)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(sat)).cov, cov)

    def test_non_contiguous_input(self):
        cov = sym_cov()
        sat = sk.satstate(T0, POS, VEL)
        sat.cov = np.asfortranarray(cov)
        np.testing.assert_array_equal(sat.cov, cov)

    def test_propagated_cross_blocks(self):
        # Phi P Phi^T with the result's own Phi, block for block
        cov = np.diag([100.0, 400.0, 900.0, 0.01, 0.04, 0.09])
        sat = sk.satstate(T0, POS, VEL, cov)
        new = sat.propagate(secs(600))
        phi = sk.propagate(STATE, T0, duration_secs=600, output_phi=True).phi
        expected = phi @ cov @ phi.T
        np.testing.assert_allclose(new.cov, expected, rtol=1e-9, atol=1e-12)
        assert not np.allclose(expected[:3, 3:], expected[:3, 3:].T, rtol=1e-3)

    @pytest.mark.parametrize("where", ["setter", "constructor"])
    def test_rejects_non_symmetric(self, where):
        cov = sym_cov()
        cov[1, 4] += 1.0
        with pytest.raises(ValueError, match="symmetric"):
            if where == "setter":
                sk.satstate(T0, POS, VEL).cov = cov
            else:
                sk.satstate(T0, POS, VEL, cov)

    def test_rejects_nan(self):
        cov = sym_cov()
        cov[2, 3] = cov[3, 2] = np.nan
        with pytest.raises(ValueError, match="symmetric"):
            sk.satstate(T0, POS, VEL).cov = cov

    def test_none_clears(self):
        sat = sk.satstate(T0, POS, VEL, sym_cov())
        assert sat.cov is not None
        sat.cov = None
        assert sat.cov is None
        assert sat.propagate(secs(60)).cov is None


class TestPropagateArguments:
    def test_pos_without_vel(self):
        with pytest.raises(TypeError, match="`pos` and `vel` together"):
            sk.propagate(begin=T0, pos=POS, duration_secs=60)
        with pytest.raises(TypeError, match="`pos` and `vel` together"):
            sk.propagate(begin=T0, vel=VEL, duration_secs=60)
        with pytest.raises(TypeError, match="`pos` and `vel` together"):
            sk.propagate(STATE, T0, pos=POS, duration_secs=60)

    def test_pos_and_vel(self):
        a = sk.propagate(begin=T0, pos=POS, vel=VEL, duration_secs=60)
        b = sk.propagate(STATE, T0, duration_secs=60)
        np.testing.assert_array_equal(a.state, b.state)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"end": T0 + secs(60), "duration_secs": 60},
            {"duration": secs(60), "duration_secs": 60},
            {"duration_days": 0.1, "duration_secs": 60},
            {"end": T0 + secs(60), "duration": secs(60)},
            {"end": T0 + secs(60), "duration": secs(1), "duration_secs": 2, "duration_days": 3},
        ],
    )
    def test_several_end_times(self, kwargs):
        with pytest.raises(TypeError, match="one end time"):
            sk.propagate(STATE, T0, **kwargs)

    def test_each_end_time_alone(self):
        ref = sk.propagate(STATE, T0, T0 + secs(864)).state
        for kwargs in (
            {"end": T0 + secs(864)},
            {"duration": secs(864)},
            {"duration_secs": 864},
            {"duration_days": 0.01},
        ):
            np.testing.assert_array_equal(sk.propagate(STATE, T0, **kwargs).state, ref)

    def test_no_end_time(self):
        with pytest.raises(TypeError, match="requires an end time"):
            sk.propagate(STATE, T0)


class TestManeuvers:
    @pytest.mark.parametrize("frame", ["ITRF", "TEME", "CIRS", "TIRS", "EME2000", "ICRF"])
    def test_unsupported_frame_refused_at_once(self, frame):
        sat = sk.satstate(T0, POS, VEL)
        with pytest.raises(ValueError, match="unsupported frame for maneuver"):
            sat.add_maneuver(T0 + secs(60), [0, 1, 0], getattr(sk.frame, frame))
        assert sat.num_maneuvers == 0
        # the state is still usable and picklable
        assert pickle.loads(pickle.dumps(sat)).num_maneuvers == 0
        sat.propagate(secs(120))

    @pytest.mark.parametrize("frame", ["GCRF", "RTN", "NTW", "LVLH", "RIC", "RSW"])
    def test_supported_frames(self, frame):
        sat = sk.satstate(T0, POS, VEL)
        sat.add_maneuver(T0 + secs(60), [0, 1, 0], getattr(sk.frame, frame))
        assert sat.num_maneuvers == 1

    def test_clear_maneuvers(self):
        sat = sk.satstate(T0, POS, VEL)
        plain = sat.propagate(secs(600))
        sat.add_prograde(T0 + secs(60), 10.0)
        sat.add_maneuver(T0 + secs(120), [1, 0, 0], sk.frame.RTN)
        assert sat.num_maneuvers == 2
        assert np.linalg.norm(sat.propagate(secs(600)).pos - plain.pos) > 100
        assert sat.clear_maneuvers() is None
        assert sat.num_maneuvers == 0 and sat.maneuvers == []
        np.testing.assert_array_equal(sat.propagate(secs(600)).pos, plain.pos)
        sat.clear_maneuvers()  # harmless when empty

    def test_maneuver_at_start_applied_at_target_not(self):
        plain = sk.satstate(T0, POS, VEL).propagate(secs(100))
        at_start = sk.satstate(T0, POS, VEL)
        at_start.add_prograde(T0, 10.0)
        assert np.linalg.norm(at_start.propagate(secs(100)).vel - plain.vel) > 9
        at_target = sk.satstate(T0, POS, VEL)
        at_target.add_prograde(T0 + secs(100), 10.0)
        np.testing.assert_array_equal(at_target.propagate(secs(100)).vel, plain.vel)


class TestThrustConstant:
    def test_end_before_start(self):
        with pytest.raises(ValueError, match="before start"):
            sk.thrust.constant([1e-4, 0, 0], T0 + secs(60), T0, sk.frame.NTW)

    def test_zero_length_arc_allowed(self):
        th = sk.thrust.constant([1e-4, 0, 0], T0, T0, sk.frame.NTW)
        assert th.start == th.end

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_accel(self, bad):
        with pytest.raises(ValueError, match="finite"):
            sk.thrust.constant([1e-4, bad, 0], T0, T0 + secs(60), sk.frame.NTW)


class TestEcomFromDict:
    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown ecomparams key 'D0'"):
            sk.ecomparams.from_dict({"D0": -1e-7})
        with pytest.raises(ValueError, match="unknown ecomparams key"):
            sk.ecomparams.from_dict({"d0": -1e-7, "b1c": 1e-9})
        with pytest.raises(ValueError, match="unknown ecomparams key 3"):
            sk.ecomparams.from_dict({3: 1.0})

    def test_missing_keys_default(self):
        e = sk.ecomparams.from_dict({"d0": -1e-7})
        assert e.d0 == -1e-7 and e.y0 == 0.0 and e.sun_relative is False
        assert sk.ecomparams.from_dict({}) == sk.ecomparams()

    def test_round_trip(self):
        e = sk.ecomparams.ecom2(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9)
        assert sk.ecomparams.from_dict(e.to_dict()) == e


class TestPropResultConstruction:
    def test_no_public_constructor(self):
        with pytest.raises(TypeError):
            sk.propresult()

    @pytest.mark.parametrize("output_phi", [False, True])
    def test_pickle_and_copy(self, output_phi):
        r = sk.propagate(STATE, T0, duration_secs=600, output_phi=output_phi)
        for dup in (pickle.loads(pickle.dumps(r)), copy.deepcopy(r), copy.copy(r)):
            assert dup.time_begin == r.time_begin and dup.time_end == r.time_end
            np.testing.assert_array_equal(dup.state, r.state)
            assert dup.can_interp
            np.testing.assert_array_equal(dup.interp(T0 + secs(300)), r.interp(T0 + secs(300)))
            assert dup.stats.num_eval == r.stats.num_eval
            if output_phi:
                np.testing.assert_array_equal(dup.phi, r.phi)
            else:
                assert dup.phi is None
