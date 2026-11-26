"""Python subclasses of ``satproperties``: user-defined drag, radiation
pressure, ECOM and additional acceleration in the propagator (#298)."""

import copy
import pickle

import numpy as np
import pytest

import satkit as sk

T0 = sk.time(2024, 3, 1)
R0 = sk.consts.earth_radius + 450e3
V0 = np.sqrt(sk.consts.mu_earth / R0)
STATE = np.array([R0, 0.0, 0.0, 0.0, 0.6 * V0, 0.8 * V0])
GEO = np.array([sk.consts.geo_r, 0, 0, 0, np.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
DUR = 1800.0


def prop(state, props, **kw):
    return sk.propagate(state, T0, duration_secs=DUR, satproperties=props, **kw)


class Calls(sk.satproperties):
    """Records the arguments of every hook"""

    def __init__(self):
        self.seen = {}

    def _note(self, name, *args):
        self.seen.setdefault(name, []).append(args)

    def cd_a_over_m(self, time, state):
        self._note("cd", time, state)
        return 0.01

    def cr_a_over_m(self, time, state):
        self._note("cr", time, state)
        return 0.02

    def srp_ecom(self, time, state):
        self._note("ecom", time, state)
        return None

    def thrust_accel(self, time, pos, vel):
        self._note("thrust", time, pos, vel)
        return None


class ThrustNTW(sk.satproperties):
    def __init__(self, accel):
        self.accel = np.asarray(accel, dtype=float)

    def thrust_accel(self, time, pos, vel):
        return sk.frametransform.to_gcrf(sk.frame.NTW, pos, vel) @ self.accel


class TestArguments:
    def test_hooks_called_with_time_and_state(self):
        p = Calls()
        prop(STATE, p)
        assert set(p.seen) == {"cd", "cr", "ecom", "thrust"}
        for name in ("cd", "cr", "ecom"):
            time, state = p.seen[name][0]
            assert isinstance(time, sk.time)
            assert isinstance(state, np.ndarray) and state.shape == (6,)
        time, pos, vel = p.seen["thrust"][0]
        assert isinstance(time, sk.time)
        assert pos.shape == (3,) and vel.shape == (3,)
        # Every evaluation lies on (or within a step of) the trajectory
        times = [(t - T0).seconds for t, *_ in p.seen["thrust"]]
        assert min(times) >= -DUR and max(times) <= 2 * DUR
        radii = [np.linalg.norm(pos) for _, pos, _ in p.seen["thrust"]]
        assert np.allclose(radii, R0, rtol=0.05)


class TestEquivalence:
    """An override that returns what the built-in model computes gives the
    built-in result."""

    def test_thrust_matches_constant_arc(self):
        accel = [0.0, 1e-4, 0.0]
        arc = sk.thrust.constant(accel, T0, T0 + sk.duration(seconds=DUR), frame=sk.frame.NTW)
        ref = prop(STATE, sk.satproperties(thrusts=[arc]))
        res = prop(STATE, ThrustNTW(accel))
        assert res.state == pytest.approx(ref.state, rel=1e-12, abs=1e-6)
        # ... and the thrust did something
        assert np.linalg.norm(res.state[:3] - prop(STATE, None).state[:3]) > 10

    def test_drag_matches_constant(self):
        class Drag(sk.satproperties):
            def cd_a_over_m(self, time, state):
                return 0.02

        ref = prop(STATE, sk.satproperties(cdaoverm=0.02))
        res = prop(STATE, Drag())
        np.testing.assert_array_equal(res.state, ref.state)
        assert np.linalg.norm(ref.state[:3] - prop(STATE, None).state[:3]) > 1e-3

    def test_srp_matches_constant(self):
        class Srp(sk.satproperties):
            def cr_a_over_m(self, time, state):
                return 0.05

        ref = prop(GEO, sk.satproperties(craoverm=0.05))
        res = prop(GEO, Srp())
        np.testing.assert_array_equal(res.state, ref.state)
        assert np.linalg.norm(ref.state[:3] - prop(GEO, None).state[:3]) > 1e-3

    def test_ecom_matches_constant(self):
        ecom = sk.ecomparams.reduced(-1e-7, 1e-9, -3e-9, 1e-9, 0.0)

        class Ecom(sk.satproperties):
            def srp_ecom(self, time, state):
                return ecom

        ref = prop(GEO, sk.satproperties(ecom=ecom))
        res = prop(GEO, Ecom())
        np.testing.assert_array_equal(res.state, ref.state)
        assert np.linalg.norm(ref.state[:3] - prop(GEO, None).state[:3]) > 1e-3

    def test_super_composes(self):
        arc = sk.thrust.constant([0, 1e-4, 0], T0, T0 + sk.duration(seconds=DUR), frame=sk.frame.NTW)

        class Doubled(sk.satproperties):
            def cd_a_over_m(self, time, state):
                return 2 * super().cd_a_over_m(time, state)

            def thrust_accel(self, time, pos, vel):
                return 2 * super().thrust_accel(time, pos, vel)

        res = prop(STATE, Doubled(cdaoverm=0.01, thrusts=[arc]))
        arc2 = sk.thrust.constant([0, 2e-4, 0], T0, T0 + sk.duration(seconds=DUR), frame=sk.frame.NTW)
        ref = prop(STATE, sk.satproperties(cdaoverm=0.02, thrusts=[arc2]))
        assert res.state == pytest.approx(ref.state, rel=1e-12, abs=1e-6)

    def test_state_dependent_force(self):
        """A force no built-in can express: acceleration along the velocity
        raises the orbit, against it lowers it."""

        class Along(sk.satproperties):
            def __init__(self, a):
                self.a = a

            def thrust_accel(self, time, pos, vel):
                return self.a * vel / np.linalg.norm(vel)

        def sma(res):
            return sk.kepler.from_pv(res.state[:3], res.state[3:]).a

        base = sma(prop(STATE, None))
        assert sma(prop(STATE, Along(1e-3))) > base + 1e3
        assert sma(prop(STATE, Along(-1e-3))) < base - 1e3

    def test_satstate_propagate(self):
        accel = [0.0, 1e-4, 0.0]
        arc = sk.thrust.constant(accel, T0, T0 + sk.duration(seconds=DUR), frame=sk.frame.NTW)
        sat = sk.satstate(T0, STATE[:3], STATE[3:])
        ref = sat.propagate(DUR, satproperties=sk.satproperties(thrusts=[arc]))
        res = sat.propagate(DUR, satproperties=ThrustNTW(accel))
        assert res.pos == pytest.approx(ref.pos, rel=1e-12, abs=1e-6)

    def test_with_state_transition_matrix(self):
        res = prop(STATE, ThrustNTW([0, 1e-4, 0]), output_phi=True)
        assert res.phi.shape == (6, 6)

    def test_gauss_jackson(self):
        settings = sk.propsettings(integrator=sk.integrator.gauss_jackson8, gj_step_seconds=30.0)
        res = prop(STATE, ThrustNTW([0, 1e-4, 0]), propsettings=settings)
        ref = prop(STATE, ThrustNTW([0, 1e-4, 0]))
        assert res.state[:3] == pytest.approx(ref.state[:3], abs=1e-2)


class TestNoOverride:
    def test_subclass_without_overrides_uses_stored_values(self):
        class Plain(sk.satproperties):
            pass

        ref = prop(STATE, sk.satproperties(cdaoverm=0.02))
        np.testing.assert_array_equal(prop(STATE, Plain(cdaoverm=0.02)).state, ref.state)

    def test_base_methods(self):
        arc = sk.thrust.constant([1e-4, 0, 0], T0, T0 + sk.duration(seconds=60), frame=sk.frame.GCRF)
        ecom = sk.ecomparams(d0=-1e-7)
        p = sk.satproperties(cdaoverm=0.01, craoverm=0.02, thrusts=[arc], ecom=ecom)
        assert p.cd_a_over_m(T0, STATE) == 0.01
        assert p.cr_a_over_m(T0, STATE) == 0.02
        assert p.srp_ecom(T0, STATE) == ecom
        assert sk.satproperties().srp_ecom(T0, STATE) is None
        np.testing.assert_array_equal(p.thrust_accel(T0, STATE[:3], STATE[3:]), [1e-4, 0, 0])
        assert p.thrust_accel(T0 + sk.duration(seconds=61), STATE[:3], STATE[3:]) is None
        assert sk.satproperties().thrust_accel(T0, STATE[:3], STATE[3:]) is None


class TestErrors:
    def test_exception_propagates(self):
        class Boom(sk.satproperties):
            def thrust_accel(self, time, pos, vel):
                raise KeyError("no such engine")

        with pytest.raises(KeyError, match="no such engine"):
            prop(STATE, Boom())
        sat = sk.satstate(T0, STATE[:3], STATE[3:])
        with pytest.raises(KeyError, match="no such engine"):
            sat.propagate(DUR, satproperties=Boom())

    def test_first_exception_is_kept_and_calls_stop(self):
        class Once(sk.satproperties):
            def __init__(self):
                self.calls = 0

            def thrust_accel(self, time, pos, vel):
                self.calls += 1
                raise RuntimeError(f"call {self.calls}")

        p = Once()
        with pytest.raises(RuntimeError, match="call 1$"):
            prop(STATE, p)
        assert p.calls == 1

    def test_instance_can_be_reused_after_an_error(self):
        class Flaky(sk.satproperties):
            def __init__(self):
                self.fail = True

            def thrust_accel(self, time, pos, vel):
                if self.fail:
                    raise ValueError("flaky")
                return None

        p = Flaky()
        with pytest.raises(ValueError, match="flaky"):
            prop(STATE, p)
        p.fail = False
        np.testing.assert_array_equal(prop(STATE, p).state, prop(STATE, None).state)

    @pytest.mark.parametrize(
        "value, exc",
        [
            ("big", TypeError),
            (None, TypeError),
            (float("nan"), ValueError),
            (-0.01, ValueError),
        ],
    )
    @pytest.mark.parametrize("hook", ["cd_a_over_m", "cr_a_over_m"])
    def test_bad_coefficient(self, hook, value, exc):
        cls = type("Bad", (sk.satproperties,), {hook: lambda self, time, state: value})
        with pytest.raises(exc, match=hook):
            prop(STATE, cls())

    @pytest.mark.parametrize(
        "value, exc",
        [
            ([1e-4, 0.0], ValueError),
            (1e-4, ValueError),
            ([np.nan, 0.0, 0.0], ValueError),
            ([np.inf, 0.0, 0.0], ValueError),
            ("abc", TypeError),
        ],
    )
    def test_bad_acceleration(self, value, exc):
        cls = type("Bad", (sk.satproperties,), {"thrust_accel": lambda self, t, p, v: value})
        with pytest.raises(exc, match="thrust_accel"):
            prop(STATE, cls())

    def test_bad_ecom(self):
        cls = type("Bad", (sk.satproperties,), {"srp_ecom": lambda self, t, s: 1.0})
        with pytest.raises(TypeError, match="srp_ecom"):
            prop(GEO, cls())

    def test_wrong_type_still_refused(self):
        with pytest.raises(TypeError, match="satproperties must be a satkit.satproperties"):
            prop(STATE, {"cdaoverm": 0.01})


class TestConstructor:
    def test_own_init_arguments(self):
        class Sail(sk.satproperties):
            def __init__(self, area, mass, *, reflectivity=1.8):
                self.craoverm = reflectivity * area / mass

        assert Sail(10.0, 100.0).craoverm == pytest.approx(0.18)
        assert Sail(10.0, mass=100.0, reflectivity=1.0).craoverm == pytest.approx(0.1)

    def test_own_init_keeps_base_keywords(self):
        class Named(sk.satproperties):
            def __init__(self, name, **kwargs):
                self.name = name

        p = Named("probe", cdaoverm=0.03)
        assert (p.name, p.cdaoverm) == ("probe", 0.03)
        with pytest.raises(ValueError):
            Named("probe", cdaoverm=-1.0)

    def test_without_init_is_as_strict_as_the_base(self):
        class Plain(sk.satproperties):
            pass

        assert Plain(craoverm=0.02).craoverm == 0.02
        for cls in (Plain, sk.satproperties):
            with pytest.raises(TypeError, match="keyword arguments only"):
                cls(0.01)
            with pytest.raises(TypeError, match="unexpected keyword argument 'cdoverm'"):
                cls(cdoverm=0.01)


class Picklable(sk.satproperties):
    """Module level, so that pickle can find the class"""

    def __init__(self, gain):
        self.gain = gain

    def thrust_accel(self, time, pos, vel):
        return self.gain * vel / np.linalg.norm(vel)


class TestPickle:
    @pytest.mark.parametrize("dup", [lambda p: pickle.loads(pickle.dumps(p)), copy.copy, copy.deepcopy])
    def test_subclass_keeps_attributes(self, dup):
        p = Picklable(1e-4)
        p.cdaoverm = 0.01
        q = dup(p)
        assert type(q) is Picklable
        assert (q.gain, q.cdaoverm) == (1e-4, 0.01)
        np.testing.assert_array_equal(prop(STATE, q).state, prop(STATE, p).state)

    def test_base_state_is_still_bytes(self):
        p = sk.satproperties(cdaoverm=0.01)
        assert isinstance(p.__getstate__(), bytes)
        assert pickle.loads(pickle.dumps(p)) == p
