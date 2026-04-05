import pytest
import numpy as np
import math as m
import os
import pickle
from sp3file import read_sp3file

import satkit as sk


class TestHighPrecisionPropagation:

    def test_interp(self):
        starttime = sk.time(2015, 3, 20, 0, 0, 0)

        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
        stoptime = starttime + sk.duration.from_days(1.0)

        settings = sk.propsettings()
        settings.precompute_terms(starttime, stoptime)

        # Propagate forward
        res1 = sk.propagate(
            np.concatenate((pos, vel)), starttime, end=stoptime, propsettings=settings
        )
        # Propagate backward and see if we recover original result
        res2 = sk.propagate(res1.state, stoptime, end=starttime, propsettings=settings)

        assert res2.state[0:3] == pytest.approx(pos, abs=0.5)
        assert res2.state[3:6] == pytest.approx(vel, abs=1e-5)

        newtime = starttime + sk.duration.from_hours(4.332)
        istate1 = res1.interp(newtime)
        istate2 = res2.interp(newtime)

        assert istate1 == pytest.approx(istate2, rel=1e-7)


    def test_gauss_jackson8(self):
        """Propagate a GEO orbit with Gauss-Jackson 8 and compare against
        the default RKV98 integrator. Also exercise dense-output interpolation
        through the Python bindings.
        """
        starttime = sk.time(2015, 3, 20, 0, 0, 0)
        stoptime = starttime + sk.duration.from_hours(6.0)

        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
        state0 = np.concatenate((pos, vel))

        # Gauss-Jackson 8 with 60-second fixed step
        settings_gj = sk.propsettings(
            integrator=sk.integrator.gauss_jackson8,
            gj_step_seconds=60.0,
        )
        # Default RKV98 reference
        settings_rk = sk.propsettings()

        res_gj = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_gj)
        res_rk = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_rk)

        # Endpoint agreement: sub-meter on a smooth 6-hour GEO arc
        assert res_gj.state[0:3] == pytest.approx(res_rk.state[0:3], abs=1.0)
        assert res_gj.state[3:6] == pytest.approx(res_rk.state[3:6], abs=1e-4)

        # Interpolation should work (quintic Hermite dense output)
        assert res_gj.can_interp is True

        mid = starttime + sk.duration.from_hours(3.7)
        istate_gj = res_gj.interp(mid)
        istate_rk = res_rk.interp(mid)
        # Quintic Hermite is 5th-order while RKV98 dense is 8th-order — a few
        # meters of disagreement is expected at GEO with 60-s steps.
        assert istate_gj[0:3] == pytest.approx(istate_rk[0:3], abs=10.0)

        # Batch interpolation
        times = [starttime + sk.duration.from_hours(h) for h in [1.0, 2.5, 4.0, 5.5]]
        batch_gj = res_gj.interp(times)
        batch_rk = res_rk.interp(times)
        assert batch_gj.shape == (4, 6)
        for i in range(4):
            assert batch_gj[i, 0:3] == pytest.approx(batch_rk[i, 0:3], abs=10.0)

    def test_ntw_prograde_adds_exactly_to_speed(self):
        """An NTW prograde (+T) burn adds its exact magnitude to |v|
        regardless of orbit eccentricity. A RIC in-track (+I) burn of the
        same magnitude does not, because the RIC I axis is perpendicular
        to position, not to velocity. This test exercises the NTW binding
        end-to-end on an eccentric orbit.
        """
        t0 = sk.time(2015, 3, 20, 0, 0, 0)

        # Eccentric orbit at mid-anomaly — non-zero flight-path angle
        a = 8000e3
        e = 0.3
        nu = m.radians(60.0)
        r_mag = a * (1.0 - e * e) / (1.0 + e * m.cos(nu))
        v_mag = m.sqrt(sk.consts.mu_earth * (2.0 / r_mag - 1.0 / a))
        gamma = m.atan(e * m.sin(nu) / (1.0 + e * m.cos(nu)))

        pos = np.array([r_mag, 0.0, 0.0])
        vel = np.array([v_mag * m.sin(gamma), v_mag * m.cos(gamma), 0.0])
        speed_before = np.linalg.norm(vel)

        # NTW +T burn — should add exactly 10 m/s to |v|
        sat_ntw = sk.satstate(time=t0, pos=pos, vel=vel)
        sat_ntw.add_prograde(t0 + sk.duration.from_seconds(1.0), 10.0)
        # Propagate just past the burn
        sat_ntw_after = sat_ntw.propagate(t0 + sk.duration.from_seconds(2.0))
        speed_after_ntw = np.linalg.norm(sat_ntw_after.vel)
        # The burn adds ~10 m/s; 1 second of propagation in the eccentric
        # orbit changes |v| by up to a few m/s due to gravity, so check a
        # loose tolerance — what we really care about is that NTW is closer
        # to +10 than RIC is.
        ntw_delta = speed_after_ntw - speed_before

        # RIC +I burn with the same magnitude
        sat_ric = sk.satstate(time=t0, pos=pos, vel=vel)
        sat_ric.add_maneuver(
            t0 + sk.duration.from_seconds(1.0),
            [0.0, 10.0, 0.0],
            frame=sk.frame.RIC,
        )
        sat_ric_after = sat_ric.propagate(t0 + sk.duration.from_seconds(2.0))
        speed_after_ric = np.linalg.norm(sat_ric_after.vel)
        ric_delta = speed_after_ric - speed_before

        # NTW should give a bigger |v| increase than RIC (by roughly
        # 10·(1-cos γ) ≈ 0.24 m/s for γ ≈ 12.7°).
        assert ntw_delta > ric_delta, (
            f"NTW prograde should add more to |v| than RIC in-track: "
            f"NTW Δ|v| = {ntw_delta:.4f}, RIC Δ|v| = {ric_delta:.4f}"
        )
        # And the difference should be roughly the expected 10·(1-cos γ).
        expected_gap = 10.0 * (1.0 - m.cos(gamma))
        assert abs((ntw_delta - ric_delta) - expected_gap) < 0.05, (
            f"Gap between NTW and RIC Δ|v| should be ≈ {expected_gap:.4f}; "
            f"got {ntw_delta - ric_delta:.4f}"
        )

    def test_lvlh_maneuver(self):
        """LVLH +x burn should give the same trajectory as the equivalent
        RIC +I burn (they are the same axis, just relabeled)."""
        t0 = sk.time(2015, 3, 20, 0, 0, 0)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])

        sat_lvlh = sk.satstate(time=t0, pos=pos, vel=vel)
        sat_ric = sk.satstate(time=t0, pos=pos, vel=vel)

        t_burn = t0 + sk.duration.from_hours(0.5)
        t_end = t0 + sk.duration.from_hours(2.0)

        # LVLH: x = in-track direction
        sat_lvlh.add_maneuver(t_burn, [10.0, 0.0, 0.0], frame=sk.frame.LVLH)
        # RIC: I = in-track direction (same axis)
        sat_ric.add_maneuver(t_burn, [0.0, 10.0, 0.0], frame=sk.frame.RIC)

        s_lvlh = sat_lvlh.propagate(t_end)
        s_ric = sat_ric.propagate(t_end)

        assert s_lvlh.pos == pytest.approx(s_ric.pos, abs=1e-3)
        assert s_lvlh.vel == pytest.approx(s_ric.vel, abs=1e-6)

    def test_maneuver_ergonomic_constructors(self):
        """Smoke-test the add_prograde / add_retrograde / add_radial /
        add_normal helpers — they should all dispatch through the NTW
        path and leave the propagation in a valid state."""
        t0 = sk.time(2015, 3, 20, 0, 0, 0)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])

        sat = sk.satstate(time=t0, pos=pos, vel=vel)
        sat.add_prograde(t0 + sk.duration.from_hours(1.0), 1.0)
        sat.add_retrograde(t0 + sk.duration.from_hours(2.0), 0.5)
        sat.add_radial(t0 + sk.duration.from_hours(3.0), 0.5)
        sat.add_normal(t0 + sk.duration.from_hours(4.0), 0.5)

        assert sat.num_maneuvers == 4

        final = sat.propagate(t0 + sk.duration.from_hours(5.0))
        # Just check the final state is finite and reasonable
        assert np.all(np.isfinite(final.pos))
        assert np.all(np.isfinite(final.vel))
        # Still close to GEO radius
        assert abs(np.linalg.norm(final.pos) - sk.consts.geo_r) < 1e5

    def test_gauss_jackson8_rejects_stm(self):
        """GJ8 should raise when asked to propagate with state-transition
        matrix output (output_phi=True)."""
        starttime = sk.time(2015, 3, 20, 0, 0, 0)
        stoptime = starttime + sk.duration.from_hours(1.0)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
        state0 = np.concatenate((pos, vel))

        settings = sk.propsettings(
            integrator=sk.integrator.gauss_jackson8,
            gj_step_seconds=60.0,
        )
        with pytest.raises(Exception):
            sk.propagate(
                state0, starttime, end=stoptime,
                output_phi=True, propsettings=settings,
            )


    def test_state_transition(self):
        # Test that state transition matrix is computed correctly
        # Define an orbit ... 30 deg inclined at 550km perigee, 1000km apogee
        perigee = sk.consts.earth_radius + 550e3
        apogee = sk.consts.earth_radius + 1000e3
        eccentricity = (apogee - perigee) / (apogee + perigee)
        semimajor_axis = (perigee + apogee) / 2
        k = sk.kepler(semimajor_axis, eccentricity, m.radians(30), 0, 0, 0)

        state0 = np.concatenate((k.to_pv()))
        epoch = sk.time(2025, 1, 1, 0, 0, 0)
        duration = sk.duration(hours=6)

        settings = sk.propsettings()

        # a small perturbation in the initial state, used to test state transition matrix
        dstate0 = [30.3, -5.2, 8.4, 0.01, -0.02, 0.05]

        res0 = sk.propagate(state0, epoch, epoch+duration, output_phi=True, propsettings=settings)
        resd = sk.propagate(state0 + dstate0, epoch, epoch+duration, output_phi=True, propsettings=settings)

        # Check that the state transition matrix correctly maps the initial
        # state perturbation to propagated state perturbation
        assert resd.state_end == pytest.approx(
            res0.state_end + res0.phi @ dstate0, rel=1e-7
        )

        # Check on interpolated state transition
        for x in range(5):
            tinterp = epoch + sk.duration(hours=x * 6.0 / 5.0)
            mstate0, mphi = res0.interp(tinterp, output_phi=True)
            mdstate0 = resd.interp(tinterp)
            assert mdstate0 == pytest.approx(
                mstate0 + mphi @ dstate0, rel=1e-7
            )

    def test_gps(self, testvec_dir):

        # File contains test calculation vectors provided by NASA

        fname = (
            testvec_dir
            + os.path.sep
            + "orbitprop"
            + os.path.sep
            + "ESA0OPSFIN_20233640000_01D_05M_ORB.SP3"
        )

        [pitrf, timearr] = read_sp3file(fname)
        pgcrf = np.stack(
            np.fromiter(
                (q * p for q, p in zip(sk.frametransform.qitrf2gcrf(timearr), pitrf)),  # type: ignore
                list,
            ),  # type: ignore
            axis=0,
        )  # type: ignore
        settings = sk.propsettings()

        # Determined by orbitprop_gps_fit.py
        fitparam = np.array(
            [2.47130562e03, 2.94682753e03, -5.34172176e02, 2.32565692e-02]
        )

        # Values for craoverm and velocity come from orbitprop_gps_fit.py
        satprops = sk.satproperties()
        satprops.craoverm = fitparam[3]  # type: ignore

        res = sk.propagate(
            np.concatenate((pgcrf[0, :], fitparam[0:3])),
            timearr[0],
            end=timearr[-1],
            propsettings=settings,
            satproperties=satprops,
        )

        # See if propagator is accurate to < 8 meters over 1 day on
        # each Cartesian axis
        for iv in range(pgcrf.shape[0] - 5):
            state = res.interp(timearr[iv])
            for ix in range(0, 3):
                assert m.fabs(state[ix] - pgcrf[iv, ix]) < 8


class TestSatState:
    def test_lvlh(self):
        """
        Test rotations of satellite state into the LVLH frame
        """
        time = sk.time(2015, 3, 20, 0, 0, 0)
        satstate = sk.satstate(
            time,
            np.array([sk.consts.geo_r, 0, 0]),
            np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0]),
        )
        state2 = satstate.propagate(time + sk.duration.from_hours(3.5))
        h = np.cross(state2.pos, state2.vel)
        rz = -1.0 / np.linalg.norm(state2.pos) * (state2.qgcrf2lvlh * state2.pos)
        ry = -1.0 / np.linalg.norm(h) * (state2.qgcrf2lvlh * h)  # type: ignore
        rx = 1.0 / np.linalg.norm(state2.vel) * (state2.qgcrf2lvlh * state2.vel)

        # Since p & v are not quite orthoginal, we allow for more tolerance
        # on this one (v is not exactly along xhat)
        assert np.array([1.0, 0.0, 0.0]) == pytest.approx(rx, abs=1.0e-4)
        # Two tests below should be exact
        assert np.array([0.0, 1.0, 0.0]) == pytest.approx(ry, abs=1e-10)
        assert np.array([0.0, 0.0, 1.0]) == pytest.approx(rz, abs=1e-10)

    def test_satstate_pickle(self):
        """Test that satstate pickle round-trips all fields including maneuvers"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t_burn = t0 + sk.duration.from_hours(1)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RIC)
        sat.add_maneuver(t_burn + sk.duration.from_hours(1), [5, 0, 0], frame=sk.frame.GCRF)

        restored = pickle.loads(pickle.dumps(sat))

        assert restored.time == sat.time
        assert np.allclose(restored.pos, sat.pos)
        assert np.allclose(restored.vel, sat.vel)
        assert restored.num_maneuvers == 2

    def test_uncertainty_frames(self):
        """The unified set_pos_uncertainty / set_vel_uncertainty API should
        accept GCRF, LVLH, RIC, NTW frames and preserve the other block
        when called in sequence."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state0 = (np.array([r, 0, 0]), np.array([0, v, 0]))

        # Every supported frame should succeed
        for frm in [sk.frame.GCRF, sk.frame.LVLH, sk.frame.RIC, sk.frame.NTW]:
            sat = sk.satstate(time=t0, pos=state0[0], vel=state0[1])
            sat.set_pos_uncertainty(np.array([10.0, 20.0, 30.0]), frame=frm)
            assert sat.cov is not None
            pos_trace = sat.cov[0, 0] + sat.cov[1, 1] + sat.cov[2, 2]
            # Trace is frame-invariant for diagonal input
            assert abs(pos_trace - (100 + 400 + 900)) / 1400 < 1e-12

        # Calling pos then vel should preserve both blocks
        sat = sk.satstate(time=t0, pos=state0[0], vel=state0[1])
        sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)
        sat.set_vel_uncertainty(np.array([0.1, 0.2, 0.05]), frame=sk.frame.LVLH)
        # Position block trace should be preserved
        pos_trace = sat.cov[0, 0] + sat.cov[1, 1] + sat.cov[2, 2]
        vel_trace = sat.cov[3, 3] + sat.cov[4, 4] + sat.cov[5, 5]
        assert abs(pos_trace - (10000 + 40000 + 2500)) / 52500 < 1e-12
        assert abs(vel_trace - (0.01 + 0.04 + 0.0025)) / 0.0525 < 1e-12

        # The frame argument is required — calling without it raises
        sat_missing = sk.satstate(time=t0, pos=state0[0], vel=state0[1])
        with pytest.raises(TypeError):
            sat_missing.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]))

    def test_uncertainty_rejects_unsupported_frame(self):
        """Frames that aren't valid for uncertainty (ITRF, TEME, etc.)
        should raise."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))

        for bad in [sk.frame.ITRF, sk.frame.TEME, sk.frame.ICRF]:
            with pytest.raises(Exception):
                sat.set_pos_uncertainty(np.array([1.0, 1.0, 1.0]), frame=bad)

    def test_satstate_pickle_with_cov(self):
        """Test that satstate pickle round-trips covariance and maneuvers together"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)
        sat.add_maneuver(t0 + sk.duration.from_hours(1), [0, 5, 0], frame=sk.frame.RIC)

        restored = pickle.loads(pickle.dumps(sat))

        assert restored.time == sat.time
        assert np.allclose(restored.pos, sat.pos)
        assert np.allclose(restored.vel, sat.vel)
        assert restored.cov is not None
        assert np.allclose(restored.cov, sat.cov)
        assert restored.num_maneuvers == 1


class TestSatPropertiesPickle:
    def test_satproperties_pickle_basic(self):
        """Test that satproperties pickle round-trips drag/SRP coefficients"""
        props = sk.satproperties(craoverm=0.02, cdaoverm=0.01)
        restored = pickle.loads(pickle.dumps(props))
        assert restored.craoverm == pytest.approx(0.02)
        assert restored.cdaoverm == pytest.approx(0.01)

    def test_satproperties_pickle_with_thrust(self):
        """Test that satproperties pickle round-trips thrust arcs"""
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        t2 = t1 + sk.duration.from_hours(1)

        thrust1 = sk.thrust.constant([1e-4, 2e-4, 3e-4], t0, t1, frame=sk.frame.RIC)
        thrust2 = sk.thrust.constant([0, 0, 5e-3], t1, t2, frame=sk.frame.GCRF)
        props = sk.satproperties(cdaoverm=0.01, thrusts=[thrust1, thrust2])

        restored = pickle.loads(pickle.dumps(props))

        assert restored.cdaoverm == pytest.approx(0.01)
        assert len(restored.thrusts) == 2
        assert restored.thrusts[0].frame == sk.frame.RIC
        assert restored.thrusts[0].accel == [pytest.approx(1e-4), pytest.approx(2e-4), pytest.approx(3e-4)]
        assert restored.thrusts[1].frame == sk.frame.GCRF
        assert restored.thrusts[1].accel == [pytest.approx(0), pytest.approx(0), pytest.approx(5e-3)]


class TestThrust:
    def test_continuous_thrust_ric(self):
        """Test in-track thrust in RIC frame raises orbit"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t1 = t0 + sk.duration.from_hours(2)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state = np.array([r, 0, 0, 0, v, 0])

        res_no = sk.propagate(state, t0, end=t1)

        # In-track thrust in RIC: [radial, in-track, cross-track]
        thrust = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RIC)
        props = sk.satproperties(thrusts=[thrust])
        res_th = sk.propagate(state, t0, end=t1, satproperties=props)

        r_no = np.linalg.norm(res_no.pos)
        r_th = np.linalg.norm(res_th.pos)
        assert r_th > r_no, "In-track thrust should raise orbit"
        assert r_th - r_no > 100, "Thrust effect should be > 100 m"

    def test_continuous_thrust_gcrf(self):
        """Test +Z thrust in GCRF increases Z position"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t1 = t0 + sk.duration.from_minutes(10)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state = np.array([r, 0, 0, 0, v, 0])

        res_no = sk.propagate(state, t0, end=t1)
        thrust = sk.thrust.constant([0, 0, 1e-3], t0, t1, frame=sk.frame.GCRF)
        props = sk.satproperties(thrusts=[thrust])
        res_th = sk.propagate(state, t0, end=t1, satproperties=props)

        assert res_th.state[2] > res_no.state[2], "+Z thrust should increase Z position"

    def test_thrust_properties(self):
        """Test thrust object properties"""
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        thrust = sk.thrust.constant([1e-4, 2e-4, 3e-4], t0, t1, frame=sk.frame.RIC)

        assert thrust.frame == sk.frame.RIC
        assert thrust.accel == [pytest.approx(1e-4), pytest.approx(2e-4), pytest.approx(3e-4)]

    def test_multiple_thrust_arcs(self):
        """Test multiple thrust arcs"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t1 = t0 + sk.duration.from_hours(1)
        t2 = t1 + sk.duration.from_hours(1)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state = np.array([r, 0, 0, 0, v, 0])

        thrust1 = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RIC)
        thrust2 = sk.thrust.constant([0, 1e-4, 0], t1, t2, frame=sk.frame.RIC)
        props = sk.satproperties(thrusts=[thrust1, thrust2])
        assert len(props.thrusts) == 2

        res = sk.propagate(state, t0, end=t2, satproperties=props)
        res_no = sk.propagate(state, t0, end=t2)

        # Two hours of thrust should have a bigger effect than no thrust
        pos_diff = np.linalg.norm(res.pos - res_no.pos)
        assert pos_diff > 1000, f"Two thrust arcs should produce large effect: {pos_diff} m"


class TestImpulsiveManeuver:
    def test_impulsive_gcrf(self):
        """Test impulsive maneuver in GCRF raises orbit"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t_burn = t0 + sk.duration.from_hours(1)
        t_end = t0 + sk.duration.from_hours(3)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        sat_no = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))

        sat.add_maneuver(t_burn, [0, 0, 10], frame=sk.frame.GCRF)
        assert sat.num_maneuvers == 1

        result = sat.propagate(t_end)
        result_no = sat_no.propagate(t_end)

        pos_diff = np.linalg.norm(result.pos - result_no.pos)
        assert pos_diff > 100, f"Maneuver should change position: {pos_diff} m"
        assert result.num_maneuvers == 1, "Maneuvers should persist"

    def test_impulsive_ric(self):
        """Test in-track impulsive maneuver in RIC frame"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t_burn = t0 + sk.duration.from_hours(1)
        t_end = t0 + sk.duration.from_hours(3)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        sat_no = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))

        # 10 m/s in-track in RIC [radial, in-track, cross-track]
        sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RIC)
        result = sat.propagate(t_end)
        result_no = sat_no.propagate(t_end)

        pos_diff = np.linalg.norm(result.pos - result_no.pos)
        assert pos_diff > 10000, f"10 m/s prograde should produce large effect: {pos_diff} m"

    def test_backward_propagation(self):
        """Test forward then backward propagation recovers original state"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t_burn = t0 + sk.duration.from_hours(1)
        t_end = t0 + sk.duration.from_hours(2)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        sat.add_maneuver(t_burn, [0, 0, 5], frame=sk.frame.GCRF)

        fwd = sat.propagate(t_end)
        back = fwd.propagate(t0)

        assert np.linalg.norm(sat.pos - back.pos) < 1.0, "Should recover original position"
        assert np.linalg.norm(sat.vel - back.vel) < 0.01, "Should recover original velocity"

    def test_ric_to_gcrf(self):
        """Test frametransform.ric_to_gcrf and gcrf_to_ric"""
        pos = np.array([6878e3, 0, 0])
        vel = np.array([0, 7612, 0])

        dcm = sk.frametransform.ric_to_gcrf(pos, vel)
        assert dcm.shape == (3, 3)

        dcm_inv = sk.frametransform.gcrf_to_ric(pos, vel)
        assert np.allclose(dcm @ dcm_inv, np.eye(3), atol=1e-10)


class TestLambert:
    """Tests for the Lambert solver"""

    def test_90deg_transfer(self):
        """90-degree prograde transfer at constant radius"""
        r1 = np.array([7000e3, 0, 0])
        r2 = np.array([0, 7000e3, 0])
        period = 2 * np.pi * np.sqrt(7000e3**3 / sk.consts.mu_earth)
        tof = period / 4.0

        sols = sk.lambert(r1, r2, tof)
        assert len(sols) >= 1
        v1, v2 = sols[0]

        # Energy conservation
        e1 = np.dot(v1, v1) / 2 - sk.consts.mu_earth / np.linalg.norm(r1)
        e2 = np.dot(v2, v2) / 2 - sk.consts.mu_earth / np.linalg.norm(r2)
        assert e1 == pytest.approx(e2, rel=1e-8)

        # Angular momentum conservation
        h1 = np.cross(r1, v1)
        h2 = np.cross(r2, v2)
        np.testing.assert_allclose(h1, h2, rtol=1e-8)

        # Symmetric transfer: speeds should match
        assert np.linalg.norm(v1) == pytest.approx(np.linalg.norm(v2), rel=1e-6)

    def test_hohmann(self):
        """Hohmann (180-degree) transfer between circular orbits"""
        r1_mag = 7000e3
        r2_mag = 10000e3
        r1 = np.array([r1_mag, 0, 0])
        r2 = np.array([-r2_mag, 0, 0])

        a_t = (r1_mag + r2_mag) / 2
        tof = np.pi * np.sqrt(a_t**3 / sk.consts.mu_earth)

        sols = sk.lambert(r1, r2, tof)
        v1, v2 = sols[0]

        # Radial velocity should be ~0 for Hohmann
        assert abs(v1[0]) < 10.0
        # Tangential velocity should be positive (prograde)
        assert v1[1] > 0

        # Energy conservation
        e1 = np.dot(v1, v1) / 2 - sk.consts.mu_earth / r1_mag
        e2 = np.dot(v2, v2) / 2 - sk.consts.mu_earth / r2_mag
        assert e1 == pytest.approx(e2, rel=1e-8)

    def test_retrograde(self):
        """Retrograde transfer"""
        r1 = np.array([7000e3, 0, 0])
        r2 = np.array([0, 7000e3, 0])
        period = 2 * np.pi * np.sqrt(7000e3**3 / sk.consts.mu_earth)
        tof = period * 0.75

        sols = sk.lambert(r1, r2, tof, prograde=False)
        assert len(sols) >= 1
        v1, v2 = sols[0]

        e1 = np.dot(v1, v1) / 2 - sk.consts.mu_earth / np.linalg.norm(r1)
        e2 = np.dot(v2, v2) / 2 - sk.consts.mu_earth / np.linalg.norm(r2)
        assert e1 == pytest.approx(e2, rel=1e-8)

    def test_inclined(self):
        """Transfer with inclination change"""
        r1 = np.array([7000e3, 0, 0])
        r2 = np.array([0, 5000e3, 5000e3])
        tof = 3600.0

        sols = sk.lambert(r1, r2, tof)
        v1, v2 = sols[0]

        e1 = np.dot(v1, v1) / 2 - sk.consts.mu_earth / np.linalg.norm(r1)
        e2 = np.dot(v2, v2) / 2 - sk.consts.mu_earth / np.linalg.norm(r2)
        assert e1 == pytest.approx(e2, rel=1e-8)

        h1 = np.cross(r1, v1)
        h2 = np.cross(r2, v2)
        np.testing.assert_allclose(h1, h2, rtol=1e-8)

    def test_custom_mu(self):
        """Lambert with custom gravitational parameter (e.g. Sun)"""
        mu_sun = sk.consts.mu_sun
        r1 = np.array([1.496e11, 0, 0])  # ~1 AU
        r2 = np.array([0, 2.279e11, 0])  # ~Mars orbit
        tof = 200 * 86400  # 200 days

        sols = sk.lambert(r1, r2, tof, mu=mu_sun)
        v1, v2 = sols[0]

        e1 = np.dot(v1, v1) / 2 - mu_sun / np.linalg.norm(r1)
        e2 = np.dot(v2, v2) / 2 - mu_sun / np.linalg.norm(r2)
        assert e1 == pytest.approx(e2, rel=1e-8)

    def test_invalid_inputs(self):
        """Invalid inputs should raise ValueError"""
        r1 = np.array([7000e3, 0, 0])
        r2 = np.array([0, 7000e3, 0])

        with pytest.raises(ValueError):
            sk.lambert(r1, r2, -1.0)  # negative TOF

        with pytest.raises(ValueError):
            sk.lambert(r1, r2, 3600.0, mu=-1.0)  # negative mu

        with pytest.raises(ValueError):
            sk.lambert(np.array([0.0, 0.0, 0.0]), r2, 3600.0)  # zero position
