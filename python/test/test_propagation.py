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
            frame=sk.frame.RTN,
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
        sat_ric.add_maneuver(t_burn, [0.0, 10.0, 0.0], frame=sk.frame.RTN)

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

        # [vx, vy, vz, Cr*A/m]. Same values as the Rust ``test_gps``:
        # refitted against ESA SP3 truth (epochs in GPS time) with the
        # current default force model (degree-4 gravity, solid tides Step 1,
        # full IERS 10.12 relativity). Refit when the default force model
        # changes. Cr*A/m was divided by (AU / d)^2 = 1.03417 at this
        # perihelion arc when SRP gained Sun-distance scaling (#206).
        fitparam = np.array(
            [2.47517168e03, 2.94357938e03, -5.34181014e02, 2.24320920e-02]
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

        # Per-axis position residual after 1 day. Threshold tightened
        # from 8 m → 6.5 m (solid tides default-on) → 2.5 m
        # (GR Schwarzschild default-on + refitted fitparam).
        for iv in range(pgcrf.shape[0] - 5):
            state = res.interp(timearr[iv])
            for ix in range(0, 3):
                assert m.fabs(state[ix] - pgcrf[iv, ix]) < 2.5


class TestRelativisticCorrection:
    """Python bindings for the GR Schwarzschild correction."""

    def test_default_is_true(self):
        ps = sk.propsettings()
        assert ps.use_relativistic_correction is True

    def test_setter_and_kwarg(self):
        ps = sk.propsettings(use_relativistic_correction=False)
        assert ps.use_relativistic_correction is False
        ps.use_relativistic_correction = True
        assert ps.use_relativistic_correction is True

    def test_propagation_with_vs_without_gr_differs(self):
        """Toggling GR must produce a measurable position difference over
        half a day at GPS altitude (~0.4 m of cumulative drift expected)."""
        starttime = sk.time(2015, 3, 20, 0, 0, 0)
        stoptime = starttime + sk.duration.from_days(0.5)
        # GPS-like orbit (MEO, ~20200 km)
        r = sk.consts.earth_radius + 20200e3
        v = m.sqrt(sk.consts.mu_earth / r)
        state0 = np.concatenate((np.array([r, 0, 0]), np.array([0, v, 0])))

        s_on = sk.propsettings(use_relativistic_correction=True,
                               gravity_degree=8,
                               abs_error=1e-10, rel_error=1e-13)
        s_off = sk.propsettings(use_relativistic_correction=False,
                                gravity_degree=8,
                                abs_error=1e-10, rel_error=1e-13)
        r_on = sk.propagate(state0, starttime, end=stoptime, propsettings=s_on)
        r_off = sk.propagate(state0, starttime, end=stoptime, propsettings=s_off)
        diff = np.linalg.norm(r_on.state[0:3] - r_off.state[0:3])
        assert 0.05 < diff < 50.0, f"GR-induced GPS diff over 0.5d = {diff} m"


class TestSolidTides:
    """Python bindings for the solid Earth tide model."""

    def test_enum_members(self):
        assert hasattr(sk, "tidemodel")
        assert sk.tidemodel.none != sk.tidemodel.solid_step1
        assert sk.tidemodel.solid_step1 != sk.tidemodel.solid_full

    def test_default_is_solid_step1(self):
        ps = sk.propsettings()
        assert ps.tide_model == sk.tidemodel.solid_step1

    def test_setter_and_kwarg(self):
        ps = sk.propsettings(tide_model=sk.tidemodel.none)
        assert ps.tide_model == sk.tidemodel.none
        ps.tide_model = sk.tidemodel.solid_step1
        assert ps.tide_model == sk.tidemodel.solid_step1

    def test_propagation_with_vs_without_tides_differs(self):
        """Toggling tides must produce a measurable position difference
        at GEO over half a day (~0.3 m expected, well above numerical noise)."""
        starttime = sk.time(2015, 3, 20, 0, 0, 0)
        stoptime = starttime + sk.duration.from_days(0.5)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
        state0 = np.concatenate((pos, vel))

        settings_on = sk.propsettings(
            tide_model=sk.tidemodel.solid_step1,
            gravity_degree=8,
            abs_error=1e-10,
            rel_error=1e-13,
        )
        settings_off = sk.propsettings(
            tide_model=sk.tidemodel.none,
            gravity_degree=8,
            abs_error=1e-10,
            rel_error=1e-13,
        )
        res_on = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_on)
        res_off = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_off)
        diff = np.linalg.norm(res_on.state[0:3] - res_off.state[0:3])
        assert 0.05 < diff < 50.0, f"tide-induced GEO diff over 0.5d = {diff} m"


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
        """Test that satstate pickle round-trips all fields including maneuvers
        in every supported frame."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t_burn = t0 + sk.duration.from_hours(1)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        # One maneuver per supported frame, to catch frame-tag corruption
        # (NTW/LVLH used to be silently written as GCRF).
        sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
        sat.add_maneuver(t_burn + sk.duration.from_hours(1), [5, 0, 0], frame=sk.frame.GCRF)
        sat.add_maneuver(t_burn + sk.duration.from_hours(2), [0, 3, 0], frame=sk.frame.NTW)
        sat.add_maneuver(t_burn + sk.duration.from_hours(3), [1, 0, 2], frame=sk.frame.LVLH)

        restored = pickle.loads(pickle.dumps(sat))

        assert restored.time == sat.time
        assert np.allclose(restored.pos, sat.pos)
        assert np.allclose(restored.vel, sat.vel)
        assert restored.num_maneuvers == 4
        assert restored.cov is None

        # Frames must survive the round-trip: propagating past all four burns
        # must match the original trajectory. If any frame were corrupted, the
        # applied delta-v would point elsewhere and the endpoints would diverge.
        t_end = t_burn + sk.duration.from_hours(4)
        orig_end = sat.propagate(t_end)
        restored_end = restored.propagate(t_end)
        assert np.allclose(orig_end.pos, restored_end.pos)
        assert np.allclose(orig_end.vel, restored_end.vel)

    def test_satstate_pickle_many_maneuvers_no_cov(self):
        """A state with >=9 maneuvers and no covariance must not be misread as
        carrying a covariance. Regression: the old length-based heuristic saw
        9*33 = 297 >= 288 bytes, consumed them as a covariance matrix, and
        dropped every maneuver."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)

        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        for i in range(9):
            sat.add_maneuver(
                t0 + sk.duration.from_hours(i + 1), [0, 1, 0], frame=sk.frame.RTN
            )

        restored = pickle.loads(pickle.dumps(sat))
        assert restored.num_maneuvers == 9
        assert restored.cov is None

    def test_uncertainty_frames(self):
        """The unified set_pos_uncertainty / set_vel_uncertainty API should
        accept GCRF, LVLH, RIC, NTW frames and preserve the other block
        when called in sequence."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state0 = (np.array([r, 0, 0]), np.array([0, v, 0]))

        # Every supported frame should succeed
        for frm in [sk.frame.GCRF, sk.frame.LVLH, sk.frame.RTN, sk.frame.NTW]:
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


class TestThrustPickle:
    def test_thrust_pickle_all_frames(self):
        """Standalone thrust pickle must round-trip accel, frame, and times."""
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        for frm in [sk.frame.GCRF, sk.frame.RTN, sk.frame.NTW, sk.frame.LVLH]:
            th = sk.thrust.constant([1e-4, 2e-4, 3e-4], t0, t1, frame=frm)
            restored = pickle.loads(pickle.dumps(th))
            assert restored.frame == frm
            assert restored.accel == [
                pytest.approx(1e-4),
                pytest.approx(2e-4),
                pytest.approx(3e-4),
            ]
            assert abs((restored.start - t0).seconds) < 1e-6
            assert abs((restored.end - t1).seconds) < 1e-6

    def test_thrust_rejects_invalid_frame(self):
        """Constructing a thrust in an Earth frame raises instead of storing
        a value that would panic during propagation."""
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        for bad in [sk.frame.ITRF, sk.frame.TEME, sk.frame.ICRF]:
            with pytest.raises(Exception):
                sk.thrust.constant([1e-4, 0, 0], t0, t1, frame=bad)


class TestSatPropertiesPickle:
    def test_satproperties_pickle_with_thrust(self):
        """Test that satproperties pickle round-trips thrust arcs"""
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        t2 = t1 + sk.duration.from_hours(1)

        thrust1 = sk.thrust.constant([1e-4, 2e-4, 3e-4], t0, t1, frame=sk.frame.RTN)
        thrust2 = sk.thrust.constant([0, 0, 5e-3], t1, t2, frame=sk.frame.GCRF)
        thrust3 = sk.thrust.constant([1e-4, 0, 0], t0, t1, frame=sk.frame.NTW)
        thrust4 = sk.thrust.constant([0, 1e-4, 0], t1, t2, frame=sk.frame.LVLH)
        props = sk.satproperties(
            cdaoverm=0.01, thrusts=[thrust1, thrust2, thrust3, thrust4]
        )

        restored = pickle.loads(pickle.dumps(props))

        assert restored.cdaoverm == pytest.approx(0.01)
        assert len(restored.thrusts) == 4
        # All four frames must round-trip (NTW/LVLH used to become GCRF).
        assert restored.thrusts[0].frame == sk.frame.RTN
        assert restored.thrusts[0].accel == [pytest.approx(1e-4), pytest.approx(2e-4), pytest.approx(3e-4)]
        assert restored.thrusts[1].frame == sk.frame.GCRF
        assert restored.thrusts[1].accel == [pytest.approx(0), pytest.approx(0), pytest.approx(5e-3)]
        assert restored.thrusts[2].frame == sk.frame.NTW
        assert restored.thrusts[3].frame == sk.frame.LVLH


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
        thrust = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)
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
        thrust = sk.thrust.constant([1e-4, 2e-4, 3e-4], t0, t1, frame=sk.frame.RTN)

        assert thrust.frame == sk.frame.RTN
        assert thrust.accel == [pytest.approx(1e-4), pytest.approx(2e-4), pytest.approx(3e-4)]

    def test_multiple_thrust_arcs(self):
        """Test multiple thrust arcs"""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        t1 = t0 + sk.duration.from_hours(1)
        t2 = t1 + sk.duration.from_hours(1)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        state = np.array([r, 0, 0, 0, v, 0])

        thrust1 = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)
        thrust2 = sk.thrust.constant([0, 1e-4, 0], t1, t2, frame=sk.frame.RTN)
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
        sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
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

    def test_itrf_gcrf_state_transform(self):
        """The ITRF <-> GCRF state transform must handle the Earth-rotation
        sweep term that a raw quaternion rotation ignores. Demonstrates
        the ~470 m/s gap between the naive rotation and the correct full
        state transform for a LEO satellite parked on Earth.
        """
        t = sk.time(2024, 3, 15, 12, 34, 56)

        # A point at rest on Earth's surface near the equator (LEO altitude)
        pos_itrf = np.array([6.378e6 + 500e3, 0.0, 0.0])
        vel_itrf = np.array([0.0, 0.0, 0.0])

        pos_gcrf, vel_gcrf = sk.frametransform.itrf_to_gcrf_state(
            pos_itrf, vel_itrf, t
        )

        # |pos| preserved
        assert abs(np.linalg.norm(pos_gcrf) - np.linalg.norm(pos_itrf)) < 1e-6

        # |vel_gcrf| ≈ OMEGA_EARTH · |r_itrf| ≈ 501 m/s
        omega_earth = 7.2921150e-5
        expected_speed = omega_earth * np.linalg.norm(pos_itrf)
        assert abs(np.linalg.norm(vel_gcrf) - expected_speed) < 1.0, (
            f"ITRF-rest LEO GCRF velocity {np.linalg.norm(vel_gcrf):.3f} m/s, "
            f"expected ≈{expected_speed:.3f}"
        )

        # Round-trip: GCRF -> ITRF -> GCRF should recover the input
        pos_gcrf_in = np.array([6.878e6, 1.23e5, -4.56e5])
        vel_gcrf_in = np.array([-123.4, 7600.0, 89.0])
        pos_itrf_out, vel_itrf_out = sk.frametransform.gcrf_to_itrf_state(
            pos_gcrf_in, vel_gcrf_in, t
        )
        pos_back, vel_back = sk.frametransform.itrf_to_gcrf_state(
            pos_itrf_out, vel_itrf_out, t
        )
        assert np.allclose(pos_back, pos_gcrf_in, atol=1e-6)
        assert np.allclose(vel_back, vel_gcrf_in, atol=1e-9)

        # Demonstrate the gap between "naive rotation" and the correct
        # full state transform: rotating velocity with qitrf2gcrf alone
        # (ignoring the omega × r term) is wrong by ~501 m/s for a LEO
        # state at rest in ITRF.
        q = sk.frametransform.qitrf2gcrf(t)
        vel_gcrf_naive = q * vel_itrf  # zero, trivially
        assert np.linalg.norm(vel_gcrf_naive) < 1e-10
        # ...whereas the correct answer has |v| ≈ 501 m/s
        naive_vs_correct_gap = np.linalg.norm(vel_gcrf - vel_gcrf_naive)
        assert abs(naive_vs_correct_gap - expected_speed) < 1.0

    def test_frame_ric_rsw_are_aliases_for_rtn(self):
        """satkit's canonical name for the Radial / Tangential / Normal
        orbital frame is ``frame.RTN``; ``frame.RIC`` and ``frame.RSW``
        are Python-level aliases that resolve to the same enum value.
        All three should be interchangeable.
        """
        # Identity: all three compare equal
        assert sk.frame.RIC == sk.frame.RTN
        assert sk.frame.RSW == sk.frame.RTN
        assert sk.frame.RIC == sk.frame.RSW

        # Functionally equivalent when passed to the maneuver API
        t0 = sk.time(2024, 1, 1)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])

        sat_rtn = sk.satstate(time=t0, pos=pos, vel=vel)
        sat_ric = sk.satstate(time=t0, pos=pos, vel=vel)
        sat_rsw = sk.satstate(time=t0, pos=pos, vel=vel)

        t_burn = t0 + sk.duration.from_hours(0.5)
        t_end = t0 + sk.duration.from_hours(2.0)

        sat_rtn.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
        sat_ric.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RIC)
        sat_rsw.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RSW)

        s_rtn = sat_rtn.propagate(t_end)
        s_ric = sat_ric.propagate(t_end)
        s_rsw = sat_rsw.propagate(t_end)

        assert np.allclose(s_rtn.pos, s_ric.pos)
        assert np.allclose(s_rtn.pos, s_rsw.pos)
        assert np.allclose(s_rtn.vel, s_ric.vel)
        assert np.allclose(s_rtn.vel, s_rsw.vel)

    def test_frametransform_to_from_gcrf(self):
        """Test the unified frametransform.to_gcrf / from_gcrf dispatch
        across all supported satellite-local frames."""
        pos = np.array([6878e3, 0, 0])
        vel = np.array([0, 7612, 0])

        # GCRF dispatch returns identity
        dcm_gcrf = sk.frametransform.to_gcrf(sk.frame.GCRF, pos, vel)
        assert np.allclose(dcm_gcrf, np.eye(3))

        # All four supported frames: to_gcrf and from_gcrf are mutual inverses
        for frm in [sk.frame.GCRF, sk.frame.LVLH, sk.frame.RTN, sk.frame.NTW]:
            dcm = sk.frametransform.to_gcrf(frm, pos, vel)
            assert dcm.shape == (3, 3)
            dcm_inv = sk.frametransform.from_gcrf(frm, pos, vel)
            assert np.allclose(dcm @ dcm_inv, np.eye(3), atol=1e-12)

        # Unsupported frames raise
        for bad in [sk.frame.ITRF, sk.frame.TEME, sk.frame.ICRF]:
            with pytest.raises(Exception):
                sk.frametransform.to_gcrf(bad, pos, vel)


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

        # r1 == r2 used to return NaN velocities; NaN inputs a misleading
        # convergence failure
        with pytest.raises(ValueError, match="coincide"):
            sk.lambert(r1, r1.copy(), 3600.0)
        with pytest.raises(ValueError, match="tof must be finite"):
            sk.lambert(r1, r2, float("nan"))
        with pytest.raises(ValueError, match="mu must be finite"):
            sk.lambert(r1, r2, 3600.0, mu=float("inf"))
        with pytest.raises(ValueError, match="r2 must be finite"):
            sk.lambert(r1, np.array([0.0, np.nan, 0.0]), 3600.0)

    @staticmethod
    def _propagate(r0, v0, dt, mu=sk.consts.mu_earth):
        """Independent two-body propagation with universal variables
        (Curtis, Algorithms 3.3/3.4); valid for any conic."""

        def stumpff(z):
            if z > 1e-8:
                s = m.sqrt(z)
                return (1 - m.cos(s)) / z, (s - m.sin(s)) / s**3
            if z < -1e-8:
                s = m.sqrt(-z)
                return (m.cosh(s) - 1) / -z, (m.sinh(s) - s) / s**3
            return 0.5 - z / 24, 1 / 6 - z / 120

        r0n = np.linalg.norm(r0)
        vr0 = np.dot(r0, v0) / r0n
        alpha = 2 / r0n - np.dot(v0, v0) / mu
        smu = m.sqrt(mu)

        def kepler_uv(chi):
            c, s = stumpff(alpha * chi * chi)
            return (
                r0n * vr0 / smu * chi**2 * c
                + (1 - alpha * r0n) * chi**3 * s
                + r0n * chi
                - smu * dt
            )

        # F is increasing in chi with F(0) < 0: bracket, then bisect
        lo, hi = 0.0, smu * dt / r0n
        while kepler_uv(hi) < 0:
            lo, hi = hi, 2 * hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if kepler_uv(mid) < 0:
                lo = mid
            else:
                hi = mid
        chi = 0.5 * (lo + hi)
        c, s = stumpff(alpha * chi * chi)
        f = 1 - chi**2 / r0n * c
        g = dt - chi**3 / smu * s
        return f * r0 + g * v0

    def _check_reaches_r2(self, r1, r2, tof, prograde=True):
        sols = sk.lambert(r1, r2, tof, prograde=prograde)
        assert len(sols) >= 1
        for v1, _ in sols:
            r = self._propagate(r1, v1, tof)
            assert np.linalg.norm(r - r2) < 1e-7 * np.linalg.norm(r2)
            hz = np.cross(r1, v1)[2]
            assert hz > 0 if prograde else hz < 0

    def test_solutions_reach_r2(self):
        """Every solution, propagated for tof, arrives at r2 moving in the
        requested direction. Long-way (> 180 deg) and retrograde transfers
        used to leave r1 in the wrong direction and miss r2 by ~2r, and
        short hyperbolic transfers missed by kilometres."""
        r = 7000e3
        period = 2 * np.pi * np.sqrt(r**3 / sk.consts.mu_earth)
        r1 = np.array([r, 0.0, 0.0])
        # Long way, prograde and retrograde
        for deg in (240.0, 300.0):
            th = np.radians(deg)
            r2 = 1.3 * r * np.array([np.cos(th), np.sin(th), 0.0])
            self._check_reaches_r2(r1, r2, 0.75 * period)
            self._check_reaches_r2(r1, r2 * [1, -1, 1], 0.75 * period, prograde=False)
        # Retrograde to +90 deg is the long way round, with multi-rev
        self._check_reaches_r2(r1, np.array([0.0, r, 0.0]), 0.75 * period, prograde=False)
        self._check_reaches_r2(r1, np.array([0.0, r, 0.0]), 3.3 * period, prograde=False)
        # Inclined long way
        r2 = np.array([-5000e3, -3000e3, 4000e3])
        self._check_reaches_r2(r1, r2, 4000.0)
        # Near-0 deg, strongly hyperbolic
        th = np.radians(1.0)
        r2 = 1.01 * r * np.array([np.cos(th), np.sin(th), 0.0])
        for tof in (10.0, 100.0):
            self._check_reaches_r2(r1, r2, tof)


class TestManeuverInspection:
    def test_maneuvers_getter(self):
        """satstate.maneuvers must expose the scheduled maneuvers, not just a count."""
        t0 = sk.time(2024, 1, 1, 12, 0, 0)
        r = 6378e3 + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        sat = sk.satstate(time=t0, pos=np.array([r, 0, 0]), vel=np.array([0, v, 0]))
        t_burn = t0 + sk.duration.from_hours(1)
        sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)
        sat.add_maneuver(t_burn + sk.duration.from_hours(1), [1, 2, 3], frame=sk.frame.LVLH)

        mans = sat.maneuvers
        assert len(mans) == 2 == sat.num_maneuvers
        assert abs((mans[0]["time"] - t_burn).seconds) < 1e-6
        assert np.allclose(mans[0]["delta_v"], [0, 10, 0])
        assert mans[0]["frame"] == sk.frame.RTN
        assert np.allclose(mans[1]["delta_v"], [1, 2, 3])
        assert mans[1]["frame"] == sk.frame.LVLH


class TestSpaceWeather:
    def test_spaceweather_get(self):
        """spaceweather.get returns the full daily record."""
        rec = sk.spaceweather.get(sk.time(2023, 11, 14))
        assert abs((rec["date"] - sk.time(2023, 11, 14)).days) < 1.0
        assert len(rec["kp"]) == 8
        assert len(rec["ap"]) == 8
        assert rec["f10p7_adj"] > 0
        assert rec["f10p7_adj_c81"] > 0
        assert rec["ap_avg"] >= 0
        # datetime accepted too
        import datetime

        rec2 = sk.spaceweather.get(
            datetime.datetime(2023, 11, 14, tzinfo=datetime.timezone.utc)
        )
        assert rec2["f10p7_adj"] == rec["f10p7_adj"]


class TestInitialStep:
    """propsettings.initial_step_secs and propresult.next_step_secs."""

    @staticmethod
    def _leo():
        r = sk.consts.earth_radius + 550e3
        v = m.sqrt(sk.consts.mu_earth / r)
        inc = m.radians(51.6)
        state = np.array([r, 0.0, 0.0, 0.0, v * m.cos(inc), v * m.sin(inc)])
        return state, sk.time(2025, 1, 1, 12, 0, 0)

    def test_default_and_kwarg(self):
        assert sk.propsettings().initial_step_secs is None
        assert sk.propsettings(initial_step_secs=30.0).initial_step_secs == pytest.approx(30.0)
        ps = sk.propsettings()
        ps.initial_step_secs = 12.5
        assert ps.initial_step_secs == pytest.approx(12.5)
        ps.initial_step_secs = None
        assert ps.initial_step_secs is None

    def test_next_step_is_working_stride(self):
        state, t0 = self._leo()
        ps = sk.propsettings(abs_error=1e-9, rel_error=1e-9)
        res = sk.propagate(state, t0, duration_secs=3600.0, propsettings=ps)
        # RKV98 at 1e-9 on a 550 km orbit steps a few hundred seconds.
        assert 50.0 < res.next_step_secs < 1000.0
        back = sk.propagate(res.state, res.time_end, duration_secs=-3600.0, propsettings=ps)
        assert back.next_step_secs < 0.0

    def test_warm_start_saves_evals_and_matches(self):
        state, t0 = self._leo()
        ps = sk.propsettings(abs_error=1e-9, rel_error=1e-9)
        seg1 = sk.propagate(state, t0, duration_secs=3600.0, propsettings=ps)
        t1 = seg1.time_end
        # A deliberately tiny hint reproduces a cold start; the default
        # state-derived hint and the warm start must both beat it.
        cold_ps = sk.propsettings(abs_error=1e-9, rel_error=1e-9, initial_step_secs=1e-3)
        warm_ps = sk.propsettings(
            abs_error=1e-9, rel_error=1e-9, initial_step_secs=seg1.next_step_secs
        )
        cold = sk.propagate(seg1.state, t1, duration_secs=3600.0, propsettings=cold_ps)
        default = sk.propagate(seg1.state, t1, duration_secs=3600.0, propsettings=ps)
        warm = sk.propagate(seg1.state, t1, duration_secs=3600.0, propsettings=warm_ps)
        # A good default can tie the warm start to within one rkv98 step.
        assert warm.stats.num_eval <= default.stats.num_eval + 21
        assert default.stats.num_eval < cold.stats.num_eval
        assert warm.stats.num_eval < 0.6 * cold.stats.num_eval
        assert warm.pos == pytest.approx(cold.pos, abs=1e-2)
        assert default.pos == pytest.approx(cold.pos, abs=1e-2)

    def test_rkv98_without_interp_uses_16_stages(self):
        state, t0 = self._leo()
        # A hint skips the heuristic's probe evaluations: evals == stages × steps.
        with_interp = sk.propsettings(abs_error=1e-9, rel_error=1e-9, initial_step_secs=100.0)
        without = sk.propsettings(
            abs_error=1e-9, rel_error=1e-9, initial_step_secs=100.0, enable_interp=False
        )
        a = sk.propagate(state, t0, duration_secs=3600.0, propsettings=with_interp)
        b = sk.propagate(state, t0, duration_secs=3600.0, propsettings=without)
        assert a.stats.num_eval == 21 * (a.stats.num_accept + a.stats.num_reject)
        assert b.stats.num_eval == 16 * (b.stats.num_accept + b.stats.num_reject)
        assert b.stats.num_eval < a.stats.num_eval
        assert b.pos == pytest.approx(a.pos, abs=1e-2)

    def test_invalid_hint_raises(self):
        state, t0 = self._leo()
        for bad in (0.0, float("nan"), float("inf")):
            ps = sk.propsettings(initial_step_secs=bad)
            with pytest.raises(RuntimeError):
                sk.propagate(state, t0, duration_secs=600.0, propsettings=ps)

