import pytest
import numpy as np
import math as m

import satkit as sk


class TestFrameTransform:
    def test_itrf2gcrf(self):
        """
        IAU-2000 Reduction,
        Vallado Example 3-14
        """

        pITRF = np.array([-1033.479383, 7901.2952754, 6380.3565958]) * 1e3
        vITRF = np.array([-3.225636520, -2.872451450, 5.531924446]) * 1.0e3
        tm = sk.time(2004, 4, 6, 7, 51, 28.386009)

        # The example looks up dut1 from start of day, along with x and y polar motion
        # Intermediate check by getting values and comparing against
        # values used in example
        tm2 = sk.time(2004, 4, 6, 0, 0, 0)
        dut1, xp, yp, lod, dX, dY = sk.frametransform.earth_orientation_params(tm2)
        assert dut1 == pytest.approx(-0.4399619, rel=1e-3)
        assert xp == pytest.approx(-0.140857, rel=1e-3)
        assert yp == pytest.approx(0.333309, rel=1e-3)
        jd_tt = tm.as_jd(sk.timescale.TT)
        assert jd_tt == pytest.approx(2453101.828154745)
        t_tt = (tm.as_jd(sk.timescale.TT) - 2451545.0) / 36525.0

        assert t_tt == pytest.approx(0.0426236319, rel=1e-8)
        # Check transform to terrestial intermediate frame
        # with value from example
        pTIRS = sk.frametransform.qitrf2tirs(tm) * pITRF
        assert pTIRS[0] == pytest.approx(-1033475.0312, rel=1e-7)
        assert pTIRS[1] == pytest.approx(7901305.5856, rel=1e-7)
        assert pTIRS[2] == pytest.approx(6380344.5327, rel=1e-7)

        # Check transfomr to celestial intermediate frame
        # with value from example
        pCIRS = sk.quaternion.rotz(sk.frametransform.earth_rotation_angle(tm)) * pTIRS
        assert pCIRS[0] == pytest.approx(5100018.4047, rel=1e-7)
        assert pCIRS[1] == pytest.approx(6122786.3648, rel=1e-7)
        assert pCIRS[2] == pytest.approx(6380344.6237, rel=1e-7)

        # Check transform to geocentric celestial reference frame
        # with value from example
        pGCRF = sk.frametransform.qcirs2gcrf(tm) * pCIRS
        assert pGCRF[0] == pytest.approx(5102508.959, rel=1e-7)
        assert pGCRF[1] == pytest.approx(6123011.403, rel=1e-7)
        assert pGCRF[2] == pytest.approx(6378136.925, rel=1e-7)

        # Now, test the whole transform at once
        pGCRF = sk.frametransform.qitrf2gcrf(tm) * pITRF
        assert pGCRF[0] == pytest.approx(5102508.959)
        assert pGCRF[1] == pytest.approx(6123011.403)
        assert pGCRF[2] == pytest.approx(6378136.925)

    def test_itrf2gcrf_state(self):
        """
        IAU-2000 Reduction applied to a full state (position + velocity)
        in a single call via ``itrf_to_gcrf_state``.

        Vallado Example 3-14 (4th ed) provides both the ITRF state and
        the expected GCRF state. The position part is verified by
        :meth:`test_itrf2gcrf` above; this test additionally checks that
        the velocity transform correctly accounts for Earth's rotation
        (the ``omega_earth × r`` term, ~465 m/s at this altitude), which
        a plain quaternion rotation of velocity would miss entirely.

        Expected GCRF state from Vallado Example 3-14:
            r_GCRF = [ 5102.508959,  6123.011403,  6378.136925] km
            v_GCRF = [-4.7432196,    0.7905366,    5.5337561  ] km/s
        """
        pITRF = np.array([-1033.479383, 7901.2952754, 6380.3565958]) * 1e3
        vITRF = np.array([-3.225636520, -2.872451450, 5.531924446]) * 1e3
        tm = sk.time(2004, 4, 6, 7, 51, 28.386009)

        # Single-call full state transform
        pGCRF, vGCRF = sk.frametransform.itrf_to_gcrf_state(pITRF, vITRF, tm)

        # Position must match the existing Vallado 3-14 reference
        assert pGCRF[0] == pytest.approx(5102508.959, rel=1e-7)
        assert pGCRF[1] == pytest.approx(6123011.403, rel=1e-7)
        assert pGCRF[2] == pytest.approx(6378136.925, rel=1e-7)

        # Velocity must match Vallado's expected GCRF velocity.
        # The Coriolis-style omega × r term (~465 m/s) is included
        # automatically by itrf_to_gcrf_state.
        assert vGCRF[0] == pytest.approx(-4743.2196, rel=1e-6)
        assert vGCRF[1] == pytest.approx(790.5366, rel=1e-6)
        assert vGCRF[2] == pytest.approx(5533.7561, rel=1e-6)

        # Round-trip: GCRF -> ITRF -> GCRF should recover the original
        # state to well under a millimeter / nanometer per second.
        pITRF_back, vITRF_back = sk.frametransform.gcrf_to_itrf_state(
            pGCRF, vGCRF, tm
        )
        assert np.allclose(pITRF_back, pITRF, atol=1e-6)
        assert np.allclose(vITRF_back, vITRF, atol=1e-9)

        # Sanity: rotating velocity alone with qitrf2gcrf (the naive
        # approach) gives a wrong answer because it misses the
        # Earth-rotation sweep term omega × r.
        q = sk.frametransform.qitrf2gcrf(tm)
        vGCRF_naive = q * vITRF
        naive_error = np.linalg.norm(vGCRF - vGCRF_naive)
        # Expected error = |omega_earth × r| = omega_earth · |r| · sin(theta),
        # where theta is the angle between r and Earth's rotation axis.
        # Vallado 3-14 state has |r| ≈ 10208 km at ~51° latitude off
        # the equator (z_itrf ≈ 6380 km out of 10208 km radius), so
        # sin(theta) ≈ 0.78 and the expected sweep is ~581 m/s.
        r_norm = np.linalg.norm(pITRF)
        xy_norm = np.linalg.norm(pITRF[:2])
        omega_earth = 7.2921150e-5
        expected_naive_err = omega_earth * r_norm * (xy_norm / r_norm)
        assert abs(naive_error - expected_naive_err) < 0.1, (
            f"Naive-rotation velocity error = {naive_error:.3f} m/s, "
            f"expected ω⊕·|r_xy| = {expected_naive_err:.3f} m/s"
        )

    def test_itrf2gcrf_state_approx(self):
        """Approximate IAU-76/FK5 state transforms — round-trip self-
        consistency and agreement with the full IERS 2010 reduction to
        ~1 arcsec on position, ~1 m/s on velocity."""
        pITRF = np.array([-1033.479383, 7901.2952754, 6380.3565958]) * 1e3
        vITRF = np.array([-3.225636520, -2.872451450, 5.531924446]) * 1e3
        tm = sk.time(2004, 4, 6, 7, 51, 28.386009)

        pGCRF_a, vGCRF_a = sk.frametransform.itrf_to_gcrf_state_approx(
            pITRF, vITRF, tm
        )

        # Round trip
        pITRF_b, vITRF_b = sk.frametransform.gcrf_to_itrf_state_approx(
            pGCRF_a, vGCRF_a, tm
        )
        assert np.allclose(pITRF_b, pITRF, atol=1e-6)
        assert np.allclose(vITRF_b, vITRF, atol=1e-9)

        # Agreement with full IERS 2010 reduction at the advertised accuracy
        # (~1 arcsec ≈ 50 m at LEO).
        pGCRF_f, vGCRF_f = sk.frametransform.itrf_to_gcrf_state(pITRF, vITRF, tm)
        assert np.linalg.norm(pGCRF_a - pGCRF_f) < 100.0
        assert np.linalg.norm(vGCRF_a - vGCRF_f) < 1.0

    def test_itrf2gcrf_state_approx_batch(self):
        """Batched form of the approximate state transform: feeding an
        (N, 3) array and N times must match per-row scalar calls."""
        tms = [
            sk.time(2024, 1, 1, 0, 0, 0.0),
            sk.time(2024, 6, 1, 12, 0, 0.0),
            sk.time(2024, 12, 31, 23, 59, 59.0),
        ]
        pITRF = np.array(
            [
                [7.0e6, 0.0, 0.0],
                [0.0, 7.1e6, 1.0e5],
                [-6.9e6, 2.0e5, -3.0e5],
            ]
        )
        vITRF = np.array(
            [
                [0.0, 7500.0, 100.0],
                [-7400.0, 0.0, 50.0],
                [100.0, 7450.0, -80.0],
            ]
        )

        pGCRF, vGCRF = sk.frametransform.itrf_to_gcrf_state_approx(
            pITRF, vITRF, tms
        )
        assert pGCRF.shape == (3, 3)
        assert vGCRF.shape == (3, 3)

        for i, t in enumerate(tms):
            p_i, v_i = sk.frametransform.itrf_to_gcrf_state_approx(
                pITRF[i], vITRF[i], t
            )
            assert np.allclose(pGCRF[i], p_i, atol=1e-9)
            assert np.allclose(vGCRF[i], v_i, atol=1e-12)

        # Batch round trip
        pITRF_back, vITRF_back = sk.frametransform.gcrf_to_itrf_state_approx(
            pGCRF, vGCRF, tms
        )
        assert np.allclose(pITRF_back, pITRF, atol=1e-6)
        assert np.allclose(vITRF_back, vITRF, atol=1e-9)

    def test_gmst(self):
        """
        Test GMST : vallado example 3-5
        """

        tm = sk.time(1992, 8, 20, 12, 14, 0)

        # Spooof UTC as UT1 value (as is done in example from Vallado)
        tdiff = tm.as_mjd(sk.timescale.UT1) - tm.as_mjd(sk.timescale.UTC)
        tm = tm - sk.duration.from_days(tdiff)
        gmst = sk.frametransform.gmst(tm)
        truth = -207.4212121875 * m.pi / 180
        assert gmst == pytest.approx(truth)

    # ── Frame-enum dispatch (new in 0.17.0) ────────────────────────────

    def test_rotation_matches_qitrf2gcrf(self):
        """Dispatch ITRF→GCRF should match the direct quaternion helper."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        v = np.array([1000.0, 2000.0, 3000.0])
        q_dispatch = sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, tm)
        q_direct = sk.frametransform.qitrf2gcrf(tm)
        assert np.allclose(q_dispatch * v, q_direct * v, atol=1e-9)

    def test_rotation_identity_pairs(self):
        """rotation(X, X, t) must be the identity for all time-frames."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        v = np.array([1.0, 2.0, 3.0])
        for f in [
            sk.frame.ITRF,
            sk.frame.TIRS,
            sk.frame.CIRS,
            sk.frame.GCRF,
            sk.frame.TEME,
            sk.frame.EME2000,
            sk.frame.ICRF,
        ]:
            q = sk.frametransform.rotation(f, f, tm)
            assert np.allclose(q * v, v)

    def test_rotation_shortest_path_itrf_tirs(self):
        """ITRF→TIRS is a single edge (polar motion only); compare to direct."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        v = np.array([1000.0, 2000.0, 3000.0])
        q_dispatch = sk.frametransform.rotation(sk.frame.ITRF, sk.frame.TIRS, tm)
        q_direct = sk.frametransform.qitrf2tirs(tm)
        assert np.allclose(q_dispatch * v, q_direct * v, atol=1e-12)

    def test_rotation_roundtrip(self):
        """rotation(a, b, t) composed with rotation(b, a, t) must be identity."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        v = np.array([6378.0, 2000.0, 3000.0])
        for a in [
            sk.frame.ITRF,
            sk.frame.GCRF,
            sk.frame.TEME,
            sk.frame.EME2000,
            sk.frame.ICRF,
        ]:
            for b in [
                sk.frame.ITRF,
                sk.frame.GCRF,
                sk.frame.TEME,
                sk.frame.EME2000,
                sk.frame.ICRF,
            ]:
                q_ab = sk.frametransform.rotation(a, b, tm)
                q_ba = sk.frametransform.rotation(b, a, tm)
                v_round = q_ba * (q_ab * v)
                assert np.allclose(v_round, v, rtol=1e-12)

    def test_rotation_orbit_frames_rejected(self):
        """LVLH / RTN / NTW need state, not just time — must raise."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        for of in [sk.frame.LVLH, sk.frame.RTN, sk.frame.NTW]:
            with pytest.raises(RuntimeError):
                sk.frametransform.rotation(of, sk.frame.GCRF, tm)
            with pytest.raises(RuntimeError):
                sk.frametransform.rotation(sk.frame.GCRF, of, tm)

    def test_rotation_approx_rejects_intermediates(self):
        """TIRS / CIRS have no FK5 analogue — approx must raise for them."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        for f in [sk.frame.TIRS, sk.frame.CIRS]:
            with pytest.raises(RuntimeError):
                sk.frametransform.rotation_approx(f, sk.frame.GCRF, tm)

    def test_transform_state_matches_itrf_to_gcrf_state(self):
        """transform_state ITRF→GCRF must match the direct state function."""
        tm = sk.time(2026, 5, 22, 12, 0, 0)
        pITRF = np.array([6378137.0, 0.0, 0.0])
        vITRF = np.array([0.0, 0.0, 0.0])
        p_dispatch, v_dispatch = sk.frametransform.transform_state(
            sk.frame.ITRF, sk.frame.GCRF, tm, pITRF, vITRF
        )
        p_direct, v_direct = sk.frametransform.itrf_to_gcrf_state(pITRF, vITRF, tm)
        assert np.allclose(p_dispatch, p_direct, atol=1e-6)
        assert np.allclose(v_dispatch, v_direct, atol=1e-9)


class TestGravity:
    def test_gravity(self):
        """
        Reference gravity computations, using
        JGM3 model, with 16 terms, found at:
        http://icgem.gfz-potsdam.de/calcstat/
        Outputs from above web page listed
        in function below as reference values
        """

        reference_gravitation = 9.822206169031
        reference_gravity = 9.803696372738
        # Gravity deflection from normal along east-west and
        # north-south direction, in arcseconds
        reference_ew_deflection_asec = -1.283542043355
        reference_ns_deflection_asec = -1.311709802440

        latitude_deg = 42.4473
        longitude_deg = -71.2272
        altitude = 0

        itrf = sk.itrfcoord(
            latitude_deg=latitude_deg, longitude_deg=longitude_deg, altitude=altitude
        )
        gravitation = sk.gravity(itrf, degree=16, model=sk.gravmodel.jgm3)
        # Add centrifugal force @ Earth surface
        centrifugal = (
            np.array([itrf.vector[0], itrf.vector[1], 0]) * sk.consts.omega_earth**2
        )

        assert np.linalg.norm(gravitation) == pytest.approx(
            reference_gravitation, rel=1e-9
        )
        gravity = gravitation + centrifugal
        assert np.linalg.norm(gravity) == pytest.approx(reference_gravity, rel=1e-9)

        # Rotate gravity into East-North-Up frame to check deflections
        gravity_enu = itrf.qenu2itrf.conj * gravity
        ew_deflection = (
            -m.atan2(gravity_enu[0], -gravity_enu[2]) * 180.0 / m.pi * 3600.0
        )
        ns_deflection = (
            -m.atan2(gravity_enu[1], -gravity_enu[2]) * 180.0 / m.pi * 3600.0
        )
        assert ns_deflection == pytest.approx(reference_ns_deflection_asec, rel=2e-6)
        assert ew_deflection == pytest.approx(reference_ew_deflection_asec, rel=2e-6)

    def test_gravity_degree_order(self):
        """Test separate degree and order parameters for gravity function"""
        itrf = sk.itrfcoord(latitude_deg=45.0, longitude_deg=30.0, altitude=400e3)

        # degree=8, order=8 (full) should differ from degree=8, order=0 (zonal only)
        g_full = sk.gravity(itrf, degree=8, order=8)
        g_zonal = sk.gravity(itrf, degree=8, order=0)
        diff = np.linalg.norm(g_full - g_zonal)
        assert diff > 1e-6, f"Full and zonal-only gravity should differ, diff = {diff}"

        # order defaults to degree
        g_default = sk.gravity(itrf, degree=8)
        assert np.allclose(g_full, g_default), "order should default to degree"

    def test_propsettings_new_fields(self):
        """Test new propsettings fields: gravity_degree, gravity_order, use_sun_gravity, use_moon_gravity"""
        # Default values
        s = sk.propsettings()
        assert s.gravity_degree == 4
        assert s.gravity_order == 4
        assert s.use_sun_gravity == True
        assert s.use_moon_gravity == True

        # Constructor kwargs
        s = sk.propsettings(gravity_degree=10, gravity_order=5, use_sun_gravity=False, use_moon_gravity=False)
        assert s.gravity_degree == 10
        assert s.gravity_order == 5
        assert s.use_sun_gravity == False
        assert s.use_moon_gravity == False

        # Setting gravity_degree should clamp gravity_order
        s = sk.propsettings(gravity_degree=8)
        assert s.gravity_degree == 8
        assert s.gravity_order == 8  # defaults to degree when not set
        s.gravity_degree = 3
        assert s.gravity_order == 3  # clamped down

        # Setting gravity_order > gravity_degree should raise
        s = sk.propsettings(gravity_degree=6)
        with pytest.raises(ValueError):
            s.gravity_order = 10

    def test_propagate_sun_moon_toggles(self):
        """Test that disabling sun/moon gravity changes propagation results"""
        starttime = sk.time(2015, 3, 20)
        stoptime = starttime + sk.duration.from_days(0.5)
        pos = np.array([sk.consts.geo_r, 0, 0])
        vel = np.array([0, m.sqrt(sk.consts.mu_earth / sk.consts.geo_r), 0])
        state0 = np.concatenate((pos, vel))

        settings_all = sk.propsettings()
        settings_no_sun = sk.propsettings(use_sun_gravity=False)
        settings_no_moon = sk.propsettings(use_moon_gravity=False)

        res_all = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_all)
        res_no_sun = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_no_sun)
        res_no_moon = sk.propagate(state0, starttime, end=stoptime, propsettings=settings_no_moon)

        # Each should produce different final positions
        diff_sun = np.linalg.norm(res_all.state[0:3] - res_no_sun.state[0:3])
        diff_moon = np.linalg.norm(res_all.state[0:3] - res_no_moon.state[0:3])
        assert diff_sun > 1.0, f"Disabling sun gravity should matter, diff = {diff_sun} m"
        assert diff_moon > 1.0, f"Disabling moon gravity should matter, diff = {diff_moon} m"
