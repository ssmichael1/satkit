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
        gravitation = sk.gravity(itrf, degree=16)
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
