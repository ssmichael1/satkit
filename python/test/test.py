# %%
import pytest
import numpy as np
import math as m
import os
from sp3file import read_sp3file
from datetime import datetime, timezone
import satkit as sk

testvec_dir = os.getenv(
    "SATKIT_TESTVEC_ROOT", default="." + os.path.sep + "satkit-testvecs" + os.path.sep
)


class TestDateTime:
    """
    Check that function calls with satkit.time and datetime.datetime return
    the same result
    """

    def test_scalar_times(self):

        # Create times and show that they are equal
        tm1 = sk.time(2023, 3, 4, 12, 5, 6)
        tm2 = datetime(2023, 3, 4, 12, 5, 6, tzinfo=timezone.utc)

        assert tm1.datetime() == tm2
        # Check that function calls work
        # Pick gmst as the test function call for time
        # it can be anything since under the hood the same function call is used
        # to get time inputs for all python functions in package
        g1 = sk.frametransform.gmst(tm1)
        g2 = sk.frametransform.gmst(tm2)
        assert g1 == pytest.approx(g2, rel=1e-10)

    def test_list_times(self):
        timearr = range(10)
        tm1 = [sk.time(2023, 3, x + 1, 12, 0, 0) for x in timearr]
        tm2 = [datetime(2023, 3, x + 1, 12, 0, 0, tzinfo=timezone.utc) for x in timearr]
        g1 = sk.frametransform.gmst(tm1)
        g2 = sk.frametransform.gmst(tm2)
        assert g1 == pytest.approx(g2)

    def test_numpy_times(self):
        timearr = range(10)
        tm1 = np.array([sk.time(2023, 3, x + 1, 12, 0, 0) for x in timearr])
        tm2 = np.array(
            [datetime(2023, 3, x + 1, 12, 0, 0, tzinfo=timezone.utc) for x in timearr]
        )
        g1 = sk.frametransform.gmst(tm1)
        g2 = sk.frametransform.gmst(tm2)
        assert g1 == pytest.approx(g2)


class TestTime:

    def test_rfc3339(self):
        """
        Test RFC3339 conversion
        """
        t = sk.time(2021, 9, 30, 12, 45, 13.345)
        assert t.as_rfc3339() == "2021-09-30T12:45:13.345000Z"

    def test_mjd(self):
        """
        Test MJD conversion
        """
        t = sk.time(2021, 1, 1, 0, 0, 0)
        mjd = t.as_mjd(sk.timescale.UTC)
        assert mjd == pytest.approx(59215.0)

    def test_jd(self):
        """
        Test JD conversion
        """
        t = sk.time(2021, 1, 1, 0, 0, 0)
        jd = t.as_jd(sk.timescale.UTC)
        assert jd == pytest.approx(2459215.5)

    def test_duration(self):
        """
        Test duration conversion
        """
        d1 = sk.duration.from_seconds(86400)
        assert d1.seconds == 86400
        assert d1.days == 1.0

        d2 = sk.duration.from_days(1)
        assert d1 == d2
        assert d1 >= d2
        assert d1 <= d2
        assert d1 + d1 > d2
        assert d1 - d1 == sk.duration.from_seconds(0)
        assert d1 - sk.duration.from_seconds(43200) == sk.duration.from_seconds(43200)
        assert d2 < d1 + d1
        assert d1 != d2 + d1
        assert d1 < d1 + d2
        assert d1 + d2 > d1

    def test_comparison_operators(self):

        t1 = sk.time(2021, 1, 1, 0, 0, 0)
        t2 = sk.time(2021, 1, 1, 0, 0, 0)
        d = sk.duration.from_days(1)
        assert t1 == t2
        assert t1 + d > t2
        assert t1 - d < t2
        assert t1 >= t2
        assert t1 <= t2
        assert t1 != sk.time(2020, 12, 31, 0, 0, 0)

    def test_time_diff(self):
        """
        Test time difference
        """
        t1 = sk.time(2021, 1, 1, 0, 0, 0)
        t2 = sk.time(2021, 1, 2, 0, 0, 0)
        d = t2 - t1
        assert d.days == 1.0

    def test_time_add(self):
        """
        Test time addition
        """
        t1 = sk.time(2021, 1, 1, 0, 0, 0)
        d = sk.duration.from_days(1)
        t2 = t1 + d
        assert t2 == sk.time(2021, 1, 2, 0, 0, 0)

    def test_time_sub(self):
        """
        Test time subtraction
        """
        t1 = sk.time(2021, 1, 1, 0, 0, 0)
        d = sk.duration.from_days(1)
        t2 = t1 - d
        assert t2 == sk.time(2020, 12, 31, 0, 0, 0)

    def test_time_gregorian(self):
        """
        Test conversion to Gregorian calendar
        """
        t = sk.time(2021, 1, 1, 0, 0, 0)
        (year, mon, day, hour, minute, sec) = t.as_gregorian()
        assert year == 2021
        assert mon == 1
        assert day == 1
        assert hour == 0
        assert minute == 0
        assert sec == 0


class TestKepler:
    def test_kepler_from_pv(self):
        """
        Test creation of Kepler elements from position and velocity
        """

        # Test case from Vallado, example 2-6
        r = np.array([6524.834, 6862.875, 6448.296]) * 1.0e3
        v = np.array([4.901327, 5.533756, -1.976341]) * 1.0e3
        kep = sk.kepler.from_pv(r, v)
        rad2deg = 180.0 / m.pi
        print(kep)
        assert kep.a == pytest.approx(36127343, 1.0e-3)
        assert kep.eccen == pytest.approx(0.83285, 1.0e-5)
        assert kep.inclination * rad2deg == pytest.approx(87.87, 1.0e-3)
        assert kep.raan * rad2deg == pytest.approx(227.89, 1.0e-3)
        assert kep.w * rad2deg == pytest.approx(53.38, 1.0e-3)
        assert kep.nu * rad2deg == pytest.approx(92.335, 1.0e-3)

    def test_kepler_to_pv(self):
        """
        Test conversion of Kepler elements to position and velocity
        """
        p = 11067790
        eccen = 0.83285
        incl = 87.87 * m.pi / 180
        raan = 227.89 * m.pi / 180
        w = 53.38 * m.pi / 180
        nu = 92.335 * m.pi / 180

        a = p / (1 - eccen**2)
        kep = sk.kepler(a, eccen, incl, raan, w, nu)
        pos, vel = kep.to_pv()
        assert pos == pytest.approx(
            np.array([6525.368, 6861.532, 6449.119]) * 1.0e3, 1.0e-3
        )
        assert vel == pytest.approx(
            np.array([4.902279, 5.533140, -1.975710]) * 1.0e3, 1.0e-3
        )


class TestJPLEphem:
    def test_jplephem_testvecs(self):
        """
        Test JPL ephemeris against test vectors provided by JPL
        """

        # File contains test calculation vectors provided by NASA

        fname = testvec_dir + os.path.sep + "jplephem" + os.path.sep + "testpo.440"

        # Read in the test vectors
        with open(fname, "r") as fd:
            lines = fd.readlines()

        # Function to convert integer index to solar system body
        def int_to_ss(ix: int) -> sk.solarsystem:
            if ix == 0:
                return sk.solarsystem.Mercury
            elif ix == 1:
                return sk.solarsystem.Venus
            elif ix == 2:
                return sk.solarsystem.EMB
            elif ix == 3:
                return sk.solarsystem.Mars
            elif ix == 4:
                return sk.solarsystem.Jupiter
            elif ix == 5:
                return sk.solarsystem.Saturn
            elif ix == 6:
                return sk.solarsystem.Uranus
            elif ix == 7:
                return sk.solarsystem.Neptune
            elif ix == 8:
                return sk.solarsystem.Pluto
            elif ix == 9:
                return sk.solarsystem.Moon
            elif ix == 10:
                return sk.solarsystem.Sun
            else:
                raise ValueError(f"Unknown solar system body index: {ix}")

        # Go through the test vectors
        # each test vecxtor is a line in the file
        for line in lines[14:]:
            s = line.split()
            assert len(s) >= 7
            # get the fields in the test vector
            jd = float(s[2])
            tar = int(s[3])
            src = int(s[4])
            coord = int(s[5])
            truth = float(s[6])
            time = sk.time.from_jd(jd, sk.timescale.TT)
            # Don't handle any of the exotic test vectors, just do sun, moon,
            # and planetary ephemerides
            if tar <= 10 and src <= 10 and coord <= 6:
                sksrc = int_to_ss(src - 1)
                sktar = int_to_ss(tar - 1)
                tpos, tvel = sk.jplephem.geocentric_state(sktar, time)
                spos, svel = sk.jplephem.geocentric_state(sksrc, time)

                # In test vectors, index 3 is not EMB, but Earth
                # (not obvious...)
                if tar == 3:
                    _mpos, mvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.Moon, time
                    )
                    tvel = tvel - mvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    tpos = np.array([0, 0, 0])
                if src == 3:
                    spos = np.array([0, 0, 0])
                    _mpos, mvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.Moon, time
                    )
                    svel = svel - mvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                if src == 10:
                    embpos, embvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.EMB, time
                    )
                    svel = svel + (
                        embvel - svel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    )
                if tar == 10:
                    embpos, embvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.EMB, time
                    )
                    tvel = tvel + (
                        embvel - tvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    )
                # Position test
                if coord <= 3:
                    calc = (tpos - spos)[coord - 1] / sk.consts.au
                    assert calc == pytest.approx(truth, rel=1e-12)
                # Velocity test
                else:
                    calc = (tvel - svel)[coord - 4] / sk.consts.au * 86400.0
                    assert calc == pytest.approx(truth, rel=1e-12)


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
        gravitation = sk.gravity(itrf, order=16)
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


class TestITRFCoord:
    def test_geodetic(self):
        """
        Test geodetic conversions
        """
        latitude_deg = 42.46
        longitude_deg = -71.1516
        altitude = 1000
        itrf = sk.itrfcoord(
            latitude_deg=latitude_deg, longitude_deg=longitude_deg, altitude=altitude
        )
        assert itrf.latitude_deg == pytest.approx(latitude_deg)
        assert itrf.longitude_deg == pytest.approx(longitude_deg)
        assert itrf.altitude == pytest.approx(altitude)

    def test_geodetic2(self):
        """
        Vallado example 3.3
        """
        itrf = sk.itrfcoord(6524.834 * 1e3, 6862.875 * 1e3, 6448.296 * 1e3)
        assert itrf.latitude_deg == pytest.approx(34.352496)
        assert itrf.longitude_deg == pytest.approx(46.4464)


class TestMoon:
    def test_moonpos(self):
        """
        Vallado example 5-3 for
        computing position of the moon
        """
        t0 = sk.time(1994, 4, 28)
        # Vallado approximates UTC as TBD, so we will
        # make the same approximation
        # for the purposes of this test case
        t1 = sk.time.from_mjd(t0.as_mjd(sk.timescale.UTC), sk.timescale.TDB)
        p = sk.moon.pos_gcrf(t1)
        ref_pos = np.array([-134240.626e3, -311571.590e3, -126693.785e3])
        assert p == pytest.approx(ref_pos)


class TestSun:
    def test_sunpos_mod(self):
        """
        Vallado example 5-1 for computing position of sun
        """
        t0 = sk.time(2006, 4, 2)
        # Vallado approximates UTC as TBD, so we will
        # make the same approximation
        # for the purposes of this test case
        t1 = sk.time.from_mjd(t0.as_mjd(sk.timescale.UTC), sk.timescale.TDB)
        p = sk.sun.pos_gcrf(t1)
        pref = np.array([146259922.0e3, 28585947.0e3, 12397430.0e3])
        assert p == pytest.approx(pref, 5e-4)

    def test_sun_rise_set(self):
        """
        Vallado example 5-2
        """
        coord = sk.itrfcoord(latitude_deg=40.0, longitude_deg=0.0)
        tm = sk.time(1996, 3, 23, 0, 0, 0)
        sunrise, sunset = sk.sun.rise_set(tm, coord)
        (year, mon, day, hour, minute, sec) = sunrise.as_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 5
        assert minute == 58
        assert sec == pytest.approx(21.97, 1e-3)
        (year, mon, day, hour, minute, sec) = sunset.as_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 18
        assert minute == 15
        assert sec == pytest.approx(17.76, 1.0e-3)

    def test_sun_rise_set_error(self):
        coord = sk.itrfcoord(latitude_deg=85.0, longitude_deg=30.0)
        tm = sk.time(2020, 6, 20)
        try:
            sunrise, sunset = sk.sun.rise_set(tm, coord)
        except:
            # This should throw exception ... there is no sunrise or sunset
            # at this time of year at the specified location; sun is always up
            pass
        else:
            assert 1 == 0


class TestQuaternion:
    def test_rotations(self):
        """
        Test coordinate rotations with quaternions
        """
        xhat = np.array([1.0, 0.0, 0.0])
        yhat = np.array([0.0, 1.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])

        # Test rotations of 90 degrees with right-hande rule of 3 coordinate axes
        assert sk.quaternion.rotz(m.pi / 2) * xhat == pytest.approx(yhat, 1.0e-10)
        assert sk.quaternion.rotx(m.pi / 2) * yhat == pytest.approx(zhat, 1.0e-10)
        assert sk.quaternion.roty(m.pi / 2) * zhat == pytest.approx(xhat, 1.0e-10)

    def test_dcm_conversion(self):
        xhat = np.array([1.0, 0.0, 0.0])
        yhat = np.array([0.0, 1.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])

        # Test rotations of 90 degrees with right-hande rule of 3 coordinate axes
        assert sk.quaternion.rotz(
            m.pi / 2
        ).as_rotation_matrix() @ xhat == pytest.approx(yhat, 1.0e-10)
        assert sk.quaternion.rotx(
            m.pi / 2
        ).as_rotation_matrix() @ yhat == pytest.approx(zhat, 1.0e-10)
        assert sk.quaternion.roty(
            m.pi / 2
        ).as_rotation_matrix() @ zhat == pytest.approx(xhat, 1.0e-10)

    def test_dcm2quaternion(self):
        """
        Test conversion of DCM to quaternion
        """
        xhat = np.array([1.0, 0.0, 0.0])
        yhat = np.array([0.0, 1.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])

        q = sk.quaternion.from_rotation_matrix(
            sk.quaternion.rotz(m.pi / 2).as_rotation_matrix()
        )
        assert q * xhat == pytest.approx(yhat, 1.0e-10)

    def test_quaternion2dcm(self):
        """
        Test conversion of quaternion to DCM
        """
        xhat = np.array([1.0, 0.0, 0.0])
        yhat = np.array([0.0, 1.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])
        q = sk.quaternion.from_rotation_matrix(
            sk.quaternion.rotz(m.pi / 2).as_rotation_matrix()
        )
        dcm = q.as_rotation_matrix()
        assert dcm @ xhat == pytest.approx(yhat, 1.0e-10)

    def test_quaternion2euler(self):
        """
        Test conversion of quaternion to Euler angles
        """
        q = sk.quaternion.rotz(m.pi / 3)
        euler = q.as_euler()
        assert euler[0] == pytest.approx(0.0)
        assert euler[1] == pytest.approx(0.0)
        assert euler[2] == pytest.approx(m.pi / 3)

        q = sk.quaternion.rotx(m.pi / 3)
        euler = q.as_euler()
        assert euler[0] == pytest.approx(m.pi / 3)
        assert euler[1] == pytest.approx(0.0)
        assert euler[2] == pytest.approx(0.0)

        q = sk.quaternion.roty(m.pi / 3)
        euler = q.as_euler()
        assert euler[0] == pytest.approx(0.0)
        assert euler[1] == pytest.approx(m.pi / 3)
        assert euler[2] == pytest.approx(0.0)


class TestGeodesicDistance:

    newyork = sk.itrfcoord(latitude_deg=40.6446, longitude_deg=-73.7797)
    london = sk.itrfcoord(latitude_deg=51.4680, longitude_deg=0.4551)

    def test_geodesic_distance(self):
        """
        Check distances between two locations
        """

        [dist, heading_start, heading_end] = self.newyork.geodesic_distance(self.london)
        [dist2, heading2_start, heading2_end] = self.london.geodesic_distance(
            self.newyork
        )

        # Check that distances and headings match going in reverse direction
        assert dist == pytest.approx(dist2, 1.0e-8)
        assert heading_start - m.pi == pytest.approx(heading2_end, 1.0e-6)
        assert heading_end - m.pi == pytest.approx(heading2_start, 1.0e-6)

        # per google new york to london distance is 3,459 miles
        # Convert to meters
        print(f"dist = {dist}")
        dist_ref = 3459 * 5280 * 12 * 2.54 / 100
        assert dist == pytest.approx(dist_ref, 1.0e-2)

    def test_heading_dist(self):
        """
        test that moving a distance at a given heading along surface of
        Earth calculation is correct
        """
        [dist, heading_start, heading_end] = self.newyork.geodesic_distance(self.london)
        loc2 = self.newyork.move_with_heading(dist, heading_start)
        diff = self.london - loc2
        assert np.linalg.norm(diff) < 1e-8


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
            np.concatenate((pos, vel)), starttime, stop=stoptime, propsettings=settings
        )
        # Propagate backward and see if we recover original result
        res2 = sk.propagate(res1.state, stoptime, stop=starttime, propsettings=settings)

        assert res2.state[0:3] == pytest.approx(pos, abs=0.5)
        assert res2.state[3:6] == pytest.approx(vel, abs=1e-5)

        newtime = starttime + sk.duration.from_hours(4.332)
        istate1 = res1.interp(newtime)
        istate2 = res2.interp(newtime)

        assert istate1 == pytest.approx(istate2, rel=1e-7)

    def test_gps(self):

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
        satprops = sk.satproperties_static()
        satprops.craoverm = fitparam[3]  # type: ignore

        res = sk.propagate(
            np.concatenate((pgcrf[0, :], fitparam[0:3])),
            timearr[0],
            stop=timearr[-1],
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


class TestSGP4:
    def test_sgp4_multiple(self):
        """
        Check propagating multiple TLEs at once
        """

        lines = [
            "0 STARLINK-3118",
            "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
            "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
            "0 STARLINK-3093",
            "1 49141U 21082M   24030.50141584 -.00000431  00000-0 -28322-4 0  9990",
            "2 49141  70.0000  73.8654 0002647 256.8611 103.2253 14.98324813131968",
            "0 STARLINK-3042",
            "1 49142U 21082N   24030.19218442  .00000448  00000-0  45331-4 0  9999",
            "2 49142  70.0005  34.6319 0002749 265.6056  94.4790 14.98327526131704",
            "0 STARLINK-3109",
            "1 49143U 21082P   24030.20076173 -.00000320  00000-0 -19071-4 0  9998",
            "2 49143  70.0002  54.6139 0002526 255.5608 104.5271 14.98327699131201",
        ]
        tles = sk.TLE.from_lines(lines)
        print(tles)
        tm = [
            sk.time(2024, 1, 15) + sk.duration.from_seconds(x * 10) for x in range(100)
        ]
        [p, v] = sk.sgp4(tles, tm)
        [p2, v2] = sk.sgp4(tles[2], tm)  # type: ignore
        # Verify that propagating multiple TLEs matches propagation of a single TLE
        assert p2 == pytest.approx(np.squeeze(p[2, :, :]))
        assert v2 == pytest.approx(np.squeeze(v[2, :, :]))

    def test_sgp4_vallado(self):
        """
        SGP4 Test Vectors from vallado
        """

        basedir = testvec_dir + os.path.sep + "sgp4"

        tlefile = basedir + os.path.sep + "SGP4-VER.TLE"
        with open(tlefile, "r") as fh:
            lines = fh.readlines()

        lines = list(filter(lambda x: x[0] != "#", lines))

        lines = [l.strip() for l in lines]
        lines = [l[0:69] for l in lines]

        tles = sk.TLE.from_lines(lines)
        for tle in tles:  # type: ignore
            fname = f"{basedir}{os.path.sep}{tle.satnum:05}.e"
            with open(fname, "r") as fh:
                testvecs = fh.readlines()
            for testvec in testvecs:
                stringvals = testvec.split()

                # Valid lines are all floats of length 7
                if len(stringvals) != 7:
                    continue
                try:
                    vals = [float(s) for s in stringvals]
                except ValueError:
                    continue
                time = tle.epoch + sk.duration.from_seconds(vals[0])
                try:
                    [p, v, eflag] = sk.sgp4(  # type: ignore
                        tle,
                        time,
                        opsmode=sk.sgp4_opsmode.afspc,
                        gravconst=sk.sgp4_gravconst.wgs72,
                        errflag=True,
                    )

                    if eflag == sk.sgp4_error.success:
                        ptest = np.array([vals[1], vals[2], vals[3]]) * 1e3
                        vtest = np.array([vals[4], vals[5], vals[6]]) * 1e3
                        assert p == pytest.approx(ptest, rel=1e-4)
                        assert v == pytest.approx(vtest, rel=1e-2)
                    else:
                        # We know which one is supposed to fail in the test vectors
                        # Make sure we pick the correcxt one
                        assert tle.satnum == 33334
                        assert eflag == sk.sgp4_error.perturb_eccen
                except RuntimeError:
                    print("Caught runtime error; this is expected in test vectors")
