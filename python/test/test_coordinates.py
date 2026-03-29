import pytest
import numpy as np
import math as m

import satkit as sk


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


    def test_ned_enu(self):

        """
        Test NED and ENU conversions
        """
        refcoord = sk.itrfcoord(
            latitude_deg=30.0, longitude_deg=-90.0, altitude=0.0
        )
        testcoord = sk.itrfcoord(
            latitude_deg=30.0, longitude_deg=-90.0, altitude=100.0
        )

        ned = testcoord.to_ned(refcoord)
        enu = testcoord.to_enu(refcoord)

        # Check NED values
        # North component
        assert ned[0] == pytest.approx(0, abs=1e-8)
        # East component
        assert ned[1] == pytest.approx(0, abs=1e-8)
        # Down component
        assert ned[2] == pytest.approx(-100.0, rel=1e-8)

        # Check ENU values
        # East component
        assert enu[0] == pytest.approx(0, abs=1e-8)
        # North component
        assert enu[1] == pytest.approx(0, abs=1e-8)
        # Up component
        assert enu[2] == pytest.approx(100.0, abs=1e-8)

        coord1 = sk.itrfcoord(latitude_deg=42.466, longitude_deg=-71.1516, altitude=10.0)

        # Go east 10 meters and check
        coord2 = sk.itrfcoord(coord1.vector + coord1.qenu2itrf * np.array([10.0, 0.0, 0.0]))
        enu = coord2.to_enu(coord1)
        assert enu[0] == pytest.approx(10.0, abs=1e-8)
        assert enu[1] == pytest.approx(0.0, abs=1e-8)
        assert enu[2] == pytest.approx(0.0, abs=1e-8)

        # Go north 10 meters and check
        coord2 = sk.itrfcoord(coord1.vector + coord1.qenu2itrf * np.array([0.0, 10.0, 0.0]))
        enu = coord2.to_enu(coord1)
        assert enu[0] == pytest.approx(0.0, abs=1e-8)
        assert enu[1] == pytest.approx(10.0, abs=1e-8)
        assert enu[2] == pytest.approx(0.0, abs=1e-8)

        # Go up 10 meters and check
        coord2 = sk.itrfcoord(coord1.vector + coord1.qenu2itrf * np.array([0.0, 0.0, 10.0]))
        enu = coord2.to_enu(coord1)
        assert enu[0] == pytest.approx(0.0, abs=1e-8)
        assert enu[1] == pytest.approx(0.0, abs=1e-8)
        assert enu[2] == pytest.approx(10.0, abs=1e-8)

        for ix in range(50):
            # Create random coordinates
            lat1 = np.random.uniform(-90.0, 90.0)
            lon1 = np.random.uniform(-180.0, 180.0)
            alt1 = np.random.uniform(100.0, 40000.0)
            lat2 = np.random.uniform(-90.0, 90.0)
            lon2 = np.random.uniform(-180.0, 180.0)
            alt2 = np.random.uniform(100.0, 40000.0)
            coord1 = sk.itrfcoord(latitude_deg=lat1, longitude_deg=lon1, altitude=alt1)
            coord2 = sk.itrfcoord(latitude_deg=lat2, longitude_deg=lon2, altitude=alt2)

            # Check to_ned, to_enu against manually computed values
            ned = coord2.to_ned(coord1)
            enu = coord2.to_enu(coord1)
            ned2 = coord1.qned2itrf.conj * (coord2-coord1)
            enu2 = coord1.qenu2itrf.conj * (coord2-coord1)
            assert ned[0] == pytest.approx(ned2[0], rel=1e-8)
            assert ned[1] == pytest.approx(ned2[1], rel=1e-8)
            assert ned[2] == pytest.approx(ned2[2], rel=1e-8)
            assert enu[0] == pytest.approx(enu2[0], rel=1e-8)
            assert enu[1] == pytest.approx(enu2[1], rel=1e-8)
            assert enu[2] == pytest.approx(enu2[2], rel=1e-8)


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

    def test_construction(self):
        """
        Test construction of quaternions from scalars
        """
        s = 1 / m.sqrt(2.0)
        q = sk.quaternion(s, s, 0, 0)
        assert q.axis == pytest.approx(np.array([1.0, 0.0, 0.0]), 1.0e-10)
        assert q.angle == pytest.approx(m.pi / 2, 1.0e-10)
        q = sk.quaternion(s, 0, s, 0)
        assert q.axis == pytest.approx(np.array([0.0, 1.0, 0.0]), 1.0e-10)
        assert q.angle == pytest.approx(m.pi / 2, 1.0e-10)
        q = sk.quaternion(s, 0, 0, s)
        assert q.axis == pytest.approx(np.array([0.0, 0.0, 1.0]), 1.0e-10)
        assert q.angle == pytest.approx(m.pi / 2, 1.0e-10)

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
