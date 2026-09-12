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
        assert kep.argp * rad2deg == pytest.approx(53.38, 1.0e-3)
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

    def test_kepler_kwargs_construction(self):
        """All six elements can be passed by keyword, matching the stub."""
        k_pos = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7)
        k_kw = sk.kepler(a=7000e3, eccen=0.1, incl=0.5, raan=1.0, w=0.3, nu=0.7)
        assert k_kw == k_pos
        # Mixed positional / keyword
        k_mix = sk.kepler(7000e3, 0.1, 0.5, raan=1.0, w=0.3, nu=0.7)
        assert k_mix == k_pos
        # Anomaly by keyword
        k_ta = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, true_anomaly=0.7)
        assert k_ta == k_pos
        k_ma = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, mean_anomaly=0.7)
        assert k_ma.mean_anomaly == pytest.approx(0.7, abs=1e-12)
        k_ea = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, eccentric_anomaly=0.7)
        assert k_ea.eccentric_anomaly == pytest.approx(0.7, abs=1e-12)
        # Exactly one anomaly must be given
        with pytest.raises(ValueError):
            sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3)
        with pytest.raises(ValueError):
            sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7, mean_anomaly=0.7)
        with pytest.raises(TypeError):
            sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7, bogus=1.0)

    def test_anomaly_setters_reject_non_finite_and_high_e_converges(self):
        """Regression for the setter that used to spin forever on NaN.

        Non-finite input is now refused before reaching the solver (the
        solver itself is iteration-capped; see the Rust test
        ``test_mean2eccentric_nan_returns``), so no setter can leave a NaN
        element behind. A high-eccentricity solve still runs, capped, in a
        watchdog thread.
        """
        import threading

        k = sk.kepler(7000e3, 0.5, 0.5, 1.0, 0.3, 0.0)
        k0 = sk.kepler(7000e3, 0.5, 0.5, 1.0, 0.3, 0.0)
        for attr in ("mean_anomaly", "eccentric_anomaly"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                with pytest.raises(ValueError, match=attr):
                    setattr(k, attr, bad)
        assert k == k0
        with pytest.raises(ValueError):
            k.eccen = 1.5
        assert k == k0

        results = {}

        def worker():
            k.eccen = 0.999
            k.mean_anomaly = 6.0
            results["high_e"] = k.nu

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=10.0)
        assert not t.is_alive(), "mean_anomaly setter hung"
        assert m.isfinite(results["high_e"])

    def test_mean_anomaly_setter_roundtrip(self):
        """Set M, read it back: |dM| < 1e-12 at e = 0.9 over a full revolution."""
        k = sk.kepler(7000e3, 0.9, 0.5, 1.0, 0.3, 0.0)
        for i in range(64):
            m0 = 2 * m.pi * i / 64
            k.mean_anomaly = m0
            dm = (k.mean_anomaly - m0 + m.pi) % (2 * m.pi) - m.pi
            assert abs(dm) < 1e-12, f"M={m0}: dM={dm:e}"
            # E must be consistent with M via Kepler's equation
            ea = k.eccentric_anomaly
            assert (ea - 0.9 * m.sin(ea) - m0 + m.pi) % (2 * m.pi) - m.pi == pytest.approx(
                0.0, abs=1e-12
            )
        # Eccentric-anomaly setter round-trip
        for i in range(64):
            e0 = 2 * m.pi * i / 64
            k.eccentric_anomaly = e0
            de = (k.eccentric_anomaly - e0 + m.pi) % (2 * m.pi) - m.pi
            assert abs(de) < 1e-12

    def test_kepler_pickle_roundtrip(self):
        import pickle

        k = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7, mu=sk.consts.mu_moon)
        k2 = pickle.loads(pickle.dumps(k))
        assert k2 == k
        assert (k2.a, k2.eccen, k2.inclination, k2.raan, k2.argp, k2.nu, k2.mu) == (
            k.a,
            k.eccen,
            k.inclination,
            k.raan,
            k.argp,
            k.nu,
            k.mu,
        )
        # mu is part of the state: a different central body is a different set
        k_earth = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7)
        assert k_earth != k
        assert k_earth.mu == sk.consts.mu_earth

    def test_kepler_validation_raises_value_error(self):
        good = dict(a=7000e3, eccen=0.1, incl=0.5, raan=1.0, argp=0.3, nu=0.7)
        sk.kepler(**good)
        for bad in (
            dict(a=0.0),
            dict(a=-7000e3),
            dict(a=float("nan")),
            dict(eccen=1.0),
            dict(eccen=-1e-3),
            dict(eccen=float("inf")),
            dict(incl=-1e-9),
            dict(incl=m.pi + 1e-9),
            dict(raan=float("nan")),
            dict(nu=float("nan")),
            dict(mu=0.0),
            dict(mu=-1.0),
        ):
            with pytest.raises(ValueError):
                sk.kepler(**{**good, **bad})
        # Boundaries are inclusive where the domain is closed
        sk.kepler(**{**good, "eccen": 0.0, "incl": 0.0})
        sk.kepler(**{**good, "incl": m.pi})
        # Setters validate too, and leave the element set unchanged on failure
        k = sk.kepler(**good)
        for attr, val in (
            ("a", -1.0),
            ("a", float("nan")),
            ("eccen", 1.0),
            ("inclination", 4.0),
            ("mu", 0.0),
            ("raan", float("nan")),
            ("raan", float("inf")),
            ("argp", float("nan")),
            ("nu", float("-inf")),
        ):
            with pytest.raises(ValueError):
                setattr(k, attr, val)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            k.w = float("nan")
        assert k == sk.kepler(**good)
        # Finite angles of any size are accepted (stored as given)
        k.raan = -10.0
        k.argp = 100.0
        k.nu = 7.0
        assert (k.raan, k.argp, k.nu) == (-10.0, 100.0, 7.0)

    def test_kepler_from_pv_open_or_rectilinear_is_value_error(self):
        r = np.array([7000e3, 0.0, 0.0])
        # Escape velocity and beyond: hyperbolic
        v_esc = m.sqrt(2 * sk.consts.mu_earth / 7000e3)
        with pytest.raises(ValueError, match="[Ee]ccentricity"):
            sk.kepler.from_pv(r, np.array([0.0, 1.5 * v_esc, 0.0]))
        # Parallel r and v: zero angular momentum
        with pytest.raises(ValueError, match="angular momentum"):
            sk.kepler.from_pv(r, np.array([1000.0, 0.0, 0.0]))
        # Malformed input is still a RuntimeError (shape, not physics)
        with pytest.raises(RuntimeError):
            sk.kepler.from_pv(np.zeros(4), np.zeros(3))

    def test_kepler_argp_and_deprecated_w(self):
        k_argp = sk.kepler(7000e3, 0.1, 0.5, 1.0, argp=0.3, nu=0.7)
        k_w = sk.kepler(7000e3, 0.1, 0.5, 1.0, w=0.3, nu=0.7)
        k_pos = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7)
        assert k_argp == k_w == k_pos
        assert k_argp.argp == 0.3
        with pytest.raises(ValueError):
            sk.kepler(7000e3, 0.1, 0.5, 1.0, argp=0.3, w=0.3, nu=0.7)
        with pytest.raises(TypeError):
            sk.kepler(7000e3, 0.1, 0.5, 1.0, nu=0.7)
        with pytest.warns(DeprecationWarning, match="argp"):
            assert k_argp.w == 0.3
        with pytest.warns(DeprecationWarning, match="argp"):
            k_argp.w = 0.4
        assert k_argp.argp == 0.4

    def test_kepler_mu(self):
        k = sk.kepler(2000e3, 0.05, 1.0, 0.2, 0.3, 0.4)
        k_moon = sk.kepler(2000e3, 0.05, 1.0, 0.2, 0.3, 0.4, mu=sk.consts.mu_moon)
        assert k.mu == sk.consts.mu_earth
        assert k_moon.mu == sk.consts.mu_moon
        assert k_moon.period / k.period == pytest.approx(
            m.sqrt(sk.consts.mu_earth / sk.consts.mu_moon), rel=1e-12
        )
        assert k_moon.period == pytest.approx(8022.0, abs=5.0)
        # from_pv with the same mu round-trips; with Earth's it does not
        r, v = k_moon.to_pv()
        back = sk.kepler.from_pv(r, v, mu=sk.consts.mu_moon)
        assert back.mu == sk.consts.mu_moon
        assert back.a == pytest.approx(k_moon.a, rel=1e-9)
        assert abs(sk.kepler.from_pv(r, v).a - k_moon.a) / k_moon.a > 0.1
        with pytest.raises(ValueError):
            sk.kepler.from_pv(r, v, mu=0.0)
        # propagate keeps mu, and the mu setter works
        assert k_moon.propagate(10.0).mu == sk.consts.mu_moon
        k.mu = sk.consts.mu_moon
        assert k == k_moon

    def test_kepler_derived_helpers(self):
        a, e = 26600e3, 0.74
        k = sk.kepler(a, e, 1.1, 5.0, 4.0, 0.0)
        assert k.periapsis == pytest.approx(a * (1 - e))
        assert k.apoapsis == pytest.approx(a * (1 + e))
        assert k.specific_energy == pytest.approx(-sk.consts.mu_earth / (2 * a), rel=1e-12)
        r, v = k.to_pv()
        assert k.angular_momentum == pytest.approx(np.linalg.norm(np.cross(r, v)), rel=1e-12)
        assert k.flight_path_angle == 0.0
        k_out = sk.kepler(a, e, 1.1, 5.0, 4.0, 1.0)
        r, v = k_out.to_pv()
        gamma = m.asin(np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v)))
        assert k_out.flight_path_angle == pytest.approx(gamma, abs=1e-12)
        assert k_out.flight_path_angle > 0
        assert k_out.argument_of_latitude == pytest.approx(5.0, abs=1e-12)
        assert k_out.true_longitude == pytest.approx(10.0 - 2 * m.pi, abs=1e-12)

    def test_kepler_repr_and_satstate_from_kepler(self):
        k = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7)
        text = repr(k)
        assert text.startswith("kepler(a=7.000000e6") and "argp=" in text and "mu=" in text
        assert "\n" not in text
        assert str(k).startswith("Keplerian Elements:")
        t0 = sk.time(2024, 1, 1)
        s = sk.satstate.from_kepler(t0, k)
        r, v = k.to_pv()
        np.testing.assert_allclose(s.pos, r)
        np.testing.assert_allclose(s.vel, v)
        assert s.time == t0

    def test_kepler_propagate_accepts_int_and_duration(self):
        k = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.3, 0.7)
        k_f = k.propagate(600.0)
        k_i = k.propagate(600)
        k_d = k.propagate(sk.duration.from_minutes(10))
        assert k_i == k_f
        assert k_d.nu == pytest.approx(k_f.nu, abs=1e-12)
        with pytest.raises(TypeError):
            k.propagate("600")

    def test_kepler_from_pv_tiny_inclination(self):
        """from_pv keeps precision for e up to 0.999 and i down to 1e-9 rad."""
        for eccen in (0.0, 0.5, 0.99, 0.999):
            for incl in (1e-9, 1e-6, 0.3):
                for nu in (0.0, 1.0, m.pi, 4.0):
                    k = sk.kepler(12000e3, eccen, incl, 1.1, 0.7, nu)
                    r, v = k.to_pv()
                    k2 = sk.kepler.from_pv(r, v)
                    assert k2.inclination == pytest.approx(incl, rel=1e-6, abs=1e-15)
                    assert k2.eccen == pytest.approx(eccen, abs=1e-9)
                    r2, v2 = k2.to_pv()
                    assert np.linalg.norm(r - r2) / np.linalg.norm(r) < 1e-6
                    assert np.linalg.norm(v - v2) / np.linalg.norm(v) < 1e-6


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


class TestNewBindings:
    def test_kepler_semiparameter(self):
        k = sk.kepler(7000e3, 0.1, 0.5, 1.0, 0.5, 0.0)
        assert k.semiparameter == pytest.approx(7000e3 * (1 - 0.1**2), rel=1e-12)

    def test_itrfcoord_distance_to(self):
        boston = sk.itrfcoord(latitude_deg=42.36, longitude_deg=-71.06, altitude=0)
        nyc = sk.itrfcoord(latitude_deg=40.71, longitude_deg=-74.01, altitude=0)
        d = boston.distance_to(nyc)
        dist, _, _ = boston.geodesic_distance(nyc)
        assert d == pytest.approx(dist, rel=1e-12)
        assert boston.distance_to(boston) == pytest.approx(0.0, abs=1e-6)

    def test_quaternion_new_methods(self):
        import math

        # from_euler is the inverse of as_euler
        q = sk.quaternion.from_euler(0.1, -0.2, 0.3)
        roll, pitch, yaw = q.as_euler()
        assert roll == pytest.approx(0.1, abs=1e-12)
        assert pitch == pytest.approx(-0.2, abs=1e-12)
        assert yaw == pytest.approx(0.3, abs=1e-12)

        # identity rotates nothing
        qi = sk.quaternion.identity()
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(qi * v, v)

        # norm / normalize / inverse / dot
        qr = sk.quaternion.rotz(math.radians(30))
        assert qr.norm() == pytest.approx(1.0, abs=1e-12)
        assert qr.normalize().norm() == pytest.approx(1.0, abs=1e-12)
        # inverse of a unit quaternion undoes the rotation
        assert np.allclose(qr.inverse() * (qr * v), v)
        assert qr.dot(qr) == pytest.approx(1.0, abs=1e-12)
        # slerp no longer takes epsilon
        q_mid = sk.quaternion.rotz(0.0).slerp(sk.quaternion.rotz(1.0), 0.5)
        assert q_mid.angle == pytest.approx(0.5, abs=1e-9)
