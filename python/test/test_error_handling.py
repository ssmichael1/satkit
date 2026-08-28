"""
Regression tests: invalid or unusual inputs must raise clean Python
exceptions rather than Rust panics (pyo3.PanicException), and valid
non-contiguous numpy input (strided views, Fortran order) must work.
"""

import numpy as np
import pytest

import satkit as sk


class TestInvalidInputRaises:
    def test_propagate_wrong_size_state(self):
        t0 = sk.time(2015, 3, 20)
        t1 = t0 + sk.duration.from_minutes(30)
        with pytest.raises(RuntimeError):
            sk.propagate(np.zeros(5), t0, t1)

    def test_kepler_from_pv_wrong_sizes(self):
        with pytest.raises(RuntimeError):
            sk.kepler.from_pv(np.zeros(4), np.zeros(3))

    def test_kepler_non_numeric_positional(self):
        with pytest.raises(TypeError):
            sk.kepler("7000e3", 0, 0, 0, 0, 0)

    def test_gravity_wrong_size(self):
        with pytest.raises(RuntimeError):
            sk.gravity(np.zeros(2))

    def test_quaternion_axis_angle_wrong_size(self):
        with pytest.raises(ValueError):
            sk.quaternion.from_axis_angle(np.array([1.0]), 0.5)

    def test_tle_from_lines_empty(self):
        with pytest.raises(ValueError):
            sk.TLE.from_lines([])

    def test_tle_from_lines_no_valid_tles(self):
        with pytest.raises(ValueError):
            sk.TLE.from_lines(["not a tle"])

    def test_tle_huge_day_of_year(self):
        line1 = "1 26900U 01039A   06-99999999999  .00000045  00000-0  10000-3 0  8290"
        line2 = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."
        with pytest.raises(RuntimeError):
            sk.TLE.from_lines([line1, line2])

    def test_sgp4_empty_list(self):
        with pytest.raises(RuntimeError):
            sk.sgp4([], sk.time(2015, 3, 20))

    def test_heliocentric_pos_invalid_body(self):
        # Sun/Moon have no heliocentric position; must raise, not panic
        # (the array-of-times path used to panic with the GIL released)
        with pytest.raises(RuntimeError):
            sk.planets.heliocentric_pos(sk.solarsystem.Moon, sk.time.now())
        with pytest.raises(RuntimeError):
            sk.planets.heliocentric_pos(
                sk.solarsystem.Sun, [sk.time.now(), sk.time.now()]
            )

    def test_time_from_rfc3339_non_ascii(self):
        with pytest.raises(ValueError):
            sk.time.from_rfc3339("ααα:00")

    def test_time_arithmetic_huge_int(self):
        # A Python int too large for f64 must raise OverflowError, not panic
        with pytest.raises(OverflowError):
            sk.time.now() + 10**400
        with pytest.raises(OverflowError):
            sk.time.now() - 10**400

    def test_sgp4_wrong_size_state(self):
        # propagate with an empty time list must raise cleanly
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        tle = sk.TLE.from_lines([line1, line2])
        with pytest.raises(RuntimeError):
            sk.sgp4([tle], [])


class TestNonContiguousInput:
    def test_itrfcoord_strided(self):
        x = np.array([7.0e6, 0, 0, 0.0, 0, 0, 0.0, 0, 0])
        c = sk.itrfcoord(x[::3])
        assert c.altitude > 0

    def test_gravity_strided(self):
        x = np.array([7.0e6, 0, 0, 0.0, 0, 0, 0.0, 0, 0])
        g = sk.gravity(x[::3])
        assert np.linalg.norm(g) > 1

    def test_satstate_fortran_order_cov(self):
        t0 = sk.time(2015, 3, 20)
        s = sk.satstate(
            t0,
            np.array([7.0e6, 0, 0]),
            np.array([0, 7.5e3, 0]),
            cov=np.asfortranarray(np.eye(6)),
        )
        assert np.allclose(s.cov, np.eye(6))

    def test_satstate_strided_sigma(self):
        t0 = sk.time(2015, 3, 20)
        s = sk.satstate(t0, np.array([7.0e6, 0, 0]), np.array([0, 7.5e3, 0]))
        s.set_pos_uncertainty(np.ones(9)[::3], sk.frame.RTN)
        assert s.cov is not None

    def test_gravity_degree_above_max_rejected(self):
        # Degrees above 40 used to be accepted and silently evaluated at 40.
        s = sk.propsettings()
        s.gravity_degree = 40
        s.gravity_order = 40
        with pytest.raises(ValueError):
            s.gravity_degree = 41
        with pytest.raises(ValueError):
            s.gravity_order = 41
        with pytest.raises(ValueError):
            sk.propsettings(gravity_degree=360, gravity_order=360)
        assert (s.gravity_degree, s.gravity_order) == (40, 40)

    def test_precompute_table_size_cap(self):
        # A tiny step would need billions of entries; must raise, not OOM.
        s = sk.propsettings()
        t0 = sk.time(2023, 5, 16, 20, 0, 0)
        with pytest.raises(RuntimeError, match="entries"):
            s.precompute_terms(t0, t0 + sk.duration.from_hours(1), 1e-6)

