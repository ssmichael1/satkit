"""
Regression tests: invalid or unusual inputs must raise clean Python
exceptions rather than Rust panics (pyo3.PanicException), and valid
non-contiguous numpy input (strided views, Fortran order) must work.
"""

import numpy as np
import pytest

import satkit as sk
from shared import ISS_2021


class TestInvalidInputRaises:
    def test_propagate_wrong_size_state(self):
        t0 = sk.time(2015, 3, 20)
        t1 = t0 + sk.duration.from_minutes(30)
        with pytest.raises(ValueError):
            sk.propagate(np.zeros(5), t0, t1)

    def test_kepler_from_pv_wrong_sizes(self):
        with pytest.raises(RuntimeError):
            sk.kepler.from_pv(np.zeros(4), np.zeros(3))

    def test_kepler_non_numeric_positional(self):
        with pytest.raises(TypeError):
            sk.kepler("7000e3", 0, 0, 0, 0, 0)

    def test_gravity_wrong_size(self):
        with pytest.raises(ValueError):
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
        # An empty TLE list gives empty arrays, not an error
        p, v = sk.sgp4([], sk.time(2015, 3, 20))
        assert p.shape == v.shape == (0, 3)

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
        # An empty time list gives empty arrays, as sgp4(tle, []) does
        tle = sk.TLE.from_lines(ISS_2021)[0]
        p, v = sk.sgp4([tle], [])
        assert p.shape == v.shape == (1, 0, 3)


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
        # Degrees above the cap used to be accepted and silently evaluated
        # at the cap (40 before 0.23, 70 since).
        s = sk.propsettings()
        s.gravity_degree = 70
        s.gravity_order = 70
        with pytest.raises(ValueError):
            s.gravity_degree = 71
        with pytest.raises(ValueError):
            s.gravity_order = 71
        with pytest.raises(ValueError):
            sk.propsettings(gravity_degree=360, gravity_order=360)
        assert (s.gravity_degree, s.gravity_order) == (70, 70)

    def test_gravity_order_above_degree_rejected(self):
        # The constructor used to clamp the order to the degree silently,
        # while the setter raised.
        with pytest.raises(ValueError, match="gravity_order"):
            sk.propsettings(gravity_degree=4, gravity_order=10)
        s = sk.propsettings(gravity_degree=10, gravity_order=4)
        assert (s.gravity_degree, s.gravity_order) == (10, 4)
        assert sk.propsettings(gravity_degree=8).gravity_order == 8
        with pytest.raises(ValueError, match="gravity_order"):
            s.gravity_order = 11

    def test_precompute_table_size_cap(self):
        # A tiny step would need billions of entries; must raise, not OOM.
        s = sk.propsettings()
        t0 = sk.time(2023, 5, 16, 20, 0, 0)
        with pytest.raises(RuntimeError, match="entries"):
            s.precompute_terms(t0, t0 + sk.duration.from_hours(1), 1e-6)



class TestTleOnError:
    """`on_error` of the TLE loaders: keep the good records of a bad file"""

    GOOD_A = [
        "0 ISS (ZARYA)",
        "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9991",
        "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302878",
    ]
    # A letter O in the epoch day
    BAD = [
        "0 STARLINK-3118",
        "1 49140U 21082L   24O30.39663557  .00000076  00000-0  14180-4 0  9995",
        "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
    ]
    GOOD_B = [
        "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
        "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
    ]
    LINES = GOOD_A + BAD + GOOD_B

    def test_raise_is_default(self):
        with pytest.raises(RuntimeError, match="line 4"):
            sk.TLE.from_lines(self.LINES)
        with pytest.raises(RuntimeError, match="line 4"):
            sk.TLE.from_lines(self.LINES, on_error="raise")

    def test_skip(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tles = sk.TLE.from_lines(self.LINES, on_error="skip")
        assert [t.satnum for t in tles] == [25544, 49140]

    def test_warn(self):
        with pytest.warns(sk.TLEWarning) as rec:
            tles = sk.TLE.from_lines(self.LINES, on_error="warn")
        assert [t.satnum for t in tles] == [25544, 49140]
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert msg.startswith('TLE record starting at line 4 (sat 49140 "STARLINK-3118")')
        # The same text on_error="raise" raises
        with pytest.raises(RuntimeError) as exc:
            sk.TLE.from_lines(self.LINES)
        assert str(exc.value) == msg
        # Attributed to the calling line, and a UserWarning
        assert rec[0].filename == __file__
        assert issubclass(sk.TLEWarning, UserWarning)

    def test_warning_as_error(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", sk.TLEWarning)
            with pytest.raises(sk.TLEWarning):
                sk.TLE.from_lines(self.LINES, on_error="warn")

    def test_checksum_failures_are_skipped_too(self):
        # Line 1 of GOOD_A carries a wrong checksum digit (1, not 6)
        lines = self.GOOD_B + self.GOOD_A
        assert len(sk.TLE.from_lines(lines, on_error="skip")) == 2
        with pytest.warns(sk.TLEWarning, match="checksum"):
            tles = sk.TLE.from_lines(lines, check_checksum=True, on_error="warn")
        assert [t.satnum for t in tles] == [49140]

    def test_all_bad_is_an_empty_list(self):
        assert sk.TLE.from_lines(self.BAD, on_error="skip") == []

    def test_no_records_still_raises(self):
        for mode in ("raise", "warn", "skip"):
            with pytest.raises(ValueError):
                sk.TLE.from_lines([], on_error=mode)
            with pytest.raises(ValueError):
                sk.TLE.from_lines(["not a tle"], on_error=mode)

    def test_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            sk.TLE.from_lines(self.GOOD_A, on_error="ignore")

    def test_from_file(self, tmp_path):
        f = tmp_path / "cat.txt"
        f.write_text("\n".join(self.LINES) + "\n")
        with pytest.raises(RuntimeError):
            sk.TLE.from_file(str(f))
        assert len(sk.TLE.from_file(str(f), on_error="skip")) == 2
        with pytest.warns(sk.TLEWarning):
            assert len(sk.TLE.from_file(str(f), on_error="warn")) == 2
