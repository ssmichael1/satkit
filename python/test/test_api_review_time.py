"""Regression tests for the Python API review: ``time``, ``duration``,
``timescale`` and ``weekday``."""

import math
from fractions import Fraction

import numpy as np
import pytest

import satkit as sk

NONFINITE = [math.nan, math.inf, -math.inf]


# ───────────────────── non-finite and out-of-range input ─────────────────────


class TestNonFiniteTimeInput:
    """NaN used to give the constructor's epoch (or leave the time unchanged)
    and infinities saturated at the year ±292,000"""

    @pytest.mark.parametrize("x", NONFINITE)
    def test_from_jd(self, x):
        with pytest.raises(ValueError, match="jd must be finite"):
            sk.time.from_jd(x)
        with pytest.raises(ValueError, match="jd must be finite"):
            sk.time.from_jd(x, sk.timescale.TT)

    @pytest.mark.parametrize("x", NONFINITE)
    def test_from_mjd(self, x):
        with pytest.raises(ValueError, match="mjd must be finite"):
            sk.time.from_mjd(x)

    @pytest.mark.parametrize("x", NONFINITE)
    def test_from_unixtime(self, x):
        with pytest.raises(ValueError, match="unixtime must be finite"):
            sk.time.from_unixtime(x)

    @pytest.mark.parametrize("x", NONFINITE)
    def test_from_gps_week_and_second(self, x):
        with pytest.raises(ValueError, match="seconds must be finite"):
            sk.time.from_gps_week_and_second(2000, x)

    @pytest.mark.parametrize("x", NONFINITE)
    def test_add_utc_days(self, x):
        with pytest.raises(ValueError, match="days must be finite"):
            sk.time(2024, 1, 1).add_utc_days(x)

    def test_out_of_range_is_overflow_error(self):
        # These saturated at +294135-11-26 / +294247-01-10
        for call in (
            lambda: sk.time.from_mjd(1e20),
            lambda: sk.time.from_mjd(-1e20, sk.timescale.TAI),
            lambda: sk.time.from_jd(1e20),
            lambda: sk.time.from_unixtime(1e300),
            lambda: sk.time.from_gps_week_and_second(2**31 - 1, 0.0),
            lambda: sk.time.from_gps_week_and_second(0, 1e300),
            lambda: sk.time(2024, 1, 1).add_utc_days(1e20),
            lambda: sk.time.from_mjd(1.0e8).add_utc_days(1.0e7),
        ):
            with pytest.raises(OverflowError, match="292,000 years"):
                call()

    def test_valid_input_unchanged(self):
        assert str(sk.time.from_mjd(60000.0)) == "2023-02-25T00:00:00.000000Z"
        assert str(sk.time.from_jd(2460000.5)) == "2023-02-25T00:00:00.000000Z"
        assert str(sk.time.from_unixtime(1700000000)) == "2023-11-14T22:13:20.000000Z"
        assert sk.time.from_gps_week_and_second(0, 0.0) == sk.time.GPS_EPOCH
        assert sk.time.from_gps_week_and_second(1, 604800.0) == sk.time.from_gps_week_and_second(2, 0.0)
        # Far, but inside the range
        assert sk.time.from_mjd(1.0e8).to_mjd() == 1.0e8
        assert sk.time.from_mjd(-1.0e8).to_mjd() == -1.0e8
        t = sk.time(2016, 12, 31, 12, 0, 0)
        assert str(t.add_utc_days(1)) == "2017-01-01T12:00:00.000000Z"
        assert str(t.add_utc_days(0.5)) == "2017-01-01T00:00:00.000000Z"
        assert str(t.add_utc_days(-1)) == "2016-12-30T12:00:00.000000Z"
        assert t.add_utc_days(0) == t


class TestInvalidTimescale:
    """``timescale.Invalid`` gave the year −290308 on the way in and NaN on
    the way out"""

    def test_raises_wherever_a_scale_is_taken(self):
        bad = sk.timescale.Invalid
        t = sk.time(2024, 1, 1)
        for call in (
            lambda: sk.time.from_mjd(59000.0, bad),
            lambda: sk.time.from_jd(2459000.5, bad),
            lambda: sk.time(2024, 1, 1, scale=bad),
            lambda: sk.time(2024, 1, 1, 0, 0, 0.0, scale=bad),
            lambda: t.to_mjd(bad),
            lambda: t.to_jd(bad),
            lambda: t.to_mjd(scale=bad),
        ):
            with pytest.raises(ValueError, match="timescale.Invalid is not a time scale"):
                call()

    def test_deprecated_aliases_too(self):
        t = sk.time(2024, 1, 1)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            t.as_mjd(sk.timescale.Invalid)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            t.as_jd(sk.timescale.Invalid)

    @pytest.mark.parametrize("name", ["UTC", "TT", "UT1", "TAI", "GPS", "TDB"])
    def test_real_scales_round_trip(self, name):
        scale = getattr(sk.timescale, name)
        t = sk.time.from_mjd(59000.25, scale)
        assert t.to_mjd(scale) == pytest.approx(59000.25, abs=1e-9)
        assert t.to_jd(scale) == pytest.approx(2459000.75, abs=1e-8)


# ───────────────────────── correctly rounded seconds ─────────────────────────


class TestCorrectlyRoundedSeconds:
    def test_reported_values(self):
        # Were 1678014225.4529998 and 45.452999999999996
        assert sk.time.from_unixtime(1678014225.453).to_unixtime() == 1678014225.453
        assert sk.time(2023, 3, 5, 11, 3, 45.453).to_gregorian() == (2023, 3, 5, 11, 3, 45.453)

    def test_every_millisecond_of_a_minute(self):
        t0 = sk.time(2024, 2, 29, 13, 7, 0.0)
        for ms in range(0, 60000, 7):
            t = t0 + sk.duration(microseconds=ms * 1000)
            assert t.to_gregorian()[5] == float(f"{ms // 1000}.{ms % 1000:03d}")

    def test_unixtime_is_nearest_float(self):
        rng = np.random.default_rng(7)
        for us in rng.integers(0, 4_000_000_000_000_000, 500):
            us = int(us)
            t = sk.time.from_unixtime(us / 1e6)
            # from_unixtime rounds to the microsecond, so the float nearest
            # the decimal is returned unchanged
            assert t.to_unixtime() == us / 1e6

    def test_leap_second(self):
        t = sk.time(2016, 12, 31, 23, 59, 60.25)
        assert t.to_gregorian() == (2016, 12, 31, 23, 59, 60.25)
        assert t.to_unixtime() == sk.time(2016, 12, 31, 23, 59, 59.25).to_unixtime()

    def test_gregorian_round_trip(self):
        t = sk.time(2023, 3, 5, 11, 3, 45.453)
        assert sk.time(*t.to_gregorian()) == t
        assert sk.time.from_gregorian(*t.to_gregorian()) == t


# ───────────────────────── exact duration * and / ─────────────────────────────


class TestDurationScaling:
    BIG = 2**53 + 1  # µs: 285 years, not representable in float seconds

    def test_integer_factor_is_exact(self):
        d = sk.duration(microseconds=self.BIG)
        assert (d * 1).microseconds == self.BIG  # was BIG - 1
        assert (d / 1).microseconds == self.BIG  # was BIG - 1
        assert (d * 3).microseconds == 3 * self.BIG
        assert (d * -1).microseconds == -self.BIG
        assert (d * np.int64(2)).microseconds == 2 * self.BIG
        assert (d * True).microseconds == self.BIG

    def test_float_factor_is_exact(self):
        d = sk.duration(microseconds=self.BIG)
        assert (d * 1.0).microseconds == self.BIG
        assert (d / 1.0).microseconds == self.BIG
        assert (d * 0.5).microseconds == (self.BIG + 1) // 2  # half rounds away from zero
        assert (d / 2.0).microseconds == (self.BIG + 1) // 2
        assert (d * -0.5).microseconds == -(self.BIG + 1) // 2
        assert (d * np.float64(2.0)).microseconds == 2 * self.BIG
        assert (d * np.float32(2.0)).microseconds == 2 * self.BIG

    def test_matches_exact_rational_arithmetic(self):
        rng = np.random.default_rng(11)

        def rnd(q):  # nearest integer, halves away from zero
            n = math.floor(abs(q) + Fraction(1, 2))
            return n if q >= 0 else -n

        for _ in range(400):
            us = int(rng.integers(-(2**62), 2**62))
            f = float(rng.normal()) * 10.0 ** int(rng.integers(-8, 3))
            if f == 0.0:
                continue
            d = sk.duration(microseconds=us)
            want = rnd(Fraction(us) * Fraction(f))
            if abs(want) < 2**63:
                assert (d * f).microseconds == want
            want = rnd(Fraction(us) / Fraction(f))
            if abs(want) < 2**63:
                assert (d / f).microseconds == want
            k = int(rng.integers(1, 1000))
            assert (d / k).microseconds == rnd(Fraction(us, k))
            assert (d / -k).microseconds == rnd(Fraction(us, -k))

    def test_ordinary_values(self):
        day = sk.duration(days=1)
        assert (day * 2.5).hours == 60.0
        assert (day / 2).hours == 12.0
        assert (day / 0.1).days == 10.0
        assert (day * 0).microseconds == 0
        assert (day * 0.0).microseconds == 0
        assert (day * 1e-300).microseconds == 0
        assert (day / 1e300).microseconds == 0
        assert (day / math.inf).microseconds == 0
        assert (sk.duration(microseconds=3) / 2).microseconds == 2
        assert (sk.duration(microseconds=-3) / 2).microseconds == -2
        assert (sk.duration(microseconds=1) * 5e-324).microseconds == 0
        assert day / sk.duration(hours=6) == 4.0

    def test_errors(self):
        day = sk.duration(days=1)
        for x in NONFINITE:
            with pytest.raises(ValueError):
                day * x
        with pytest.raises(ValueError):
            day / math.nan
        for call in (
            lambda: day * 1e300,
            lambda: day * 10**12,
            lambda: day / 1e-300,
            lambda: day / 5e-324,
            lambda: sk.duration(microseconds=2**62) * 2,
            lambda: sk.duration(microseconds=2**62) / 0.5,
        ):
            with pytest.raises(OverflowError, match="292,000 years"):
                call()
        for zero in (0, 0.0, -0.0, np.float64(0), False, sk.duration()):
            with pytest.raises(ZeroDivisionError):
                day / zero
        with pytest.raises(TypeError):
            day * "a"
        with pytest.raises(TypeError):
            day * None
        with pytest.raises(TypeError):
            day / "a"
        with pytest.raises(TypeError):
            day * day

    def test_largest_values(self):
        top = sk.duration(microseconds=2**63 - 1)
        bottom = sk.duration(microseconds=-(2**63))
        assert (top * 1).microseconds == 2**63 - 1
        assert (bottom * 1.0).microseconds == -(2**63)
        assert (bottom / 1).microseconds == -(2**63)
        with pytest.raises(OverflowError):
            bottom * -1
        with pytest.raises(OverflowError):
            bottom / -1.0
