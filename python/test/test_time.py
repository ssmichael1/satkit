import pytest
import numpy as np
import pickle
import warnings
from datetime import datetime, timezone

import satkit as sk
from shared import ISS_2024


class TestDateTime:
    """
    Check that function calls with satkit.time and datetime.datetime return
    the same result
    """

    def test_scalar_times(self):

        # Create times and show that they are equal
        tm1 = sk.time(2023, 3, 4, 12, 5, 6)
        tm2 = datetime(2023, 3, 4, 12, 5, 6, tzinfo=timezone.utc)

        assert tm1.to_datetime() == tm2
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
        assert t.to_rfc3339() == "2021-09-30T12:45:13.345000Z"

    def test_mjd(self):
        """
        Test MJD conversion
        """
        t = sk.time(2021, 1, 1, 0, 0, 0)
        mjd = t.to_mjd(sk.timescale.UTC)
        assert mjd == pytest.approx(59215.0)

    def test_jd(self):
        """
        Test JD conversion
        """
        t = sk.time(2021, 1, 1, 0, 0, 0)
        jd = t.to_jd(sk.timescale.UTC)
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

        d3 = sk.duration.from_hours(2.0)
        assert d3.seconds == 7200
        assert (d3 / 2.0).seconds == 3600
        d4 = sk.duration.from_hours(1.0)
        assert d3 / d4 == 2.0


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
        (year, mon, day, hour, minute, sec) = t.to_gregorian()
        assert year == 2021
        assert mon == 1
        assert day == 1
        assert hour == 0
        assert minute == 0
        assert sec == 0

    def test_weekday_is_property(self):
        # 2024-03-01 was a Friday
        t = sk.time(2024, 3, 1)
        assert t.weekday == sk.weekday.Friday
        assert sk.time(2024, 3, 3).weekday == sk.weekday.Sunday
        with pytest.raises(TypeError):
            t.weekday()  # type: ignore[operator]

    def test_day_of_year_is_property(self):
        t = sk.time(2024, 3, 1)
        assert isinstance(t.day_of_year, int)
        with pytest.raises(TypeError):
            t.day_of_year()  # type: ignore[operator]

    def test_day_of_year(self):
        t = sk.time(2021, 1, 1)
        assert t.day_of_year == 1

        t = sk.time(2021, 12, 31)
        assert t.day_of_year == 365

        t = sk.time(2020, 12, 31)
        assert t.day_of_year == 366

        t = sk.time(2100, 12, 31)
        assert t.day_of_year == 365

        t = sk.time(2400, 12, 31)
        assert t.day_of_year == 366

        t = sk.time(2024, 2, 29)
        assert t.day_of_year == 60

        t = sk.time(2025, 8, 16)
        assert t.day_of_year == 228


class TestParseErrors:
    """Every time-string parser raises ValueError for a string it cannot
    parse (strptime, from_string and time(str) raised RuntimeError in 0.23)"""

    GOOD = sk.time(2024, 1, 4, 13, 14, 12.5)

    def test_good_strings_parse(self):
        assert sk.time.from_rfc3339("2024-01-04T13:14:12.5Z") == self.GOOD
        assert sk.time.from_string("2024-01-04 13:14:12.5") == self.GOOD
        assert sk.time("2024-01-04T13:14:12.5Z") == self.GOOD
        assert sk.time("2024-01-04 13:14:12.5") == self.GOOD
        assert (
            sk.time.strptime("2024-01-04 13:14:12.5", "%Y-%m-%d %H:%M:%S.%f")
            == self.GOOD
        )

    @pytest.mark.parametrize(
        "parse",
        [
            sk.time.from_rfc3339,
            sk.time.from_string,
            sk.time,
            lambda s: sk.time.strptime(s, "%Y"),
            lambda s: sk.time.strptime(s, "%Y-%m-%d %H:%M:%S"),
        ],
        ids=["from_rfc3339", "from_string", "time", "strptime_Y", "strptime_full"],
    )
    @pytest.mark.parametrize(
        "bad",
        ["garbage", "", "2024-13-45 99:99:99", "2024-01-04 13:14:12 trailing 7"],
    )
    def test_bad_strings_raise_value_error(self, parse, bad):
        with pytest.raises(ValueError) as ei:
            parse(bad)
        # The parser's reason is in the message
        assert str(ei.value)

    def test_strptime_trailing_input(self):
        with pytest.raises(ValueError):
            sk.time.strptime("2024-01-04x", "%Y-%m-%d")

    def test_non_string_errors_keep_their_types(self):
        with pytest.raises(TypeError):
            sk.time.from_string(2024)
        with pytest.raises(TypeError):
            sk.time.strptime("2024", 4)
        with pytest.raises(TypeError):
            sk.time(2024.5)


class TestEpochConstants:
    def test_epoch_constants(self):
        # J2000: 2000-01-01 12:00:00 TT
        assert sk.time.J2000.to_mjd(sk.timescale.TT) == pytest.approx(51544.5, abs=1e-9)
        # GPS epoch: 1980-01-06 00:00:00 UTC
        g = sk.time.GPS_EPOCH.to_datetime()
        assert (g.year, g.month, g.day) == (1980, 1, 6)
        # Unix epoch: 1970-01-01 00:00:00 UTC
        assert sk.time.UNIX_EPOCH.to_unixtime() == pytest.approx(0.0, abs=1e-6)
        # MJD epoch: MJD 0
        assert sk.time.MJD_EPOCH.to_mjd(sk.timescale.UTC) == pytest.approx(0.0, abs=1e-9)


class TestArithmeticOverflow:
    """Time and duration arithmetic beyond the ±2^63-microsecond range
    (about ±292,000 years) raises OverflowError; it used to wrap around
    silently (duration(days=1e8, seconds=1e12) was negative, and adding 1e8
    days twice to 2024 gave year -34949)."""

    T = sk.time(2024, 1, 1)

    def test_duration_constructor_sum(self):
        with pytest.raises(OverflowError):
            sk.duration(days=1e8, seconds=1e12)
        # Each argument alone fits
        assert sk.duration(days=1e8).days == 1e8
        assert sk.duration(seconds=1e12).seconds == 1e12

    def test_duration_add_sub(self):
        big = sk.duration(days=1e8)
        with pytest.raises(OverflowError):
            big + big
        with pytest.raises(OverflowError):
            big - sk.duration(days=-1e8)
        with pytest.raises(OverflowError):
            sk.duration(microseconds=2**63 - 1) + sk.duration(microseconds=1)
        with pytest.raises(OverflowError):
            sk.duration(microseconds=-(2**63)) - sk.duration(microseconds=1)
        assert (big - big).microseconds == 0

    def test_time_plus_days(self):
        far = self.T + 1e8
        assert far > self.T
        with pytest.raises(OverflowError):
            far + 1e8
        with pytest.raises(OverflowError):
            (self.T - 1e8) - 1e8
        with pytest.raises(OverflowError):
            far + [0.0, 1e8]
        with pytest.raises(OverflowError):
            far + np.array([1e8])

    def test_time_plus_duration(self):
        big = sk.duration(days=1e8)
        with pytest.raises(OverflowError):
            self.T + big + big
        with pytest.raises(OverflowError):
            big + (self.T + big)
        with pytest.raises(OverflowError):
            self.T - big - big
        with pytest.raises(OverflowError):
            (self.T + big) + [big]

    def test_time_difference(self):
        far = self.T + 1e8
        near = self.T - 1e8
        with pytest.raises(OverflowError):
            far - near
        with pytest.raises(OverflowError):
            far - [near]
        assert (far - self.T).days == pytest.approx(1e8)


class TestTimeArrayOperands:
    """A real numeric 1-D numpy array of days is accepted whatever its
    dtype (integer and float32 arrays used to raise TypeError)"""

    T = sk.time(2024, 1, 1)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.uint8])
    def test_numeric_dtypes(self, dtype):
        days = np.array([1, 2], dtype=dtype)
        out = self.T + days
        assert list(out) == [self.T + 1.0, self.T + 2.0]
        out = self.T - days
        assert list(out) == [self.T - 1.0, self.T - 2.0]

    def test_non_numeric_arrays_raise(self):
        with pytest.raises(TypeError):
            self.T + np.array([True, False])
        with pytest.raises(TypeError):
            self.T + np.zeros((2, 2))
        with pytest.raises(TypeError):
            self.T + np.array([sk.duration(days=1)], dtype=object)


class TestFromStringYearSign:
    """from_string keeps the sign of an expanded-form year when it falls
    back from RFC 3339 (the sign used to be dropped: 44 BC became AD 44)"""

    @pytest.mark.parametrize(
        "s, year",
        [
            ("-0044-03-15 12:00:00", -44),
            ("-0044-03-15", -44),
            ("+0044-03-15 12:00:00", 44),
            ("+10000-03-15", 10000),
        ],
    )
    def test_signed_year(self, s, year):
        t = sk.time.from_string(s)
        assert t.to_gregorian()[0] == year
        assert t == sk.time.from_rfc3339(
            s.replace(" ", "T") + ("Z" if " " in s else "T00:00:00Z")
        )


class TestDurationUnits:
    def test_from_milliseconds_and_microseconds(self):
        d = sk.duration.from_milliseconds(1500.0)
        assert d.seconds == pytest.approx(1.5, rel=1e-12)
        assert d.microseconds == 1_500_000
        assert sk.duration(seconds=2).microseconds == 2_000_000


class TestLeapSeconds:
    def test_midnight_after_leap_second(self):
        # 00:00:00 on the day after a leap second is the end of the leap
        # second, not its start; the leap-second day is 86401 s long
        t = sk.time(2017, 1, 1)
        assert str(t) == "2017-01-01T00:00:00.000000Z"
        assert (t - sk.time(2016, 12, 31)).seconds == pytest.approx(86401.0, abs=1e-9)
        assert sk.time.from_mjd(57754.0) == t
        assert sk.time.from_jd(2457754.5) == t
        assert sk.time.from_unixtime(1483228800.0) == t
        assert sk.time(2016, 12, 31).add_utc_days(1.0) == t

    def test_leap_second_label(self):
        # The leap second can be entered by its own label and round-trips
        t = sk.time(2016, 12, 31, 23, 59, 60.5)
        s = t.to_rfc3339()
        assert s == "2016-12-31T23:59:60.500000Z"
        assert str(t) == s
        assert sk.time.from_rfc3339(s) == t
        assert sk.time(s) == t
        assert t - sk.time(2016, 12, 31, 23, 59, 59) == sk.duration(seconds=1.5)
        g = t.to_gregorian()
        assert g[:5] == (2016, 12, 31, 23, 59)
        assert g[5] == pytest.approx(60.5)
        # Exactly :60, from a string
        t60 = sk.time.from_rfc3339("2016-12-31T23:59:60Z")
        assert t60 == sk.time(2016, 12, 31, 23, 59, 60.0)
        assert (t - t60).seconds == pytest.approx(0.5)
        # :60 on a day without a leap second is an error
        with pytest.raises(Exception):
            sk.time(2024, 12, 31, 23, 59, 60.0)
        with pytest.raises(Exception):
            sk.time.from_rfc3339("2024-12-31T23:59:60Z")

    def test_1972_step(self):
        # Pre-1972 TAI - UTC drifts to 9.892242 s at 1972-01-01 00:00:00 UTC
        # and steps to 10 s there: a 0.107758 s inserted interval labelled
        # 1971-12-31T23:59:60.0 .. 23:59:60.107757
        t = sk.time(1972, 1, 1)
        assert str(t) == "1972-01-01T00:00:00.000000Z"
        assert (t - sk.time(1971, 12, 31, 23, 59, 59)).microseconds == 1_107_758
        assert str(sk.time(1971, 12, 31, 23, 59, 60.1)) == "1971-12-31T23:59:60.100000Z"
        with pytest.raises(Exception):
            sk.time(1971, 12, 31, 23, 59, 60.2)
        # 1970-01-01 00:00:00 UTC is 8.000082 s of TAI after 1970-01-01 TAI
        assert sk.time.UNIX_EPOCH == sk.time(1970, 1, 1)
        tai_1970 = sk.time.from_mjd(40587.0, sk.timescale.TAI)
        assert (sk.time.UNIX_EPOCH - tai_1970).microseconds == 8_000_082

    def test_pre1972_pickle_keeps_the_instant(self):
        t = sk.time(1965, 6, 1, 12, 30, 15.25)
        t2 = pickle.loads(pickle.dumps(t))
        assert t2 == t and str(t2) == "1965-06-01T12:30:15.250000Z"


class TestPre1970:
    def test_time_of_day_before_1970(self):
        t = sk.time(1960, 1, 1, 12, 0, 0)
        assert str(t) == "1960-01-01T12:00:00.000000Z"
        assert t.to_rfc3339() == "1960-01-01T12:00:00.000000Z"
        g = t.to_gregorian()
        assert g[:5] == (1960, 1, 1, 12, 0)
        assert g[5] == pytest.approx(0.0)
        t = sk.time(1969, 12, 31, 23, 59, 59.5)
        assert str(t) == "1969-12-31T23:59:59.500000Z"


@pytest.mark.skipif(not hasattr(__import__("time"), "tzset"), reason="needs time.tzset")
class TestDatetimeTimeZone:
    """A naive datetime is local time (Python's convention); an aware one
    uses its own offset."""

    def test_naive_is_local_aware_uses_offset(self):
        import os
        import time as systime
        from datetime import timedelta

        old_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            systime.tzset()
            # Naive: 12:30 US Eastern (EDT, UTC-4) = 16:30 UTC
            naive = datetime(2024, 6, 15, 12, 30)
            assert sk.time.from_datetime(naive) == sk.time(2024, 6, 15, 16, 30, 0)
            # Aware UTC: 12:30 UTC on any machine
            aware = datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)
            assert sk.time.from_datetime(aware) == sk.time(2024, 6, 15, 12, 30, 0)
            # Aware with another offset
            plus2 = datetime(2024, 6, 15, 12, 30, tzinfo=timezone(timedelta(hours=2)))
            assert sk.time.from_datetime(plus2) == sk.time(2024, 6, 15, 10, 30, 0)
            # Functions that take a datetime use the same convention
            gmst = sk.frametransform.gmst(naive)
            assert gmst == pytest.approx(sk.frametransform.gmst(sk.time(2024, 6, 15, 16, 30, 0)))
            # to_datetime(utc=False) is naive local time and round-trips
            t = sk.time(2024, 6, 15, 16, 30, 0)
            local = t.to_datetime(False)
            assert local.tzinfo is None
            assert (local.hour, local.minute) == (12, 30)
            assert sk.time.from_datetime(local) == t
            assert t.to_datetime().tzinfo is not None
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            systime.tzset()


def _as_time(t):
    """The satkit.time a scalar time argument names (satstate stores it)"""
    return sk.satstate(t, np.zeros(3), np.zeros(3)).time


def _qs(qs):
    return [(q.w, q.x, q.y, q.z) for q in qs]


class TestDatetime64:
    """numpy datetime64 times, arrays and scalars, are UTC labels: the same
    instants as the equivalent satkit.time / aware-UTC datetime"""

    LABELS = [
        "2024-01-01T12:00:00.000000",
        "2024-01-01T12:07:18.123456",
        "2024-02-29T23:59:59.999999",
        "2016-12-31T23:59:59.500000",  # just before a leap second
        "2017-01-01T00:00:00.000000",  # just after it
        "2031-07-04T04:05:06.000001",
    ]

    def times(self):
        return [sk.time.from_rfc3339(s + "Z") for s in self.LABELS]

    def array(self, unit="us"):
        return np.array(self.LABELS, dtype=f"datetime64[{unit}]")

    def test_same_results_as_satkit_time(self):
        tl, ta = self.times(), self.array()
        tle = sk.TLE.from_lines(ISS_2024)[0]
        for a, b in zip(sk.sgp4(tle, ta), sk.sgp4(tle, tl)):
            assert np.array_equal(a, b)
        ft = sk.frametransform
        assert _qs(ft.qitrf2gcrf(ta)) == _qs(ft.qitrf2gcrf(tl))
        assert _qs(ft.rotation(sk.frame.ITRF, sk.frame.GCRF, ta)) == _qs(
            ft.rotation(sk.frame.ITRF, sk.frame.GCRF, tl)
        )
        assert np.array_equal(sk.sun.pos_gcrf(ta), sk.sun.pos_gcrf(tl))
        assert np.array_equal(sk.moon.pos_gcrf(ta), sk.moon.pos_gcrf(tl))
        assert ft.gmst(ta) == ft.gmst(tl)
        # A datetime64 array keeps its time axis, like a list
        assert sk.sgp4(tle, ta[:1])[0].shape == (1, 3)
        try:
            jpl = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, tl)
        except Exception:
            pytest.skip("JPL ephemeris not available")
        assert np.array_equal(sk.jplephem.geocentric_pos(sk.solarsystem.Moon, ta), jpl)

    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"])
    def test_units(self, unit):
        """Every unit gives the instant of its UTC label, including before
        1972 (calendar units through the calendar, not a fixed length)"""
        src = [
            "1965-06-01T12:30:15.250000",
            "2024-02-29T13:45:12.123456",
            "1969-12-31T23:59:59.999999",
        ]
        arr = np.array(src, dtype=f"datetime64[{unit}]")
        # numpy's own (exact) conversion to microseconds names the label
        expect = [sk.time.from_rfc3339(str(x) + "Z") for x in arr.astype("datetime64[us]")]
        assert sk.frametransform.gmst(arr) == sk.frametransform.gmst(expect)
        assert [_as_time(x) for x in arr] == expect

    def test_unit_multiplier(self):
        """A unit with a multiplier (datetime64[15m]) counts that many units"""
        assert _as_time(np.array([7], dtype="datetime64[15m]")[0]) == sk.time(1970, 1, 1, 1, 45, 0)
        assert _as_time(np.array([3], dtype="datetime64[5M]")[0]) == sk.time(1971, 4, 1)
        assert _as_time(np.array([151], dtype="datetime64[10ns]")[0]) == sk.time(
            1970, 1, 1
        ) + sk.duration(microseconds=2)

    @pytest.mark.parametrize("day", ["2024-01-01", "1970-01-01", "1969-06-01"])
    @pytest.mark.parametrize(
        "ns, us",
        [(0, 0), (499, 0), (500, 1), (1_499, 1), (1_500, 2), (-499, 0), (-500, 0), (-501, -1), (-1_500, -1)],
    )
    def test_sub_microsecond_rounds_to_nearest(self, day, ns, us):
        """Finer than a microsecond rounds to the nearest microsecond, halves
        to the later one on either side of 1970, as the calendar constructor
        and the string parser round the seconds: the value names the same
        time as its ISO string"""
        t64 = np.datetime64(day + "T00:00:00", "ns") + np.timedelta64(ns, "ns")
        expect = sk.time(day + "T00:00:00Z") + sk.duration(microseconds=us)
        assert _as_time(t64) == expect
        assert sk.time(str(t64) + "Z") == expect
        assert sk.frametransform.gmst(np.array([t64])) == sk.frametransform.gmst([expect])

    @pytest.mark.parametrize("unit, count", [("ps", 1_500_000), ("fs", 1_500_000_000), ("as", 1_500_000_000_000)])
    def test_finer_units_round(self, unit, count):
        # 1.5 us after the Unix epoch rounds to 2 us; 1.5 us before, to 1 us before
        t64 = np.array([count, -count], dtype=f"datetime64[{unit}]")
        assert _as_time(t64[0]) == sk.time.UNIX_EPOCH + sk.duration(microseconds=2)
        assert _as_time(t64[1]) == sk.time.UNIX_EPOCH - sk.duration(microseconds=1)

    def test_strided_and_byte_swapped(self):
        tl, ta = self.times(), self.array()
        gmst = sk.frametransform.gmst
        assert gmst(ta[::2]) == gmst(tl[::2])
        assert gmst(ta[::-1]) == gmst(tl[::-1])
        swapped = ta.astype(ta.dtype.newbyteorder("S"))
        assert not swapped.dtype.isnative
        assert gmst(swapped) == gmst(tl)
        assert gmst(swapped[::2]) == gmst(tl[::2])
        assert gmst(np.array([], dtype="datetime64[us]")) == []

    def test_nat_raises_value_error(self):
        arr = self.array()
        arr[2] = np.datetime64("NaT", "us")
        with pytest.raises(ValueError, match="index 2 is NaT"):
            sk.frametransform.gmst(arr)
        with pytest.raises(ValueError, match="NaT"):
            sk.frametransform.gmst(np.datetime64("NaT", "s"))
        # numpy deprecates the generic unit, which can only hold NaT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            generic = np.array(["NaT", "NaT"], dtype="datetime64")
        with pytest.raises(ValueError, match="index 0 is NaT"):
            sk.frametransform.gmst(generic)

    def test_out_of_range_raises_overflow_error(self):
        for arr in (
            np.array([0, 2**62], dtype="datetime64[s]"),
            np.array([0, -(2**62)], dtype="datetime64[s]"),
            np.array([0, 10**12], dtype="datetime64[Y]"),
            np.array([0, 2**62], dtype="datetime64[W]"),
        ):
            with pytest.raises(OverflowError, match="index 1"):
                sk.frametransform.gmst(arr)
        with pytest.raises(OverflowError):
            sk.frametransform.gmst(np.datetime64(2**62, "s"))

    def test_multidimensional_array_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            sk.frametransform.gmst(self.array().reshape(2, 3))

    def test_scalar_accepted_where_times_are(self):
        t = sk.time(2024, 6, 15, 16, 30, 0.25)
        d64 = np.datetime64("2024-06-15T16:30:00.250")
        assert _as_time(d64) == t
        assert sk.frametransform.gmst(d64) == sk.frametransform.gmst(t)
        # a scalar (or 0-d array) gives scalar output, as for satkit.time
        assert sk.sun.pos_gcrf(d64).shape == (3,)
        assert np.array_equal(sk.sun.pos_gcrf(np.asarray(d64)), sk.sun.pos_gcrf(t))
        tle = sk.TLE.from_lines(ISS_2024)[0]
        assert np.array_equal(sk.sgp4(tle, d64)[0], sk.sgp4(tle, t)[0])
        coord = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=0)
        noon = np.datetime64("2024-06-15T16:00")
        assert sk.sun.rise_set(noon, coord) == sk.sun.rise_set(sk.time(2024, 6, 15, 16, 0, 0), coord)
        assert sk.density.nrlmsise(coord, d64) == sk.density.nrlmsise(coord, t)
        assert sk.density.nrlmsise(400e3, d64) == sk.density.nrlmsise(400e3, t)
        # lists and object arrays of datetime64 scalars
        assert sk.frametransform.gmst([d64, t]) == sk.frametransform.gmst([t, t])
        obj = np.array([d64, d64], dtype=object)
        assert sk.frametransform.gmst(obj) == sk.frametransform.gmst([t, t])
        # an OMM dict's EPOCH
        omm = tle.to_omm()
        omm64 = {**omm, "EPOCH": np.datetime64(omm["EPOCH"].rstrip("Z"), "us")}
        assert np.array_equal(sk.sgp4(omm64, t)[0], sk.sgp4(omm, t)[0])

    def test_pre1972_matches_datetime(self):
        """Pre-1972 labels go through the same UTC model as datetime"""
        d64 = np.datetime64("1965-06-01T12:30:15.250000")
        dt = datetime(1965, 6, 1, 12, 30, 15, 250000, tzinfo=timezone.utc)
        assert _as_time(d64) == sk.time.from_datetime(dt) == sk.time(1965, 6, 1, 12, 30, 15.25)
        assert str(_as_time(np.datetime64("1960-01-01T12:00"))) == "1960-01-01T12:00:00.000000Z"
