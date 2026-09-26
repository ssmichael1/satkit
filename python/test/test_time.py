import pytest
import numpy as np
import math as m
import pickle
from datetime import datetime, timezone

import satkit as sk


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


class TestPickle:

    def test_time_pickle(self):
        """
        Test pickling and unpickling of time objects
        """
        t1 = sk.time(2021, 9, 30, 12, 45, 13.345)
        p = pickle.dumps(t1)
        t2 = pickle.loads(p)
        assert t1 == t2

    def test_quaternion_pickle(self):
        """
        Test pickling and unpickling of quaternion objects
        """
        q1 = sk.quaternion.rotz(m.pi / 4)
        p = pickle.dumps(q1)
        q2 = pickle.loads(p)
        assert q1.x == pytest.approx(q2.x)
        assert q1.y == pytest.approx(q2.y)
        assert q1.z == pytest.approx(q2.z)
        assert q1

    def test_duration_pickle(self):
        """
        Test pickling and unpickling of duration objects
        """
        d1 = sk.duration.from_days(10)
        p = pickle.dumps(d1)
        d2 = pickle.loads(p)
        assert d1 == d2

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

    def test_from_gps_week_and_second(self):
        """
        Test GPS week and second-of-week conversion
        """
        # GPS epoch: January 6, 1980 00:00:00 UTC
        gps_epoch = sk.time.from_gps_week_and_second(0, 0)
        g = gps_epoch.to_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 6
        assert g[3] == 0

        # Week 1 should be 7 days later: January 13, 1980
        week1 = sk.time.from_gps_week_and_second(1, 0)
        g = week1.to_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 13

        # Difference between week 0 and week 1 should be exactly 7 days
        diff = week1 - gps_epoch
        assert diff.seconds == pytest.approx(604800.0, abs=1e-3)

        # Day 2 of week 0: January 7, 1980
        day2 = sk.time.from_gps_week_and_second(0, 86400)
        g = day2.to_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 7

        # Consistency: week * 604800 + sow seconds from GPS epoch
        t = sk.time.from_gps_week_and_second(2, 43200)
        expected_sec = 2 * 604800 + 43200
        assert (t - gps_epoch).seconds == pytest.approx(expected_sec, abs=1e-3)


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
        import pickle

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
