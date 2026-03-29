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

        assert tm1.as_datetime() == tm2
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
        (year, mon, day, hour, minute, sec) = t.as_gregorian()
        assert year == 2021
        assert mon == 1
        assert day == 1
        assert hour == 0
        assert minute == 0
        assert sec == 0

    def test_day_of_year(self):
        t = sk.time(2021, 1, 1)
        assert t.day_of_year() == 1

        t = sk.time(2021, 12, 31)
        assert t.day_of_year() == 365

        t = sk.time(2020, 12, 31)
        assert t.day_of_year() == 366

        t = sk.time(2100, 12, 31)
        assert t.day_of_year() == 365

        t = sk.time(2400, 12, 31)
        assert t.day_of_year() == 366

        t = sk.time(2024, 2, 29)
        assert t.day_of_year() == 60

        t = sk.time(2025, 8, 16)
        assert t.day_of_year() == 228

    def test_from_gps_week_and_second(self):
        """
        Test GPS week and second-of-week conversion
        """
        # GPS epoch: January 6, 1980 00:00:00 UTC
        gps_epoch = sk.time.from_gps_week_and_second(0, 0)
        g = gps_epoch.as_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 6
        assert g[3] == 0

        # Week 1 should be 7 days later: January 13, 1980
        week1 = sk.time.from_gps_week_and_second(1, 0)
        g = week1.as_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 13

        # Difference between week 0 and week 1 should be exactly 7 days
        diff = week1 - gps_epoch
        assert diff.seconds == pytest.approx(604800.0, abs=1e-3)

        # Day 2 of week 0: January 7, 1980
        day2 = sk.time.from_gps_week_and_second(0, 86400)
        g = day2.as_gregorian()
        assert g[0] == 1980
        assert g[1] == 1
        assert g[2] == 7

        # Consistency: week * 604800 + sow seconds from GPS epoch
        t = sk.time.from_gps_week_and_second(2, 43200)
        expected_sec = 2 * 604800 + 43200
        assert (t - gps_epoch).seconds == pytest.approx(expected_sec, abs=1e-3)
