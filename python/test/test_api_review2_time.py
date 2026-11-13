"""API review round 2: time and duration additions.

GPS week round trip, ``scale=`` on the date/Gregorian conversions,
``duration.from_microseconds`` / ``milliseconds``, floor division and modulo
of durations, ``time`` with numpy ``timedelta64`` / ``datetime64`` and
``datetime.datetime`` operands, and ``time.__format__``.
"""

import datetime

import numpy as np
import pytest

import satkit as sk

UTC = datetime.timezone.utc


# --- to_gps_week_and_second ---------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        sk.time(2024, 1, 1),
        sk.time(2016, 12, 31, 23, 59, 60.5),  # inside a leap second
        sk.time(1980, 1, 6),  # the GPS epoch
        sk.time(1979, 6, 1, 3, 4, 5.123456),  # before it: negative week
        sk.time(2100, 7, 4, 12, 0, 0.000001),
    ],
)
def test_gps_week_round_trip(t):
    week, sow = t.to_gps_week_and_second()
    assert isinstance(week, int) and isinstance(sow, float)
    assert 0.0 <= sow < 604800.0
    assert sk.time.from_gps_week_and_second(week, sow) == t


def test_gps_week_values():
    assert sk.time.GPS_EPOCH.to_gps_week_and_second() == (0, 0.0)
    # Sunday 2023-12-31 starts week 2295; GPS is 18 s ahead of UTC in 2024
    assert sk.time(2024, 1, 1).to_gps_week_and_second() == (2295, 86418.0)
    assert sk.time(1980, 1, 5, 23, 59, 59).to_gps_week_and_second() == (-1, 604799.0)


# --- scale= on from_date / from_gregorian / to_date / to_gregorian -------------

SCALES = [
    sk.timescale.UTC,
    sk.timescale.TAI,
    sk.timescale.TT,
    sk.timescale.GPS,
    sk.timescale.UT1,
    sk.timescale.TDB,
]


@pytest.mark.parametrize("scale", SCALES)
def test_from_gregorian_scale_matches_constructor(scale):
    comps = (2024, 3, 5, 12, 34, 56.789012)
    t = sk.time.from_gregorian(*comps, scale=scale)
    assert t == sk.time(*comps, scale=scale)
    assert sk.time.from_date(2024, 3, 5, scale=scale) == sk.time(2024, 3, 5, scale=scale)


@pytest.mark.parametrize("scale", SCALES)
def test_to_gregorian_scale_round_trip(scale):
    comps = (2024, 3, 5, 12, 34, 56.789012)
    g = sk.time.from_gregorian(*comps, scale=scale).to_gregorian(scale=scale)
    assert g[:5] == comps[:5]
    if scale in (sk.timescale.UT1, sk.timescale.TDB):
        assert g[5] == pytest.approx(comps[5], abs=1.0e-6)
    else:
        assert g[5] == comps[5]
    t = sk.time.from_date(2024, 3, 5, scale=scale)
    assert t.to_date(scale=scale) == (2024, 3, 5)
    assert t.to_date(scale) == (2024, 3, 5)  # positional, as to_mjd(scale)


def test_scale_values():
    t = sk.time(2024, 1, 1)
    assert t.to_gregorian(scale=sk.timescale.TAI) == (2024, 1, 1, 0, 0, 37.0)
    assert t.to_gregorian(scale=sk.timescale.TT) == (2024, 1, 1, 0, 1, 9.184)
    assert t.to_gregorian(scale=sk.timescale.GPS) == (2024, 1, 1, 0, 0, 18.0)
    assert sk.time.J2000.to_gregorian(scale=sk.timescale.TT) == (2000, 1, 1, 12, 0, 0.0)
    # Midnight TAI is 37 s before midnight UTC, on the previous UTC day
    t = sk.time.from_date(2024, 1, 1, scale=sk.timescale.TAI)
    assert t == sk.time(2023, 12, 31, 23, 59, 23)
    assert t.to_date() == (2023, 12, 31)
    assert t.to_date(scale=sk.timescale.TAI) == (2024, 1, 1)
    # A UTC leap second has no label in the uniform scales
    leap = sk.time(2016, 12, 31, 23, 59, 60.5)
    assert leap.to_gregorian() == (2016, 12, 31, 23, 59, 60.5)
    assert leap.to_gregorian(scale=sk.timescale.TAI) == (2017, 1, 1, 0, 0, 36.5)
    with pytest.raises(ValueError):
        sk.time.from_gregorian(2016, 12, 31, 23, 59, 60.5, scale=sk.timescale.TAI)


def test_scale_default_unchanged():
    t = sk.time(2024, 3, 5, 12, 34, 56.789012)
    assert t.to_gregorian() == t.to_gregorian(scale=sk.timescale.UTC) == (2024, 3, 5, 12, 34, 56.789012)
    assert t.to_date() == (2024, 3, 5)
    assert sk.time.from_date(2024, 3, 5) == sk.time(2024, 3, 5)
    assert sk.time.from_gregorian(2016, 12, 31, 23, 59, 60.5) == sk.time(2016, 12, 31, 23, 59, 60.5)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: sk.time.from_date(2024, 1, 1, scale=s),
        lambda s: sk.time.from_gregorian(2024, 1, 1, 0, 0, 0.0, scale=s),
        lambda s: sk.time(2024, 1, 1).to_date(scale=s),
        lambda s: sk.time(2024, 1, 1).to_gregorian(scale=s),
    ],
)
def test_scale_invalid(call):
    with pytest.raises(ValueError):
        call(sk.timescale.Invalid)
    with pytest.raises(TypeError):
        call("TAI")


# --- duration.from_microseconds / milliseconds --------------------------------


def test_duration_microseconds_milliseconds():
    d = sk.duration.from_microseconds(1500)
    assert d.microseconds == 1500
    assert d.milliseconds == 1.5
    assert d == sk.duration(microseconds=1500) == sk.duration.from_milliseconds(1.5)
    assert sk.duration.from_microseconds(-7).milliseconds == -0.007
    assert sk.duration.from_seconds(90).milliseconds == 90000.0
    big = 2**62 + 1
    assert sk.duration.from_microseconds(big).microseconds == big  # exact
    with pytest.raises(TypeError):
        sk.duration.from_microseconds(1.5)
    with pytest.raises(OverflowError):
        sk.duration.from_microseconds(2**63)


# --- duration // and % ---------------------------------------------------------


@pytest.mark.parametrize("a", [7, -7, 0, 6, 2**62 + 3, -(2**63)])
@pytest.mark.parametrize("b", [2, -2, 3, 7, 2**61])
def test_duration_floordiv_mod_match_python_ints(a, b):
    da, db = sk.duration(microseconds=a), sk.duration(microseconds=b)
    q = da // db
    assert isinstance(q, int)
    assert q == a // b
    assert (da % db).microseconds == a % b
    assert (da // b).microseconds == a // b


def test_duration_floordiv_most_negative_by_minus_one():
    # -2**63 // -1 is a fine Python int, though not a duration
    d = sk.duration(microseconds=-(2**63))
    assert d // sk.duration(microseconds=-1) == 2**63
    with pytest.raises(OverflowError):
        d // -1


def test_duration_floordiv_number():
    d = sk.duration.from_seconds(7)
    assert d // 2 == sk.duration.from_seconds(3.5)
    assert d // 2.5 == sk.duration.from_seconds(2.8)
    assert d // np.int64(2) == sk.duration.from_seconds(3.5)
    assert d // np.float64(2.0) == sk.duration.from_seconds(3.5)
    # Floored, not rounded: 7 us / 2 is 3 us, -7 us / 2 is -4 us
    assert (sk.duration(microseconds=7) // 2).microseconds == 3
    assert (sk.duration(microseconds=-7) // 2).microseconds == -4
    assert (sk.duration(microseconds=7) // 2.0).microseconds == 3
    assert (sk.duration(microseconds=-7) // 2.0).microseconds == -4
    assert (sk.duration(microseconds=1) // 3.0).microseconds == 0
    assert (sk.duration(microseconds=-1) // 1e300).microseconds == -1
    # Exact above 2**53 microseconds
    big = 2**60 + 1
    assert (sk.duration(microseconds=big) // 1.0).microseconds == big


def test_duration_floordiv_mod_errors():
    d = sk.duration.from_seconds(7)
    for zero in (0, 0.0, -0.0, sk.duration()):
        with pytest.raises(ZeroDivisionError):
            d // zero
    with pytest.raises(ZeroDivisionError):
        d % sk.duration()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            d // bad
    with pytest.raises(TypeError):
        d // "2"
    with pytest.raises(TypeError):
        d % 2
    with pytest.raises(TypeError):
        2 // d


# --- time with numpy timedelta64 ------------------------------------------------


def test_time_plus_timedelta64():
    t = sk.time(2024, 1, 1)
    assert t + np.timedelta64(90, "s") == sk.time(2024, 1, 1, 0, 1, 30)
    assert np.timedelta64(90, "s") + t == sk.time(2024, 1, 1, 0, 1, 30)
    assert t - np.timedelta64(1, "D") == sk.time(2023, 12, 31)
    assert t + np.timedelta64(5, "us") == t + sk.duration(microseconds=5)
    # Finer than a microsecond: rounded to the nearest, halves later
    assert t + np.timedelta64(1500, "ns") == t + sk.duration(microseconds=2)
    assert np.timedelta64(1500, "ns") + t == t + sk.duration(microseconds=2)
    assert t + np.timedelta64(-1500, "ns") == t - sk.duration(microseconds=1)
    # Exact for counts beyond 2**53 microseconds
    us = 2**60 + 1
    assert (t + np.timedelta64(us, "us")) - t == sk.duration(microseconds=us)


def test_time_plus_timedelta64_array():
    t = sk.time(2024, 1, 1)
    out = t + np.arange(3) * np.timedelta64(1, "h")
    assert out.dtype == object
    assert list(out) == [t, t + sk.duration(hours=1), t + sk.duration(hours=2)]
    out = t - np.array([1, 2], dtype="timedelta64[m]")
    assert list(out) == [t - sk.duration(minutes=1), t - sk.duration(minutes=2)]


def test_time_timedelta64_errors():
    t = sk.time(2024, 1, 1)
    with pytest.raises(ValueError, match="NaT"):
        t + np.timedelta64("NaT", "s")
    with pytest.raises(ValueError, match="index 1"):
        t + np.array([1, "NaT"], dtype="timedelta64[s]")
    with pytest.raises(ValueError, match="fixed"):
        t + np.timedelta64(1, "M")
    with pytest.raises(ValueError):
        t + np.zeros((2, 2), dtype="timedelta64[s]")
    with pytest.raises(OverflowError):
        t + np.timedelta64(2**62, "s")


# --- time against datetime.datetime and numpy.datetime64 -----------------------


def test_time_ordering_with_datetime():
    t = sk.time(2024, 1, 1, 12, 0, 0)
    before = datetime.datetime(2024, 1, 1, 11, tzinfo=UTC)
    after = datetime.datetime(2024, 1, 1, 13, tzinfo=UTC)
    assert t > before and t >= before and not t < before and not t <= before
    assert t < after and t <= after
    assert before < t and after > t  # reflected
    same = datetime.datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert t <= same and t >= same and not t < same


def test_time_ordering_with_datetime64():
    t = sk.time(2024, 1, 1, 0, 30, 0)
    d = np.datetime64("2024-01-01T00:00:00.000000")
    assert t > d and t >= d and not t < d
    # Reflected: numpy defers to satkit.time (a datetime64 is UTC; converting
    # it to a naive datetime would have read it as local time)
    assert d < t and d <= t and not d > t
    for unit in ("D", "s", "ns"):
        assert np.datetime64("2024-01-01", unit) < t
        assert t > np.datetime64("2024-01-01", unit)
    with pytest.raises(ValueError, match="NaT"):
        t < np.datetime64("NaT", "s")


def test_time_ordering_other_types():
    t = sk.time(2024, 1, 1)
    for bad in (1.0, "2024-01-01", sk.duration(), np.int64(1)):
        with pytest.raises(TypeError):
            t < bad
        with pytest.raises(TypeError):
            bad > t


def test_time_minus_datetime():
    t = sk.time(2024, 1, 1, 12, 0, 0)
    dt = datetime.datetime(2024, 1, 1, tzinfo=UTC)
    assert t - dt == sk.duration(hours=12)
    assert dt - t == sk.duration(hours=-12)
    # A naive datetime is local time, as in time(dt)
    naive = datetime.datetime(2024, 1, 1, 6, 0, 0, 123456)
    assert t - naive == t - sk.time(naive)
    assert naive - t == sk.time(naive) - t


def test_time_minus_datetime64():
    t = sk.time(2024, 1, 1, 12, 0, 0)
    d = np.datetime64("2024-01-01T00:00:00")
    assert t - d == sk.duration(hours=12)
    assert d - t == sk.duration(hours=-12)
    assert np.datetime64("2024-01-01T12:00:00.0000015", "ns") - t == sk.duration(microseconds=2)
    with pytest.raises(ValueError, match="NaT"):
        t - np.datetime64("NaT", "s")


def test_time_eq_unchanged_for_datetime():
    # == stays satkit.time only: hash(time) and hash(datetime) differ
    t = sk.time(2024, 1, 1)
    dt = datetime.datetime(2024, 1, 1, tzinfo=UTC)
    assert not (t == dt) and t != dt
    assert not (t == np.datetime64("2024-01-01")) and t != np.datetime64("2024-01-01")
    assert t == sk.time(dt)


def test_object_arrays_still_elementwise():
    # __array_priority__ only makes numpy *scalars* defer; arrays keep
    # their element-wise operators
    t = sk.time(2024, 1, 1)
    times = np.array([sk.time(2023, 1, 1), sk.time(2025, 1, 1)], dtype=object)
    assert list(times < t) == [True, False]
    assert list(t > times) == [True, False]
    durs = np.array([sk.duration(seconds=1), sk.duration(seconds=2)], dtype=object)
    assert list(durs + t) == [t + sk.duration(seconds=1), t + sk.duration(seconds=2)]


# --- __format__ ----------------------------------------------------------------


def test_time_format():
    t = sk.time(2023, 6, 3, 6, 19, 34.5)
    assert f"{t}" == str(t) == format(t, "")
    assert f"{t:%Y-%m-%d}" == "2023-06-03"
    assert f"{t:%H:%M:%S.%f}" == "06:19:34.500000"
    assert format(t, "%A %%") == t.strftime("%A %%")
    assert "{:%B}".format(t) == "June"
    with pytest.raises(ValueError):
        f"{t:%Q}"


# --- non-time inputs are TypeError ---------------------------------------------


def test_time_input_wrong_type_is_type_error():
    for bad in (1.0, "2024-01-01", [1.0, 2.0], np.zeros(2)):
        with pytest.raises(TypeError):
            sk.frametransform.gmst(bad)
