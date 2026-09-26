"""Property-based tests (hypothesis) for the Python binding layer.

The Rust property suites (``tests/properties.rs``, ``tests/properties_data.rs``)
cover the time and frame arithmetic; this file checks what only the bindings
can get wrong: ``datetime`` conversion and time zones, string round-trips,
pickling of every picklable public type, and vectorised (array-of-times)
calls against scalar calls.

Generators are edge-biased the same way as the Rust ones: times near every
leap second and positive pre-1972 UTC step (including ``23:59:60.x``),
pre-1970 dates back to 1900 (with the 1961-1971 rubber-second era), and a few
fixed edges, mixed with uniform sampling.

Properties that exposed a defect (all since fixed) carry the minimal
counterexample as an ``@example``, so it replays on every run whatever the
random draw. A newly found defect is marked ``xfail(strict=True)`` with the
root cause until it is fixed; being strict, the marker then fails the run and
is removed with the fix. No property is currently marked.

Case counts: hypothesis's default 100 examples per test (fewer for the
expensive ones); set ``HYPOTHESIS_MAX_EXAMPLES`` for a deeper run.

Seeding: ``conftest.py`` loads the hypothesis profile named by
``HYPOTHESIS_PROFILE``. Pull-request CI selects ``ci`` (derandomized, so a
PR's result does not depend on the draw); the weekly scheduled job selects
``deep`` (random, with a large ``HYPOTHESIS_MAX_EXAMPLES``).
"""

import math
import os
import pickle
import struct
import time as systime
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import satkit as sk  # noqa: E402
from shared import ISS_2024, qtuple, same  # noqa: E402

_MAX = int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "100"))


def _settings(n=_MAX):
    # No deadline: the first call into some bindings loads data files.
    return settings(
        max_examples=n,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )


# ───────────────────────── generators ─────────────────────────

US = 1_000_000

# Every UTC day that ends with inserted time: (year, month, day, inserted us).
# Hard-coded from IERS Bulletin C (1972 on) and USNO tai-utc.dat (the positive
# pre-1972 steps), not read back from satkit. 1960-12-31 is satkit's
# convention for the start of its UTC model (TAI - UTC = 0 before 1961-01-01,
# 1.422818 s from it). Each step is labelled 23:59:60.x on the day it ends.
LEAP_DAYS = [
    (1960, 12, 31, 1_422_818), (1963, 10, 31, 100_000), (1964, 3, 31, 100_000),
    (1964, 8, 31, 100_000), (1964, 12, 31, 100_000), (1965, 2, 28, 100_000),
    (1965, 6, 30, 100_000), (1965, 8, 31, 100_000), (1971, 12, 31, 107_758),
    (1972, 6, 30, US), (1972, 12, 31, US), (1973, 12, 31, US), (1974, 12, 31, US),
    (1975, 12, 31, US), (1976, 12, 31, US), (1977, 12, 31, US), (1978, 12, 31, US),
    (1979, 12, 31, US), (1981, 6, 30, US), (1982, 6, 30, US), (1983, 6, 30, US),
    (1985, 6, 30, US), (1987, 12, 31, US), (1989, 12, 31, US), (1990, 12, 31, US),
    (1992, 6, 30, US), (1993, 6, 30, US), (1994, 6, 30, US), (1995, 12, 31, US),
    (1997, 6, 30, US), (1998, 12, 31, US), (2005, 12, 31, US), (2008, 12, 31, US),
    (2012, 6, 30, US), (2015, 6, 30, US), (2016, 12, 31, US),
]  # fmt: skip
INSERTED_US = {lb[:3]: lb[3] for lb in LEAP_DAYS}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _label_from_offset(idx, off_us):
    """UTC label (y, mo, d, h, mi, us-into-minute) at ``off_us`` from the start
    of the inserted interval of ``LEAP_DAYS[idx]``."""
    y, mo, d, ins = LEAP_DAYS[idx]
    if off_us < ins:
        return (y, mo, d, 23, 59, 60 * US + off_us)
    nxt = datetime(y, mo, d) + timedelta(days=1, microseconds=off_us - ins)
    return (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute, nxt.second * US + nxt.microsecond)


@st.composite
def leap_edge_labels(draw):
    idx = draw(st.integers(0, len(LEAP_DAYS) - 1))
    ins = LEAP_DAYS[idx][3]
    off = draw(
        st.one_of(
            st.integers(-3 * US, ins + 3 * US - 1),
            st.sampled_from([-1, 0, 1, ins - 1, ins, ins + 1]),
        )
    )
    return _label_from_offset(idx, off)


@st.composite
def uniform_labels(draw, y0=1900, y1=2045):
    dt = draw(st.datetimes(min_value=datetime(y0, 1, 1), max_value=datetime(y1, 12, 31, 23, 59, 59, 999999)))
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second * US + dt.microsecond)


FIXED_LABELS = [
    (1900, 1, 1, 0, 0, 0),
    (1960, 12, 31, 23, 59, 61_422_817),
    (1961, 1, 1, 0, 0, 0),
    (1963, 10, 31, 23, 59, 60_050_000),
    (1968, 1, 31, 23, 59, 59_899_999),
    (1968, 2, 1, 0, 0, 0),
    (1969, 12, 31, 23, 59, 59_999_999),
    (1970, 1, 1, 0, 0, 0),
    (1971, 12, 31, 23, 59, 60_000_000),
    (1971, 12, 31, 23, 59, 60_107_757),
    (1972, 1, 1, 0, 0, 0),
    (2000, 1, 1, 11, 58, 55_816_000),  # J2000 in UTC
    (2016, 12, 31, 23, 59, 60_000_000),
    (2017, 1, 1, 0, 0, 0),
]

labels = st.one_of(
    uniform_labels(1972, 2045),
    leap_edge_labels(),
    uniform_labels(1900, 1971),
    st.sampled_from(FIXED_LABELS),
)
non_leap_labels = labels.filter(lambda lb: lb[5] < 60 * US)


def in_rubber_era(y):
    """1961-1971: pre-1972 UTC, whose second is 1 + (1.3 ... 3.0)e-8 SI s"""
    return 1961 <= y < 1972


def label_time(lb):
    """satkit.time for a label, without float rounding: whole seconds
    through the calendar constructor plus an integer-microsecond duration.
    In the rubber-second era (outside an inserted interval) a UTC second is
    not an SI second, so the seconds go through the constructor as a float,
    which rounds to the microsecond."""
    y, mo, d, h, mi, us = lb
    if in_rubber_era(y) and us < 60 * US:
        return sk.time(y, mo, d, h, mi, us * 1e-6)
    return sk.time(y, mo, d, h, mi, float(us // US)) + sk.duration(microseconds=us % US)


def label_iso(lb):
    y, mo, d, h, mi, us = lb
    return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{us // US:02d}.{us % US:06d}Z"


def label_datetime(lb, tz=timezone.utc):
    """Aware datetime for a non-leap label, in ``tz``."""
    y, mo, d, h, mi, us = lb
    return datetime(y, mo, d, h, mi, us // US, us % US, tzinfo=timezone.utc).astimezone(tz)


def us_between(a, b):
    return abs((a - b).microseconds)


def _zones():
    out = [
        timezone.utc,
        timezone(timedelta(hours=5, minutes=30)),
        timezone(timedelta(hours=-8)),
        timezone(timedelta(hours=14)),
        timezone(timedelta(hours=-12)),
        timezone(timedelta(hours=5, minutes=45)),
    ]
    try:
        from zoneinfo import ZoneInfo

        out += [ZoneInfo(z) for z in ("America/New_York", "Australia/Lord_Howe", "Europe/London")]
    except Exception:  # pragma: no cover - no tz database
        pass
    return out


ZONES = _zones()


@contextmanager
def local_tz(name):
    old = os.environ.get("TZ")
    os.environ["TZ"] = name
    systime.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        systime.tzset()


# ───────────────────────── datetime round-trips ─────────────────────────


class TestDatetime:
    @_settings()
    @example(lb=(2039, 4, 6, 11, 5, 6_748_275), tz=timezone.utc)
    @given(non_leap_labels, st.sampled_from(ZONES))
    def test_aware_datetime_roundtrip(self, lb, tz):
        """Aware datetime -> satkit.time -> to_datetime() names the same
        instant, exactly, in any time zone, and matches the calendar route
        (the datetime's UTC label). Regression (the example): from_datetime
        went through the f64 timestamp() and lost a microsecond ~2% of the
        time."""
        dt = label_datetime(lb, tz)
        t = sk.time.from_datetime(dt)
        assert t == label_time(lb), (dt, t)
        back = t.to_datetime()
        assert back.tzinfo is not None
        assert back == dt, (dt, back)

    @pytest.mark.skipif(not hasattr(systime, "tzset"), reason="needs time.tzset")
    @pytest.mark.parametrize(
        "tzname", ["America/New_York", "Asia/Kolkata", "Australia/Lord_Howe", "UTC"]
    )
    @_settings(max(_MAX // 2, 10))
    @given(lb=non_leap_labels.filter(lambda lb: lb[0] >= 1902))
    def test_naive_datetime_is_local_time(self, tzname, lb):
        """A naive datetime is local time (Python's own convention): the
        naive local rendering of an aware datetime names the same instant,
        across DST transitions (``fold`` disambiguates), and
        ``to_datetime(False)`` is naive local time that round-trips."""
        with local_tz(tzname):
            aware = label_datetime(lb)
            naive = aware.astimezone().replace(tzinfo=None)  # keeps fold
            t = sk.time.from_datetime(naive)
            assert t == sk.time.from_datetime(aware), (tzname, aware, naive)
            local = t.to_datetime(False)
            assert local.tzinfo is None
            assert local == naive, (tzname, local, naive)
            assert sk.time.from_datetime(local) == t


# ───────────────────────── strings ─────────────────────────


class TestStrings:
    @_settings()
    @example(lb=(2024, 1, 1, 0, 0, 249))
    @given(labels)
    def test_str_roundtrip_and_fields(self, lb):
        """str(t) is exactly the label (leap seconds and pre-1970 included),
        to_rfc3339() matches it, satkit.time(str(t)) re-parses to exactly t,
        and the Gregorian fields are in range. Regression (the example):
        '...00.000249Z' parsed as 248 us (truncation)."""
        t = label_time(lb)
        iso = label_iso(lb)
        assert str(t) == iso
        assert t.to_rfc3339() == iso
        assert sk.time(str(t)) == t
        assert sk.time.from_rfc3339(str(t)) == t
        y, mo, d, h, mi, s = t.to_gregorian()
        assert (y, mo, d, h, mi) == lb[:5]
        assert 0 <= h < 24 and 0 <= mi < 60
        s_max = 60 + INSERTED_US.get((y, mo, d), 0) * 1e-6 if (h, mi) == (23, 59) else 60
        assert 0 <= s < s_max

    @_settings()
    @given(st.lists(labels, min_size=2, max_size=6))
    def test_str_order_matches_time_order(self, lbs):
        """Lexicographic order of the ISO labels is time order."""
        ts = [label_time(lb) for lb in lbs]
        by_time = sorted(range(len(ts)), key=lambda i: (ts[i] - ts[0]).microseconds)
        by_str = sorted(range(len(ts)), key=lambda i: str(ts[i]))
        assert [str(ts[i]) for i in by_time] == [str(ts[i]) for i in by_str]


# ───────────────────────── pickle ─────────────────────────

finite = st.floats(allow_nan=False, allow_infinity=False, width=64)
small = st.floats(-1e3, 1e3, allow_nan=False)
times = labels.map(label_time)
frames_man = st.sampled_from([sk.frame.GCRF, sk.frame.RTN, sk.frame.NTW])


def roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


class TestPickle:
    @_settings()
    @given(times, st.integers(-(2**62), 2**62))
    def test_time_and_duration(self, t, us):
        assert roundtrip(t) == t
        d = sk.duration(microseconds=us)
        assert roundtrip(d).microseconds == us

    @_settings()
    @given(finite, finite, finite, finite)
    def test_quaternion(self, w, x, y, z):
        q = sk.quaternion(w, x, y, z)
        assert qtuple(roundtrip(q)) == qtuple(q)

    @_settings()
    @given(st.floats(-1e8, 1e8), st.floats(-1e8, 1e8), st.floats(-1e8, 1e8))
    def test_itrfcoord(self, x, y, z):
        c = sk.itrfcoord([x, y, z])
        assert list(roundtrip(c).vector) == [x, y, z]

    @_settings()
    @given(
        st.floats(6.6e6, 5e7),
        st.floats(0, 0.95),
        st.floats(0, math.pi),
        st.floats(0, 2 * math.pi),
        st.floats(0, 2 * math.pi),
        st.floats(0, 2 * math.pi),
        st.floats(1e13, 1e15),
    )
    def test_kepler(self, a, e, i, raan, argp, nu, mu):
        k = sk.kepler(a, e, i, raan, argp, nu, mu=mu)
        k2 = roundtrip(k)
        assert k2 == k
        for f in ("a", "eccen", "inclination", "raan", "argp", "nu", "mu"):
            assert getattr(k2, f) == getattr(k, f), f

    @staticmethod
    def _tle(sat_num, incl, raan, ecc, argp, ma, mm, bstar, epoch, name, rev):
        tle = sk.TLE.from_lines(ISS_2024)[0]
        tle.satnum = sat_num
        tle.inclination = incl
        tle.raan = raan
        tle.eccen = ecc
        tle.arg_of_perigee = argp
        tle.mean_anomaly = ma
        tle.mean_motion = mm
        tle.bstar = bstar
        tle.epoch = epoch
        tle.name = name
        tle.rev_num = rev
        return tle

    tle_args = (
        st.integers(1, 99999),
        st.floats(0, 180),
        st.floats(0, 360),
        st.floats(0, 0.99),
        st.floats(0, 360),
        st.floats(0, 360),
        st.floats(0.5, 17),
        st.floats(-1e-2, 1e-2),
        times,
        st.text(max_size=24),
        st.integers(0, 99999),
    )

    @staticmethod
    def _tle_fields(tle):
        return [
            getattr(tle, f)
            for f in (
                "satnum", "name", "intl_desig", "desig_year", "desig_launch", "desig_piece",
                "inclination", "raan", "eccen", "arg_of_perigee", "mean_anomaly", "mean_motion",
                "mean_motion_dot", "mean_motion_dot_dot", "bstar", "rev_num", "element_num",
                "ephem_type",
            )
        ]  # fmt: skip

    @_settings()
    @given(st.tuples(*tle_args))
    def test_tle(self, args):
        """Every TLE field survives pickling, the epoch exactly."""
        tle = self._tle(*args)
        tle2 = roundtrip(tle)
        assert self._tle_fields(tle2) == self._tle_fields(tle)
        assert tle2.epoch == tle.epoch

    @_settings()
    @example(t=sk.time(2039, 2, 25, 21, 42, 35.0) + sk.duration(microseconds=990_070))
    @given(times)
    def test_embedded_times_exact(self, t):
        """Regression: these pickles stored times as a TAI MJD f64 and lost a
        microsecond ~1% of the time; they now store the raw microseconds."""
        tle = self._tle(25544, 51.6, 10.0, 0.001, 30.0, 40.0, 15.5, 1e-4, t, "X", 1)
        assert roundtrip(tle).epoch == t
        s = sk.satstate(t, np.array([7e6, 0, 0.0]), np.array([0, 7.5e3, 0.0]))
        assert roundtrip(s).time == t
        props = sk.satproperties(thrusts=[sk.thrust.constant([0, 1e-4, 0], t, t + sk.duration(hours=1), sk.frame.RTN)])
        assert roundtrip(props).thrusts[0].start == t

    def test_old_time_formats_load(self):
        """Pickles written by satkit <= 0.23 (times as a TAI MJD f64: TLE v1,
        satstate v1, satproperties v1/v2) still load, rounded to the nearest
        microsecond, which recovers the time exactly before ~2038. The old
        bytes are rebuilt from the new ones: same layout, version byte and
        time fields swapped."""
        t = sk.time(2024, 3, 1, 12, 0, 0.0) + sk.duration(microseconds=990_070)
        t2 = t + sk.duration(seconds=100.000249)

        def mjd(tm):
            return struct.pack("<d", tm.to_mjd(sk.timescale.TAI))

        def patch(b, version, fields):
            b = bytearray(b)
            b[0] = version
            for at, tm in fields:
                b[at : at + 8] = mjd(tm)
            return bytes(b)

        def load(obj, state):
            fresh = roundtrip(obj)
            fresh.__setstate__(state)
            return fresh

        tle = self._tle(25544, 51.6, 10.0, 0.001, 30.0, 40.0, 15.5, 1e-4, t, "X", 1)
        new = tle.__getstate__()
        assert new[0] == 2
        old = load(tle, patch(new, 1, [(85, t)]))
        assert old.epoch == t
        assert self._tle_fields(old) == self._tle_fields(tle)

        s = sk.satstate(t, np.array([7e6, 0, 0.0]), np.array([0, 7.5e3, 0.0]))
        s.add_maneuver(t2, [0.0, 1.0, 0.0], sk.frame.RTN)
        new = s.__getstate__()
        assert new[0] == 2
        # header 58 bytes, no covariance, 4-byte count, then the maneuver
        old = load(s, patch(new, 1, [(1, t), (58 + 4, t2)]))
        assert old.time == t and old.maneuvers[0]["time"] == t2
        np.testing.assert_array_equal(old.pos, s.pos)

        p = sk.satproperties(
            cdaoverm=0.01, craoverm=0.02, thrusts=[sk.thrust.constant([0, 1e-4, 0], t, t2, sk.frame.RTN)]
        )
        new = p.__getstate__()
        assert new[0] == 3
        # 21-byte header, then 24 accel + 1 frame, start, end
        for version in (1, 2):
            b = patch(new, version, [(21 + 25, t), (21 + 33, t2)])
            if version == 1:
                b = b[:-1]  # v1 has no ECOM flag
            old = load(p, b)
            assert (old.thrusts[0].start, old.thrusts[0].end) == (t, t2), version
            assert (old.cdaoverm, old.craoverm) == (0.01, 0.02)

    @_settings()
    @given(
        times,
        st.lists(small, min_size=6, max_size=6),
        st.booleans(),
        st.lists(st.tuples(times, st.lists(small, min_size=3, max_size=3), frames_man), max_size=3),
    )
    def test_satstate(self, t, pv, with_cov, mans):
        s = sk.satstate(t, np.array(pv[:3]) * 1e4, np.array(pv[3:]))
        if with_cov:
            a = np.arange(36.0).reshape(6, 6) * pv[0]
            s.cov = a @ a.T
        for mt, dv, fr in mans:
            s.add_maneuver(mt, dv, fr)
        s2 = roundtrip(s)
        assert s2.time == s.time
        np.testing.assert_array_equal(s2.pos, s.pos)
        np.testing.assert_array_equal(s2.vel, s.vel)
        if with_cov:
            np.testing.assert_array_equal(s2.cov, s.cov)
        else:
            assert s2.cov is None
        m1, m2 = s.maneuvers, s2.maneuvers
        assert len(m1) == len(m2)
        for a, b in zip(m1, m2):
            assert a["time"] == b["time"]
            np.testing.assert_array_equal(a["delta_v"], b["delta_v"])
            assert a["frame"] == b["frame"]

    ecom_fields = ("d0", "y0", "b0", "dc", "ds", "yc", "ys", "bc", "bs", "d2c", "d2s", "d4c", "d4s")

    @_settings()
    @given(
        st.floats(0, 1),
        st.floats(0, 1),
        st.lists(st.tuples(st.lists(small, min_size=3, max_size=3), times, st.floats(1, 1e5), frames_man), max_size=3),
        st.one_of(st.none(), st.tuples(st.lists(small, min_size=13, max_size=13), st.booleans())),
    )
    def test_satproperties_and_ecom(self, cd, cr, arcs, ecom):
        thrusts = [sk.thrust.constant(a, t0, t0 + sk.duration(seconds=dur), fr) for a, t0, dur, fr in arcs]
        e = None
        if ecom is not None:
            vals, sun_rel = ecom
            e = sk.ecomparams(**dict(zip(self.ecom_fields, vals)), sun_relative=sun_rel)
            e2 = roundtrip(e)
            assert [getattr(e2, f) for f in self.ecom_fields] == vals
            assert e2.sun_relative == sun_rel
        p = sk.satproperties(cdaoverm=cd, craoverm=cr, thrusts=thrusts, ecom=e)
        p2 = roundtrip(p)
        assert (p2.cdaoverm, p2.craoverm) == (cd, cr)
        assert len(p2.thrusts) == len(thrusts)
        for a, b in zip(p.thrusts, p2.thrusts):
            assert list(a.accel) == list(b.accel)
            assert a.frame == b.frame
            assert (a.start, a.end) == (b.start, b.end)
            b3 = roundtrip(a)  # thrust on its own (__reduce__)
            assert list(b3.accel) == list(a.accel) and b3.frame == a.frame
            assert (b3.start, b3.end) == (a.start, a.end)
        if e is None:
            assert p2.ecom is None
        else:
            assert [getattr(p2.ecom, f) for f in self.ecom_fields] == [getattr(e, f) for f in self.ecom_fields]
            assert p2.ecom == e

    @_settings()
    @example(cd=0.0, cr=1.0)
    @given(st.floats(0, 1), st.floats(0, 1))
    def test_satproperties_positional_rejected(self, cd, cr):
        """Found by the pickle property: positional arguments used to bind as
        (craoverm, cdaoverm), the reverse of the documented order. The
        constructor is now keyword-only, so a positional call raises rather
        than silently swapping drag and radiation pressure."""
        with pytest.raises(TypeError, match="keyword arguments only: cdaoverm=, craoverm="):
            sk.satproperties(cd, cr)
        with pytest.raises(TypeError, match="keyword arguments only"):
            sk.satproperties(cd)
        p = sk.satproperties(cdaoverm=cd, craoverm=cr)
        assert (p.cdaoverm, p.craoverm) == (cd, cr)

    @_settings()
    @given(
        st.floats(1e-14, 1e-4),
        st.floats(1e-14, 1e-4),
        st.integers(0, 70),
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.sampled_from([sk.integrator.rkv98, sk.integrator.rkts54, sk.integrator.gauss_jackson8]),
        st.sampled_from([sk.tidemodel.none, sk.tidemodel.solid_step1]),
        st.floats(1, 600),
        st.integers(1, 10**7),
        st.one_of(st.none(), st.floats(1e-3, 1e3)),
    )
    def test_propsettings(self, abs_e, rel_e, deg, sw, sun, moon, rel, interp, integ, tide, gj, steps, h0):
        ps = sk.propsettings(
            abs_error=abs_e,
            rel_error=rel_e,
            gravity_degree=deg,
            gravity_order=deg // 2,
            use_spaceweather=sw,
            use_sun_gravity=sun,
            use_moon_gravity=moon,
            use_relativistic_correction=rel,
            enable_interp=interp,
            integrator=integ,
            tide_model=tide,
            gj_step_seconds=gj,
            max_steps=steps,
            require_eop_coverage=not sw,
            initial_step_secs=h0,
        )
        ps2 = roundtrip(ps)
        for f in (
            "abs_error", "rel_error", "gravity_degree", "gravity_order", "gravity_model",
            "use_spaceweather", "use_sun_gravity", "use_moon_gravity",
            "use_relativistic_correction", "enable_interp", "integrator", "tide_model",
            "gj_step_seconds", "max_steps", "require_eop_coverage", "initial_step_secs",
        ):  # fmt: skip
            assert getattr(ps2, f) == getattr(ps, f), f

    @_settings(max(_MAX // 10, 5))
    @given(labels.filter(lambda lb: 1990 <= lb[0] <= 2020), st.floats(600, 3600))
    def test_propresult(self, lb, dur):
        """A propagation result (two-body + J2 only, so it is cheap and needs
        no space weather) pickles with its states and its interpolant."""
        t0 = label_time(lb)
        ps = sk.propsettings(
            gravity_degree=2,
            use_sun_gravity=False,
            use_moon_gravity=False,
            use_spaceweather=False,
            tide_model=sk.tidemodel.none,
            use_relativistic_correction=False,
        )
        r = sk.propagate(np.array([7e6, 0, 0, 0, 7.5e3, 1e3]), t0, duration_secs=dur, propsettings=ps)
        r2 = roundtrip(r)
        assert r2.time_begin == r.time_begin and r2.time_end == r.time_end
        np.testing.assert_array_equal(r2.state_end, r.state_end)
        tm = t0 + sk.duration(seconds=dur / 3)
        np.testing.assert_array_equal(r2.interp(tm), r.interp(tm))


# ───────────────────────── vectorised vs scalar ─────────────────────────

# One-element lists included: list input always gives list / array output
# (see test_binding_edges.py::TestOneElementTimeLists).
time_lists = st.lists(times, min_size=1, max_size=6)
recent_time_lists = st.lists(labels.filter(lambda lb: 1990 <= lb[0] <= 2030).map(label_time), min_size=1, max_size=6)


class TestVectorised:
    """Array-of-times calls give element-wise exactly the scalar results,
    for a list and for a numpy array of times."""

    @_settings()
    @given(
        axis=st.tuples(*[st.floats(-1, 1, allow_nan=False)] * 3).filter(
            lambda a: sum(x * x for x in a) > 1e-6
        ),
        angle=st.floats(-2 * math.pi, 2 * math.pi, allow_nan=False),
        n=st.integers(1, 20),
        seed=st.integers(0, 2**32 - 1),
        layout=st.sampled_from(["C", "F", "strided", "list"]),
    )
    def test_quaternion_times_array_is_rowwise(self, axis, angle, n, seed, layout):
        """q * V (Nx3) equals stacking q * v for each row, and V @ R.T, for
        any memory layout of V. The Nx3 path applied the inverse rotation in
        0.14.1-0.23.1 (a column-major matrix read as row-major)."""
        q = sk.quaternion.from_axis_angle(np.array(axis), angle)
        base = np.random.default_rng(seed).normal(size=(n, 6)) * 1e3
        V = base[:, :3].copy()
        rows = np.array([q * v for v in V])
        if layout == "F":
            arg = np.asfortranarray(V)
        elif layout == "strided":
            arg = base[:, ::2]  # non-contiguous view
            V = arg.copy()
            rows = np.array([q * v for v in V])
        elif layout == "list":
            arg = V.tolist()
        else:
            arg = V
        out = q * arg
        np.testing.assert_allclose(out, rows, rtol=0, atol=1e-9)
        np.testing.assert_allclose(out, V @ q.to_rotation_matrix().T, rtol=0, atol=1e-9)

    FT = [
        "gmst", "gast", "eqeq", "earth_rotation_angle", "qitrf2gcrf", "qgcrf2itrf",
        "qitrf2gcrf_approx", "qgcrf2itrf_approx", "qteme2gcrf", "qteme2itrf",
        "qitrf2tirs", "qtirs2cirs", "qcirs2gcrf", "qmod2gcrf", "qtod2mod_approx",
    ]  # fmt: skip

    @pytest.mark.parametrize("fname", FT)
    @_settings(max(_MAX // 4, 10))
    @given(tl=time_lists)
    def test_frametransform(self, fname, tl):
        f = getattr(sk.frametransform, fname)
        for arg in (tl, np.array(tl)):
            vec = f(arg)
            assert len(vec) == len(tl)
            for t, v in zip(tl, vec):
                assert same(v, f(t)), (fname, t)

    @pytest.mark.parametrize(
        "fn",
        [
            sk.sun.pos_gcrf,
            sk.sun.pos_mod,
            sk.moon.pos_gcrf,
            sk.moon.pos_mod,
            sk.moon.illumination,
            sk.moon.phase,
        ],
        ids=[
            "sun.pos_gcrf",
            "sun.pos_mod",
            "moon.pos_gcrf",
            "moon.pos_mod",
            "moon.illumination",
            "moon.phase",
        ],
    )
    @_settings(max(_MAX // 4, 10))
    @given(tl=time_lists)
    def test_sun_moon(self, fn, tl):
        vec = np.asarray(fn(tl))
        assert vec.shape[0] == len(tl)
        for t, v in zip(tl, vec):
            assert same(v, fn(t)), t

    @_settings(max(_MAX // 4, 10))
    @given(tl=recent_time_lists)
    def test_sgp4(self, tl):
        tle = sk.TLE.from_lines(ISS_2024)[0]
        pos, vel = sk.sgp4(tle, tl)
        assert pos.shape == vel.shape == (len(tl), 3)
        for i, t in enumerate(tl):
            p, v = sk.sgp4(tle, t)
            np.testing.assert_array_equal(pos[i], p)
            np.testing.assert_array_equal(vel[i], v)

    @_settings(max(_MAX // 4, 10))
    @given(tl=recent_time_lists)
    def test_jplephem(self, tl):
        try:
            sk.jplephem.geocentric_pos(sk.solarsystem.Moon, tl[0])
        except Exception as e:  # no ephemeris file available
            pytest.skip(f"JPL ephemeris unavailable: {e}")
        for body in (sk.solarsystem.Moon, sk.solarsystem.Sun, sk.solarsystem.Mars):
            vec = sk.jplephem.geocentric_pos(body, tl)
            assert vec.shape == (len(tl), 3)
            for t, v in zip(tl, vec):
                np.testing.assert_array_equal(v, sk.jplephem.geocentric_pos(body, t))
