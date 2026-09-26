"""
Differential tests of satkit's time scales and Earth-frame rotations against
ERFA (``pyerfa``, the C library behind astropy's SOFA-equivalent routines).

The aim is to compare like with like: each satkit quantity is checked against
the ERFA routine for the same model (IAU 1982 GMST against ``gmst82``, the
IAU 2006/2000A CIP series against ``xy06``/``s06``, and so on), and ERFA is fed
the Earth orientation parameters satkit itself uses, so data differences
cannot mask algorithm differences. Where satkit deliberately uses a simpler
model (the ``_approx`` reduction, the one-term TDB - TT series, the
two-term equation of the equinoxes) the test checks the documented accuracy,
and the tolerance comment says where the number comes from.

Cases marked ``xfail(strict=True)`` are confirmed satkit defects; being
strict, they fail once the defect is fixed, so remove the marker with the fix.

Instants are read and built through the internal microsecond count, which is
TAI (``raw / 86400e6 + 40587`` is the TAI MJD), via the public duration API
relative to 1970-01-01 00:00:00 TAI (``TAI_1970`` below):
``(t - TAI_1970).microseconds`` and ``TAI_1970 + duration(microseconds=raw)``.
Both are exact. (``time.UNIX_EPOCH`` is the UTC label 1970-01-01T00:00:00,
8.000082 s of TAI later.)
"""

import calendar
import math
import os
import warnings

import numpy as np
import pytest

erfa = pytest.importorskip("erfa")

import satkit as sk  # noqa: E402

ft = sk.frametransform
TS = sk.timescale

pytestmark = pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")

AS2RAD = math.pi / 180.0 / 3600.0
MAS2RAD = AS2RAD * 1.0e-3
UAS2RAD = AS2RAD * 1.0e-6
MJD_UNIX = 40587.0  # MJD of 1970-01-01
R_LEO = 7.0e6  # m, for converting angles to metres in failure messages
R_GEO = 42.164e6

# Seconds: an f64 MJD near 60000 has an ulp of 0.63 us, and satkit converts
# MJD to its microsecond count by truncation, so any comparison that goes
# through `to_mjd` / `from_mjd` carries up to ~1.3 us of rounding.
TOL_MJD_S = 1.5e-6



# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


# 1970-01-01 00:00:00 TAI, where satkit's microsecond count is zero (TAI MJD
# 40587 is exact in f64, so this is exact)
TAI_1970 = sk.time.from_mjd(MJD_UNIX, TS.TAI)


def raw_us(t):
    """satkit's internal microsecond count (TAI since 1970-01-01 00:00:00 TAI)"""
    return (t - TAI_1970).microseconds


def from_raw(raw):
    return TAI_1970 + sk.duration(microseconds=int(raw))


def tai_pair(raw):
    """ERFA two-part TAI Julian date from satkit's microsecond count(s)"""
    raw = np.asarray(raw, dtype=np.int64)
    days = np.floor_divide(raw, 86_400_000_000)
    rem = raw - days * 86_400_000_000
    return 2400000.5 + MJD_UNIX + days.astype(float), rem / 86_400.0e6


def pair_to_tai_seconds(d1, d2):
    """Seconds since 1970-01-01 00:00:00 TAI from an ERFA two-part TAI JD"""
    return ((np.asarray(d1) - 2400000.5 - MJD_UNIX) + np.asarray(d2)) * 86400.0


def pair_to_us(d1, d2):
    """Integer microseconds since 1970-01-01 00:00:00 TAI from a two-part TAI JD
    (the whole-day part is exact in f64; the sum keeps 0.25 us resolution)"""
    return int(round((d1 - 2400000.5 - MJD_UNIX) * 86400e6 + d2 * 86400e6))


def erfa_us_from_utc(y, mo, d, h, mi, s):
    """ERFA's TAI for a UTC label, as satkit's integer microsecond count"""
    return pair_to_us(*erfa.utctai(*erfa.dtf2d("UTC", y, mo, d, h, mi, s)))


def assert_us(t, expected_us, what, tol_us=1):
    """Instant equals an expected microsecond count. 1 us of slack: satkit
    truncates seconds to microseconds (see test_gregorian_seconds_round_to_microsecond)"""
    got = raw_us(t)
    assert abs(got - expected_us) <= tol_us, f"{what}: satkit - ERFA = {(got - expected_us) * 1e-6:+.6f} s"


def mjd_minus_pair_seconds(mjd, d1, d2):
    """(satkit MJD) - (ERFA two-part JD), in seconds"""
    return ((np.asarray(mjd) - (np.asarray(d1) - 2400000.5)) - np.asarray(d2)) * 86400.0


def matrices(quats):
    return np.array([q.to_rotation_matrix() for q in quats])


def rot_angle(a, b):
    """Angle (rad) of the small rotation a @ b.T, stable to ~1e-16 rad"""
    d = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    v = 0.5 * np.stack(
        [d[..., 2, 1] - d[..., 1, 2], d[..., 0, 2] - d[..., 2, 0], d[..., 1, 0] - d[..., 0, 1]],
        axis=-1,
    )
    return np.linalg.norm(v, axis=-1)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def leap_seconds():
    """(year, month) of every 1st-of-month at which ERFA's TAI - UTC steps by 1 s"""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", erfa.ErfaWarning)  # "dubious year" past ERFA's release
        prev = erfa.dat(1972, 1, 1, 0.0)
        for y in range(1972, 2036):
            for m in (1, 7):
                d = erfa.dat(y, m, 1, 0.0)
                if d != prev:
                    assert d - prev == 1.0
                    out.append((y, m))
                prev = d
    return out


LEAPS = leap_seconds()


def day_before(y, m):
    return (y - 1, 12, 31) if m == 1 else (y, 6, 30)


def eop_file_rows():
    """The loaded EOP file's rows as (mjd, xp, yp, dut1, dX, dY), or None"""
    src = ft.eop_source()
    for d in sk.utils.data_search_dirs():
        if src == "finals2000A":
            p = os.path.join(d, "finals2000A.all")
            if not os.path.isfile(p):
                continue
            rows = []
            with open(p) as f:
                for line in f:
                    line = line.rstrip()
                    if len(line) < 17 or line[16] not in "IP":
                        continue

                    def col(a, n, line=line):
                        return line[a - 1 : a - 1 + n].strip()

                    if not col(59, 10):
                        continue
                    rows.append(
                        (
                            float(col(8, 8)),
                            float(col(19, 9)),
                            float(col(38, 9)),
                            float(col(59, 10)),
                            float(col(98, 9) or 0.0),
                            float(col(117, 9) or 0.0),
                        )
                    )
            return np.array(rows)
        if src == "celestrak":
            p = os.path.join(d, "EOP-All.csv")
            if not os.path.isfile(p):
                continue
            rows = []
            with open(p) as f:
                next(f)
                for line in f:
                    v = line.strip().split(",")
                    if len(v) < 12:
                        continue
                    rows.append(
                        (float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[8]) * 1e3, float(v[9]) * 1e3)
                    )
            return np.array(rows)
    return None


# ----------------------------------------------------------------------------
# Time scales: UTC <-> TAI and leap seconds
# ----------------------------------------------------------------------------


class TestLeapSeconds:
    def test_table_matches_erfa_dat(self):
        """TAI - UTC at noon on every 1st and 15th of the month, 1972-2035"""
        for y in range(1972, 2036):
            for m in range(1, 13):
                for d in (1, 15):
                    got = raw_us(sk.time(y, m, d, 12, 0, 0.0)) - pair_to_us(*erfa.dtf2d("TAI", y, m, d, 12, 0, 0.0))
                    assert got == round(erfa.dat(y, m, d, 0.5) * 1e6), (y, m, d)

    @pytest.mark.parametrize("leap", LEAPS, ids=[f"{y}-{m:02d}" for y, m in LEAPS])
    def test_utc_labels_around_leap_second(self, leap):
        """Gregorian UTC -> instant within +/-2 s of each leap second, and back"""
        y, m = leap
        py, pm, pd = day_before(y, m)
        labels = [(py, pm, pd, 23, 59, s) for s in (58.0, 58.5, 59.0, 59.5, 59.999999)]
        labels += [(y, m, 1, 0, 0, s) for s in (0.000001, 0.5, 1.0, 1.5, 2.0)]
        for c in labels:
            t = sk.time(*c)
            assert_us(t, erfa_us_from_utc(*c), c)
            g = t.to_gregorian()
            assert g[:5] == c[:5] and g[5] == pytest.approx(c[5], abs=1e-7), (c, g)

    @pytest.mark.parametrize("leap", LEAPS, ids=[f"{y}-{m:02d}" for y, m in LEAPS])
    def test_instant_to_utc_labels_around_leap_second(self, leap):
        """TAI instants within +/-2.5 s of each leap second (including every
        microsecond boundary of the leap second itself) -> UTC Gregorian,
        against ERFA taiutc + d2dtf. The 23:59:60.x labels are checked here."""
        y, m = leap
        r0 = erfa_us_from_utc(y, m, 1, 0, 0, 0.0)
        offsets = list(range(-2_500_000, 2_500_001, 125_000)) + [-1_000_001, -1_000_000, -999_999, -1, 1]
        raws = np.array([r0 + o for o in offsets], dtype=np.int64)
        u1, u2 = erfa.taiutc(*tai_pair(raws))
        iy, im, iday, ihmsf = erfa.d2dtf("UTC", 6, u1, u2)
        for k, r in enumerate(raws):
            g = from_raw(r).to_gregorian()
            exp = (int(iy[k]), int(im[k]), int(iday[k]), int(ihmsf["h"][k]), int(ihmsf["m"][k]))
            sec = ihmsf["s"][k] + ihmsf["f"][k] * 1e-6
            assert g[:5] == exp and g[5] == pytest.approx(sec, abs=1e-7), (int(r - r0), g, exp, sec)

    def test_midnight_after_leap_second_gregorian(self):
        for y, m in LEAPS:
            t = sk.time(y, m, 1, 0, 0, 0.0)
            assert_us(t, erfa_us_from_utc(y, m, 1, 0, 0, 0.0), (y, m))

    def test_midnight_after_leap_second_from_mjd(self):
        for y, m in LEAPS:
            t = sk.time.from_mjd(erfa.cal2jd(y, m, 1)[1])
            assert_us(t, erfa_us_from_utc(y, m, 1, 0, 0, 0.0), (y, m))

    def test_midnight_after_leap_second_from_unixtime(self):
        for y, m in LEAPS:
            t = sk.time.from_unixtime(float(calendar.timegm((y, m, 1, 0, 0, 0))))
            assert_us(t, erfa_us_from_utc(y, m, 1, 0, 0, 0.0), (y, m))

    def test_midnight_after_leap_second_add_utc_days(self):
        for y, m in LEAPS:
            t = sk.time(*day_before(y, m), 12, 0, 0.0).add_utc_days(0.5)
            assert_us(t, erfa_us_from_utc(y, m, 1, 0, 0, 0.0), (y, m))

    def test_enter_leap_second_label(self):
        for y, m in LEAPS:
            for s in (60.0, 60.5, 60.999999):
                c = (*day_before(y, m), 23, 59, s)
                t = sk.time(*c)
                assert_us(t, erfa_us_from_utc(*c), c)
                assert t.to_gregorian()[3:5] == (23, 59), c
            py, pm, pd = day_before(y, m)
            t = sk.time.from_rfc3339(f"{py:04d}-{pm:02d}-{pd:02d}T23:59:60.5Z")
            assert_us(t, erfa_us_from_utc(py, pm, pd, 23, 59, 60.5), "rfc3339")

    def test_first_seconds_of_1972(self):
        for s in (0.0, 0.5, 5.0, 9.0):
            c = (1972, 1, 1, 0, 0, s)
            assert_us(sk.time(*c), erfa_us_from_utc(*c), c)

    def test_1972_after_first_ten_seconds(self):
        for c in [(1972, 1, 1, 0, 0, 10.0), (1972, 1, 1, 6, 0, 0.0), (1972, 3, 1, 0, 0, 0.0)]:
            assert_us(sk.time(*c), erfa_us_from_utc(*c), c)

    def test_pre1961_utc_is_treated_as_tai(self):
        """Convention: satkit models UTC from 1961-01-01 (the first line of
        USNO tai-utc.dat) and takes TAI - UTC = 0 before it. ERFA's dat also
        has a 1960 entry (1.42 s), so 1960 labels deliberately differ from
        ERFA; see TestPre1972UTC for 1961 on."""
        for c in [(1950, 6, 1, 12, 0, 0.0), (1960, 6, 1, 12, 0, 0.0), (1960, 12, 31, 23, 59, 59.0)]:
            assert_us(sk.time(*c), pair_to_us(*erfa.dtf2d("TAI", *c)), c, tol_us=0)

    def test_pre1970_construction(self):
        c = (1965, 6, 1, 12, 30, 15.25)
        assert_us(sk.time(*c), erfa_us_from_utc(*c), c)

    def test_pre1970_gregorian_fields(self):
        for c in [(1969, 12, 31, 23, 0, 0.0), (1965, 6, 1, 12, 30, 15.25), (1901, 3, 1, 1, 2, 3.5)]:
            g = sk.time(*c).to_gregorian()
            assert g[:5] == c[:5] and g[5] == pytest.approx(c[5], abs=1e-7), (c, g)

    def test_random_utc_labels(self):
        """2000 random UTC labels 1973-2035 on 1/64 s steps (exactly
        representable, so satkit's seconds->microsecond truncation cannot
        bite) against ERFA dtf2d + utctai. First minute of the 1st of the
        month is excluded (leap-second boundaries are tested above)."""
        rng = np.random.default_rng(1)
        n = 0
        while n < 2000:
            c = (
                int(rng.integers(1973, 2036)),
                int(rng.integers(1, 13)),
                int(rng.integers(1, 29)),
                int(rng.integers(0, 24)),
                int(rng.integers(0, 60)),
                float(rng.integers(0, 60 * 64)) / 64.0,
            )
            if c[2] == 1 and c[3] == 0 and c[4] == 0:
                continue
            n += 1
            assert_us(sk.time(*c), erfa_us_from_utc(*c), c)

    def test_gregorian_seconds_round_to_microsecond(self):
        bad = []
        for us in range(0, 1_000_000, 997):
            s = 12.0 + us * 1e-6
            got = raw_us(sk.time(2020, 1, 1, 0, 0, s)) % 1_000_000
            if got != us:
                bad.append((us, got))
        assert not bad, f"{len(bad)} inputs truncated, e.g. {bad[:3]}"


# ----------------------------------------------------------------------------
# Pre-1972 ("rubber second") UTC, 1961-01-01 .. 1972-01-01
# ----------------------------------------------------------------------------

# 1st-of-month 00:00 UTC at which a pre-1972 segment of ERFA's dat starts
# (1961-01-01 excluded: satkit starts its model there, see
# test_pre1961_utc_is_treated_as_tai), plus 1972-01-01
PRE72_BOUNDARIES = [
    (1961, 8), (1962, 1), (1963, 11), (1964, 1), (1964, 4), (1964, 9), (1965, 1),
    (1965, 3), (1965, 7), (1965, 9), (1966, 1), (1968, 2), (1972, 1),
]  # fmt: skip


def prev_day(y, m, d=1):
    iy, im, iday, _ = erfa.jd2cal(2400000.5, erfa.cal2jd(y, m, d)[1] - 1.0)
    return int(iy), int(im), int(iday)


def erfa_step(y, m):
    """ERFA's TAI - UTC step (s) at 00:00 UTC on the 1st of (y, m): the new
    value less the preceding day's linear TAI - UTC carried to midnight"""
    pd = prev_day(y, m)
    end_of_prev = 2.0 * erfa.dat(*pd, 0.5) - erfa.dat(*pd, 0.0)
    return erfa.dat(y, m, 1, 0.0) - end_of_prev


def is_pre72_step_day(mjd_utc_day):
    """True for the UTC day that ends in a nonzero pre-1972 step"""
    for y, m in PRE72_BOUNDARIES:
        if abs(erfa_step(y, m)) > 1e-9 and erfa.cal2jd(y, m, 1)[1] - 1 == mjd_utc_day:
            return True
    return False


PRE72_STEP_DAYS = None


def pre72_step_days():
    """(y, m, d) of the days that end in a nonzero pre-1972 step"""
    global PRE72_STEP_DAYS
    if PRE72_STEP_DAYS is None:
        PRE72_STEP_DAYS = {prev_day(y, m) for y, m in PRE72_BOUNDARIES if abs(erfa_step(y, m)) > 1e-9}
    return PRE72_STEP_DAYS


def assert_label_matches_erfa(r, g, iy, im, iday, ihmsf):
    """satkit's label g of TAI microsecond count r against ERFA.

    ERFA's d2dtf only treats a day as a leap-second day when its step
    exceeds 0.5 s, so on the days that end in a fractional pre-1972 step it
    prints the quasi-JD fraction times 86400 (off by up to the step) rather
    than the label its own dtf2d + utctai map to that instant. satkit follows
    dtf2d + utctai; there the label must map back to r through ERFA."""
    if g[:3] in pre72_step_days():
        assert abs(erfa_us_from_utc(*g) - r) <= 1, (int(r), g)
        return
    exp = (int(iy), int(im), int(iday), int(ihmsf["h"]), int(ihmsf["m"]))
    sec = ihmsf["s"] + ihmsf["f"] * 1e-6
    assert g[:5] == exp and g[5] == pytest.approx(sec, abs=1.5e-6), (int(r), g, exp, sec)


def random_pre72_labels(n, seed):
    """Random microsecond UTC labels 1961-01-01 .. 1971-12-31"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        c = (
            int(rng.integers(1961, 1972)),
            int(rng.integers(1, 13)),
            int(rng.integers(1, 29)),
            int(rng.integers(0, 24)),
            int(rng.integers(0, 60)),
            float(rng.integers(0, 60_000_000)) * 1e-6,
        )
        out.append(c)
    return out


class TestPre1972UTC:
    def test_steps(self):
        """The steps of ERFA's table (and so of satkit's): +-0.05/0.1 s,
        zero where only the rate changes, +0.107758 s at 1972-01-01"""
        steps = {ym: round(erfa_step(*ym), 7) for ym in PRE72_BOUNDARIES}
        assert steps == {
            (1961, 8): -0.05, (1962, 1): 0.0, (1963, 11): 0.1, (1964, 1): 0.0, (1964, 4): 0.1,
            (1964, 9): 0.1, (1965, 1): 0.1, (1965, 3): 0.1, (1965, 7): 0.1, (1965, 9): 0.1,
            (1966, 1): 0.0, (1968, 2): -0.1, (1972, 1): 0.107758,
        }  # fmt: skip
        # satkit's step: elapsed TAI over the last UTC second of the day,
        # less that second's length (1 s plus its drift)
        for y, m in PRE72_BOUNDARIES:
            pd = prev_day(y, m)
            if steps[(y, m)] < 0:
                continue  # 23:59:59 + 1 s lands in the next day; covered below
            dt = (sk.time(y, m, 1) - sk.time(*pd, 23, 59, 59.0)).microseconds
            drift = erfa.dat(*pd, 1.0) - erfa.dat(*pd, 1.0 - 1.0 / 86400.0)
            assert abs(dt - 1e6 * (1.0 + drift + steps[(y, m)])) <= 1, (y, m, dt)

    def test_dat_every_day(self):
        """TAI - UTC at 00:00, 12:00 and 23:59:59 of every day 1961-1971
        against erfa.dat"""
        d0 = erfa.cal2jd(1961, 1, 1)[1]
        d1 = erfa.cal2jd(1972, 1, 1)[1]
        worst = 0
        for mjd in np.arange(d0, d1):
            y, m, d, _ = erfa.jd2cal(2400000.5, mjd)
            y, m, d = int(y), int(m), int(d)
            for h, mi, s, fd in ((0, 0, 0.0, 0.0), (12, 0, 0.0, 0.5), (23, 59, 59.0, 86399.0 / 86400.0)):
                got = raw_us(sk.time(y, m, d, h, mi, s)) - pair_to_us(*erfa.dtf2d("TAI", y, m, d, h, mi, s))
                err = got - erfa.dat(y, m, d, fd) * 1e6
                worst = max(worst, abs(err))
                assert abs(err) <= 0.5 + 1e-6, ((y, m, d, h, mi, s), err)
        assert worst <= 0.5 + 1e-6

    def test_random_labels(self):
        """3000 random microsecond labels against ERFA dtf2d + utctai"""
        for c in random_pre72_labels(3000, 11):
            assert_us(sk.time(*c), erfa_us_from_utc(*c), c)

    @pytest.mark.parametrize("ym", PRE72_BOUNDARIES, ids=[f"{y}-{m:02d}" for y, m in PRE72_BOUNDARIES])
    def test_labels_around_step(self, ym):
        """Labels across each boundary, including 23:59:60.x inside a positive
        step and the never-occurring labels of a negative step, which both
        satkit and ERFA put on the first |step| of the next day"""
        y, m = ym
        pd = prev_day(y, m)
        step = round(erfa_step(y, m), 7)
        labels = [(*pd, 23, 59, s) for s in (58.0, 58.5, 59.0, 59.5, 59.85, 59.899999, 59.9, 59.94, 59.96, 59.999999)]
        if step > 0:
            labels += [(*pd, 23, 59, 60.0 + f * step) for f in (0.0, 0.5)]
            labels.append((*pd, 23, 59, round(60.0 + step - 1e-6, 6)))
        labels += [(y, m, 1, 0, 0, s) for s in (0.0, 0.000001, 0.02, 0.05, 0.1, 0.5, 1.0)]
        for c in labels:
            t = sk.time(*c)
            assert_us(t, erfa_us_from_utc(*c), c)
            g = t.to_gregorian()
            missing = step < 0 and c[:3] == pd and c[5] >= 60.0 + step
            if missing:
                # ... and reads back as the next day's label
                assert g[:5] == (y, m, 1, 0, 0) and g[5] == pytest.approx(c[5] - 60.0 - step, abs=1.5e-6), (c, g)
            else:
                assert g[:5] == c[:5] and g[5] == pytest.approx(c[5], abs=1e-7), (c, g)
        if step <= 0:
            with pytest.raises(Exception):
                sk.time(*pd, 23, 59, 60.0)
        else:
            with pytest.raises(Exception):
                sk.time(*pd, 23, 59, 60.0 + step + 1e-6)

    @pytest.mark.parametrize("ym", PRE72_BOUNDARIES, ids=[f"{y}-{m:02d}" for y, m in PRE72_BOUNDARIES])
    def test_instants_to_labels_around_step(self, ym):
        """TAI instants within +/-2.5 s of each boundary, and every microsecond
        near both ends of a step, against ERFA (see assert_label_matches_erfa),
        and increasing labels (adjacent microseconds may share a label: a
        pre-1972 UTC microsecond is slightly longer than an SI one)"""
        y, m = ym
        r0 = erfa_us_from_utc(y, m, 1, 0, 0, 0.0)
        step_us = round(erfa_step(y, m) * 1e6)
        offsets = list(range(-2_500_000, 2_500_001, 125_000))
        for edge in {0, -step_us}:
            offsets += [edge + k for k in range(-3, 4)]
        raws = np.array(sorted({r0 + o for o in offsets}), dtype=np.int64)
        u1, u2 = erfa.taiutc(*tai_pair(raws))
        iy, im, iday, ihmsf = erfa.d2dtf("UTC", 6, u1, u2)
        prev = None
        for k, r in enumerate(raws):
            g = from_raw(r).to_gregorian()
            assert_label_matches_erfa(r, g, iy[k], im[k], iday[k], ihmsf[k])
            lab = (g[:5], g[5])
            if prev is not None:
                assert lab > prev[0] or (r - prev[1] == 1 and lab == prev[0]), (int(r - r0), prev, g)
            prev = (lab, r)

    def test_random_instants_to_labels(self):
        """2000 random TAI instants 1961-1971 against ERFA taiutc + d2dtf"""
        lo = erfa_us_from_utc(1961, 1, 1, 0, 0, 0.0)
        hi = erfa_us_from_utc(1972, 1, 1, 0, 0, 0.0)
        raws = np.random.default_rng(12).integers(lo, hi, 2000)
        u1, u2 = erfa.taiutc(*tai_pair(raws))
        iy, im, iday, ihmsf = erfa.d2dtf("UTC", 6, u1, u2)
        for k, r in enumerate(raws):
            assert_label_matches_erfa(r, from_raw(r).to_gregorian(), iy[k], im[k], iday[k], ihmsf[k])

    def test_to_mjd_utc_off_step_days(self):
        """UTC MJD against ERFA's quasi-JD, except on the days that end in a
        step (ERFA stretches or shrinks those days' fraction; satkit keeps
        86400 s, as for leap seconds)"""
        lo = erfa_us_from_utc(1961, 1, 1, 0, 0, 0.0)
        hi = erfa_us_from_utc(1972, 1, 1, 0, 0, 0.0)
        raws = np.random.default_rng(13).integers(lo, hi, 2000)
        u1, u2 = erfa.taiutc(*tai_pair(raws))
        keep = np.array([not is_pre72_step_day(np.floor((a - 2400000.5) + b)) for a, b in zip(u1, u2)])
        got = np.array([from_raw(r).to_mjd(TS.UTC) for r in raws])
        err = mjd_minus_pair_seconds(got, u1, u2)[keep]
        assert np.max(np.abs(err)) < TOL_MJD_S
        # and from_mjd inverts it
        for r, mjd in zip(raws[keep][:300], got[keep][:300]):
            assert abs(raw_us(sk.time.from_mjd(mjd)) - r) <= 2

    def test_tt_gps_tdb_of_labels(self):
        """TT / GPS of pre-1972 UTC labels (TDB follows TT)"""
        for c in random_pre72_labels(300, 14):
            a1, a2 = erfa.utctai(*erfa.dtf2d("UTC", *c))
            t = sk.time(*c)
            tt1, tt2 = erfa.taitt(a1, a2)
            assert mjd_minus_pair_seconds(t.to_mjd(TS.TT), tt1, tt2) == pytest.approx(0.0, abs=TOL_MJD_S), c
            gps = mjd_minus_pair_seconds(t.to_mjd(TS.GPS), a1, a2 - 19.0 / 86400.0)
            assert gps == pytest.approx(0.0, abs=TOL_MJD_S), c

    def test_unix_epoch_and_unixtime(self):
        """UNIX_EPOCH is the UTC label 1970-01-01T00:00:00 (TAI - UTC =
        8.000082 s), and Unix time stays UTC-based before 1972"""
        assert sk.time.UNIX_EPOCH == sk.time(1970, 1, 1)
        assert raw_us(sk.time.UNIX_EPOCH) == erfa_us_from_utc(1970, 1, 1, 0, 0, 0.0) == 8_000_082
        assert sk.time.UNIX_EPOCH.to_unixtime() == 0.0
        for c in random_pre72_labels(300, 15):
            ut = calendar.timegm((*c[:5], 0)) + c[5]
            t = sk.time.from_unixtime(ut)
            assert_us(t, erfa_us_from_utc(*c), c)
            assert t.to_unixtime() == pytest.approx(ut, abs=1e-6)

    def test_1961_step(self):
        """satkit's own 1.422818 s step from TAI-aligned labels to the 1961
        segment: an inserted interval 1960-12-31T23:59:60 .. 23:59:61.422817"""
        t = sk.time(1961, 1, 1)
        assert raw_us(t) == erfa_us_from_utc(1961, 1, 1, 0, 0, 0.0)
        assert (t - sk.time(1960, 12, 31, 23, 59, 59.0)).microseconds == 2_422_818
        t = sk.time(1960, 12, 31, 23, 59, 61.25)
        assert str(t) == "1960-12-31T23:59:61.250000Z"
        assert sk.time.from_rfc3339(str(t)) == t
        assert str(from_raw(raw_us(sk.time(1961, 1, 1)) - 1)) == "1960-12-31T23:59:61.422817Z"
        with pytest.raises(Exception):
            sk.time(1960, 12, 31, 23, 59, 61.422818)


# ----------------------------------------------------------------------------
# Time scales: MJD in UTC / TAI / TT / GPS / TDB
# ----------------------------------------------------------------------------


def _random_raws(n, seed, y0=1973, y1=2030):
    rng = np.random.default_rng(seed)
    lo = int((y0 - 1970) * 365.25 * 86400e6)
    hi = int((y1 - 1970) * 365.25 * 86400e6)
    return rng.integers(lo, hi, n)


class TestScales:
    def test_to_mjd_tai_tt_gps(self):
        raws = _random_raws(2000, 2)
        a1, a2 = tai_pair(raws)
        t1, t2 = erfa.taitt(a1, a2)
        times = [from_raw(r) for r in raws]
        for scale, (e1, e2) in (
            (TS.TAI, (a1, a2)),
            (TS.TT, (t1, t2)),
            (TS.GPS, (a1, a2 - 19.0 / 86400.0)),  # GPS = TAI - 19 s
        ):
            got = np.array([t.to_mjd(scale) for t in times])
            err = mjd_minus_pair_seconds(got, e1, e2)
            assert np.max(np.abs(err)) < TOL_MJD_S, (scale, np.max(np.abs(err)))

    def test_to_mjd_utc_off_leap_days(self):
        """UTC MJD against ERFA's quasi-JD, excluding days that end in a leap
        second: ERFA stretches such a day to 86401 s (fraction =
        seconds/86401), satkit keeps 86400 s and repeats 23:59:59 through the
        leap second. Both are conventions."""
        raws = _random_raws(2000, 3)
        u1, u2 = erfa.taiutc(*tai_pair(raws))
        leap_days = {erfa.cal2jd(*day_before(y, m))[1] for (y, m) in LEAPS}
        keep = np.array([np.floor((a - 2400000.5) + b) not in leap_days for a, b in zip(u1, u2)])
        got = np.array([from_raw(r).to_mjd(TS.UTC) for r in raws])
        err = mjd_minus_pair_seconds(got, u1, u2)[keep]
        assert np.max(np.abs(err)) < TOL_MJD_S

    def test_constructors_with_scale(self):
        """time(y, m, d, h, mi, s, scale=...) for TAI / TT / GPS"""
        rng = np.random.default_rng(4)
        for i in range(300):
            c = (
                int(rng.integers(1975, 2035)),
                int(rng.integers(1, 13)),
                int(rng.integers(1, 29)),
                int(rng.integers(0, 24)),
                int(rng.integers(0, 60)),
                float(rng.integers(0, 60 * 64)) / 64.0,
            )
            exp = {
                "TAI": erfa.dtf2d("TAI", *c),
                "TT": erfa.tttai(*erfa.dtf2d("TT", *c)),
                "GPS": np.add(erfa.dtf2d("TAI", *c), (0.0, 19.0 / 86400.0)),
            }
            for name, (d1, d2) in exp.items():
                got = raw_us(sk.time(*c, scale=getattr(TS, name))) * 1e-6
                assert got == pytest.approx(float(pair_to_tai_seconds(d1, d2)), abs=TOL_MJD_S), (name, c)

    def test_j2000(self):
        assert_us(sk.time.J2000, pair_to_us(*erfa.tttai(2451545.0, 0.0)), "J2000", tol_us=0)
        assert raw_us(sk.time(2000, 1, 1, 12, 0, 0.0, scale=TS.TT)) == raw_us(sk.time.J2000)

    def test_gps_week_and_second(self):
        epoch = erfa_us_from_utc(1980, 1, 6, 0, 0, 0.0)
        for wk, sow in [(0, 0.0), (1, 0.5), (1042, 604799.5), (2238, 345600.25), (2400, 0.0)]:
            t = sk.time.from_gps_week_and_second(wk, sow)
            assert_us(t, epoch + round((wk * 604800 + sow) * 1e6), (wk, sow), tol_us=0)

    def test_from_unixtime(self):
        """Unix time counts UTC days of 86400 s (away from leap-second midnights)"""
        rng = np.random.default_rng(5)
        for i in range(500):
            c = (int(rng.integers(1973, 2035)), int(rng.integers(1, 13)), int(rng.integers(2, 29)),
                 int(rng.integers(0, 24)), int(rng.integers(0, 60)), float(rng.integers(0, 60 * 64)) / 64.0)
            ut = calendar.timegm((*c[:5], 0)) + c[5]
            t = sk.time.from_unixtime(ut)
            assert_us(t, erfa_us_from_utc(*c), c)
            assert t.to_unixtime() == pytest.approx(ut, abs=1e-6)


class TestTDB:
    """satkit's TDB - TT is the one-term series 0.001657 s sin(628.3076 T +
    6.2401) (Vallado Eq. 3-50, leading Fairhead-Bretagnon term). Against the
    full geocentric series (erfa.dtdb, observer at the geocentre) the one-term
    series is good to 54 us max / 20 us RMS over 1900-2100 (checked below), so
    60 us is the tolerance against dtdb."""

    TOL_ONE_TERM_VS_DTDB = 60e-6

    @staticmethod
    def _tt_pairs(n=2000):
        a1, a2 = tai_pair(_random_raws(n, 6, 1901, 2099))
        return erfa.taitt(a1, a2)

    def test_one_term_series_accuracy(self):
        """Pure ERFA check that justifies the tolerance above"""
        t1, t2 = self._tt_pairs()
        T = ((t1 - 2451545.0) + t2) / 36525.0
        d = 0.001657 * np.sin(628.3076 * T + 6.2401) - erfa.dtdb(t1, t2, 0.0, 0.0, 0.0, 0.0)
        assert np.max(np.abs(d)) < self.TOL_ONE_TERM_VS_DTDB
        assert np.sqrt(np.mean(d**2)) < 25e-6

    def test_tdb_minus_tt_is_the_one_term_series(self):
        t1, t2 = self._tt_pairs()
        T = ((t1 - 2451545.0) + t2) / 36525.0
        expected = 0.001657 * np.sin(628.3076 * T + 6.2401)
        mjd_tt = (t1 - 2400000.5) + t2
        times = [sk.time.from_mjd(m, TS.TT) for m in mjd_tt]
        got = np.array([(t.to_mjd(TS.TDB) - t.to_mjd(TS.TT)) * 86400.0 for t in times])
        assert np.max(np.abs(got - expected)) < 2 * TOL_MJD_S

    def test_tdb_vs_erfa_dtdb(self):
        t1, t2 = self._tt_pairs()
        mjd_tt = (t1 - 2400000.5) + t2
        times = [sk.time.from_mjd(m, TS.TT) for m in mjd_tt]
        got = np.array([(t.to_mjd(TS.TDB) - t.to_mjd(TS.TT)) * 86400.0 for t in times])
        assert np.max(np.abs(got - erfa.dtdb(t1, t2, 0.0, 0.0, 0.0, 0.0))) < self.TOL_ONE_TERM_VS_DTDB

    def test_tdb_constructor(self):
        rng = np.random.default_rng(7)
        for i in range(200):
            c = (int(rng.integers(1975, 2035)), int(rng.integers(1, 13)), int(rng.integers(1, 29)),
                 int(rng.integers(0, 24)), int(rng.integers(0, 60)), float(rng.integers(0, 60 * 64)) / 64.0)
            b1, b2 = erfa.dtf2d("TDB", *c)
            tt1, tt2 = erfa.tdbtt(b1, b2, erfa.dtdb(b1, b2, 0.0, 0.0, 0.0, 0.0))
            exp = float(pair_to_tai_seconds(*erfa.tttai(tt1, tt2)))
            assert raw_us(sk.time(*c, scale=TS.TDB)) * 1e-6 == pytest.approx(exp, abs=self.TOL_ONE_TERM_VS_DTDB), c


# ----------------------------------------------------------------------------
# UT1 and Earth orientation parameters
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eop_span():
    """(first, last_observed) UTC MJD of the loaded EOP table"""
    cov = ft.eop_coverage()
    if cov is None:
        pytest.skip("no EOP table loaded")
    return cov[0].to_mjd(), cov[1].to_mjd()


@pytest.fixture(scope="module")
def eop_rows():
    rows = eop_file_rows()
    if rows is None or len(rows) < 100:
        pytest.skip("EOP file not found in the data search directories")
    return rows


def _erfa_ut1(times):
    """ERFA UT1 two-part JD from satkit's TAI, with satkit's UT1 - UTC.

    Before 1972 erfa.utcut1 forms UT1 - TAI with TAI - UTC at 0h of the day,
    so it drops that day's drift of TAI - UTC (up to 2.6 ms by 24h) and is
    not UT1 = UTC + (UT1 - UTC). There UT1 = TAI + (UT1 - UTC) - (TAI - UTC)
    is formed with erfa.taiut1 and erfa.dat at the instant instead."""
    raws = np.array([raw_us(t) for t in times], dtype=np.int64)
    a1, a2 = tai_pair(raws)
    u1, u2 = erfa.taiutc(a1, a2)
    dut1 = np.array([ft.earth_orientation_params(t)[0] for t in times])
    e1, e2 = erfa.utcut1(u1, u2, dut1)
    for k in np.nonzero((u1 - 2400000.5) + u2 < 41317.0)[0]:
        iy, im, iday, fd = erfa.jd2cal(u1[k], u2[k])
        if fd > 1.0 - 1e-12:
            # taiutc can return 00:00 of a step day as 1e-14 d before it
            iy, im, iday, _ = erfa.jd2cal(u1[k], u2[k] + 1e-9)
            fd = 0.0
        e1[k], e2[k] = erfa.taiut1(a1[k], a2[k], dut1[k] - erfa.dat(iy, im, iday, fd))
    return e1, e2


class TestUT1:
    def test_ut1_vs_erfa_utcut1(self, eop_span):
        """UT1 = UTC + (UT1 - UTC) with satkit's own UT1 - UTC fed to ERFA"""
        rng = np.random.default_rng(8)
        times = [sk.time.from_mjd(m) for m in rng.uniform(eop_span[0] + 1, eop_span[1] - 1, 1000)]
        e1, e2 = _erfa_ut1(times)
        err = mjd_minus_pair_seconds([t.to_mjd(TS.UT1) for t in times], e1, e2)
        assert np.max(np.abs(err)) < TOL_MJD_S

    def test_ut1_round_trip(self, eop_span):
        """from_mjd(UT1) inverts to_mjd(UT1). from_mjd evaluates UT1 - UTC at the
        UT1 value itself (< 0.9 s off), which moves it by < 0.1 us except on a
        leap-second day, which is excluded here (defect #6)."""
        rng = np.random.default_rng(9)
        leap_days = {erfa.cal2jd(*day_before(y, m))[1] for (y, m) in LEAPS}
        mjds = [m for m in rng.uniform(eop_span[0] + 1, eop_span[1] - 1, 1000) if np.floor(m) not in leap_days]
        for m in mjds:
            t = sk.time.from_mjd(m)
            t2 = sk.time.from_mjd(t.to_mjd(TS.UT1), TS.UT1)
            assert abs(raw_us(t2) - raw_us(t)) <= 2, m

    def test_ut1_across_every_step(self, eop_span):
        """Across every UTC step inside the EOP table (the pre-1972 steps and
        rate changes, 1972-01-01's 0.107758 s, every leap second), UT1 - TAI
        is continuous to f64 MJD resolution (UT1 - UTC is interpolated as UT1
        - TAI), UT1 is monotonic and invertible, and it matches ERFA. The
        pre-1972 steps need an EOP table reaching before 1972 (CelesTrak's
        EOP-All.csv next to finals2000A.all)."""
        steps = [
            (y, m)
            for (y, m) in PRE72_BOUNDARIES + LEAPS
            if eop_span[0] < erfa.cal2jd(y, m, 1)[1] - 1 and erfa.cal2jd(y, m, 1)[1] + 1 < eop_span[1]
        ]
        assert steps
        for y, m in steps:
            r0 = erfa_us_from_utc(y, m, 1, 0, 0, 0.0)
            # every 50 ms over +/-2 s, plus inside a 0.1 s inserted interval
            offsets = sorted(set(range(-2_000_000, 2_000_001, 50_000)) | {-50_000, -1, 1})
            times = [from_raw(r0 + o) for o in offsets]
            ut1 = np.array([t.to_mjd(TS.UT1) for t in times])
            d = (ut1 - np.array([t.to_mjd(TS.TAI) for t in times])) * 86400.0
            # UT1 - TAI changes by < 5 ms/day, i.e. < 0.25 us over these 4 s;
            # an f64 MJD resolves ~0.6 us
            assert np.ptp(d) < 2e-6, (y, m, np.ptp(d))
            assert np.all(np.diff(ut1) > 0), (y, m)
            for t, u in zip(times, ut1):
                assert abs(raw_us(sk.time.from_mjd(u, TS.UT1)) - raw_us(t)) <= 2, (y, m, str(t))
            e1, e2 = _erfa_ut1(times)
            err = mjd_minus_pair_seconds(ut1, e1, e2)
            assert np.max(np.abs(err)) < TOL_MJD_S, (y, m, np.max(np.abs(err)))

    def test_ut1_is_utc_without_eop(self, eop_span):
        """Documented fallback: before the EOP table (or with none), UT1 - UTC
        is 0, so UT1 is the UTC MJD and repeats UTC's steps (as erfa.utcut1
        with dut1 = 0). Checked at the steps before the loaded table."""
        steps = [(y, m) for (y, m) in [(1961, 1)] + PRE72_BOUNDARIES + LEAPS if erfa.cal2jd(y, m, 1)[1] + 1 < eop_span[0]]
        if not steps:
            pytest.skip("EOP table covers every step")
        for y, m in steps:
            for c in [(*prev_day(y, m), 23, 59, 59.5), (y, m, 1, 0, 0, 0.5)]:
                t = sk.time(*c)
                assert t.to_mjd(TS.UT1) == t.to_mjd(TS.UTC), c

    def test_ut1_inside_leap_second(self, eop_span):
        times = []
        for y, m in LEAPS:
            if erfa.cal2jd(y, m, 1)[1] - 1 <= eop_span[0] or erfa.cal2jd(y, m, 1)[1] > eop_span[1]:
                continue
            r0 = erfa_us_from_utc(y, m, 1, 0, 0, 0.0)
            times += [from_raw(r0 - 1_000_000 + o) for o in (1, 250_000, 500_000, 999_999)]
        assert times
        e1, e2 = _erfa_ut1(times)
        err = mjd_minus_pair_seconds([t.to_mjd(TS.UT1) for t in times], e1, e2)
        assert np.max(np.abs(err)) < TOL_MJD_S, np.max(np.abs(err))
        for t in times:
            t2 = sk.time.from_mjd(t.to_mjd(TS.UT1), TS.UT1)
            assert abs(raw_us(t2) - raw_us(t)) <= 20, str(t)

    def test_eop_at_table_rows(self, eop_rows, eop_span):
        """Every 5th row returned exactly (UT1-UTC s, xp/yp arcsec, dX/dY mas)"""
        for r in eop_rows[::5]:
            if not (eop_span[0] <= r[0] <= eop_span[1]):
                continue
            dut1, xp, yp, _, dx, dy = ft.earth_orientation_params(sk.time.from_mjd(r[0]))
            assert (dut1, xp, yp, dx, dy) == pytest.approx((r[3], r[1], r[2], r[4], r[5]), abs=1e-12), r[0]

    def test_eop_linear_between_rows(self, eop_rows, eop_span):
        """Linear interpolation between adjacent rows. UT1 - UTC is interpolated
        as UT1 - TAI, so every Delta-AT step between two rows (leap seconds and
        the pre-1972 fractional steps) and the pre-1972 daily drift are removed
        exactly; Delta-AT here comes independently from erfa.dat."""

        def dat_at(mjd):
            iy, im, iday, fd = erfa.jd2cal(2400000.5, mjd)
            return erfa.dat(iy, im, iday, fd)

        for i in range(0, len(eop_rows) - 1, 7):
            r0, r1 = eop_rows[i], eop_rows[i + 1]
            if not (eop_span[0] <= r0[0] and r1[0] <= eop_span[1]):
                continue
            d0, d1 = dat_at(r0[0]), dat_at(r1[0])
            for g in (0.25, 0.5, 0.9):
                v = ft.earth_orientation_params(sk.time.from_mjd(r0[0] + g))
                exp = (1 - g) * r0 + g * r1
                exp_dut1 = (1 - g) * (r0[3] - d0) + g * (r1[3] - d1) + dat_at(r0[0] + g)
                assert v[0] == pytest.approx(exp_dut1, abs=2e-6), r0[0] + g
                assert (v[1], v[2], v[4], v[5]) == pytest.approx(
                    (exp[1], exp[2], exp[4], exp[5]), abs=1e-9
                ), r0[0] + g

    def test_eop_interpolation_across_leap_second(self, eop_rows):
        """UT1 - UTC steps by 1 s at 00:00 UTC after a leap second; on the day
        before, interpolation must use the continuous UT1 - TAI"""
        jumps = np.nonzero(np.abs(np.diff(eop_rows[:, 3])) > 0.5)[0]
        assert len(jumps) > 0
        for i in jumps:
            r0, r1 = eop_rows[i], eop_rows[i + 1]
            step = np.round(r1[3] - r0[3])
            for g in (0.25, 0.5, 0.9):
                got = ft.earth_orientation_params(sk.time.from_mjd(r0[0] + g))[0]
                assert got == pytest.approx((1 - g) * r0[3] + g * (r1[3] - step), abs=1e-6), r0[0] + g


# ----------------------------------------------------------------------------
# Earth rotation, sidereal time, precession-nutation, full reduction
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref(eop_span):
    """Shared sample of 400 UTC epochs in the observed EOP span with ERFA
    reference quantities, using satkit's EOP values throughout"""
    rng = np.random.default_rng(20260925)
    first = max(eop_span[0], 41684.0) + 1.0
    times = [sk.time.from_mjd(m) for m in rng.uniform(first, eop_span[1] - 1.0, 400)]
    raws = np.array([raw_us(t) for t in times], dtype=np.int64)
    tai = tai_pair(raws)
    utc = erfa.taiutc(*tai)
    tt = erfa.taitt(*tai)
    eop = np.array([ft.earth_orientation_params(t) for t in times])
    dut1, xp, yp, dx, dy = eop[:, 0], eop[:, 1] * AS2RAD, eop[:, 2] * AS2RAD, eop[:, 4] * MAS2RAD, eop[:, 5] * MAS2RAD
    ut1 = erfa.utcut1(*utc, dut1)
    x, y = erfa.xy06(*tt)
    x, y = x + dx, y + dy
    s = erfa.s06(*tt, x, y)
    c2i = erfa.c2ixys(x, y, s)  # GCRS -> CIRS
    pom = erfa.pom00(xp, yp, erfa.sp00(*tt))  # TIRS -> ITRS
    era = erfa.era00(*ut1)
    c2t = erfa.c2tcio(c2i, era, pom)  # GCRS -> ITRS
    gmst82 = erfa.gmst82(*ut1)
    teme2itrf = pom @ erfa.rz(gmst82, np.broadcast_to(np.eye(3), (len(times), 3, 3)))
    return dict(times=times, tt=tt, ut1=ut1, x=x, y=y, s=s, c2i=c2i, pom=pom, era=era, c2t=c2t,
                gmst82=gmst82, teme2itrf=teme2itrf)


def _assert_rot(sk_mats, erfa_mats, tol_rad, what):
    err = rot_angle(sk_mats, erfa_mats)
    i = int(np.argmax(err))
    assert err[i] < tol_rad, (
        f"{what}: {err[i] / MAS2RAD:.4f} mas = {err[i] * R_LEO * 100:.2f} cm at LEO, "
        f"{err[i] * R_GEO * 100:.2f} cm at GEO (sample {i})"
    )


# Earth rotation angle, evaluated like ERFA era00 from the two-part date
# (2400000.5, MJD(UT1)): UT1 as an f64 MJD (<= 0.6 us here) is < 10 us of arc
# (measured 8.9 uas max; 0.3 mm at LEO). Before the two-part evaluation,
# JD(UT1) = MJD + 2400000.5 as one f64 (ulp 40 us) cost up to 0.30 mas. ERA
# precision is the largest term in the full reduction, so 20 uas is also the
# tolerance for every ERA-dependent rotation below.
TOL_ERA = 20 * UAS2RAD
# CIP X, Y (Tables 5.2a/b) against ERFA's xy06 of the same series: measured
# 3.3 uas over 1973-2026 (ERFA's own xy06 and xys06a differ by 1.5 uas).
# 10 uas = 0.3 mm at LEO.
TOL_CIP = 10 * UAS2RAD


class TestEarthRotation:
    def test_era_vs_era00(self, ref):
        got = np.array(ft.earth_rotation_angle(ref["times"]))
        assert np.max(np.abs(wrap(got - ref["era"]))) < TOL_ERA

    def test_gmst_is_gmst82(self, ref):
        """satkit's GMST is the IAU 1982 expression (Vallado Alg. 15), UT1 based.
        30 uas covers the f64 MJD(UT1)."""
        got = np.array(ft.gmst(ref["times"]))
        assert np.max(np.abs(wrap(got - ref["gmst82"]))) < 30 * UAS2RAD

    def test_equation_of_equinoxes(self, ref):
        """satkit's eqeq is the two-term approximation (dPsi = -17.2" sin Omega
        - 1.3" sin 2L, times cos eps). Against the IAU 1994 equation of the
        equinoxes (full 1980 nutation) it is good to 0.56" (37 ms of time)
        over this sample and 0.65" (43 ms) over a dense 1950-2100 sample (the
        figure the docstrings quote); the tolerance is 0.7", since the sample
        moves with the end of the EOP table."""
        got = np.array(ft.eqeq(ref["times"]))
        assert np.max(np.abs(got - erfa.eqeq94(*ref["tt"]))) < 0.7 * AS2RAD
        gast = np.array(ft.gast(ref["times"]))
        assert np.max(np.abs(wrap(gast - erfa.gst94(*ref["ut1"])))) < 0.7 * AS2RAD


class TestPrecessionNutation:
    def test_cip_xy_and_cio_locator(self, ref):
        c2i = matrices(ft.qcirs2gcrf(ref["times"])).transpose(0, 2, 1)  # GCRS -> CIRS
        x, y = c2i[:, 2, 0], c2i[:, 2, 1]
        assert np.max(np.abs(x - ref["x"])) < TOL_CIP
        assert np.max(np.abs(y - ref["y"])) < TOL_CIP
        # c2ixys(X, Y, s) = R3(-s) . c2ixys(X, Y, 0), so s is the residual z rotation
        d = c2i @ erfa.c2ixys(x, y, 0.0).transpose(0, 2, 1)
        s = -np.arctan2(d[:, 0, 1], d[:, 0, 0])
        assert np.max(np.abs(s - ref["s"])) < 1 * UAS2RAD
        _assert_rot(c2i, ref["c2i"], TOL_CIP, "GCRS->CIRS")

    def test_cirs_model_only(self):
        """Epochs before any EOP table (1900-1960), where satkit applies no
        dX/dY: the bare IAU 2006/2000A GCRS->CIRS matrix against ERFA's
        series (xy06 + s06) and its matrix route (c2i06a). The difference
        grows slowly with |T| (3 uas at 1973-2026, 12 uas at 1900); ERFA's
        own two routes differ by 4 uas at 1900. Tolerance 20 uas (0.7 mm at
        LEO)."""
        raws = _random_raws(200, 10, 1900, 1960)
        times = [from_raw(r) for r in raws]
        tt = erfa.taitt(*tai_pair(raws))
        c2i = matrices(ft.qcirs2gcrf(times)).transpose(0, 2, 1)
        x, y = erfa.xy06(*tt)
        _assert_rot(c2i, erfa.c2ixys(x, y, erfa.s06(*tt, x, y)), 20 * UAS2RAD, "GCRS->CIRS vs xy06 (no EOP)")
        _assert_rot(c2i, erfa.c2i06a(*tt), 20 * UAS2RAD, "GCRS->CIRS vs c2i06a (no EOP)")

    def test_polar_motion(self, ref):
        """W = R3(-s') R2(xp) R1(yp), s' = -47 uas T (sp00)"""
        w = matrices(ft.qitrf2tirs(ref["times"])).transpose(0, 2, 1)  # TIRS -> ITRS
        _assert_rot(w, ref["pom"], 1e-3 * UAS2RAD, "polar motion")

    def test_full_reduction(self, ref):
        _assert_rot(matrices(ft.qgcrf2itrf(ref["times"])), ref["c2t"], TOL_ERA, "qgcrf2itrf")
        _assert_rot(
            matrices(ft.rotation(sk.frame.GCRF, sk.frame.ITRF, ref["times"])), ref["c2t"], TOL_ERA,
            "rotation(GCRF, ITRF)",
        )

    def test_intermediate_frames(self, ref):
        F = sk.frame
        n = len(ref["times"])
        eye = np.broadcast_to(np.eye(3), (n, 3, 3))
        _assert_rot(matrices(ft.rotation(F.GCRF, F.CIRS, ref["times"])), ref["c2i"], TOL_CIP, "GCRF->CIRS")
        _assert_rot(
            matrices(ft.rotation(F.CIRS, F.TIRS, ref["times"])), erfa.rz(ref["era"], eye), TOL_ERA, "CIRS->TIRS"
        )
        _assert_rot(matrices(ft.rotation(F.TIRS, F.ITRF, ref["times"])), ref["pom"], 1e-3 * UAS2RAD, "TIRS->ITRF")
        _assert_rot(matrices(ft.rotation(F.GCRF, F.ICRF, ref["times"])), eye, 1e-15, "GCRF->ICRF")

    def test_frame_bias_eme2000(self, ref):
        """GCRF -> EME2000 is the IERS 2010 frame bias; against the IAU 2006
        bias matrix of bp06 (0.2 uas apart: bi00 vs Fukushima-Williams)"""
        rb, _, _ = erfa.bp06(*ref["tt"])
        _assert_rot(matrices(ft.rotation(sk.frame.GCRF, sk.frame.EME2000, ref["times"])), rb, 1 * UAS2RAD, "bias")

    def test_mod_precession(self, ref):
        """qmod2gcrf is IAU 2006 precession (zeta_A, z_A, theta_A) from the
        J2000 mean equator/equinox, without frame bias: it matches the
        precession-only matrix rp of bp06, and so differs from the GCRS-based
        rbp by the 23 mas frame bias despite its name"""
        _, rp, rbp = erfa.bp06(*ref["tt"])
        m = matrices(ft.qmod2gcrf(ref["times"])).transpose(0, 2, 1)
        _assert_rot(m, rp, 1 * UAS2RAD, "MOD precession")
        assert np.all(np.abs(rot_angle(m, rbp) / MAS2RAD - 23.1) < 0.2)

    def test_tod2mod_approx(self, ref):
        """Two-term nutation (documented as ~0.9"); measured 0.80" max on this
        sample and 0.88" over a dense 1950-2100 sample against IAU 2006/2000A
        nutation"""
        tod2mod = matrices(ft.qtod2mod_approx(ref["times"]))
        mod2tod = erfa.numat(erfa.obl06(*ref["tt"]), *erfa.nut06a(*ref["tt"]))
        _assert_rot(tod2mod.transpose(0, 2, 1), mod2tod, 1.0 * AS2RAD, "TOD->MOD approx")


class TestApproxReduction:
    """The '_approx' reduction (IAU 2006 precession + two-term nutation +
    two-term equation of the equinoxes) is documented as ~1 arcsec. Against
    the full ERFA reduction GCRF <-> ITRF measures 0.98" max (34 m at LEO,
    200 m at GEO) over 1973-2026, of which up to 0.6" is polar motion, which
    the approx chain neglects. Tolerance 1.05".

    TEME -> GCRF involves no polar motion, so the same chain applied to PEF
    (TEME rotated by GMST82 alone) is better: 0.55" max (the nutation
    approximations); tolerance 0.6"."""

    TOL = 1.05 * AS2RAD
    TOL_TEME = 0.6 * AS2RAD

    def test_gcrf_itrf_approx(self, ref):
        _assert_rot(matrices(ft.qgcrf2itrf_approx(ref["times"])), ref["c2t"], self.TOL, "qgcrf2itrf_approx")
        _assert_rot(
            matrices(ft.rotation_approx(sk.frame.GCRF, sk.frame.ITRF, ref["times"])), ref["c2t"], self.TOL,
            "rotation_approx(GCRF, ITRF)",
        )

    def test_teme_gcrf_approx(self, ref):
        teme2gcrf = ref["c2t"].transpose(0, 2, 1) @ ref["teme2itrf"]
        F = sk.frame
        _assert_rot(matrices(ft.qteme2gcrf(ref["times"])), teme2gcrf, self.TOL_TEME, "qteme2gcrf")
        for to, pre in ((F.GCRF, None), (F.ICRF, None), (F.EME2000, erfa.bp06(*ref["tt"])[0])):
            got = matrices(ft.rotation_approx(F.TEME, to, ref["times"]))
            _assert_rot(got, teme2gcrf if pre is None else pre @ teme2gcrf, self.TOL_TEME, f"rotation_approx(TEME, {to})")

    def test_teme_gcrf_approx_has_no_polar_motion(self, ref):
        """TEME -> GCRF involves no polar motion. Against the same approx chain
        applied to PEF (= ITRF without polar motion), qteme2gcrf and
        rotation_approx(TEME, GCRF) agree to rounding."""
        n = len(ref["times"])
        pef2teme = erfa.rz(-np.array(ft.gmst(ref["times"])), np.broadcast_to(np.eye(3), (n, 3, 3)))
        expected = matrices(ft.qitrf2gcrf_approx(ref["times"])) @ pef2teme.transpose(0, 2, 1)
        _assert_rot(matrices(ft.qteme2gcrf(ref["times"])), expected, 1 * UAS2RAD, "qteme2gcrf vs PM-free chain")
        _assert_rot(
            matrices(ft.rotation_approx(sk.frame.TEME, sk.frame.GCRF, ref["times"])), expected, 1 * UAS2RAD,
            "rotation_approx(TEME, GCRF) vs PM-free chain",
        )


class TestTEME:
    """satkit's TEME is Vallado et al. (2006): TEME -> PEF by GMST82 alone
    (no equation of the equinoxes), then polar motion PEF -> ITRF."""

    def test_teme_itrf(self, ref):
        _assert_rot(matrices(ft.qteme2itrf(ref["times"])), ref["teme2itrf"], 30 * UAS2RAD, "qteme2itrf")
        _assert_rot(
            matrices(ft.rotation(sk.frame.TEME, sk.frame.ITRF, ref["times"])), ref["teme2itrf"], 30 * UAS2RAD,
            "rotation(TEME, ITRF)",
        )

    def test_teme_gcrf_full(self, ref):
        teme2gcrf = ref["c2t"].transpose(0, 2, 1) @ ref["teme2itrf"]
        _assert_rot(
            matrices(ft.rotation(sk.frame.TEME, sk.frame.GCRF, ref["times"])), teme2gcrf, TOL_ERA,
            "rotation(TEME, GCRF)",
        )
        rb, _, _ = erfa.bp06(*ref["tt"])
        _assert_rot(
            matrices(ft.rotation(sk.frame.TEME, sk.frame.EME2000, ref["times"])), rb @ teme2gcrf, TOL_ERA,
            "rotation(TEME, EME2000)",
        )

    def test_teme_vs_iau80_construction(self, ref):
        """The other classical TEME: TOD (IAU 1976/1980, pnm80) rotated by the
        IAU 1994 equation of the equinoxes. It differs from the GMST/ITRF route
        by the IAU 1980 vs 2006/2000A model difference, the EOP nutation
        corrections it omits, and the 23 mas frame bias: measured 50 mas; the
        tolerance is 0.1"."""
        eye = np.broadcast_to(np.eye(3), (len(ref["times"]), 3, 3))
        teme2j2000 = erfa.pnm80(*ref["tt"]).transpose(0, 2, 1) @ erfa.rz(-erfa.eqeq94(*ref["tt"]), eye)
        _assert_rot(
            matrices(ft.rotation(sk.frame.TEME, sk.frame.GCRF, ref["times"])), teme2j2000, 0.1 * AS2RAD, "TEME (IAU80)"
        )


class TestStateTransform:
    def test_itrf_to_gcrf_velocity(self, ref):
        """Velocity against the numerical derivative of ERFA's full matrix.
        satkit keeps only omega x r (omega = 7.292115e-5 rad/s, no LOD) and
        treats precession-nutation as static: 50"/yr x 7000 km = 5e-5 m/s.
        Tolerance 1e-4 m/s; position to 1 mm (ERA precision, above: 20 uas
        is 0.7 mm at this radius)."""
        r = np.array([4066.8e3, 4337.9e3, 3253.4e3])
        v = np.array([-2000.0, 5000.0, 4000.0])
        for i, t in enumerate(ref["times"][:60]):
            pos, vel = ft.itrf_to_gcrf_state(r, v, t)
            m = ref["c2t"][i].T
            h = 1.0
            ts = [t - sk.duration(seconds=h), t + sk.duration(seconds=h)]
            e = _erfa_c2t_at(ts)
            v_exp = m @ v + (e[1].T - e[0].T) @ r / (2 * h)
            assert np.linalg.norm(pos - m @ r) < 1e-3
            assert np.linalg.norm(vel - v_exp) < 1e-4


    def test_teme_to_itrf_state(self, ref):
        """TEME -> ITRF state against Vallado et al. (2006) built from ERFA:
        r = W R3(gmst82) r_teme, v = W (R3(gmst82) v_teme - w x R3(gmst82) r_teme),
        w = 7.292115146706979e-5 (1 - LOD / 86400) rad/s. satkit uses the
        nominal 7.292115e-5 rad/s and no LOD (1.7e-5 m/s at LEO), and goes
        TEME -> PEF -> ITRF directly (no GCRF leg), identically in the full and
        approximate modes. Tolerance 1e-4 m/s; position to 1 mm (GMST82
        precision, 30 uas = 1 mm at LEO)."""
        r = np.array([4066.8e3, 4337.9e3, 3253.4e3])
        v = np.array([-2000.0, 5000.0, 4000.0])
        eye = np.eye(3)
        for i, t in enumerate(ref["times"][:60]):
            lod = ft.earth_orientation_params(t)[3]
            omega = np.array([0.0, 0.0, 7.292115146706979e-5 * (1.0 - lod / 86400.0)])
            r3 = erfa.rz(ref["gmst82"][i], eye)
            w = ref["pom"][i]
            r_exp = w @ r3 @ r
            v_exp = w @ (r3 @ v - np.cross(omega, r3 @ r))
            for approx in (False, True):
                fn = ft.transform_state_approx if approx else ft.transform_state
                pos, vel = fn(sk.frame.TEME, sk.frame.ITRF, t, r, v)
                assert np.linalg.norm(pos - r_exp) < 1e-3, (i, approx)
                assert np.linalg.norm(vel - v_exp) < 1e-4, (i, approx)
                # and back
                pos2, vel2 = fn(sk.frame.ITRF, sk.frame.TEME, t, pos, vel)
                assert np.linalg.norm(pos2 - r) < 1e-6 and np.linalg.norm(vel2 - v) < 1e-9, (i, approx)


def _erfa_c2t_at(times):
    raws = np.array([raw_us(t) for t in times], dtype=np.int64)
    tai = tai_pair(raws)
    tt = erfa.taitt(*tai)
    eop = np.array([ft.earth_orientation_params(t) for t in times])
    ut1 = erfa.utcut1(*erfa.taiutc(*tai), eop[:, 0])
    x, y = erfa.xy06(*tt)
    x, y = x + eop[:, 4] * MAS2RAD, y + eop[:, 5] * MAS2RAD
    c2i = erfa.c2ixys(x, y, erfa.s06(*tt, x, y))
    pom = erfa.pom00(eop[:, 1] * AS2RAD, eop[:, 2] * AS2RAD, erfa.sp00(*tt))
    return erfa.c2tcio(c2i, erfa.era00(*ut1), pom)


# ----------------------------------------------------------------------------
# Geodetic (WGS 84)
# ----------------------------------------------------------------------------


class TestGeodetic:
    @staticmethod
    def _samples():
        rng = np.random.default_rng(11)
        n = 2000
        lat = rng.uniform(-90.0, 90.0, n)
        lon = rng.uniform(-180.0, 180.0, n)
        h = rng.uniform(-1000.0, 40_000e3, n)
        h[::4] = rng.uniform(-1000.0, 2000.0, len(h[::4]))  # dense near the surface
        lat[:6] = [90.0, -90.0, 89.9999999, -89.9999999, 0.0, 45.0]
        h[:6] = [0.0, 40_000e3, -1000.0, 1000.0, -1000.0, 0.0]
        return lat, lon, h

    def test_geodetic_to_cartesian(self):
        lat, lon, h = self._samples()
        xyz = erfa.gd2gc(1, np.radians(lon), np.radians(lat), h)
        for i in range(len(lat)):
            c = sk.itrfcoord(latitude_deg=lat[i], longitude_deg=lon[i], altitude=h[i])
            assert np.linalg.norm(c.vector - xyz[i]) < 1e-6, (lat[i], lon[i], h[i])

    def test_cartesian_to_geodetic(self):
        """Inverse on gd2gc(lat, lon, h): satkit recovers the input to 1e-6 m.
        ERFA's own gc2gd is only good to ~1 mm in latitude at 40,000 km
        altitude, so the direct comparison with it is at 2 mm."""
        lat, lon, h = self._samples()
        xyz = erfa.gd2gc(1, np.radians(lon), np.radians(lat), h)
        elong, phi, height = erfa.gc2gd(1, xyz)
        for i in range(len(lat)):
            c = sk.itrfcoord(xyz[i])
            r = np.linalg.norm(xyz[i])
            assert abs(c.latitude_rad - np.radians(lat[i])) * r < 1e-6, (lat[i], lon[i], h[i])
            if abs(lat[i]) < 90.0:
                dlon = wrap(c.longitude_rad - np.radians(lon[i]))
                assert abs(dlon) * r * math.cos(np.radians(lat[i])) < 1e-6, (lat[i], lon[i], h[i])
            assert abs(c.altitude - h[i]) < 1e-6, (lat[i], lon[i], h[i])
            assert abs(c.latitude_rad - phi[i]) * r < 2e-3
            assert abs(c.altitude - height[i]) < 1e-6
