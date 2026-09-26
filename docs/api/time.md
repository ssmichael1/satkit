# Time Representation

The `satkit` package makes use of a custom time class. This is a wrapper around a custom Rust class that adds the ability to represent time with different scales, or epochs, which is often necessary in the calculation of astronomical phenomena.

However, *all* functions in the `satkit` package that take time as an input can accept either the `satkit.time` class or the more commonly-used `datetime.datetime` class. A `datetime.datetime` is converted with Python's own convention (`datetime.timestamp()`):

- A *naive* datetime (no `tzinfo`) is interpreted in the machine's **local time zone**, not UTC.
- An *aware* datetime uses its own UTC offset.

For UTC, pass `tzinfo=datetime.timezone.utc` or build a `satkit.time` directly. `satkit.time.to_datetime()` returns an aware UTC datetime; `to_datetime(utc=False)` returns a naive local-time datetime, which round-trips through `satkit.time.from_datetime`.

```python
import datetime
import satkit as sk

sk.time.from_datetime(datetime.datetime(2024, 6, 15, 12, 30, tzinfo=datetime.timezone.utc))
# 2024-06-15T12:30:00.000000Z, on any machine

sk.time.from_datetime(datetime.datetime(2024, 6, 15, 12, 30))
# 12:30 local time: 2024-06-15T16:30:00.000000Z on a machine set to US Eastern (EDT)
```

The time scales and their relationships follow the IERS Conventions (2010), Chapter 10 ([Petit & Luzum 2010](../guide/references.md#petit2010)); UTC and leap seconds are defined by [ITU-R TF.460-6](../guide/references.md#itu460), with the leap-second table taken from [IERS Bulletin C](../guide/references.md#bulletinc).

## UTC before 1972

From 1961-01-01 to 1972-01-01, UTC ran on "rubber seconds": TAI − UTC drifted linearly, $\Delta AT = A + (\mathrm{MJD_{UTC}} - \mathrm{MJD_0}) \times r$, and stepped by fractions of a second at the start of some months. `satkit` uses the coefficients of [USNO `tai-utc.dat`](../guide/references.md#usnotaiutc), which are also those of [ERFA](../guide/references.md#erfa) `dat`, so TAI, TT, TDB and GPS built from a pre-1972 UTC label agree with ERFA to a microsecond. TAI − UTC was 1.422818 s at 1961-01-01, 8.000082 s at 1970-01-01 and 9.892242 s at the end of 1971, before the step to 10 s.

- **Positive steps** (+0.1 s on seven dates from 1963-11-01 to 1965-09-01; +0.107758 s at 1972-01-01) insert time, labelled `23:59:60.x` on the preceding day like a leap second (e.g. `1963-10-31T23:59:60.05`).
- **Negative steps** (−0.05 s at 1961-08-01, −0.1 s at 1968-02-01) remove time: the last 0.05 s / 0.1 s of labels on 1961-07-31 and 1968-01-31 never occurred. They are accepted and, as in ERFA `utctai`, land on the first 0.05 s / 0.1 s of the next day, so they do not round-trip.
- **Before 1961** UTC is taken to equal TAI (TAI − UTC = 0). This deliberately differs from ERFA, which also has a 1960 entry. The resulting 1.422818 s step at 1961-01-01 is an inserted interval labelled `1960-12-31T23:59:60.0` to `23:59:61.422817`.

On the days that end in a step, labels follow ERFA `dtf2d`/`utctai`. ERFA's `d2dtf` treats only steps over 0.5 s as leap seconds and prints labels up to the step size off on those days, and ERFA's quasi-JD stretches the day's fraction, while `to_mjd(timescale.UTC)` keeps 86,400 s days as it does for leap seconds.

`time.UNIX_EPOCH` is the UTC label 1970-01-01T00:00:00, and Unix time (`from_unixtime`, `to_unixtime`, `datetime` conversion) stays UTC-based. UT1 − UTC from the Earth orientation table is interpolated as UT1 − TAI, so the pre-1972 steps and drift, and leap seconds, do not leak into UT1: UT1 is continuous, monotonic and invertible across every step inside the table. The table starts at 1973-01-02 (`finals2000A.all`) or 1962-01-01 (with CelesTrak's `EOP-All.csv`). Before it, or with no table loaded, UT1 − UTC is taken as 0, so UT1 equals UTC and repeats its steps, stepping back at 00:00 after each leap second or positive pre-1972 step (as ERFA `utcut1` with `dut1 = 0`).

::: satkit.timescale

::: satkit.time

::: satkit.duration
