# Time Representation

`satkit.time` is an instant with microsecond resolution that converts between the UTC, TAI, TT, TDB, UT1 and GPS time scales; `satkit.duration` is a time interval. What the scales are, how leap seconds and Earth orientation data enter, and which scale each part of the library uses are explained, with runnable examples, in [Time Systems](../tutorials/Time%20Systems.ipynb).

## Python `datetime` interoperability

*All* functions in the `satkit` package that take time as an input accept either the `satkit.time` class or the more commonly-used `datetime.datetime` class, as well as a `numpy.datetime64` (see [below](#numpy-datetime64)). Functions that take many times accept a list or 1-D numpy object array of any of these, or a 1-D `numpy.datetime64` array. A `datetime.datetime` is converted with Python's own convention (`datetime.timestamp()`):

- A *naive* datetime (no `tzinfo`) is interpreted in the machine's **local time zone**, not UTC.
- An *aware* datetime uses its own UTC offset.

For UTC, pass `tzinfo=datetime.timezone.utc` or build a `satkit.time` directly. `satkit.time.to_datetime()` returns an aware UTC datetime; `to_datetime(utc=False)` returns a naive local-time datetime, which round-trips through `satkit.time.from_datetime`. Both directions are exact to the microsecond.

```python
import datetime
import satkit as sk

sk.time.from_datetime(datetime.datetime(2024, 6, 15, 12, 30, tzinfo=datetime.timezone.utc))
# 2024-06-15T12:30:00.000000Z, on any machine

sk.time.from_datetime(datetime.datetime(2024, 6, 15, 12, 30))
# 12:30 local time: 2024-06-15T16:30:00.000000Z on a machine set to US Eastern (EDT)
```

## numpy `datetime64`

A `numpy.datetime64` scalar, or a 1-D `datetime64` array of any unit (years through attoseconds), is read as a **UTC** label. numpy's `datetime64` has no time zone and no leap seconds, so unlike a naive `datetime` it is never local time; it is converted exactly as an aware-UTC `datetime` with the same fields would be, including the pre-1972 UTC model. Like a `datetime`, it cannot name a leap second (`23:59:60`).

- Values finer than a microsecond round to the nearest microsecond; a value exactly halfway goes to the later microsecond.
- Year and month units go through the calendar (a `datetime64[M]` is the first of its month).
- `NaT` raises `ValueError`, and a value beyond satkit's range (about ±292,000 years) raises `OverflowError`; both name the array index.
- An array is read as its int64 counts, not element by element, which makes large time arrays cheap to build and pass: `np.arange` or arithmetic on a `datetime64` array replaces a Python loop over `satkit.time` objects.

```python
import numpy as np
import satkit as sk

tle = sk.TLE.from_lines([
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005",
    "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560000 00001",
])[0]
times = np.datetime64("2024-01-01T12:00") + np.arange(1440) * np.timedelta64(1, "m")
pos, vel = sk.sgp4(tle, times)  # one day at one-minute steps, shape (1440, 3)

sk.frametransform.gmst(np.datetime64("2024-01-01T12:00:00.5"))  # scalars work too
```

::: satkit.timescale

::: satkit.time

::: satkit.duration
