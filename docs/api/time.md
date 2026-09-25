# Time Representation

`satkit.time` is an instant with microsecond resolution that converts between the UTC, TAI, TT, TDB, UT1 and GPS time scales; `satkit.duration` is a time interval. What the scales are, how leap seconds and Earth orientation data enter, and which scale each part of the library uses are covered in [Theory: Time Systems](../guide/time.md); worked examples are in the [Time Systems tutorial](../tutorials/Time%20Systems.ipynb).

## Python `datetime` interoperability

*All* functions in the `satkit` package that take time as an input accept either the `satkit.time` class or the more commonly-used `datetime.datetime` class. A `datetime.datetime` is converted with Python's own convention (`datetime.timestamp()`):

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

::: satkit.timescale

::: satkit.time

::: satkit.duration
