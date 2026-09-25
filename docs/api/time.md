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

::: satkit.timescale

::: satkit.time

::: satkit.duration
