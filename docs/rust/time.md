# Time and Time Scales

`satkit::Instant` is a point in time, stored as integer TAI microseconds since 1970-01-01 00:00:00 TAI, and `satkit::Duration` is an exact microsecond interval. Arithmetic on them is integer arithmetic that counts leap seconds, and saturates at the ends of the ±292,000-year range (`checked_add` / `checked_sub` report overflow instead). The [Time Systems](../tutorials/Time%20Systems.ipynb) tutorial explains the scales and how satkit converts between them; this page shows the Rust API.

## Time scales

`TimeScale` has `UTC`, `TAI`, `TT`, `GPS`, `UT1` and `TDB`. Calendar constructors take UTC unless they say otherwise (`from_datetime_with_scale`), and the scale of a Julian date is always explicit: `as_mjd_with_scale`, `as_jd_with_scale`, `from_mjd_with_scale`, `from_jd_with_scale`, and the `_utc` shortcuts (`as_mjd_utc`, `from_jd_utc`, ...). UT1 uses the Earth orientation table (UT1 = UTC with a warning when it is not available).

Every function in the crate that takes a time is generic over the `TimeLike` trait, implemented by `Instant` and, with the `chrono` feature, by `chrono::DateTime`.

## Parsing and formatting

- `Instant::from_rfc3339` for RFC 3339 / ISO 8601 (`2024-06-15T12:00:00.25Z`, with or without an offset).
- `Instant::from_string` for free-form dates (`"June 15 2024 13:30"`, `"2024-06-15 13:30:00"`); it tries RFC 3339 first. Locale-ordered numeric dates such as `06/15/2024` are ambiguous and not supported; use `strptime` for those.
- `Instant::strptime(s, fmt)` with an explicit format (`%Y %m %d %H %M %S %f %z %b %B`).
- `Display` and `as_rfc3339()` print `2024-06-15T12:00:00.000000Z`; `strftime` formats with the same codes plus weekday names.

Parse failures are `InstantError`.

## Example

```rust
--8<-- "examples/time_scales.rs"
```

```text
$ cargo run --example time_scales
t = 2024-06-15T12:00:00.000000Z
  TAI - UTC = +37.0000 s
  TT - UTC = +69.1840 s
  GPS - UTC = +18.0000 s
  UT1 - UTC = -0.0164 s
  TDB - UTC = +69.1845 s
12:00 UTC - 12:00 TT = 69184000 us
leap second: 2016-12-31T23:59:59.000000Z -> 2016-12-31T23:59:60.000000Z -> 2017-01-01T00:00:00.000000Z
2016-12-31 lasted 86401 s
parsed: 2024-06-15T12:00:00.250000Z, 2024-06-15T13:30:00.000000Z, 2024-06-15T14:45:10.000000Z
strftime: Saturday 15 June 2024, 14:45
...
GPS week 2318, 561618 s = 2024-06-15T12:00:00.000000Z
```

The UT1 line depends on the Earth orientation table; everything else is fixed by the compiled-in leap-second table.

## chrono interoperability

With the `chrono` feature, `Instant` and `chrono::DateTime` convert both ways with `From` / `Into`, exactly to the microsecond, and a `chrono::DateTime` can be passed directly to any function that takes a time. chrono has no leap seconds: an instant inside one (`23:59:60.5`) converts to `23:59:59.5`, as Unix time does.

```rust
--8<-- "examples/chrono_interop.rs"
```

```text
$ cargo run --example chrono_interop --features chrono
from chrono: 2024-06-15T12:00:00.000000Z
to chrono:   2024-06-15 13:30:00 UTC
MJD (TT) = 60476.500801, Sun distance = 1.5196e11 m
2016-12-31T23:59:60.500000Z -> 2016-12-31 23:59:59.500 UTC
```

## See also

- [Time Systems](../tutorials/Time%20Systems.ipynb): the scales, leap seconds, UT1 and TDB.
- [`Instant`](https://docs.rs/satkit/latest/satkit/struct.Instant.html), [`Duration`](https://docs.rs/satkit/latest/satkit/struct.Duration.html) and [`TimeScale`](https://docs.rs/satkit/latest/satkit/enum.TimeScale.html) on docs.rs.
