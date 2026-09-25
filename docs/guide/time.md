# Time Systems

This page is the reference for how `satkit` handles time: the scales it
supports, which one each part of the library uses, how an instant is stored,
and how leap seconds, Earth orientation data and TDB enter the conversions.
Runnable Python examples are in the
[Time Systems tutorial](../tutorials/Time%20Systems.ipynb); the
[API reference](../api/time.md) covers constructors, methods and how Python
`datetime` values are interpreted.

`satkit` has its own time type, [`satkit.time`](../api/time.md) (Rust:
`satkit::Instant`), rather than using `datetime` or `datetime64`, because the
library needs what those lack: an explicit time scale, leap seconds that
exist, exact integer arithmetic, and UT1 and TDB. Every function that takes a
time accepts one, in Python and Rust alike.

## The Time Scales

A time scale is a rule for labelling instants. The scales differ in what
"one second" means (an atomic SI second, or a fraction of one rotation of the
Earth), where they are defined (on the geoid or at the solar-system
barycenter), and whether they follow the Earth's rotation. The definitions
follow Chapter 10 of the IERS Conventions
([Petit & Luzum 2010](references.md#petit2010)); UTC and leap seconds are
defined by [ITU-R TF.460-6](references.md#itu460).

`satkit.timescale` (Rust: `satkit::TimeScale`) has six members:

| Scale | Relation used by satkit | What it is for |
|-------|-------------------------|----------------|
| **TAI**, International Atomic Time | reference scale | The average of several hundred atomic clocks, in SI seconds on the geoid. It never jumps, so `satkit` stores instants and measures elapsed time in it. |
| **UTC**, Coordinated Universal Time | $\text{UTC} = \text{TAI} - \Delta AT$ | Civil time: TAI minus a whole number of leap seconds that keeps it within 0.9 s of UT1 ([Leap Seconds](#leap-seconds)). Almost every external time — a calendar date, an ISO string, a Unix timestamp, a TLE epoch — is UTC. |
| **TT**, Terrestrial Time | $\text{TT} = \text{TAI} + 32.184$ s | The time argument of geocentric theories (precession–nutation, the lunar and planetary series). J2000 is 2000-01-01 12:00:00 TT. |
| **GPS**, GPS system time | $\text{GPS} = \text{TAI} - 19$ s | Equal to UTC at the GPS epoch (1980-01-06) and has not followed a leap second since. GPS receivers and SP3 orbit files label epochs in it. |
| **UT1**, Universal Time | $\text{UT1} = \text{UTC} + \Delta\text{UT1}$ | The Earth's rotation angle expressed as time; $\Delta\text{UT1}$ is measured, not computed ([UT1 and EOP](#ut1-and-earth-orientation-parameters)). |
| **TDB**, Barycentric Dynamical Time | $\text{TDB} pprox \text{TT} + 1.657\ \text{ms}\,\sin g$ | The time argument of the solar-system ephemerides ([TDB and TT](#tdb-and-tt)). |

There is also an `Invalid` member (converting to it returns NaN). TCG, TCB
and other GNSS system times are not represented.

## Which Scale satkit Uses Where

The scale is always chosen inside the library; you pass a `satkit.time` and
each algorithm reads it in the scale it needs.

| Component | Scale |
|-----------|-------|
| Internal instant count, `time` − `time`, `time` ± `duration` | TAI (continuous SI microseconds) |
| Numerical propagator independent variable; SGP4 time since epoch | elapsed SI seconds between instants (TAI) |
| Calendar components, strings, Unix time (input and output) | UTC, unless `scale=` is given |
| Python `datetime` input | its own time zone, or local time if naive ([API reference](../api/time.md)) |
| TLE and OMM epochs; SGP4 initialization | UTC |
| EOP lookup ($\Delta$UT1, polar motion, $dX$/$dY$) | UTC (daily rows at 0 h UTC) |
| Space-weather lookup for the density models | UTC calendar day |
| Earth rotation angle (ITRF ↔ GCRF), GMST (TEME ↔ ITRF) | UT1 |
| IAU 2006/2000A precession–nutation, TIO locator $s'$, equation of the equinoxes | TT |
| JPL DE ephemerides (`satkit.jplephem`) | TDB |
| Low-precision Sun and Moon (`satkit.sun`, `satkit.moon`) | TDB |
| Low-precision planets (`satkit.planets`) | TT |

Because the frame transforms read UT1 and TT from the same instant, a time
given in the wrong scale is not a small error: labelling a GPS-time epoch as
UTC shifts it by 18 s, which is 18 s of Earth rotation (about 1.3 mrad, or
8 km at the equator) in any ITRF ↔ GCRF rotation. Build such times with
`scale=satkit.timescale.GPS` or `satkit.time.from_gps_week_and_second()`.

## How an Instant Is Stored

An `Instant` is a single signed 64-bit integer, `raw`: the number of SI
microseconds since 1970-01-01 00:00:00, counted continuously through every
leap second. In terms of the Unix time $t_\text{unix}$ (which ignores leap
seconds),

$$
\text{raw} = 10^6\,\big(t_\text{unix} + \Delta AT(t)\big)
\qquad\Longleftrightarrow\qquad
\text{MJD}_\text{TAI} = 40587 + \frac{\text{raw}}{86\,400 \times 10^6}.
$$

That is, `raw` is a TAI count, zeroed at the Unix epoch (where `satkit` takes
TAI = UTC; see [Leap Seconds](#leap-seconds)). Every other scale is computed
from it on demand:

- **UTC**: subtract $\Delta AT$ from the leap-second table.
- **TT, GPS**: add the fixed offsets above; exact in integer microseconds.
- **UT1**: add the interpolated $\Delta\text{UT1}$ to UTC.
- **TDB**: add the periodic series to TT.

Consequences of the integer representation:

- **Resolution is 1 µs** over a range of about ±292,000 years. Equality,
  ordering and hashing are exact, and the difference of two instants is an
  exact integer number of microseconds that includes any leap seconds in
  between.
- **`satkit.duration`** is also an integer microsecond count. Fractional
  microseconds passed to its constructors are truncated toward zero, and so
  are fractional microseconds in calendar seconds and in floating-point
  MJD/JD input. A decimal like `0.253922` s has no exact binary
  representation and is stored just below its value, so about 1 % of
  microsecond-exact inputs land 1 µs early (`0.253922` → `0.253921`).
- **Floating-point dates lose precision.** `to_mjd()` and `to_jd()` return
  `f64` days. Near the present an MJD is resolved to about 0.6 µs, so MJD
  round trips are exact at the microsecond level, but a Julian Date (a number
  near 2.46 million) is resolved only to about 40 µs, and a JD round trip can
  be off by ~20 µs. Compute intervals by subtracting `time` objects, not
  their MJDs or JDs.
- **Adding a float adds SI days.** `time + 1.0` adds one `duration` of
  exactly 86,400 s. To step by UTC *calendar* days — which differ from
  86,400 s on a day with a leap second — use `add_utc_days()`.

## Leap Seconds

The $\Delta AT$ table is **compiled into the library**: 28 entries, from
10 s at 1972-01-01 to 37 s at 2017-01-01, transcribed from
[IERS Bulletin C](references.md#bulletinc). No data file is read for it and
nothing is downloaded, so a leap second announced in the future requires a
new `satkit` release. After the last entry $\Delta AT$ is held at 37 s, which
remains correct until IERS announces another leap second.

**During a leap second** the internal count advances normally; one second of
`raw` values has no counterpart in a uniform-day representation of UTC.
`satkit` handles that second as follows:

- Calendar output (`str()`, `to_gregorian()`, `to_rfc3339()`) shows it as
  `23:59:60.xxxxxx`, and calendar and RFC 3339 input accept that label on a
  day that has a leap second (`satkit.time(2016, 12, 31, 23, 59, 60.5)`);
  on any other day `:60` is an error.
- `to_mjd()` / `to_jd()` in UTC, and `to_unixtime()`, map it onto a repeat of
  `23:59:59.xxxxxx`, the POSIX convention: every UTC day is 86,400 s long in
  these units, so the UTC MJD is not monotonic across a leap second. The TAI,
  TT and GPS MJDs are.
- `to_datetime()` goes through Unix time, so Python's `datetime`, which has
  no second 60, also receives `23:59:59.xxxxxx`.
- Durations and `time` differences count the leap second: adding 2 s to
  2016-12-31 23:59:59 UTC gives 2017-01-01 00:00:00 UTC, and
  `time(2017, 1, 1) - time(2016, 12, 31)` is 86,401 s.
  `add_utc_days(1.0)` steps one calendar day instead.

**Before 1972** UTC was not a leap-second scale: from 1961 it ran at an offset
rate with fractional-second steps, and TAI − UTC grew to about 10 s by the end
of 1971. `satkit` does not model this and takes $\Delta AT = 0$ before
1972-01-01. Pre-1972 UTC input is therefore converted to TAI, TT and GPS with
an error of up to ~10 s (1.4 s in 1961, rising to 9.9 s at the end of 1971).
The 10 s step is represented as one 10-second inserted interval at the end of
1971-12-31, labelled `23:59:60` through `23:59:69`, so that day is 86,410 s
long and the mapping between labels and instants stays one-to-one. (UT1 uses
the EOP table, which reaches back to 1962, and is unaffected.)

## UT1 and Earth Orientation Parameters

$\Delta\text{UT1}$ comes from the Earth orientation parameter (EOP) table,
loaded from the IERS `finals2000A.all` file
([IERS Rapid Service](references.md#iers-finals2000a)) with CelesTrak's
`EOP-All.csv` as the fallback. The table has one row per day at 0 h UTC and
is looked up by UTC MJD; values between rows are linearly interpolated.
$\Delta\text{UT1}$ jumps by +1 s at every leap second, so when two rows
straddle one the step is removed before interpolating: conversions go through
$\text{UT1} - \text{TAI}$, which is continuous, and UT1 advances smoothly
through the leap second. The inverse conversion
(`from_mjd(..., scale=UT1)`) uses the same continuous quantity and
round-trips to the microsecond.

What happens outside the table depends on where the epoch falls, as reported
by `satkit.frametransform.eop_status(t)`:

| Status | Meaning | $\Delta\text{UT1}$ used |
|--------|---------|--------------------|
| `"observed"` | inside the table, on or before the last measured row | interpolated |
| `"predicted"` | inside the table, in the IERS prediction (about a year for `finals2000A.all`) | interpolated prediction |
| `"extrapolated"` | after the last row | last row held constant; one-time warning on stderr |
| `"before_table"` | before the first row (1962 when CelesTrak's history is present, else 1973) | zero; one-time warning |
| `"not_loaded"` | no EOP table available | zero; one-time warning |

A zero $\Delta\text{UT1}$ means UT1 = UTC, an error of up to 0.9 s in UT1,
i.e. up to ~420 m of Earth rotation at the equator. Held-constant values drift
by roughly 10 ms over a few months. Refresh the table with
`satkit.utils.update_datafiles()`; check the covered span with
`satkit.frametransform.eop_coverage()`; silence the warnings with
`satkit.frametransform.disable_eop_time_warning()`; or set
`propsettings.require_eop_coverage = True` to make the propagator raise
rather than extrapolate. [Data Coverage](../getting-started/datacoverage.md#eop-coverage)
gives the span of each source.

## TDB and TT

TDB and TT tick at the same average rate but differ periodically, because a
clock on the moving, gravitating Earth runs at a varying rate relative to one
at the barycenter. `satkit` keeps only the dominant annual term
([Vallado 2013](references.md#vallado2013), Eq. 3-50;
[Petit & Luzum 2010](references.md#petit2010), §10.1):

$$
\text{TDB} - \text{TT} \approx 0.001657\ \text{s}\ \sin\!\big(628.3076\,T + 6.2401\big),
\qquad T = \frac{\text{JD}_\text{TT} - 2451545.0}{36525},
$$

with the argument in radians (628.3076 rad per century is one revolution per
year; the phase is the Earth's mean anomaly at J2000). The terms left out are
individually small; against the full series (ERFA `dtdb`) the one-term
formula is within 54 µs over 1900–2100 (20 µs RMS). The inverse
(`from_mjd(..., scale=TDB)`) evaluates the same term at the TDB date, which is
equivalent to well under a microsecond.

TDB is read by the JPL DE ephemerides (whose native argument is TDB, strictly
$T_\text{eph}$) and by the low-precision Sun and Moon models.

## Rust Usage

The [tutorial](../tutorials/Time%20Systems.ipynb) covers the Python API. The
Rust equivalents are methods on `satkit::Instant`, with the scale passed
explicitly:

```rust
use satkit::{Duration, Instant, TimeScale};

// Calendar input is UTC
let t = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0)?;
let tt_minus_utc =
    (t.as_mjd_with_scale(TimeScale::TT) - t.as_mjd_with_scale(TimeScale::UTC)) * 86400.0;
println!("TT - UTC = {tt_minus_utc:.3} s"); // 69.184 s

// The same calendar components interpreted in TT
let t_tt = Instant::from_datetime_with_scale(2024, 6, 15, 12, 0, 0.0, TimeScale::TT)?;
let dt: Duration = t - t_tt; // exact integer microseconds
assert_eq!(dt.as_microseconds(), 69_184_000);

// Crossing a leap second
let t0 = Instant::from_datetime(2016, 12, 31, 23, 59, 59.0)?;
println!("{}", t0 + Duration::from_seconds(1.0)); // 2016-12-31T23:59:60.000000Z
```

## See Also

- **Tutorial**: [Time Systems](../tutorials/Time%20Systems.ipynb) — runnable Python: creating times, converting between scales, durations, leap seconds, EOP coverage and GPS week/second, with plots of the offsets.
- **Theory**: [TLEs, SGP4 & OMMs](tle.md) for UTC element-set epochs; [Force Model](forces.md#future-propagation) for how stale EOP data affects propagation.
- **Data**: [Data Coverage](../getting-started/datacoverage.md#eop-coverage) for the span of the EOP table.
- **API**: [`satkit.time`, `satkit.timescale`, `satkit.duration`](../api/time.md); [`satkit.frametransform`](../api/frametransform.md) (`eop_status`, `eop_coverage`, `disable_eop_time_warning`).
- **References**: [Petit & Luzum 2010](references.md#petit2010) (Ch. 10), [ITU-R TF.460-6](references.md#itu460), [IERS Bulletin C](references.md#bulletinc), [Vallado 2013](references.md#vallado2013) (Eq. 3-50), [IERS `finals2000A.all`](references.md#iers-finals2000a).
