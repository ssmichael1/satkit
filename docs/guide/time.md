# Time Systems

This page describes the time scales `satkit` supports, how
[`satkit.time`](../api/time.md) (Rust: `satkit::Instant`) represents an
instant internally, how leap seconds and Earth orientation data enter the
conversions, and which scale each part of the library runs on. It is the
reference; the [Time Systems tutorial](../tutorials/Time%20Systems.ipynb)
works through the same material in runnable Python — creating times,
converting between scales, durations, leap seconds and GPS week/second.

## Why a Dedicated Time Type

Python already has `datetime`, NumPy `datetime64` and astropy `Time`.
`satkit.time` exists because the rest of the library needs properties none of
the first two have, without taking on astropy as a dependency:

- **The time scale is explicit.** A `datetime` cannot say whether it is UTC,
  TAI or TT, and subtracting two of them across a leap second is silently
  wrong by a second. A `satkit.time` is an unambiguous instant; the scale is
  chosen when you build it and when you read it out.
- **Leap seconds exist.** `23:59:60` has its own label and a leap-second day
  is 86,401 s long (see [Leap Seconds](#leap-seconds)).
- **Exact integer arithmetic.** Instants and durations are integer
  microsecond counts (see [How an Instant Is Stored](#how-an-instant-is-stored)),
  so differences, ordering and hashing are exact.
- **UT1 and TDB come for free.** The Earth orientation table and the TDB
  series are already part of the library for the frame transforms and
  ephemerides.
- **One type everywhere.** The propagator, frame transforms, SGP4, ephemerides
  and ground-contact search all take a `satkit.time`, in Python and Rust
  alike.

Every public Python API that takes a time also accepts a `datetime`, and
`satkit.time` converts to and from one. A naive `datetime` (no `tzinfo`)
follows Python's convention and is read as the machine's **local** time; an
aware one uses its own offset.

## The Time Scales

A time scale is a rule for labelling instants. The scales used in
astrodynamics differ in what "one second" means (an atomic SI second, or a
fraction of one rotation of the Earth), where they are defined (on the geoid
or at the solar-system barycenter), and whether they are adjusted to follow
the Earth's rotation. The definitions below follow Chapter 10 of the IERS
Conventions ([Petit & Luzum 2010](references.md#petit2010)); UTC and leap
seconds are defined by [ITU-R TF.460-6](references.md#itu460).

`satkit.timescale` (Rust: `satkit::TimeScale`) has six members:

| Scale   | Name                           | Relation used by satkit                                          | Kind |
|---------|--------------------------------|------------------------------------------------------------------|------|
| **TAI** | International Atomic Time      | reference scale                                                  | uniform (SI seconds) |
| **UTC** | Coordinated Universal Time     | $\text{UTC} = \text{TAI} - \Delta AT$, $\Delta AT$ from the leap-second table | uniform between leap seconds |
| **TT**  | Terrestrial Time               | $\text{TT} = \text{TAI} + 32.184$ s                              | uniform |
| **GPS** | GPS system time                | $\text{GPS} = \text{TAI} - 19$ s                                 | uniform |
| **UT1** | Universal Time                 | $\text{UT1} = \text{UTC} + \Delta\text{UT1}$, $\Delta\text{UT1}$ from the EOP table | follows Earth rotation |
| **TDB** | Barycentric Dynamical Time     | $\text{TDB} \approx \text{TT} + 1.657\ \text{ms}\,\sin g$        | uniform on average; periodic w.r.t. TT |

(There is also an `Invalid` member; converting to it returns NaN.) TCG, TCB
and other GNSS system times are not represented.

**TAI** is the weighted average of several hundred atomic clocks, counting SI
seconds on the rotating geoid. It never jumps, which makes it the natural
scale for storing an instant and for measuring elapsed time.

**UTC** is civil time. It ticks in SI seconds at the TAI rate but is held
within 0.9 s of UT1 by inserting (in principle also removing) a *leap second*
at the end of June or December. The offset $\Delta AT = \text{TAI} - \text{UTC}$
is therefore a step function: 10 s from 1972-01-01, 37 s since 2017-01-01.
Almost every external time you will hand to `satkit` — a calendar date, an ISO
string, a Unix timestamp, a TLE epoch — is UTC.

**TT** is the time argument of geocentric theories: precession–nutation,
the lunar and planetary series, and the J2000 epoch itself (2000-01-01
12:00:00 **TT**, which is 11:58:55.816 UTC). It differs from TAI by the fixed
32.184 s inherited from the older Ephemeris Time.

**GPS time** is the GNSS system time: it matched UTC at the GPS epoch
(1980-01-06 00:00:00 UTC, when $\Delta AT$ was 19 s) and has not followed any
leap second since, so $\text{GPS} - \text{UTC} = \Delta AT - 19$ s = 18 s
today. GPS receivers, and precise-orbit products such as SP3 files, label
epochs in GPS time.

**UT1** is not an atomic scale at all: it is the angle the Earth has rotated
through, expressed as time. The difference $\Delta\text{UT1} = \text{UT1} -
\text{UTC}$ wanders by milliseconds per day with the Earth's irregular
rotation and can only be measured (by VLBI) and predicted, not computed; see
[UT1 and Earth Orientation Parameters](#ut1-and-earth-orientation-parameters).

**TDB** is the time argument of the barycentric solar-system ephemerides.
Relative to TT it carries periodic relativistic terms, dominated by an annual
term of 1.657 ms amplitude from the eccentricity of the Earth's orbit; see
[TDB and TT](#tdb-and-tt).

## Which Scale satkit Uses Where

The scale is always chosen inside the library; you pass a `satkit.time` and
each algorithm reads it in the scale it needs.

| Component | Scale |
|-----------|-------|
| Internal instant count, `time` − `time`, `time` ± `duration` | TAI (continuous SI microseconds) |
| Numerical propagator independent variable; SGP4 time since epoch | elapsed SI seconds between instants (TAI) |
| Calendar components, strings, Unix time (input and output) | UTC, unless `scale=` is given |
| Python `datetime` input | its own time zone if aware; the machine's local time zone if naive (Python's convention) |
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

## Limitations

- **Pre-1972 UTC is not modelled**: $\Delta AT = 0$ before 1972, an error
  of up to ~10 s in TAI/TT/GPS for UTC labels in 1961–1971 (see
  [Leap Seconds](#leap-seconds)).
- **Future leap seconds need a new release**: the $\Delta AT$ table is
  compiled in.
- **TDB is a one-term series**, good to ~50 µs, with no observer-dependent
  (topocentric) terms.
- **Input is truncated, not rounded, to the microsecond** (see
  [How an Instant Is Stored](#how-an-instant-is-stored)).
- **UT1 needs EOP data**: outside the table it is extrapolated or taken equal
  to UTC (see [UT1 and Earth Orientation Parameters](#ut1-and-earth-orientation-parameters)).

## See Also

- **Tutorial**: [Time Systems](../tutorials/Time%20Systems.ipynb) — runnable Python: creating times, converting between scales, durations, leap seconds, EOP coverage and GPS week/second, with plots of the offsets.
- **Theory**: [TLEs, SGP4 & OMMs](tle.md) for UTC element-set epochs; [Force Model](forces.md#future-propagation) for how stale EOP data affects propagation.
- **Data**: [Data Coverage](../getting-started/datacoverage.md#eop-coverage) for the span of the EOP table.
- **API**: [`satkit.time`, `satkit.timescale`, `satkit.duration`](../api/time.md); [`satkit.frametransform`](../api/frametransform.md) (`eop_status`, `eop_coverage`, `disable_eop_time_warning`).
- **References**: [Petit & Luzum 2010](references.md#petit2010) (Ch. 10), [ITU-R TF.460-6](references.md#itu460), [IERS Bulletin C](references.md#bulletinc), [Vallado 2013](references.md#vallado2013) (Eq. 3-50), [IERS `finals2000A.all`](references.md#iers-finals2000a).
